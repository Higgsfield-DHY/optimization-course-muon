"""Deferred experiment runners. This module never runs training at import time.

The setup session verifies helpers/components only. CLI execution requires --execute.
"""

import math
from pathlib import Path
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

from .data import ByteDataset
from .model import TinyTransformer
from .optim import Muon, build_optimizers
from .tracking import RunStore, capture_rng, restore_rng
from .budget import BudgetExhausted,require_study_accounting


def lr_factor(step, total, warmup, minimum):
    if warmup and step <= warmup:
        return step / warmup
    progress = min(1., max(0., (step - warmup) / (total - warmup)))
    return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))


def make_matrix_problem(rows, cols, condition_number, seed):
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(rows, rows, generator=generator, dtype=torch.float64))
    singular = torch.logspace(0., .5 * math.log10(condition_number), rows, dtype=torch.float64)
    a = singular[:, None] * q.T
    target = torch.randn(rows, cols, generator=generator, dtype=torch.float64)
    initial = torch.randn(rows, cols, generator=generator, dtype=torch.float64)
    return tuple(item.float() for item in (a, a @ target, target, initial))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _restore_optimizers(optimizers, states):
    if len(optimizers) != len(states):
        raise ValueError("Checkpoint optimizer count mismatch")
    for optimizer, state in zip(optimizers, states):
        optimizer.load_state_dict(state)


def _open_store(config, resume, dataset_fingerprint=None):
    if resume:
        import json
        metadata = json.loads((Path(resume) / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("prepared_dataset_fingerprint") != dataset_fingerprint:
            raise ValueError("Resume dataset checksum does not match the original run")
        return RunStore.resume(resume, config)
    return RunStore.create(config["run"]["output_root"], config), None


@torch.no_grad()
def _run_matrix(config, store, device, checkpoint, budget=None):
    settings, opt = config["matrix"], config["optimizer"]
    a, b, _, initial = make_matrix_problem(settings["rows"], settings["cols"],
                                          settings["condition_number"], config["run"]["seed"])
    a, b = a.to(device), b.to(device)
    weight = torch.nn.Parameter(initial.to(device))
    if opt["name"] == "muon":
        optimizer = Muon([weight], **{key: opt[key] for key in
            ("lr", "weight_decay", "momentum", "nesterov", "method", "ns_steps", "ns_eps", "aspect_scale", "normalization")})
    else:
        optimizer = torch.optim.AdamW([weight], lr=opt["lr"], betas=tuple(opt["betas"]),
                                       eps=opt["eps"], weight_decay=opt["weight_decay"])
    generator = torch.Generator().manual_seed(config["run"]["seed"] + 1)
    step, elapsed_base = 0, 0.
    if checkpoint:
        weight.copy_(checkpoint["weight"].to(device))
        optimizer.load_state_dict(checkpoint["optimizers"][0])
        restore_rng(checkpoint["rng"], generator)
        step, elapsed_base = checkpoint["step"], checkpoint["elapsed_seconds"]
    store.update(problem="0.5 * ||A W - B||_F^2; Hessian condition number specified",
                 gradient_noise="independent Gaussian per entry; noise_std is per-entry std",
                 parameter_count=weight.numel())
    _sync(device)
    started = time.perf_counter()

    def elapsed():
        _sync(device)
        return elapsed_base + time.perf_counter() - started

    def record(current):
        residual = a @ weight - b
        true_gradient = a.T @ residual
        store.record(dict(step=current, objective=.5 * residual.square().sum().item(),
                          grad_norm_F=true_gradient.norm().item(),
                          grad_norm_nuclear=torch.linalg.svdvals(true_gradient).sum().item(),
                          elapsed_seconds=elapsed(), lr=opt["lr"]))

    def save(current):
        store.save_checkpoint(dict(step=current, weight=weight.detach().clone(),
                                   optimizers=[optimizer.state_dict()], rng=capture_rng(generator),
                                   elapsed_seconds=elapsed()))

    if checkpoint is None:
        record(0)
        save(0)
    for step in range(step + 1, settings["steps"] + 1):
        if budget is not None:
            try:
                budget.check(1.5*budget.operation_costs.get('update',0.))
            except BudgetExhausted:
                save(step-1)
                store.update(completed_steps=step-1, elapsed_seconds=elapsed())
                raise
        with budget.operation('update') if budget is not None else nullcontext():
            gradient = a.T @ (a @ weight - b)
            if settings["noise_std"]:
                noise = torch.randn(weight.shape, generator=generator).to(device)
                gradient = gradient + settings["noise_std"] * noise
            weight.grad = gradient
            optimizer.step()
        if step % settings["log_every"] == 0 or step == settings["steps"]:
            record(step)
            print(f"{store.path.name}: matrix step {step}/{settings['steps']}", file=sys.stderr)
        if step % settings["checkpoint_every"] == 0 or step == settings["steps"]:
            save(step)
        if budget is not None:
            try:
                budget.check()
            except BudgetExhausted:
                save(step)
                store.update(completed_steps=step,elapsed_seconds=elapsed())
                raise
    store.update(completed_steps=step, elapsed_seconds=elapsed())


@torch.no_grad()
def _validation_loss(model, dataset, settings, device, budget=None):
    if settings.get('eval_mode') == 'full':
        from .evaluation import evaluate_split
        return evaluate_split(model,dataset,'val',settings['block_size'],settings['batch_size'],device,budget)['loss']
    was_training=model.training
    model.eval()
    generator = torch.Generator().manual_seed(12345)
    losses = []
    try:
        for _ in range(settings["eval_batches"]):
            with budget.operation('evaluation_batch') if budget is not None else nullcontext():
                x, y = dataset.batch("val", settings["batch_size"], settings["block_size"], generator, device)
                logits = model(x)
                losses.append(F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).item())
        if budget is not None:
            budget.check()
        return sum(losses) / len(losses)
    finally:
        model.train(was_training)


def _run_language(config, store, device, dataset, checkpoint, budget=None):
    settings=config['language']
    model=TinyTransformer(settings).to(device)
    optimizers,groups=build_optimizers(model,config['optimizer'])
    base_lrs=[[group['lr'] for group in optimizer.param_groups] for optimizer in optimizers]
    generator=torch.Generator().manual_seed(config['run']['seed']+1)
    current,elapsed_base,best_validation,previous_peak=0,0.,float('inf'),0.
    best_model,best_step=None,0
    pending_validation,pending_loss=True,None
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        _restore_optimizers(optimizers,checkpoint['optimizers'])
        restore_rng(checkpoint['rng'],generator)
        current,elapsed_base=checkpoint['step'],checkpoint['elapsed_seconds']
        best_validation=checkpoint['best_validation'] if checkpoint['best_validation'] is not None else float('inf')
        previous_peak=checkpoint['peak_memory_MiB']
        best_model,best_step=checkpoint['best_model'],checkpoint['best_step']
        pending_validation=checkpoint.get('pending_validation',False)
        pending_loss=checkpoint.get('pending_train_loss')
    store.update(parameter_count=sum(p.numel() for p in model.parameters()),optimizer_groups=groups,
        dataset_source_sha256=dataset.manifest['source_sha256'],prepared_dataset_fingerprint=dataset.fingerprint,
        vocabulary='256 raw UTF-8 byte values',loss_unit='natural-log cross entropy per byte',dtype='float32',
        test_set_evaluated=False,validation_sampling_seed=12345 if settings['eval_mode']=='sampled' else None,
        evaluation_mode=settings['eval_mode'])
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started=time.perf_counter()
    def elapsed():
        _sync(device)
        return elapsed_base+time.perf_counter()-started
    def peak():
        return max(previous_peak,torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0.)
    def save():
        store.save_checkpoint(dict(step=current,model=model.state_dict(),
            optimizers=[optimizer.state_dict() for optimizer in optimizers],rng=capture_rng(generator),
            best_validation=best_validation if math.isfinite(best_validation) else None,
            best_model=best_model,best_step=best_step,elapsed_seconds=elapsed(),peak_memory_MiB=peak(),
            pending_validation=pending_validation,pending_train_loss=pending_loss))
    def validate_and_record(train_loss):
        nonlocal best_validation,best_model,best_step,pending_validation
        validation=_validation_loss(model,dataset,settings,device,budget)
        if not math.isfinite(validation):
            raise RuntimeError('Nonfinite validation loss')
        if validation<best_validation:
            best_validation,best_step=validation,current
            best_model={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
            store.save_checkpoint(dict(step=best_step,model=best_model,validation_loss=best_validation),best=True)
        pending_validation=False
        record(train_loss,validation)
    def record(train_loss,validation):
        store.record(dict(step=current,train_loss=train_loss,validation_loss=validation,
            elapsed_seconds=elapsed(),tokens=current*settings['batch_size']*settings['block_size']*settings['grad_accum'],
            peak_memory_MiB=peak(),lr=optimizers[0].param_groups[0]['lr']))
    try:
        if checkpoint is None:
            save()  # A resumable step-zero state, even if initial validation exhausts the budget.
        if pending_validation:
            validate_and_record(pending_loss)
            save()
        for step in range(current+1,settings['max_steps']+1):
            pending_loss=None
            with budget.operation('update') if budget is not None else nullcontext():
                factor=lr_factor(step,settings['max_steps'],settings['warmup_steps'],settings['min_lr_ratio'])
                for optimizer,rates in zip(optimizers,base_lrs):
                    optimizer.zero_grad(set_to_none=True)
                    for group,rate in zip(optimizer.param_groups,rates):
                        group['lr']=rate*factor
                loss_value=0.
                for _ in range(settings['grad_accum']):
                    x,y=dataset.batch('train',settings['batch_size'],settings['block_size'],generator,device)
                    logits=model(x)
                    loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
                    if not torch.isfinite(loss).item():
                        raise RuntimeError(f'Nonfinite training loss at step {step}')
                    (loss/settings['grad_accum']).backward()
                    loss_value+=loss.detach().item()/settings['grad_accum']
                torch.nn.utils.clip_grad_norm_(model.parameters(),settings['clip_grad'],error_if_nonfinite=True)
                for optimizer in optimizers:
                    optimizer.step()
                current=step
            pending_validation=step%settings['eval_every']==0 or step==settings['max_steps']
            pending_loss=loss_value if pending_validation else None
            if budget is not None:
                budget.check()
            if pending_validation:
                validate_and_record(loss_value)
            elif step%settings['log_every']==0:
                record(loss_value,None)
            if step%settings['log_every']==0 or step%settings['eval_every']==0 or step==settings['max_steps']:
                print(f"{store.path.name}: language step {step}/{settings['max_steps']}",file=sys.stderr)
            if step%settings['checkpoint_every']==0 or step==settings['max_steps']:
                save()
            if budget is not None:
                budget.check()
    except BudgetExhausted:
        save()
        store.update(completed_steps=current,elapsed_seconds=elapsed())
        raise
    store.update(completed_steps=current,best_validation=best_validation,elapsed_seconds=elapsed(),peak_memory_MiB=peak())


def run_experiment(config, resume=None, budget=None, on_store=None):
    require_study_accounting(config,budget,resume)
    if budget is not None:
        budget.check()
    device = torch.device(config["run"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; activate the project environment or choose device=cpu explicitly")
    seed = config["run"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    dataset = ByteDataset(config["language"]["data_dir"]) if config["run"]["task"] == "language" else None
    if dataset:
        for split in ("train", "val"):
            if len(dataset.parts[split]) <= config["language"]["block_size"]:
                raise ValueError(f"Split {split} is too short for block_size")
    store, checkpoint = _open_store(config, resume, dataset.fingerprint if dataset else None)
    store.update(phase=config['run'].get('phase','development'),cohort=config['run'].get('cohort','development'))
    if on_store is not None:
        on_store(store)
    try:
        if dataset:
            _run_language(config, store, device, dataset, checkpoint, budget)
        else:
            _run_matrix(config, store, device, checkpoint, budget)
        if budget is not None:
            budget.check()
        store.finish("COMPLETED")
    except BudgetExhausted:
        store.finish('INTERRUPTED','GPU_BUDGET_EXHAUSTED')
    except KeyboardInterrupt:
        store.finish("INTERRUPTED", "KeyboardInterrupt; resume from the latest saved step boundary")
        raise
    except Exception as error:
        store.finish("FAILED", f"{type(error).__name__}: {error}")
        raise
    return store.path
