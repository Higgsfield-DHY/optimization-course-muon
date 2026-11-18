"""Checkpointable token training with full prepared-step interventions."""

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from ..budget import BudgetExhausted
from .instruments import evaluate_candidates
from .language import ResearchTransformer,token_loss,microbatch_token_loss
from .optimizer import ResearchOptimizer,spectral_candidate_errors

SOURCE_FILES=('geometry.py','instruments.py','optimizer.py','language.py','data.py','training.py','../model.py')
SOURCE_HASHES={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
INITIAL_TRAINER_SHA256='751de549d5c3c81fd94a485c172fa35db87780db0b10b2557ba9681cdd52f518'
INITIAL_MODEL_SHA256='350052b18145372763b11b3cb7dc67cf7535d4de1a64ae579657cc0db6a37f4d'


def compatible_source(hashes):
    if hashes==SOURCE_HASHES:
        return True
    # The initial final runs predate the transaction and replay fixes.
    # Pencil in their original trainer hash only when every shared numerical
    # module, including the imported Transformer blocks, still matches.
    legacy={name:digest for name,digest in SOURCE_HASHES.items() if name!='../model.py'}
    legacy['training.py']=INITIAL_TRAINER_SHA256
    return hashes==legacy and SOURCE_HASHES['../model.py']==INITIAL_MODEL_SHA256


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)


def atomic_torch(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);temporary.replace(path)


def fingerprint(config):return hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest()


def amp(device,precision):return torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and precision=='bfloat16')


@torch.no_grad()
def evaluate(model,dataset,split,context,batch_size,device,precision='float32',targets=None,budget=None):
    training=model.training;model.eval();total=0.;count=0
    try:
        for x,y in dataset.evaluation_batches(split,context,batch_size,device):
            if targets is not None and count>=targets:break
            if targets is not None:
                valid=torch.nonzero(y.reshape(-1)>=0).flatten()
                if len(valid)>targets-count:
                    y=y.clone();y.reshape(-1)[valid[targets-count:]]=-1
            n=(y>=0).sum().item()
            with budget.operation('evaluation_batch') if budget else nullcontext():
                with amp(device,precision):loss=token_loss(model,(x,y))
                total+=loss.item()*n;count+=n
        if budget:budget.check()
        return dict(loss=total/count,targets=count)
    finally:model.train(training)


def run_training(config,dataset,directory,budget=None,max_updates=None):
    started=time.perf_counter()
    device=torch.device(config['device'])
    if device.type=='cuda' and budget is None:raise ValueError('CUDA training requires the shared research ledger')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    identity=fingerprint(config);metadata_path=directory/'metadata.json'
    if metadata_path.exists():
        previous=json.loads(metadata_path.read_text(encoding='utf-8'))
        if previous['config_fingerprint']!=identity or previous['dataset_fingerprint']!=dataset.fingerprint:
            raise ValueError('Run configuration/data identity changed')
        if config['phase']=='final' and not compatible_source(previous.get('source_hashes')):
            raise ValueError('Run training source identity changed')
        if previous['status']=='COMPLETED':return previous
    else:previous={}
    torch.set_num_threads(1);torch.manual_seed(config['seed'])
    if device.type=='cuda':
        torch.cuda.manual_seed_all(config['seed']);torch.cuda.reset_peak_memory_stats(device)
    model=ResearchTransformer(**config['model']).to(device)
    optimizer=ResearchOptimizer(model,config['method'],config['lr'],config['aux_lr'],config['weight_decay'],ns_dtype=config['ns_dtype'])
    sampler=torch.Generator().manual_seed(config['seed']+1)
    probe_generator=torch.Generator().manual_seed(config['seed']+10001)
    step=0;metrics=[];selections=[];best_model=None;best_validation=float('inf');best_step=None;pending=True
    elapsed_base=previous.get('elapsed_seconds',0.)
    checkpoint=directory/'latest.pt'
    if checkpoint.exists():
        payload=torch.load(checkpoint,map_location=device,weights_only=False)
        if payload['config_fingerprint']!=identity or payload['dataset_fingerprint']!=dataset.fingerprint:raise ValueError('Checkpoint identity mismatch')
        if config['phase']=='final' and not compatible_source(payload.get('source_hashes',previous.get('source_hashes'))):
            raise ValueError('Checkpoint training source identity changed')
        model.load_state_dict(payload['model']);optimizer.load_state_dict(payload['optimizer'])
        sampler.set_state(payload['sampler'].cpu());probe_generator.set_state(payload['probe_generator'].cpu())
        torch.set_rng_state(payload['cpu_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in payload['cuda_rng']])
        step=payload['step'];metrics=payload['metrics'];selections=payload['selections']
        best_model={k:v.cpu() for k,v in payload['best_model'].items()} if payload['best_model'] is not None else None
        best_validation=payload['best_validation'];best_step=payload['best_step'];pending=payload['pending_validation']
        del payload
        if best_model is not None:atomic_torch(directory/'best.pt',dict(model=best_model,step=best_step,config_fingerprint=identity,dataset_fingerprint=dataset.fingerprint))
    metadata=dict(config=config,config_fingerprint=identity,dataset_fingerprint=dataset.fingerprint,
        source_hashes=SOURCE_HASHES,
        parameter_count=sum(p.numel() for p in model.parameters()),hidden_parameters=sum(model.get_parameter(n).numel() for n in optimizer.hidden_names),
        auxiliary_parameters=sum(model.get_parameter(n).numel() for n in optimizer.auxiliary_names),status='RUNNING',seed=config['seed'],method=config['method'])
    if previous and previous.get('source_hashes')!=SOURCE_HASHES:
        metadata['previous_source_hashes']=previous.get('source_hashes')
    if budget:metadata['budget_provenance']=budget.provenance()
    atomic_json(metadata_path,metadata)
    model.train()

    def elapsed():
        if device.type=='cuda':torch.cuda.synchronize(device)
        return elapsed_base+time.perf_counter()-started

    def save():
        atomic_torch(checkpoint,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
            sampler=sampler.get_state(),probe_generator=probe_generator.get_state(),cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],metrics=metrics,selections=selections,
            best_model=best_model,best_validation=best_validation,best_step=best_step,pending_validation=pending,
            config_fingerprint=identity,dataset_fingerprint=dataset.fingerprint,source_hashes=SOURCE_HASHES))
        atomic_json(directory/'metrics.json',metrics);atomic_json(directory/'selections.json',selections)

    def validate(train_loss=None):
        nonlocal best_validation,best_step,best_model,pending
        value=evaluate(model,dataset,'val',config['model']['context'],config['batch_size'],device,config['precision'],config.get('validation_targets'),budget)
        metrics.append(dict(step=step,tokens=step*config['batch_size']*config['accumulation']*config['model']['context'],
            train_loss=train_loss,validation_loss=value['loss'],validation_targets=value['targets'],elapsed_seconds=elapsed()))
        if value['loss']<best_validation:
            best_validation=value['loss'];best_step=step
            best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            atomic_torch(directory/'best.pt',dict(model=best_model,step=best_step,config_fingerprint=identity,dataset_fingerprint=dataset.fingerprint))
        pending=False;save()

    status='COMPLETED';failure=None;in_update=False;sampler_before=None;probe_before=None
    try:
        if pending:validate()
        final_step=min(config['steps'],max_updates if max_updates is not None else config['steps'])
        for current in range(step+1,final_step+1):
            if budget:budget.check(1.5*budget.operation_costs.get('update',0.))
            sampler_before=sampler.get_state();probe_before=probe_generator.get_state();in_update=True
            with budget.operation('update') if budget else nullcontext():
                optimizer.zero_grad(set_to_none=True);loss_value=0.
                for _ in range(config['accumulation']):
                    batch=dataset.batch('train',config['batch_size'],config['model']['context'],sampler,device)
                    with amp(device,config['precision']):loss=token_loss(model,batch)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    (loss/config['accumulation']).backward();loss_value+=loss.detach().item()/config['accumulation']
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                warmup=config['warmup_steps']
                factor=current/warmup if warmup and current<=warmup else .1+.9*.5*(1+math.cos(math.pi*(current-warmup)/max(1,config['steps']-warmup)))
                probing=config['method'] in {'adaptive','replay','spectral'} and current>=config['probe_interval'] and current%config['probe_interval']==0
                anchoring=config.get('save_anchors') and current in config.get('anchor_steps',[])
                prepared=optimizer.prepare(factor,depths=(1,3,5) if probing or anchoring else None)
                if probing:
                    probe_started=time.perf_counter()
                    if config['method']=='spectral':
                        errors=spectral_candidate_errors(model,optimizer,prepared,factor)
                        scores={key:dict(polar_error=value) for key,value in errors.items()}
                        choice=min(errors,key=lambda key:(errors[key],int(key[2:])))
                    else:
                        batch=dataset.batch('train',config['probe_batch_size'],config['model']['context'],probe_generator,device)
                        scores=evaluate_candidates(model,prepared.deltas,batch,microbatch_token_loss)
                        if config['method']=='replay':choice=f"ns{config['depth_schedule'][len(selections)]}"
                        else:choice=min(scores,key=lambda key:(scores[key]['loss'],int(key[2:])))
                    decision=dict(step=current,choice=choice,scores=scores,probe_seconds=time.perf_counter()-probe_started)
                else:choice=f'ns{optimizer.depth}' if config['method'] in {'adaptive','replay','spectral'} else config['method']
                if anchoring:
                    cache={}
                    def cpu_copy(value):
                        key=(value.data_ptr(),tuple(value.shape),tuple(value.stride()),value.dtype,value.device)
                        if key not in cache:cache[key]=value.detach().clone() if value.device.type=='cpu' else value.detach().cpu()
                        return cache[key]
                    with torch.no_grad():
                        live_momenta={name:optimizer.momentum*prepared.states[model.get_parameter(name)]['buffer']+
                            (1-optimizer.momentum)*model.get_parameter(name).grad for name in optimizer.hidden_names}
                        momenta={name:cpu_copy(value) for name,value in live_momenta.items()}
                        anchor=dict(model={n:cpu_copy(v) for n,v in model.state_dict().items()},
                            candidates={key:{n:cpu_copy(v) for n,v in values.items()} for key,values in prepared.deltas.items()},
                            momenta=momenta,hidden_names=sorted(optimizer.hidden_names),selected=choice,config=config,
                            step=current,base_completed_updates=step,lr_factor=factor,sampler_before=sampler_before,
                            config_fingerprint=identity,dataset_fingerprint=dataset.fingerprint)
                        anchors=directory/'anchors';anchors.mkdir(exist_ok=True);atomic_torch(anchors/f'step-{current}.pt',anchor)
                    del anchor,cache,momenta,live_momenta
                optimizer.commit(prepared,choice);step=current;in_update=False
                if probing:
                    optimizer.depth=int(choice[2:])
                    selections.append(decision)
            pending=step%config['validation_every']==0 or step==config['steps']
            if pending:
                save();validate(loss_value)
            elif step%config['log_every']==0:
                metrics.append(dict(step=step,tokens=step*config['batch_size']*config['accumulation']*config['model']['context'],
                    train_loss=loss_value,validation_loss=None,validation_targets=None,elapsed_seconds=elapsed()))
            if step%config['log_every']==0:print(f"{directory.name}: step {step}/{config['steps']} loss {loss_value:.4f}",flush=True)
            if budget:budget.check()
        if step<config['steps']:status='INTERRUPTED';failure='REQUESTED_UPDATE_BOUNDARY'
    except BudgetExhausted as error:status='INTERRUPTED';failure=str(error)
    except BaseException as error:
        status='FAILED';failure=f'{type(error).__name__}: {error}'
        if in_update:
            sampler.set_state(sampler_before);probe_generator.set_state(probe_before)
        save();metadata.update(status=status,failure_reason=failure,completed_steps=step,elapsed_seconds=elapsed());atomic_json(metadata_path,metadata)
        raise
    if in_update:
        sampler.set_state(sampler_before);probe_generator.set_state(probe_before)
    save();metadata.update(status=status,failure_reason=failure,completed_steps=step,best_validation=best_validation,
        best_step=best_step,elapsed_seconds=elapsed(),peak_memory_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0.)
    atomic_json(metadata_path,metadata)
    return metadata
