"""Measure forward/backward, clipping and optimizer work on a common batch."""

import gc
import json
from pathlib import Path
from statistics import median
import time

import torch

from .data import TokenDataset, sha256
from .instruments import evaluate_candidates
from .language import ResearchTransformer, token_loss, microbatch_token_loss
from .optimizer import ResearchOptimizer, spectral_candidate_errors
from .training import amp, atomic_json


def benchmark(anchor_path, dataset, budget, iterations=12, burn_in=4):
    device = torch.device("cuda")
    anchor = torch.load(anchor_path, map_location="cpu", weights_only=True)
    config = anchor["config"]
    if anchor["dataset_fingerprint"] != dataset.fingerprint:
        raise ValueError("Profiling anchor and dataset differ")
    generator = torch.Generator().manual_seed(73891)
    batches = [dataset.batch("train", config["batch_size"], config["model"]["context"], generator, device)
               for _ in range(config["accumulation"])]
    probe_batch = dataset.batch("train", 16, config["model"]["context"], generator, device)
    rows = []
    for method in ("adamw", "momentum", "ns1", "ns3", "ns5", "canonical5", "adaptive", "spectral"):
        budget.check()
        gc.collect()
        torch.cuda.empty_cache()
        model = ResearchTransformer(**config["model"])
        model.load_state_dict(anchor["model"])
        model.to(device)
        optimizer = ResearchOptimizer(model, method, config["lr"], config["aux_lr"],
                                      config["weight_decay"], ns_dtype=config["ns_dtype"])
        torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(burn_in + iterations):
            with budget.operation("profile_step"):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                for batch in batches:
                    with amp(device, config["precision"]):
                        loss = token_loss(model, batch)
                    (loss / len(batches)).backward()
                torch.cuda.synchronize(device)
                after_model = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                torch.cuda.synchronize(device)
                after_clip = time.perf_counter()
                if method in ("adaptive", "spectral"):
                    prepared = optimizer.prepare(lr_factor=0.2, depths=(1, 3, 5))
                    if method == "adaptive":
                        scores = evaluate_candidates(model, prepared.deltas, probe_batch, microbatch_token_loss)
                        choice = min(scores, key=lambda key: (scores[key]["loss"], int(key[2:])))
                    else:
                        errors = spectral_candidate_errors(model, optimizer, prepared, 0.2)
                        choice = min(errors, key=lambda key: (errors[key], int(key[2:])))
                else:
                    prepared = optimizer.prepare(lr_factor=0.2)
                    choice = method
                optimizer.commit(prepared, choice)
                torch.cuda.synchronize(device)
                end = time.perf_counter()
            if iteration >= burn_in:
                rows.append({
                    "method": method,
                    "iteration": iteration - burn_in,
                    "model_seconds": after_model - start,
                    "clip_seconds": after_clip - after_model,
                    "optimizer_seconds": end - after_clip,
                    "total_seconds": end - start,
                    "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20,
                })
        del model, optimizer, prepared, loss
    return rows


def profile_formal(root):
    from .protocol import ledger

    root = Path(root)
    dataset = TokenDataset(Path("data/research_v2/tokens/wikitext103"))
    budget = ledger(root)
    torch.set_num_threads(1)
    results = {}
    for cohort, scale in (("main", "small"), ("scale", "medium")):
        anchor = root / "language" / f"{cohort}_wikitext103_{scale}_ns5_s0" / "anchors/step-500.pt"
        output = root / f"profile_{scale}.json"
        identity = sha256(anchor)
        if output.exists():
            previous = json.loads(output.read_text(encoding="utf-8"))
            if previous["anchor_sha256"] != identity:
                raise ValueError("Profiling cache anchor changed")
            budget.validate_provenance(previous["budget_provenance"])
            results[scale] = previous
            continue
        with budget.session(f"profile_{scale}"):
            rows = benchmark(anchor, dataset, budget)
            result = {
                "rows": rows,
                "anchor_sha256": identity,
                "dataset_fingerprint": dataset.fingerprint,
                "device": torch.cuda.get_device_name(),
                "precision": "bfloat16",
                "timed_updates": 12,
                "warmup_updates": 4,
                "budget_provenance": budget.provenance(),
            }
        atomic_json(output, result)
        results[scale] = result
        for method in dict.fromkeys(row["method"] for row in rows):
            values = [row["optimizer_seconds"] for row in rows if row["method"] == method]
            print(f"{scale} {method}: optimizer median {1000*median(values):.2f} ms", flush=True)
    return results
