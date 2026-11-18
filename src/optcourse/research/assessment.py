"""Evaluate frozen validation winners on their complete held-out splits."""

import gc
import json
import math
from pathlib import Path

import torch

from ..budget import BudgetExhausted
from .data import TokenDataset, sha256
from .language import ResearchTransformer
from .training import atomic_json, evaluate, fingerprint


def read_json(path):
    return Path(path).read_text(encoding="utf-8")


def checked_checkpoint(run, dataset, metadata):
    config = metadata["config"]
    identity = fingerprint(config)
    if metadata["config_fingerprint"] != identity or metadata["dataset_fingerprint"] != dataset.fingerprint:
        raise ValueError(f"Run identity changed: {run.name}")

    metrics = json.loads(read_json(run / "metrics.json"))
    validations = [row for row in metrics if row["validation_loss"] is not None]
    best = min(validations, key=lambda row: (row["validation_loss"], row["step"]))
    tail = next(row for row in validations if row["step"] == config["steps"])
    if best["step"] != metadata["best_step"] or best["validation_loss"] != metadata["best_validation"]:
        raise ValueError(f"Validation selection changed: {run.name}")

    path = run / "best.pt"
    receipt = {
        "config_fingerprint": identity,
        "dataset_fingerprint": dataset.fingerprint,
        "checkpoint_sha256": sha256(path),
        "best_step": best["step"],
        "best_validation": best["validation_loss"],
        "terminal_validation": tail["validation_loss"],
        "targets": len(dataset.parts["test"]) - 1,
        "precision": config["precision"],
        "split": "test",
    }
    return path, receipt


def evaluate_checkpoint(run, dataset, metadata, budget):
    path, receipt = checked_checkpoint(run, dataset, metadata)
    cache = run / "test.json"
    if cache.exists():
        cached = json.loads(read_json(cache))
        if any(cached.get(key) != value for key, value in receipt.items()):
            raise ValueError(f"Test cache identity changed: {run.name}")
        if cached.get("status") != "COMPLETED" or not math.isfinite(cached["loss"]):
            raise ValueError(f"Invalid cached test result: {run.name}")
        budget.validate_provenance(cached["budget_provenance"])
        return cached

    torch.set_num_threads(1)
    config = metadata["config"]
    device = torch.device(config["device"])
    with budget.session("test_" + run.name):
        gc.collect()
        torch.cuda.empty_cache()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("config_fingerprint", "dataset_fingerprint"):
            if payload[key] != receipt[key]:
                raise ValueError(f"Test checkpoint identity changed: {run.name}")
        if payload["step"] != receipt["best_step"]:
            raise ValueError(f"Test checkpoint step changed: {run.name}")
        model = ResearchTransformer(**config["model"])
        model.load_state_dict(payload["model"])
        model.to(device)
        result = evaluate(model, dataset, "test", config["model"]["context"],
                          config["batch_size"], device, config["precision"], budget=budget)
        if result["targets"] != receipt["targets"] or not math.isfinite(result["loss"]):
            raise ValueError(f"Incomplete test result: {run.name}")
        record = receipt | result | {"status": "COMPLETED", "budget_provenance": budget.provenance()}
        del model, payload
    atomic_json(cache, record)
    return record


def evaluate_formal(root):
    from .protocol import ledger

    root = Path(root)
    formal = json.loads(read_json(root / "formal_manifest.json"))
    jobs = json.loads(read_json(root / "frozen_protocol.json"))["expected_jobs"]
    keys = lambda row: (row["cohort"], row["corpus"], row["scale"], row["method"], row["seed"])
    if sorted(map(keys, formal)) != sorted(map(keys, jobs)):
        raise ValueError("Formal manifest does not match the registered jobs")

    manifest_path = root / "test_manifest.json"
    budget = ledger(root)
    datasets = {}
    entries = [
        {**dict(zip(("cohort", "corpus", "scale", "method", "seed"), keys(row))), "id": row.get("id"),
         "status": "NOT_RUN", "training_status": row["status"]}
        for row in formal
    ]
    atomic_json(manifest_path, entries)

    for entry, job in zip(entries, formal):
        if job["status"] != "COMPLETED":
            continue
        run = root / "language" / job["id"]
        if job["corpus"] not in datasets:
            datasets[job["corpus"]] = TokenDataset(Path("data/research_v2/tokens") / job["corpus"])
        dataset = datasets[job["corpus"]]
        try:
            entry.update(evaluate_checkpoint(run, dataset, job["metadata"], budget))
            print(f"{job['id']}: test loss {entry['loss']:.6f}, {entry['targets']} targets", flush=True)
        except BudgetExhausted as error:
            entry.update(status="INTERRUPTED", failure_reason=str(error))
            atomic_json(manifest_path, entries)
            break
        except Exception as error:
            entry.update(status="FAILED", failure_reason=f"{type(error).__name__}: {error}")
            atomic_json(manifest_path, entries)
            raise
        atomic_json(manifest_path, entries)
    return entries
