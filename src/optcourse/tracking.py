"""Run metadata, finite metrics, and resumable state; no experiments on import."""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import uuid

import numpy as np
import torch

from .config import fingerprint, public_config

METRICS = ["step", "objective", "grad_norm_F", "grad_norm_nuclear", "train_loss",
           "validation_loss", "elapsed_seconds", "tokens", "peak_memory_MiB", "lr"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def environment_info():
    import matplotlib
    info = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__,
                cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                torch_cpu_threads=torch.get_num_threads())
    if info["cuda_available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return info


def capture_rng(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                sampler=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    generator.set_state(state["sampler"].cpu())


def load_checkpoint(path, config):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("config_fingerprint") != fingerprint(config):
        raise ValueError("Checkpoint configuration does not match the effective configuration")
    return payload


class RunStore:
    def __init__(self, path, config, metadata):
        self.path, self.config, self.metadata = Path(path), config, metadata

    @classmethod
    def create(cls, root, config):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        path = Path(root) / f"{config['run']['name']}_s{config['run']['seed']}_{run_id}"
        path.mkdir(parents=True, exist_ok=False)
        root = config.get("_root", Path.cwd())
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            revision, dirty = None, None
        metadata = dict(schema_version=1, status="RUNNING", started_at=utc_now(),
                        config=public_config(config), config_fingerprint=fingerprint(config),
                        environment=environment_info(), git_revision=revision, git_dirty=dirty,
                        attempts=[], failure_reason=None)
        store = cls(path, config, metadata)
        store.update()
        return store

    @classmethod
    def resume(cls, path, config):
        path = Path(path)
        payload = load_checkpoint(path / "latest.pt", config)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("config_fingerprint") != fingerprint(config):
            raise ValueError("Run metadata configuration mismatch")
        if payload.get("prepared_dataset_fingerprint") != metadata.get("prepared_dataset_fingerprint"):
            raise ValueError("Checkpoint dataset identity does not match run metadata")
        if metadata["status"] == "COMPLETED":
            raise ValueError("This run is already completed")
        metadata["attempts"].append(dict(previous_status=metadata["status"],
                                          failure_reason=metadata.get("failure_reason"), resumed_at=utc_now()))
        metadata.update(status="RUNNING", failure_reason=None, ended_at=None)
        metrics_path = path / "metrics.csv"
        if metrics_path.exists():
            with metrics_path.open(newline="", encoding="utf-8") as stream:
                rows = [row for row in csv.DictReader(stream) if int(row["step"]) <= payload["step"]]
            temporary_metrics = metrics_path.with_suffix(".csv.tmp")
            with temporary_metrics.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=METRICS)
                writer.writeheader()
                writer.writerows(rows)
            temporary_metrics.replace(metrics_path)
        store = cls(path, config, metadata)
        if "best_model" in payload:
            store.save_checkpoint(dict(step=payload["best_step"], model=payload["best_model"],
                                       validation_loss=payload["best_validation"]), best=True)
        store.update()
        return store, payload

    def update(self, **values):
        self.metadata.update(values)
        write_json(self.path / "metadata.json", self.metadata)

    def record(self, row):
        if set(row) - set(METRICS):
            raise ValueError("Unknown metric column")
        if any(value is not None and not math.isfinite(float(value)) for value in row.values()):
            raise ValueError("Metric values must be finite")
        path = self.path / "metrics.csv"
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRICS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def save_checkpoint(self, payload, best=False):
        destination = self.path / ("best.pt" if best else "latest.pt")
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(dict(payload, config_fingerprint=fingerprint(self.config),
                        prepared_dataset_fingerprint=self.metadata.get("prepared_dataset_fingerprint")), temporary)
        temporary.replace(destination)

    def finish(self, status, reason=None):
        if status not in {"COMPLETED", "FAILED", "INTERRUPTED"}:
            raise ValueError("Invalid terminal run status")
        self.update(status=status, failure_reason=reason, ended_at=utc_now())
