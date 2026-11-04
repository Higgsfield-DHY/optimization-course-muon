"""Strict, side-effect-free experiment configuration."""

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib

OPTIM_DEFAULTS = dict(name="muon", lr=.02, weight_decay=0., betas=[.9, .95],
                      eps=1e-8, momentum=.95, nesterov=True, ns_steps=5,
                      ns_eps=1e-7, method="ns", aspect_scale=True, aux_lr=3e-4,
                      normalization="plus")
MATRIX_DEFAULTS = dict(rows=32, cols=16, condition_number=100., noise_std=0.,
                       steps=1000, log_every=10, checkpoint_every=100)
LANGUAGE_DEFAULTS = dict(n_layer=6, n_head=4, n_embd=256, block_size=128,
                         dropout=.1, batch_size=8, grad_accum=4, max_steps=1000,
                         eval_every=100, eval_batches=10, log_every=10,
                         checkpoint_every=100, warmup_steps=100, min_lr_ratio=.1,
                         clip_grad=1., data_dir="data/shakespeare_byte", eval_mode="sampled")
RUN_DEFAULTS = dict(name="experiment", task="matrix", seed=0, device="cpu", output_root="runs",
                    phase="development", cohort="development")


def _merge(defaults, values, section):
    if not isinstance(values, dict):
        raise ValueError(f"{section} must be a TOML table")
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {section} option(s): {sorted(unknown)}")
    return copy.deepcopy(defaults) | values


def _number(value, name, minimum=0., strict=False, integer=False):
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'}")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} must be {'greater than' if strict else 'at least'} {minimum}")


def load_config(path, seed=None):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    unknown = set(raw) - {"run", "optimizer", "matrix", "language"}
    if unknown:
        raise ValueError(f"Unknown top-level table(s): {sorted(unknown)}")
    run = _merge(RUN_DEFAULTS, raw.get("run", {}), "run")
    if seed is not None:
        run["seed"] = seed
    if run["task"] not in {"matrix", "language"}:
        raise ValueError("run.task must be matrix or language")
    if run["phase"] not in {"development", "pilot", "tuning", "final", "verification"}:
        raise ValueError("Unknown run.phase")
    if not isinstance(run["cohort"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run["cohort"]):
        raise ValueError("Invalid run.cohort")
    if not isinstance(run["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run["name"]):
        raise ValueError("run.name must contain only letters, numbers, underscores and hyphens")
    _number(run["seed"], "run.seed", integer=True)
    if not isinstance(run["device"], str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", run["device"]):
        raise ValueError("run.device must be cpu, cuda or cuda:N")
    if not isinstance(run["output_root"], str) or not run["output_root"]:
        raise ValueError("run.output_root must be a nonempty path")
    root = next((p for p in path.parents if (p / "pyproject.toml").is_file()), path.parent)
    run["output_root"] = str((root / run["output_root"]).resolve())
    opt = _merge(OPTIM_DEFAULTS, raw.get("optimizer", {}), "optimizer")
    if opt["name"] not in {"muon", "adamw"} or opt["method"] not in {"ns", "svd"}:
        raise ValueError("Unknown optimizer name or orthogonalization method")
    if opt["normalization"] not in {"plus", "max"}:
        raise ValueError("Unknown optimizer.normalization")
    for key in ("lr", "aux_lr", "eps", "ns_eps"):
        _number(opt[key], f"optimizer.{key}", strict=True)
    _number(opt["weight_decay"], "optimizer.weight_decay")
    _number(opt["momentum"], "optimizer.momentum")
    _number(opt["ns_steps"], "optimizer.ns_steps", strict=True, integer=True)
    if opt["momentum"] >= 1 or not isinstance(opt["betas"], list) or len(opt["betas"]) != 2:
        raise ValueError("Momentum must be in [0,1); betas must be a pair")
    for beta in opt["betas"]:
        _number(beta, "optimizer.betas")
        if beta >= 1:
            raise ValueError("Betas must be in [0,1)")
    if any(not isinstance(opt[key], bool) for key in ("nesterov", "aspect_scale")):
        raise ValueError("nesterov and aspect_scale must be boolean")
    if max(opt["lr"], opt["aux_lr"]) * opt["weight_decay"] >= 1:
        raise ValueError("Learning rate times weight decay must be less than one")
    task = run["task"]
    other = "language" if task == "matrix" else "matrix"
    if other in raw:
        raise ValueError(f"[{other}] is not used by task {task}; remove it")
    settings = _merge(MATRIX_DEFAULTS if task == "matrix" else LANGUAGE_DEFAULTS,
                      raw.get(task, {}), task)
    if task == "matrix":
        for key in ("rows", "cols", "steps", "log_every", "checkpoint_every"):
            _number(settings[key], f"matrix.{key}", strict=True, integer=True)
        _number(settings["condition_number"], "matrix.condition_number", minimum=1.)
        _number(settings["noise_std"], "matrix.noise_std")
    else:
        if settings["eval_mode"] not in {"sampled", "full"}:
            raise ValueError("Unknown language.eval_mode")
        for key in ("n_layer", "n_head", "n_embd", "block_size", "batch_size", "grad_accum",
                    "max_steps", "eval_every", "eval_batches", "log_every", "checkpoint_every"):
            _number(settings[key], f"language.{key}", strict=True, integer=True)
        _number(settings["warmup_steps"], "language.warmup_steps", integer=True)
        if settings["n_embd"] % settings["n_head"]:
            raise ValueError("n_embd must be divisible by n_head")
        for key in ("dropout", "min_lr_ratio"):
            _number(settings[key], f"language.{key}")
            if settings[key] >= 1:
                raise ValueError(f"{key} must be less than one")
        _number(settings["clip_grad"], "language.clip_grad", strict=True)
        if settings["warmup_steps"] >= settings["max_steps"]:
            raise ValueError("warmup_steps must be smaller than max_steps")
        if not isinstance(settings["data_dir"], str) or not settings["data_dir"]:
            raise ValueError("data_dir must be a nonempty path")
        settings["data_dir"] = str((root / settings["data_dir"]).resolve())
    return {"run": run, "optimizer": opt, task: settings,
            "_config_path": str(path), "_root": str(root)}


def public_config(config):
    return copy.deepcopy({key: value for key, value in config.items() if not key.startswith("_")})


def fingerprint(config, ignore_seed=False):
    clean = public_config(config)
    clean["run"].pop("output_root", None)
    if ignore_seed:
        clean["run"].pop("seed", None)
    return hashlib.sha256(json.dumps(clean, sort_keys=True, allow_nan=False).encode()).hexdigest()
