import copy
import csv
import importlib
import json
import random

import pytest
import torch


def trial_config(tmp_path, seed=0):
    return dict(run=dict(name="unit", task="matrix", seed=seed, device="cpu", output_root=str(tmp_path)),
                optimizer=dict(name="adamw", lr=.01), matrix=dict(steps=3))


def test_checkpoint_rejects_a_different_effective_configuration(tmp_path):
    tracking = importlib.import_module("optcourse.tracking")
    config = trial_config(tmp_path)
    store = tracking.RunStore.create(tmp_path / "runs", config)
    store.save_checkpoint(dict(step=2, value=torch.tensor([4.])))
    loaded = tracking.load_checkpoint(store.path / "latest.pt", config)
    assert loaded["step"] == 2
    torch.testing.assert_close(loaded["value"], torch.tensor([4.]))
    changed = copy.deepcopy(config)
    changed["optimizer"]["lr"] = .5
    with pytest.raises(ValueError, match="configuration"):
        tracking.load_checkpoint(store.path / "latest.pt", changed)


def test_rng_snapshot_restores_global_and_batch_sampling_streams():
    tracking = importlib.import_module("optcourse.tracking")
    sampler = torch.Generator().manual_seed(99)
    snapshot = tracking.capture_rng(sampler)
    expected_sample, expected_global = torch.rand(3, generator=sampler), torch.rand(3)
    expected_python = random.random()
    tracking.restore_rng(snapshot, sampler)
    torch.testing.assert_close(torch.rand(3, generator=sampler), expected_sample)
    torch.testing.assert_close(torch.rand(3), expected_global)
    assert random.random() == expected_python


def test_metrics_reject_nonfinite_values_and_record_failure_reason(tmp_path):
    tracking = importlib.import_module("optcourse.tracking")
    store = tracking.RunStore.create(tmp_path / "runs", trial_config(tmp_path))
    with pytest.raises(ValueError, match="finite"):
        store.record(dict(step=0, objective=float("nan")))
    store.finish("FAILED", "unit fixture failure")
    meta = json.loads((store.path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "FAILED" and meta["failure_reason"] == "unit fixture failure"


def test_resume_rolls_best_model_back_with_latest_checkpoint_and_metrics(tmp_path):
    tracking = importlib.import_module("optcourse.tracking")
    config = trial_config(tmp_path)
    store = tracking.RunStore.create(tmp_path / "fixtures", config)
    store.record(dict(step=100, validation_loss=.5))
    store.save_checkpoint(dict(step=100, model={"weight": torch.tensor([10.])},
                               best_model={"weight": torch.tensor([8.])},
                               best_step=80, best_validation=.5))
    # A later validation improvement happened, but latest.pt has not advanced.
    store.record(dict(step=150, validation_loss=.4))
    store.save_checkpoint(dict(step=150, model={"weight": torch.tensor([15.])},
                               validation_loss=.4), best=True)
    store.finish("INTERRUPTED")
    resumed, _ = tracking.RunStore.resume(store.path, config)
    best = tracking.load_checkpoint(resumed.path / "best.pt", config)
    assert best["step"] == 80 and best["validation_loss"] == .5
    torch.testing.assert_close(best["model"]["weight"], torch.tensor([8.]))
    with (resumed.path / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        assert [int(row["step"]) for row in csv.DictReader(stream)] == [100]
