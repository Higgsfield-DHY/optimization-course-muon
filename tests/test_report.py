import importlib

import pytest

from test_tracking import trial_config


def test_summary_excludes_failed_runs_and_groups_seed_replicates(tmp_path):
    tracking = importlib.import_module("optcourse.tracking")
    report = importlib.import_module("optcourse.report")
    root = tmp_path / "fixtures"
    for seed, objective, status in [(0, 2., "COMPLETED"), (1, 4., "COMPLETED"), (2, 0., "FAILED")]:
        store = tracking.RunStore.create(root, trial_config(tmp_path, seed))
        store.record(dict(step=3, objective=objective, elapsed_seconds=1.))
        store.finish(status)
    result = report.summarize(root, tmp_path / "reports", make_plots=False)
    assert result["completed_runs"] == 2
    assert result["discovered_runs"] == 3
    assert result["status_counts"] == {"COMPLETED": 2, "FAILED": 1}
    assert result["excluded_count"] == 1
    assert result["excluded_runs"][0]["seed"] == 2
    assert len(result["groups"]) == 1
    assert result["groups"][0]["n"] == 2
    assert result["groups"][0]["objective_mean"] == 3.
    assert result["groups"][0]["objective_std"] == pytest.approx(2 ** .5)


def test_no_completed_runs_produces_no_report_directory(tmp_path):
    report = importlib.import_module("optcourse.report")
    with pytest.raises(ValueError, match="completed"):
        report.summarize(tmp_path / "missing", tmp_path / "output", make_plots=False)
    assert not (tmp_path / "output").exists()


def test_repeated_same_seed_is_not_counted_as_an_independent_replicate(tmp_path):
    tracking = importlib.import_module("optcourse.tracking")
    report = importlib.import_module("optcourse.report")
    for _ in range(2):
        store = tracking.RunStore.create(tmp_path / "fixtures", trial_config(tmp_path))
        store.record(dict(step=3, objective=2.))
        store.finish("COMPLETED")
    with pytest.raises(ValueError, match="Duplicate seed"):
        report.summarize(tmp_path / "fixtures", tmp_path / "output", make_plots=False)


def test_same_source_with_different_prepared_splits_is_not_one_group(tmp_path):
    tracking = importlib.import_module("optcourse.tracking")
    report = importlib.import_module("optcourse.report")
    config = trial_config(tmp_path)
    config["run"]["task"] = "language"
    config["language"] = dict(data_dir="same_path")
    config.pop("matrix")
    for prepared_identity in ("split_A", "split_B"):
        store = tracking.RunStore.create(tmp_path / "fixtures", config)
        store.update(dataset_source_sha256="same_original_source",
                     prepared_dataset_fingerprint=prepared_identity)
        store.record(dict(step=3, validation_loss=2., elapsed_seconds=1.))
        store.finish("COMPLETED")
    result = report.summarize(tmp_path / "fixtures", tmp_path / "reports", make_plots=False)
    assert len(result["groups"]) == 2
