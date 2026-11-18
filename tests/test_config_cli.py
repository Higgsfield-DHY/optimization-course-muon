import importlib
import json
from pathlib import Path

import pytest


def config_file(tmp_path, extra=""):
    path = tmp_path / "trial.toml"
    path.write_text('''[run]
name = "trial"
task = "matrix"
device = "cpu"
output_root = "runs"
[optimizer]
name = "muon"
lr = 0.02
''' + extra, encoding="utf-8")
    return path


def test_run_without_execute_only_plans_and_creates_no_results(tmp_path, capsys):
    cli = importlib.import_module("optcourse.cli")
    code = cli.main(["run", "--config", str(config_file(tmp_path))])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["status"] == "NOT_RUN"
    assert result["execution_requested"] is False
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("extra", ['[matrix]\nsteps = 0\n', '[matrix]\nconditon_number = 10\n'])
def test_invalid_or_misspelled_configuration_is_rejected(tmp_path, extra):
    config = importlib.import_module("optcourse.config")
    with pytest.raises(ValueError):
        config.load_config(config_file(tmp_path, extra))


def test_seed_override_is_effective_but_summary_identity_excludes_seed(tmp_path):
    config = importlib.import_module("optcourse.config")
    path = config_file(tmp_path)
    first, second = config.load_config(path, seed=3), config.load_config(path, seed=8)
    assert first["run"]["seed"] == 3
    assert second["run"]["seed"] == 8
    assert config.fingerprint(first) != config.fingerprint(second)
    assert config.fingerprint(first, ignore_seed=True) == config.fingerprint(second, ignore_seed=True)
    assert Path(first["run"]["output_root"]) == tmp_path / "runs"


def test_language_width_must_be_divisible_by_number_of_heads(tmp_path):
    config = importlib.import_module("optcourse.config")
    path = tmp_path / "bad.toml"
    path.write_text('[run]\nname="bad"\ntask="language"\n[language]\nn_embd=10\nn_head=3\n')
    with pytest.raises(ValueError):
        config.load_config(path)
