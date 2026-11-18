"""Opt-in actual training verification, never ordinary unit-test evidence."""
import json

import pytest
import torch


@pytest.mark.training
def test_language_training_resume_matches_uninterrupted_in_temporary_fixture(tmp_path, monkeypatch):
    from optcourse.config import load_config
    from optcourse.data import prepare_data
    from optcourse.experiments import run_experiment
    from optcourse.tracking import RunStore
    source=tmp_path/'input.txt'
    source.write_bytes(b'abcde '*200)
    prepare_data(source,tmp_path/'data')
    cfg=tmp_path/'config.toml'
    cfg.write_text('[run]\nname="recovery_fixture"\ntask="language"\ndevice="cpu"\nphase="verification"\noutput_root="fixtures"\n[optimizer]\nname="muon"\nnormalization="max"\n[language]\ndata_dir="data"\nn_layer=1\nn_embd=8\nn_head=2\nblock_size=4\nbatch_size=2\ngrad_accum=2\nmax_steps=8\nwarmup_steps=2\neval_every=2\ncheckpoint_every=4\nlog_every=2\neval_mode="full"\n',encoding='utf-8')
    config=load_config(cfg)
    uninterrupted=run_experiment(config)
    original=RunStore.save_checkpoint
    fired=False
    def interrupt_after_saved_step(self,payload,best=False):
        nonlocal fired
        original(self,payload,best)
        if not best and payload['step']==4 and not fired:
            fired=True
            raise KeyboardInterrupt()
    monkeypatch.setattr(RunStore,'save_checkpoint',interrupt_after_saved_step)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(config)
    candidate=next(path for path in (tmp_path/'fixtures').iterdir() if path != uninterrupted)
    monkeypatch.setattr(RunStore,'save_checkpoint',original)
    resumed=run_experiment(config,resume=candidate)
    first=torch.load(uninterrupted/'latest.pt',weights_only=False)
    second=torch.load(resumed/'latest.pt',weights_only=False)
    for key,value in first['model'].items():
        torch.testing.assert_close(value,second['model'][key],rtol=0,atol=0)
    assert first['best_step']==second['best_step']
    assert first['best_validation']==second['best_validation']
    assert torch.equal(first['rng']['sampler'],second['rng']['sampler'])
    assert json.loads((resumed/'metadata.json').read_text(encoding='utf-8'))['status']=='COMPLETED'
