import importlib

import numpy as np
import pytest
import torch


@pytest.mark.training
@pytest.mark.parametrize('method',['ns3','adaptive'])
def test_research_training_resume_matches_continuous_updates(tmp_path,method):
    data=importlib.import_module('optcourse.research.data')
    runner=importlib.import_module('optcourse.research.training')
    root=tmp_path/'data'
    gen=np.random.default_rng(7)
    data.save_arrays(root,{k:gen.integers(0,32,n,dtype=np.uint16) for k,n in [('train',300),('val',35),('test',31)]},dict(tokenizer='fixture',vocabulary=32))
    dataset=data.TokenDataset(root)
    config=dict(seed=5,method=method,model=dict(depth=1,width=16,heads=2,context=8,vocabulary=32),steps=4,
        batch_size=2,accumulation=1,lr=.001,aux_lr=.001,weight_decay=.01,device='cpu',precision='float32',
        ns_dtype='float32',validation_every=2,validation_targets=None,log_every=1,warmup_steps=0,
        probe_interval=2,probe_batch_size=2,phase='verification')
    a=runner.run_training(config,dataset,tmp_path/'continuous')
    b=runner.run_training(config,dataset,tmp_path/'resumed',max_updates=2)
    assert b['status']=='INTERRUPTED'
    c=runner.run_training(config,dataset,tmp_path/'resumed')
    assert a['status']==c['status']=='COMPLETED'
    full=torch.load(tmp_path/'continuous/latest.pt',weights_only=False,map_location='cpu')
    resumed=torch.load(tmp_path/'resumed/latest.pt',weights_only=False,map_location='cpu')
    assert all(torch.equal(value,resumed['model'][name]) for name,value in full['model'].items())
    assert torch.equal(full['sampler'],resumed['sampler'])
    assert full['best_step']==resumed['best_step']
    assert full['best_validation']==resumed['best_validation']


def test_research_preview_has_no_run_directory_side_effect(tmp_path):
    module=importlib.import_module('optcourse.research.protocol')
    result=module.preview(tmp_path/'study')
    assert result['status']=='NOT_RUN'
    assert result['aggregate_gpu_limit_seconds']==28800
    assert result['prior_gpu_seconds']==pytest.approx(3507.4571810862326)
    assert not (tmp_path/'study').exists()


def test_replay_shuffles_each_mixed_stage_and_preserves_realized_depth_counts():
    from collections import Counter
    from optcourse.research.protocol import stage_matched_replay

    choices=[1,3,1,5,3,3,1,5,3,1]
    selections=[{'step':100*(i+1),'choice':f'ns{k}'} for i,k in enumerate(choices)]
    replay=stage_matched_replay(selections,0)
    for indices in (slice(0,3),slice(3,6),slice(6,8)):
        assert Counter(replay[indices])==Counter(choices[indices])
        assert replay[indices]!=choices[indices]
    assert replay[-2:]==choices[-2:]


@pytest.mark.training
def test_failed_anchor_write_does_not_consume_a_probe_selection(tmp_path,monkeypatch):
    data=importlib.import_module('optcourse.research.data')
    runner=importlib.import_module('optcourse.research.training')
    root=tmp_path/'data'
    gen=np.random.default_rng(17)
    data.save_arrays(root,{k:gen.integers(0,32,n,dtype=np.uint16) for k,n in
                           [('train',300),('val',35),('test',31)]},dict(tokenizer='fixture',vocabulary=32))
    dataset=data.TokenDataset(root)
    config=dict(seed=5,method='adaptive',model=dict(depth=1,width=8,heads=2,context=4,vocabulary=32),steps=4,
        batch_size=2,accumulation=1,lr=.001,aux_lr=.001,weight_decay=.01,device='cpu',precision='float32',
        ns_dtype='float32',validation_every=2,validation_targets=None,log_every=1,warmup_steps=0,
        probe_interval=2,probe_batch_size=2,phase='verification',save_anchors=True,anchor_steps=[2])
    runner.run_training(config,dataset,tmp_path/'continuous')
    original=runner.atomic_torch
    def fail_on_anchor(path,value):
        if path.parent.name=='anchors':
            raise OSError('injected anchor write failure')
        original(path,value)
    with monkeypatch.context() as patch:
        patch.setattr(runner,'atomic_torch',fail_on_anchor)
        with pytest.raises(OSError,match='injected anchor write failure'):
            runner.run_training(config,dataset,tmp_path/'recovered')
    after_failure=torch.load(tmp_path/'recovered/latest.pt',weights_only=True)
    assert after_failure['step']==1
    assert after_failure['selections']==[]
    runner.run_training(config,dataset,tmp_path/'recovered')
    continuous=torch.load(tmp_path/'continuous/latest.pt',weights_only=True)
    recovered=torch.load(tmp_path/'recovered/latest.pt',weights_only=True)
    assert [row['choice'] for row in continuous['selections']]==[row['choice'] for row in recovered['selections']]
    assert all(torch.equal(value,recovered['model'][name]) for name,value in continuous['model'].items())


@pytest.mark.training
def test_frozen_anchor_captures_base_model_and_exact_applied_candidates(tmp_path):
    data=importlib.import_module('optcourse.research.data');runner=importlib.import_module('optcourse.research.training')
    language=importlib.import_module('optcourse.research.language')
    parts={k:np.arange(n,dtype=np.uint16)%32 for k,n in [('train',300),('val',35),('test',31)]}
    data.save_arrays(tmp_path/'data',parts,dict(tokenizer='fixture',vocabulary=32))
    config=dict(seed=5,method='ns3',model=dict(depth=1,width=16,heads=2,context=8,vocabulary=32),steps=2,
        batch_size=2,accumulation=1,lr=.001,aux_lr=.001,weight_decay=.01,device='cpu',precision='float32',
        ns_dtype='float32',validation_every=2,validation_targets=None,log_every=1,warmup_steps=0,
        probe_interval=2,probe_batch_size=2,phase='verification',save_anchors=True,anchor_steps=[2])
    runner.run_training(config,data.TokenDataset(tmp_path/'data'),tmp_path/'run')
    anchor=torch.load(tmp_path/'run/anchors/step-2.pt',weights_only=False,map_location='cpu')
    latest=torch.load(tmp_path/'run/latest.pt',weights_only=False,map_location='cpu')
    assert anchor['base_completed_updates']==1
    assert set(anchor['candidates'])=={'ns1','ns3','ns5'}
    model=language.ResearchTransformer(**config['model']);model.load_state_dict(anchor['model'])
    for name,p in model.named_parameters():
        assert torch.equal(p-anchor['candidates']['ns3'][name],latest['model'][name])
    from optcourse.research.geometry import candidate_directions,match_rms
    for name,momentum in anchor['momenta'].items():
        direction=match_rms(candidate_directions(momentum,(3,))['ns3'],.2)
        expected=config['lr']*anchor['lr_factor']*(direction+config['weight_decay']*anchor['model'][name])
        assert torch.allclose(expected,anchor['candidates']['ns3'][name],atol=1e-9,rtol=1e-5)
