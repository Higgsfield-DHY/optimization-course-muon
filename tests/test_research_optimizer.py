import importlib

import pytest
import torch
from torch import nn


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(8,3);self.hidden=nn.Linear(3,3);self.head=nn.Linear(3,8,bias=False);self.head.weight=self.embedding.weight


def api():return importlib.import_module('optcourse.research.optimizer')


def grads(model):
    gen=torch.Generator().manual_seed(19)
    for p in model.parameters():p.grad=torch.randn(p.shape,generator=gen)


def test_prepared_steps_are_pure_and_commit_only_once():
    g=api();model=Toy();grads(model)
    opt=g.ResearchOptimizer(model,'ns5',lr=.001,aux_lr=.001,weight_decay=.1)
    before={n:p.clone() for n,p in model.named_parameters()}
    prepared=opt.prepare(depths=(1,3,5))
    assert len(opt.state)==0
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters())
    opt.commit(prepared,'ns3')
    assert all(torch.allclose(p,before[n]-prepared.deltas['ns3'][n]) for n,p in model.named_parameters())
    with pytest.raises(ValueError,match='already|stale'):opt.commit(prepared,'ns3')


def test_all_candidates_share_auxiliary_updates_and_hidden_rms():
    g=api();model=Toy();grads(model)
    opt=g.ResearchOptimizer(model,'ns5',lr=.003,aux_lr=.001,weight_decay=0.)
    prepared=opt.prepare(depths=(1,3,5))
    for name in opt.auxiliary_names:
        assert torch.equal(prepared.deltas['ns1'][name],prepared.deltas['ns5'][name])
    for name in opt.hidden_names:
        for candidate in prepared.deltas.values():
            assert candidate[name].square().mean().sqrt().item()==pytest.approx(.003*.2,rel=1e-5)
    assert set(opt.hidden_names).isdisjoint(opt.auxiliary_names)
    assert set(opt.hidden_names)|set(opt.auxiliary_names)==set(dict(model.named_parameters()))


def test_adamw_matches_native_parameter_group_updates():
    g=api();torch.manual_seed(3);model=Toy();reference=Toy();reference.load_state_dict(model.state_dict())
    opt=g.ResearchOptimizer(model,'adamw',lr=.002,aux_lr=.001,weight_decay=.1)
    names=dict(reference.named_parameters())
    groups=[]
    for name,p in names.items():groups.append(dict(params=[p],lr=.002 if name in opt.hidden_names else .001,weight_decay=.1 if p.ndim>=2 else 0.))
    native=torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8)
    for _ in range(3):
        grads(model)
        for name,p in model.named_parameters():names[name].grad=p.grad.clone()
        prepared=opt.prepare();opt.commit(prepared,'adamw');native.step()
    assert all(torch.allclose(p,names[name],atol=2e-7,rtol=1e-6) for name,p in model.named_parameters())


def test_polar_error_selector_scores_the_same_normalized_candidates():
    g=api()
    from optcourse.research.geometry import candidate_directions,match_rms
    model=nn.Sequential(nn.Linear(2,2,bias=False));model[0].weight.grad=torch.diag(torch.tensor([1.,.08]))
    opt=g.ResearchOptimizer(model,'spectral',lr=.001,aux_lr=.001,weight_decay=0.)
    prepared=opt.prepare(depths=(1,3,5));scores=g.spectral_candidate_errors(model,opt,prepared,1.)
    reference=match_rms(torch.eye(2),.2)
    for key in ('ns1','ns3','ns5'):
        expected=((prepared.deltas[key]['0.weight']/.001-reference).norm()/reference.norm()).item()
        assert scores[key]==pytest.approx(expected,rel=1e-5,abs=1e-7)
    assert len(opt.state)==0
