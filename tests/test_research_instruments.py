import importlib

import pytest
import torch
from torch import nn


def api():return importlib.import_module('optcourse.research.instruments')


def test_virtual_step_restores_parameters_gradients_buffers_rng_and_mode_on_exception():
    g=api();model=nn.Sequential(nn.Linear(2,2),nn.BatchNorm1d(2))
    model.train();model[1].eval()
    for p in model.parameters():p.grad=torch.ones_like(p)
    state={k:v.clone() for k,v in model.state_dict().items()}
    grads={n:p.grad.clone() for n,p in model.named_parameters()}
    rng=torch.get_rng_state().clone()
    delta={n:torch.full_like(p,.1) for n,p in model.named_parameters()}
    with pytest.raises(RuntimeError,match='probe failure'):
        with g.virtual_step(model,delta):
            model.eval();model[1].running_mean.add_(3)
            for p in model.parameters():p.grad=None
            torch.rand(10)
            raise RuntimeError('probe failure')
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert all(torch.equal(grads[n],p.grad) for n,p in model.named_parameters())
    assert torch.equal(rng,torch.get_rng_state())
    assert model.training and not model[1].training


def test_neural_hvp_includes_cross_parameter_curvature():
    g=api()
    class Coupled(nn.Module):
        def __init__(self):
            super().__init__();self.a=nn.Parameter(torch.tensor([2.],dtype=torch.float64));self.b=nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    model=Coupled();directions={'x':{'a':torch.tensor([.2],dtype=torch.float64),'b':torch.tensor([-.1],dtype=torch.float64)}}
    loss=lambda m,batch:.5*(m.a+2*m.b).square().sum()
    scores=g.directional_geometry(model,directions,None,loss)
    assert scores['x']['alignment']==pytest.approx(0.)
    assert scores['x']['curvature']==pytest.approx(0.)
    assert all(p.grad is None for p in model.parameters())


def test_direction_is_frozen_even_if_caller_built_it_from_parameters():
    g=api();model=nn.Linear(1,1,bias=False,dtype=torch.float64)
    with torch.no_grad():model.weight.fill_(2.)
    deltas={'weight':model.weight.square()}
    loss=lambda m,b:.5*m.weight.square().sum()
    row=g.directional_geometry(model,{'candidate':deltas},None,loss)['candidate']
    assert row['alignment']==pytest.approx(8.)
    assert row['curvature']==pytest.approx(16.)


def test_neural_quadratic_prediction_equals_actual_virtual_gain():
    g=api();model=nn.Linear(2,1,bias=False,dtype=torch.float64)
    with torch.no_grad():model.weight.copy_(torch.tensor([[1.,-.3]],dtype=torch.float64))
    x=torch.tensor([[1.,2.],[2.,-1.],[.3,.7]],dtype=torch.float64);y=torch.tensor([[.2],[.4],[-.3]],dtype=torch.float64)
    loss=lambda m,b:.5*(m(b[0])-b[1]).square().mean()
    candidates={'a':{'weight':torch.tensor([[.05,-.02]],dtype=torch.float64)},'b':{'weight':torch.tensor([[-.02,.01]],dtype=torch.float64)}}
    rows=g.directional_geometry(model,candidates,(x,y),loss)
    actual=g.evaluate_candidates(model,candidates,(x,y),loss)
    for name in candidates:
        assert rows[name]['predicted_gain']==pytest.approx(actual[name]['gain'],abs=1e-12)
        assert rows[name]['linear_gain']==pytest.approx(rows[name]['alignment'])
        assert rows[name]['cross_curvature'][name]==pytest.approx(rows[name]['curvature'])
    assert rows['a']['cross_curvature']['b']==pytest.approx(rows['b']['cross_curvature']['a'])


def test_transformer_directional_curvature_uses_double_backward_capable_attention():
    g=api()
    from optcourse.research.language import ResearchTransformer
    from torch.nn import functional as F
    model=ResearchTransformer(depth=1,width=8,heads=2,context=4,vocabulary=16).double()
    batch=(torch.randint(0,16,(1,4)),torch.randint(0,16,(1,4)))
    deltas={n:torch.randn_like(p)*.001 for n,p in model.named_parameters()}
    loss=lambda m,b:F.cross_entropy(m(b[0]).reshape(-1,16),b[1].reshape(-1))
    result=g.directional_geometry(model,{'candidate':deltas},batch,loss)
    assert torch.isfinite(torch.tensor(result['candidate']['curvature']))
