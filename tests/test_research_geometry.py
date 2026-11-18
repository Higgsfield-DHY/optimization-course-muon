import importlib

import pytest
import torch


def api():
    return importlib.import_module('optcourse.research.geometry')


def test_ns_candidates_preserve_input_and_scalar_polynomial():
    g=api()
    matrix=torch.diag(torch.tensor([3.,1.],dtype=torch.float64))
    original=matrix.clone()
    rows=g.candidate_directions(matrix,(0,1,3,5),include_polar=True)
    values=matrix.diag()/matrix.norm()
    assert torch.equal(rows['ns0'].diag(),values)
    for step in range(1,6):
        values=3.4445*values-4.7750*values**3+2.0315*values**5
        if step in (1,3,5):
            assert torch.allclose(rows[f'ns{step}'].diag(),values,atol=1e-12,rtol=1e-12)
    assert torch.equal(original,matrix)
    assert torch.allclose(rows['polar'],torch.eye(2,dtype=torch.float64))


def test_match_rms_matches_energy_and_retains_zero():
    g=api(); x=torch.randn(3,5,dtype=torch.float64)
    y=g.match_rms(x,.2)
    assert y.square().mean().sqrt().item()==pytest.approx(.2)
    assert torch.allclose(y/y.norm(),x/x.norm())
    assert torch.equal(g.match_rms(torch.zeros_like(x),.2),torch.zeros_like(x))


def test_large_zero_rms_direction_stays_finite():
    g=api();x=torch.zeros(128,512)
    assert torch.equal(g.match_rms(x,.2),x)


def test_quadratic_gain_uses_full_cross_direction_curvature():
    g=api()
    h=torch.tensor([[3.,1.,0.,0.],[1.,2.,.5,0.],[0.,.5,4.,1.],[0.,0.,1.,2.]],dtype=torch.float64)
    w=torch.tensor([[1.,-.5],[.8,.3]],dtype=torch.float64)
    direction=torch.tensor([[.1,.3],[-.2,.7]],dtype=torch.float64)
    gradient=(h@w.flatten()).reshape_as(w)
    eta=.12
    score=g.quadratic_step_score(gradient,direction,lambda d:(h@d.flatten()).reshape_as(d),eta)
    objective=lambda z:.5*torch.dot(z.flatten(),h@z.flatten()).item()
    assert score['gain']==pytest.approx(objective(w)-objective(w-eta*direction),abs=1e-12)
    assert score['curvature']==pytest.approx(torch.dot(direction.flatten(),h@direction.flatten()).item())


def test_line_optimized_gain_is_scale_invariant():
    g=api(); grad=torch.tensor([[2.,.7]],dtype=torch.float64);d=torch.tensor([[.8,.4]],dtype=torch.float64)
    h=lambda x:x*torch.tensor([[1.,3.]],dtype=torch.float64)
    a=g.quadratic_step_score(grad,d,h,.1);b=g.quadratic_step_score(grad,11*d,h,.1)
    assert a['optimal_gain']==pytest.approx(b['optimal_gain'])
    assert a['optimal_eta']==pytest.approx(11*b['optimal_eta'])


def test_two_mode_boundary_predicts_reversed_ranking():
    g=api();rho=.15;r1=.85;r2=.3
    boundary=g.two_mode_boundary(rho,r1,r2)
    assert g.two_mode_gain(rho,boundary*.4,r1)>g.two_mode_gain(rho,boundary*.4,r2)
    assert g.two_mode_gain(rho,boundary*4,r1)<g.two_mode_gain(rho,boundary*4,r2)
    assert g.two_mode_gain(rho,boundary,r1)==pytest.approx(g.two_mode_gain(rho,boundary,r2))


def test_nonpositive_curvature_reports_finite_step_without_unbounded_optimum():
    g=api();x=torch.ones(2,2,dtype=torch.float64)
    row=g.quadratic_step_score(x,x,lambda d:-d,.1)
    assert row['gain']==pytest.approx(.42)
    assert row['optimal_eta'] is None and row['optimal_gain'] is None


def test_polar_and_stable_rank_support_rank_deficiency():
    g=api();x=torch.diag(torch.tensor([3.,0.],dtype=torch.float64))
    row=g.candidate_directions(x,(1,),include_polar=True)
    assert torch.equal(row['polar'],torch.diag(torch.tensor([1.,0.],dtype=torch.float64)))
    assert g.stable_rank(x)==pytest.approx(1.)
    assert g.stable_rank(torch.zeros_like(x))==0.


def test_classical_cubic_reference_contracts_singular_residual():
    g=api();x=torch.diag(torch.tensor([1.,.8],dtype=torch.float64))
    rows=g.candidate_directions(x,(1,3,5,8),family='cubic')
    errors=[(rows[f'ns{k}']-torch.eye(2,dtype=x.dtype)).norm().item() for k in (1,3,5,8)]
    assert all(b<a for a,b in zip(errors,errors[1:]))
