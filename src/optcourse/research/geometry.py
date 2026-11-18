"""Reference spectral maps and scale-invariant directional descent scores."""

import math
from typing import Callable

import torch


@torch.no_grad()
def candidate_directions(matrix: torch.Tensor, depths=(0,1,3,5), *, include_polar=False,family='empirical'):
    if matrix.ndim != 2 or not matrix.is_floating_point():
        raise ValueError('Expected a floating-point matrix')
    depths=tuple(sorted(set(depths)))
    if not depths or any(type(k) is not int or k<0 for k in depths):
        raise ValueError('Depths must be nonnegative integers')
    if not torch.isfinite(matrix).all():
        raise ValueError('Input contains nonfinite values')
    if family not in {'empirical','cubic'}:raise ValueError('Unknown polynomial family')
    x=matrix.clone()
    transposed=x.shape[0]>x.shape[1]
    if transposed:x=x.T
    x=x/x.norm().clamp_min(torch.finfo(x.dtype).tiny)
    result={}
    if 0 in depths:result['ns0']=(x.T if transposed else x).clone()
    for k in range(1,max(depths)+1):
        gram=x@x.T
        if family=='cubic':x=1.5*x-.5*gram@x
        else:x=3.4445*x+(-4.7750*gram+2.0315*(gram@gram))@x
        if k in depths:result[f'ns{k}']=(x.T if transposed else x).clone()
    if include_polar:
        u,s,vh=torch.linalg.svd(matrix,full_matrices=False)
        active=s>torch.finfo(s.dtype).eps*max(matrix.shape)*s.max()
        result['polar']=u[:,active]@vh[active,:]
    return result


@torch.no_grad()
def match_rms(direction: torch.Tensor,target=.2):
    if not math.isfinite(target) or target<=0:
        raise ValueError('Target RMS must be finite and positive')
    norm=direction.norm()
    denominator=torch.where(norm>0,norm,torch.ones_like(norm))
    return direction*(target*math.sqrt(direction.numel())/denominator)


@torch.no_grad()
def quadratic_step_score(gradient: torch.Tensor,direction: torch.Tensor,
                         hessian_action: Callable[[torch.Tensor],torch.Tensor],eta: float):
    a=torch.sum(gradient*direction).item()
    b=torch.sum(direction*hessian_action(direction)).item()
    norm2=direction.square().sum().item()
    gradnorm=gradient.norm().item()
    optimal_eta=a/b if a>0 and b>0 else None
    return dict(alignment=a,curvature=b,gain=eta*a-.5*eta**2*b,
                optimal_eta=optimal_eta,optimal_gain=a*a/(2*b) if optimal_eta is not None else None,
                normalized_curvature=b/norm2 if norm2>0 else 0.,
                cosine=a/(math.sqrt(norm2)*gradnorm) if norm2>0 and gradnorm>0 else 0.)


def two_mode_gain(gradient_ratio: float,curvature_ratio: float,direction_ratio: float):
    if min(gradient_ratio,curvature_ratio,direction_ratio)<0 or curvature_ratio==0:
        raise ValueError('Ratios must be nonnegative, with positive curvature')
    return (1+gradient_ratio*direction_ratio)**2/(2*(1+curvature_ratio*direction_ratio**2))


def two_mode_boundary(gradient_ratio: float,first_ratio: float,second_ratio: float):
    rho,r,s=gradient_ratio,first_ratio,second_ratio
    if min(rho,r,s)<=0 or r==s:
        raise ValueError('Use distinct positive direction ratios and a positive gradient ratio')
    return (2*rho+rho*rho*(r+s))/(r+s+2*rho*r*s)


@torch.no_grad()
def stable_rank(matrix: torch.Tensor):
    singular=torch.linalg.svdvals(matrix)
    spectral2=singular[0].square().item() if singular.numel() else 0.
    return matrix.square().sum().item()/spectral2 if spectral2>0 else 0.
