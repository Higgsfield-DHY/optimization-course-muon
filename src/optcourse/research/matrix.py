"""Registered exact-quadratic interventions and paired noisy trajectories."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from .geometry import candidate_directions,match_rms,quadratic_step_score,two_mode_boundary


def write_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def phase_diagram(include_flat=False):
    rows=[]
    for rho in list(np.geomspace(.01,.95,48))+([1.] if include_flat else []):
        gradient=torch.diag(torch.tensor([1.,rho],dtype=torch.float64))
        candidates=candidate_directions(gradient,(0,1,2,3,5,8),include_polar=True)
        candidates={name:match_rms(d,.2) for name,d in candidates.items()}
        reference=match_rms(candidates['polar'],.2)
        for curvature_ratio in np.geomspace(.01,100.,64):
            h=torch.tensor([1.,curvature_ratio],dtype=torch.float64)
            h=h/h.max()
            for name,d in candidates.items():
                score=quadratic_step_score(gradient,d,lambda x:h[:,None]*x,.1)
                r=(d[1,1]/d[0,0]).item()
                rows.append(dict(gradient_ratio=float(rho),curvature_ratio=float(curvature_ratio),variant=name,
                    direction_ratio=r,polar_error=((d-reference).norm()/reference.norm()).item(),
                    alignment=score['alignment'],curvature=score['curvature'],gain=score['gain'],
                    optimal_gain=score['optimal_gain'],optimal_eta=score['optimal_eta']))
    return rows


def fixed_momentum_phase():
    rows=[];momentum=torch.diag(torch.tensor([1.,.08],dtype=torch.float64))
    directions=candidate_directions(momentum,(1,5),include_polar=True)
    directions={name:match_rms(value,.2) for name,value in directions.items()}
    reference=directions['polar']
    for rho in np.geomspace(.01,1.,64):
        gradient=torch.diag(torch.tensor([1.,rho],dtype=torch.float64))
        for kappa in np.geomspace(.01,100.,80):
            h=torch.tensor([1.,kappa],dtype=torch.float64);h/=h.max()
            for name in ('ns1','ns5'):
                direction=directions[name];score=quadratic_step_score(gradient,direction,lambda x:h[:,None]*x,.1)
                rows.append(dict(gradient_ratio=float(rho),curvature_ratio=float(kappa),momentum_ratio=.08,variant=name,
                    direction_ratio=(direction[1,1]/direction[0,0]).item(),
                    polar_error=((direction-reference).norm()/reference.norm()).item(),**score))
    return rows


def general_matrix_interventions(exponent=2.,jitter=False):
    rows=[]
    for seed in range(10):
        gen=torch.Generator().manual_seed(seed)
        u=torch.linalg.qr(torch.randn(16,16,generator=gen,dtype=torch.float64))[0]
        v=torch.linalg.qr(torch.randn(16,16,generator=gen,dtype=torch.float64))[0]
        singular=torch.logspace(0.,-exponent,16,dtype=torch.float64)
        if jitter and exponent>0:singular=torch.sort(singular*torch.exp(.1*torch.randn(16,generator=gen,dtype=torch.float64)),descending=True).values
        gradient=u@torch.diag(singular)@v.T;gradient/=gradient.norm()
        random_basis=torch.linalg.qr(torch.randn(16,16,generator=gen,dtype=torch.float64))[0]
        base_noise=torch.randn(16,16,generator=gen,dtype=torch.float64);base_noise/=base_noise.norm()
        for condition in (1.,10.,100.):
            eigen=torch.logspace(-math.log10(condition),0.,16,dtype=torch.float64)
            for orientation in ('aligned','reversed','rotated'):
                basis=u if orientation=='aligned' else (u.flip(1) if orientation=='reversed' else random_basis)
                h=(basis*eigen)@basis.T
                for noise_ratio in (0.,.1,1.):
                    momentum=gradient+noise_ratio*base_noise
                    directions=candidate_directions(momentum,(0,1,2,3,5,8),include_polar=True)
                    reference=match_rms(directions['polar'],.2)
                    for name,raw in directions.items():
                        d=match_rms(raw,.2);score=quadratic_step_score(gradient,d,lambda x:h@x,.1)
                        rows.append(dict(seed=seed,exponent=exponent,condition=condition,orientation=orientation,noise_ratio=noise_ratio,
                            variant=name,polar_error=((d-reference).norm()/reference.norm()).item(),**score))
    return rows


def same_momentum_swaps(exponent=2.,jitter=False):
    rows=[]
    for seed in range(10):
        gen=torch.Generator().manual_seed(1000+seed)
        u=torch.linalg.qr(torch.randn(16,16,generator=gen,dtype=torch.float64))[0]
        v=torch.linalg.qr(torch.randn(16,16,generator=gen,dtype=torch.float64))[0]
        singular=torch.logspace(0.,-exponent,16,dtype=torch.float64)
        if jitter and exponent>0:singular=torch.sort(singular*torch.exp(.1*torch.randn(16,generator=gen,dtype=torch.float64)),descending=True).values
        gradient=u@torch.diag(singular)@v.T
        gradient/=gradient.norm()
        directions=candidate_directions(gradient,(1,3,5,8),include_polar=True)
        directions.update({'cubic'+name[2:]:value for name,value in candidate_directions(gradient,(3,5,8),family='cubic').items()})
        directions={name:match_rms(value,.2) for name,value in directions.items()}
        reference=directions['polar']
        for epsilon in (1e-1,1e-2,1e-3,1e-4,1e-5):
            for favored in ('ns3','ns5'):
                unit=directions[favored]/directions[favored].norm()
                action=lambda x:x-(1-epsilon)*(x*unit).sum()*unit
                for name,direction in directions.items():
                    score=quadratic_step_score(gradient,direction,action,.1)
                    rows.append(dict(seed=seed,exponent=exponent,epsilon=epsilon,favored=favored,variant=name,
                        polar_error=((direction-reference).norm()/reference.norm()).item(),**score))
    return rows


def _trajectory(condition,orientation,noise_ratio,seed,variant,rate,steps=500,save=False):
    gen=torch.Generator().manual_seed(seed)
    u=torch.linalg.qr(torch.randn(16,16,generator=gen,dtype=torch.float64))[0]
    v=torch.linalg.qr(torch.randn(16,16,generator=gen,dtype=torch.float64))[0]
    eigen=torch.logspace(-math.log10(condition),0.,16,dtype=torch.float64)
    basis=u if orientation=='aligned' else u.flip(1)
    h=(basis*eigen)@basis.T
    w=u@torch.diag(torch.logspace(0.,-2.,16,dtype=torch.float64))@v.T
    w/=math.sqrt((.5*(w*(h@w)).sum()).item())
    buffer=torch.zeros_like(w);initial_gradient=(h@w).norm().item();records=[]
    for step in range(steps+1):
        if save and (step%10==0 or step==steps):
            records.append(dict(step=step,objective=(.5*(w*(h@w)).sum()).item(),gradient_norm=(h@w).norm().item()))
        if step==steps:break
        gradient=h@w
        if noise_ratio:
            noise=torch.randn(w.shape,generator=gen,dtype=w.dtype)
            gradient=gradient+noise*noise_ratio*initial_gradient/math.sqrt(w.numel())
        buffer=.95*buffer+.05*gradient;momentum=.95*buffer+.05*gradient
        if variant=='polar':raw=candidate_directions(momentum,(0,),include_polar=True)['polar']
        else:raw=candidate_directions(momentum,(int(variant[2:]),))[variant]
        d=match_rms(raw,.2)
        factor=.01+.99*.5*(1+math.cos(math.pi*step/steps))
        w=w-rate*factor*d
    return (.5*(w*(h@w)).sum()).item(),records


def trajectories():
    tuning=[];final=[];curves=[]
    variants=('ns0','ns1','ns3','ns5','polar')
    for condition in (1.,100.):
        for orientation in ('aligned','reversed'):
            for noise_ratio in (0.,.1):
                for variant in variants:
                    candidates=[]
                    for rate in (.0003,.001,.003,.01,.03,.1,.3,1.):
                        loss,_=_trajectory(condition,orientation,noise_ratio,101,variant,rate)
                        row=dict(condition=condition,orientation=orientation,noise_ratio=noise_ratio,variant=variant,lr=rate,objective=loss)
                        tuning.append(row);candidates.append(row)
                    winner=min(candidates,key=lambda row:(row['objective'],row['lr']))
                    if winner['lr'] in (.0003,1.):
                        rate=winner['lr']/3 if winner['lr']==.0003 else 3.
                        loss,_=_trajectory(condition,orientation,noise_ratio,101,variant,rate)
                        row=dict(condition=condition,orientation=orientation,noise_ratio=noise_ratio,variant=variant,lr=rate,objective=loss)
                        tuning.append(row);candidates.append(row)
                    rate=min(candidates,key=lambda row:(row['objective'],row['lr']))['lr']
                    for seed in range(10):
                        loss,trace=_trajectory(condition,orientation,noise_ratio,seed,variant,rate,save=True)
                        keys=dict(condition=condition,orientation=orientation,noise_ratio=noise_ratio,variant=variant,lr=rate,seed=seed)
                        final.append(keys|dict(objective=loss));curves.extend(keys|row for row in trace)
                print(f'matrix complete: condition={condition} orientation={orientation} noise={noise_ratio}',flush=True)
    return tuning,final,curves


def run_synthetic(output):
    torch.set_num_threads(1);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    phase=phase_diagram();write_rows(output/'phase.csv',phase)
    general=general_matrix_interventions();write_rows(output/'frozen_matrix.csv',general)
    tune,final,curves=trajectories()
    for name,rows in [('tuning',tune),('final',final),('curves',curves)]:write_rows(output/f'{name}.csv',rows)
    summary=dict(phase_rows=len(phase),matrix_intervention_rows=len(general),trajectory_runs=len(final),paired_seeds=10,
                 fixed_maximum_curvature=1.,rms=.2,methods=['ns0','ns1','ns3','ns5','polar'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def run_extended(output):
    torch.set_num_threads(1)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    phase=phase_diagram(include_flat=True)
    write_rows(output/'phase.csv',phase)
    fixed=fixed_momentum_phase()
    write_rows(output/'phase_fixed_momentum.csv',fixed)
    matrix=[]
    swaps=[]
    for exponent in (0.,.5,1.,2.):
        matrix.extend(general_matrix_interventions(exponent,jitter=True))
        swaps.extend(same_momentum_swaps(exponent,jitter=True))
    write_rows(output/'frozen_matrix_controlled.csv',matrix)
    write_rows(output/'same_momentum_swaps.csv',swaps)
    return dict(phase_rows=len(phase),fixed_momentum_rows=len(fixed),matrix_rows=len(matrix),swap_rows=len(swaps))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/research_v2/synthetic');parser.add_argument('--execute',action='store_true');parser.add_argument('--extended',action='store_true')
    args=parser.parse_args()
    run=run_extended if args.extended else run_synthetic
    print(json.dumps(run(args.output) if args.execute else dict(status='NOT_RUN',output=args.output)))
