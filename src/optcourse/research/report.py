"""Figures and seed-level summaries from registered research evidence."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from .geometry import two_mode_boundary

COLORS=dict(adamw='#34383c',momentum='#998b47',ns0='#998b47',ns1='#759c51',ns3='#3a91aa',ns5='#cb6a3b',
    adaptive='#9c4874',replay='#7f7fba',spectral='#58857d',adam_rms='#7d7d7d',canonical5='#bb9148',polar='#654a8b')
LABELS=dict(adamw='AdamW',momentum='Momentum',ns0='Momentum',ns1='NS1',ns3='NS3',ns5='NS5',
    adaptive='Probe-NS',replay='Stage replay',spectral='Error-NS',adam_rms='RMS-Adam',canonical5='Canonical NS5',polar='Exact polar')


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def style():
    plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'normal',
        'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42,'ps.fonttype':42,'lines.linewidth':1.5,'legend.frameon':False})


def save(fig,output,name):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/f'{name}.pdf',bbox_inches='tight');fig.savefig(output/f'{name}.png',dpi=200,bbox_inches='tight');plt.close(fig)


def synthetic_figures(root,output):
    style();root=Path(root)
    phase=read_csv(root/'synthetic_extended/phase_fixed_momentum.csv')
    rhos=sorted({float(r['gradient_ratio']) for r in phase});kappas=sorted({float(r['curvature_ratio']) for r in phase})
    table={(float(r['gradient_ratio']),float(r['curvature_ratio']),r['variant']):r for r in phase}
    ratio=np.asarray([[np.log10(float(table[(rho,kappa,'ns5')]['optimal_gain'])/float(table[(rho,kappa,'ns1')]['optimal_gain'])) for rho in rhos] for kappa in kappas])
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
    limit=max(.03,float(np.max(np.abs(ratio))));norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    image=axes[0].pcolormesh(rhos,kappas,ratio,shading='auto',cmap='RdBu_r',norm=norm,rasterized=True)
    boundary=[]
    for rho in rhos[:-1]:
        r=float(table[(rho,kappas[0],'ns5')]['direction_ratio']);s=float(table[(rho,kappas[0],'ns1')]['direction_ratio'])
        boundary.append(two_mode_boundary(rho,r,s))
    axes[0].plot(rhos[:-1],boundary,color='#222222',lw=1,ls='--',label='Analytic boundary')
    axes[0].set(xscale='log',yscale='log',xlabel='Gradient ratio $\\rho$',ylabel='Curvature ratio $\\kappa$',title='(a) Fixed momentum, equal RMS')
    axes[0].legend(fontsize=8,loc='lower left');fig.colorbar(image,ax=axes[0],label='$\\log_{10}(J_5/J_1)$',shrink=.85)
    swaps=read_csv(root/'synthetic_extended/same_momentum_swaps.csv');groups=defaultdict(dict)
    for row in swaps:
        if row['variant'] in ('ns3','ns5'):
            groups[(float(row['exponent']),int(row['seed']),float(row['epsilon']),row['favored'])][row['variant']]=float(row['optimal_gain'])
    for exponent,ls in [(1.,'-'),(2.,'--')]:
        for favored,color in [('ns3',COLORS['ns3']),('ns5',COLORS['ns5'])]:
            xs=[];ys=[]
            for epsilon in sorted({key[2] for key in groups},reverse=True):
                values=[value['ns3']/value['ns5'] for key,value in groups.items() if key[0]==exponent and key[2]==epsilon and key[3]==favored]
                xs.append(1/epsilon);ys.append(np.exp(np.mean(np.log(values))))
            axes[1].plot(xs,ys,color=color,ls=ls,label=f'$H_{3 if favored=="ns3" else 5}$, span {exponent:g}')
    axes[1].axhline(1,color='#555555',lw=.7);axes[1].set(xscale='log',yscale='log',xlabel='$1/\\epsilon$',ylabel='$J_3/J_5$',title='(b) Same momentum, swapped curvature')
    axes[1].legend(fontsize=8,ncol=2)
    save(fig,output,'v2_geometry')

    frozen=read_csv(root/'synthetic_extended/frozen_matrix_controlled.csv');group=defaultdict(list)
    for row in frozen:group[tuple(row[k] for k in ('seed','exponent','condition','orientation','noise_ratio'))].append(row)
    summary=[]
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.7),layout='constrained',sharey=True)
    for ax,orientation in zip(axes,('aligned','reversed')):
        points=defaultdict(list)
        for key,items in group.items():
            if key[3]!=orientation or float(key[2])!=100. or float(key[4])!=0. or float(key[1])==0.:continue
            best=max(float(x['optimal_gain']) for x in items)
            for row in items:
                if row['variant'] in ('ns1','ns3','ns5'):points[row['variant']].append((float(row['polar_error']),float(row['optimal_gain'])/best))
        for method,values in points.items():
            data=np.asarray(values);ax.scatter(data[:,0],data[:,1],s=16,color=COLORS[method],label=LABELS[method],alpha=.65,edgecolors='none')
        ax.set(xlabel='RMS-normalized polar error',title=f'{orientation.capitalize()} curvature',ylim=(0,1.05))
    axes[0].set_ylabel('Gain / best candidate gain');axes[1].legend(fontsize=9)
    save(fig,output,'v2_error_gain')
    for exponent in (0.,.5,1.,2.):
        total=0;disagreements=0;ties=0
        for key,items in group.items():
            if float(key[1])!=exponent:continue
            items=[row for row in items if row['variant'] in ('ns1','ns3','ns5')]
            errors=np.asarray([float(row['polar_error']) for row in items]);gains=np.asarray([float(row['optimal_gain']) for row in items])
            if gains.max()-gains.min()<=1e-10:ties+=1;continue
            total+=1;disagreements+=gains[np.argmin(errors)]<gains.max()-1e-10
        summary.append(dict(exponent=exponent,states=total,tied_states=ties,error_winner_suboptimal=int(disagreements)))

    curves=read_csv(root/'synthetic/curves.csv');fig,axes=plt.subplots(2,2,figsize=(7.2,4.8),layout='constrained')
    for ax,(condition,noise) in zip(axes.flat,[(1.,0.),(100.,0.),(1.,.1),(100.,.1)]):
        rows=[r for r in curves if float(r['condition'])==condition and float(r['noise_ratio'])==noise and r['orientation']=='reversed']
        for method in ('ns0','ns1','ns3','ns5','polar'):
            by_step=defaultdict(list)
            for row in rows:
                if row['variant']==method:by_step[int(row['step'])].append(float(row['objective']))
            steps=sorted(by_step);values=np.asarray([by_step[s] for s in steps]);mean=values.mean(1);std=values.std(1,ddof=1)
            ax.plot(steps,mean,color=COLORS[method],label=LABELS[method]);ax.fill_between(steps,np.maximum(mean-std,1e-16),mean+std,color=COLORS[method],alpha=.09,lw=0)
        ax.set(yscale='log',xlabel='Updates',ylabel='Objective',title=f'$\\kappa={condition:g}$, noise={noise:g}')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=5,fontsize=9)
    save(fig,output,'v2_matrix_trajectories')
    (Path(root)/'synthetic_extended/statistics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='runs/research_v2');parser.add_argument('--output',default='paper/figures_v2');parser.add_argument('--synthetic',action='store_true')
    args=parser.parse_args()
    if args.synthetic:print(json.dumps(synthetic_figures(args.root,args.output),indent=2))
