"""Manifest-bounded scientific summaries; never starts model evaluation."""

from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import statistics

from .config import fingerprint
from .evaluation import file_sha256
from .tracking import write_json


def compare_study(state_path, output, make_plots=True):
    state=json.loads(Path(state_path).read_text(encoding='utf-8'))
    expected=[row for row in state['jobs'].values() if row['phase']=='final']
    if not expected:
        raise ValueError('No expected final cohort in study manifest')
    groups=defaultdict(list)
    excluded=[]
    for job in expected:
        if not job.get('run_path'):
            excluded.append(dict(id=job['id'],status='NOT_RUN',reason='Not started'))
            continue
        run=Path(job['run_path'])
        meta=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
        config=meta['config']
        if (meta['config_fingerprint']!=job['config_fingerprint'] or fingerprint(config)!=job['config_fingerprint']
            or config['run']['seed']!=job['seed'] or config['run'].get('phase')!='final'):
            raise ValueError('Final manifest/config/phase identity mismatch')
        if meta['status']!='COMPLETED':
            excluded.append(dict(id=job['id'],status=meta['status'],reason=meta.get('failure_reason')))
            continue
        with (run/'metrics.csv').open(encoding='utf-8',newline='') as stream:
            metrics=list(csv.DictReader(stream))
        if not metrics:
            raise ValueError('Completed final run has no metrics')
        task=job['task']
        score_column='objective' if task=='matrix' else 'validation_loss'
        scores=[float(row[score_column]) for row in metrics if row.get(score_column)]
        if not scores:
            raise ValueError('Final run has no required terminal score')
        expected_steps=config[task]['steps' if task=='matrix' else 'max_steps']
        if int(metrics[-1]['step'])!=expected_steps:
            raise ValueError('Completed run did not finish its frozen horizon')
        test=None
        if task=='language':
            if meta.get('prepared_dataset_fingerprint')!=state.get('dataset_fingerprint'):
                raise ValueError('Language cohort dataset changed')
            evaluation=run/'evaluations/test_best.json'
            if evaluation.exists():
                test=json.loads(evaluation.read_text(encoding='utf-8'))
                if (test['config_fingerprint']!=job['config_fingerprint'] or
                    test['prepared_dataset_fingerprint']!=state['dataset_fingerprint'] or
                    test['checkpoint_sha256']!=file_sha256(run/'best.pt')):
                    raise ValueError('Held-out evaluation identity mismatch')
        key=(task,job['condition_number'],job['variant'])
        groups[key].append(dict(job=job,meta=meta,metrics=metrics,score=scores[-1],test=test))
    summary=[]
    for (task,condition,variant),items in groups.items():
        seeds=[item['job']['seed'] for item in items]
        if len(seeds)!=len(set(seeds)):
            raise ValueError('Duplicate independent final seed')
        scores=[item['score'] for item in items]
        test_scores=[item['test']['loss'] for item in items if item['test'] is not None]
        summary.append(dict(task=task,condition_number=condition,variant=variant,n=len(items),
            expected_n=sum(row['task']==task and row['condition_number']==condition and row['variant']==variant for row in expected),
            score_mean=statistics.mean(scores),score_std=statistics.stdev(scores) if len(scores)>1 else None,
            elapsed_mean=statistics.mean(item['job']['wall_seconds'] for item in items),
            trajectory_elapsed_mean=statistics.mean(float(item['metrics'][-1]['elapsed_seconds']) for item in items),
            peak_memory_MiB_mean=statistics.mean(float(item['metrics'][-1]['peak_memory_MiB'] or 0) for item in items),
            test_n=len(test_scores),test_loss_mean=statistics.mean(test_scores) if test_scores else None,
            test_loss_std=statistics.stdev(test_scores) if len(test_scores)>1 else None))
    result=dict(expected_final_runs=len(expected),completed_final_runs=sum(len(x) for x in groups.values()),
        groups=summary,excluded=excluded,phase_status_counts=dict(Counter(
            f"{row['task']}/{row['phase']}/{row['status']}" for row in state['jobs'].values())),
        note='Final manifest only; terminal full validation uses equal data budgets; test uses validation-selected best checkpoints')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'comparison.json',result)
    if summary:
        with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary[0]))
            writer.writeheader();writer.writerows(summary)
    if make_plots and groups:
        _plots(groups,output)
    return result


def _plots(groups,output):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'pdf.fonttype':42,'font.size':13})
    from matplotlib import pyplot as plt
    import numpy as np
    colors=dict(adamw='#333333',svd='#7570b3',ns1='#e6ab02',ns3='#1b9e77',ns5='#d95f02')
    labels=dict(adamw='AdamW',svd='Muon-SVD',ns1='Muon-NS1',ns3='Muon-NS3',ns5='Muon-NS5')
    def save(fig,name):
        fig.savefig(output/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(output/f'{name}.png',dpi=180,bbox_inches='tight')
        plt.close(fig)
    for metric,name,ylabel in (('objective','matrix_objective','Objective'),('grad_norm_F','matrix_gradient','True gradient norm')):
        matrix_keys=[key for key in groups if key[0]=='matrix']
        if not matrix_keys:
            continue
        conditions=sorted(set(key[1] for key in matrix_keys))
        fig,axes=plt.subplots(1,len(conditions),figsize=(10,2.8),squeeze=False,layout='constrained')
        for index,condition in enumerate(conditions):
            axis=axes[0,index]
            for key in matrix_keys:
                if key[1]!=condition:
                    continue
                variant=key[2]
                rows=groups[key]
                for item in rows:
                    useful=[row for row in item['metrics'] if row.get(metric)]
                    axis.plot([int(row['step']) for row in useful],[float(row[metric]) for row in useful],
                        color=colors[variant],alpha=.65,lw=.9,label=labels[variant] if item is rows[0] else None)
            axis.set(xlabel='Updates',ylabel=ylabel,title=f'Condition number {condition}',yscale='log')
            axis.grid(alpha=.2)
        axes[0,0].legend(fontsize=10)
        save(fig,name)
    language_keys=[key for key in groups if key[0]=='language']
    if language_keys:
        for x,name,xlabel in (('tokens','language_bytes','Training bytes'),('elapsed_seconds','language_time','Optimization-trajectory seconds')):
            fig,axis=plt.subplots(figsize=(5.6,3.2),layout='constrained')
            for key in language_keys:
                variant=key[2]
                items=groups[key]
                if x=='tokens':
                    maps=[{int(row[x]):float(row['validation_loss']) for row in item['metrics'] if row.get('validation_loss')} for item in items]
                    positions=sorted(set.intersection(*(set(mapping) for mapping in maps)))
                    values=np.array([[mapping[position] for position in positions] for mapping in maps])
                    mean=values.mean(0)
                    axis.plot(positions,mean,color=colors[variant],label=labels[variant])
                    if len(items)>1:
                        std=values.std(0,ddof=1)
                        axis.fill_between(positions,mean-std,mean+std,color=colors[variant],alpha=.15)
                else:
                    for index,item in enumerate(items):
                        useful=[row for row in item['metrics'] if row.get('validation_loss')]
                        axis.plot([float(row[x]) for row in useful],[float(row['validation_loss']) for row in useful],
                            color=colors[variant],alpha=.7,label=labels[variant] if index==0 else None)
            axis.set(xlabel=xlabel,ylabel='Full validation loss (nats/byte)')
            axis.grid(alpha=.2);axis.legend()
            save(fig,name)
