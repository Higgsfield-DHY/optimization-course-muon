"""Research-v2 ledger and explicit stage entry points."""

import gc
from itertools import permutations
import json
from pathlib import Path

import numpy as np

from ..budget import BudgetLedger
from .training import atomic_json

PRIOR_GPU_SECONDS=3507.4571810862326
SCALES=dict(small=dict(depth=4,width=384,heads=6,context=256,vocabulary=50304),
            medium=dict(depth=8,width=512,heads=8,context=256,vocabulary=50304))


def preview(root):
    return dict(status='NOT_RUN',root=str(root),aggregate_gpu_limit_seconds=28800,prior_gpu_seconds=PRIOR_GPU_SECONDS,
        scales=SCALES,corpora=['wikitext103','tinystories'],principal_methods=['adamw','momentum','ns1','ns3','ns5','adaptive','replay'])


def ledger(root):
    path=Path(root)/'gpu_budget.json'
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(path,dict(limit_seconds=28800.,used_seconds=PRIOR_GPU_SECONDS,
            attempts=[dict(label='carried_course_v1',seconds=PRIOR_GPU_SECONDS,outcome='CLOSED',reason='Previously consumed aggregate budget',session_id='course_v1_carried')],active=None))
    return BudgetLedger(path)


def base_config(method='ns5',scale='small',seed=0,steps=100,phase='pilot',corpus='wikitext103',lr=.001):
    return dict(seed=seed,method=method,model=SCALES[scale],scale=scale,corpus=corpus,steps=steps,batch_size=4,accumulation=8,
        lr=lr,aux_lr=.001,weight_decay=.1,device='cuda',precision='bfloat16',ns_dtype='bfloat16',validation_every=250,
        validation_targets=4096 if phase=='pilot' else 262144,log_every=50,warmup_steps=max(1,steps//10),
        probe_interval=100,probe_batch_size=2,phase=phase)


def run_one(root,identifier,config,dataset):
    import torch
    from .training import run_training
    budget=ledger(root)
    with budget.session(identifier):
        gc.collect();torch.cuda.empty_cache()
        return run_training(config,dataset,Path(root)/'language'/identifier,budget)


def pilot(root):
    from .data import TokenDataset
    dataset=TokenDataset('data/research_v2/tokens/wikitext103')
    rows=[]
    for scale in ('small','medium'):
        for method in (('adamw','ns3','ns5','adaptive','ns1','momentum','adam_rms','canonical5') if scale=='small' else ('adamw','ns3','ns5','adaptive')):
            config=base_config(method,scale,99,100)
            identifier=f'pilot_{scale}_{method}'
            row=run_one(root,identifier,config,dataset);rows.append(row)
            if row['status']!='COMPLETED':raise RuntimeError(f'Pilot did not complete: {identifier}')
    atomic_json(Path(root)/'pilot.json',rows)
    return rows


def freeze_protocol(root):
    root=Path(root);path=root/'frozen_protocol.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    pilots=json.loads((root/'pilot.json').read_text(encoding='utf-8'))
    latency={}
    for row in pilots:
        identifier=f"pilot_{row['config']['scale']}_{row['method']}"
        metrics=json.loads((root/'language'/identifier/'metrics.json').read_text(encoding='utf-8'))
        a=next(x for x in metrics if x['step']==50);b=next(x for x in metrics if x['step']==100)
        latency[(row['config']['scale'],row['method'])]=(b['elapsed_seconds']-a['elapsed_seconds'])/50
    small=max(latency[('small','adaptive')],latency[('small','ns5')])
    medium=max(latency[('medium','adaptive')],latency[('medium','ns5')])
    main_methods=['adamw','momentum','ns1','ns3','ns5','adaptive','replay']
    def estimate(long_steps):
        cost=lambda scale,method:latency[(scale,'adaptive' if method=='replay' else method)]
        training_seconds=3*1000*sum(cost('small',m) for m in main_methods)
        training_seconds+=3*1000*sum(cost('small',m) for m in ('adam_rms','canonical5'))
        training_seconds+=3*1000*sum(cost('medium',m) for m in ('adamw','ns5','adaptive'))
        training_seconds+=3*1000*sum(cost('small',m) for m in ('adamw','ns5','adaptive'))
        training_seconds+=3*long_steps*sum(cost('small',m) for m in ('ns3','ns5'))
        training_seconds+=3*1000*cost('small','ns5')
        return 1.15*training_seconds+4800+1200+150
    budget=ledger(root)
    long_steps=2500
    projected=estimate(long_steps)
    if projected>budget.remaining_seconds-300:
        long_steps=2000;projected=estimate(long_steps)
    if projected>budget.remaining_seconds-300:raise ValueError('Registered study exceeds the remaining aggregate budget')
    result=dict(core_steps=1000,ablation_steps=1000,long_steps=long_steps,token_batch=8192,
        main_methods=['adamw','momentum','ns1','ns3','ns5','adaptive','replay'],ablation_methods=['adam_rms','canonical5'],
        confirmation_methods=['adamw','ns5','adaptive'],seeds=[0,1,2],tuning_steps=400,tuning_seed=101,confirmation_seed=102,
        tuning_grid=[.0003,.001,.003],core_formal_runs=54,projection_seconds=projected,
        online_probe_sequences=16,frozen_probe_B_sequences=4,frozen_probe_C_sequences=16,
        projected_update_latency=dict(small=small,medium=medium),prior_used_seconds=budget.used_seconds,
        expected_jobs=[dict(cohort=cohort,corpus=corpus,scale=scale,method=method,seed=seed,steps=steps,status='NOT_RUN')
            for cohort,corpus,scale,methods,steps in [
                ('main','wikitext103','small',['adamw','momentum','ns1','ns3','ns5','adaptive','replay'],1000),
                ('criterion','wikitext103','small',['spectral'],1000),
                ('ablation','wikitext103','small',['adam_rms','canonical5'],1000),
                ('scale','wikitext103','medium',['adamw','ns5','adaptive'],1000),
                ('transfer','tinystories','small',['adamw','ns5','adaptive'],1000),
                ('long','wikitext103','small',['ns3','ns5'],long_steps)]
            for seed in (0,1,2) for method in methods])
    atomic_json(path,result);return result


def tune(root):
    from .data import TokenDataset
    root=Path(root);protocol=freeze_protocol(root);records=[];selections={}
    for corpus,scale,methods in [('wikitext103','small',('adamw','momentum','ns3','ns5')),
                                 ('wikitext103','medium',('adamw','ns5')),
                                 ('tinystories','small',('adamw','ns5'))]:
        dataset=TokenDataset(Path('data/research_v2/tokens')/corpus)
        key=f'{corpus}_{scale}';selections[key]={}
        for method in methods:
            candidates={}
            def trial(rate,seed):
                identifier=f'tune_{key}_{method}_lr{rate:g}_s{seed}'.replace('.','p')
                config=base_config(method,scale,seed,protocol['tuning_steps'],'tuning',corpus,rate)
                config['validation_targets']=65536
                try:row=run_one(root,identifier,config,dataset)
                except RuntimeError as error:
                    if 'nonfinite' not in str(error).lower() and 'non-finite' not in str(error).lower():raise
                    row=json.loads((root/'language'/identifier/'metadata.json').read_text(encoding='utf-8'))
                score=row['best_validation'] if row['status']=='COMPLETED' else None
                record=dict(id=identifier,corpus=corpus,scale=scale,method=method,lr=rate,seed=seed,status=row['status'],score=score)
                records.append(record);atomic_json(root/'tuning_records.json',records)
                return score if score is not None else float('inf')
            for rate in protocol['tuning_grid']:candidates[rate]=trial(rate,101)
            for _ in range(3):
                best=min(candidates,key=lambda x:(candidates[x],x));ordered=sorted(candidates)
                if best==ordered[0]:rate=best/3
                elif best==ordered[-1]:rate=best*3
                else:break
                candidates[rate]=trial(rate,101)
            top=sorted(candidates,key=lambda x:(candidates[x],x))[:2]
            means={rate:.5*(candidates[rate]+trial(rate,102)) for rate in top}
            rate=min(means,key=lambda x:(means[x],x))
            if not np.isfinite(means[rate]):raise RuntimeError(f'No finite selected rate for {key}/{method}')
            selections[key][method]=dict(lr=rate,selection_mean=means[rate],seed101_scores={str(k):(v if np.isfinite(v) else None) for k,v in candidates.items()},seed102_candidates=top)
            atomic_json(root/'selections.json',selections)
            print(f'selected {key} {method}: {rate}',flush=True)
    return selections


def stage_matched_replay(selections,seed):
    sequence=[int(row['choice'][2:]) for row in selections]
    rng=np.random.default_rng(12345+seed)
    result=[]
    for block in np.array_split(np.arange(len(sequence)),4):
        # The final selection has a shorter residence time. Keep it at its
        # original step so the control matches realized update counts too.
        indices=[int(i) for i in block if i<len(sequence)-1]
        original=tuple(sequence[i] for i in indices)
        alternatives=sorted(set(permutations(original))-{original})
        result.extend(alternatives[rng.integers(len(alternatives))] if alternatives else original)
        if len(block) and block[-1]==len(sequence)-1:
            result.append(sequence[-1])
    return result


def formal(root):
    from .data import TokenDataset
    root=Path(root);protocol=freeze_protocol(root)
    selected=json.loads((root/'selections.json').read_text(encoding='utf-8'))
    records=protocol['expected_jobs']
    datasets={}
    manifest=root/'formal_manifest.json'
    if manifest.exists():
        key=lambda row:(row['cohort'],row['corpus'],row['scale'],row['method'],row['seed'])
        previous={key(row):row for row in json.loads(manifest.read_text(encoding='utf-8'))}
        if set(previous)!=set(map(key,records)):
            raise ValueError('Existing formal manifest differs from registered jobs')
        records=[previous[key(row)] for row in records]
    atomic_json(manifest,records)
    for cohort in ('main','criterion','ablation','scale','transfer','long'):
        for seed in protocol['seeds']:
            subset=[row for row in records if row['cohort']==cohort and row['seed']==seed]
            if cohort=='main':subset=subset[seed*2:]+subset[:seed*2]
            for row in subset:
                corpus,scale,method=row['corpus'],row['scale'],row['method'];key=f'{corpus}_{scale}'
                rate_method='adamw' if method=='adam_rms' else ('momentum' if method=='momentum' else ('ns3' if cohort=='long' and method=='ns3' else ('adamw' if method=='adamw' else 'ns5')))
                rate=selected[key][rate_method]['lr']
                config=base_config(method,scale,seed,row['steps'],'final',corpus,rate);config['cohort']=cohort
                config['probe_batch_size']=16
                if method=='replay':
                    source=root/'language'/f'main_{key}_adaptive_s{seed}'/'selections.json'
                    config['depth_schedule']=stage_matched_replay(json.loads(source.read_text(encoding='utf-8')),seed)
                if (method=='ns5' and cohort in {'main','scale','transfer'}) or (method=='adaptive' and cohort=='main'):
                    config['save_anchors']=True;config['anchor_steps']=[250,500,row['steps']]
                if corpus not in datasets:datasets[corpus]=TokenDataset(Path('data/research_v2/tokens')/corpus)
                identifier=f'{cohort}_{key}_{method}_s{seed}'
                try:
                    metadata=run_one(root,identifier,config,datasets[corpus])
                except BaseException as error:
                    metadata_file=root/'language'/identifier/'metadata.json'
                    metadata=json.loads(metadata_file.read_text(encoding='utf-8')) if metadata_file.is_file() else {}
                    row.update(status='FAILED',id=identifier,lr=rate,metadata=metadata,
                               failure_reason=f'{type(error).__name__}: {error}')
                    atomic_json(manifest,records)
                    raise
                row.update(status=metadata['status'],id=identifier,lr=rate,metadata=metadata)
                row.pop('failure_reason',None)
                atomic_json(manifest,records)
                if row['status']!='COMPLETED':raise RuntimeError(f'Formal task did not complete: {identifier}')
    return records


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='runs/research_v2');parser.add_argument('--stage',default='pilot');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:print(json.dumps(preview(args.root),indent=2))
    elif args.stage=='pilot':pilot(args.root)
    elif args.stage=='freeze':print(json.dumps(freeze_protocol(args.root),indent=2))
    elif args.stage=='tuning':tune(args.root)
    elif args.stage=='formal':formal(args.root)
    elif args.stage=='test':
        from .assessment import evaluate_formal
        evaluate_formal(args.root)
    elif args.stage=='profile':
        from .performance import profile_formal
        profile_formal(args.root)
    elif args.stage=='schedule-control':
        from .schedule_control import schedule_control
        print(json.dumps(schedule_control(args.root),indent=2))
    else:raise ValueError('Unknown research stage')
