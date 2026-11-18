"""CPU Hessian instruments for exact GPU-captured parameter-step candidates."""

import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .data import TokenDataset,sha256
from .geometry import match_rms
from .instruments import directional_geometry,evaluate_candidates
from .language import ResearchTransformer,token_loss,microbatch_token_loss
from .training import atomic_json

INSTRUMENT_HASHES = {
    name: sha256(Path(__file__).parent / name)
    for name in ("probes.py", "instruments.py", "geometry.py", "data.py", "language.py", "../model.py")
}


def batch_hash(batch):
    digest=hashlib.sha256()
    for tensor in batch:digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


def investigate_anchor(anchor_path,output,dataset,repeats=2,b_sequences=4,c_sequences=16,threads=2,expected=None):
    torch.set_num_threads(threads)
    anchor_path,output=Path(anchor_path),Path(output);output.mkdir(parents=True,exist_ok=True)
    identity=sha256(anchor_path);receipt=output/'result.json'
    instrument=dict(schema=3,repeats=repeats,b_sequences=b_sequences,c_sequences=c_sequences,threads=threads,
                    multipliers=[.5,1.,2.],source_hashes=INSTRUMENT_HASHES)
    if receipt.exists():
        previous=json.loads(receipt.read_text(encoding='utf-8'))
        if (previous['source_anchor_sha256']==identity and previous['dataset_fingerprint']==dataset.fingerprint
                and previous.get('instrument')==instrument and previous['status']=='COMPLETED'
                and (expected is None or previous.get('registered_anchor')==expected)):return previous
        raise ValueError('Frozen-probe cache identity changed')
    started=time.perf_counter();anchor=torch.load(anchor_path,map_location='cpu',weights_only=True)
    if anchor['dataset_fingerprint']!=dataset.fingerprint:raise ValueError('Frozen dataset identity changed')
    if expected is not None:
        for key in ('step','config_fingerprint'):
            if anchor[key]!=expected[key]:raise ValueError(f'Frozen anchor differs from registered {key}')
        if anchor['base_completed_updates']!=anchor['step']-1:
            raise ValueError('Frozen anchor is not a pre-update state')
    config=anchor['config'];model=ResearchTransformer(**config['model']);model.load_state_dict(anchor['model']);model.eval()
    candidates={key:{n:value.detach() for n,value in values.items()} for key,values in anchor['candidates'].items()}
    parameters=set(dict(model.named_parameters()))
    if set(candidates)!={'ns1','ns3','ns5'} or any(set(delta)!=parameters for delta in candidates.values()):
        raise ValueError('Frozen anchor must contain three full candidate steps')
    if not set(anchor['hidden_names']) or set(anchor['hidden_names'])!=set(anchor['momenta']):
        raise ValueError('Frozen anchor is missing hidden-matrix momenta')
    candidates['polar']=dict(candidates['ns1']);candidates['cubic5']=dict(candidates['ns1'])
    rate=config['lr']*anchor['lr_factor'];numerators={key:0. for key in candidates};denominator=0.;spectral=[];bank={}
    depth=config['model']['depth']
    selected_names={f'blocks.0.attention.qkv.weight',f'blocks.{depth//2}.mlp.0.weight',f'blocks.{depth-1}.attention.projection.weight',f'blocks.{depth-1}.mlp.2.weight'}
    with torch.no_grad():
        for name,momentum in anchor['momenta'].items():
            m=momentum.double();u,s,vh=torch.linalg.svd(m,full_matrices=False)
            active=s>torch.finfo(s.dtype).eps*max(m.shape)*s.max()
            polar=match_rms(u[:,active]@vh[active,:],.2)
            values=s/s.norm().clamp_min(torch.finfo(s.dtype).tiny)
            for _ in range(5):values=1.5*values-.5*values**3
            cubic=match_rms((u*values)@vh,.2)
            base=anchor['model'][name];decay=rate*config['weight_decay']*base
            candidates['polar'][name]=rate*polar.float()+decay
            candidates['cubic5'][name]=rate*cubic.float()+decay
            denominator+=polar.square().sum().item()
            for key,delta in candidates.items():
                direction=(delta[name].double()-decay.double())/rate
                numerators[key]+=(direction-polar).square().sum().item()
            if name in selected_names:
                probability=s.square()/s.square().sum().clamp_min(torch.finfo(s.dtype).tiny)
                positive=probability>0
                spectral.append(dict(layer=name,shape=list(m.shape),stable_rank=s.square().sum().item()/s[0].square().item() if s[0]>0 else 0.,
                    entropy_rank=float(torch.exp(-(probability[positive]*probability[positive].log()).sum())),
                    normalized_quantiles=torch.quantile(s/s.norm(),torch.tensor([.01,.1,.5,.9,.99],dtype=s.dtype)).tolist()))
                bank[name.replace('.','__')]=momentum.numpy()
    np.savez_compressed(output/'momenta.npz',**bank)
    errors={key:math.sqrt(value/denominator) for key,value in numerators.items()}
    rows=[];run_id=anchor_path.parent.parent.name
    for repeat in range(repeats):
        seed=int.from_bytes(hashlib.sha256(f'{run_id}|{anchor["step"]}|{repeat}'.encode()).digest()[:8],'little')%(2**63-1)
        b=dataset.batch('train',b_sequences,config['model']['context'],torch.Generator().manual_seed(seed),'cpu')
        c=dataset.batch('train',c_sequences,config['model']['context'],torch.Generator().manual_seed((seed+987654321)%(2**63-1)),'cpu')
        scores=directional_geometry(model,candidates,b,token_loss,layer_names=selected_names)
        actual_b=evaluate_candidates(model,candidates,b,microbatch_token_loss)
        for multiplier in (.5,1.,2.):
            scaled={};cache={}
            for key,values in candidates.items():
                scaled[key]={}
                for name,value in values.items():
                    if id(value) not in cache:cache[id(value)]=value*multiplier
                    scaled[key][name]=cache[id(value)]
            actual_c=evaluate_candidates(model,scaled,c,microbatch_token_loss)
            for key in candidates:
                score=scores[key]
                rows.append(dict(run_id=run_id,seed=config['seed'],cohort=config.get('cohort','pilot'),corpus=config['corpus'],scale=config['scale'],
                    anchor_step=anchor['step'],base_completed_updates=anchor['base_completed_updates'],repeat=repeat,variant=key,multiplier=multiplier,
                    polar_error=errors[key],linear_gain=multiplier*score['alignment'],curvature=multiplier**2*score['curvature'],
                    predicted_gain=multiplier*score['alignment']-.5*multiplier**2*score['curvature'],actual_gain=actual_c[key]['gain'],
                    actual_b_unit_gain=actual_b[key]['gain'],c_base_loss=actual_c[key]['base_loss'],
                    cross_curvature=score['cross_curvature'],layers=score.get('layers',{}),
                    b_batch_sha256=batch_hash(b),c_batch_sha256=batch_hash(c),source_anchor_sha256=identity))
    result=dict(status='COMPLETED',source_anchor_sha256=identity,dataset_fingerprint=dataset.fingerprint,run_id=run_id,
        anchor_step=anchor['step'],device='cpu',precision='float32',instrument=instrument,spectral=spectral,
        registered_anchor=expected,rows=rows,elapsed_seconds=time.perf_counter()-started)
    atomic_json(receipt,result);print(f'probed {run_id} step {anchor["step"]}: {result["elapsed_seconds"]:.1f}s',flush=True)
    return result


def run_bank(root,wait=False):
    root=Path(root);datasets={};receipts={}
    while True:
        manifest_path=root/'formal_manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
        for row in manifest:
            if row['status']!='COMPLETED':continue
            run=root/'language'/row['id']
            metadata=row['metadata']
            config=metadata['config']
            anchor_steps=config.get('anchor_steps',[]) if config.get('save_anchors') else []
            for step in anchor_steps:
                anchor=run/'anchors'/f'step-{step}.pt'
                identifier=row['id']+'_'+anchor.stem
                if identifier in receipts:continue
                try:
                    if row['corpus'] not in datasets:datasets[row['corpus']]=TokenDataset(Path('data/research_v2/tokens')/row['corpus'])
                    result=investigate_anchor(anchor,root/'probes'/identifier,datasets[row['corpus']],
                                              expected=dict(step=step,config_fingerprint=metadata['config_fingerprint']))
                    receipts[identifier]=dict(status=result['status'],path=str((root/'probes'/identifier/'result.json').resolve()),rows=len(result['rows']))
                except Exception as error:
                    receipts[identifier]=dict(status='FAILED',reason=f'{type(error).__name__}: {error}',rows=0)
                atomic_json(root/'probe_manifest.json',receipts);gc.collect()
                if receipts[identifier]['status']=='FAILED':
                    raise RuntimeError(f"Probe {identifier} failed: {receipts[identifier]['reason']}")
        complete=bool(manifest) and all(row['status']=='COMPLETED' for row in manifest)
        if complete or not wait:return receipts
        time.sleep(15)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='runs/research_v2');parser.add_argument('--execute',action='store_true');parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    if args.execute:run_bank(args.root,args.wait)
    else:print(json.dumps(dict(status='NOT_RUN',device='cpu',root=args.root)))
