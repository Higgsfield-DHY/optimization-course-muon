"""Independent readout of raw final evidence; CPU file checks, no inference."""
from pathlib import Path
from collections import defaultdict
import csv,json,math,sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from optcourse.evaluation import file_sha256
from optcourse.config import fingerprint

study=ROOT/'runs/course_v1'
state=json.loads((study/'study_state.json').read_text(encoding='utf-8'))
budget=json.loads((study/'gpu_budget.json').read_text(encoding='utf-8'))
summary=json.loads((study/'reports/comparison.json').read_text(encoding='utf-8'))
assert state['status']=='EXPERIMENTS_COMPLETE' and budget['active'] is None
assert budget['used_seconds']<=28800
assert math.isclose(budget['used_seconds'],sum(row['seconds'] for row in budget['attempts']),abs_tol=1e-6)
groups=defaultdict(list);initials=defaultdict(list);tests=[]
for job in state['jobs'].values():
    if job['phase']!='final':continue
    assert job['status']=='COMPLETED'
    run=Path(job['run_path'])
    meta=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    assert meta['status']=='COMPLETED'
    assert meta['config_fingerprint']==job['config_fingerprint']==fingerprint(meta['config'])
    with (run/'metrics.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    if job['task']=='matrix':
        assert int(rows[-1]['step'])==1000
        for row in rows:
            assert float(row['grad_norm_nuclear'])+1e-5>=float(row['grad_norm_F'])
        score=float(rows[-1]['objective'])
        initials['matrix',job['condition_number'],job['seed']].append(float(rows[0]['objective']))
    else:
        assert int(rows[-1]['step'])==state['profile']['steps']
        assert int(rows[-1]['tokens'])==4096*state['profile']['steps']
        val=[row for row in rows if row['validation_loss']]
        chosen=min(val,key=lambda row:float(row['validation_loss']))
        payload=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        assert payload['step']==int(chosen['step'])
        assert math.isclose(payload['validation_loss'],float(chosen['validation_loss']),abs_tol=1e-12)
        evaluation=json.loads((run/'evaluations/test_best.json').read_text(encoding='utf-8'))
        assert evaluation['checkpoint_sha256']==file_sha256(run/'best.pt')
        assert evaluation['predicted_bytes']==55769
        assert evaluation['config_fingerprint']==job['config_fingerprint']
        assert any(row.get('session_id')==evaluation['accounting']['session_id'] and row['outcome']=='CLOSED' for row in budget['attempts'])
        assert meta['test_set_evaluated'] is True
        initials['language',None,job['seed']].append(float(rows[0]['validation_loss']))
        tests.append(dict(variant=job['variant'],seed=job['seed'],best_step=payload['step'],test_loss=evaluation['loss']))
        score=float(rows[-1]['validation_loss'])
    groups[job['task'],job['condition_number'],job['variant']].append(score)
for values in initials.values():assert max(values)-min(values)<1e-7
for row in summary['groups']:
    values=np.array(groups[row['task'],row['condition_number'],row['variant']])
    assert len(values)==row['n']==3
    assert math.isclose(values.mean(),row['score_mean'],abs_tol=1e-12)
    assert math.isclose(values.std(ddof=1),row['score_std'],abs_tol=1e-12)
result=dict(final_runs=sum(map(len,groups.values())),test_runs=len(tests),
    gpu_seconds=budget['used_seconds'],all_checks_passed=True,per_seed_test=tests)
(ROOT/'work/final_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
