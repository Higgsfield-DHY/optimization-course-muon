"""Approved study recipes. Planning is pure; execution is always explicit."""

from contextlib import nullcontext
import csv
import json
import math
from pathlib import Path
import time

from .config import RUN_DEFAULTS, OPTIM_DEFAULTS, MATRIX_DEFAULTS, LANGUAGE_DEFAULTS


def study_plan():
    return dict(status='NOT_RUN', gpu_budget_seconds=28800,
        counts=dict(matrix_tuning=18, matrix_final=45, language_pilot=3, language_tuning=6, language_final=9),
        final_seeds=[0,1,2], tuning_seed=101, pilot_seed=99, horizon_candidates=[2000,1000,500])


def recipes(task, phase, steps=None, selected=None, micro_batch=8):
    if task not in {'matrix','language'} or phase not in {'pilot','tuning','final'}:
        raise ValueError('Unknown study task/phase')
    if micro_batch not in {8,4,2}:
        raise ValueError('Unsupported microbatch')
    if phase == 'final' and selected is None:
        raise ValueError('Final recipes require frozen tuning selections')
    rows = []
    conditions = (1,10,100) if task == 'matrix' else (None,)
    variants = ('adamw','svd','ns1','ns3','ns5') if task == 'matrix' and phase=='final' else (
        ('adamw','ns5','ns3') if phase in {'pilot','final'} else ('adamw','ns5'))
    seeds = (0,1,2) if phase=='final' else ((101,) if phase=='tuning' else (99,))
    for condition in conditions:
        for variant in variants:
            name = 'adamw' if variant == 'adamw' else 'muon'
            if phase == 'tuning':
                rates = (.003,.01,.03) if task=='matrix' and name=='adamw' else (
                    (.0001,.0003,.001) if name=='adamw' else (.005,.02,.05))
            elif phase == 'final':
                rates = ((selected[str(condition)][name] if task=='matrix' else selected[name]),)
            else:
                rates = (.0003 if name=='adamw' else .02,)
            for rate in rates:
                for seed in seeds:
                    count = steps or (300 if phase=='tuning' else (100 if phase=='pilot' else 1000))
                    label = f'{task}_{phase}_{variant}' + (f'_k{condition}' if condition else '')
                    if task=='language':
                        label += f'_b{micro_batch}'
                    config = dict(run=RUN_DEFAULTS | dict(name=label,task=task,seed=seed,
                        device='cpu' if task=='matrix' else 'cuda',phase=phase,cohort='course_v1'),
                        optimizer=OPTIM_DEFAULTS | dict(name=name,lr=rate,normalization='max',
                            ns_steps=int(variant[-1]) if variant.startswith('ns') else 5,
                            method='svd' if variant=='svd' else 'ns',
                            weight_decay=0. if task=='matrix' else .01,aspect_scale=task=='language'))
                    if task=='matrix':
                        config['matrix'] = MATRIX_DEFAULTS | dict(condition_number=float(condition),steps=count)
                    else:
                        config['language'] = LANGUAGE_DEFAULTS | dict(max_steps=count,warmup_steps=count//10,
                            batch_size=micro_batch,grad_accum=32//micro_batch,eval_mode='full')
                    job_id = f'{label}_lr{rate:g}_s{seed}'.replace('.','p')
                    rows.append(dict(id=job_id,variant=variant,condition_number=condition,config=config))
    if task=='language' and phase=='final':
        order=('adamw','ns5','ns3')
        rows.sort(key=lambda row:(row['config']['run']['seed'],
            (order.index(row['variant'])-row['config']['run']['seed'])%3))
    return rows


def choose_trial(trials):
    valid = [row for row in trials if row['status']=='COMPLETED' and
             row.get('score') is not None and math.isfinite(row['score'])]
    if not valid:
        raise ValueError('No completed finite tuning candidate')
    return min(valid, key=lambda row:(row['score'],row['lr']))


def select_horizon(seconds_per_step):
    costs = [seconds_per_step[key] for key in ('adamw','ns5','ns3')]
    if any(not math.isfinite(value) or value<=0 for value in costs):
        raise ValueError('Invalid pilot timing')
    tuning = 3*300*(costs[0]+costs[1])
    for steps in (2000,1000,500):
        estimate = 1.3*(tuning+3*steps*sum(costs))
        if estimate <= 21600:
            return dict(steps=steps,projected_tuning_final_seconds=estimate,safety_factor=1.3,
                        measured_seconds_per_step=seconds_per_step)
    raise ValueError('Minimum horizon exceeds the approved GPU budget')


def write_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for section, values in config.items():
        sections.append(f'[{section}]')
        for key, value in values.items():
            sections.append(f'{key} = {json.dumps(value,ensure_ascii=False,allow_nan=False)}')
        sections.append('')
    text = '\n'.join(sections)
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError(f'Refusing to alter a frozen configuration: {path}')
    path.write_text(text,encoding='utf-8')
    return path


class Study:
    def __init__(self, root):
        from .budget import BudgetLedger
        from .tracking import write_json
        import torch
        torch.set_num_threads(1)
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.path = self.root/'study_state.json'
        self.state = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else dict(
            protocol_version=1,jobs={},created_from_plan='2026-11-04-course-study',status='IN_PROGRESS')
        if self.state['protocol_version'] != 1:
            raise ValueError('Unknown study protocol version')
        self.write_json = write_json
        self.budget = BudgetLedger(self.root/'gpu_budget.json',28800)

    def save(self):
        self.write_json(self.path,self.state)

    def run_job(self, row):
        from .config import load_config, fingerprint
        from .experiments import run_experiment
        config = row['config']
        config['run']['output_root'] = str(self.root/config['run']['task'])
        path = write_config(config,self.root/'configs'/f"{row['id']}.toml")
        effective = load_config(path)
        identity = fingerprint(effective)
        entry = self.state['jobs'].setdefault(row['id'],dict(id=row['id'],variant=row['variant'],
            condition_number=row['condition_number'],phase=config['run']['phase'],task=config['run']['task'],
            seed=config['run']['seed'],config_fingerprint=identity,lr=config['optimizer']['lr'],
            status='NOT_RUN',run_path=None,wall_seconds=0.,attempts=[]))
        if entry['config_fingerprint'] != identity:
            raise ValueError('Study job configuration changed')
        if entry['status']=='COMPLETED':
            meta=json.loads((Path(entry['run_path'])/'metadata.json').read_text(encoding='utf-8'))
            if meta['config_fingerprint'] != identity or meta['status']!='COMPLETED':
                raise ValueError('Completed study record does not match run metadata')
            return entry
        resume = None
        if entry['run_path'] and (Path(entry['run_path'])/'latest.pt').exists():
            resume=Path(entry['run_path'])
        gpu=config['run']['device'].startswith('cuda')
        entry['score']=None
        snapshot=None
        if gpu:
            import subprocess
            probe=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.free',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=False)
            snapshot=probe.stdout.strip() if probe.returncode==0 else 'unavailable'
        def register(store):
            if snapshot is not None:
                store.update(gpu_baseline_utilization_percent_and_free_MiB=snapshot)
                import torch
                store.update(gpu_process_start_allocated_MiB=torch.cuda.memory_allocated()/2**20)
            entry['run_path']=str(store.path)
            entry['status']='RUNNING'
            if str(store.path) not in entry['attempts']:
                entry['attempts'].append(str(store.path))
            self.save()
        started=time.perf_counter()
        error=None
        try:
            with self.budget.session(row['id']) if gpu else nullcontext():
                if gpu:
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                run_experiment(effective,resume=resume,budget=self.budget if gpu else None,on_store=register)
        except Exception as problem:
            error=problem
        finally:
            entry['wall_seconds']+=time.perf_counter()-started
            if entry['run_path']:
                meta=json.loads((Path(entry['run_path'])/'metadata.json').read_text(encoding='utf-8'))
                entry['status']=meta['status']
                entry['failure_reason']=meta.get('failure_reason')
                if entry['status']=='COMPLETED':
                    if config['run']['task']=='language':
                        entry['score']=meta['best_validation']
                    else:
                        with (Path(entry['run_path'])/'metrics.csv').open(encoding='utf-8',newline='') as stream:
                            entry['score']=float(list(csv.DictReader(stream))[-1]['objective'])
            self.save()
        if error is not None and entry['status'] not in {'FAILED'}:
            raise error
        if entry['status']=='INTERRUPTED':
            raise RuntimeError(f"Study interrupted: {entry.get('failure_reason')}")
        return entry

    def run_rows(self, rows, require_complete=True):
        result=[]
        # Materialize the entire expected cohort before running its first job.
        for row in rows:
            config=row['config']
            config['run']['output_root']=str(self.root/config['run']['task'])
            from .config import load_config,fingerprint
            path=write_config(config,self.root/'configs'/f"{row['id']}.toml")
            self.state['jobs'].setdefault(row['id'],dict(id=row['id'],variant=row['variant'],
                condition_number=row['condition_number'],phase=config['run']['phase'],task=config['run']['task'],
                seed=config['run']['seed'],config_fingerprint=fingerprint(load_config(path)),lr=config['optimizer']['lr'],
                status='NOT_RUN',run_path=None,wall_seconds=0.,attempts=[]))
        self.save()
        for row in rows:
            entry=self.run_job(row)
            result.append(entry)
            if require_complete and entry['status']!='COMPLETED':
                raise RuntimeError(f"Study job failed: {entry['id']}: {entry.get('failure_reason')}")
        return result

    def matrix(self):
        import torch
        torch.set_num_threads(1)
        tuning=self.run_rows(recipes('matrix','tuning'),require_complete=False)
        selected={}
        for condition in (1,10,100):
            selected[str(condition)]={name:choose_trial([row for row in tuning if row['condition_number']==condition
                and row['variant']==variant])['lr'] for name,variant in (('adamw','adamw'),('muon','ns5'))}
        if 'matrix_selection' in self.state and self.state['matrix_selection'] != selected:
            raise ValueError('Frozen matrix selection changed')
        self.state['matrix_selection']=selected
        self.save()
        self.run_rows(recipes('matrix','final',selected=selected))
        self.matrix_diagnostics()

    def matrix_diagnostics(self):
        import torch
        from .experiments import make_matrix_problem
        from .optim import orthogonalize
        rows=[]
        for condition in (1,10,100):
            for seed in (0,1,2):
                a,b,_,initial=make_matrix_problem(32,16,condition,seed)
                gradient=a.T@(a@initial-b)
                reference=orthogonalize(gradient,method='svd')
                for variant,steps in (('ns1',1),('ns3',3),('ns5',5)):
                    update=orthogonalize(gradient,steps=steps,normalization='max')
                    singular=torch.linalg.svdvals(update)
                    rows.append(dict(condition_number=condition,seed=seed,variant=variant,
                        relative_error_F=((update-reference).norm()/reference.norm()).item(),
                        spectral_error=torch.linalg.matrix_norm(update-reference,ord=2).item(),
                        min_singular=singular.min().item(),max_singular=singular.max().item()))
        self.write_json(self.root/'matrix_diagnostics.json',dict(input='same initial true gradient for all NS variants',rows=rows))

    def pilot(self):
        from .data import ByteDataset
        from .config import LANGUAGE_DEFAULTS
        identity=ByteDataset(Path(__file__).resolve().parents[2]/LANGUAGE_DEFAULTS['data_dir']).fingerprint
        if self.state.get('dataset_fingerprint',identity)!=identity:
            raise ValueError('Study dataset identity changed')
        self.state['dataset_fingerprint']=identity
        if 'profile' in self.state:
            return
        for micro in (8,4,2):
            rows=self.run_rows(recipes('language','pilot',micro_batch=micro),require_complete=False)
            if all(row['status']=='COMPLETED' for row in rows):
                profile=select_horizon({row['variant']:row['wall_seconds']/100 for row in rows})
                profile['micro_batch']=micro
                self.state['profile']=profile
                self.save()
                return
            failures=[row for row in rows if row['status']!='COMPLETED']
            if not all('out of memory' in row.get('failure_reason','').lower() for row in failures):
                raise RuntimeError(f'Pilot failed: {failures}')
            import torch
            torch.cuda.empty_cache()
        raise RuntimeError('All approved microbatch sizes failed')

    def tuning(self):
        self.check_dataset()
        profile=self.state['profile']
        rows=self.run_rows(recipes('language','tuning',micro_batch=profile['micro_batch']),require_complete=False)
        selected={name:choose_trial([row for row in rows if row['variant']==variant])['lr']
                  for name,variant in (('adamw','adamw'),('muon','ns5'))}
        if self.state.get('language_selection',selected)!=selected:
            raise ValueError('Frozen language selection changed')
        self.state['language_selection']=selected
        self.save()

    def final(self):
        self.check_dataset()
        profile=self.state['profile']
        self.run_rows(recipes('language','final',steps=profile['steps'],
            selected=self.state['language_selection'],micro_batch=profile['micro_batch']))

    def evaluate(self):
        self.check_dataset()
        from .evaluation import evaluate_run
        expected=[row for row in self.state['jobs'].values() if row['task']=='language' and row['phase']=='final']
        if len(expected)!=9 or any(row['status']!='COMPLETED' for row in expected):
            raise ValueError('All nine final language runs must complete before test evaluation')
        for row in expected:
            with self.budget.session('test_'+row['id']):
                row['test_evaluation']=evaluate_run(row['run_path'],'test',budget=self.budget)
            self.save()
        self.state['status']='EXPERIMENTS_COMPLETE'
        self.save()

    def check_dataset(self):
        from .data import ByteDataset
        identity=ByteDataset(Path(__file__).resolve().parents[2]/LANGUAGE_DEFAULTS['data_dir']).fingerprint
        if identity!=self.state.get('dataset_fingerprint'):
            raise ValueError('Frozen study dataset identity mismatch')


def execute_study(stage,root):
    study=Study(root)
    stages=('matrix','pilot','tuning','final','evaluate','report') if stage=='all' else (stage,)
    for name in stages:
        if name=='report':
            from .comparison import compare_study
            compare_study(study.path,study.root/'reports')
        else:
            getattr(study,name)()
    return dict(status=study.state['status'],root=str(study.root),gpu_used_seconds=study.budget.used_seconds,
                jobs=len(study.state['jobs']))
