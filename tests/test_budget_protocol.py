import importlib
import json

import pytest


class FakeClock:
    def __init__(self):
        self.now = 0.
    def __call__(self):
        return self.now


def test_shared_budget_charges_failures_and_stops_at_limit(tmp_path):
    module = importlib.import_module('optcourse.budget')
    clock = FakeClock()
    ledger = module.BudgetLedger(tmp_path/'budget.json', 10, clock=clock)
    with pytest.raises(RuntimeError, match='fixture'):
        with ledger.session('failed job'):
            clock.now = 3.
            raise RuntimeError('fixture')
    assert ledger.used_seconds == 3.
    restored = module.BudgetLedger(tmp_path/'budget.json', 10, clock=clock)
    with restored.session('next job'):
        clock.now = 11.
        with pytest.raises(module.BudgetExhausted):
            restored.check()
    assert restored.used_seconds == 11.
    assert len(json.loads((tmp_path/'budget.json').read_text())['attempts']) == 2
    with pytest.raises(module.BudgetExhausted):
        with restored.session('must not run'):
            pytest.fail('exhausted budget executed')


def test_protocol_counts_seed_isolation_and_full_evaluation():
    module = importlib.import_module('optcourse.study')
    assert module.study_plan()['counts'] == dict(matrix_tuning=18, matrix_final=45,
        language_pilot=3, language_tuning=6, language_final=9)
    tuning = module.recipes('language', 'tuning')
    assert len(tuning) == 6
    assert {row['config']['run']['seed'] for row in tuning} == {101}
    assert all(row['config']['language']['eval_mode'] == 'full' for row in tuning)
    chosen = {'adamw':.0003, 'muon':.02}
    final = module.recipes('language','final',steps=500,selected=chosen)
    assert len(final) == 9
    assert {row['config']['run']['seed'] for row in final} == {0,1,2}
    assert all(row['config']['language']['warmup_steps']==50 for row in final)
    assert all(row['config']['optimizer']['normalization']=='max' for row in final)
    assert [row['variant'] for row in final[:3]]==['adamw','ns5','ns3']
    assert [row['config']['run']['seed'] for row in final[:3]]==[0,0,0]


def test_horizon_uses_measured_conservative_cost_and_selection_ignores_failures():
    module = importlib.import_module('optcourse.study')
    assert module.select_horizon({'adamw':1.,'ns5':1.,'ns3':1.})['steps'] == 1000
    assert module.select_horizon({'adamw':2.,'ns5':2.,'ns3':2.})['steps'] == 500
    with pytest.raises(ValueError, match='budget'):
        module.select_horizon({'adamw':10.,'ns5':10.,'ns3':10.})
    trials = [dict(status='FAILED',lr=.01,score=0.),dict(status='COMPLETED',lr=.03,score=2.),
              dict(status='COMPLETED',lr=.003,score=2.)]
    assert module.choose_trial(trials)['lr'] == .003
    with pytest.raises(ValueError):
        module.choose_trial(trials[:1])


@pytest.mark.parametrize('total_steps',[2,4])
def test_budget_stop_saves_last_complete_update_in_temporary_fixture(tmp_path, monkeypatch,total_steps):
    import torch
    from optcourse.config import load_config
    from optcourse.optim import Muon
    from optcourse.experiments import run_experiment
    budget_module = importlib.import_module('optcourse.budget')
    clock = FakeClock()
    budget = budget_module.BudgetLedger(tmp_path/'budget.json',1.5,clock=clock)
    path=tmp_path/'config.toml'
    path.write_text(f'[run]\nname="fixture"\nphase="verification"\noutput_root="fixtures"\n[matrix]\nrows=2\ncols=2\nsteps={total_steps}\ncheckpoint_every=4\n',encoding='utf-8')
    config=load_config(path)
    original=Muon.step
    def counted_step(self, *args, **kwargs):
        result=original(self,*args,**kwargs)
        clock.now+=1
        return result
    monkeypatch.setattr(Muon,'step',counted_step)
    with budget.session('manual-gradient fixture'):
        run=run_experiment(config,budget=budget)
    meta=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    checkpoint=torch.load(run/'latest.pt',weights_only=False)
    assert meta['status']=='INTERRUPTED'
    assert meta['failure_reason']=='GPU_BUDGET_EXHAUSTED'
    assert meta['completed_steps']==checkpoint['step']==1
    assert budget.used_seconds<=1.5


def test_study_cli_preview_creates_no_outputs(tmp_path, capsys):
    from optcourse.cli import main
    target=tmp_path/'no-side-effects'
    assert main(['study','--root',str(target)]) == 0
    assert json.loads(capsys.readouterr().out)['status']=='NOT_RUN'
    assert not target.exists()


def test_full_training_validation_obeys_budget_and_keeps_pending_checkpoint(tmp_path,monkeypatch):
    import torch
    from optcourse.config import load_config
    from optcourse.data import prepare_data
    from optcourse.experiments import run_experiment
    from optcourse.model import TinyTransformer
    from optcourse.budget import BudgetLedger
    source=tmp_path/'text.txt';source.write_bytes(b'abcde'*100)
    prepare_data(source,tmp_path/'data')
    path=tmp_path/'config.toml'
    path.write_text('[run]\ntask="language"\ndevice="cpu"\nphase="verification"\noutput_root="fixtures"\n[language]\ndata_dir="data"\nn_layer=1\nn_embd=8\nn_head=2\nblock_size=4\nbatch_size=1\nmax_steps=2\nwarmup_steps=0\neval_mode="full"\n',encoding='utf-8')
    clock=FakeClock();budget=BudgetLedger(tmp_path/'budget.json',1.5,clock=clock)
    original=TinyTransformer.forward
    calls=[]
    def slow(self,tokens):
        result=original(self,tokens);clock.now+=1;calls.append(1);return result
    monkeypatch.setattr(TinyTransformer,'forward',slow)
    with budget.session('validation fixture'):
        run=run_experiment(load_config(path),budget=budget)
    meta=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    checkpoint=torch.load(run/'latest.pt',weights_only=False)
    assert len(calls)==1
    assert meta['status']=='INTERRUPTED'
    assert checkpoint['step']==0 and checkpoint['pending_validation']
    assert budget.used_seconds<=1.5


def test_evaluation_cli_is_dry_and_routes_course_work_to_owner_budget(tmp_path,monkeypatch,capsys):
    from optcourse.cli import main
    import optcourse.evaluation as evaluation
    root=tmp_path/'study';run=root/'language'/'trial';run.mkdir(parents=True)
    (root/'study_state.json').write_text('{}')
    (run/'metadata.json').write_text(json.dumps({'config':{'run':{'device':'cuda','cohort':'course_v1'}}}))
    calls=[]
    def fake_evaluate(path,split,checkpoint,budget=None):
        assert budget is not None and budget.start is not None
        assert budget.path.resolve()==(root/'gpu_budget.json').resolve()
        calls.append(1)
        return {'loss':2.}
    monkeypatch.setattr(evaluation,'evaluate_run',fake_evaluate)
    assert main(['evaluate','--run',str(run)])==0
    assert not calls
    assert json.loads(capsys.readouterr().out)['status']=='NOT_RUN'
    assert main(['evaluate','--run',str(run),'--execute','--budget-seconds','99999'])==2
    assert not calls and not (root/'gpu_budget.json').exists()
    assert main(['evaluate','--run',str(run),'--execute'])==0
    assert len(calls)==1
    assert len(json.loads((root/'gpu_budget.json').read_text())['attempts'])==1


@pytest.mark.training
@pytest.mark.parametrize('evaluation_step',[1,2])
def test_pending_periodic_or_final_validation_resumes_without_repeating_update(tmp_path,monkeypatch,evaluation_step):
    import torch
    import optcourse.experiments as experiments
    from optcourse.config import load_config
    from optcourse.data import prepare_data
    from optcourse.model import TinyTransformer
    from optcourse.budget import BudgetLedger
    source=tmp_path/'text.txt';source.write_bytes(b'abcde'*100)
    prepare_data(source,tmp_path/'data')
    path=tmp_path/'config.toml'
    path.write_text(f'[run]\ntask="language"\ndevice="cpu"\nphase="verification"\noutput_root="fixtures"\n[language]\ndata_dir="data"\nn_layer=1\nn_embd=8\nn_head=2\nblock_size=4\nbatch_size=1\ngrad_accum=1\nmax_steps=2\nwarmup_steps=0\neval_every={evaluation_step}\neval_mode="full"\n',encoding='utf-8')
    config=load_config(path)
    baseline=experiments.run_experiment(config)
    clock=FakeClock();budget=BudgetLedger(tmp_path/'budget.json',1.5,clock=clock)
    original_forward=TinyTransformer.forward
    original_validation=experiments._validation_loss
    armed=False;validation_calls=0
    def validation(*args,**kwargs):
        nonlocal armed,validation_calls
        validation_calls+=1;armed=validation_calls>=2
        try:return original_validation(*args,**kwargs)
        finally:armed=False
    def delayed(self,tokens):
        result=original_forward(self,tokens)
        if armed:clock.now+=1
        return result
    monkeypatch.setattr(experiments,'_validation_loss',validation)
    monkeypatch.setattr(TinyTransformer,'forward',delayed)
    with budget.session('pending validation fixture'):
        interrupted=experiments.run_experiment(config,budget=budget)
    checkpoint=torch.load(interrupted/'latest.pt',weights_only=False)
    assert checkpoint['step']==evaluation_step and checkpoint['pending_validation']
    monkeypatch.setattr(experiments,'_validation_loss',original_validation)
    monkeypatch.setattr(TinyTransformer,'forward',original_forward)
    resumed=experiments.run_experiment(config,resume=interrupted)
    expected=torch.load(baseline/'latest.pt',weights_only=False)
    actual=torch.load(resumed/'latest.pt',weights_only=False)
    for name,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)
    assert actual['step']==2 and not actual['pending_validation']


def test_gpu_jobs_collect_unreachable_previous_task_objects(tmp_path,monkeypatch):
    import gc,weakref
    import optcourse.experiments as experiments
    from optcourse.study import Study,recipes
    from optcourse.tracking import RunStore
    class Retained:
        pass
    old=Retained();old.cycle=old;reference=weakref.ref(old);del old
    gc.disable()
    def fake_run(config,resume=None,budget=None,on_store=None):
        assert reference() is None,'previous task cycle retained at next GPU job'
        store=RunStore.create(config['run']['output_root'],config)
        on_store(store)
        store.update(best_validation=2.)
        store.finish('COMPLETED')
        return store.path
    monkeypatch.setattr(experiments,'run_experiment',fake_run)
    try:
        study=Study(tmp_path/'study')
        result=study.run_job(recipes('language','pilot')[0])
        assert result['status']=='COMPLETED'
    finally:
        gc.enable();gc.collect()
