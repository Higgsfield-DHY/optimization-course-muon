import importlib
import json

import pytest


def test_comparison_uses_expected_final_cohort_and_keeps_failed_denominator(tmp_path):
    from optcourse.tracking import RunStore
    from optcourse.config import fingerprint
    from test_tracking import trial_config
    comparison=importlib.import_module('optcourse.comparison')
    jobs={}
    for seed,phase,status,value in ((0,'final','COMPLETED',2.),(1,'final','COMPLETED',4.),
                                    (2,'final','FAILED',0.),(101,'tuning','COMPLETED',.001)):
        config=trial_config(tmp_path,seed)
        config['run'].update(phase=phase,cohort='course_v1')
        store=RunStore.create(tmp_path/'fixtures',config)
        store.record(dict(step=3,objective=value,elapsed_seconds=2.))
        store.finish(status,'fixture failure' if status=='FAILED' else None)
        jobs[str(seed)]=dict(id=str(seed),phase=phase,task='matrix',variant='adamw',condition_number=1,
            seed=seed,status=status,run_path=str(store.path),config_fingerprint=fingerprint(config),wall_seconds=100.)
    state=tmp_path/'study_state.json'
    state.write_text(json.dumps(dict(jobs=jobs)),encoding='utf-8')
    result=comparison.compare_study(state,tmp_path/'reports',make_plots=False)
    assert result['expected_final_runs']==3
    assert result['completed_final_runs']==2
    assert result['groups'][0]['n']==2
    assert result['groups'][0]['score_mean']==3.
    assert result['groups'][0]['elapsed_mean']==100.
    assert result['groups'][0]['trajectory_elapsed_mean']==2.
    assert result['groups'][0]['score_std']==pytest.approx(2**.5)
    assert result['excluded'][0]['reason']=='fixture failure'
    jobs['missing']=dict(id='missing',phase='final',task='matrix',variant='ns5',condition_number=1,
        seed=0,status='NOT_RUN',run_path=None,config_fingerprint='not-started')
    state.write_text(json.dumps(dict(jobs=jobs)),encoding='utf-8')
    result=comparison.compare_study(state,tmp_path/'reports',make_plots=False)
    assert result['expected_final_runs']==4
    assert len(result['excluded'])==2
