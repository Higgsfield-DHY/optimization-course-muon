import importlib

import numpy as np
import pytest
import torch


@pytest.mark.training
def test_probe_bank_reads_exact_candidates_and_generates_independent_outcomes(tmp_path):
    probe=importlib.import_module('optcourse.research.probes')
    from optcourse.research.data import save_arrays,TokenDataset
    from optcourse.research.language import ResearchTransformer,token_loss
    from optcourse.research.optimizer import ResearchOptimizer
    gen=np.random.default_rng(8)
    save_arrays(tmp_path/'data',{k:gen.integers(0,32,n,dtype=np.uint16) for k,n in [('train',300),('val',35),('test',31)]},dict(tokenizer='fixture',vocabulary=32))
    dataset=TokenDataset(tmp_path/'data');model=ResearchTransformer(depth=1,width=8,heads=2,context=4,vocabulary=32)
    optimizer=ResearchOptimizer(model,'ns5',lr=.001,aux_lr=.001,weight_decay=.1)
    batch=dataset.batch('train',2,4,torch.Generator().manual_seed(5),'cpu');token_loss(model,batch).backward()
    prepared=optimizer.prepare(depths=(1,3,5))
    momenta={n:.95*prepared.states[model.get_parameter(n)]['buffer']+.05*model.get_parameter(n).grad for n in optimizer.hidden_names}
    config=dict(model=model.config,lr=.001,aux_lr=.001,weight_decay=.1,method='ns5',seed=0,corpus='fixture',scale='tiny',cohort='verification')
    path=tmp_path/'run/anchors/step-1.pt';path.parent.mkdir(parents=True)
    torch.save(dict(model=model.state_dict(),candidates=prepared.deltas,momenta=momenta,hidden_names=sorted(momenta),selected='ns5',config=config,
        step=1,base_completed_updates=0,lr_factor=1.,config_fingerprint='fixture',dataset_fingerprint=dataset.fingerprint),path)
    result=probe.investigate_anchor(path,tmp_path/'probe',dataset,repeats=1)
    assert result['status']=='COMPLETED'
    assert len(result['rows'])==15
    assert {r['variant'] for r in result['rows']}=={'ns1','ns3','ns5','cubic5','polar'}
    assert all(r['b_batch_sha256']!=r['c_batch_sha256'] for r in result['rows'])
    assert all(np.isfinite(r['actual_gain']) for r in result['rows'])
    assert all(r['source_anchor_sha256']==result['source_anchor_sha256'] for r in result['rows'])
