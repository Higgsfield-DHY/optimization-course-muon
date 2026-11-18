import importlib
import json

import numpy as np
import pytest
import torch


def test_story_split_keeps_duplicate_documents_in_one_partition():
    g=importlib.import_module('optcourse.research.data')
    stories=['first story','second story','first story','third story']
    val,test=g.partition_stories(stories)
    assert set(val).isdisjoint(test)
    assert sorted(val+test)==sorted(stories)
    assert g.partition_stories(stories)==(val,test)


def test_token_dataset_sampler_and_tail_cover_every_target_once(tmp_path):
    g=importlib.import_module('optcourse.research.data')
    parts={name:np.arange(length,dtype=np.uint16)%17 for name,length in [('train',41),('val',22),('test',19)]}
    g.save_arrays(tmp_path,parts,dict(tokenizer='fixture',vocabulary=17))
    ds=g.TokenDataset(tmp_path)
    gen=torch.Generator().manual_seed(42);state=gen.get_state()
    x,y=ds.batch('train',2,5,gen,'cpu');gen.set_state(state)
    xx,yy=ds.batch('train',2,5,gen,'cpu')
    assert torch.equal(x,xx) and torch.equal(y,yy)
    targets=[]
    for _,y in ds.evaluation_batches('val',5,3,'cpu'):targets.extend(y[y>=0].tolist())
    assert targets==parts['val'][1:].tolist()
    assert len(targets)==21


def test_token_identity_detects_changed_split(tmp_path):
    g=importlib.import_module('optcourse.research.data')
    parts={k:np.arange(10,dtype=np.uint16) for k in ('train','val','test')}
    g.save_arrays(tmp_path,parts,dict(tokenizer='fixture',vocabulary=17))
    path=tmp_path/'test.bin';payload=bytearray(path.read_bytes());payload[0]^=1;path.write_bytes(payload)
    with pytest.raises(ValueError,match='hash|identity'):g.TokenDataset(tmp_path)


def test_token_model_is_causal_and_ties_head():
    g=importlib.import_module('optcourse.research.language')
    torch.manual_seed(3);model=g.ResearchTransformer(depth=2,width=16,heads=2,context=8,vocabulary=64)
    x=torch.randint(0,64,(2,7));y=x.clone();y[:,4:]=torch.randint(0,64,(2,3))
    assert model(x).shape==(2,7,64)
    assert torch.equal(model(x)[:,:4],model(y)[:,:4])
    assert model.head.weight is model.token_embedding.weight


def test_microbatch_probe_loss_matches_full_batch_and_gradients():
    g=importlib.import_module('optcourse.research.language')
    model=g.ResearchTransformer(depth=1,width=8,heads=2,context=4,vocabulary=16)
    batch=(torch.randint(0,16,(5,4)),torch.randint(0,16,(5,4)))
    full=g.token_loss(model,batch)
    reference=torch.autograd.grad(full,tuple(model.parameters()))
    chunked=g.microbatch_token_loss(model,batch,2)
    actual=torch.autograd.grad(chunked,tuple(model.parameters()))
    assert full.item()==pytest.approx(chunked.item(),rel=1e-6)
    assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-5) for a,b in zip(reference,actual))
