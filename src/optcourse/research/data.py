"""Pinned GPT-2 token arrays and document-stable held-out partitions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(2**20):digest.update(chunk)
    return digest.hexdigest()


def partition_stories(records):
    groups=([],[])
    for text in records:groups[hashlib.sha256(text.encode('utf-8')).digest()[0]%2].append(text)
    return groups


def save_arrays(root,parts,metadata):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise FileExistsError('Prepared dataset already exists')
    splits={}
    for name,values in parts.items():
        values=np.asarray(values,dtype=np.uint16);path=root/f'{name}.bin';values.tofile(path)
        splits[name]=dict(tokens=len(values),sha256=sha256(path))
    manifest=dict(schema=1,dtype='uint16',**metadata,splits=splits)
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


class TokenDataset:
    def __init__(self,root,verify=True):
        self.root=Path(root)
        self.manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        self.parts={}
        for name,row in self.manifest['splits'].items():
            path=self.root/f'{name}.bin'
            if path.stat().st_size!=2*row['tokens'] or (verify and sha256(path)!=row['sha256']):
                raise ValueError(f'Split hash/identity mismatch: {name}')
            self.parts[name]=np.memmap(path,mode='r',dtype=np.uint16)
        self.fingerprint=hashlib.sha256(json.dumps(self.manifest,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    def batch(self,split,batch_size,context,generator,device):
        data=self.parts[split]
        if len(data)<=context:raise ValueError('Split too short')
        offsets=torch.randint(0,len(data)-context,(batch_size,),generator=generator).tolist()
        x=torch.from_numpy(np.stack([data[i:i+context].astype(np.int64) for i in offsets]))
        y=torch.from_numpy(np.stack([data[i+1:i+context+1].astype(np.int64) for i in offsets]))
        return x.to(device),y.to(device)

    def evaluation_batches(self,split,context,batch_size,device):
        data=self.parts[split];starts=range(0,len(data)-1,context)
        for group in range(0,len(starts),batch_size):
            offsets=list(starts[group:group+batch_size]);length=max(min(context,len(data)-1-i) for i in offsets)
            x=np.zeros((len(offsets),length),dtype=np.int64);y=np.full_like(x,-1)
            for row,i in enumerate(offsets):
                n=min(context,len(data)-1-i);x[row,:n]=data[i:i+n];y[row,:n]=data[i+1:i+n+1]
            yield torch.from_numpy(x).to(device),torch.from_numpy(y).to(device)


def prepare_datasets(raw_root,output_root,train_cap=50_000_000):
    import pyarrow.parquet as pq
    import tiktoken
    raw_root,output_root=Path(raw_root),Path(output_root)
    source_manifest=json.loads((raw_root/'source_manifest.json').read_text(encoding='utf-8'))
    encoding=tiktoken.get_encoding('gpt2')
    ranks=sorted((key.hex(),value) for key,value in encoding._mergeable_ranks.items())
    encoding_hash=hashlib.sha256(json.dumps(ranks,separators=(',',':')).encode()).hexdigest()
    def encode_records(records,cap=None):
        chunks=[];total=0;count=0;document_hash=hashlib.sha256()
        for start in range(0,len(records),512):
            batch=records[start:start+512]
            encoded=encoding.encode_ordinary_batch(batch,num_threads=8)
            for text,tokens in zip(batch,encoded):
                tokens.append(encoding.eot_token)
                if cap is not None:tokens=tokens[:max(0,cap-total)]
                if not tokens:break
                chunks.append(np.asarray(tokens,dtype=np.uint16));total+=len(tokens);count+=1
                document_hash.update(hashlib.sha256(text.encode()).digest())
                if cap is not None and total>=cap:break
            if cap is not None and total>=cap:break
        return np.concatenate(chunks),dict(documents=count,document_sequence_sha256=document_hash.hexdigest())
    summaries={}
    for corpus,prefix in [('wikitext103','wiki'),('tinystories','stories')]:
        destination=output_root/corpus
        if (destination/'manifest.json').exists():
            summaries[corpus]=TokenDataset(destination).manifest;continue
        text=pq.read_table(raw_root/f'{prefix}_train.parquet',columns=['text']).column('text').to_pylist()
        train,train_meta=encode_records(text,train_cap);del text
        held=pq.read_table(raw_root/f'{prefix}_val.parquet',columns=['text']).column('text').to_pylist()
        if prefix=='stories':selection,test=partition_stories(held)
        else:selection,test=held,pq.read_table(raw_root/'wiki_test.parquet',columns=['text']).column('text').to_pylist()
        val,val_meta=encode_records(selection);test,test_meta=encode_records(test)
        relevant=[row for row in source_manifest if row['name'].startswith(prefix)]
        metadata=dict(corpus=corpus,tokenizer='tiktoken-0.9.0/gpt2',tokenizer_ranks_sha256=encoding_hash,
            vocabulary=50304,actual_vocabulary=encoding.n_vocab,eot_token=encoding.eot_token,sources=relevant,
            document_rule='official splits' if prefix=='wiki' else 'official validation content SHA256 first-byte parity',
            document_metadata=dict(train=train_meta,val=val_meta,test=test_meta))
        summaries[corpus]=save_arrays(destination,dict(train=train,val=val,test=test),metadata)
        print(json.dumps({'corpus':corpus,'tokens':{k:v['tokens'] for k,v in summaries[corpus]['splits'].items()}},indent=2),flush=True)
    return summaries


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--train-cap',type=int,default=50_000_000)
    args=parser.parse_args()
    if args.execute:prepare_datasets('data/research_v2/raw','data/research_v2/tokens',args.train_cap)
    else:print(json.dumps(dict(status='NOT_RUN',train_cap=args.train_cap)))
