"""Tokenized causal Transformer used in the revised optimizer study."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..model import Block


class ResearchTransformer(nn.Module):
    def __init__(self,depth=4,width=384,heads=6,context=256,vocabulary=50304):
        super().__init__()
        self.config=dict(depth=depth,width=width,heads=heads,context=context,vocabulary=vocabulary)
        self.context,self.vocabulary=context,vocabulary
        self.token_embedding=nn.Embedding(vocabulary,width)
        self.position_embedding=nn.Embedding(context,width)
        self.blocks=nn.Sequential(*[Block(width,heads,0.) for _ in range(depth)])
        self.final_norm=nn.LayerNorm(width);self.head=nn.Linear(width,vocabulary,bias=False)
        self.apply(self._init);self.head.weight=self.token_embedding.weight
        for block in self.blocks:
            nn.init.normal_(block.attention.projection.weight,std=.02/math.sqrt(2*depth))
            nn.init.normal_(block.mlp[2].weight,std=.02/math.sqrt(2*depth))

    @staticmethod
    def _init(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=.02)
            if getattr(module,'bias',None) is not None:nn.init.zeros_(module.bias)

    def forward(self,tokens):
        if tokens.ndim!=2 or tokens.shape[1]>self.context:raise ValueError('Invalid token context')
        positions=torch.arange(tokens.shape[1],device=tokens.device)
        x=self.token_embedding(tokens)+self.position_embedding(positions)
        return self.head(self.final_norm(self.blocks(x)))


def token_loss(model,batch):
    x,y=batch;logits=model(x)
    return F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-1)


def microbatch_token_loss(model,batch,micro_batch=4):
    x,y=batch;total=0.;weight=0
    for start in range(0,len(x),micro_batch):
        xx,yy=x[start:start+micro_batch],y[start:start+micro_batch]
        n=(yy>=0).sum().item()
        total=total+token_loss(model,(xx,yy))*n;weight+=n
    return total/weight
