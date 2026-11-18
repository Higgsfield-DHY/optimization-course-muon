"""Pure prepared steps with common auxiliary updates and RMS-matched matrices."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .geometry import candidate_directions,match_rms


@dataclass
class PreparedStep:
    owner: int
    version: int
    deltas: dict
    states: dict


class ResearchOptimizer(torch.optim.Optimizer):
    def __init__(self,model,method,lr=.001,aux_lr=.001,weight_decay=.1,rms=.2,
                 momentum=.95,ns_dtype='float32'):
        if method not in {'adamw','adam_rms','momentum','ns1','ns3','ns5','canonical5','adaptive','replay','spectral'}:
            raise ValueError('Unknown research optimizer')
        self.named_parameters=dict(model.named_parameters())
        excluded={id(p) for module in model.modules() if isinstance(module,nn.Embedding) for p in module.parameters(recurse=False)}
        if hasattr(model,'head'):excluded.update(id(p) for p in model.head.parameters())
        hidden={id(module.weight) for module in model.modules() if isinstance(module,nn.Linear) and id(module.weight) not in excluded}
        self.hidden_names={n for n,p in self.named_parameters.items() if id(p) in hidden}
        self.auxiliary_names=set(self.named_parameters)-self.hidden_names
        self.method,self.lr,self.aux_lr,self.decay,self.rms=method,lr,aux_lr,weight_decay,rms
        self.momentum,self.ns_dtype,self.version,self.depth=momentum,ns_dtype,0,5
        super().__init__(list(self.named_parameters.values()),dict(lr=lr))

    @torch.no_grad()
    def prepare(self,lr_factor=1.,depths=None):
        if depths is not None:
            keys=[f'ns{k}' for k in depths]
        elif self.method in {'adaptive','replay','spectral'}:keys=[f'ns{self.depth}']
        else:keys=[self.method]
        proposals={key:{} for key in keys};states={}
        for name,p in self.named_parameters.items():
            if p.grad is None:continue
            gradient=p.grad.detach().float();old=self.state.get(p,{})
            hidden=name in self.hidden_names
            lr=(self.lr if hidden else self.aux_lr)*lr_factor
            decay=self.decay if p.ndim>=2 else 0.
            if not hidden or self.method in {'adamw','adam_rms'}:
                step=old.get('step',0)+1
                m=.9*old.get('m',torch.zeros_like(gradient))+.1*gradient
                v=.95*old.get('v',torch.zeros_like(gradient))+.05*gradient.square()
                direction=(m/(1-.9**step))/(torch.sqrt(v/(1-.95**step))+1e-8)
                if hidden and self.method=='adam_rms':direction=match_rms(direction,self.rms)
                delta=lr*(direction+decay*p)
                for key in keys:proposals[key][name]=delta
                states[p]=dict(step=step,m=m,v=v)
            else:
                buffer=self.momentum*old.get('buffer',torch.zeros_like(gradient))+(1-self.momentum)*gradient
                momentum=self.momentum*buffer+(1-self.momentum)*gradient
                if self.method=='momentum':raw={keys[0]:momentum}
                else:
                    counts=tuple(int(k[2:]) if k.startswith('ns') else 5 for k in keys)
                    source=momentum.to(torch.bfloat16) if self.ns_dtype=='bfloat16' and p.device.type=='cuda' else momentum
                    candidates=candidate_directions(source,counts)
                    raw={key:candidates[f'ns{count}'].float() for key,count in zip(keys,counts)}
                for key,direction in raw.items():
                    if self.method=='canonical5':direction=direction*(.2*math.sqrt(max(p.shape)))
                    else:direction=match_rms(direction,self.rms)
                    proposals[key][name]=lr*(direction+decay*p)
                states[p]=dict(buffer=buffer)
        return PreparedStep(id(self),self.version,proposals,states)

    @torch.no_grad()
    def commit(self,prepared,choice):
        if prepared.owner!=id(self) or prepared.version!=self.version:
            raise ValueError('Prepared step is stale or already committed')
        if choice not in prepared.deltas:raise ValueError('Unknown prepared candidate')
        before={name:p.clone() for name,p in self.named_parameters.items()}
        old_states={p:self.state.get(p) for p in prepared.states}
        try:
            for name,delta in prepared.deltas[choice].items():self.named_parameters[name].sub_(delta)
            for p,state in prepared.states.items():self.state[p]=state
        except BaseException:
            for name,p in self.named_parameters.items():p.copy_(before[name])
            for p,state in old_states.items():
                if state is None:self.state.pop(p,None)
                else:self.state[p]=state
            raise
        self.version+=1

    def state_dict(self):
        result=super().state_dict();result['research']=dict(version=self.version,depth=self.depth,method=self.method)
        return result

    def load_state_dict(self,state_dict):
        data=dict(state_dict);metadata=data.pop('research')
        if metadata['method']!=self.method:raise ValueError('Optimizer method changed')
        super().load_state_dict(data);self.version=metadata['version'];self.depth=metadata['depth']


@torch.no_grad()
def spectral_candidate_errors(model,optimizer,prepared,lr_factor):
    rate=optimizer.lr*lr_factor;errors={key:0. for key in prepared.deltas};denominator=0.
    for name in optimizer.hidden_names:
        p=model.get_parameter(name)
        momentum=optimizer.momentum*prepared.states[p]['buffer']+(1-optimizer.momentum)*p.grad
        reference=match_rms(candidate_directions(momentum.detach().cpu().double(),(0,),include_polar=True)['polar'],optimizer.rms)
        base=p.detach().cpu().double();denominator+=reference.square().sum().item()
        for key,values in prepared.deltas.items():
            direction=values[name].detach().cpu().double()/rate-optimizer.decay*base
            errors[key]+=(direction-reference).square().sum().item()
    return {key:math.sqrt(value/denominator) if denominator else 0. for key,value in errors.items()}
