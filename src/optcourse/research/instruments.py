"""Immutable virtual steps and full-vector neural directional curvature."""

from contextlib import contextmanager

import torch


@contextmanager
def virtual_step(model,deltas):
    parameters=dict(model.named_parameters())
    if set(deltas)-set(parameters):raise ValueError('Unknown parameter in virtual step')
    if any(d.shape!=parameters[n].shape for n,d in deltas.items()):raise ValueError('Virtual direction shape mismatch')
    values={n:p.detach().clone() for n,p in parameters.items()}
    grads={n:(p.grad,p.grad.clone() if p.grad is not None else None) for n,p in parameters.items()}
    buffers={n:b.clone() for n,b in model.named_buffers()}
    modes={module:module.training for module in model.modules()}
    rng=torch.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        with torch.no_grad():
            for name,delta in deltas.items():parameters[name].sub_(delta)
        yield model
    finally:
        with torch.no_grad():
            for name,p in parameters.items():
                p.copy_(values[name]);p.grad=grads[name][0]
                if p.grad is not None:p.grad.copy_(grads[name][1])
            for name,b in model.named_buffers():b.copy_(buffers[name])
        for module,training in modes.items():module.training=training
        torch.set_rng_state(rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)


@torch.no_grad()
def evaluate_candidates(model,candidates,batch,loss_fn):
    result={}
    with virtual_step(model,{}):
        model.eval();base=loss_fn(model,batch).item()
        for name,delta in candidates.items():
            with virtual_step(model,delta):value=loss_fn(model,batch).item()
            result[name]=dict(base_loss=base,loss=value,gain=base-value)
    return result


def directional_geometry(model,candidates,batch,loss_fn,layer_names=()):
    candidates={key:{name:delta.detach() for name,delta in values.items()} for key,values in candidates.items()}
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    device=next(model.parameters()).device
    from torch.nn.attention import sdpa_kernel,SDPBackend
    backend=sdpa_kernel([SDPBackend.MATH])
    rows={}
    with virtual_step(model,{}),torch.enable_grad(),backend,torch.autocast(device.type,enabled=False):
        model.eval();loss=loss_fn(model,batch)
        gradients=torch.autograd.grad(loss,tuple(parameters.values()),create_graph=True,allow_unused=True)
        gradients=dict(zip(parameters,gradients))
        for index,(key,deltas) in enumerate(candidates.items()):
            terms=[(gradients[n]*delta).sum() for n,delta in deltas.items() if n in gradients and gradients[n] is not None]
            projection=sum(terms,torch.zeros((),device=device,dtype=loss.dtype))
            a=projection.detach().item()
            if projection.requires_grad:
                hv=torch.autograd.grad(projection,tuple(parameters.values()),retain_graph=index<len(candidates)-1,allow_unused=True)
                b=sum((value*deltas[n]).sum().detach().item() for n,value in zip(parameters,hv) if value is not None and n in deltas)
                hv_map=dict(zip(parameters,hv))
            else:b=0.;hv_map={}
            norm2=sum(delta.square().sum().item() for delta in deltas.values())
            rows[key]=dict(alignment=a,linear_gain=a,curvature=b,predicted_gain=a-.5*b,
                normalized_curvature=b/norm2 if norm2 else 0.,step_norm=norm2**.5)
            rows[key]['cross_curvature']={other:sum((value*other_delta[n]).sum().detach().item()
                for n,value in hv_map.items() if value is not None and n in other_delta)
                for other,other_delta in candidates.items()}
            if layer_names:
                layers={}
                for name in layer_names:
                    gradient=gradients.get(name);delta=deltas.get(name);hvp=hv_map.get(name)
                    if gradient is None or delta is None:continue
                    layers[name]=dict(alignment=(gradient*delta).sum().detach().item(),gradient_norm=gradient.norm().detach().item(),
                        step_rms=delta.square().mean().sqrt().item(),curvature_contribution=(hvp*delta).sum().detach().item() if hvp is not None else 0.)
                rows[key]['layers']=layers
    return rows
