"""Single-device float32 Muon, adapted from KellerJordan/Muon (MIT).

Upstream commit: f98f1cacc0263b04290753e32be8d498c1efc806.
See third_party/Muon-LICENSE.txt and docs/algorithm.md for attribution and differences.
"""

import torch
from torch import nn

from .config import OPTIM_DEFAULTS


def orthogonalize(matrix, method="ns", steps=5, eps=1e-7, normalization="plus"):
    if matrix.ndim != 2:
        raise ValueError("Muon requires a two-dimensional matrix")
    if normalization not in {"plus", "max"}:
        raise ValueError("Unknown normalization")
    if method == "svd":
        u, singular, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
        tolerance = torch.finfo(singular.dtype).eps * max(matrix.shape) * singular.max()
        active = singular > tolerance
        return (u[:, active] @ vh[active, :]).to(matrix.dtype)
    if method != "ns" or not isinstance(steps, int) or steps < 1:
        raise ValueError("Use method=ns with positive steps, or method=svd")
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    norm = x.norm()
    if eps == 0 and norm == 0:
        return torch.zeros_like(matrix)
    x = x / (norm + eps if normalization == "plus" else norm.clamp_min(eps))
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return (x.T if transposed else x).to(matrix.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=.02, weight_decay=0., momentum=.95, nesterov=True,
                 method="ns", ns_steps=5, ns_eps=1e-7, aspect_scale=True, normalization="plus"):
        params = list(params)
        if not params or any(p.ndim != 2 for p in params):
            raise ValueError("Muon accepts a nonempty list of matrix parameters only")
        if lr <= 0 or weight_decay < 0 or not 0 <= momentum < 1 or lr * weight_decay >= 1:
            raise ValueError("Invalid Muon learning rate, decay or momentum")
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                                     nesterov=nesterov, method=method, ns_steps=ns_steps,
                                     ns_eps=ns_eps, aspect_scale=aspect_scale, normalization=normalization))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise ValueError("Sparse Muon gradients are unsupported")
                gradient = parameter.grad.detach().float()
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1 - group["momentum"])
                direction = gradient.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                update = orthogonalize(direction, group["method"], group["ns_steps"], group["ns_eps"], group["normalization"])
                if group["aspect_scale"]:
                    update = update * max(1., parameter.shape[0] / parameter.shape[1]) ** .5
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        return loss


def build_optimizers(model, optimizer_config):
    config = OPTIM_DEFAULTS | optimizer_config
    excluded = {id(p) for module in model.modules() if isinstance(module, nn.Embedding)
                for p in module.parameters(recurse=False)}
    if hasattr(model, "head"):
        excluded.update(id(p) for p in model.head.parameters())
    hidden = {id(module.weight) for module in model.modules() if isinstance(module, nn.Linear)
              and module.weight.ndim == 2 and id(module.weight) not in excluded}
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    muon_params = [(name, p) for name, p in named if config["name"] == "muon" and id(p) in hidden]
    aux_params = [(name, p) for name, p in named if not (config["name"] == "muon" and id(p) in hidden)]
    bundle = []
    if muon_params:
        bundle.append(Muon([p for _, p in muon_params], **{k: config[k] for k in
            ("lr", "weight_decay", "momentum", "nesterov", "method", "ns_steps", "ns_eps", "aspect_scale", "normalization")}))
    if aux_params:
        groups = [dict(params=[p for _, p in aux_params if p.ndim >= 2], weight_decay=config["weight_decay"]),
                  dict(params=[p for _, p in aux_params if p.ndim < 2], weight_decay=0.)]
        groups = [group for group in groups if group["params"]]
        bundle.append(torch.optim.AdamW(groups, lr=config["aux_lr"] if config["name"] == "muon" else config["lr"],
                                       betas=tuple(config["betas"]), eps=config["eps"]))
    owned = [id(p) for item in bundle for group in item.param_groups for p in group["params"]]
    if len(owned) != len(set(owned)) or set(owned) != {id(p) for _, p in named}:
        raise RuntimeError("Optimizer groups must cover every parameter exactly once")
    return bundle, dict(muon_parameter_count=sum(p.numel() for _, p in muon_params),
                        adamw_parameter_count=sum(p.numel() for _, p in aux_params),
                        muon_names=[name for name, _ in muon_params],
                        adamw_names=[name for name, _ in aux_params])
