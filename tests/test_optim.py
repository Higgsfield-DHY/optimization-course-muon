import copy
import importlib

import pytest
import torch


def test_muon_exact_polar_step_and_decoupled_decay_have_hand_computed_values():
    optim = importlib.import_module("optcourse.optim")
    weight = torch.nn.Parameter(torch.ones(2, 2))
    weight.grad = torch.tensor([[3., 0.], [0., 4.]])
    method = optim.Muon([weight], lr=0.1, weight_decay=0.2,
                        momentum=0., nesterov=False, method="svd", aspect_scale=False)
    method.step()
    torch.testing.assert_close(weight, torch.tensor([[0.88, 0.98], [0.98, 0.88]]))


def test_ns_uses_empirical_polynomial_and_handles_zero_and_tall_matrices():
    optim = importlib.import_module("optcourse.optim")
    # One scalar step at singular value 1: 3.4445 - 4.7750 + 2.0315 = 0.7010.
    value = optim.orthogonalize(torch.ones(1, 1), method="ns", steps=1, eps=0.)
    torch.testing.assert_close(value, torch.tensor([[0.7010]]), atol=1e-6, rtol=1e-6)
    zero = optim.orthogonalize(torch.zeros(3, 2), method="svd")
    assert torch.count_nonzero(zero).item() == 0
    tall = optim.orthogonalize(torch.tensor([[3., 0.], [0., 4.], [0., 0.]]), method="svd")
    torch.testing.assert_close(tall, torch.tensor([[1., 0.], [0., 1.], [0., 0.]]))


def test_muon_rejects_vectors_and_does_not_modify_supplied_gradient():
    optim = importlib.import_module("optcourse.optim")
    with pytest.raises(ValueError):
        optim.Muon([torch.nn.Parameter(torch.ones(3))])
    weight = torch.nn.Parameter(torch.ones(2, 2))
    weight.grad = torch.tensor([[1., 2.], [3., 4.]])
    original = weight.grad.clone()
    optim.Muon([weight]).step()
    torch.testing.assert_close(weight.grad, original)


def test_parameter_ownership_excludes_embedding_head_and_norm_from_muon():
    model_module = importlib.import_module("optcourse.model")
    optim = importlib.import_module("optcourse.optim")
    model = model_module.TinyTransformer(dict(n_layer=1, n_head=2, n_embd=8,
                                              block_size=8, dropout=0.))
    bundle, summary = optim.build_optimizers(model, dict(name="muon", lr=.02))
    all_ids = [id(p) for item in bundle for group in item.param_groups for p in group["params"]]
    assert len(all_ids) == len(set(all_ids)) == len(list(model.parameters()))
    muon_ids = {id(p) for item in bundle if isinstance(item, optim.Muon)
                for group in item.param_groups for p in group["params"]}
    assert id(model.token_embedding.weight) not in muon_ids
    assert id(model.head.weight) not in muon_ids
    assert all(p.ndim == 2 for p in model.parameters() if id(p) in muon_ids)
    assert summary["muon_parameter_count"] > 0


def test_optimizer_state_roundtrip_preserves_next_manual_gradient_update():
    optim = importlib.import_module("optcourse.optim")
    first = torch.nn.Parameter(torch.ones(2, 2))
    one = optim.Muon([first], momentum=.9)
    first.grad = torch.tensor([[1., 2.], [0., 1.]])
    one.step()
    second = torch.nn.Parameter(first.detach().clone())
    two = optim.Muon([second], momentum=.9)
    two.load_state_dict(copy.deepcopy(one.state_dict()))
    for p in (first, second):
        p.grad = torch.tensor([[0., 1.], [3., 2.]])
    one.step()
    two.step()
    torch.testing.assert_close(first, second)
