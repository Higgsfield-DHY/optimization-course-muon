import importlib

import pytest
import torch


def test_schedule_reaches_warmup_peak_and_final_floor_without_training():
    module = importlib.import_module("optcourse.experiments")
    assert module.lr_factor(0, 100, 10, .1) == 0.
    assert module.lr_factor(10, 100, 10, .1) == 1.
    assert module.lr_factor(100, 100, 10, .1) == pytest.approx(.1)


def test_matrix_problem_has_requested_hessian_condition_and_exact_target():
    module = importlib.import_module("optcourse.experiments")
    a, b, target, initial = module.make_matrix_problem(2, 3, 100., 7)
    eigenvalues = torch.linalg.eigvalsh(a.T @ a)
    torch.testing.assert_close(eigenvalues, torch.tensor([1., 100.]), atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(a @ target, b)
    assert initial.shape == (2, 3)
