"""Numerical checks for the single-variable Stage-I pattern-cost ablation."""
import pytest
import torch

from model.Meta_Models.pattern_moe import (
    _combine_history_future_pattern_costs,
    _fixed_router_gate,
)


def test_zero_pattern_weight_removes_only_pattern_cost_and_gradient():
    history = torch.tensor([[1.0, 2.0]], requires_grad=True)
    future = torch.tensor([[3.0, 4.0]], requires_grad=True)
    pattern = torch.tensor([[15.0, 30.0]], requires_grad=True)
    result = _combine_history_future_pattern_costs(
        history, future, pattern, future_weight=1.0, pattern_weight=0.0
    )
    assert torch.allclose(result, 0.5 * (history + future))
    result.sum().backward()
    assert torch.allclose(history.grad, torch.full_like(history, 0.5))
    assert torch.allclose(future.grad, torch.full_like(future, 0.5))
    assert pattern.grad is None


def test_uniform_pattern_weight_is_active_and_normalized():
    history = torch.tensor([[1.0, 2.0]])
    future = torch.tensor([[3.0, 4.0]])
    pattern = torch.tensor([[15.0, 30.0]], requires_grad=True)
    result = _combine_history_future_pattern_costs(
        history, future, pattern, future_weight=1.0, pattern_weight=0.5
    )
    assert torch.allclose(result, (history + future + 0.5 * pattern) / 2.5)
    result.sum().backward()
    assert torch.allclose(pattern.grad, torch.full_like(pattern, 0.2))


def test_uniform_router_remains_fixed_in_both_arms():
    gate = _fixed_router_gate(
        "uniform", 2, 4, torch.device("cpu"), torch.float32
    )
    assert torch.allclose(gate.sum(dim=-1), torch.ones(2, 4))
    assert torch.allclose(gate, torch.full((2, 4, 3), 1.0 / 3.0))


def test_negative_pattern_weight_is_rejected():
    with pytest.raises(ValueError):
        _combine_history_future_pattern_costs(
            torch.ones(2, 2),
            torch.ones(2, 2),
            torch.ones(2, 2),
            future_weight=1.0,
            pattern_weight=-0.1,
        )
