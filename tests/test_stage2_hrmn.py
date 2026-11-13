import torch

from model.Meta_Models.horizon_conditioned_residual_memory import (
    HorizonConditionedResidualMemory,
)


def test_hrmn_shape_and_bounded_correction():
    torch.manual_seed(11)
    model = HorizonConditionedResidualMemory(
        rep_dim=128, pred_len=36, memory_size=32,
        horizon_dim=8, hidden_dim=32, max_correction_ratio=0.25,
        dropout=0.0,
    )
    rep = torch.randn(4, 7, 128)
    base = torch.randn(4, 7, 36) * 10
    scale = torch.full((1, 7, 1), 5.0)
    correction = model(rep, base, scale)
    assert correction.shape == base.shape
    assert torch.isfinite(correction).all()
    assert correction.abs().max() <= 0.25 * 5.0 + 1e-6


def test_hrmn_parameters_receive_gradients():
    torch.manual_seed(13)
    model = HorizonConditionedResidualMemory(
        rep_dim=128, pred_len=36, memory_size=32,
        horizon_dim=8, hidden_dim=32, dropout=0.0,
    )
    rep = torch.randn(2, 5, 128)
    base = torch.randn(2, 5, 36)
    target = torch.randn(2, 5, 36)
    scale = torch.ones(1, 5, 1)
    prediction = base + model(rep, base, scale)
    loss = torch.nn.functional.smooth_l1_loss(prediction, target)
    loss.backward()
    assert model.memory_keys.grad is not None
    assert model.memory_values.grad is not None
    assert model.horizon_embedding.weight.grad is not None
    assert model.residual_decoder[0].weight.grad is not None


def test_memory_is_compact():
    model = HorizonConditionedResidualMemory(memory_size=256)
    assert model.memory_keys.shape == (256, 128)
    assert model.memory_values.shape == (256, 128)
