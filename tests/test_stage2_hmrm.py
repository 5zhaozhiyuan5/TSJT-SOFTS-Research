import torch

from model.Meta_Models.hierarchical_horizon_residual_memory import (
    MultiScaleHorizonResidualMemory,
)


def make_model():
    return MultiScaleHorizonResidualMemory(
        rep_dim=128, pred_len=36, memory_size=10000,
        horizon_dim=128, decoder_dim=256, heads=4,
        decoder_layers=2, dropout=0.0,
    )


def test_hierarchical_memory_shape_and_finite_output():
    torch.manual_seed(5)
    model = make_model()
    rep = torch.randn(2, 5, 128)
    base = torch.randn(2, 5, 36) * 10
    scale = torch.full((5,), 5.0)
    correction = model(rep, base, scale)
    assert correction.shape == base.shape
    assert torch.isfinite(correction).all()
    assert correction.abs().max() <= 0.5 * 5.0 + 1e-5


def test_all_memory_levels_and_decoder_receive_gradients():
    torch.manual_seed(7)
    model = make_model()
    rep = torch.randn(2, 4, 128)
    base = torch.randn(2, 4, 36)
    target = torch.randn(2, 4, 36)
    correction = model(rep, base, torch.ones(4))
    loss = torch.nn.functional.smooth_l1_loss(base + correction, target)
    loss.backward()
    assert model.memory_keys.grad is not None
    assert model.memory_values.grad is not None
    assert model.horizon_embedding.weight.grad is not None
    assert model.horizon_interaction.layers[0].linear1.weight.grad is not None
    assert model.residual_head[0].weight.grad is not None
    assert model.gate_head[0].weight.grad is not None


def test_memory_capacity_matches_official_scale():
    model = make_model()
    assert model.total_slots == 10000
    assert model.global_slots + 3 * model.band_slots == 10000
    assert len(model.band_readers) == 3


def test_sampled_memory_regularizer_is_finite():
    model = make_model()
    penalty = model.regularization_loss()
    assert torch.isfinite(penalty)
    penalty.backward()
    assert model.memory_keys.grad is not None
