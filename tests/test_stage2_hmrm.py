import torch

from model.Meta_Models.hierarchical_horizon_residual_memory import (
    MultiScaleHorizonResidualMemory,
    BidirectionalHorizonMamba,
)


def make_model():
    return MultiScaleHorizonResidualMemory(
        rep_dim=128, pred_len=36, memory_size=10000,
        horizon_dim=128, decoder_dim=256, heads=4,
        dropout=0.0, mamba_inner_dim=64, mamba_state_dim=8,
    )


def test_hierarchical_mamba_shape_and_finite_output():
    torch.manual_seed(5)
    model = make_model()
    rep = torch.randn(2, 5, 128)
    base = torch.randn(2, 5, 36) * 10
    scale = torch.full((5,), 5.0)
    correction = model(rep, base, scale)
    assert correction.shape == base.shape
    assert torch.isfinite(correction).all()
    assert correction.abs().max() <= 0.5 * 5.0 + 1e-5


def test_bidirectional_selective_ssm_gradients():
    torch.manual_seed(7)
    module = BidirectionalHorizonMamba(
        model_dim=256, inner_dim=64, state_dim=8, dropout=0.0
    )
    x = torch.randn(3, 36, 256, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert module.forward_ssm.A_log.grad is not None
    assert module.backward_ssm.A_log.grad is not None
    assert module.fusion.weight.grad is not None


def test_memory_and_residual_heads_receive_gradients():
    torch.manual_seed(11)
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
    assert model.horizon_interaction.forward_ssm.A_log.grad is not None
    assert model.residual_head[0].weight.grad is not None
    assert model.gate_head[0].weight.grad is not None


def test_memory_capacity_is_10000_and_three_bands():
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
