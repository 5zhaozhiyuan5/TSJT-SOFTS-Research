"""Synthetic invariance checks for history-only, node-agnostic Graph WaveNet.

These tests are not four-city MAE/RMSE evidence.
"""
import torch

from tsjt_pgrad.models.graph_wavenet_backbone import GraphWaveNetBackbone


def _model():
    torch.manual_seed(20261109)
    return GraphWaveNetBackbone(
        residual_channels=8, dilation_channels=8, skip_channels=16,
        end_channels=16, blocks=1, layers=2, diffusion_order=1,
        dropout=0.0, use_knowledge_graph=False,
    ).eval()


def test_node_permutation_equivariance():
    """No fixed node ID embeddings: permuting sensors must permute predictions."""
    model = _model()
    x = torch.randn(2, 12, 7, 3)
    perm = torch.tensor([6, 2, 0, 5, 1, 4, 3])
    with torch.no_grad():
        original = model(x, horizon=12)
        permuted = model(x[:, :, perm, :], horizon=12)
    torch.testing.assert_close(permuted, original[:, :, perm, :], rtol=1e-4, atol=1e-5)


def test_historical_support_is_row_stochastic_and_permutation_equivariant():
    model = _model()
    x = torch.randn(2, 12, 7, 3)
    perm = torch.tensor([6, 2, 0, 5, 1, 4, 3])
    with torch.no_grad():
        support = model._historical_support(x)
        changed = model._historical_support(x[:, :, perm, :])
    assert support.shape == (2, 7, 7)
    torch.testing.assert_close(support.sum(dim=-1), torch.ones(2, 7))
    torch.testing.assert_close(changed, support[:, perm][:, :, perm])


def test_earlier_history_changes_do_not_require_future_labels():
    """Support and forecast accept only the input history, without target labels."""
    model = _model()
    x = torch.randn(1, 6, 5, 3, requires_grad=True)
    forecast = model(x, horizon=6)
    grad = torch.autograd.grad(forecast.sum(), x)[0]
    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()
    assert forecast.shape == (1, 6, 5, 1)
