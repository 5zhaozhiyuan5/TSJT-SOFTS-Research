"""Graph WaveNet adapter tests; synthetic data, no traffic accuracy claims."""
import torch
from torch.func import functional_call
from tsjt_pgrad.models.graph_wavenet_backbone import GraphWaveNetBackbone
from tsjt_pgrad.models.pgrad_model import TSJTPGrad


def small():
    return GraphWaveNetBackbone(residual_channels=8, dilation_channels=8,
        skip_channels=16, end_channels=16, blocks=1, layers=2,
        diffusion_order=1, dropout=0.)


def test_shapes_across_city_node_counts_and_resolutions():
    model = small().eval()
    for t, n in [(12, 207), (6, 524), (6, 627), (12, 325)]:
        with torch.no_grad():
            result = model(torch.randn(1, t, n, 3), horizon=t)
        assert result.shape == (1, t, n, 1)
        assert torch.isfinite(result).all()


def test_adjacency_ignored_when_knowledge_graph_disabled():
    model = small().eval()
    x = torch.randn(1, 12, 5, 3)
    with torch.no_grad():
        a = model(x, torch.eye(5))
        b = model(x, torch.rand(5, 5))
    torch.testing.assert_close(a, b)


def test_horizon_physical_time_alignment():
    model = small().eval()
    x = torch.randn(1, 6, 4, 3)
    with torch.no_grad():
        out6 = model(x, horizon=6)
        out12 = model(x, horizon=12)
    torch.testing.assert_close(out6, out12[:, 1::2])


def test_gradient_and_virtual_forward():
    model = small()
    x = torch.randn(2, 6, 5, 3)
    loss = (model(x, horizon=6) - torch.randn(2, 6, 5, 1)).abs().mean()
    grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    # The final post-skip graph path does not affect the prediction loss.
    virtual = {k: p.detach() - .001 * (torch.zeros_like(p) if g is None else g)
               for (k, p), g in zip(model.named_parameters(), grads)}
    assert functional_call(model, virtual, (x, None), {'horizon': 6}).shape == (2, 6, 5, 1)


def test_tsjt_wrapper_preserves_angle_controller():
    model = TSJTPGrad(residual_channels=8, dilation_channels=8,
        skip_channels=16, end_channels=16, blocks=1, layers=1)
    assert model.angle_controller is not None
    assert model(torch.randn(1, 12, 5, 3), None).shape == (1, 12, 5, 1)


def test_prediction_head_axis_regression():
    model = small().eval()
    with torch.no_grad():
        y = model(torch.randn(2, 12, 7, 3), horizon=12)
    assert y.shape == (2, 12, 7, 1)
