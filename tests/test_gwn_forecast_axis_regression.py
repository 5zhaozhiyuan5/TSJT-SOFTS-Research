"""Regression: the Graph WaveNet forecast axis must not be swapped with nodes."""
import pytest
import torch

from tsjt_pgrad.models.graph_wavenet_backbone import GraphWaveNetBackbone


@pytest.mark.parametrize("nodes,horizon", [(5, 12), (17, 12), (5, 6), (17, 6)])
def test_forecast_axis_is_horizon_not_nodes(nodes, horizon):
    model = GraphWaveNetBackbone(input_dim=3, residual_channels=4,
                                 dilation_channels=4, skip_channels=8,
                                 end_channels=8, blocks=1, layers=1, dropout=0.0)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.randn(2, horizon, nodes, 3), horizon=horizon)
    assert prediction.shape == (2, horizon, nodes, 1)
    assert torch.isfinite(prediction).all()


def test_six_step_output_samples_every_second_canonical_horizon():
    model = GraphWaveNetBackbone(input_dim=3, residual_channels=4,
                                 dilation_channels=4, skip_channels=8,
                                 end_channels=8, blocks=1, layers=1, dropout=0.0)
    model.eval()
    x = torch.randn(2, 12, 7, 3)
    with torch.no_grad():
        full = model(x, horizon=12)
        reduced = model(x, horizon=6)
    torch.testing.assert_close(reduced, full[:, 1::2])
