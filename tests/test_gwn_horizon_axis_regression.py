"""Regression: horizon axis must be time, not the city node axis."""
import torch
from tsjt_pgrad.models.graph_wavenet_backbone import GraphWaveNetBackbone


def test_city_node_axis_independent_of_prediction_horizon():
    torch.manual_seed(19)
    model=GraphWaveNetBackbone(residual_channels=4,dilation_channels=4,
        skip_channels=8,end_channels=8,blocks=1,layers=1,
        diffusion_order=1,dropout=0.).eval()
    for history,nodes,horizon in ((12,7,12),(6,9,6),(6,5,12),(12,3,6)):
        x=torch.randn(2,history,nodes,3)
        with torch.no_grad():
            prediction=model(x,horizon=horizon)
        assert prediction.shape==(2,horizon,nodes,1)
        assert torch.isfinite(prediction).all()


def test_six_step_is_physical_10min_subsample_of_twelve_step_5min():
    torch.manual_seed(20)
    model=GraphWaveNetBackbone(residual_channels=4,dilation_channels=4,
        skip_channels=8,end_channels=8,blocks=1,layers=1,
        diffusion_order=1,dropout=0.).eval()
    x=torch.randn(1,6,7,3)
    with torch.no_grad():
        six=model(x,horizon=6)
        twelve=model(x,horizon=12)
    torch.testing.assert_close(six,twelve[:,1::2,:,:])
