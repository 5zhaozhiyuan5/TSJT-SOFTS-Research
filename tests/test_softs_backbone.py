"""SOFTS/TSJT adapter tests: no real traffic data or tsjt_c1 install needed."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.func import functional_call

# Import by path so these checks run even when the original TSJT test fixture
# installs mock `tsjt_pgrad.models.pgrad_model` for legacy tests.
PATH=Path(__file__).resolve().parents[1]/'src/tsjt_pgrad/models/softs_backbone.py'
spec=importlib.util.spec_from_file_location('_tvsgi_softs_backbone', PATH)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SOFTSBackbone=module.SOFTSBackbone


def model(**kwargs):
    return SOFTSBackbone(input_dim=3, canonical_steps=12, supported_horizons=(6,12),
                         d_model=16,d_core=8,d_ff=24,e_layers=1,dropout=0.,**kwargs)


def test_shapes_for_4_cities_and_mixed_temporal_resolution():
    torch.manual_seed(21)
    m=model()
    for length,n in [(12,207),(6,524),(6,627),(12,325)]:
        x=torch.randn(2,length,n,3)
        y=m(x,torch.eye(n),horizon=length)
        assert y.shape==(2,length,n,1)
        assert torch.isfinite(y).all()


def test_history_10min_to_5min_uses_only_historical_data_and_clock_alignment():
    m=model()
    x=torch.arange(6.).view(1,6,1,1).repeat(1,1,2,3)
    y=m.align_history(x)
    assert y.shape==(1,12,2,3)
    assert y[0,0,0,0]==0  # earlier -55min held at earliest -50min value
    assert y[0,1,0,0]==0  # -50min
    assert y[0,2,0,0]==.5 # -45min
    assert y[0,-1,0,0]==5 # now


def test_6step_readout_matches_10min_positions_of_12step_prediction():
    m=model()
    x=torch.randn(2,6,4,3)
    m.eval()
    out6=m(x,horizon=6)
    out12=m(x,horizon=12)
    torch.testing.assert_close(out6,out12[:,1::2])


def test_different_node_counts_have_same_shared_weights_and_no_graph_dependency():
    m=model()
    before={k:v.detach().clone() for k,v in m.named_parameters()}
    x=torch.randn(2,12,3,3)
    m.eval()
    y1=m(x,torch.eye(3),horizon=12)
    y2=m(x,torch.rand(3,3),horizon=12)
    torch.testing.assert_close(y1,y2)
    y3=m(torch.randn(1,12,5,3),torch.eye(5),horizon=12)
    assert y3.shape==(1,12,5,1)
    for k,v in m.named_parameters():torch.testing.assert_close(v,before[k])


def test_real_gradients_and_virtual_parameter_call():
    torch.manual_seed(22)
    m=model()
    x=torch.randn(2,6,5,3)
    y=torch.randn(2,6,5,1)
    loss=(m(x,horizon=6)-y).abs().mean()
    grads=torch.autograd.grad(loss,tuple(m.parameters()))
    assert any(g.abs().sum()>0 for g in grads)
    virtual={name:(p.detach()-0.01*g) for (name,p),g in zip(m.named_parameters(),grads)}
    baseline={name:p.detach().clone() for name,p in m.named_parameters()}
    vpred=functional_call(m,virtual,(x,None),{'horizon':6})
    assert vpred.shape==y.shape
    assert torch.isfinite(vpred).all()
    for k,p in m.named_parameters():torch.testing.assert_close(p,baseline[k])


def test_training_soft_pool_is_deterministic_without_dropout():
    torch.manual_seed(23)
    m=model(star_pooling='soft')
    m.train()
    x=torch.randn(2,12,3,3)
    a=m(x)
    b=m(x)
    torch.testing.assert_close(a,b)


def test_training_stochastic_pool_is_available_as_official_ablation():
    m=model(star_pooling='stochastic')
    m.train()
    y=m(torch.randn(2,12,3,3))
    assert y.shape==(2,12,3,1)


def test_parameter_count_independent_of_city_nodes():
    m=model()
    before=sum(p.numel() for p in m.parameters())
    m(torch.randn(2,12,3,3));m(torch.randn(2,6,7,3),horizon=6)
    after=sum(p.numel() for p in m.parameters())
    assert before==after


def test_integration_tvsgi_rotation_meta_fusion_with_real_softs_backbone():
    """Real SOFTS + unchanged training module (mocked external tsjt_c1 data engine)."""
    from tsjt_pgrad.training import trainer
    from tsjt_pgrad.pgrad.rotation import AngleController
    torch.manual_seed(29)

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=model(use_time_features=True)
            self.angle_controller=AngleController(input_dim=3,hidden_dim=16)
        def forward(self,x,adjacency,horizon=None):
            return self.backbone(x,adjacency,horizon=horizon)

    wrapped=Wrapped()
    params=list(wrapped.backbone.parameters())
    tx=torch.randn(2,12,5,3)
    ty=torch.randn(2,12,5,1)
    sources=[]
    for n,h in [(4,12),(6,6),(7,6)]:
        sx=torch.randn(2,h,n,3)
        sy=torch.randn(2,h,n,1)
        sources.append((sx,sy,torch.eye(n)))
    cfg={'train':{
       'gradient_method':'full_method','gamma_target':.0002,'gamma_source':.0001,
       'pgrad_virtual_lr':.001,'pgrad_num_components':2,'pgrad_rounds':1,
       'source_interaction_mode':'source_interaction_v2',
       'source_interaction_order_count':3,'meta_interval':1,
       'meta_step_scale':1.,'meta_lr':.001,'fusion_interval':1,
       'fusion_lambda':1.,'fusion_mu_w':1e-3,'source_budget_ratio':.3,
    }}
    before=[p.detach().clone() for p in params]
    r=(torch.randn(2,12,5,3),torch.randn(2,12,5,1))
    w=(torch.randn(2,12,5,3),torch.randn(2,12,5,1))
    diag=trainer._train_step(wrapped,params,tx,ty,torch.eye(5),sources,cfg,
                             torch.device('cpu'),0,state={},query_r=r,query_w=w)
    assert diag['pgrad']['mode']=='source_interaction_v2'
    assert len(diag['pgrad']['city_orders'])==3
    assert len(diag['rotation'])==3
    assert len(diag['fusion']['weights'])==3
    assert all(torch.isfinite(p).all() for p in params)
    assert any(not torch.equal(b,p) for b,p in zip(before,params))
