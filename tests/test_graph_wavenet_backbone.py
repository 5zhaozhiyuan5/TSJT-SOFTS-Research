"""Graph WaveNet backbone regression tests; synthetic inputs are NOT traffic results."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.func import functional_call

PATH=Path(__file__).resolve().parents[1]/'src/tsjt_pgrad/models/graph_wavenet_backbone.py'
spec=importlib.util.spec_from_file_location('_tsjt_gwn_backbone',PATH)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GraphWaveNetBackbone=module.GraphWaveNetBackbone


def make(**kwargs):
    return GraphWaveNetBackbone(input_dim=3,canonical_steps=12,
        supported_horizons=(6,12),residual_channels=8,dilation_channels=8,
        skip_channels=16,end_channels=16,blocks=2,layers=2,kernel_size=2,
        diffusion_order=2,adaptive_dim=8,dropout=0.,**kwargs)


def test_four_city_node_counts_and_horizons_share_parameters():
    torch.manual_seed(42)
    m=make().eval()
    count=sum(p.numel() for p in m.parameters())
    for t,n in [(12,207),(6,524),(6,627),(12,325)]:
        with torch.no_grad():
            out=m(torch.randn(1,t,n,3),horizon=t)
        assert out.shape==(1,t,n,1)
        assert bool(torch.isfinite(out).all())
        assert sum(p.numel() for p in m.parameters())==count


def test_alignment_never_reads_future_labels():
    m=make()
    x=torch.arange(6.).reshape(1,6,1,1).expand(1,6,2,3)
    a=m.align_history(x)
    assert tuple(a.shape)==(1,12,2,3)
    torch.testing.assert_close(a[0,:,0,0],torch.tensor([0.,0.,.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.]))


def test_six_step_horizon_reads_10min_slots():
    m=make().eval()
    x=torch.randn(2,6,4,3)
    with torch.no_grad():
        a=m(x,horizon=6)
        b=m(x,horizon=12)
    torch.testing.assert_close(a,b[:,1::2])


def test_trainable_adaptive_graph_and_virtual_pgrad_unchanged_parameters():
    torch.manual_seed(10)
    m=make()
    x=torch.randn(2,6,7,3)
    loss=(m(x,horizon=6)-torch.randn(2,6,7,1)).abs().mean()
    named=list(m.named_parameters())
    gradients=torch.autograd.grad(loss,[p for _,p in named])
    assert torch.isfinite(loss)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum().item() for n,g in zip((n for n,_ in named),gradients) if n.startswith('node_'))>0
    before={n:p.detach().clone() for n,p in named}
    virtual={n:(p.detach()-0.01*g) for (n,p),g in zip(named,gradients)}
    vm=functional_call(m,virtual,(x,None),{'horizon':6})
    assert vm.shape==(2,6,7,1)
    assert bool(torch.isfinite(vm).all())
    for n,p in named:
        torch.testing.assert_close(p,before[n])


def test_optional_real_adjacency_and_input_validation():
    m=make(use_knowledge_graph=True).eval()
    x=torch.randn(2,12,5,3)
    with torch.no_grad():
        assert m(x,torch.eye(5)).shape==(2,12,5,1)
    with pytest.raises(ValueError,match='requires supplied adjacency'):
        m(x,None)
    with pytest.raises(ValueError,match='adjacency must have shape'):
        m(x,torch.eye(4))
    with pytest.raises(ValueError,match='unsupported horizon'):
        m(x,torch.eye(5),horizon=8)


def test_no_road_adjacency_needed_for_default():
    m=make().eval()
    x=torch.randn(1,12,4,3)
    with torch.no_grad():
        a=m(x,None)
        b=m(x,torch.eye(4))
    torch.testing.assert_close(a,b)
