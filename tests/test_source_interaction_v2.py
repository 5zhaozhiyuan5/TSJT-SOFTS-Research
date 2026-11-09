"""Source-only module regression tests: rotation/fusion/backbone are unchanged."""
import math

import pytest
import torch
from torch import nn

from conftest import ToyModel
from tsjt_pgrad.pgrad.source_interaction import (
    balanced_city_orders, combine_source_interaction, source_interaction_principal_gradient,
)
from tsjt_pgrad.training import trainer


def _parameters():
    m=nn.Linear(4, 1, bias=False)
    return m, tuple(m.parameters())


def _g(x):
    return (torch.tensor([x], dtype=torch.float32).reshape(1, 4),)


def test_01_balanced_city_orders():
    assert balanced_city_orders(3, 3) == [(0,1,2),(1,2,0),(2,0,1)]
    orders=balanced_city_orders(3,6)
    assert len(orders)==6 and len(set(orders))==6
    with pytest.raises(ValueError): balanced_city_orders(3,0)


def test_02_source_matrix_and_weights_are_valid():
    m,p=_parameters()
    s=[_g([1,0,0,0]),_g([0,1,0,0]),_g([0,-1,0,0])]
    t=_g([1,0,0,0])
    pg=[_g([1,1,1,0]),_g([1,2,1,0]),_g([1,1,2,0])]
    output,meta=combine_source_interaction(p,s,t,pg,weight_ema=0)
    assert len(output)==len(p)
    assert torch.isfinite(output[0]).all()
    assert math.isclose(sum(meta['source_weights']),1,abs_tol=1e-6)
    assert meta['source_weights'][0]>meta['source_weights'][2]
    assert meta['pairwise_cosine'][1][2] == pytest.approx(-1.0,abs=1e-6)
    assert abs(meta['target_orthogonal_dot'])<1e-5


def test_03_redundancy_logged_for_near_duplicates():
    _,p=_parameters()
    s=[_g([0,1,0,0]),_g([0,1,0,0]),_g([0,0,1,0])]
    _,meta=combine_source_interaction(p,s,_g([1,0,0,0]),[_g([0,1,1,0])])
    assert meta['source_redundancy'][0]>0
    assert meta['source_redundancy'][1]>0
    assert meta['source_redundancy'][2]==0


def test_04_source_weight_ema_only_applies_to_scores():
    _,p=_parameters()
    s=[_g([1,1,0,0]),_g([-1,0,1,0]),_g([-1,0,0,1])]
    _,meta=combine_source_interaction(p,s,_g([1,0,0,0]),[_g([1,1,1,0])],
                                       previous_weights=[0,1,0],weight_ema=.9)
    assert meta['source_weights'][1]>.8


def test_05_zero_source_returns_finite_zero_vector():
    _,p=_parameters()
    s=[_g([0,0,0,0]) for _ in range(3)]
    output,meta=combine_source_interaction(p,s,_g([1,0,0,0]),[_g([0,1,0,0])])
    assert meta['zero_source_gradients']
    assert torch.count_nonzero(output[0])==0


def test_06_real_virtual_source_gradients_and_model_unchanged():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.tensor([1.,-1.,1.,0.]))
    m=Tiny(); before=m.weight.detach().clone();p=[m.weight]
    centers=[torch.tensor([3.,0.,0.,0.]),torch.tensor([0.,2.,0.,0.]),torch.tensor([0.,0.,3.,0.])]
    def cb(city,pp):
        loss=((pp['weight']-centers[city])**2).sum()/2
        return torch.autograd.grad(loss,[pp['weight']])
    per_city=[(m.weight.detach()-c,) for c in centers]
    output,meta=source_interaction_principal_gradient(
        m,cb,p,per_city,(torch.tensor([1.,0.,0.,0.]),),
        virtual_lr=.03,order_count=3,num_components=2)
    assert meta['mode']=='source_interaction_v2'
    assert len(meta['city_orders'])==3
    assert len(meta['source_weights'])==3
    assert torch.equal(m.weight.detach(),before)
    assert torch.isfinite(output[0]).all()


def test_07_rotation_and_fusion_unchanged_call_sites_execute():
    # End-to-end _train_step mock contract: only new source-interaction parameter.
    model=ToyModel();params=list(model.backbone.parameters())
    target_x=torch.tensor([[1.,0.,0.],[1.,0.,0.]])
    target_y=torch.ones(2,1)
    chunks=[
        (torch.tensor([[1.,1.,0.],[1.,1.,0.]]),-torch.ones(2,1),torch.eye(1)),
        (torch.tensor([[1.,0.,1.],[1.,0.,1.]]),-torch.ones(2,1),torch.eye(1)),
        (torch.tensor([[1.,1.,1.],[1.,1.,1.]]),-torch.ones(2,1),torch.eye(1)),
    ]
    cfg={'train':{
       'gradient_method':'full_method','gamma_target':.02,'gamma_source':.01,
       'pgrad_virtual_lr':.01,'pgrad_num_components':2,'pgrad_rounds':1,
       'source_interaction_mode':'source_interaction_v2',
       'source_interaction_order_count':3,
       'meta_interval':1,'meta_step_scale':5.,'meta_lr':.001,
       'fusion_interval':1,'fusion_lambda':1.,'fusion_mu_w':1e-3,
       'source_budget_ratio':0.0,
    }}
    r=(torch.tensor([[0.,0.,1.]]),torch.tensor([[1.]]))
    w=(torch.tensor([[0.,1.,0.]]),torch.tensor([[1.]]))
    state={}
    d=trainer._train_step(model,params,target_x,target_y,torch.eye(1),chunks,
         cfg,torch.device('cpu'),0,state=state,query_r=r,query_w=w)
    assert d['path']==['per_city_gradient','pgrad','rotation','residual_matching']
    assert d['pgrad']['mode']=='source_interaction_v2'
    assert len(d['rotation'])==3
    assert len(d['fusion']['weights'])==3
    assert len(state['source_interaction_weights'])==3
    assert torch.isfinite(model.backbone.weight).all()


def test_08_reject_bad_configuration():
    _,p=_parameters()
    with pytest.raises(ValueError):
        combine_source_interaction(p,[_g([1,0,0,0]),_g([0,1,0,0])],_g([1,0,0,0]),
                                   [_g([1,1,0,0])],temperature=0)


def test_09_inactive_city_gets_no_weight_even_with_ema_and_uniform_floor():
    _, p = _parameters()
    sources = [_g([0, 1, 0, 0]), _g([0, 0, 0, 0]), _g([0, 0, 1, 0])]
    _, meta = combine_source_interaction(
        p, sources, _g([1, 0, 0, 0]), [_g([0, 1, 1, 0])],
        previous_weights=[0.0, 1.0, 0.0], weight_ema=0.9,
        uniform_mix=0.3,
    )
    assert meta['source_weights'][1] == pytest.approx(0.0, abs=1e-8)
    assert sum(meta['source_weights']) == pytest.approx(1.0, abs=1e-6)
    assert meta['source_weights'][0] > 0.0
    assert meta['source_weights'][2] > 0.0


def test_10_single_active_city_has_unit_weight():
    _, p = _parameters()
    sources = [_g([0, 1, 0, 0]), _g([0, 0, 0, 0]), _g([0, 0, 0, 0])]
    out, meta = combine_source_interaction(
        p, sources, _g([1, 0, 0, 0]), [_g([0, 1, 0, 0])],
        previous_weights=[0.0, 0.5, 0.5], weight_ema=0.95,
    )
    assert meta['source_weights'] == pytest.approx([1.0, 0.0, 0.0])
    assert torch.isfinite(out[0]).all()
