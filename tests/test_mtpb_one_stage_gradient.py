"""Synthetic one-stage MTPB checks (not real traffic MAE/RMSE)."""
from __future__ import annotations
import torch
import pytest
import tsjt_pgrad.mtpb.integrated_meta as m


class ToyMTPB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor([.1,.3]))
    def forward(self,data,adj,stage='train'):
        x=data.x[...,0]
        features=torch.stack([x[:,:,-1],x[:,:,-2]],dim=-1)
        pred=(features*self.weight).sum(-1,keepdim=True).expand(-1,-1,36)
        return pred,data.y,adj


class ToyRep:
    device='cpu'
    def __init__(self):
        self.model=ToyMTPB()
        self.current_epoch=0
    def meta_train_revise(self,*a):
        raise AssertionError('native meta-update must NOT run in integrated mode')


class ToyDataset:
    def __init__(self,cities,*,samples=32,horizon=36):
        self.data_list=cities
        self.stage='target_maml' if cities == ['pems-bay'] else 'source_train'
        self.task_args={'batch_size':3, 'pred_num':horizon}
        self.x_list={}
        self.y_list={}
        self.A_list={}
        self.means_list={}
        self.stds_list={}
        for k,city in enumerate(cities):
            self.x_list[city]=torch.randn(samples,4,288,2)*.3+.7
            self.y_list[city]=torch.ones(samples,4,horizon)*(k+1)
            self.A_list[city]=torch.eye(4)
            self.means_list[city]=[0.]
            self.stds_list[city]=[1.]


@pytest.fixture
def geometry_stubs(monkeypatch):
    def principal(model,callback,params,sources,target,**kwargs):
        state={name:p.detach().clone().requires_grad_(True)
               for name,p in model.named_parameters() if p.requires_grad}
        # Exercise the functional-call virtual gradient interface.
        assert len(callback(0,state))==len(params)
        return sources[0],{
            'source_weights':[1./3]*3,
            'source_target_cosine':[1.]*3,
            'pairwise_cosine':[[1.]*3]*3,
            'principal_norm':1.}
    def rotation(source,target,principal,params,angle_controller,
                 return_phi_tensor=False):
        direction=source
        return (direction,{'rotated':False},torch.tensor(0.)) if return_phi_tensor \
            else (direction,{'rotated':False})
    def fusion(target,directions,params,**kwargs):
        return tuple(sum(g[i] for g in directions)/len(directions)
                     for i in range(len(params))),{
            'weights':[1./len(directions)]*len(directions)}
    monkeypatch.setattr(m,'source_interaction_principal_gradient',principal)
    monkeypatch.setattr(m,'rotate_source_gradient',rotation)
    monkeypatch.setattr(m,'residual_matching_fusion',fusion)


def test_replaces_native_source_meta_update(geometry_stubs):
    torch.manual_seed(11)
    rep=ToyRep()
    updater=m.TargetGuidedMTPBMetaUpdate(
        rep,m.IntegratedMetaConfig(inner_steps=1,order_count=1))
    old=rep.model.weight.detach().clone()
    result=updater.step(ToyDataset(['metr-la','chengdu_m','shenzhen']),
                        ToyDataset(['pems-bay']))
    assert rep.current_epoch==1
    assert result['target_train_only'] and result['horizon']==36
    assert result['realized_update_norm']>0
    assert not torch.equal(old,rep.model.weight.detach())


def test_reject_source_target_overlap():
    updater=m.TargetGuidedMTPBMetaUpdate(ToyRep())
    with pytest.raises(ValueError,match='excluding the target'):
        updater.step(ToyDataset(['pems-bay','chengdu_m','shenzhen']),
                     ToyDataset(['pems-bay']))


def test_enforce_mtpb_36_step_prediction(geometry_stubs):
    updater=m.TargetGuidedMTPBMetaUpdate(
        ToyRep(),m.IntegratedMetaConfig(inner_steps=1))
    with pytest.raises(ValueError,match='36'):
        updater.step(ToyDataset(['metr-la','chengdu_m','shenzhen'],horizon=12),
                     ToyDataset(['pems-bay']))


def test_support_query_gradients_do_not_update_actual_weights():
    rep=ToyRep()
    updater=m.TargetGuidedMTPBMetaUpdate(rep,m.IntegratedMetaConfig(inner_steps=2))
    dataset=ToyDataset(['metr-la'])
    support,_=updater.sample(dataset,'metr-la')
    query,_=updater.sample(dataset,'metr-la')
    initial=dict(rep.model.named_parameters())
    old=[p.detach().clone() for p in rep.model.parameters()]
    gradients,loss=updater.task_gradient(rep.model,support,query,initial)
    assert len(gradients)==len(initial) and loss>=0
    assert all(torch.equal(a,b) for a,b in zip(old,rep.model.parameters()))


def test_target_guidance_gradient_is_direct_and_does_not_update_weights():
    """PEMS-BAY gradients must be evaluated at the current shared weights."""
    rep=ToyRep()
    updater=m.TargetGuidedMTPBMetaUpdate(
        rep,m.IntegratedMetaConfig(inner_steps=2))
    guide=ToyDataset(['pems-bay'])
    pair,_=updater.sample(guide,'pems-bay')
    init=dict(rep.model.named_parameters())
    before=[p.detach().clone() for p in rep.model.parameters()]
    actual=updater.guidance_gradient(rep.model,pair,init)
    w=torch.nn.Parameter(before[0])
    x=pair[0].x[...,0]
    feats=torch.stack([x[:,:,-1],x[:,:,-2]],dim=-1)
    prediction=(feats*w).sum(-1,keepdim=True).expand(-1,-1,36)
    expected=torch.autograd.grad(
        torch.nn.functional.mse_loss(prediction,pair[0].y),w)[0]
    assert torch.allclose(actual[0],expected)
    assert all(torch.equal(a,b.detach()) for a,b in zip(before,rep.model.parameters()))


def test_no_target_support_adaptation_during_guided_step(geometry_stubs):
    torch.manual_seed(13)
    rep=ToyRep()
    updater=m.TargetGuidedMTPBMetaUpdate(
        rep,m.IntegratedMetaConfig(inner_steps=1,order_count=1))
    original=updater.task_gradient
    task_cities=[]
    def guarded(model,support,query,state):
        city=support[0].data_name
        task_cities.append(city)
        assert city!='pems-bay', 'target support adaptation is forbidden'
        return original(model,support,query,state)
    updater.task_gradient=guarded
    report=updater.step(ToyDataset(['metr-la','chengdu_m','shenzhen']),
                        ToyDataset(['pems-bay']))
    assert len(task_cities)>=3
    assert report['target_finetune_epochs']==0
    assert report['target_inner_steps']==0
    assert report['training_protocol']=='source_meta_with_target_gradient_guidance'


def test_target_gradient_cannot_update_model_when_all_source_gradients_zero(geometry_stubs):
    """Target guidance is not a hidden target SGD step."""
    torch.manual_seed(44)
    rep=ToyRep()
    updater=m.TargetGuidedMTPBMetaUpdate(
        rep,m.IntegratedMetaConfig(inner_steps=1,order_count=1))
    original=rep.model.weight.detach().clone()
    def zero_source_tasks(model,support,query,initial):
        return tuple(torch.zeros_like(p) for p in initial.values()),0.
    updater.task_gradient=zero_source_tasks
    report=updater.step(ToyDataset(['metr-la','chengdu_m','shenzhen']),
                        ToyDataset(['pems-bay']))
    assert report['realized_update_norm']==0
    assert torch.equal(original,rep.model.weight.detach())


def test_target_guidance_query_labels_are_disjoint():
    """MTPB target_maml stride=12, horizon=36: indices 1 apart leak labels."""
    torch.manual_seed(23)
    dataset=ToyDataset(['pems-bay'], samples=46, horizon=36)
    updater=m.TargetGuidedMTPBMetaUpdate(ToyRep())
    reference,query,ref_idx,query_idx=updater.target_guidance_pairs(
        dataset,'pems-bay')
    assert reference[0].x.shape[0] == 3
    assert query[0].x.shape[0] == 3
    assert (query_idx.min()-ref_idx.max()).item()*12 >= 36
    assert len(torch.unique(torch.cat((ref_idx,query_idx)))) == 6


def test_target_guidance_rejects_insufficient_separation():
    dataset=ToyDataset(['pems-bay'], samples=8, horizon=36)
    updater=m.TargetGuidedMTPBMetaUpdate(ToyRep())
    with pytest.raises(ValueError,match='label-disjoint'):
        updater.target_guidance_pairs(dataset,'pems-bay')


def test_target_guidance_rejects_test_split():
    dataset=ToyDataset(['pems-bay'], samples=46)
    dataset.stage='test'
    with pytest.raises(ValueError,match='target_maml'):
        m.TargetGuidedMTPBMetaUpdate.target_guidance_pairs(dataset,'pems-bay')


def test_pattern_entropy_reaches_actual_source_interaction(monkeypatch, geometry_stubs):
    """Enabled mode must condition real source-interaction call, not a dead helper."""
    from contextlib import contextmanager
    recorded = []
    @contextmanager
    def fake_capture(model):
        # The official retrieval hook is independently exercised in
        # test_mtpb_retrieval_confidence.py; this checks training integration.
        yield [(.25, {'1':.25,'3':.25,'6':.25,'12':.25,'24':.25})]
    def principal(model, callback, params, sources, target, **kwargs):
        recorded.append(kwargs['target_strength'])
        state={name:p.detach().clone().requires_grad_(True)
               for name,p in model.named_parameters() if p.requires_grad}
        assert len(callback(0,state)) == len(params)
        return sources[0], {
            'source_weights':[1./3]*3,
            'source_target_cosine':[1.]*3,
            'pairwise_cosine':[[1.]*3]*3,
            'principal_norm':1.}
    monkeypatch.setattr(m,'capture_retrieval_confidence',fake_capture)
    monkeypatch.setattr(m,'source_interaction_principal_gradient',principal)
    torch.manual_seed(42)
    rep=ToyRep()
    updater=m.TargetGuidedMTPBMetaUpdate(
        rep,m.IntegratedMetaConfig(
            inner_steps=1,order_count=1,retrieval_guidance_strength=1.))
    result=updater.step(ToyDataset(['metr-la','chengdu_m','shenzhen']),
                        ToyDataset(['pems-bay']))
    assert recorded == pytest.approx([.25])
    assert result['effective_target_alignment_strength'] == pytest.approx(.25)
    assert result['pattern_retrieval_confidence_by_hour']['24'] == pytest.approx(.25)
    assert result['realized_update_norm'] > 0
