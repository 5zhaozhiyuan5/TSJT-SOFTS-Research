"""Synthetic integration checks; no claim of MTPB four-city MAE/RMSE."""
from __future__ import annotations
import torch
import pytest
import tsjt_pgrad.mtpb.integrated_meta as m


class ToyMTPB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor([.1,.3]))
    def forward(self,data,adj):
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
        self.task_args={'batch_size':3}
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
