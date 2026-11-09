"""Adapter tests with MTPB-shaped synthetic tensors; not traffic benchmarks.

The pre-existing source interaction/rotation/fusion algorithms have separate
tests in this repository. This file isolates their orchestration around STRep.
"""
from types import SimpleNamespace
import torch
import pytest
import tsjt_pgrad.mtpb.adapter as bridge


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor([.1,.2,.3]))
    def forward(self,data,adjacency):
        features=data.x[...,:3,0]
        out=(features*self.weight).sum(-1,keepdim=True).expand(-1,-1,36)
        return out,data.y,adjacency


class DummyRep:
    device='cpu'
    def __init__(self):
        self.model=DummyModel()
        self.called=0
    def meta_train_revise(self,*args):
        self.called+=1
        with torch.no_grad():
            self.model.weight.add_(torch.tensor([.1,0,0]))
        return (1.,2.,3.,4.,5.)


class DummyDataset:
    def __init__(self,cities,*,wrong_horizon=False,small=False):
        self.data_list=cities
        self.task_args={'batch_size':2}
        self.x_list={};self.y_list={}
        self.means_list={};self.stds_list={};self.A_list={}
        for k,c in enumerate(cities):
            n=3 if small else 10
            self.x_list[c]=torch.randn(n,4,288,2)+(k*.15)
            self.y_list[c]=torch.ones(n,4,12 if wrong_horizon else 36)*(k+1.)
            self.means_list[c]=[0.]
            self.stds_list[c]=[1.]
            self.A_list[c]=torch.eye(4)


@pytest.fixture
def isolate_gradient_modules(monkeypatch):
    def principal(model,callback,params,source_grads,target_grad,**kwargs):
        ref=tuple(sum(g[i] for g in source_grads)/len(source_grads)
                  for i in range(len(params)))
        return ref,dict(source_weights=[1/len(source_grads)]*len(source_grads),
                        source_target_cosine=[0.]*len(source_grads),
                        pairwise_cosine=[[1.]]*len(source_grads))
    def rotation(source,target,principal,params,controller):
        return source,{'rotated':False}
    def fusion(target,directions,params,**kwargs):
        result=tuple(-sum(g[i] for g in directions)/len(directions)
                     for i in range(len(params)))
        return result,{'weights':[1/len(directions)]*len(directions)}
    monkeypatch.setattr(bridge,'source_interaction_principal_gradient',principal)
    monkeypatch.setattr(bridge,'rotate_source_gradient',rotation)
    monkeypatch.setattr(bridge,'residual_matching_fusion',fusion)


def test_mtpb_contract_and_native_update(isolate_gradient_modules):
    torch.manual_seed(7)
    rep=DummyRep()
    adapter=bridge.MTPBGradientAdapter(rep,bridge.MTPBGradientConfig(
        order_count=1,correction_cap=.1))
    result=adapter.train_step([],[],[],[],DummyDataset([
        'metr-la','chengdu_m','shenzhen']),DummyDataset(['pems-bay']))
    assert result==(1.,2.,3.,4.,5.)
    assert rep.called==1
    diag=adapter.last_diagnostics
    assert (diag['history'],diag['horizon'])==(288,36)
    assert diag['source_cities']==['metr-la','chengdu_m','shenzhen']
    assert set(diag['target_support_indices']).isdisjoint(diag['target_query_indices'])
    assert diag['correction_scale']>0
    native=torch.tensor([.2,.2,.3])
    final=rep.model.weight.detach()
    assert not torch.allclose(native,final)
    assert (final-native).norm()<=.010001


def test_reject_target_city_in_source(isolate_gradient_modules):
    rep=DummyRep()
    adapter=bridge.MTPBGradientAdapter(rep)
    with pytest.raises(ValueError,match='exclude'):
        adapter.train_step([],[],[],[],DummyDataset(['pems-bay','metr-la']),
                           DummyDataset(['pems-bay']))
    assert rep.called==0


def test_reject_wrong_horizon(isolate_gradient_modules):
    rep=DummyRep()
    adapter=bridge.MTPBGradientAdapter(rep)
    with pytest.raises(ValueError,match='36'):
        adapter.train_step([],[],[],[],DummyDataset([
            'metr-la','chengdu_m','shenzhen'],wrong_horizon=True),
            DummyDataset(['pems-bay']))
    assert rep.called==0


def test_disjoint_train_batches_required(isolate_gradient_modules):
    rep=DummyRep()
    adapter=bridge.MTPBGradientAdapter(rep)
    with pytest.raises(ValueError,match='disjoint'):
        adapter.train_step([],[],[],[],DummyDataset([
            'metr-la','chengdu_m','shenzhen']),DummyDataset(['pems-bay'],small=True))
    assert rep.called==0
