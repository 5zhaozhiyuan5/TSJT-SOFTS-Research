"""Synthetic MTPB pattern retrieval / gradient-alignment smoke, not MAE/RMSE."""
import torch
import pytest
from torch import nn
from tsjt_pgrad.mtpb.retrieval_confidence import retrieval_confidence, capture_retrieval_confidence

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hour_list=['1','3','6','12','24']
        self.Qnet_dict=nn.ModuleDict({h:nn.Linear(12,2,bias=False) for h in self.hour_list})
        self.qry_mat_dict=nn.ParameterDict({
            h:nn.Parameter(torch.zeros(2*int(h),3)) for h in self.hour_list})
    def forward(self,x):
        return x.mean()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_list=nn.ModuleList([Encoder()])
        self.w=nn.Parameter(torch.ones(1))
    def forward(self,x):
        return self.model_list[0](x)*self.w

def test_uniform_bank_is_zero_confidence_and_hook_is_removed():
    m=Model()
    with capture_retrieval_confidence(m) as record:
        loss=m(torch.randn(2,3,24,12))
        loss.backward()
    assert len(record)==1
    confidence, scales=record[0]
    assert confidence==pytest.approx(0,abs=1e-6)
    assert set(scales)==set(m.model_list[0].hour_list)
    assert len(m.model_list[0]._forward_pre_hooks)==0
    assert m.w.grad is not None

def test_concentrated_retrieval_has_high_confidence():
    m=Model()
    with torch.no_grad():
        for q in m.model_list[0].Qnet_dict.values():
            q.weight.zero_()
            q.weight[0,0]=1
        for mat in m.model_list[0].qry_mat_dict.values():
            mat[:,0]=1
            mat[:,1:]=-1
    c,_=retrieval_confidence(m.model_list[0],torch.ones(1,2,24,12)*20)
    assert .9<c<=1

def test_invalid_scale_and_invalid_model_fail_closed():
    m=Model()
    with pytest.raises(ValueError,match='does not divide'):
        retrieval_confidence(m.model_list[0],torch.ones(1,2,23,12))
    with pytest.raises(ValueError,match='model_list'):
        with capture_retrieval_confidence(nn.Linear(2,2)):
            pass

def test_guidance_strength_ablation_validation():
    from tsjt_pgrad.mtpb.integrated_meta import IntegratedMetaConfig,TargetGuidedMTPBMetaUpdate
    class Rep:
        device='cpu'
    for invalid in (-.1,1.1):
        with pytest.raises(ValueError,match='retrieval_guidance_strength'):
            TargetGuidedMTPBMetaUpdate(Rep(),IntegratedMetaConfig(
                retrieval_guidance_strength=invalid))
    assert IntegratedMetaConfig().retrieval_guidance_strength==0.
