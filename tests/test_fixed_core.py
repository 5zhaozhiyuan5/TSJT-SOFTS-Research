"""Correctness tests for real sequential PGrad, rotation, Query meta, fusion and 8 modes.

Test stubs emulate the unavailable TSJT backbone, *not* its real prediction model.
"""
import math
import torch
import pytest
from torch import nn
from conftest import ToyModel
from tsjt_pgrad.pgrad.grad_utils import flatten_gradients,unflatten_gradients,principal_directions_gram
from tsjt_pgrad.pgrad.pgrad import sequential_virtual_trajectory,pgrad_principal_gradient
from tsjt_pgrad.pgrad.rotation import AngleController,rotate_source_gradient
from tsjt_pgrad.pgrad.fusion import residual_matching_fusion,project_nonneg_l1_ball
from tsjt_pgrad.training.meta import virtual_parameter_update,angle_meta_loss,apply_meta_update
from tsjt_pgrad.training import trainer


def toy_grads(tensors):
    m=nn.Module()
    m.weight=nn.Parameter(torch.zeros_like(tensors))
    return [m.weight]


def test_01_roundtrip():
    params=toy_grads(torch.zeros(3))
    grads=(torch.tensor([1.,2.,3.]),)
    assert torch.allclose(unflatten_gradients(params,flatten_gradients(params,grads))[0],grads[0])


def test_02_length_mismatch_rejected():
    with pytest.raises(ValueError):flatten_gradients(toy_grads(torch.zeros(2)),[])


def test_03_pca_translation_invariant():
    sequence=[torch.tensor([0.,0.,0.]),torch.tensor([1.,0.,0.]),torch.tensor([1.,2.,0.])]
    d1,e1=principal_directions_gram(sequence,2)
    shift=torch.tensor([1e4,2e4,3e4])
    d2,e2=principal_directions_gram([x+shift for x in sequence],2)
    assert torch.allclose(e1,e2,atol=1e-5)
    assert torch.allclose(d1.abs(),d2.abs(),atol=1e-5)


def test_04_pca_zero_rank():
    direction,eigs=principal_directions_gram([torch.zeros(3)]*4,2)
    assert direction.shape==(0,3)


def test_05_real_sequential_order_matters():
    class Quad(nn.Module):
        def __init__(self):
            super().__init__();self.w=nn.Parameter(torch.tensor([1.,-1.]))
    model=Quad()
    # Hessians different per source => true sequential updates depend on order.
    centers=[torch.tensor([3.,0.]),torch.tensor([-1.,4.]),torch.tensor([0.,-2.])]
    scales=[torch.tensor([1.,2.]),torch.tensor([3.,1.]),torch.tensor([2.,3.])]
    calls=[]
    def cb(i,params):
        w=params['w']; calls.append((i,w.detach().clone()))
        loss=((w-centers[i]).square()*scales[i]).sum()/2
        return torch.autograd.grad(loss,(w,))
    t1,_=sequential_virtual_trajectory(model,cb,virtual_lr=.1,num_cities=3,city_order=[0,1,2])
    t2,_=sequential_virtual_trajectory(model,cb,virtual_lr=.1,num_cities=3,city_order=[2,1,0])
    assert not torch.allclose(t1[-1],t2[-1])
    assert calls[1][0]==1 and not torch.allclose(calls[1][1],model.w.detach())


def test_06_pgrad_preserves_original_parameters():
    m=ToyModel();before=m.backbone.weight.detach().clone()
    centers=[torch.tensor([2.,1.,0.]),torch.tensor([0.,1.,2.]),torch.tensor([1.,0.,2.])]
    def cb(i,params):
        w=params['weight'];return torch.autograd.grad(((w-centers[i])**2).sum()/2,(w,))
    out,meta=pgrad_principal_gradient(m.backbone,cb,virtual_lr=.1,num_cities=3)
    assert torch.equal(before,m.backbone.weight.detach())
    assert meta['trajectory_len']==4
    assert meta['num_components']<=2
    assert torch.isfinite(out[0]).all()


def test_07_pgrad_gradient_sign_correct():
    m=ToyModel()
    centers=[torch.tensor([2.,1.,0.]),torch.tensor([2.,0.,1.]),torch.tensor([2.,1.,1.])]
    def cb(i,params):
        w=params['weight'];return torch.autograd.grad(((w-centers[i])**2).sum()/2,(w,))
    out,meta=pgrad_principal_gradient(m.backbone,cb,virtual_lr=.01,num_cities=3,num_components=1)
    mean_source=sum([-v for v in centers])/3
    assert (out[0]*mean_source).sum()>0


def test_08_rotation_no_target_conflict():
    params=toy_grads(torch.zeros(3));model=AngleController()
    s=(torch.tensor([-1.,3.,0.]),);t=(torch.tensor([1.,0.,0.]),)
    p=(torch.tensor([0.,1.,1.]),)
    result,meta=rotate_source_gradient(s,t,p,params,model,fixed_angle=math.pi/8)
    assert abs(float((result[0]*t[0]).sum()))<1e-6
    assert meta['rotated']
    assert torch.allclose(result[0].norm(),torch.tensor(3.),atol=1e-5)
    assert result[0][2]>0


def test_09_rotation_zero_recovers_tsb():
    params=toy_grads(torch.zeros(3))
    v,_=rotate_source_gradient((torch.tensor([-1.,3.,0.]),),(torch.tensor([1.,0.,0.]),),
        (torch.tensor([0.,1.,1.]),),params,None,fixed_angle=0)
    assert torch.allclose(v[0],torch.tensor([0.,3.,0.]))


def test_10_rotation_fallback_no_secondary_direction():
    params=toy_grads(torch.zeros(3))
    result,meta=rotate_source_gradient((torch.tensor([-1.,3.,0.]),),(torch.tensor([1.,0.,0.]),),
        (torch.tensor([0.,9.,0.]),),params,None,fixed_angle=math.pi/4)
    assert torch.allclose(result[0],torch.tensor([0.,3.,0.]),atol=1e-6)
    assert not meta['rotated']


def test_11_angle_init_near_zero():
    torch.manual_seed(0)
    c=AngleController();phi=c(torch.zeros(1,3))
    assert 0<=phi.item()<.07


def test_12_true_query_gradient_reaches_controller_without_regularizer():
    m=ToyModel();ps=list(m.backbone.parameters())
    src=(torch.tensor([-1.,3.,0.]),);tg=(torch.tensor([1.,0.,0.]),)
    principal=(torch.tensor([0.,1.,1.]),)
    direction,metadata,phi=rotate_source_gradient(src,tg,principal,ps,m.angle_controller,return_phi_tensor=True)
    assert metadata['rotated']
    # Predict on a feature aligned with the new q direction, not with g_T.
    joint=tuple(.2*t+.5*s for t,s in zip(tg,direction))
    updated=virtual_parameter_update(m,ps,joint,step_scale=1.)
    qx=torch.tensor([[0.,0.,1.],[0.,0.,2.]])
    qy=torch.tensor([[1.],[1.]])
    loss,qloss,_=angle_meta_loss(m,updated,qx,qy,torch.eye(1),phi_tensors=[phi],mu_phi=0.)
    grad=torch.autograd.grad(qloss,list(m.angle_controller.parameters()),retain_graph=True,allow_unused=True)
    assert sum(float(g.abs().sum()) for g in grad if g is not None)>1e-6
    result=apply_meta_update(m.angle_controller,loss,qloss,learning_rate=.01)
    assert result['meta_updated']
    assert result['param_delta']>0


def test_13_meta_rejects_regularizer_only_training():
    m=ToyModel();ps=list(m.backbone.parameters())
    g=(torch.ones(3),)
    updated=virtual_parameter_update(m,ps,g,step_scale=.1)
    phi=m.angle_controller(torch.zeros(1,3))
    loss,qloss,_=angle_meta_loss(m,updated,torch.ones(1,3),torch.ones(1,1),torch.eye(1),phi_tensors=[phi],mu_phi=1.)
    result=apply_meta_update(m.angle_controller,loss,qloss,learning_rate=1.)
    assert not result['meta_updated']
    assert result['param_delta']==0


def test_14_fusion_nonzero_reference_yields_nonzero_weights():
    ps=toy_grads(torch.zeros(3));t=(torch.tensor([1.,0.,0.]),)
    q=(torch.tensor([1.,0.,2.]),)
    dirs=[(torch.tensor([0.,0.,1.]),),(torch.tensor([0.,1.,0.]),)]
    source,info=residual_matching_fusion(t,dirs,ps,target_query_gradients=q,return_source_only=True)
    assert info['weights'][0]>.5
    assert info['weights'][1]<1e-3
    assert source[0][0]==0


def test_15_fusion_zero_query_forces_zero_weights():
    ps=toy_grads(torch.zeros(3));t=(torch.tensor([1.,0.,0.]),)
    direction=(torch.tensor([0.,1.,0.]),)
    src,info=residual_matching_fusion(t,[direction],ps,target_query_gradients=t,return_source_only=True)
    assert info['all_zero']
    assert torch.equal(src[0],torch.zeros(3))


def test_16_fusion_requires_query():
    ps=toy_grads(torch.zeros(3));t=(torch.tensor([1.,0.,0.]),)
    with pytest.raises(ValueError):residual_matching_fusion(t,[t],ps,target_query_gradients=None)


def test_17_exact_simplex_projection():
    w=project_nonneg_l1_ball(torch.tensor([3.,2.,-1.]))
    assert torch.allclose(w,torch.tensor([1.,0.,0.]))
    assert abs(float(w.sum())-1.)<1e-7


def test_18_query_sampler_disjoint_labels():
    class DS:
        indices=list(range(170))
        def __len__(self):return len(self.indices)
        def __getitem__(self,i):return torch.tensor([float(i)]),torch.tensor([float(i)])
    sampler=trainer.TrainOnlyQuerySampler(DS(),history=8,horizon=6)
    r,w,info=sampler.sample([3,30],query_r_size=2,query_w_size=2,seed=12)
    selected=info['support_ids']+info['query_r_ids']+info['query_w_ids']
    intervals=[sampler.intervals[i] for i in selected]
    for i in range(len(intervals)):
        for j in range(i+1,len(intervals)):
            assert intervals[i][1]<=intervals[j][0] or intervals[j][1]<=intervals[i][0]
    assert r[0].shape[0]==2 and w[0].shape[0]==2


def _inputs():
    target_x=torch.tensor([[1.,0.,0.],[1.,0.,0.]])
    target_y=torch.ones(2,1)
    # Conflict with target, and have mutually different orthogonal components.
    source_chunks=[
        (torch.tensor([[1.,1.,0.],[1.,1.,0.]]),-torch.ones(2,1),torch.eye(1)),
        (torch.tensor([[1.,0.,1.],[1.,0.,1.]]),-torch.ones(2,1),torch.eye(1)),
        (torch.tensor([[1.,1.,1.],[1.,1.,1.]]),-torch.ones(2,1),torch.eye(1)),
    ]
    return target_x,target_y,source_chunks


def _config(method):
    return {'train':{'gradient_method':method,'gamma_target':.02,'gamma_source':.01,
        'pgrad_virtual_lr':.01,'pgrad_num_components':2,'meta_interval':1,
        'meta_step_scale':5.,'meta_lr':.001,'fusion_interval':1,
        'fusion_lambda':1.,'fusion_mu_w':1e-3,'source_budget_ratio':0.}}


def test_19_all_eight_modes_execute():
    for method in sorted(trainer.GRADIENT_METHODS):
        model=ToyModel();params=list(model.backbone.parameters())
        x,y,chunks=_inputs()
        r=(torch.tensor([[0.,0.,1.]]),torch.tensor([[1.]]))
        w=(torch.tensor([[0.,1.,0.]]),torch.tensor([[1.]]))
        diag=trainer._train_step(model,params,x,y,torch.eye(1),chunks,
            _config(method),torch.device('cpu'),0,state={},query_r=r,query_w=w)
        assert math.isfinite(diag['effective_source_ratio'])
        assert len(diag['path'])>0
        assert torch.isfinite(model.backbone.weight).all()


def test_20_full_method_calls_all_three_modules(monkeypatch):
    model=ToyModel();params=list(model.backbone.parameters())
    x,y,chunks=_inputs()
    r=(torch.tensor([[0.,0.,1.]]),torch.tensor([[1.]]))
    w=(torch.tensor([[0.,1.,0.]]),torch.tensor([[1.]]))
    diag=trainer._train_step(model,params,x,y,torch.eye(1),chunks,_config('full_method'),
        torch.device('cpu'),0,state={},query_r=r,query_w=w)
    assert diag['path']==['per_city_gradient','pgrad','rotation','residual_matching']
    assert diag['num_source_directions']==3
    assert 'fusion' in diag
    assert len(diag['fusion']['weights'])==3
    assert 'pgrad' in diag


def test_21_rotation_only_does_not_use_fusion():
    model=ToyModel();params=list(model.backbone.parameters())
    x,y,chunks=_inputs()
    r=(torch.tensor([[0.,0.,1.]]),torch.tensor([[1.]]))
    diag=trainer._train_step(model,params,x,y,torch.eye(1),chunks,
        _config('pgrad_rotation_only'),torch.device('cpu'),0,state={},query_r=r)
    assert 'residual_matching' not in diag['path']
    assert 'fusion' not in diag


def test_22_meta_uses_independent_qr_not_support():
    model=ToyModel();params=list(model.backbone.parameters())
    x,y,chunks=_inputs()
    with pytest.raises(ValueError,match='Query-R'):
        trainer._train_step(model,params,x,y,torch.eye(1),chunks,
            _config('pgrad_rotation_only'),torch.device('cpu'),0,state={})


def test_23_fusion_uses_independent_qw_not_target():
    model=ToyModel();params=list(model.backbone.parameters())
    x,y,chunks=_inputs()
    with pytest.raises(ValueError,match='Query-W'):
        trainer._train_step(model,params,x,y,torch.eye(1),chunks,
            _config('residual_only'),torch.device('cpu'),0,state={})


def test_24_full_method_three_step_smoke():
    model=ToyModel();ps=list(model.backbone.parameters())
    state={};x,y,chunks=_inputs()
    r=(torch.tensor([[0.,0.,1.],[0.,0.,2.]]),torch.ones(2,1))
    w=(torch.tensor([[0.,1.,0.],[0.,2.,0.]]),torch.ones(2,1))
    before=model.backbone.weight.detach().clone()
    for step in range(3):
        diag=trainer._train_step(model,ps,x,y,torch.eye(1),chunks,_config('full_method'),
            torch.device('cpu'),step,state=state,query_r=r,query_w=w)
        assert 'pgrad' in diag and 'fusion' in diag
        assert len(diag['fusion']['weights'])==3
        assert torch.isfinite(model.backbone.weight).all()
    assert not torch.equal(before,model.backbone.weight.detach())


def test_25_source_budget_cap():
    ps=toy_grads(torch.zeros(3))
    t=(torch.tensor([1.,0.,0.]),)
    src=(torch.tensor([0.,1000.,0.]),)
    cap,ratio=trainer._effective_source_cap(ps,t,src,gamma_t=.1,gamma_s=.1,budget_ratio=.3)
    assert abs(ratio*cap-.3)<1e-5


# Regression checks added after a second audit of the handoff package.
def test_26_completed_smoke_cannot_silently_reuse_when_skip_complete_false(tmp_path):
    (tmp_path/'complete.json').write_text('{}')
    conf=_config('full_method')
    conf['run']={'skip_complete':False,'resume':False}
    with pytest.raises(FileExistsError,match='already marked complete'):
        trainer._run_all_pgrad(conf,tmp_path,'hash',torch.device('cpu'),None)


def test_27_skip_complete_true_requires_matching_checkpoint_and_metrics(tmp_path):
    (tmp_path/'complete.json').write_text('{}')
    (tmp_path/'final_metrics.json').write_text('{}')
    torch.save({'config_hash':'hash'},tmp_path/'last.pt')
    conf=_config('full_method')
    conf['run']={'skip_complete':True,'resume':False}
    result=trainer._run_all_pgrad(conf,tmp_path,'hash',torch.device('cpu'),None)
    assert result==str(tmp_path/'final_metrics.json')


def test_28_partial_checkpoint_no_resume_rejected(tmp_path):
    torch.save({'config_hash':'hash'},tmp_path/'last.pt')
    conf=_config('full_method')
    conf['run']={'skip_complete':False,'resume':False}
    with pytest.raises(FileExistsError,match='resume=false'):
        trainer._run_all_pgrad(conf,tmp_path,'hash',torch.device('cpu'),None)


def test_29_mismatched_epoch_config_rejected():
    conf=_config('full_method')
    conf['train'].update({'epochs':1,'max_epochs':50})
    with pytest.raises(ValueError,match='max_epochs'):
        trainer._validate_pgrad_config(conf)


def test_30_smoke_yaml_epoch_evaluation_consistent():
    import yaml
    from pathlib import Path
    conf=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs'/'pgrad_pemsbay_smoke.yaml').read_text())
    assert conf['train']['epochs']==conf['train']['max_epochs']==conf['evaluation']['evaluate_once_after_epoch']==1
    assert conf['evaluation']['skip_final_evaluation'] is True



def test_31_meta_update_reports_actual_not_theoretical_delta():
    ctrl=AngleController()
    with torch.no_grad():
        for p in ctrl.parameters():
            p.fill_(1.)
    parameters=list(ctrl.parameters())
    before=[p.detach().clone() for p in parameters]
    # Positive autograd gradient, but fp32 cannot represent a 1e-10 change around 1.0.
    query_loss=1e-7*sum(p.sum() for p in parameters)
    info=apply_meta_update(ctrl,query_loss,query_loss,learning_rate=1e-3)
    assert info['query_grad_norm']>0
    assert info['proposed_param_delta']>0
    assert info['param_delta']==0
    assert not info['meta_updated']
    for p,b in zip(parameters,before):
        assert torch.equal(p,b)


def test_32_realized_source_effect_can_round_to_zero():
    params=toy_grads(torch.ones(3))
    with torch.no_grad():
        params[0].fill_(1.)
    # A source step of 1e-10 on fp32 weights at 1.0 is numerically invisible.
    old=params[0].detach().clone()
    target=torch.zeros(3)
    source=torch.ones(3)
    with torch.no_grad():
        target_only=old-.0002*target
        params[0].add_(.0002*target+1e-10*source,alpha=-1.)
    assert torch.equal(params[0],old)
    assert (params[0]-target_only).norm().item()==0
