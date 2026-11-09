"""Differentiable one-step target-Query feedback for gradient rotation angles."""
from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn.functional as F
from torch.func import functional_call


def virtual_parameter_update(model,parameters:Sequence[torch.nn.Parameter],gradients,*,step_scale:float=1.)->dict[str,torch.Tensor]:
    """θ' = θ - step_scale*gradient, preserving d(θ')/dφ.

    Gradients here MUST be the same gamma_t*g_T + gamma_s*source_combo
    used by the real training step. step_scale is a virtual step multiplier.
    """
    named=[(n,p) for n,p in model.backbone.named_parameters() if p.requires_grad]
    if len(parameters)!=len(named):
        raise ValueError('virtual update must use only backbone parameters')
    if len(gradients)!=len(parameters):
        raise ValueError('gradient length mismatch')
    if any(id(a)!=id(b) for a,(_,b) in zip(parameters,named)):
        raise ValueError('parameter ordering mismatch')
    return {name:p.detach()-(step_scale*g if g is not None else 0.) for (name,p),g in zip(named,gradients)}

def angle_meta_loss(model,updated_params,query_x,query_y,adjacency,*,phi_tensors,mu_phi:float):
    """Return total and its two terms; no dummy-input angle regularizer."""
    if not phi_tensors:
        raise ValueError('no actual rotation angles supplied')
    buffer_copies={k:v.detach().clone() for k,v in model.backbone.named_buffers()}
    pred=functional_call(model.backbone, (updated_params,buffer_copies),
                         (query_x,adjacency),{'horizon':query_y.shape[1]})
    if isinstance(pred,dict):
        pred=pred['prediction']
    query_loss=F.l1_loss(pred,query_y)
    penalty=torch.stack([phi.square().mean() for phi in phi_tensors]).mean()
    return query_loss+mu_phi*penalty,query_loss,penalty

def apply_meta_update(controller,loss,query_loss,*,learning_rate:float,eps:float=1e-15)->dict:
    """Refuse to pretend an angle update from regularizer alone is target feedback."""
    params=list(controller.parameters())
    q_grads=(torch.autograd.grad(query_loss,params,allow_unused=True,retain_graph=True)
             if query_loss.requires_grad else [None]*len(params))
    query_grad_norm=sum(float(g.detach().square().sum()) for g in q_grads if g is not None)**.5
    all_grads=torch.autograd.grad(loss,params,allow_unused=True)
    total_grad_norm=sum(float(g.detach().square().sum()) for g in all_grads if g is not None)**.5
    if not torch.isfinite(loss) or query_grad_norm<=eps:
        return {'query_grad_norm':query_grad_norm,'total_grad_norm':total_grad_norm,
                'param_delta':0.0,'meta_updated':False}
    # In fp32, a proposed update of ~1e-10 on a weight of magnitude ~1e-1
    # may round to zero. Report the *realized* change, not lr*grad_norm.
    with torch.no_grad():
        before=[p.detach().clone() for p in params]
        for p,g in zip(params,all_grads):
            if g is not None:
                p.add_(g,alpha=-learning_rate)
        realized_sq=sum(float((p.detach()-b).double().square().sum())
                        for p,b in zip(params,before))
    actual_delta=realized_sq**0.5
    updated=actual_delta>0
    return {'query_grad_norm':query_grad_norm,'total_grad_norm':total_grad_norm,
            'proposed_param_delta':float(learning_rate*total_grad_norm),
            'param_delta':actual_delta,'meta_updated':updated,
            'reason':None if updated else 'update_rounded_to_zero_in_parameter_dtype'}
