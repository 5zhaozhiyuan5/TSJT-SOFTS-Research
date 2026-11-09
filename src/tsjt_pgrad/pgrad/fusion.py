"""Target-gradient residual matching on a nonnegative l1 unit ball."""
from __future__ import annotations
from collections.abc import Sequence
import torch
from .grad_utils import flatten_gradients,unflatten_gradients

def target_residual(target_query_gradients,target_gradients,parameters,*,eps=1e-12):
    q=flatten_gradients(parameters,target_query_gradients)
    t=flatten_gradients(parameters,target_gradients)
    denom=(t*t).sum(dim=1,keepdim=True)
    if (denom<=eps).item():
        return q
    return q - (q*t).sum(dim=1,keepdim=True)/denom*t

@torch.no_grad()
def project_nonneg_l1_ball(v:torch.Tensor, radius:float=1.)->torch.Tensor:
    """Euclidean projection onto {w >= 0, sum(w) <= radius}."""
    positive=v.clamp_min(0)
    if positive.sum() <= radius:
        return positive
    sorted_v=torch.sort(v,descending=True).values
    cssv=torch.cumsum(sorted_v,dim=0)-radius
    ranks=torch.arange(1,v.numel()+1,device=v.device,dtype=v.dtype)
    flags=sorted_v-cssv/ranks>0
    rho=torch.nonzero(flags,as_tuple=False)[-1,0]
    theta=cssv[rho]/(rho+1).to(v.dtype)
    return (v-theta).clamp_min(0)

@torch.no_grad()
def _solve_nonneg_weights(direction_matrix:torch.Tensor,residual:torch.Tensor,*,lam:float,mu_w:float,iterations:int=300)->torch.Tensor:
    """Projected gradient with exact Euclidean l1-ball projection; joint convex QP."""
    if lam<=0 or mu_w<=0:
        raise ValueError('lam and mu_w must be >0')
    d=direction_matrix
    r=residual.reshape(-1)
    if d.ndim !=2 or d.shape[1]!=r.numel():
        raise ValueError('direction shape mismatch')
    if not bool(torch.isfinite(d).all() and torch.isfinite(r).all()):
        raise FloatingPointError('nonfinite fusion inputs')
    if d.numel()==0:
        return r.new_zeros((d.shape[0],))
    # Normalize both residual and candidates by the *same* scale so mu_w
    # has stable meaning regardless of gradient size.
    scale=torch.stack([r.norm(),d.norm(dim=1).max()]).max().clamp_min(1e-12)
    a=lam*d/scale
    b=r/scale
    gram=a@a.T+mu_w*torch.eye(d.shape[0],device=d.device,dtype=d.dtype)
    c=a@b
    # Lipschitz constant of grad (Gram normalization removes factor 2).
    step=1./torch.linalg.eigvalsh(gram).max().clamp_min(1e-9)
    weights=torch.zeros_like(c)
    for _ in range(iterations):
        next_w=project_nonneg_l1_ball(weights-step*(gram@weights-c))
        if torch.max((next_w-weights).abs()) < 1e-9:
            weights=next_w
            break
        weights=next_w
    return weights

def residual_matching_fusion(
    target_gradients,
    source_directions,
    parameters,
    *, target_query_gradients=None,
    lam:float=1.0,mu_w:float=1e-3,
    return_source_only:bool=False,
):
    """Returns source contribution only when return_source_only=True.

    Crucial: the trainer already adds gamma_target * target_grad. Never embed
    the target gradient in a 'source' term and accidentally update it twice.
    """
    if target_query_gradients is None:
        raise ValueError('Query-W gradient is required: using g_Q=g_T forces zero residual')
    directions=torch.cat([flatten_gradients(parameters,g) for g in source_directions],dim=0)
    target=flatten_gradients(parameters,target_gradients)
    r=target_residual(target_query_gradients,target_gradients,parameters)
    weights=_solve_nonneg_weights(directions,r,lam=lam,mu_w=mu_w)
    source=lam*(weights.view(1,-1)@directions)
    result=source if return_source_only else source+target
    metrics={
        'weights':[float(w) for w in weights], 'weight_sum':float(weights.sum()),
        'all_zero':bool((weights.abs().sum()<1e-10).item()),
        'residual_norm':float(r.detach().norm()),'source_norm':float(source.detach().norm()),
        'match_error':float((r-source).detach().norm()),
    }
    return unflatten_gradients(parameters,result),metrics
