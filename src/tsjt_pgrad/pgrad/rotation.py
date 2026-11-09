"""Target-orthogonal rotation towards a PGrad-derived multi-city direction.

The previous implementation rotated into the *target-parallel source component*,
which reintroduced negative transfer. Here both rotation axes lie in g_T^perp.
"""
from __future__ import annotations
from collections.abc import Sequence
import math
import torch
from torch import nn
from .grad_utils import flatten_gradients, unflatten_gradients

class AngleController(nn.Module):
    def __init__(self, input_dim: int=3, hidden_dim: int=16, phi_max: float=math.pi/4):
        super().__init__()
        self.phi_max=float(phi_max)
        self.mlp=nn.Sequential(nn.Linear(input_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,1))
        # ~2 degrees at initialization, not the ~22.5 degrees of sigmoid(0).
        with torch.no_grad():
            nn.init.normal_(self.mlp[-1].weight, mean=0, std=0.005)
            nn.init.constant_(self.mlp[-1].bias,-3.0)
    def forward(self, stats: torch.Tensor)->torch.Tensor:
        return self.phi_max*torch.sigmoid(self.mlp(stats))

def _cos(x: torch.Tensor,y: torch.Tensor, eps=1e-12)->torch.Tensor:
    return (x*y).sum(dim=-1)/(x.norm(dim=-1)*y.norm(dim=-1)).clamp_min(eps)

def angle_statistics(source_gradients, target_gradients, principal_gradients, parameters)->torch.Tensor:
    s=flatten_gradients(parameters,source_gradients)
    t=flatten_gradients(parameters,target_gradients)
    p=flatten_gradients(parameters,principal_gradients)
    return torch.stack([_cos(s,t),_cos(s,p),_cos(p,t)],dim=-1).detach()

def orthogonal_component(x:torch.Tensor, ref:torch.Tensor,eps:float=1e-12)->torch.Tensor:
    norm2=(ref*ref).sum(dim=-1,keepdim=True)
    if bool((norm2 <= eps).any()):
        return x.clone()
    return x - (x*ref).sum(dim=-1,keepdim=True)/norm2*ref

def decompose_parallel_orthogonal(source_gradients,target_gradients,parameters,*,eps:float=1e-12):
    s=flatten_gradients(parameters,source_gradients); t=flatten_gradients(parameters,target_gradients)
    ortho=orthogonal_component(s,t,eps)
    return unflatten_gradients(parameters,ortho),unflatten_gradients(parameters,s-ortho)

def rotate_source_gradient(
    source_gradients:Sequence[torch.Tensor|None],
    target_gradients:Sequence[torch.Tensor|None],
    principal_gradients:Sequence[torch.Tensor|None],
    parameters:Sequence[nn.Parameter],
    angle_controller:AngleController|None,
    *, fixed_angle:float|None=None, eps:float=1e-12,
    return_phi_tensor:bool=False,
):
    """Return target-orthogonal direction, metrics, optionally differentiable phi.

    Non-conflicting source directions are NOT rotated but their target-parallel
    part is removed for the orthogonal-complement fusion design.
    """
    s=flatten_gradients(parameters,source_gradients)
    t=flatten_gradients(parameters,target_gradients)
    p=flatten_gradients(parameters,principal_gradients)
    ortho=orthogonal_component(s,t,eps)
    principal_ortho=orthogonal_component(p,t,eps)
    # Remove source's own orthogonal direction from the PGrad reference.
    reference=orthogonal_component(principal_ortho,ortho,eps)
    o_norm=ortho.norm(dim=-1,keepdim=True)
    q_norm=reference.norm(dim=-1,keepdim=True)
    s_t_dot=(s*t).sum(dim=-1,keepdim=True)
    conflict=bool((s_t_dot<0).item())
    relative_floor=1e-6 * principal_ortho.norm(dim=-1,keepdim=True)
    valid=conflict and bool((o_norm>eps).item()) and bool((q_norm>torch.maximum(relative_floor,q_norm.new_tensor(eps))).item()) and bool((t.norm()>eps).item())
    if fixed_angle is None:
        if angle_controller is None:
            raise ValueError('angle_controller required for learned rotation')
        phi=angle_controller(angle_statistics(source_gradients,target_gradients,principal_gradients,parameters))
    else:
        phi=s.new_tensor([[fixed_angle]])
    if valid:
        u=ortho/o_norm.clamp_min(eps)
        v=reference/q_norm.clamp_min(eps)
        out=o_norm*(torch.cos(phi)*u+torch.sin(phi)*v)
    else:
        out=ortho
    meta={
        'phi':float(phi.detach().item()), 'rotated':valid,
        'conflict':conflict, 'orthogonal_norm':float(o_norm.detach().item()),
        'reference_norm':float(q_norm.detach().item()),
        'rotated_norm':float(out.detach().norm()),
        'target_dot':float((out.detach()*t.detach()).sum()),
    }
    result=(unflatten_gradients(parameters,out),meta)
    return (*result,phi) if return_phi_tensor else result
