"""PGrad-inspired sequential virtual adaptation across source cities.

This deliberately re-evaluates each city's loss at *updated virtual parameters*.
A list of precomputed fixed gradients would NOT implement PGrad interaction.
"""
from __future__ import annotations
import random
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
import torch
from .grad_utils import principal_directions_gram

GradientCallback = Callable[[int, Mapping[str, torch.Tensor]], Sequence[torch.Tensor | None]]

def sequential_virtual_trajectory(
    backbone: torch.nn.Module,
    gradient_callback: GradientCallback,
    *, virtual_lr: float,
    num_cities: int,
    city_order: Sequence[int] | None = None,
    rounds: int = 1,
) -> tuple[list[torch.Tensor], list[int]]:
    """Return displacement snapshots relative to the original backbone parameters.

    The callback must compute gradients via functional_call on the given virtual
    parameter dict; each subsequent city sees the previous city's updated weights.
    """
    if num_cities < 1 or rounds < 1 or virtual_lr <= 0:
        raise ValueError("invalid PGrad trajectory configuration")
    base = OrderedDict((name, p.detach().clone()) for name,p in backbone.named_parameters() if p.requires_grad)
    current = OrderedDict((name, value.detach().clone().requires_grad_(True)) for name,value in base.items())
    flat_base = torch.cat([p.reshape(-1) for p in base.values()])
    trajectories = [torch.zeros_like(flat_base)]
    executed = []
    for rnd in range(rounds):
        order = list(city_order) if city_order is not None else random.sample(range(num_cities), num_cities)
        if sorted(order) != list(range(num_cities)):
            raise ValueError("city_order must be a permutation")
        for city in order:
            grads = list(gradient_callback(city, current))
            if len(grads) != len(current):
                raise ValueError("callback must provide gradients for every named backbone parameter")
            current = OrderedDict((name, (value - virtual_lr*(grad.detach() if grad is not None else torch.zeros_like(value))).detach().requires_grad_(True))
                                  for (name,value),grad in zip(current.items(),grads))
            trajectories.append(torch.cat([(v.detach()-base[n]).reshape(-1) for n,v in current.items()]))
            executed.append(city)
    return trajectories, executed

def pgrad_principal_gradient(
    backbone: torch.nn.Module,
    gradient_callback: GradientCallback,
    *, virtual_lr: float,
    num_cities: int,
    num_components: int=2,
    city_order: Sequence[int] | None=None,
    rounds: int=1,
) -> tuple[tuple[torch.Tensor, ...], dict]:
    traj, executed = sequential_virtual_trajectory(backbone, gradient_callback,
        virtual_lr=virtual_lr, num_cities=num_cities, city_order=city_order, rounds=rounds)
    delta_descent = -traj[-1] # current theta - last theta; true descent gradient sign
    directions, eigvals = principal_directions_gram(traj, num_components=num_components)
    delta_norm = delta_descent.norm()
    if directions.shape[0] == 0 or delta_norm <= 1e-12:
        grad = delta_descent
        fallback = True
    else:
        # Use singular-value weights; anchor every principal axis to the descent gradient.
        signs = torch.where(directions @ delta_descent >= 0, 1.0, -1.0)
        scores = eigvals.clamp_min(0).sqrt()
        weights = scores / scores.sum().clamp_min(1e-12)
        mix = (signs[:,None]*directions*weights[:,None]).sum(dim=0)
        grad = mix / mix.norm().clamp_min(1e-12) * delta_norm
        fallback = False
    # Source virtual_lr is an arbitrary numerical trajectory step length. Calibrate
    # gradient magnitude back to gradient scale, rather than carrying an extra 1e-3.
    grad = grad / virtual_lr
    parts=[]; off=0
    for p in (p for p in backbone.parameters() if p.requires_grad):
        parts.append(grad[off:off+p.numel()].reshape_as(p))
        off+=p.numel()
    if not all(torch.isfinite(x).all() for x in parts):
        raise FloatingPointError("PGrad returned nonfinite direction")
    metadata = {
        'num_cities':num_cities, 'trajectory_len':len(traj), 'city_order':executed,
        'delta_norm':float(delta_norm), 'principal_norm':float(grad.norm()),
        'num_components':int(directions.shape[0]),
        'eigenvalues':[float(v) for v in eigvals], 'pgrad_fallback':fallback,
        'virtual_lr':virtual_lr,
    }
    return tuple(parts), metadata
