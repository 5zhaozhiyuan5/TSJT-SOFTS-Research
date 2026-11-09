"""Source-city interaction: order-balanced PGrad + target-aware Gram collaboration.

This module ONLY produces the common principal gradient passed to the existing
rotation and residual fusion modules. It never updates real parameters and does
not alter the per-source gradients supplied to the existing rotation.

References/inspiration (not claimed as identical implementations):
    PGrad, ICLR 2023: sequential virtual optimization trajectory
    PiKE, NeurIPS 2025: positive multi-task gradient interaction
    Aligned Multi Objective Optimization, ICML 2025: aligned objective progress
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from .grad_utils import flatten_gradients, unflatten_gradients
from .pgrad import pgrad_principal_gradient


def balanced_city_orders(n: int, count: int) -> list[tuple[int, ...]]:
    """Deterministic cyclic orders first; reverse orders follow when requested."""
    if n < 1 or count < 1:
        raise ValueError('number of cities/orders must be positive')
    base = tuple(range(n))
    cyclic = [base[k:] + base[:k] for k in range(n)]
    if count <= n:
        return cyclic[:count]
    reverse = tuple(reversed(base))
    reverse_cyclic = [reverse[k:] + reverse[:k] for k in range(n)]
    orders = list(dict.fromkeys(cyclic + reverse_cyclic))
    if count > len(orders):
        extra = [p for p in itertools.permutations(base) if p not in orders]
        orders.extend(extra)
    return orders[:min(count, len(orders))]


def _unit(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _remove_target_parallel(x: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = (target * target).sum()
    if float(denom) < eps:
        return x
    return x - (x * target).sum(dim=-1, keepdim=True) / denom * target


@torch.no_grad()
def combine_source_interaction(
    parameters: Sequence[torch.nn.Parameter],
    source_gradients: Sequence[Sequence[torch.Tensor | None]],
    target_gradients: Sequence[torch.Tensor | None],
    order_principals: Sequence[Sequence[torch.Tensor | None]],
    *,
    previous_weights: Sequence[float] | None = None,
    target_strength: float = 1.0,
    synergy_strength: float = 0.50,
    target_consistent_synergy: float = 0.0,
    conflict_strength: float = 0.50,
    redundancy_strength: float = 0.60,
    redundancy_threshold: float = 0.80,
    temperature: float = 0.35,
    uniform_mix: float = 0.15,
    weight_ema: float = 0.65,
    trajectory_blend: float = 0.50,
    eps: float = 1e-12,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Compute one *reference* vector for the unchanged rotation module.

    Only normalized source directions are used in relation computation, so city
    batch-size/gradient norm differences cannot overwhelm source interactions.
    Conflicting source pairs are symmetrically corrected **only inside the
    principal reference**; the actual per-source gradients remain untouched.
    """
    n = len(source_gradients)
    if n < 2 or not order_principals:
        raise ValueError('requires >=2 cities and >=1 PGrad reference')
    if len(target_gradients) != len(parameters):
        raise ValueError('target gradient shape mismatch')
    if not (0 <= uniform_mix <= 1 and 0 <= weight_ema < 1 and 0 <= trajectory_blend <= 1):
        raise ValueError('invalid source interaction convex weighting')
    if not (0 <= target_consistent_synergy <= 1):
        raise ValueError('target_consistent_synergy must be in [0,1]')
    if temperature <= 0 or not (0 <= redundancy_threshold <= 1):
        raise ValueError('invalid source interaction temperature/redundancy threshold')

    raw = torch.cat([flatten_gradients(parameters, g) for g in source_gradients], dim=0)
    target = flatten_gradients(parameters, target_gradients).reshape(-1)
    norms = raw.norm(dim=1)
    valid = norms > eps
    if not bool(valid.any()):
        return tuple(torch.zeros_like(p) for p in parameters), {
            'source_weights': [1.0/n]*n, 'zero_source_gradients': True,
            'pairwise_cosine': [[0.0]*n for _ in range(n)],
        }
    unit = raw / norms[:, None].clamp_min(eps)
    unit = torch.where(valid[:, None], unit, torch.zeros_like(unit))
    target_unit = _unit(target.unsqueeze(0), eps).reshape(-1)
    target_cos = unit @ target_unit
    pairwise_cos = (unit @ unit.T).clamp(-1., 1.)
    mask = 1.0 - torch.eye(n, dtype=unit.dtype, device=unit.device)
    pairwise = pairwise_cos * mask
    # Positive synergy is useful, but near-duplicate sources (e.g. correlated
    # source cities) must not receive disproportionate influence.
    pos = pairwise.clamp_min(0.)
    # Target-consistent positive synergy: two mutually aligned sources should
    # not receive a synergy bonus if either points against the target.
    # Setting strength=0 recovers the v2 baseline exactly.
    target_gate = target_cos.clamp_min(0.).clamp_max(1.)
    pair_gate = target_gate[:, None] * target_gate[None, :]
    gated_pos = pos.clamp(max=redundancy_threshold) * (
        1.0 - target_consistent_synergy + target_consistent_synergy * pair_gate)
    pos_synergy = gated_pos.sum(dim=1) / (n-1)
    redundancy = (pos - redundancy_threshold).clamp_min(0.).sum(dim=1) / (n-1)
    conflict = (-pairwise).clamp_min(0.).sum(dim=1) / (n-1)
    score = (target_strength * target_cos
             + synergy_strength * pos_synergy
             - conflict_strength * conflict
             - redundancy_strength * redundancy)
    score = score.masked_fill(~valid, -1e4)
    weights = torch.softmax(score / temperature, dim=0)
    weights = (1-uniform_mix)*weights + uniform_mix/n
    # Smooth scalar city weights across PGrad refreshes (no per-parameter EMA).
    if previous_weights is not None:
        prev = torch.as_tensor(previous_weights, dtype=unit.dtype, device=unit.device)
        if prev.shape == weights.shape and bool(torch.isfinite(prev).all()) and float(prev.sum()) > eps:
            prev = prev.clamp_min(0)
            prev = prev / prev.sum()
            weights = weight_ema*prev + (1-weight_ema)*weights
    weights = weights / weights.sum().clamp_min(eps)

    # Symmetric source-to-source conflict adjustment. This affects ONLY the
    # PGrad reference vector; the subsequent per-city rotation stays identical.
    neg = pairwise.clamp(max=0.)
    adjusted = unit - (neg @ unit)/(n-1)
    adjusted = torch.where(valid[:, None], adjusted, torch.zeros_like(adjusted))
    collaborative = (weights[:, None] * adjusted).sum(dim=0, keepdim=True)
    collaborative = _remove_target_parallel(collaborative, target, eps)

    # Average trajectory principals from balanced city orders. Orient each to
    # the mean source descent vector, guarding against axis sign ambiguity.
    anchor = (unit[valid]).mean(dim=0)
    oriented = []
    magnitudes = []
    for ref in order_principals:
        p = flatten_gradients(parameters, ref).reshape(-1)
        if not bool(torch.isfinite(p).all()):
            raise FloatingPointError('non-finite PGrad reference')
        magnitude = p.norm()
        if float(magnitude) <= eps:
            continue
        if float(torch.dot(p, anchor)) < 0:
            p = -p
        oriented.append(_unit(p.unsqueeze(0), eps).reshape(-1))
        magnitudes.append(magnitude)
    if not oriented:
        trajectory = torch.zeros_like(collaborative)
        reference_norm = norms[valid].mean()
        consistency = 0.0
    else:
        stacked = torch.stack(oriented)
        trajectory = stacked.mean(dim=0, keepdim=True)
        reference_norm = torch.stack(magnitudes).mean()
        consistency = float((_unit(stacked) @ _unit(trajectory).T).mean())
    trajectory = _remove_target_parallel(trajectory, target, eps)

    # Unlike ordinary trajectory PCA, downstream rotation only uses the
    # principal component orthogonal to the target gradient. Optimize that
    # active subspace here, without altering the rotation operator itself.
    if float(collaborative.norm()) <= eps:
        mixed = trajectory
    elif float(trajectory.norm()) <= eps:
        mixed = collaborative
    else:
        t = _unit(trajectory, eps)
        c = _unit(collaborative, eps)
        # Do not flip source collaboration merely to match a noisy trajectory.
        mixed = (1.0-trajectory_blend)*t + trajectory_blend*c
        if float(mixed.norm()) <= eps:
            mixed = c
    if float(mixed.norm()) <= eps:
        mixed = collaborative if float(collaborative.norm()) > eps else raw.mean(dim=0, keepdim=True)
        mixed = _remove_target_parallel(mixed, target, eps)
    principal_flat = _unit(mixed, eps) * reference_norm
    if not bool(torch.isfinite(principal_flat).all()):
        raise FloatingPointError('non-finite source interaction direction')
    result = unflatten_gradients(parameters, principal_flat)
    ref_unit = _unit(principal_flat, eps).reshape(-1)
    meta: dict[str, Any] = {
        'source_weights': [float(x) for x in weights],
        'source_target_cosine': [float(x) for x in target_cos],
        'pairwise_cosine': [[float(x) for x in row] for row in pairwise_cos],
        'source_positive_synergy': [float(x) for x in pos_synergy],
        'target_consistent_synergy': float(target_consistent_synergy),
        'target_gate': [float(x) for x in target_gate],
        'source_conflict': [float(x) for x in conflict],
        'source_redundancy': [float(x) for x in redundancy],
        'order_consistency': consistency,
        'principal_cosine_by_city': [float(x) for x in unit @ ref_unit],
        'principal_norm': float(principal_flat.norm()),
        'trajectory_principal_norm': float(reference_norm),
        'target_orthogonal_dot': float(torch.dot(ref_unit, target_unit)),
        'zero_source_gradients': False,
    }
    return result, meta


def source_interaction_principal_gradient(
    backbone: torch.nn.Module,
    gradient_callback: Callable[[int, Mapping[str, torch.Tensor]], Sequence[torch.Tensor | None]],
    parameters: Sequence[torch.nn.Parameter],
    source_gradients: Sequence[Sequence[torch.Tensor | None]],
    target_gradients: Sequence[torch.Tensor | None],
    *, virtual_lr: float,
    num_components: int = 2,
    rounds: int = 1,
    order_count: int = 3,
    previous_weights: Sequence[float] | None = None,
    target_strength: float = 1.0,
    synergy_strength: float = 0.5,
    target_consistent_synergy: float = 0.0,
    conflict_strength: float = 0.5,
    redundancy_strength: float = 0.6,
    redundancy_threshold: float = 0.8,
    temperature: float = 0.35,
    uniform_mix: float = 0.15,
    weight_ema: float = 0.65,
    trajectory_blend: float = 0.5,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Same gradient output signature as PGrad, now encoding source interactions."""
    if len(source_gradients) < 2:
        raise ValueError('requires at least two source cities')
    orders = balanced_city_orders(len(source_gradients), order_count)
    pgrads = []
    order_metas = []
    for order in orders:
        ref, info = pgrad_principal_gradient(
            backbone, gradient_callback,
            virtual_lr=virtual_lr, num_cities=len(source_gradients),
            num_components=num_components, city_order=order, rounds=rounds,
        )
        pgrads.append(ref)
        order_metas.append(info)
    result, meta = combine_source_interaction(
        parameters, source_gradients, target_gradients, pgrads,
        previous_weights=previous_weights,
        target_strength=target_strength, synergy_strength=synergy_strength,
        target_consistent_synergy=target_consistent_synergy,
        conflict_strength=conflict_strength, redundancy_strength=redundancy_strength,
        redundancy_threshold=redundancy_threshold, temperature=temperature,
        uniform_mix=uniform_mix, weight_ema=weight_ema,
        trajectory_blend=trajectory_blend,
    )
    meta.update({
        'mode': 'source_interaction_v2',
        'num_cities': len(source_gradients),
        'order_count': len(orders),
        'city_orders': [list(order) for order in orders],
        'trajectory_principal_norms': [float(v['principal_norm']) for v in order_metas],
        'num_components': int(num_components),
        'virtual_lr': float(virtual_lr),
        'pgrad_rounds': int(rounds),
        'pgrad_fallback': any(bool(v['pgrad_fallback']) for v in order_metas),
    })
    return result, meta


__all__ = ['balanced_city_orders', 'combine_source_interaction', 'source_interaction_principal_gradient']
