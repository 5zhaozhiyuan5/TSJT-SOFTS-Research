"""Deterministic OT sample-weight strategies for Stage-I ablations.

Input probabilities follow official PMME semantics:
source = P(target domain | source), target = P(source domain | target).
This module changes weighting, not the domain classifier or the OT solver.
"""
from __future__ import annotations

import torch


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    return 1.0 / weights.square().sum().clamp_min(1e-12)


def build_ot_weights(
    source_probs: torch.Tensor,
    target_probs: torch.Tensor,
    strategy: str = "legacy",
    scale_strategy: str = "legacy",
    threshold: float = 0.3,
    floor: float = 0.05,
    power: float = 1.0,
):
    """Return (alpha, beta, OT_scale, active, diagnostics).

    legacy: original hard-threshold and original sqrt(mean_s * mean_t).
    uniform: no probability-based selection.
    smooth: relative positive weights; never zero out every sample.

    scale_strategy=legacy preserves the original scale from selected weights.
    scale_strategy=fixed sets scale to one regardless of classifier confidence.
    For inactive legacy masses, the returned uniform weights are a safe
    placeholder: active=False forces OT loss zero in the caller.
    """
    if strategy not in {"legacy", "uniform", "smooth"}:
        raise ValueError(f"Unsupported OT weight strategy: {strategy}")
    if scale_strategy not in {"legacy", "fixed"}:
        raise ValueError(f"Unsupported OT scale strategy: {scale_strategy}")
    if not 0.0 <= floor < 1.0:
        raise ValueError("OT weight floor must be in [0,1)")
    if power <= 0:
        raise ValueError("OT weight power must be positive")
    ps = source_probs.detach().reshape(-1)
    pt = target_probs.detach().reshape(-1)
    if ps.numel() == 0 or pt.numel() == 0:
        raise ValueError("Empty OT probability inputs")
    if not torch.isfinite(ps).all() or not torch.isfinite(pt).all():
        raise ValueError("Nonfinite OT probabilities")
    if ((ps < 0) | (ps > 1)).any() or ((pt < 0) | (pt > 1)).any():
        raise ValueError("OT probabilities must lie in [0,1]")

    raw_source_accept = (ps > threshold).float().mean()
    raw_target_accept = (pt > threshold).float().mean()
    if strategy == "uniform":
        ws = torch.ones_like(ps)
        wt = torch.ones_like(pt)
    elif strategy == "smooth":
        # Domain probability only determines relative preference.  Use
        # max-normalization to prevent near-zero probability scales.
        ws = floor + (1.0 - floor) * (ps / ps.max().clamp_min(1e-8)).pow(power)
        wt = floor + (1.0 - floor) * (pt / pt.max().clamp_min(1e-8)).pow(power)
    else:
        ws = torch.where(ps > threshold, ps, torch.zeros_like(ps))
        wt = torch.where(pt > threshold, pt, torch.zeros_like(pt))

    active = bool((ws.sum() > 0).item() and (wt.sum() > 0).item())
    if active:
        alpha = ws / ws.sum()
        beta = wt / wt.sum()
    else:
        alpha = torch.full_like(ps, 1.0 / ps.numel())
        beta = torch.full_like(pt, 1.0 / pt.numel())
    if scale_strategy == "fixed":
        scale = ps.new_tensor(1.0)
    else:
        scale = torch.sqrt(ws.mean() * wt.mean())

    diagnostics = {
        "strategy": strategy,
        "scale_strategy": scale_strategy,
        "active": active,
        "raw_source_mean": float(ps.mean().cpu()),
        "raw_target_mean": float(pt.mean().cpu()),
        "raw_source_fraction_above_threshold": float(raw_source_accept.cpu()),
        "raw_target_fraction_above_threshold": float(raw_target_accept.cpu()),
        "selected_source_fraction": float((ws > 0).float().mean().cpu()),
        "selected_target_fraction": float((wt > 0).float().mean().cpu()),
        "source_ess": float(effective_sample_size(alpha).cpu()) if active else 0.0,
        "target_ess": float(effective_sample_size(beta).cpu()) if active else 0.0,
        "source_ess_fraction": float((effective_sample_size(alpha) / ps.numel()).cpu()) if active else 0.0,
        "target_ess_fraction": float((effective_sample_size(beta) / pt.numel()).cpu()) if active else 0.0,
        "ot_scale": float(scale.cpu()),
    }
    return alpha.detach(), beta.detach(), scale.detach(), active, diagnostics
