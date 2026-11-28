"""Prediction-reliability calibration for the original F2 source marginal.

F2 continues to estimate transfer compatibility from shared-head gradients.
This module independently measures each source window/node item's forecasting
error from the same forward pass used by the source task loss, then uses that
error only to calibrate F2's source marginal. It does not change target beta,
the OT cost, or Sinkhorn.

The reliability is relative within each source city/batch: this avoids comparing
raw MAE across cities with different traffic scales. Each marginal item is one
(batch window, sensor node) row, matching projection2's [B,N,128] features.
"""
from __future__ import annotations

import math
import torch


def _ess_fraction(weights: torch.Tensor, eps: float = 1e-12) -> float:
    weights = weights.float().reshape(-1)
    n = max(int(weights.numel()), 1)
    return float((1.0 / weights.square().sum().clamp_min(eps) / n).item())


def _error_correlation(error: torch.Tensor, weights: torch.Tensor,
                       eps: float = 1e-12) -> float:
    x = error.float().reshape(-1)
    y = weights.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if not bool(torch.isfinite(denom)) or float(denom) <= eps:
        return 0.0
    return float(((x * y).sum() / denom).clamp(-1.0, 1.0).item())


@torch.no_grad()
def calibrate_source_reliability(
    alpha_prior: torch.Tensor,
    source_prediction: torch.Tensor,
    source_target: torch.Tensor,
    blend: float = 0.5,
    sensitivity: float = 0.5,
    z_clip: float = 3.0,
    multiplier_min: float = 0.5,
    multiplier_max: float = 2.0,
    eps: float = 1e-8,
):
    """Calibrate original-F2 alpha using per-window/node source forecast error.

    Args:
        alpha_prior: Original F2 source marginal, flattened [B*N].
        source_prediction/source_target: Matching source forecasts and labels,
            normally [B,N,36]. Flattening the first B*N rows preserves exactly
            the same order as projection2's source features and F2 alpha.
        blend: Convex blend between original F2 alpha and reliability-tilted
            alpha. 0 preserves F2; 1 applies the full reliability tilt.
        sensitivity: Strength of the robust error-to-reliability mapping.
        z_clip: Clip robust standardized errors before exponentiation.
        multiplier_min/max: Bounds preventing a single prediction error from
            collapsing or dominating a source item's F2 mass.

    Returns:
        (alpha_new, stats). The caller must retain the original beta unchanged.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0, 1]")
    if not math.isfinite(sensitivity) or sensitivity < 0:
        raise ValueError("sensitivity must be finite and >= 0")
    if not math.isfinite(z_clip) or z_clip <= 0:
        raise ValueError("z_clip must be finite and > 0")
    if (not math.isfinite(multiplier_min) or
            not math.isfinite(multiplier_max) or
            multiplier_min <= 0 or multiplier_max < multiplier_min):
        raise ValueError("Invalid reliability multiplier bounds")

    alpha = alpha_prior.detach().float().reshape(-1)
    if alpha.numel() < 1 or not bool(torch.isfinite(alpha).all()):
        raise ValueError("alpha_prior must be non-empty and finite")
    if bool((alpha < 0).any()) or float(alpha.sum()) <= eps:
        raise ValueError("alpha_prior must be non-negative with positive mass")
    alpha = alpha / alpha.sum()

    prediction = source_prediction.detach().to(
        device=alpha.device, dtype=torch.float32
    )
    target = source_target.detach().to(
        device=alpha.device, dtype=torch.float32
    )
    if prediction.shape != target.shape:
        raise ValueError(
            "Source prediction/target shape mismatch: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if prediction.numel() != alpha.numel() * (
        prediction.numel() // max(alpha.numel(), 1)
    ):
        raise ValueError("Invalid source prediction item count")
    if prediction.numel() % alpha.numel() != 0:
        raise ValueError(
            "Source prediction rows do not align with F2 alpha: "
            f"{prediction.numel()} values for {alpha.numel()} marginal items"
        )

    prediction = prediction.reshape(alpha.numel(), -1)
    target = target.reshape(alpha.numel(), -1)
    if prediction.shape[1] < 1:
        raise ValueError("Each source marginal item needs prediction values")
    if (not bool(torch.isfinite(prediction).all()) or
            not bool(torch.isfinite(target).all())):
        raise ValueError("Non-finite source predictions/targets in PRC")

    # Each row corresponds to one source window x sensor item and contains
    # the complete forecast horizon. Use the same physical scale as task loss.
    sample_mae = (prediction - target).abs().mean(dim=1)
    median = torch.quantile(sample_mae, 0.5)
    mad = torch.quantile((sample_mae - median).abs(), 0.5)
    robust_scale = 1.4826 * mad
    fallback_scale = sample_mae.std(unbiased=False)
    scale = torch.where(
        robust_scale > eps, robust_scale, fallback_scale
    ).clamp_min(eps)

    z = ((sample_mae - median) / scale).clamp(-z_clip, z_clip)
    raw_multiplier = torch.exp(-sensitivity * z)
    reliability = raw_multiplier.clamp(
        min=multiplier_min, max=multiplier_max
    )

    tilted = alpha * reliability
    tilted = tilted / tilted.sum().clamp_min(eps)
    alpha_new = (1.0 - blend) * alpha + blend * tilted
    alpha_new = alpha_new / alpha_new.sum().clamp_min(eps)

    top_k = min(5, alpha_new.numel())
    top_weight, top_index = torch.topk(alpha_new, k=top_k)
    worst_error = torch.topk(sample_mae, k=top_k, largest=True).indices
    stats = {
        "method": "f2_prc",
        "source_items": int(alpha_new.numel()),
        "forecast_values_per_item": int(prediction.shape[1]),
        "source_mae_mean": float(sample_mae.mean().item()),
        "source_mae_median": float(median.item()),
        "source_mae_p90": float(torch.quantile(sample_mae, 0.90).item()),
        "source_mae_mad": float(mad.item()),
        "robust_error_scale": float(scale.item()),
        "error_z_mean": float(z.mean().item()),
        "error_z_std": float(z.std(unbiased=False).item()),
        "high_error_fraction_z_gt_1": float((z > 1.0).float().mean().item()),
        "reliability_mean": float(reliability.mean().item()),
        "reliability_min": float(reliability.min().item()),
        "reliability_p10": float(torch.quantile(reliability, 0.10).item()),
        "reliability_p90": float(torch.quantile(reliability, 0.90).item()),
        "reliability_max": float(reliability.max().item()),
        "reliability_gt_1_fraction": float(
            (reliability > 1.0).float().mean().item()
        ),
        "source_ess_frac_before": _ess_fraction(alpha),
        "source_ess_frac_after": _ess_fraction(alpha_new),
        "source_l1_shift": float((alpha_new - alpha).abs().sum().item()),
        "error_alpha_corr_before": _error_correlation(sample_mae, alpha),
        "error_alpha_corr_after": _error_correlation(sample_mae, alpha_new),
        "top_source_indices": top_index.detach().cpu().tolist(),
        "top_source_weights": top_weight.detach().cpu().tolist(),
        "top_source_reliability": reliability[top_index].detach().cpu().tolist(),
        "worst_error_indices": worst_error.detach().cpu().tolist(),
        "worst_error_values": sample_mae[worst_error].detach().cpu().tolist(),
        "blend": float(blend),
        "sensitivity": float(sensitivity),
        "z_clip": float(z_clip),
        "target_marginal_unchanged": True,
    }
    return alpha_new.detach(), stats
