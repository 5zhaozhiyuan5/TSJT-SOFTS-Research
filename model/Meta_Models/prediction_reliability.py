"""Prediction-reliability calibration (PRC) for the ORIGINAL F2 OT source marginal.

One addition: forecast accuracy of the source traffic pattern influences
its F2 OT marginal alpha. F2 gradient compatibility remains unchanged.

Use current *training-source* predictions and labels only. Compare model
errors to the persistence predictor (repeat the last observed speed) so
trivially easy traffic is not automatically interpreted as useful skill.
These are training errors, not out-of-sample estimates of transfer utility.
"""
import math

import torch


@torch.no_grad()
def source_prediction_quality(prediction, truth, history, city_mean, city_std):
    """Return detached [B,N] accuracy quality in [0,1] and diagnostics.

    prediction/truth: [B,N,36] REAL city-speed units, as returned by PMME.
    history: [B,N,T,2] normalized traffic dataset input, channel 0 = speed.
    city_mean/std: scalar arrays used by the dataset for speed normalization.

    Absolute error and persistence error are divided by city std to compare
    across cities with different native speed scales. Score combines actual
    model accuracy and forecast skill relative to a naive persistence model.
    """
    if prediction.ndim != 3 or prediction.shape != truth.shape:
        raise ValueError("Prediction and truth must have matching [B,N,H] shapes")
    if history.ndim != 4 or history.shape[:2] != prediction.shape[:2]:
        raise ValueError("history must have shape [B,N,T,C] matching prediction")
    if history.shape[-1] < 1 or history.shape[-2] < 1:
        raise ValueError("history must contain normalized speed observations")
    if not torch.isfinite(prediction).all() or not torch.isfinite(truth).all():
        raise ValueError("Nonfinite prediction or source label")
    pred, target = prediction.detach().float(), truth.detach().float()
    h = history.detach().to(device=pred.device, dtype=pred.dtype)
    mean = torch.as_tensor(city_mean, device=pred.device,
                           dtype=pred.dtype).reshape(-1)[0]
    std = torch.as_tensor(city_std, device=pred.device,
                          dtype=pred.dtype).reshape(-1)[0]
    if not (torch.isfinite(mean) and torch.isfinite(std) and std > 0):
        raise ValueError("Invalid city mean/std")
    persistence = h[:, :, -1, 0] * std + mean
    # Physically meaningful absolute error; city-scale normalized for fairness.
    raw_error = (pred - target).abs().mean(dim=-1)
    error = raw_error / std
    persistence_error = (
        persistence.unsqueeze(-1) - target
    ).abs().mean(dim=-1) / std
    # Robust within-city scales; this is *not* a learnable calibrator.
    accuracy_scale = error.median().clamp_min(0.1)
    persistence_scale = persistence_error.median().clamp_min(0.1)
    absolute_accuracy = torch.exp(-error / accuracy_scale)
    skill = torch.sigmoid(
        ((persistence_error - error) / persistence_scale).clamp(-20, 20)
    )
    quality = (absolute_accuracy * skill).clamp(0.0, 1.0)
    if not torch.isfinite(quality).all():
        raise ValueError("Nonfinite prediction reliability")
    return quality, {
        "source_mae": float(raw_error.mean().item()),
        "persistence_mae": float(
            (persistence_error * std).mean().item()
        ),
        "source_normalized_mae": float(error.mean().item()),
        "model_beats_persistence_frac": float(
            (error < persistence_error).float().mean().item()
        ),
        "quality_mean": float(quality.mean().item()),
        "quality_std": float(quality.std(unbiased=False).item()),
        "samples": quality.numel(),
    }


@torch.no_grad()
def calibrate_source_alpha(f2_alpha, quality, strength=0.5):
    """Bounded multiplicative reliability correction to ORIGINAL F2 alpha.

    strength=0 gives F2 *bitwise unchanged* and adds no new model parameters.
    quality is relative-normalized and clipped to [0,2], so at strength=.5
    the multiplier lies in [0.5,1.5]; no sample is forcibly discarded.

    Target beta, OT ground cost and all prediction / PCGrad gradients remain
    untouched; quality/gates are detached so there is no second-order path.
    """
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("PRC strength must be finite in [0,1]")
    if f2_alpha.ndim != 1 or f2_alpha.numel() != quality.numel():
        raise ValueError("alpha / prediction quality element count mismatch")
    if not torch.isfinite(f2_alpha).all() or not torch.isfinite(quality).all():
        raise ValueError("Nonfinite alpha or quality")
    if (f2_alpha < 0).any() or not torch.isclose(
        f2_alpha.sum(), f2_alpha.new_tensor(1.0), atol=1e-5
    ):
        raise ValueError("F2 alpha must be normalized and nonnegative")
    if (quality < 0).any() or (quality > 1).any():
        raise ValueError("Prediction quality must be within [0,1]")
    ess_original = float((1.0 / f2_alpha.square().sum()).item())
    if strength == 0:
        return f2_alpha, {
            "strength": 0.0,
            "ess_f2": ess_original,
            "ess_after": ess_original,
            "min_multiplier": 1.0,
            "max_multiplier": 1.0,
        }
    q = quality.detach().reshape(-1).to(
        device=f2_alpha.device, dtype=f2_alpha.dtype
    )
    if torch.count_nonzero(q).item() == 0:
        relative = torch.ones_like(q)
    else:
        relative = (q / q.mean().clamp_min(1e-8)).clamp(0, 2)
    multiplier = (1.0 - strength) + strength * relative
    corrected = f2_alpha.detach() * multiplier
    corrected = corrected / corrected.sum().clamp_min(1e-12)
    return corrected.detach(), {
        "strength": float(strength),
        "ess_f2": ess_original,
        "ess_after": float((1.0 / corrected.square().sum()).item()),
        "min_multiplier": float(multiplier.min().item()),
        "max_multiplier": float(multiplier.max().item()),
    }
