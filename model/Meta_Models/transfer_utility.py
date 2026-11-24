"""Task-utility OT weights; no DVGP, domain classification or pseudo-domain labels.

The score is a true first-order cosine between each sensor/window's
gradient of the *shared final forecasting head* and the target-support
mean gradient of the same head. It is not a full-backbone parameter
gradient and should not be described as such.

This module does not modify the Sinkhorn OT operator or either backbone.
"""
import torch
import torch.nn.functional as F

VARIANTS = ("uniform", "gradient", "horizon", "reliable")


@torch.no_grad()
def _head_gradient_cosine(source_features, source_delta,
                          target_features, target_delta, start, end,
                          eps=1e-12):
    """Cosine of rank-1 per-node readout gradient with target gradient.

    Shared readout W has shape [H,D]. Each node's contribution equals
    delta[:, :, start:end] outer features (no huge [nodes,H,D] tensor).
    """
    ds = source_delta[..., start:end].float()
    dt = target_delta[..., start:end].float()
    fs = source_features.float()
    ft = target_features.float()
    g_target = torch.einsum("bnh,bnd->hd", dt, ft)
    dot = torch.einsum("bnh,hd,bnd->bn", ds, g_target, fs)
    denominator = (ds.norm(dim=-1) * fs.norm(dim=-1) *
                   g_target.norm() + eps)
    return torch.nan_to_num(dot / denominator).clamp(-1.0, 1.0)


@torch.no_grad()
def _target_band_reliability(target_features, target_delta, start, end,
                             eps=1e-12):
    """Check gradient direction agreement between two target support halves."""
    b = target_features.shape[0]
    cut = max(1, b // 2)
    if cut == b:
        return target_features.new_tensor(0.0)
    f = target_features.float()
    d = target_delta[..., start:end].float()
    a = torch.einsum("bnh,bnd->hd", d[:cut], f[:cut])
    c = torch.einsum("bnh,bnd->hd", d[cut:], f[cut:])
    cos = (a * c).sum() / (a.norm() * c.norm() + eps)
    return cos.clamp(min=0.0, max=1.0)


@torch.no_grad()
def compute_weights(source_features, source_delta, target_features,
                    target_delta, variant, temperature=8.0,
                    min_ess_fraction=0.5):
    """Return source/target Sinkhorn marginals and diagnostics.

    Input feature shapes: [B,N,D] (actual input to projection2).
    Delta shapes: [B,N,36] (autograd loss gradient wrt projection2 output).
    No trained probability estimator is used.

    F1 uniform: both marginals uniform.
    F2 gradient: one 36-step shared-head gradient agreement.
    F3 horizon: equal short/mid/long 12-step gradient agreements.
    F4 reliable: target support split-half stability weighting plus ESS floor.
    """
    if variant not in VARIANTS:
        raise ValueError("Unknown transfer variant: " + str(variant))
    if source_delta.shape[-1] != 36 or target_delta.shape[-1] != 36:
        raise ValueError("Expected 36-step prediction-head deltas")
    if source_features.shape[:2] != source_delta.shape[:2]:
        raise ValueError("Source gradient/representation shape mismatch")
    if target_features.shape[:2] != target_delta.shape[:2]:
        raise ValueError("Target gradient/representation shape mismatch")

    count_s = source_features.shape[0] * source_features.shape[1]
    count_t = target_features.shape[0] * target_features.shape[1]
    source_uniform = source_features.new_full((count_s,), 1.0 / count_s)
    target_uniform = target_features.new_full((count_t,), 1.0 / count_t)
    if variant == "uniform":
        return source_uniform, target_uniform, {
            "ess_frac": 1.0, "mean_cos": 0.0,
            "positive_frac": 0.0, "stabilities": [0.0, 0.0, 0.0]
        }

    if variant == "gradient":
        scores = _head_gradient_cosine(source_features, source_delta,
                                       target_features, target_delta, 0, 36)
        stability = [0.0, 0.0, 0.0]
    else:
        intervals = ((0, 12), (12, 24), (24, 36))
        bands = torch.stack([
            _head_gradient_cosine(source_features, source_delta,
                                  target_features, target_delta, left, right)
            for left, right in intervals
        ], dim=0)
        if variant == "reliable":
            reliabilities = torch.stack([
                _target_band_reliability(target_features, target_delta,
                                         left, right)
                for left, right in intervals
            ])
            # 0.15 prior avoids a zero-weight band on tiny target sets.
            coefficients = (reliabilities + 0.15)
            coefficients = coefficients / coefficients.sum()
            scores = (bands * coefficients[:, None, None]).sum(dim=0)
            stability = reliabilities.cpu().tolist()
        else:
            scores = bands.mean(dim=0)
            stability = [0.0, 0.0, 0.0]

    flat = scores.reshape(-1)
    importance = torch.softmax(temperature * flat, dim=0)
    if variant == "reliable":
        # Analytic convex shrinkage to ensure ESS >= min_ess_fraction*N.
        # For w = (1-lambda)*uniform + lambda*importance,
        # N * sum(w^2) = 1 + lambda^2*(N*sum(importance^2)-1).
        n = float(count_s)
        concentration = (n * importance.square().sum() - 1.0).clamp(min=0)
        permitted = ((1.0 / min_ess_fraction - 1.0) /
                     (concentration + 1e-12)).sqrt().clamp(max=0.85)
        importance = (1.0 - permitted) * source_uniform + permitted * importance

    ess = (1.0 / importance.square().sum()) / count_s
    return importance.detach(), target_uniform.detach(), {
        "ess_frac": float(ess.cpu()),
        "mean_cos": float(flat.mean().cpu()),
        "positive_frac": float((flat > 0).float().mean().cpu()),
        "stabilities": stability,
    }
