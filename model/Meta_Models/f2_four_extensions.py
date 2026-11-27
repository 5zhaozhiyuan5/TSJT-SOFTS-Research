"""Four independent F2 marginal estimators (no PFR, gradient surgery or OT-cost edits).

All sample gradients are the per-node outer products of the original SOFTS
projection2 output loss gradients and its 128-D input representation.
The original F2 source weights are passed as a prior. This module returns
detached source and target marginals to the UNMODIFIED original Sinkhorn.

Inspirations, NOT exact reproductions:
  mag  : SAM-GS (ECML-PKDD 2025): relative gradient norm mismatch.
  sign : GradDrop (NeurIPS 2020): coordinate sign-disagreement purity.
  ema  : GradVac (ICLR 2021): historical target-gradient reference.
  biot : importance-weighted / unbalanced OT (ICML 2026): two-sided coupling.
  stability : sample-gradient similarity (TSMC 2024) and gradient-variance
              calibration (CVPR 2026 Findings): target-support sensitivity.
"""
import math
import torch
import torch.nn.functional as F

METHODS = ("original", "mag", "sign", "ema", "biot", "stability")


@torch.no_grad()
def _target_gradient(target_features, target_delta):
    return torch.einsum(
        "nh,nd->hd", target_delta.float(), target_features.float()
    )


@torch.no_grad()
def _rankone_cos(source_features, source_delta, target_grad, eps=1e-12):
    numerator = torch.einsum(
        "nh,hd,nd->n", source_delta.float(), target_grad.float(),
        source_features.float()
    )
    denominator = (source_delta.float().norm(dim=-1) *
                   source_features.float().norm(dim=-1) *
                   target_grad.float().norm() + eps)
    return (numerator / denominator).nan_to_num().clamp(-1, 1)


@torch.no_grad()
def _target_subsample_cosine_dispersion(
    source_features, source_delta, target_features, target_delta,
    n_subsamples=6, sample_ratio=0.8, seed=0,
):
    """Estimate per-source cosine uncertainty by resampling target windows.

    The resampling unit is the target support window (dimension B), not an
    individual sensor/node and not a prediction horizon. Every sampled window
    retains all of its sensors, preserving within-window spatial dependence.
    Gradients are measured on the same full prediction-head dimensions as F2.
    """
    if (target_features.ndim != 3 or target_delta.ndim != 3 or
            target_features.shape[:2] != target_delta.shape[:2] or
            target_features.shape[-1] != source_features.shape[-1] or
            target_delta.shape[-1] != source_delta.shape[-1]):
        raise ValueError("Target window feature/delta shapes do not match source head")
    if not isinstance(n_subsamples, int) or n_subsamples < 2:
        raise ValueError("n_subsamples must be >= 2")
    if not (0.5 <= sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in [0.5, 1.0]")

    n_windows = target_features.shape[0]
    if n_windows < 3:
        zeros = source_features.new_zeros(
            source_features.shape[0], dtype=torch.float32
        )
        return zeros, zeros, {
            "target_subsample_active": False,
            "target_subsample_reason": "fewer_than_3_target_windows",
            "target_windows": int(n_windows),
            "target_subset_windows": int(n_windows),
            "target_subsamples": 0,
        }

    subset_windows = max(
        2, min(n_windows, int(math.ceil(sample_ratio * n_windows)))
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**31 - 1))
    scores = []
    for _ in range(n_subsamples):
        # Repeated random subsampling without replacement.
        ids = torch.randperm(n_windows, generator=generator)[:subset_windows]
        ids = ids.to(device=target_features.device)
        f_sub = target_features.index_select(0, ids).reshape(
            -1, target_features.shape[-1]
        ).float()
        d_sub = target_delta.index_select(0, ids).reshape(
            -1, target_delta.shape[-1]
        ).float()
        g_target_sub = _target_gradient(f_sub, d_sub)
        scores.append(_rankone_cos(source_features, source_delta, g_target_sub))

    score_matrix = torch.stack(scores, dim=0)
    return score_matrix.mean(dim=0), score_matrix.std(dim=0, unbiased=False), {
        "target_subsample_active": True,
        "target_windows": int(n_windows),
        "target_subset_windows": int(subset_windows),
        "target_subsamples": int(n_subsamples),
        "target_sample_ratio": float(sample_ratio),
        "target_subsample_seed": int(seed),
    }


@torch.no_grad()
def _partition_by_features(features, k, seed):
    """Small, reproducible feature-based coarsening; never builds N_source x N_target."""
    n = features.shape[0]
    k = min(k, n)
    generator = torch.Generator(device="cpu").manual_seed(seed + n)
    anchors = torch.randperm(n, generator=generator)[:k].to(features.device)
    normalized = F.normalize(features.float(), dim=-1, eps=1e-8)
    labels = torch.cdist(normalized, normalized[anchors]).argmin(dim=1)
    # Each sampled anchor is the closest point to itself; therefore groups are populated.
    return labels, k


@torch.no_grad()
def _group_stats(features, deltas, weights, labels, k):
    """Return group masses, mean features, and mean EXACT rank-one head gradients."""
    dim = features.shape[-1]
    horizon = deltas.shape[-1]
    mass = features.new_zeros(k, dtype=torch.float32)
    centroid = features.new_zeros((k, dim), dtype=torch.float32)
    grads = features.new_zeros((k, horizon, dim), dtype=torch.float32)
    f = features.float()
    d = deltas.float()
    w = weights.float()
    mass.index_add_(0, labels, w)
    centroid.index_add_(0, labels, f * w[:, None])
    centroid /= mass[:, None].clamp_min(1e-12)
    for group in range(k):
        mask = labels == group
        if mask.any():
            grads[group] = (d[mask] * w[mask, None]).T @ f[mask]
    grads /= mass[:, None, None].clamp_min(1e-12)
    return mass, centroid, grads


@torch.no_grad()
def _biot_marginals(fs, ds, ft, dt, alpha0, beta0,
                    groups=32, blur=0.5, rho_s=0.8, rho_t=1.6,
                    blend=0.40):
    """Coarsened generalized Sinkhorn with relaxed source and target marginals.

    The group coupling is an INTERNAL mass estimator, not the final OT plan.
    The final PMME GeomLoss still sees the original joint Euclidean cost.
    """
    s_labels, ks = _partition_by_features(fs, groups, 1231)
    t_labels, kt = _partition_by_features(ft, groups, 1733)
    a, sh, sg = _group_stats(fs, ds, alpha0, s_labels, ks)
    b, th, tg = _group_stats(ft, dt, beta0, t_labels, kt)
    shn = F.normalize(sg.reshape(ks, -1), dim=-1, eps=1e-8)
    thn = F.normalize(tg.reshape(kt, -1), dim=-1, eps=1e-8)
    pair_cos = (shn @ thn.T).clamp(-1, 1)
    feature_distance = torch.cdist(sh.float(), th.float())
    feature_distance = feature_distance / feature_distance.median().clamp_min(1e-6)
    internal_cost = feature_distance + 0.5 * (1 - pair_cos)
    # This is a small group-to-group coupling, typically 32 x 32.
    kernel = torch.exp(-internal_cost / blur).clamp_min(1e-15)
    kernel = kernel * a[:, None] * b[None, :]
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    expo_s = rho_s / (rho_s + blur)
    expo_t = rho_t / (rho_t + blur)
    for _ in range(35):
        u = (a / (kernel @ v + 1e-15)).clamp(1e-6, 1e6).pow(expo_s)
        v = (b / (kernel.T @ u + 1e-15)).clamp(1e-6, 1e6).pow(expo_t)
    candidate = (u[:, None] * kernel) * v[None, :]
    candidate = candidate / candidate.sum().clamp_min(1e-15)
    src_group_mass = candidate.sum(dim=1)
    tgt_group_mass = candidate.sum(dim=0)
    # Preserve within-group F2 preference, while changing group-level mass.
    src = alpha0 * src_group_mass[s_labels] / a[s_labels].clamp_min(1e-15)
    tgt = beta0 * tgt_group_mass[t_labels] / b[t_labels].clamp_min(1e-15)
    src = (1 - blend) * alpha0 + blend * src
    tgt = (1 - blend) * beta0 + blend * tgt
    return src, tgt, {
        "coarse_source_groups": ks,
        "coarse_target_groups": kt,
        "coarse_pair_cos_mean": float(pair_cos.mean()),
        "coarse_group_source_ess": float(1 / src_group_mass.square().sum() / ks),
        "coarse_group_target_ess": float(1 / tgt_group_mass.square().sum() / kt),
    }


@torch.no_grad()
def compute_f2_extension(source_features, source_delta,
                         target_features, target_delta,
                         alpha_prior, beta_prior, method,
                         state=None, state_key=None,
                         mag_lambda=1.0, sign_lambda=1.5,
                         ema_momentum=0.85, ema_blend=0.40,
                         temperature=8.0, biot_blend=0.40,
                         stability_lambda=0.5, stability_subsamples=6,
                         stability_sample_ratio=0.8, stability_seed=0):
    """Compute enhanced marginal weights under one mutually exclusive method.

    Source/target features: [B,N,128], output deltas: [B,N,36].
    alpha_prior is the ORIGINAL F2 softmax; beta_prior is target-uniform.
    state is a caller-owned detached EMA dictionary, used only by 'ema'.
    """
    if method not in METHODS:
        raise ValueError("Unknown F2 extension: %s" % method)
    if method == "original":
        return alpha_prior.detach(), beta_prior.detach(), {"method": method}
    fs = source_features.detach().reshape(-1, source_features.shape[-1]).float()
    ft = target_features.detach().reshape(-1, target_features.shape[-1]).float()
    ds = source_delta.detach().reshape(-1, source_delta.shape[-1]).float()
    dt = target_delta.detach().reshape(-1, target_delta.shape[-1]).float()
    a0 = alpha_prior.detach().float().reshape(-1)
    b0 = beta_prior.detach().float().reshape(-1)
    if (ds.shape != (fs.shape[0], 36) or dt.shape != (ft.shape[0], 36) or
            a0.numel() != fs.shape[0] or b0.numel() != ft.shape[0]):
        raise ValueError("F2 source/target gradient and marginal shape mismatch")
    if not (0 <= ema_momentum < 1 and 0 <= ema_blend <= 1 and
            0 <= biot_blend <= 1 and mag_lambda >= 0 and
            sign_lambda >= 0 and temperature > 0):
        raise ValueError("Invalid extension coefficient")
    a, b = a0, b0
    info = {"method": method}
    if method == "mag":
        # IMPORTANT: raw autograd delta norms have different loss-reduction
        # factors for source and target; compare WITHIN-DOMAIN relative norms.
        source_norm = (ds.norm(dim=-1) * fs.norm(dim=-1)).clamp_min(1e-15).log()
        target_norm = (dt.norm(dim=-1) * ft.norm(dim=-1)).clamp_min(1e-15).log()
        source_relative = source_norm - source_norm.median()
        target_relative = target_norm - target_norm.median()
        low = torch.quantile(target_relative, 0.25)
        high = torch.quantile(target_relative, 0.75)
        deviation = (low - source_relative).clamp_min(0) + (
            source_relative - high
        ).clamp_min(0)
        a = torch.softmax(a0.clamp_min(1e-30).log() -
                          mag_lambda * deviation, dim=0)
        info["relative_magnitude_deviation_mean"] = float(deviation.mean())
    elif method == "sign":
        gt = _target_gradient(ft, dt)
        numerator = torch.einsum("nh,hd,nd->n", ds, gt, fs)
        absolute = torch.einsum(
            "nh,hd,nd->n", ds.abs(), gt.abs(), fs.abs()
        )
        # Magnitude-weighted opposite-sign fraction. Zero gradients are neutral.
        disagreement = torch.where(
            absolute > 1e-12,
            (0.5 * (1 - numerator / absolute.clamp_min(1e-12))).clamp(0, 1),
            torch.full_like(absolute, 0.5)
        )
        a = torch.softmax(a0.clamp_min(1e-30).log() -
                          sign_lambda * disagreement, dim=0)
        info["sign_conflict_mean"] = float(disagreement.mean())
        info["sign_conflict_high_frac"] = float(
            (disagreement > 0.5).float().mean()
        )
    elif method == "ema":
        if state is None or state_key is None:
            raise ValueError("EMA needs persistent state and group key")
        gt = _target_gradient(ft, dt)
        current = F.normalize(gt.reshape(-1), dim=0, eps=1e-12)
        previous = state.get(state_key)
        if previous is None:
            direction = current
            state[state_key] = current.detach().clone()
            info["ema_initialized"] = True
        else:
            previous = previous.to(current.device)
            direction = F.normalize(
                (1 - ema_blend) * current + ema_blend * previous,
                dim=0, eps=1e-12
            )
            state[state_key] = F.normalize(
                ema_momentum * previous + (1 - ema_momentum) * current,
                dim=0, eps=1e-12
            ).detach()
            info["ema_initialized"] = False
            info["target_direction_cos_to_ema"] = float(
                (current * previous).sum()
            )
        cosine = _rankone_cos(fs, ds, direction.reshape(36, fs.shape[-1]))
        a = torch.softmax(temperature * cosine, dim=0)
        info["ema_cos_mean"] = float(cosine.mean())
    elif method == "biot":
        a, b, grouped = _biot_marginals(
            fs, ds, ft, dt, a0, b0, blend=biot_blend
        )
        info.update(grouped)
    elif method == "stability":
        if (not math.isfinite(stability_lambda) or stability_lambda < 0 or
                not isinstance(stability_subsamples, int) or
                stability_subsamples < 2 or
                not 0.5 <= stability_sample_ratio <= 1.0):
            raise ValueError("Invalid F2 gradient-stability settings")
        full_target_gradient = _target_gradient(ft, dt)
        full_cosine = _rankone_cos(fs, ds, full_target_gradient)
        _, cosine_std, sampling = _target_subsample_cosine_dispersion(
            fs, ds,
            target_features.detach().float(),
            target_delta.detach().float(),
            n_subsamples=stability_subsamples,
            sample_ratio=stability_sample_ratio,
            seed=stability_seed,
        )
        # Lower-confidence-bound calibration: retain F2's full-support
        # alignment score, subtract only its target-support sensitivity.
        robust_score = full_cosine - stability_lambda * cosine_std
        a = torch.softmax(temperature * robust_score, dim=0)
        info.update(sampling)
        info.update({
            "gradient_dimensions": int(ds.shape[-1]),
            "prediction_horizon_split": False,
            "stability_lambda": float(stability_lambda),
            "full_cos_mean": float(full_cosine.mean()),
            "full_cos_std": float(full_cosine.std(unbiased=False)),
            "target_cosine_dispersion_mean": float(cosine_std.mean()),
            "target_cosine_dispersion_median": float(cosine_std.median()),
            "target_cosine_dispersion_p90": float(torch.quantile(
                cosine_std, torch.tensor(0.90, device=cosine_std.device)
            )),
            "uncertainty_penalty_mean": float(
                (stability_lambda * cosine_std).mean()
            ),
            "robust_score_mean": float(robust_score.mean()),
            "robust_score_std": float(robust_score.std(unbiased=False)),
            "raw_source_ess_frac": float(
                1.0 / a0.square().sum().clamp_min(1e-15) / a0.numel()
            ),
        })
    a = a.clamp_min(0)
    b = b.clamp_min(0)
    a /= a.sum().clamp_min(1e-15)
    if method == "stability":
        # F2-GS is source-only calibration: preserve F2's target beta exactly.
        b = beta_prior.detach().float().reshape(-1)
    else:
        b /= b.sum().clamp_min(1e-15)
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise FloatingPointError("F2 extension produced nonfinite marginals")
    if a.min() < 0 or b.min() < 0:
        raise FloatingPointError("F2 extension produced negative marginals")
    info["source_ess_frac"] = float(1 / a.square().sum() / a.numel())
    info["target_ess_frac"] = float(1 / b.square().sum() / b.numel())
    info["source_l1_shift"] = float((a - a0).abs().sum())
    info["target_l1_shift"] = float((b - b0).abs().sum())
    return a.detach(), b.detach(), info
