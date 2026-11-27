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
"""
import math
import torch
import torch.nn.functional as F

METHODS = ("original", "mag", "sign", "ema", "biot", "geomed", "influence", "cvar", "triple")


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
def _target_window_gradients(target_features, target_delta):
    """One 36 x D shared-readout gradient per target time window.

    Sensors are aggregated inside each window, so correlated sensors are not
    incorrectly treated as independent target observations.
    """
    if (target_features.ndim != 3 or target_delta.ndim != 3 or
            target_features.shape[:2] != target_delta.shape[:2] or
            target_delta.shape[-1] != 36):
        raise ValueError("Expected target features [B,N,D] and deltas [B,N,36]")
    return torch.einsum(
        "bnh,bnd->bhd", target_delta.detach().float(),
        target_features.detach().float()
    )


@torch.no_grad()
def _geometric_median(points, iterations=6, eps=1e-8):
    """Weiszfeld approximation to the geometric median of window gradients."""
    if points.ndim < 2 or points.shape[0] < 1:
        raise ValueError("Geometric median needs at least one point")
    flat = points.reshape(points.shape[0], -1).float()
    center = flat.mean(dim=0)
    for _ in range(int(iterations)):
        distance = torch.linalg.vector_norm(flat - center, dim=1).clamp_min(eps)
        weight = distance.reciprocal()
        updated = (flat * weight[:, None]).sum(dim=0) / weight.sum().clamp_min(eps)
        center = updated
    return center.reshape(points.shape[1:])


@torch.no_grad()
def _diagonal_fisher_precision(window_gradients, damping=0.1, eps=1e-12):
    """Damped inverse diagonal empirical-Fisher proxy from target windows."""
    fisher_diag = window_gradients.float().square().mean(dim=0)
    scale = fisher_diag.mean().clamp_min(eps)
    precision = (fisher_diag + float(damping) * scale + eps).reciprocal()
    return precision, fisher_diag


@torch.no_grad()
def _rankone_metric_cos(source_features, source_delta, target_grad,
                        precision, eps=1e-12):
    """Fisher-metric cosine without materializing dense per-source gradients."""
    fs = source_features.float()
    ds = source_delta.float()
    target_grad = target_grad.float()
    precision = precision.float()
    dot = torch.einsum("nh,hd,nd->n", ds, precision * target_grad, fs)
    source_norm_sq = torch.einsum(
        "nh,hd,nd->n", ds.square(), precision, fs.square()
    ).clamp_min(0)
    target_norm_sq = (precision * target_grad.square()).sum().clamp_min(0)
    denominator = (source_norm_sq * target_norm_sq).sqrt() + eps
    return torch.nan_to_num(dot / denominator).clamp(-1.0, 1.0)


@torch.no_grad()
def _temporal_block_gradients(window_gradients, requested_blocks=5):
    """Chronologically split target windows into contiguous, balanced blocks."""
    n_windows = int(window_gradients.shape[0])
    if n_windows < 2:
        return window_gradients.mean(dim=0, keepdim=True)
    # Keep at least five windows in a block where possible; the PMME target
    # groups can be as small as 15 windows (three blocks of five).
    n_blocks = min(int(requested_blocks), max(2, n_windows // 5), n_windows)
    chunks = torch.tensor_split(window_gradients, n_blocks, dim=0)
    return torch.stack([chunk.mean(dim=0) for chunk in chunks if chunk.shape[0]], dim=0)


@torch.no_grad()
def _lower_tail_mean(scores, tail_fraction=0.4):
    """Empirical lower-tail CVaR of compatibility scores [blocks, samples]."""
    if scores.ndim != 2 or scores.shape[0] < 1:
        raise ValueError("CVaR expects [num_blocks, num_samples] scores")
    tail_count = max(1, int(math.ceil(float(tail_fraction) * scores.shape[0])))
    return scores.sort(dim=0).values[:tail_count].mean(dim=0), tail_count


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
                         geomed_iters=6, fisher_damping=0.1,
                         cvar_blocks=5, cvar_tail_fraction=0.4):
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
    if (not isinstance(geomed_iters, int) or geomed_iters < 1 or
            not math.isfinite(fisher_damping) or fisher_damping <= 0 or
            not isinstance(cvar_blocks, int) or cvar_blocks < 2 or
            not math.isfinite(cvar_tail_fraction) or
            not 0 < cvar_tail_fraction <= 1):
        raise ValueError("Invalid robust/Fisher/CVaR extension settings")
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
    elif method in ("geomed", "influence", "cvar", "triple"):
        # Keep [target windows, sensors, ...] until after target-window
        # gradients are formed; do not treat sensors as independent windows.
        target_window_grad = _target_window_gradients(
            target_features, target_delta
        )
        mean_target_grad = target_window_grad.mean(dim=0)
        robust_target_grad = _geometric_median(
            target_window_grad, iterations=geomed_iters
        )
        block_grads = _temporal_block_gradients(
            target_window_grad, requested_blocks=cvar_blocks
        )
        precision, fisher_diag = _diagonal_fisher_precision(
            target_window_grad, damping=fisher_damping
        )

        # Original F2 reference score (mean target gradient).
        f2_score = _rankone_cos(fs, ds, mean_target_grad)
        # Robust-statistics score: geometric median suppresses outlying
        # target windows before measuring source/target compatibility.
        geomed_score = _rankone_cos(fs, ds, robust_target_grad)
        # Influence-inspired score: a damped diagonal empirical-Fisher
        # inverse supplies a parameter-sensitive metric. This is NOT an
        # exact Hessian influence function and needs no second derivatives.
        fisher_score = _rankone_metric_cos(
            fs, ds, mean_target_grad, precision
        )
        robust_fisher_score = _rankone_metric_cos(
            fs, ds, robust_target_grad, precision
        )

        block_cos_scores = torch.stack([
            _rankone_cos(fs, ds, block_grads[k])
            for k in range(block_grads.shape[0])
        ], dim=0)
        block_fisher_scores = torch.stack([
            _rankone_metric_cos(fs, ds, block_grads[k], precision)
            for k in range(block_grads.shape[0])
        ], dim=0)
        cvar_score, tail_count = _lower_tail_mean(
            block_cos_scores, tail_fraction=cvar_tail_fraction
        )
        fisher_cvar_score, _ = _lower_tail_mean(
            block_fisher_scores, tail_fraction=cvar_tail_fraction
        )

        if method == "geomed":
            final_score = geomed_score
        elif method == "influence":
            final_score = fisher_score
        elif method == "cvar":
            # Keep the original F2 score as an anchor and add a lower-tail
            # compatibility term across contiguous target time blocks.
            final_score = 0.5 * f2_score + 0.5 * cvar_score
        else:
            # Three-way mechanism: robust center + Fisher geometry +
            # worst-block risk. Original F2 remains as a conservative anchor.
            final_score = (
                0.20 * f2_score +
                0.40 * robust_fisher_score +
                0.40 * fisher_cvar_score
            )

        a = torch.softmax(temperature * final_score, dim=0)
        mean_grad_norm = mean_target_grad.norm().clamp_min(1e-12)
        info.update({
            "target_windows": int(target_window_grad.shape[0]),
            "target_sensors_per_window": int(target_features.shape[1]),
            "geomed_iterations": int(geomed_iters),
            "geomed_shift_rel": float(
                (robust_target_grad - mean_target_grad).norm() / mean_grad_norm
            ),
            "geomed_vs_mean_cos": float(F.cosine_similarity(
                robust_target_grad.reshape(1, -1),
                mean_target_grad.reshape(1, -1), dim=1
            )[0]),
            "fisher_damping": float(fisher_damping),
            "fisher_diag_mean": float(fisher_diag.mean()),
            "fisher_diag_p90": float(torch.quantile(
                fisher_diag.reshape(-1),
                torch.tensor(0.90, device=fisher_diag.device)
            )),
            "fisher_precision_p90_p10": float(
                torch.quantile(precision.reshape(-1), torch.tensor(
                    0.90, device=precision.device
                )) /
                torch.quantile(precision.reshape(-1), torch.tensor(
                    0.10, device=precision.device
                )).clamp_min(1e-12)
            ),
            "f2_score_mean": float(f2_score.mean()),
            "geomed_score_mean": float(geomed_score.mean()),
            "fisher_score_mean": float(fisher_score.mean()),
            "robust_fisher_score_mean": float(robust_fisher_score.mean()),
            "cvar_blocks": int(block_grads.shape[0]),
            "cvar_tail_fraction": float(cvar_tail_fraction),
            "cvar_tail_count": int(tail_count),
            "cvar_score_mean": float(cvar_score.mean()),
            "cvar_score_p10": float(torch.quantile(
                cvar_score, torch.tensor(0.10, device=cvar_score.device)
            )),
            "fisher_cvar_score_mean": float(fisher_cvar_score.mean()),
            "block_score_std_mean": float(
                block_cos_scores.std(dim=0, unbiased=False).mean()
            ),
            "final_score_mean": float(final_score.mean()),
            "final_score_std": float(final_score.std(unbiased=False)),
            "prediction_horizon_split": False,
            "target_marginal_unchanged": True,
        })
    a = a.clamp_min(0)
    b = b.clamp_min(0)
    a /= a.sum().clamp_min(1e-15)
    if method in ("geomed", "influence", "cvar", "triple"):
        # These are source-only F2 refinements; preserve target beta exactly.
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
