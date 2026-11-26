"""D1/D2: detached, counterfactual audit of ORIGINAL F2 sample ranking.

D1: Does a high source-vs-target prediction-head cosine predict a MORE
    helpful *real Sinkhorn OT gradient* on projection1.weight?
D2: After controlling for that cosine, does GradDrop-style coordinate
    sign disagreement add useful information?

All OT candidates use EXACTLY the same, small, fixed source and target
support sets and the ORIGINAL GeomLoss cost. Only alpha differs. The target
probe windows are EXCLUDED from the audit's alpha estimation AND candidate
OT target points. The probe gradient is computed from those excluded
windows and is used ONLY as a diagnostic evaluation signal.

This is a first-order, one-step, within-three-day-support diagnostic:
    <g_probe, g_OT>
is not the full PCGrad + AdamW gain, not a causal sample-pair effect, and
not an independent chronological validation score. The original training
path, parameter .grad fields, optimizer state and final Sinkhorn remain
untouched. No PFR or new trainable model is used.
"""
from __future__ import annotations

import math
import torch


def probe_window_masks(n_windows: int, window_stride: int = 5, device=None):
    """Window-level split; never split individual sensor nodes of a window."""
    if n_windows < 8 or window_stride < 2:
        raise ValueError("Need >=8 windows and stride>=2")
    index = torch.arange(n_windows, device=device)
    probe = index.remainder(window_stride).eq(0)
    fit = ~probe
    if probe.sum() < 1 or fit.sum() < 1:
        raise ValueError("Empty target probe or fit partition")
    return fit, probe


@torch.no_grad()
def source_gradient_scores(source_features, source_deltas,
                           target_features, target_deltas, eps=1e-12):
    """Exact rank-one 36x128 source-head gradient vs pooled target head."""
    fs = source_features.detach().reshape(-1, source_features.shape[-1]).float()
    ds = source_deltas.detach().reshape(-1, source_deltas.shape[-1]).float()
    ft = target_features.detach().reshape(-1, target_features.shape[-1]).float()
    dt = target_deltas.detach().reshape(-1, target_deltas.shape[-1]).float()
    if ds.shape[-1] != 36 or dt.shape[-1] != 36:
        raise ValueError("Expected 36-step forecast gradients")
    if fs.shape[-1] != ft.shape[-1]:
        raise ValueError("Shared readout feature dimensions must match")
    gt = torch.einsum("nh,nd->hd", dt, ft)
    dot = torch.einsum("nh,hd,nd->n", ds, gt, fs)
    norms = ds.norm(dim=-1) * fs.norm(dim=-1) * gt.norm()
    cosine = (dot / (norms + eps)).nan_to_num().clamp(-1, 1)
    absolute = torch.einsum("nh,hd,nd->n", ds.abs(), gt.abs(), fs.abs())
    conflict = torch.where(
        absolute > eps,
        (0.5 * (1 - dot / absolute.clamp_min(eps))).clamp(0, 1),
        torch.full_like(dot, 0.5)
    )
    return cosine, conflict


@torch.no_grad()
def _fixed_pool(scores, max_source: int):
    """Top / bottom / evenly spread sources; all policies share this pool."""
    n = scores.numel()
    if n <= max_source:
        return torch.arange(n, device=scores.device)
    n_extreme = max(1, max_source // 3)
    order = scores.argsort()
    high = order[-n_extreme:]
    low = order[:n_extreme]
    regular = torch.linspace(0, n - 1, max_source, device=scores.device).round().long()
    combined = torch.cat((low, high, regular)).unique(sorted=True)
    # Keep the extreme samples; optionally thin the middle of the pool.
    if combined.numel() > max_source:
        extreme = torch.cat((low, high)).unique(sorted=True)
        middle = regular[~torch.isin(regular, extreme)].unique(sorted=True)
        remain = max_source - extreme.numel()
        if remain > 0 and middle.numel() > remain:
            select = torch.linspace(
                0, middle.numel()-1, remain, device=scores.device
            ).round().long()
            middle = middle[select]
        combined = torch.cat((extreme, middle[:max(0, remain)])).unique(sorted=True)
    return combined


@torch.no_grad()
def _selected_weights(cosine, conflict, sign_lambda: float, temperature: float):
    """Seven candidate distributions, same source/target sample support."""
    if not 0 <= sign_lambda <= 100:
        raise ValueError("Invalid sign_lambda")
    n = cosine.numel()
    if n < 8:
        raise ValueError("Need >=8 sampled source sensor/window points")
    k = max(1, n // 4)
    top = cosine.topk(k).indices
    bottom = cosine.topk(k, largest=False).indices
    # D2 cosine-stratified comparison: within each cosine quartile
    # select equally many low/high-conflict samples. This suppresses
    # confounding by the original F2 cosine score (not an exact match).
    cosine_bins = torch.tensor_split(cosine.argsort(), 4)
    low_parts, high_parts = [], []
    for bin_ids in cosine_bins:
        if len(bin_ids) < 2:
            continue
        rank = bin_ids[conflict[bin_ids].argsort()]
        m = max(1, len(rank) // 4)
        low_parts.append(rank[:m])
        high_parts.append(rank[-m:])
    low_sign = torch.cat(low_parts)
    high_sign = torch.cat(high_parts)

    def subset(indices):
        # Strictly positive mass for GeomLoss Sinkhorn log-weights;
        # 5% uniform floor, 95% allocated to the chosen stratum.
        vec = torch.full_like(cosine, 0.05 / n)
        vec[indices] += 0.95 / len(indices)
        return vec

    f2 = torch.softmax(temperature * cosine, dim=0)
    sign = torch.softmax(temperature * cosine - sign_lambda * conflict, dim=0)
    output = {
        "uniform": torch.full_like(cosine, 1 / n),
        "f2": f2,
        "sign": sign,
        "cos_high": subset(top),
        "cos_low": subset(bottom),
        "sign_low_cosmatched": subset(low_sign),
        "sign_high_cosmatched": subset(high_sign)
    }
    chosen = {
        "cos_high": top, "cos_low": bottom,
        "sign_low_cosmatched": low_sign,
        "sign_high_cosmatched": high_sign
    }
    return output, chosen


@torch.no_grad()
def _pearson(a, b):
    x = a.float() - a.float().mean()
    y = b.float() - b.float().mean()
    return float(((x*y).sum() / (x.norm()*y.norm()+1e-12)).item())


@torch.no_grad()
def _metrics(gradient, probe_gradient, uniform_gradient=None):
    g = gradient.detach().float()
    t = probe_gradient.detach().float()
    dot = float((t*g).sum().item())
    tnorm = t.norm()
    gnorm = g.norm()
    result = {
        "ot_probe_dot": dot,
        "ot_probe_cos": float((t*g).sum().div(tnorm*gnorm+1e-12).item()),
        "ot_grad_norm": float(gnorm.item()),
        "probe_grad_norm": float(tnorm.item()),
        "linear_gain_unit_lr": dot,
    }
    if uniform_gradient is not None:
        d = g - uniform_gradient.detach().float()
        result["gain_vs_uniform_unit_lr"] = float((t*d).sum().item())
        result["grad_shift_vs_uniform_norm"] = float(d.norm().item())
    return result


def audit_f2_source_utility(*, source_rep, source_future, source_features,
                            source_delta, target_fit_rep, target_fit_future,
                            target_fit_features, target_fit_delta,
                            probe_gradient, shared_parameter, criterion,
                            ot_scale: float, xy_scale: float, epoch: int,
                            city: str, target_group: int,
                            source_budget: int = 192,
                            target_budget: int = 96,
                            sign_lambda: float = 1.5,
                            temperature: float = 8.0):
    """Run D1 and D2 counterfactuals on a common, small active OT support.

    source_rep and target_fit_rep HAVE the real model autograd graph.
    Futures and output-head deltas are observed SUPPORT data, not test labels.
    No parameter mutation or backward() is executed by this function.
    """
    if source_budget < 8 or target_budget < 2:
        raise ValueError("Source and target budgets too small")
    if xy_scale <= 0 or ot_scale <= 0 or not math.isfinite(ot_scale):
        raise ValueError("Expected positive OT scale and future feature scale")
    if temperature <= 0:
        raise ValueError("Positive F2 temperature required")
    if probe_gradient is None:
        raise ValueError("Missing target probe gradient")

    ns = source_rep.shape[0]
    nt = target_fit_rep.shape[0]
    if (source_rep.ndim != 2 or target_fit_rep.ndim != 2 or
            ns != source_features.reshape(-1, source_features.shape[-1]).shape[0] or
            nt != target_fit_features.reshape(-1, target_fit_features.shape[-1]).shape[0] or
            source_future.shape != (ns, 36) or
            target_fit_future.shape != (nt, 36) or
            ns < 8 or nt < 2):
        raise ValueError("Input shape mismatch or insufficient OT points")

    cos_all, conflict_all = source_gradient_scores(
        source_features, source_delta, target_fit_features, target_fit_delta
    )
    with torch.no_grad():
        source_ids = _fixed_pool(cos_all, source_budget)
        target_ids = torch.linspace(
            0, nt-1, min(nt, target_budget), device=target_fit_rep.device
        ).round().long().unique(sorted=True)
        cos = cos_all[source_ids]
        conflict = conflict_all[source_ids]
        weights, sets = _selected_weights(cos, conflict, sign_lambda, temperature)
        report = {
            "audit_version": "d1d2-v1",
            "epoch": int(epoch),
            "city": str(city),
            "target_group": int(target_group),
            "source_total": int(ns),
            "source_batch_windows": int(source_features.shape[0]),
            "source_node_count": int(source_features.shape[1]),
            "source_subset": int(source_ids.numel()),
            "target_fit_windows": int(target_fit_features.shape[0]),
            "target_node_count": int(target_fit_features.shape[1]),
            "target_fit_total": int(nt),
            "target_fit_subset": int(target_ids.numel()),
            "source_subset_flat_indices": source_ids.cpu().tolist(),
            "target_fit_subset_flat_indices": target_ids.cpu().tolist(),
            "source_cos_mean": float(cos.mean().item()),
            "source_conflict_mean": float(conflict.mean().item()),
            "cos_conflict_pearson": _pearson(cos, conflict),
            "candidate_groups": {},
            "candidate_OT": {},
            "f2_temperature": float(temperature),
            "d2_conflict_lambda": float(sign_lambda),
        }
        for key, inds in sets.items():
            report["candidate_groups"][key] = {
                "source_subset_positions": inds.cpu().tolist(),
                "source_flat_indices": source_ids[inds].cpu().tolist(),
                "mean_cos": float(cos[inds].mean().item()),
                "mean_conflict": float(conflict[inds].mean().item())
            }

    # Re-use the SAME source, target, original OT geometry, and coefficient.
    # Each candidate changes only source alpha; beta remains uniform.
    x = torch.cat(
        [source_rep.index_select(0, source_ids),
         source_future.detach().index_select(0, source_ids) * xy_scale],
        dim=-1
    )
    y = torch.cat(
        [target_fit_rep.index_select(0, target_ids),
         target_fit_future.detach().index_select(0, target_ids) * xy_scale],
        dim=-1
    )
    beta = x.new_full((len(target_ids),), 1.0/len(target_ids))
    gradients = {}
    for name, alpha in weights.items():
        ot_loss = (criterion(alpha, x, beta, y).sum() * ot_scale)
        grad = torch.autograd.grad(
            ot_loss, shared_parameter, retain_graph=True,
            allow_unused=False, create_graph=False
        )[0].detach()
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError("Nonfinite diagnostic OT gradient: "+name)
        gradients[name] = grad
    uniform = gradients["uniform"]
    for name, grad in gradients.items():
        report["candidate_OT"][name] = _metrics(grad, probe_gradient, uniform)
    report["contrasts"] = {
        "D1_cos_high_minus_low_gain": (
            report["candidate_OT"]["cos_high"]["linear_gain_unit_lr"] -
            report["candidate_OT"]["cos_low"]["linear_gain_unit_lr"]
        ),
        "D2_sign_low_minus_high_gain_given_mid_cos": (
            report["candidate_OT"]["sign_low_cosmatched"]["linear_gain_unit_lr"] -
            report["candidate_OT"]["sign_high_cosmatched"]["linear_gain_unit_lr"]
        ),
        "D2_sign_minus_f2_gain": (
            report["candidate_OT"]["sign"]["linear_gain_unit_lr"] -
            report["candidate_OT"]["f2"]["linear_gain_unit_lr"]
        ),
        "D1_f2_minus_uniform_gain": (
            report["candidate_OT"]["f2"]["linear_gain_unit_lr"] -
            report["candidate_OT"]["uniform"]["linear_gain_unit_lr"]
        ),
    }
    return report
