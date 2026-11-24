"""G1-G3: low-cost counterfactual OT utility probe on PMME's shared readout.

Not DVGP, not a source/target domain classifier, and not a learned OT solver.

The proxy changes only projection1.weight in a VIRTUAL one-step update. The
candidate OT gradients come from the REAL GeomLoss OT operator on a small
deterministically chosen source/target subset, with encoder outputs frozen.
The virtual delta includes PMME's one-sided gradient projection, but not the
full AdamW update, encoder update or interactions between all source cities.
Thus 'gain' means measured loss change under this specified local proxy,
NOT realized validation/test benefit.

The target probe uses only the three-day target SUPPORT batch. It is held
out from this proxy's small OT subset but is not an independent chronological
validation set, since the original PMME training sees these support windows.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


RHO_CHOICES = (0.0, 0.5, 1.0)


def _indices(n: int, k: int, device):
    """Evenly spaced deterministic sample, independent of random number state."""
    if n <= 0 or k <= 0:
        raise ValueError("Empty counterfactual OT subset")
    count = min(n, k)
    if count == n:
        return torch.arange(n, device=device)
    return torch.linspace(0, n - 1, count, device=device).round().long()


def _pcgrad_one_sided(g, target, eps=1e-8):
    dot = (g * target).sum()
    if dot < 0:
        return g - dot / (target.square().sum() + eps) * target
    return g


def blend_source_weights(f2_alpha, rho: float):
    if not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0,1]")
    n = f2_alpha.numel()
    uniform = torch.full_like(f2_alpha, 1.0 / n)
    return ((1.0 - rho) * uniform + rho * f2_alpha).detach()


def _probe_loss(prediction, target, start=0, end=36):
    return F.smooth_l1_loss(
        prediction[..., start:end], target[..., start:end],
        beta=5e-4, reduction="mean"
    )


def _proxy_virtual_gain(ot_head_grad, task_head_grad, target_head_grad,
                        probe_input, probe_prediction, probe_target,
                        probe_physical_scale, projection2_weight,
                        step_size, max_step_norm):
    """Exact change of fixed-encoder probe forecast under head-only delta W.

    The additional projected OT gradient is computed relative to the same
    source-task-only projected update; probe loss is measured at current
    parameters, not after the task-only update. This is a local marginal
    counterfactual, NOT an exact replay of AdamW and the full backbone.
    """
    with torch.no_grad():
        g_task = task_head_grad.float()
        g_target = target_head_grad.float()
        g_ot = ot_head_grad.float()
        delta_g = (_pcgrad_one_sided(g_task + g_ot, g_target) -
                   _pcgrad_one_sided(g_task, g_target))
        update = -step_size * delta_g
        update_norm = update.norm()
        if not bool(torch.isfinite(update_norm)):
            return None
        if update_norm > max_step_norm:
            update = update * (max_step_norm / (update_norm + 1e-12))
        # projection1: [128,512], projection2: [36,128]
        # forecast_delta = projection2.weight @ (update @ encoder_output).
        hidden_delta = torch.einsum(
            "bnd,rd->bnr", probe_input.float(), update
        )
        forecast_delta = torch.einsum(
            "bnr,hr->bnh", hidden_delta, projection2_weight.float()
        )
        forecast_delta = forecast_delta * probe_physical_scale.float()
        base_pred = probe_prediction.float()
        true_y = probe_target.float()
        shifted = base_pred + forecast_delta
        base_loss = _probe_loss(base_pred, true_y)
        candidate_loss = _probe_loss(shifted, true_y)
        gains = [
            float((_probe_loss(base_pred, true_y, left, right) -
                   _probe_loss(shifted, true_y, left, right)).item())
            for left, right in ((0, 12), (12, 24), (24, 36))
        ]
        return {
            "gain": float((base_loss - candidate_loss).item()),
            "bands": gains,
            "update_norm": float(update.norm().item()),
            "ot_grad_norm": float(g_ot.norm().item()),
            "ot_target_cosine": float(
                ((g_ot * g_target).sum() /
                 (g_ot.norm() * g_target.norm() + 1e-12)).item()
            ),
        }


def evaluate_feedback_candidates(
    *, source_rep, target_rep, source_y, target_y,
    source_pre, target_pre, f2_alpha, target_probe_pre,
    target_probe_output, target_probe_y, target_probe_scale,
    projection2_weight, task_head_grad, target_head_grad,
    criterion, ot_coefficient, xy_scale,
    n_source=256, n_target=128,
    step_size=0.05, max_step_norm=0.02,
    risk_penalty=1.0,
):
    """Compute F2-based counterfactual candidate utilities for one source city.

    source/target_rep are [N,128], pre-projection1 inputs [N,512].
    source/target_y are [N,36] RevIN-normalized futures.
    target_pre must exclude target probe *windows* by caller mask.

    Returns only Python scalars and strings; does not mutate any parameters,
    optimizer buffers, forward-mode flags, torch RNG or existing .grad fields.
    """
    if source_rep.ndim != 2 or target_rep.ndim != 2:
        raise ValueError("Expected flattened sample-node representations")
    if source_pre.shape[0] != source_rep.shape[0]:
        raise ValueError("Source pre-head shape mismatch")
    if target_pre.shape[0] != target_rep.shape[0]:
        raise ValueError("Target pre-head shape mismatch")
    if not (len(f2_alpha) == len(source_rep)):
        raise ValueError("F2 source weight size mismatch")
    if ot_coefficient <= 0 or step_size <= 0 or max_step_norm <= 0:
        raise ValueError("Positive OT coefficient, virtual step, norm cap required")
    if not all(math.isfinite(v) for v in
               (ot_coefficient, step_size, max_step_norm, risk_penalty)):
        raise ValueError("Nonfinite counterfactual configuration")

    src_indices = _indices(len(source_rep), n_source, source_rep.device)
    tgt_indices = _indices(len(target_rep), n_target, target_rep.device)

    # Freeze shared encoder: autograd is needed ONLY for the local proxy OT
    # with respect to projected representations, not the model itself.
    with torch.enable_grad():
        rs = source_rep.detach().index_select(0, src_indices).float().requires_grad_(True)
        rt = target_rep.detach().index_select(0, tgt_indices).float().requires_grad_(True)
        ys = source_y.detach().index_select(0, src_indices).float() * xy_scale
        yt = target_y.detach().index_select(0, tgt_indices).float() * xy_scale
        in_s = source_pre.detach().index_select(0, src_indices).float()
        in_t = target_pre.detach().index_select(0, tgt_indices).float()
        beta = torch.full((len(rt),), 1.0 / len(rt), device=rt.device)
        results = []
        for rho in RHO_CHOICES:
            # Rebuild the pairwise OT graph for each candidate.
            x = torch.cat([rs, ys], dim=-1)
            y = torch.cat([rt, yt], dim=-1)
            alpha_full = blend_source_weights(f2_alpha.detach(), rho)
            alpha = alpha_full.index_select(0, src_indices)
            alpha = alpha / alpha.sum().clamp_min(1e-12)
            loss = criterion(alpha, x, beta, y).sum() * ot_coefficient
            grad_s, grad_t = torch.autograd.grad(
                loss, (rs, rt), retain_graph=False, create_graph=False
            )
            g_head = grad_s.detach().T @ in_s + grad_t.detach().T @ in_t
            measured = _proxy_virtual_gain(
                g_head, task_head_grad, target_head_grad,
                target_probe_pre, target_probe_output, target_probe_y,
                target_probe_scale, projection2_weight,
                step_size=step_size, max_step_norm=max_step_norm
            )
            if measured is None:
                raise FloatingPointError("Nonfinite counterfactual virtual update")
            measured["rho"] = rho
            measured["horizon_score"] = (
                measured["gain"] -
                risk_penalty * max(0.0, -min(measured["bands"]))
            )
            results.append(measured)
    # "off" is the no-OT reference, which has zero marginal benefit.
    results.append({
        "rho": "off", "gain": 0.0, "bands": [0.0, 0.0, 0.0],
        "horizon_score": 0.0, "update_norm": 0.0, "ot_grad_norm": 0.0,
        "ot_target_cosine": 0.0,
    })
    return results


def select_feedback_policy(results, mode: str, tolerance=1e-8):
    """G2 picks max average gain; G3 adds negative-band risk penalty.

    A candidate must beat 'off' by tolerance, otherwise no source-specific
    OT update is used. All choices use identical candidate budgets.
    """
    if mode not in ("g2", "g3"):
        raise ValueError("Unsupported feedback selection mode " + str(mode))
    if not results:
        raise ValueError("No candidate results")
    metric = "gain" if mode == "g2" else "horizon_score"
    candidates = [r for r in results if r["rho"] != "off"]
    viable = [r for r in candidates if r[metric] > tolerance]
    if not viable:
        return "off"
    return max(viable, key=lambda r: r[metric])["rho"]
