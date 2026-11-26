"""H3 read-only city counterfactual (real OT, full-parameter PCGrad + AdamW).

Seven main variants: F2 for all sources, each city's F2 replaced by uniform
(three variants), and each city's OT disabled (three variants).
An eighth, no-OT, reference is reported separately.

All OT gradients use exact GeomLoss on identical full source/target-fit sets.
Probe labels are NOT involved in alpha, beta, OT geometry or optimizer
directions; only the shadow prediction evaluation reads probe labels.
Probe belongs to previously seen three-day support: NOT independent testing.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

try:
    from .h1_shadow_audit import projected_directions, simulate_adamw_step, _grads
    from .h2_candidate_audit import sum_gradient_vectors
except ImportError:
    from h1_shadow_audit import projected_directions, simulate_adamw_step, _grads
    from h2_candidate_audit import sum_gradient_vectors


def masked_support_split(n_windows, device, stride=5):
    """Split whole windows, never individual sensor nodes."""
    if n_windows < 15 or stride < 2:
        raise ValueError("Need at least 15 target windows and stride>=2")
    indices = torch.arange(n_windows, device=device)
    probe = indices % stride == 0
    fit = ~probe
    if not bool(probe.any()) or not bool(fit.any()):
        raise ValueError("Empty target fit/probe split")
    return fit, probe


def make_policy_grads(city_entries):
    """Seven candidate OT-gradient tuples plus the all-off reference."""
    if len(city_entries) != 3:
        raise ValueError("Exactly 3 cities required")
    names = [entry["city"] for entry in city_entries]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate city name")
    for entry in city_entries:
        if "f2" not in entry or "uniform" not in entry:
            raise ValueError("Both OT variants required")
        if len(entry["f2"]) != len(entry["uniform"]):
            raise ValueError("Gradient parameter dimension mismatch")
    full = sum_gradient_vectors([entry["f2"] for entry in city_entries])
    candidates = {"f2": full}
    for i, entry in enumerate(city_entries):
        candidates["city_uniform:" + entry["city"]] = sum_gradient_vectors([
            city["uniform"] if j == i else city["f2"]
            for j, city in enumerate(city_entries)
        ])
        candidates["city_off:" + entry["city"]] = sum_gradient_vectors([
            city["f2"] for j, city in enumerate(city_entries) if j != i
        ])
    candidates["all_off"] = [torch.zeros_like(x) for x in full]
    return candidates


@torch.no_grad()
def vector_alignment(a, b):
    if len(a) != len(b):
        raise ValueError("Gradient tuple mismatch")
    dot = sum(float((x.float()*y.float()).sum().item())
              for x,y in zip(a,b))
    a2 = sum(float(x.float().square().sum().item()) for x in a)
    b2 = sum(float(y.float().square().sum().item()) for y in b)
    return {
        "dot": dot,
        "cos": dot / (math.sqrt(a2*b2)+1e-15),
        "norm_a": math.sqrt(a2), "norm_b": math.sqrt(b2),
    }


@torch.no_grad()
def evaluate_probes(pred, truth, beta=5e-4):
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[-1] != 36:
        raise ValueError("Expected [probe_windows,nodes,36] predictions and labels")
    if not bool(torch.isfinite(pred).all()):
        raise FloatingPointError("Nonfinite shadow prediction")
    err = pred.float()-truth.float()
    h_mae = err.abs().mean(dim=(0,1))
    h_rmse = err.square().mean(dim=(0,1)).sqrt()
    return {
        "smooth_l1": float(F.smooth_l1_loss(pred.float(),truth.float(),beta=beta)),
        "mae": float(err.abs().mean()),
        "rmse": float(err.square().mean().sqrt()),
        "band_smooth_l1": [
            float(F.smooth_l1_loss(pred[...,a:b].float(),
                                   truth[...,a:b].float(),beta=beta))
            for a,b in ((0,12),(12,24),(24,36))
        ],
        "mae_at_minutes": {
            str(m): float(h_mae[i])
            for m,i in ((10,1),(60,11),(120,23),(180,35))
        },
        "rmse_at_minutes": {
            str(m): float(h_rmse[i])
            for m,i in ((10,1),(60,11),(120,23),(180,35))
        },
    }


def shadow_city_audit(*, model, optimizer, source_task_loss,
                      target_fit_task_loss, city_entries, target_data,
                      probe_mask, epoch, sample_index, beta=5e-4):
    """Evaluate the actual one-step PCGrad+AdamW policy counterfactuals."""
    named = list(model.named_parameters())
    params = [p for _,p in named]
    if not params:
        raise ValueError("Model has no parameters")
    if probe_mask.dtype != torch.bool:
        raise ValueError("probe mask must be Boolean")
    truth = torch.as_tensor(
        target_data.y, dtype=params[0].dtype,device=params[0].device
    )
    if len(probe_mask) != len(truth) or not bool(probe_mask.any()):
        raise ValueError("Invalid target probe mask")
    # Force the shadow forward to see zero labels. Forecast path reads X only.
    blind_target = SimpleNamespace(
        x=target_data.x,
        y=torch.zeros_like(truth),
        means=target_data.means,
        stds=target_data.stds,
    )
    probe_truth = truth[probe_mask].detach()
    source_grad = _grads(source_task_loss,params)
    target_fit_grad = _grads(target_fit_task_loss,params)
    policies = make_policy_grads(city_entries)
    report = {
        "audit_version":"h3-v1",
        "epoch_trigger":int(epoch),
        "probe_batch_index":int(sample_index),
        "probe_window_indices":probe_mask.nonzero().flatten().cpu().tolist(),
        "fit_window_indices":(~probe_mask).nonzero().flatten().cpu().tolist(),
        "city_names":[c["city"] for c in city_entries],
        "training_updates":0,
        "candidate_scores":{},
        "gradient_diagnostics":{},
    }
    for entry in city_entries:
        report["gradient_diagnostics"][entry["city"]] = {
            "f2_vs_fit_target":vector_alignment(entry["f2"],target_fit_grad),
            "uniform_vs_fit_target":vector_alignment(entry["uniform"],target_fit_grad),
        }
    report["gradient_diagnostics"]["source_vs_fit_target"] = vector_alignment(
        source_grad,target_fit_grad
    )
    saved_training = {n:m.training for n,m in model.named_modules()}
    dynamic_names = ("RevIn_mean","RevIn_std","eps")
    saved_dynamic = {a:(hasattr(model,a),getattr(model,a,None))
                     for a in dynamic_names}
    buffers = {n:b.detach().clone() for n,b in model.named_buffers()}
    device = params[0].device
    cuda_devices = ([device.index if device.index is not None
                     else torch.cuda.current_device()] if device.type=="cuda" else [])
    from torch.func import functional_call
    try:
        with torch.no_grad(),torch.random.fork_rng(devices=cuda_devices):
            for name,ot_grads in policies.items():
                _,projected = projected_directions(
                    source_grad,target_fit_grad,ot_grads
                )
                shadow_params = simulate_adamw_step(named,optimizer,projected)
                # All candidates see identical stochastic pooling draws.
                torch.manual_seed(310079+int(sample_index))
                shadow_buffers = {n:v.clone() for n,v in buffers.items()}
                prediction,*_ = functional_call(
                    model,(shadow_params,shadow_buffers),
                    (blind_target,),{"stage":"train"}
                )
                report["candidate_scores"][name] = evaluate_probes(
                    prediction[probe_mask].detach(),probe_truth,beta
                )
                del projected,shadow_params,shadow_buffers,prediction
    finally:
        modules = dict(model.named_modules())
        for n,state in saved_training.items():
            modules[n].training = state
        for a,(present,value) in saved_dynamic.items():
            if present:
                setattr(model,a,value)
            elif hasattr(model,a):
                delattr(model,a)
    f2 = report["candidate_scores"]["f2"]
    report["gain_vs_f2"] = {}
    for name,score in report["candidate_scores"].items():
        report["gain_vs_f2"][name] = {
            "smooth_l1":f2["smooth_l1"]-score["smooth_l1"],
            "mae":f2["mae"]-score["mae"],
            "bands":[x-y for x,y in zip(
                f2["band_smooth_l1"],score["band_smooth_l1"])],
            "horizons_10_60_120_180_mae":{
                key:f2["mae_at_minutes"][key]-score["mae_at_minutes"][key]
                for key in ("10","60","120","180")
            },
        }
    report["best_by_probe_smooth_l1"] = min(
        report["candidate_scores"],
        key=lambda name: report["candidate_scores"][name]["smooth_l1"]
    )
    return report
