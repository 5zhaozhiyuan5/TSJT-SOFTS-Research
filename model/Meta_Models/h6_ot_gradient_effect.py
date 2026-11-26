"""H6 read-only real-GeomLoss OT strength/city gradient counterfactuals.

Replays the ORIGINAL F2 Stage-I all-parameter one-sided PCGrad + optimizer-state
AdamW shadow. Does not use probe futures in any policy gradient or intervention.
No optimizer.step, model parameter or optimizer state mutation, no training.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

try:
    from .h1_shadow_audit import _grads, projected_directions, simulate_adamw_step
    from .h2_candidate_audit import sum_gradient_vectors
    from .h3_city_counterfactual import evaluate_probes, vector_alignment
except ImportError:
    from h1_shadow_audit import _grads, projected_directions, simulate_adamw_step
    from h2_candidate_audit import sum_gradient_vectors
    from h3_city_counterfactual import evaluate_probes, vector_alignment


SCALES = (0.0, 0.25, 0.5, 1.0)


def gradient_tuple_scaled(grads, factor):
    if not math.isfinite(float(factor)):
        raise ValueError("OT factor must be finite")
    return [x * float(factor) for x in grads]


def gradient_tuple_difference(a, b):
    if len(a) != len(b):
        raise ValueError("Gradient tuple length mismatch")
    return [x-y for x,y in zip(a,b)]


@torch.no_grad()
def gradient_norm(grads):
    return math.sqrt(sum(float(x.detach().float().square().sum()) for x in grads))


def make_h6_policies(city_entries):
    """F2, no OT, global 0.25/0.5, and each city's 0/0.5 OT.

    IMPORTANT: only scales the precomputed *real original F2 OT gradients*,
    not alpha/beta, the Sinkhorn cost, or the gradient selection function.
    """
    if len(city_entries) != 3:
        raise ValueError("H6 requires exactly three source cities")
    cities = [str(x["city"]) for x in city_entries]
    if len(set(cities)) != 3:
        raise ValueError("Source-city names must be distinct")
    n = len(city_entries[0]["f2"])
    if not n or any(len(e["f2"]) != n for e in city_entries):
        raise ValueError("Inconsistent OT gradient tuples")
    full=sum_gradient_vectors([x["f2"] for x in city_entries])
    policies={
        "f2":full,
        "all_off":gradient_tuple_scaled(full,0.0),
        "global_quarter":gradient_tuple_scaled(full,0.25),
        "global_half":gradient_tuple_scaled(full,0.5),
    }
    for entry in city_entries:
        city=entry["city"]
        policies[f"city_off:{city}"]=gradient_tuple_difference(
            full,entry["f2"]
        )
        policies[f"city_half:{city}"]=gradient_tuple_difference(
            full,gradient_tuple_scaled(entry["f2"],0.5)
        )
    policies["identity"]=full  # replay noise control, bitwise identical OT tuple
    return policies


@torch.no_grad()
def _parameter_displacement(shadow, named_params):
    return [shadow[name]-p.detach() for name,p in named_params]


@torch.no_grad()
def _step_difference(a, b, named_params):
    return [a[name]-b[name] for name,_ in named_params]


def shadow_h6_audit(*,model,optimizer,source_task_loss,target_fit_task_loss,
                    city_entries,target_data,probe_mask,sample_index,
                    checkpoint_epoch,beta=5e-4):
    """One-step 11-shadow controlled audit; returns scores AND optimizer traces.

    Fit/probe windows must be specified before call; probe_mask is used only
    after all policy OT grads are computed. Before returning, source model
    parameters, optimizer, train/eval mode, and buffers are unchanged.
    """
    policies=make_h6_policies(city_entries)
    named=list(model.named_parameters())
    params=[p for _,p in named]
    if not named or probe_mask.dtype != torch.bool:
        raise ValueError("Expected nonempty model and Boolean probe mask")
    source_grad=_grads(source_task_loss,params)
    target_grad=_grads(target_fit_task_loss,params)
    raw_gradient=policies["f2"]
    _,source_pc=projected_directions(
        source_grad,target_grad,policies["all_off"]
    )
    _,f2_pc=projected_directions(source_grad,target_grad,raw_gradient)
    effective=gradient_tuple_difference(f2_pc,source_pc)
    raw_plus_source=[
        a+b for a,b in zip(source_grad,raw_gradient)
    ]
    grad_report={
        "source_to_target_fit":vector_alignment(source_grad,target_grad),
        "total_ot_to_target_fit":vector_alignment(raw_gradient,target_grad),
        "total_ot_to_source":vector_alignment(raw_gradient,source_grad),
        "raw_source_plus_ot_to_target_fit":
            vector_alignment(raw_plus_source,target_grad),
        "projected_source_to_target_fit":
            vector_alignment(source_pc,target_grad),
        "projected_source_plus_ot_to_target_fit":
            vector_alignment(f2_pc,target_grad),
        "effective_ot_after_pcgrad_vs_target":
            vector_alignment(effective,target_grad),
        "effective_ot_after_pcgrad_vs_raw_ot":
            vector_alignment(effective,raw_gradient),
        "raw_ot_norm":gradient_norm(raw_gradient),
        "effective_ot_after_pcgrad_norm":gradient_norm(effective),
        "source_gradient_norm":gradient_norm(source_grad),
        "target_fit_gradient_norm":gradient_norm(target_grad),
        "city":{}
    }
    for entry in city_entries:
        city=entry["city"]
        off_grad=policies[f"city_off:{city}"]
        _,city_off_pc=projected_directions(
            source_grad,target_grad,off_grad
        )
        effect=gradient_tuple_difference(f2_pc,city_off_pc)
        grad_report["city"][city]={
            "raw_ot_vs_target_fit":
                vector_alignment(entry["f2"],target_grad),
            "raw_ot_vs_source":
                vector_alignment(entry["f2"],source_grad),
            "conditional_after_pcgrad_vs_target_fit":
                vector_alignment(effect,target_grad),
            "conditional_after_pcgrad_vs_raw_city_ot":
                vector_alignment(effect,entry["f2"]),
            "raw_ot_norm":gradient_norm(entry["f2"]),
            "conditional_pcgrad_effect_norm":gradient_norm(effect),
        }
    old_truth=torch.as_tensor(target_data.y,dtype=params[0].dtype,
                              device=params[0].device)
    if probe_mask.numel()!=old_truth.shape[0] or not bool(probe_mask.any()):
        raise ValueError("Invalid probe windows")
    # The model sees history only in shadow forward: probe future replaced.
    blind=SimpleNamespace(
        x=target_data.x,y=torch.zeros_like(old_truth),
        means=target_data.means,stds=target_data.stds
    )
    probe_truth=old_truth[probe_mask].detach()
    saved_training={name:m.training for name,m in model.named_modules()}
    saved_dynamic={k:(hasattr(model,k),getattr(model,k,None))
                   for k in ("RevIn_mean","RevIn_std","eps")}
    original_buffers={k:b.detach().clone() for k,b in model.named_buffers()}
    device=params[0].device
    cuda_devices=([device.index if device.index is not None
                   else torch.cuda.current_device()] if device.type=="cuda" else [])
    from torch.func import functional_call
    scores={};step_diagnostics={};shadows={}
    try:
        with torch.no_grad(),torch.random.fork_rng(devices=cuda_devices):
            # Build both reference states first, then one candidate at a time.
            for name in ("f2","all_off"):
                _,direction=projected_directions(source_grad,target_grad,
                                                  policies[name])
                shadows[name]=simulate_adamw_step(named,optimizer,direction)

            base_step=_parameter_displacement(shadows["all_off"],named)
            f2_step=_parameter_displacement(shadows["f2"],named)
            step_diagnostics["f2_vs_no_ot"]={
                "parameter_delta":vector_alignment(
                    _step_difference(shadows["f2"],shadows["all_off"],named),
                    gradient_tuple_scaled(target_grad,-1)
                ),
                "no_ot_step_vs_target_descent":
                    vector_alignment(base_step,gradient_tuple_scaled(target_grad,-1)),
                "f2_step_vs_target_descent":
                    vector_alignment(f2_step,gradient_tuple_scaled(target_grad,-1)),
            }
            for name in policies:
                if name in shadows:
                    shadow=shadows[name]
                else:
                    _,direction=projected_directions(
                        source_grad,target_grad,policies[name]
                    )
                    shadow=simulate_adamw_step(named,optimizer,direction)
                # Each policy runs the identical stochastic SOFTS pooling draw.
                torch.manual_seed(681947+int(sample_index))
                new_buffers={k:v.clone() for k,v in original_buffers.items()}
                output,*_=functional_call(
                    model,(shadow,new_buffers),(blind,),{"stage":"train"}
                )
                scores[name]=evaluate_probes(
                    output[probe_mask].detach(),probe_truth,beta
                )
                delta_from_zero=_step_difference(
                    shadow,shadows["all_off"],named
                )
                step_diagnostics[name]={
                    "vs_no_ot_step":vector_alignment(
                        delta_from_zero,gradient_tuple_scaled(target_grad,-1)
                    ),
                    "step_norm":gradient_norm(
                        _parameter_displacement(shadow,named)
                    ),
                    "raw_policy_ot_norm":gradient_norm(policies[name]),
                }
                del output,new_buffers,delta_from_zero
                if name not in ("f2","all_off"):
                    del shadow
            # Replayed F2 should be numerically equal to identity.
            for entry in city_entries:
                city=entry["city"]
                off=f"city_off:{city}"
                _,s=projected_directions(
                    source_grad,target_grad,policies[off]
                )
                # Step counterpart of the *conditional* city contribution.
                off_shadow=simulate_adamw_step(named,optimizer,s)
                city_step_delta=_step_difference(shadows["f2"],off_shadow,named)
                step_diagnostics[off]["city_f2_minus_city_off_step"]={
                    **vector_alignment(
                        city_step_delta,gradient_tuple_scaled(target_grad,-1)
                    )
                }
                del off_shadow,city_step_delta
    finally:
        modules=dict(model.named_modules())
        for name,flag in saved_training.items():
            modules[name].training=flag
        for key,(exists,value) in saved_dynamic.items():
            if exists:
                setattr(model,key,value)
            elif hasattr(model,key):
                delattr(model,key)
        # torch.func.functional_call copies supplied buffers, not the model's.
        # Explicitly restore in case an implementation changes module internals.
        with torch.no_grad():
            for k,v in original_buffers.items():
                current=dict(model.named_buffers())[k]
                if not torch.equal(current,v):
                    current.copy_(v)
    baseline=scores["f2"]
    gains={}
    for name,sc in scores.items():
        gains[name]={
            "gain_mae":baseline["mae"]-sc["mae"],
            "gain_smooth_l1":baseline["smooth_l1"]-sc["smooth_l1"],
            "gain_bands":[a-b for a,b in zip(
                baseline["band_smooth_l1"],sc["band_smooth_l1"])
            ],
            "gain_mae_at_minutes":{
                t:baseline["mae_at_minutes"][t]-sc["mae_at_minutes"][t]
                for t in ("10","60","120","180")
            }
        }
    return {
        "audit_version":"h6-v1",
        "checkpoint_epoch":int(checkpoint_epoch),
        "batch_index":int(sample_index),
        "n_shadow_candidates":len(scores),
        "training_updates":0,"stage2_runs":0,
        "fit_window_indices":(~probe_mask).nonzero().flatten().cpu().tolist(),
        "probe_window_indices":probe_mask.nonzero().flatten().cpu().tolist(),
        "gradient_report":grad_report,
        "step_diagnostics":step_diagnostics,
        "f2_baseline":baseline,
        "candidate_scores":scores,
        "gains":gains,
    }
