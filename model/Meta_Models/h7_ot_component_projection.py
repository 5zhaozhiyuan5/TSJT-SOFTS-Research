"""H7: checkpoint-only target-FIT OT-gradient projection counterfactuals.

This module is a DIAGNOSTIC of existing PCGrad ideas, not a novel algorithm.
Original F2 alpha/beta, Sinkhorn geometry, source gradient, optimizer state and
Stage-II are unchanged. Target PROBE labels are used only AFTER policy updates,
not in projection. All gradients are full-parameter, no optimizer.step().
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

try:
    from .h1_shadow_audit import _grads, projected_directions, simulate_adamw_step
    from .h2_candidate_audit import sum_gradient_vectors
    from .h3_city_counterfactual import evaluate_probes, vector_alignment
    from .h6_ot_gradient_effect import gradient_tuple_scaled, gradient_tuple_difference, gradient_norm
except ImportError:
    from h1_shadow_audit import _grads, projected_directions, simulate_adamw_step
    from h2_candidate_audit import sum_gradient_vectors
    from h3_city_counterfactual import evaluate_probes, vector_alignment
    from h6_ot_gradient_effect import gradient_tuple_scaled, gradient_tuple_difference, gradient_norm


# Exactly six preregistered policies plus an identical-F2 numerical control.
POLICY_NAMES = (
    "f2", "global_half", "city_off:shenzhen",
    "global_projection", "city_projection", "city_projection_half",
    "identity",
)


@torch.no_grad()
def dot_tuple(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("Empty or differently shaped gradients")
    out = None
    for x, y in zip(a, b):
        if x.shape != y.shape:
            raise ValueError("Gradient shapes mismatch")
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError("Nonfinite gradient")
        v=(x.detach().float()*y.detach().float()).sum()
        out=v if out is None else out+v
    return out


@torch.no_grad()
def project_conflicting_ot(ot_grad, target_fit_grad, eps=1e-8):
    """One-sided PCGrad on the OT component BEFORE source+OT aggregation.

    g' = g - min(0, <g,t>)/(||t||^2+eps) * t.
    A zero target gradient is a no-op. Positive-alignment OT is untouched.
    This operation is not differentiable through its gradient inputs.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    dot = dot_tuple(ot_grad,target_fit_grad)
    target_sq=dot_tuple(target_fit_grad,target_fit_grad)
    conflicting=bool(dot<0) and bool(target_sq>0)
    if conflicting:
        coef=(dot/(target_sq+eps)).detach()
        projected=[
            (x.detach()-coef.to(dtype=x.dtype,device=x.device)*y.detach()).detach()
            for x,y in zip(ot_grad,target_fit_grad)
        ]
    else:
        projected=[x.detach().clone() for x in ot_grad]
    removed=gradient_tuple_difference(ot_grad,projected)
    return projected,{
        "raw_dot":float(dot.item()),
        "target_fit_norm_sq":float(target_sq.item()),
        "is_conflicting":conflicting,
        "removed_norm":gradient_norm(removed),
        "raw_ot_norm":gradient_norm(ot_grad),
        "projected_ot_norm":gradient_norm(projected),
        "raw_vs_target":vector_alignment(ot_grad,target_fit_grad),
        "projected_vs_target":vector_alignment(projected,target_fit_grad),
    }


@torch.no_grad()
def make_h7_policies(city_entries, target_fit_grad):
    """Six fixed OT-gradient policies; all share the same true F2 OT gradients."""
    if len(city_entries)!=3:
        raise ValueError("H7 expects exactly three source cities")
    names=[str(e["city"]) for e in city_entries]
    if len(set(names))!=3 or "shenzhen" not in names:
        raise ValueError("Expected three distinct cities including shenzhen")
    length=len(target_fit_grad)
    if length==0 or any(len(e["f2"])!=length for e in city_entries):
        raise ValueError("Missing gradients or parameter mismatch")
    # Validate shapes, devices, and finite values before any shadow.
    for entry in city_entries:
        dot_tuple(entry["f2"],target_fit_grad)
    global_ot=sum_gradient_vectors([e["f2"] for e in city_entries])
    global_proj,global_info=project_conflicting_ot(global_ot,target_fit_grad)
    each=[];city_info={}
    for entry in city_entries:
        g,info=project_conflicting_ot(entry["f2"],target_fit_grad)
        each.append(g)
        city_info[entry["city"]]=info
    summed_city_proj=sum_gradient_vectors(each)
    shenzhen=next(x["f2"] for x in city_entries if x["city"]=="shenzhen")
    policies={
        "f2":global_ot,
        "global_half":gradient_tuple_scaled(global_ot,0.5),
        "city_off:shenzhen":gradient_tuple_difference(global_ot,shenzhen),
        "global_projection":global_proj,
        "city_projection":summed_city_proj,
        "city_projection_half":gradient_tuple_scaled(summed_city_proj,0.5),
        "identity":global_ot,
    }
    if tuple(policies)!=POLICY_NAMES:
        raise AssertionError("Policy set changed; experiment not preregistered")
    difference=gradient_tuple_difference(global_proj,summed_city_proj)
    # When every city conflicts, the sum-of-projections and projection-of-sum
    # are mathematically identical. Log this, not an invented city-level gain.
    diagnostics={
        "global":global_info,
        "city":city_info,
        "global_vs_city_projection_difference_norm":gradient_norm(difference),
        "global_vs_city_projection_relative_difference":(
            gradient_norm(difference)/max(gradient_norm(global_proj),1e-12)),
        "all_cities_conflicting":all(v["is_conflicting"] for v in city_info.values()),
        "target_fit_grad_norm":gradient_norm(target_fit_grad),
    }
    return policies, diagnostics


@torch.no_grad()
def _displacement(named,shadow):
    return [shadow[name]-p.detach() for name,p in named]


@torch.no_grad()
def _shadow_difference(named,sha,shb):
    return [sha[name]-shb[name] for name,_ in named]


def shadow_h7_audit(*,model,optimizer,source_task_loss,target_fit_task_loss,
                    city_entries,target_data,probe_mask,sample_index,
                    checkpoint_epoch,beta=5e-4):
    """Seven paired, stateful all-parameter AdamW shadows with identical RNG.

    Original one-sided PCGrad is applied AFTER candidate OT surgery for every
    policy. Comparing candidate gains against global_half is mandatory.
    """
    named=list(model.named_parameters())
    if not named or probe_mask.dtype!=torch.bool:
        raise ValueError("Need model with parameters and Boolean probe mask")
    params=[p for _,p in named]
    source_grad=_grads(source_task_loss,params)
    target_grad=_grads(target_fit_task_loss,params)  # target FIT ONLY
    policies,projection_info=make_h7_policies(city_entries,target_grad)
    _,zero_step_grad=projected_directions(
        source_grad,target_grad,gradient_tuple_scaled(policies["f2"],0.0)
    )
    gradient_traces={}
    for name,g in policies.items():
        _,direction=projected_directions(source_grad,target_grad,g)
        effective=gradient_tuple_difference(direction,zero_step_grad)
        gradient_traces[name]={
            "ot_raw_norm":gradient_norm(g),
            "ot_raw_target_fit_cos":vector_alignment(g,target_grad)["cos"],
            "effective_after_existing_pcgrad_norm":gradient_norm(effective),
            "effective_after_existing_pcgrad_target_cos":
                vector_alignment(effective,target_grad)["cos"],
            "source_plus_ot_vs_target_fit":
                vector_alignment([a+b for a,b in zip(source_grad,g)],target_grad),
        }

    # PROBE truth accessed only after all projection policies are finalized.
    original_truth=torch.as_tensor(
        target_data.y,device=params[0].device,dtype=params[0].dtype
    )
    if (probe_mask.numel()!=len(original_truth) or
        not bool(probe_mask.any()) or bool((~probe_mask).sum()==0)):
        raise ValueError("Invalid probe partition")
    blind=SimpleNamespace(
        x=target_data.x,y=torch.zeros_like(original_truth),
        means=target_data.means,stds=target_data.stds
    )
    probe_truth=original_truth[probe_mask].detach()
    saved_training={name:m.training for name,m in model.named_modules()}
    saved_dynamic={
        key:(hasattr(model,key),getattr(model,key,None))
        for key in ("RevIn_mean","RevIn_std","eps")
    }
    old_buffers={name:b.detach().clone() for name,b in model.named_buffers()}
    device=params[0].device
    cuda_devices=([device.index if device.index is not None
                   else torch.cuda.current_device()] if device.type=="cuda" else [])
    from torch.func import functional_call
    scores={};optimizer_traces={}
    try:
        with torch.no_grad(),torch.random.fork_rng(devices=cuda_devices):
            # Store reference optimizer outcomes, not a modification of model.
            references={}
            for name in ("f2","global_half"):
                _,update_grad=projected_directions(
                    source_grad,target_grad,policies[name]
                )
                references[name]=simulate_adamw_step(named,optimizer,update_grad)

            for name,g in policies.items():
                if name in references:
                    shadow=references[name]
                else:
                    _,update_grad=projected_directions(
                        source_grad,target_grad,g
                    )
                    shadow=simulate_adamw_step(named,optimizer,update_grad)

                # Every candidate uses the exact same random pooling draw.
                torch.manual_seed(681947+int(sample_index))
                shadow_buffers={k:v.clone() for k,v in old_buffers.items()}
                predictions,*_=functional_call(
                    model,(shadow,shadow_buffers),(blind,),{"stage":"train"}
                )
                scores[name]=evaluate_probes(
                    predictions[probe_mask].detach(),probe_truth,beta
                )
                delta_f2=_shadow_difference(named,shadow,references["f2"])
                delta_half=_shadow_difference(named,shadow,references["global_half"])
                optimizer_traces[name]={
                    "parameter_step_norm":gradient_norm(
                        _displacement(named,shadow)
                    ),
                    "delta_from_f2_norm":gradient_norm(delta_f2),
                    "delta_from_half_norm":gradient_norm(delta_half),
                    "delta_from_f2_vs_target_descent":
                        vector_alignment(
                            delta_f2,gradient_tuple_scaled(target_grad,-1.0)
                        ),
                    "delta_from_half_vs_target_descent":
                        vector_alignment(
                            delta_half,gradient_tuple_scaled(target_grad,-1.0)
                        ),
                }
                del predictions,shadow_buffers,delta_f2,delta_half
                if name not in references:
                    del shadow
    finally:
        mods=dict(model.named_modules())
        for name,flag in saved_training.items():
            mods[name].training=flag
        for key,(exists,value) in saved_dynamic.items():
            if exists:
                setattr(model,key,value)
            elif hasattr(model,key):
                delattr(model,key)
        with torch.no_grad():
            buffers=dict(model.named_buffers())
            for key,b in old_buffers.items():
                if not torch.equal(buffers[key],b):
                    buffers[key].copy_(b)

    def _gains(reference):
        baseline=scores[reference]
        return {
            name:{
                "gain_mae":baseline["mae"]-s["mae"],
                "gain_smooth_l1":baseline["smooth_l1"]-s["smooth_l1"],
                "gain_bands":[a-b for a,b in zip(
                    baseline["band_smooth_l1"],s["band_smooth_l1"])],
                "gain_mae_at_minutes":{
                    t:baseline["mae_at_minutes"][t]-s["mae_at_minutes"][t]
                    for t in ("10","60","120","180")
                },
            }
            for name,s in scores.items()
        }

    return {
        "audit_version":"h7-v1",
        "checkpoint_epoch":int(checkpoint_epoch),
        "batch_index":int(sample_index),
        "n_shadow_candidates":len(scores),
        "training_updates":0,"stage2_runs":0,
        "fit_window_indices":(~probe_mask).nonzero().flatten().cpu().tolist(),
        "probe_window_indices":probe_mask.nonzero().flatten().cpu().tolist(),
        "projection_diagnostics":projection_info,
        "gradient_traces":gradient_traces,
        "optimizer_traces":optimizer_traces,
        "candidate_scores":scores,
        "gains_vs_f2":_gains("f2"),
        "gains_vs_half":_gains("global_half"),
    }
