"""H8: read-only OT direction-versus-magnitude counterfactual audit.

H8 disentangles the gains of H7's 0.5*Proj(OT,target FIT gradient).
All policies reuse exact original F2 GeomLoss OT gradients, F2 checkpoint
and AdamW moments; only pre-PCGrad OT-gradient interventions change.

This is PREVIOUSLY-SEEN target-support one-step screening, not a new method
or a final test result. No probe label is read until all candidates,
including the AdamW-norm calibration, have been fixed.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

try:
    from .h1_shadow_audit import _grads, projected_directions, simulate_adamw_step
    from .h2_candidate_audit import sum_gradient_vectors
    from .h3_city_counterfactual import evaluate_probes, vector_alignment
    from .h6_ot_gradient_effect import (
        gradient_norm, gradient_tuple_scaled, gradient_tuple_difference
    )
    from .h7_ot_component_projection import project_conflicting_ot
except ImportError:
    from h1_shadow_audit import _grads, projected_directions, simulate_adamw_step
    from h2_candidate_audit import sum_gradient_vectors
    from h3_city_counterfactual import evaluate_probes, vector_alignment
    from h6_ot_gradient_effect import (
        gradient_norm, gradient_tuple_scaled, gradient_tuple_difference
    )
    from h7_ot_component_projection import project_conflicting_ot


POLICIES = (
    "f2",
    "global_half",
    "projected_half",
    "norm_matched_raw",
    "direction_norm_matched_half",
    "adamw_step_norm_matched_raw",
    "all_off",
    "identity",
)
EPS=1e-12


def build_h8_policies(city_entries, target_fit_gradient):
    """Build fixed pre-optimizer, same-gradient-norm controls.

    projected_half:        0.5 * P(g)                         [H7 winner]
    norm_matched_raw:      0.5 * ||P(g)||/||g|| * g             [same norm, raw direction]
    direction_norm_matched_half: 0.5 * ||g||/||P(g)|| * P(g)
                          [same norm as global_half, projected direction]
    adamw_step_norm_matched_raw is populated later from the actual AdamW
    optimizer state to match the *OT-induced parameter displacement*.
    """
    if len(city_entries)!=3:
        raise ValueError("H8 requires exactly 3 sources")
    names=[str(x["city"]) for x in city_entries]
    if len(set(names))!=3:
        raise ValueError("H8 source-city names must be unique")
    if not target_fit_gradient or any(
        len(e["f2"])!=len(target_fit_gradient) for e in city_entries
    ):
        raise ValueError("Incompatible source and target gradient lengths")
    raw=sum_gradient_vectors([e["f2"] for e in city_entries])
    projected,proj_stats=project_conflicting_ot(raw,target_fit_gradient)
    nr=gradient_norm(raw)
    np=gradient_norm(projected)
    # Handle exact zero without inventing a spurious gradient direction.
    same_factor=(0.5*np/nr) if nr>EPS else 0.0
    direction_factor=(0.5*nr/np) if np>EPS else 0.0
    fixed={
        "f2":raw,
        "global_half":gradient_tuple_scaled(raw,.5),
        "projected_half":gradient_tuple_scaled(projected,.5),
        "norm_matched_raw":gradient_tuple_scaled(raw,same_factor),
        "direction_norm_matched_half":
            gradient_tuple_scaled(projected,direction_factor),
        "all_off":gradient_tuple_scaled(raw,0.),
        "identity":raw,
    }
    # Numerically verify the design does what the names promise.
    same_err=abs(
        gradient_norm(fixed["projected_half"]) -
        gradient_norm(fixed["norm_matched_raw"])
    )
    half_err=abs(
        gradient_norm(fixed["global_half"]) -
        gradient_norm(fixed["direction_norm_matched_half"])
    )
    tol=1e-6*max(1.0,nr)
    if same_err>tol or (np>EPS and half_err>tol):
        raise RuntimeError("Gradient-norm controls were not matched")
    stats={
        "raw_gradient_norm":nr,
        "projected_gradient_norm":np,
        "raw_target_gradient_cos":proj_stats["raw_vs_target"]["cos"],
        "projected_target_gradient_cos":proj_stats["projected_vs_target"]["cos"],
        "removed_conflict_norm":proj_stats["removed_norm"],
        "projection_triggered":proj_stats["is_conflicting"],
        "norm_matched_raw_factor":same_factor,
        "projected_equal_half_norm_factor":direction_factor,
        "projected_vs_norm_matched_raw_gradient_norm_error":same_err,
        "global_half_vs_equal_norm_projected_error":half_err,
        "source_city_grad_norms":{
            x["city"]:gradient_norm(x["f2"]) for x in city_entries
        },
    }
    return fixed,stats


@torch.no_grad()
def shadow_delta(named,a,b):
    return [a[name]-b[name] for name,_ in named]


def _shadow_candidate(named,optimizer,source_gradient,target_gradient,ot_gradient):
    _,direction=projected_directions(
        source_gradient,target_gradient,ot_gradient
    )
    return simulate_adamw_step(named,optimizer,direction)


def _ot_effect_norm(named,shadow,no_ot_shadow):
    return gradient_norm(shadow_delta(named,shadow,no_ot_shadow))


def calibrate_adamw_step_norm(*,named,optimizer,source_gradient,target_gradient,
                              raw_ot,projected_half_shadow,no_ot_shadow,
                              max_factor=4.0,iterations=16):
    """Match the OT-only induced AdamW parameter displacement norm.

    The target is ||theta(projected_half)-theta(no_ot)||. Search alpha >=0 for
    ||theta(alpha*g_raw)-theta(no_ot)|| ~= target. Uses model parameters, FIT
    gradients and *saved optimizer state only*: no target PROBE labels.
    A monotonicity check records whether binary search is a sound assumption.
    On failure, returns the nearest tried factor and marks unmatched, rather
    than silently claiming a controlled comparison.
    """
    if iterations<1 or max_factor<=0:
        raise ValueError("Invalid calibration configuration")
    target=_ot_effect_norm(named,projected_half_shadow,no_ot_shadow)
    raw_base=gradient_norm(raw_ot)
    if raw_base<EPS or target<EPS:
        return 0.,{
            "target_ot_adamw_effect_norm":target,
            "actual_ot_adamw_effect_norm":0.,
            "absolute_error":target,
            "relative_error":0. if target<EPS else 1.,
            "within_one_percent":target<EPS,
            "monotonic":True,
            "calibration_evaluations":0,
            "calibration_limited":False,
        }

    # Lightweight coarse ladder also catches non-monotonic AdamW situations.
    cache={0.:0.}
    trials=0
    def value(factor):
        nonlocal trials
        f=float(factor)
        if f not in cache:
            shadow=_shadow_candidate(
                named,optimizer,source_gradient,target_gradient,
                gradient_tuple_scaled(raw_ot,f)
            )
            cache[f]=_ot_effect_norm(named,shadow,no_ot_shadow)
            del shadow
            trials+=1
        return cache[f]

    coarse=[max_factor*i/8 for i in range(9)]
    coarse_values=[value(k) for k in coarse]
    monotonic=all(
        coarse_values[i+1]+1e-7 >= coarse_values[i]
        for i in range(len(coarse_values)-1)
    )
    best=min(cache,key=lambda f:abs(cache[f]-target))
    # AdamW is nonlinear in the gradient and need not be globally monotone.
    # Bracket ANY coarse interval that crosses the desired step norm, then
    # bisect that continuous interval even if the full ladder is nonmonotone.
    brackets=[]
    for i in range(len(coarse)-1):
        left,right=coarse[i],coarse[i+1]
        yleft,yright=coarse_values[i],coarse_values[i+1]
        if (yleft-target)*(yright-target)<=0:
            brackets.append((left,right))
    if brackets:
        low,high=min(
            brackets,
            key=lambda pair:min(
                abs(value(pair[0])-target),abs(value(pair[1])-target)
            )
        )
        flow=value(low)-target
        for _ in range(iterations):
            mid=.5*(low+high)
            y=value(mid)
            if abs(y-target)<abs(cache[best]-target):
                best=mid
            if flow*(y-target)<=0:
                high=mid
            else:
                low=mid
                flow=y-target
    # Otherwise the requested norm lies outside our bounded safe range or
    # no coarse bracket exists. Keep the closest observed point, report it
    # as unmatched, and never infer a direction-only effect from it.
    actual=value(best)
    error=abs(actual-target)
    relative=error/max(target,EPS)
    return float(best),{
        "target_ot_adamw_effect_norm":target,
        "actual_ot_adamw_effect_norm":actual,
        "absolute_error":error,
        "relative_error":relative,
        "within_one_percent":relative<=.01,
        "monotonic":monotonic,
        "calibration_evaluations":trials,
        "calibration_limited":bool(best>=max_factor and relative>.01),
    }


def shadow_h8_audit(*,model,optimizer,source_task_loss,target_fit_task_loss,
                    city_entries,target_data,probe_mask,sample_index,
                    checkpoint_epoch,beta=5e-4):
    """Read-only full-parameter H8 8-policy shadow with post-AdamW norm control."""
    named=list(model.named_parameters())
    if not named or probe_mask.dtype!=torch.bool:
        raise ValueError("Need model parameters and Boolean probe mask")
    params=[p for _,p in named]
    source_gradient=_grads(source_task_loss,params)
    target_gradient=_grads(target_fit_task_loss,params)
    fixed,gradient_stats=build_h8_policies(city_entries,target_gradient)
    raw=fixed["f2"]

    saved_training={k:m.training for k,m in model.named_modules()}
    saved_dynamic={
        k:(hasattr(model,k),getattr(model,k,None))
        for k in ("RevIn_mean","RevIn_std","eps")
    }
    original_buffers={k:b.detach().clone() for k,b in model.named_buffers()}
    device=params[0].device
    cuda_devices=([device.index if device.index is not None
                  else torch.cuda.current_device()] if device.type=="cuda" else [])
    from torch.func import functional_call
    scores={}
    optimizer_traces={}
    try:
        with torch.no_grad(),torch.random.fork_rng(devices=cuda_devices):
            references={
                name:_shadow_candidate(
                    named,optimizer,source_gradient,target_gradient,fixed[name]
                )
                for name in ("f2","global_half","projected_half","all_off")
            }
            factor,calibration=calibrate_adamw_step_norm(
                named=named,optimizer=optimizer,
                source_gradient=source_gradient,target_gradient=target_gradient,
                raw_ot=raw,
                projected_half_shadow=references["projected_half"],
                no_ot_shadow=references["all_off"],
            )
            fixed["adamw_step_norm_matched_raw"]=gradient_tuple_scaled(raw,factor)
            if set(fixed)!=set(POLICIES):
                raise RuntimeError("H8 candidate set mismatched preregistration")
            gradient_stats["adamw_step_match_raw_factor"]=factor

            gradients_after_pc={}
            no_ot_grad=projected_directions(
                source_gradient,target_gradient,fixed["all_off"]
            )[1]
            for name in POLICIES:
                _,update_grad=projected_directions(
                    source_gradient,target_gradient,fixed[name]
                )
                effective=gradient_tuple_difference(update_grad,no_ot_grad)
                gradients_after_pc[name]={
                    "ot_norm_pre_pcgrad":gradient_norm(fixed[name]),
                    "ot_cos_target_fit_pre_pcgrad":
                        vector_alignment(fixed[name],target_gradient)["cos"],
                    "effective_ot_after_pcgrad_norm":gradient_norm(effective),
                    "effective_ot_after_pcgrad_target_fit_cos":
                        vector_alignment(effective,target_gradient)["cos"],
                }

            # Only after every policy and calibration is fixed, touch PROBE y.
            original_truth=torch.as_tensor(
                target_data.y,device=device,dtype=params[0].dtype
            )
            if (probe_mask.numel()!=len(original_truth) or
                not bool(probe_mask.any()) or not bool((~probe_mask).any())):
                raise ValueError("Invalid target fit/probe partition")
            blinded=SimpleNamespace(
                x=target_data.x,
                y=torch.zeros_like(original_truth),
                means=target_data.means,
                stds=target_data.stds,
            )
            probe_truth=original_truth[probe_mask].detach()
            for name in POLICIES:
                shadow=references.get(name)
                if shadow is None:
                    shadow=_shadow_candidate(
                        named,optimizer,source_gradient,target_gradient,fixed[name]
                    )
                # Exactly the same SOFTS pooling RNG as H6 / H7 for replay.
                torch.manual_seed(681947+int(sample_index))
                buffers={k:b.clone() for k,b in original_buffers.items()}
                output,*_=functional_call(
                    model,(shadow,buffers),(blinded,),{"stage":"train"}
                )
                scores[name]=evaluate_probes(
                    output[probe_mask].detach(),probe_truth,beta
                )
                optimizer_traces[name]={
                    "parameter_step_norm":gradient_norm(
                        [shadow[key]-p.detach() for key,p in named]
                    ),
                    "ot_induced_step_norm":_ot_effect_norm(
                        named,shadow,references["all_off"]
                    ),
                    "vs_projected_half_step_delta":vector_alignment(
                        shadow_delta(named,shadow,references["projected_half"]),
                        gradient_tuple_scaled(target_gradient,-1.)
                    ),
                    "vs_global_half_step_delta":vector_alignment(
                        shadow_delta(named,shadow,references["global_half"]),
                        gradient_tuple_scaled(target_gradient,-1.)
                    ),
                }
                del output,buffers
                if name not in references:
                    del shadow
    finally:
        mods=dict(model.named_modules())
        for k,flag in saved_training.items():
            mods[k].training=flag
        for key,(exists,value) in saved_dynamic.items():
            if exists:
                setattr(model,key,value)
            elif hasattr(model,key):
                delattr(model,key)
        with torch.no_grad():
            current=dict(model.named_buffers())
            for k,v in original_buffers.items():
                if not torch.equal(current[k],v):
                    current[k].copy_(v)

    def _gains(reference):
        baseline=scores[reference]
        return {
            k:{
                "mae":baseline["mae"]-s["mae"],
                "smooth_l1":baseline["smooth_l1"]-s["smooth_l1"],
                "bands":[a-b for a,b in zip(
                    baseline["band_smooth_l1"],s["band_smooth_l1"]
                )],
                "mae_minutes":{
                    t:baseline["mae_at_minutes"][t]-s["mae_at_minutes"][t]
                    for t in ("10","60","120","180")
                },
            } for k,s in scores.items()
        }

    return {
        "audit_version":"h8-v1",
        "batch_index":int(sample_index),
        "checkpoint_epoch":int(checkpoint_epoch),
        "training_updates":0,"stage2_runs":0,
        "fit_window_indices":(~probe_mask).nonzero().flatten().cpu().tolist(),
        "probe_window_indices":probe_mask.nonzero().flatten().cpu().tolist(),
        "candidate_scores":scores,
        "gains_vs_f2":_gains("f2"),
        "gains_vs_half":_gains("global_half"),
        "gains_vs_projected_half":_gains("projected_half"),
        "gains_vs_same_norm_raw":_gains("norm_matched_raw"),
        "gains_vs_step_norm_raw":_gains("adamw_step_norm_matched_raw"),
        "gradient_stats":gradient_stats,
        "pcgrad_traces":gradients_after_pc,
        "adamw_calibration":calibration,
        "adamw_traces":optimizer_traces,
        "n_shadow_candidates":len(scores),
    }
