"""H10 source/target/OT/optimizer effect ablations.

The ORIGINAL checkpoint training log states projection=1.0. Therefore the
faithful original F2 Stage-I branch has one-sided whole-update PCGrad of
(source + OT) against all 46 target support windows. T6 explicitly removes
PCGrad, isolating its actual impact. H9 used 36 target FIT windows. All updates are applied only to fresh clones.
"""
from __future__ import annotations

import copy
import math

import torch

try:
    from .h1_shadow_audit import projected_directions
    from .h2_candidate_audit import sum_gradient_vectors
    from .h6_ot_gradient_effect import gradient_norm,gradient_tuple_scaled
    from .h7_ot_component_projection import project_conflicting_ot
    from .h3_city_counterfactual import vector_alignment
    from .h9_checkpoint_adaptation import clone_f2
except ImportError:
    from h1_shadow_audit import projected_directions
    from h2_candidate_audit import sum_gradient_vectors
    from h6_ot_gradient_effect import gradient_norm,gradient_tuple_scaled
    from h7_ot_component_projection import project_conflicting_ot
    from h3_city_counterfactual import vector_alignment
    from h9_checkpoint_adaptation import clone_f2


POLICIES=(
    "t0_no_update",
    "t0f_train_forward_only",
    "t1_zero_grad_adamw",
    "t2_source_only",
    "t3_target_only",
    "t4_source_pcgrad",
    "t5_official_f2",
    "t6_full_pcgrad",
    "t7_project_ot_half",
)
UPDATE_POLICIES=POLICIES[2:]


def _sum(a,b):
    if len(a)!=len(b):
        raise ValueError("Gradient lengths differ")
    return [x+y for x,y in zip(a,b)]


def _zero(a):
    return [torch.zeros_like(x) for x in a]


def h10_direction(bank,policy):
    """Return gradient/update mask plus diagnostics, no optimizer access."""
    if policy not in UPDATE_POLICIES:
        raise ValueError("Direction unavailable for no-update control")
    source=bank["source_gradient"]
    target=bank["target_fit_gradient"]
    raw_ot=sum_gradient_vectors([c["f2"] for c in bank["city_entries"]])
    no_ot=[False]*len(source)
    source_mask=bank.get("source_present_mask",[True]*len(source))
    target_mask=bank.get("target_present_mask",[True]*len(source))
    ot_mask=bank.get("ot_present_mask",[True]*len(source))
    union_mask=[s or o for s,o in zip(source_mask,ot_mask)]
    if policy=="t1_zero_grad_adamw":
        direction=_zero(source);mask=[True]*len(source)
    elif policy=="t2_source_only":
        direction=source;mask=source_mask
    elif policy=="t3_target_only":
        direction=target;mask=target_mask
    elif policy=="t4_source_pcgrad":
        direction=projected_directions(source,target,_zero(source))[0]
        mask=[True]*len(source)
    elif policy=="t5_official_f2":
        direction=projected_directions(source,target,raw_ot)[1]
        # Original projection=1 flatten_and_pad_grads + write_back_flat_grads
        # sets a tensor grad on EVERY parameter, even if unused.
        mask=[True]*len(source)
    elif policy=="t6_full_pcgrad":
        # Historical name retained as an ablation ID: this control REMOVES
        # PCGrad, unlike original checkpoint's projection=1.
        direction=_sum(source,raw_ot)
        mask=union_mask
    elif policy=="t7_project_ot_half":
        projected,_=project_conflicting_ot(raw_ot,target)
        half=gradient_tuple_scaled(projected,.5)
        direction=projected_directions(source,target,half)[1]
        mask=[True]*len(source)
    else:
        raise ValueError(policy)
    if len(direction)!=len(mask) or any(not bool(
        torch.isfinite(x).all()
    ) for x in direction):
        raise FloatingPointError("Invalid H10 gradient direction")
    return direction,mask,{
        "source_grad_norm":gradient_norm(source),
        "target_grad_norm":gradient_norm(target),
        "ot_grad_norm":gradient_norm(raw_ot),
        "source_vs_target":vector_alignment(source,target),
        "ot_vs_target":vector_alignment(raw_ot,target),
        "source_vs_ot":vector_alignment(source,raw_ot),
        "update_grad_norm":gradient_norm(direction),
        "update_vs_target":vector_alignment(direction,target),
        "grad_tensors_present":sum(mask),
        "grad_tensors_missing":len(mask)-sum(mask),
        "ot_city_grad_norms":{
            c["city"]:gradient_norm(c["f2"]) for c in bank["city_entries"]
        },
    }


def apply_directions(model,optimizer,direction,present_mask):
    """True AdamW clone step, preserving official None-grad semantics."""
    params=list(model.parameters())
    if not(len(params)==len(direction)==len(present_mask)):
        raise ValueError("Optimizer parameter/gradient mismatch")
    if len(optimizer.param_groups)!=1:
        raise ValueError("Only original one-group AdamW supported")
    optimizer.zero_grad(set_to_none=True)
    before=[p.detach().clone() for p in params]
    with torch.no_grad():
        for p,g,present in zip(params,direction,present_mask):
            p.grad=g.detach().clone() if present else None
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    step_sq=sum(float((p.detach().float()-old.float()).square().sum())
                for p,old in zip(params,before))
    return math.sqrt(step_sq)


def clone_official_wrapper(wrapper):
    """Deep-copy both wrapped model and AdamW, rebind Python closure hooks."""
    official=copy.deepcopy(wrapper)
    # deepcopy of a registered hook copies a closure referencing the ORIGINAL
    # STRep. Rebind only hooks on this fresh clone, never on the original.
    head=official.model.forecasting_model.projection2
    head._forward_hooks.clear()
    def capture(_module,inputs,output):
        if official._capture_transfer:
            official._transfer_records.append((inputs[0],output))
    official._transfer_hook=head.register_forward_hook(capture)
    # G0 does not need feedback projection1 hook; remove copied stale closure.
    if hasattr(official.model.forecasting_model,"projection1"):
        official.model.forecasting_model.projection1._forward_hooks.clear()
    official._capture_feedback=False
    official._capture_transfer=False
    official._transfer_records=[]
    # Safety: ensure deepcopy optimizer points exactly to cloned parameters.
    cloned_parameters=list(official.model.parameters())
    optimizer_parameters=[
        p for g in official.main_optim.param_groups for p in g["params"]
    ]
    if (len(cloned_parameters)!=len(optimizer_parameters) or
        any(id(a)!=id(b) for a,b in zip(cloned_parameters,optimizer_parameters))):
        raise RuntimeError("Official cloned optimizer does not own clone weights")
    if float(official.projection_args)!=1.0:
        raise RuntimeError("Pinned checkpoint log uses projection=1.0; refusing alternate official branch")
    return official


def snapshot_tensors(module,optimizer):
    with torch.no_grad():
        weights={
            n:p.detach().cpu().clone()
            for n,p in module.named_parameters()
        }
        buffers={
            n:b.detach().cpu().clone()
            for n,b in module.named_buffers()
        }
    state=copy.deepcopy(optimizer.state_dict())
    for entry in state["state"].values():
        for key,v in entry.items():
            if torch.is_tensor(v):
                entry[key]=v.detach().cpu()
    return weights,buffers,state


def _max_dict_diff(a,b):
    if set(a)!=set(b):
        raise ValueError("Mismatched model parameters/buffers")
    if not a:
        return 0.
    out=0.
    for key in a:
        x,y=a[key].float(),b[key].float()
        if x.shape!=y.shape:
            raise ValueError("Different parameter shape "+key)
        out=max(out,float((x-y).abs().max()))
    return out


def compare_snapshots(lhs,rhs,*,atol=2e-5):
    """Return independently checked parameters, model buffers & optimizer."""
    aw,ab,ao=lhs;bw,bb,bo=rhs
    werr=_max_dict_diff(aw,bw)
    berr=_max_dict_diff(ab,bb)
    if (len(ao["param_groups"])!=len(bo["param_groups"]) or
        set(ao["state"])!=set(bo["state"])):
        raise ValueError("Different AdamW structure")
    optim_max=0.
    for k,item in ao["state"].items():
        other=bo["state"][k]
        if set(item)!=set(other):
            raise ValueError("Different optimizer state fields")
        for field,x in item.items():
            y=other[field]
            if torch.is_tensor(x):
                optim_max=max(optim_max,float(
                    (x.float()-y.float()).abs().max()
                ))
            elif x!=y:
                raise ValueError(f"AdamW scalar state mismatch {k}:{field}")
    group_equal=all(
        {k:v for k,v in ag.items() if k!="params"}==
        {k:v for k,v in bg.items() if k!="params"}
        for ag,bg in zip(ao["param_groups"],bo["param_groups"])
    )
    return {
        "parameter_max_abs_diff":werr,
        "buffer_max_abs_diff":berr,
        "adamw_state_max_abs_diff":optim_max,
        "adamw_group_hyperparameters_equal":group_equal,
        "atol":atol,
        "passed":all(x<=atol for x in (werr,berr,optim_max)) and group_equal,
    }
