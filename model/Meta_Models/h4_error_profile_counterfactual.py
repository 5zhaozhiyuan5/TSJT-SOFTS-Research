"""H4: source forecast-error *profiles* vs target OT transfer contribution.

Read-only diagnostic on ORIGINAL F2; no new predictor, source weighting
controller or trainable parameters. Grouping uses source TRAIN labels only.
All counterfactuals use unchanged OT support/cost and target-fit gradients.

All scores from previously-trained three-day support are exploratory, NOT
independent generalization estimates. See H4_ERROR_PROFILE_EXPERIMENT.md.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

try:
    from .h1_shadow_audit import _grads, projected_directions, simulate_adamw_step
    from .h2_candidate_audit import sum_gradient_vectors
    from .h3_city_counterfactual import evaluate_probes
except ImportError:
    from h1_shadow_audit import _grads, projected_directions, simulate_adamw_step
    from h2_candidate_audit import sum_gradient_vectors
    from h3_city_counterfactual import evaluate_probes


GROUP_NAMES = (
    "accurate", "inaccurate", "short_weak", "mid_weak",
    "long_weak", "balanced"
)


@torch.no_grad()
def error_profile_groups(pred, truth, history, mean, std, gap=0.15):
    """Disjoint source-point partition using task errors only.

    Inputs [B,N,36] physical speed predictions/labels and [B,N,T,2]
    city-normalized history. Flattened output has B*N elements:
      accurate   : lowest quartile of normalized 36-step MAE
      inaccurate : highest quartile
      other half : by most disproportionately high of 3 band MAEs,
                   or balanced if all 3 bands are near-equal.
    All group definitions are fixed BEFORE viewing any target probe label.
    """
    if pred.ndim != 3 or truth.shape != pred.shape or pred.shape[-1] != 36:
        raise ValueError("Expected source physical speed [B,N,36]")
    if (history.ndim != 4 or history.shape[:2] != pred.shape[:2]
            or history.shape[-2] < 1 or history.shape[-1] < 1):
        raise ValueError("Invalid history shape")
    if not 0 <= gap <= 1:
        raise ValueError("Invalid balanced gap")
    device = pred.device
    a = pred.detach().float()
    y = truth.detach().to(device=device,dtype=torch.float32)
    if not torch.isfinite(a).all() or not torch.isfinite(y).all():
        raise ValueError("Invalid source forecast / label values")
    avg = torch.as_tensor(mean,device=device,dtype=a.dtype).reshape(-1)[0]
    scale = torch.as_tensor(std,device=device,dtype=a.dtype).reshape(-1)[0]
    if not torch.isfinite(scale).item() or scale.item() <= 0:
        raise ValueError("Invalid source-city standard deviation")
    band = torch.stack([
        (a[...,left:right]-y[...,left:right]).abs().mean(-1)/scale
        for left,right in ((0,12),(12,24),(24,36))
    ],dim=-1).reshape(-1,3)
    scalar_error = band.mean(-1)
    n = scalar_error.numel()
    if n < 12:
        raise ValueError("Need >=12 source points for H4 quantiles")
    # Stable tie ordering is essential to reproducible zero-error smoke tests.
    ids = torch.argsort(scalar_error,stable=True)
    k = n//4
    names = [""]*n
    for i in ids[:k].tolist():
        names[i] = "accurate"
    for i in ids[-k:].tolist():
        names[i] = "inaccurate"
    relative = band/(scalar_error.unsqueeze(-1)+1e-6)
    weak = ("short_weak","mid_weak","long_weak")
    for i in ids[k:n-k].tolist():
        row = relative[i]
        if float((row.max()-row.min()).item()) < gap:
            names[i]="balanced"
        else:
            names[i]=weak[int(row.argmax().item())]
    masks={name:torch.tensor([v==name for v in names],device=device,
                             dtype=torch.bool) for name in GROUP_NAMES}
    if not all(names):
        raise RuntimeError("Every source point must belong to exactly one group")
    # Persistence skill: diagnostics ONLY, never enters candidate ranking.
    last = history.detach()[...,-1,0].to(device=device,dtype=a.dtype)*scale+avg
    persistence = (last.unsqueeze(-1)-y).abs().mean(-1).reshape(-1)/scale
    return masks,{
        "n":n,
        "group_counts":{name:int(mask.sum().item()) for name,mask in masks.items()},
        "mean_mae_by_band":[float(x) for x in band.mean(0).tolist()],
        "source_mean_normalized_mae":float(scalar_error.mean().item()),
        "better_than_persistence_frac":float(
            (scalar_error<persistence).float().mean().item()
        ),
        "balanced_gap":gap,
        "group_mean_normalized_mae":{
            name:(float(scalar_error[mask].mean().item()) if mask.any() else None)
            for name,mask in masks.items()
        },
    }


@torch.no_grad()
def downweight_group(alpha, mask, keep_fraction=0.5):
    """One bounded intervention, within-city mass normalized back to one."""
    if alpha.ndim!=1 or mask.ndim!=1 or len(alpha)!=len(mask):
        raise ValueError("F2 alpha and group mask must be 1D and equal length")
    if mask.dtype!=torch.bool:
        raise ValueError("Expected Boolean group mask")
    if not math.isfinite(keep_fraction) or not 0<keep_fraction<1:
        raise ValueError("keep_fraction must be in (0,1)")
    if not bool(torch.isfinite(alpha).all()) or bool((alpha<0).any()):
        raise ValueError("Invalid F2 alpha")
    if not bool(torch.isclose(alpha.sum(),alpha.new_tensor(1),atol=1e-5)):
        raise ValueError("F2 alpha is not normalized")
    if not bool(mask.any()) or bool(mask.all()):
        raise ValueError("Group cannot be empty or cover all points")
    multiplier=torch.where(mask,alpha.new_tensor(keep_fraction),alpha.new_tensor(1))
    corrected=alpha.detach()*multiplier
    corrected=corrected/corrected.sum()
    return corrected.detach(),{
        "group_count":int(mask.sum().item()),
        "group_alpha_mass_before":float(alpha[mask].sum().item()),
        "group_alpha_mass_after":float(corrected[mask].sum().item()),
        "ess_before":float((1.0/alpha.square().sum()).item()),
        "ess_after":float((1.0/corrected.square().sum()).item()),
        "keep_fraction":keep_fraction,
    }



@torch.no_grad()
def matched_placebo_mask(alpha, true_mask, seed, trials=24):
    """Reproducible count-matched false group, chosen using F2 MASS alone.

    Never looks at probe predictions/labels. Candidate masks are pseudo-random
    and kept as disjoint as possible from the original accurate group.
    The placebo mass is only approximately matched; log mismatch explicitly.
    """
    if (alpha.ndim!=1 or true_mask.ndim!=1 or
            alpha.numel()!=true_mask.numel() or true_mask.dtype!=torch.bool):
        raise ValueError("Invalid alpha/true group for placebo")
    if not bool(true_mask.any()) or bool(true_mask.all()) or trials<1:
        raise ValueError("Placebo requires nontrivial group and trials")
    n=len(alpha)
    k=int(true_mask.sum().item())
    desired_mass=float(alpha[true_mask].sum().item())
    generator=torch.Generator(device="cpu").manual_seed(int(seed))
    best=None
    objective=float("inf")
    chosen_stats=None
    for _ in range(trials):
        perm=torch.randperm(n,generator=generator)[:k]
        candidate=torch.zeros(n,dtype=torch.bool,device=alpha.device)
        candidate[perm.to(device=alpha.device)]=True
        overlap=int((candidate&true_mask).sum().item())
        mass=float(alpha[candidate].sum().item())
        # Relative alpha-mass mismatch PLUS a penalty for overlapping source
        # samples; matching decisions never use the target probe labels.
        mismatch=abs(mass-desired_mass)/(desired_mass+1e-9)
        value=mismatch+4.0*overlap/max(1,k)
        if value<objective:
            objective=value
            best=candidate
            chosen_stats={
                "count":k,
                "mass_original":desired_mass,
                "mass_placebo":mass,
                "relative_mass_mismatch":mismatch,
                "overlap_count":overlap,
                "seed":int(seed),
                "trials":trials,
            }
    return best,chosen_stats


def shadow_profile_audit(*,model,optimizer,source_task_loss,target_fit_task_loss,
                         entries,target_data,probe_mask,sample_index,
                         checkpoint_epoch,beta=5e-4):
    """Virtual full-parameter one-step PCGrad+AdamW for baseline/group ablations.

    entries: per city {city, f2: gradient tuple, changed: {group: gradient tuple}}
    Every policy changes just ONE source-city group's OT marginal.
    Does not change model weights, optimizer states, .grad, or target beta.
    """
    if len(entries)!=3 or len({e["city"] for e in entries})!=3:
        raise ValueError("Require exactly three distinct source cities")
    named=list(model.named_parameters())
    params=[p for _,p in named]
    if not params or probe_mask.dtype!=torch.bool:
        raise ValueError("Bad model / probe mask")
    names=["f2"]
    base=sum_gradient_vectors([e["f2"] for e in entries])
    changes={}
    for entry in entries:
        for group,grad in entry["changed"].items():
            label=f'down:{entry["city"]}:{group}'
            if label in changes:
                raise ValueError("Duplicate intervention label")
            changes[label]=[
                full-orig+updated
                for full,orig,updated in zip(base,entry["f2"],grad)
            ]
            names.append(label)
    src_grad=_grads(source_task_loss,params)
    tgt_grad=_grads(target_fit_task_loss,params)
    original_truth=torch.as_tensor(target_data.y,device=params[0].device,
                                  dtype=params[0].dtype)
    probe_truth=original_truth[probe_mask].detach()
    # Prediction is independent of the target's future labels, which must
    # never be passed into a shadow forward.
    blind=SimpleNamespace(
        x=target_data.x,y=torch.zeros_like(original_truth),
        means=target_data.means,stds=target_data.stds
    )
    saved_training={n:m.training for n,m in model.named_modules()}
    dynamic=("RevIn_mean","RevIn_std","eps")
    saved_dynamic={key:(hasattr(model,key),getattr(model,key,None))
                   for key in dynamic}
    buffers={key:b.detach().clone() for key,b in model.named_buffers()}
    dev=params[0].device
    cuda_devices=([dev.index if dev.index is not None
                   else torch.cuda.current_device()] if dev.type=="cuda" else [])
    scores={}
    from torch.func import functional_call
    try:
        with torch.no_grad(),torch.random.fork_rng(devices=cuda_devices):
            for name in names:
                directions=base if name=="f2" else changes[name]
                _,projected=projected_directions(src_grad,tgt_grad,directions)
                shadow=simulate_adamw_step(named,optimizer,projected)
                # Identical SOFTS random-pooling draw for every policy.
                torch.manual_seed(419227+int(sample_index))
                shadow_buffers={k:v.clone() for k,v in buffers.items()}
                prediction,*_=functional_call(
                    model,(shadow,shadow_buffers),(blind,),{"stage":"train"}
                )
                scores[name]=evaluate_probes(
                    prediction[probe_mask].detach(),probe_truth,beta
                )
                del shadow,shadow_buffers,prediction
    finally:
        modules=dict(model.named_modules())
        for name,flag in saved_training.items():
            modules[name].training=flag
        for key,(exists,value) in saved_dynamic.items():
            if exists:setattr(model,key,value)
            elif hasattr(model,key):delattr(model,key)
    baseline=scores["f2"]
    gains={}
    for name,score in scores.items():
        if name=="f2":continue
        # positive suppress_gain: downweighting HELPED probe target predictions.
        gains[name]={
            "suppress_gain_smooth_l1":
                baseline["smooth_l1"]-score["smooth_l1"],
            "suppress_gain_mae":baseline["mae"]-score["mae"],
            "suppress_gain_bands":[
                a-b for a,b in zip(baseline["band_smooth_l1"],
                                   score["band_smooth_l1"])
            ],
            "suppress_gain_at_minutes":{
                m:baseline["mae_at_minutes"][m]-score["mae_at_minutes"][m]
                for m in ("10","60","120","180")
            },
        }
    return {
        "audit_version":"h4-v1",
        "checkpoint_epoch":int(checkpoint_epoch),
        "batch_index":int(sample_index),
        "n_shadow_candidates":len(scores),
        "training_updates":0,
        "probe_window_indices":probe_mask.nonzero().flatten().cpu().tolist(),
        "fit_window_indices":(~probe_mask).nonzero().flatten().cpu().tolist(),
        "f2_baseline":baseline,
        "gains":gains,
        "candidate_scores":scores,
    }
