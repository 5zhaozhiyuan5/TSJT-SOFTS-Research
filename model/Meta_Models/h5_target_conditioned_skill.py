"""H5: target-fit demand x source forecast skill x horizon-specific head gradient.

AUDIT ONLY. No trainable parameters, no optimizer.step(), no modifications to
the original F2, its OT geometry, or official Stage-II.  This utility uses
source TRAIN labels and target FIT labels, never target probe/test labels.
"""
from __future__ import annotations

import math
import torch

try:
    from .prediction_reliability import calibrate_source_alpha
except ImportError:
    from prediction_reliability import calibrate_source_alpha

BANDS = ((0, 12), (12, 24), (24, 36))
METHODS = (
    "prc", "demand_skill", "demand_gradient", "joint",
    "rotated_demand", "shuffled_skill"
)


@torch.no_grad()
def source_band_capability(pred, truth, history, city_mean, city_std):
    """Source [n,36] speed predictions/labels, history [n,T,C] normalized.

    skill=(persistence_mae-model_mae)/max(persistence_mae,0.1*city_std).
    Returns sigmoid(2*skill) per band, not a transfer-utility probability.
    """
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] != 36:
        raise ValueError("Expected physical-speed predictions [n,36]")
    if history.ndim != 3 or history.shape[0] != len(pred) or history.shape[1] < 1:
        raise ValueError("Expected [n,T,C] source history")
    if history.shape[2] < 1:
        raise ValueError("Source history must contain speed")
    device = pred.device
    mean = torch.as_tensor(city_mean,device=device,dtype=torch.float32).reshape(-1)[0]
    std = torch.as_tensor(city_std,device=device,dtype=torch.float32).reshape(-1)[0]
    if not torch.isfinite(std) or std <= 0 or not torch.isfinite(mean):
        raise ValueError("Invalid city mean/std")
    pred,truth=pred.detach().float(),truth.detach().float()
    if not bool(torch.isfinite(pred).all() and torch.isfinite(truth).all()):
        raise ValueError("Nonfinite speed")
    last = history.detach()[:,-1,0].to(device=device,dtype=torch.float32)*std+mean
    model_error = torch.stack([
        (pred[:,a:b]-truth[:,a:b]).abs().mean(dim=1)/std
        for a,b in BANDS
    ],dim=1)
    persistence_error = torch.stack([
        (last[:,None]-truth[:,a:b]).abs().mean(dim=1)/std
        for a,b in BANDS
    ],dim=1)
    skill = ((persistence_error-model_error)/
             persistence_error.clamp_min(0.1)).clamp(-10,10)
    capability = torch.sigmoid((2*skill).clamp(-20,20))
    return capability.detach(), {
        "normalized_model_band_mae":model_error.mean(0).tolist(),
        "normalized_persistence_band_mae":persistence_error.mean(0).tolist(),
        "skill_mean_by_band":skill.mean(0).tolist(),
        "capability_mean_by_band":capability.mean(0).tolist(),
        "model_beats_persistence_by_band":
            (model_error < persistence_error).float().mean(0).tolist(),
    }


@torch.no_grad()
def target_fit_demand(pred, truth, city_std):
    """Target FIT ONLY: mean normalized forecast error in 3 fixed bands."""
    if pred.shape!=truth.shape or pred.ndim!=3 or pred.shape[-1]!=36:
        raise ValueError("Expected [fit_windows,nodes,36]")
    if not pred.shape[0]:
        raise ValueError("Empty target fit windows")
    scale=torch.as_tensor(city_std,device=pred.device,dtype=torch.float32).reshape(-1)[0]
    if not torch.isfinite(scale) or scale<=0:
        raise ValueError("Bad target scale")
    err=torch.stack([
        (pred[...,a:b].detach()-truth[...,a:b].detach()).abs().mean()/scale
        for a,b in BANDS
    ])
    if not torch.isfinite(err).all():
        raise ValueError("Nonfinite target fit error")
    weights=(err+1e-6)/(err.sum()+3e-6)
    return weights.detach(),{"target_fit_band_mae_normalized":err.tolist(),
                             "demand":weights.tolist(),
                             "demand_range":float(weights.max()-weights.min())}


@torch.no_grad()
def band_head_cosine(source_features,source_deltas,target_features,target_deltas):
    """cos(vec(delta_i^band outer f_i), mean_j vec(delta_tj^band outer f_tj)).

    Each band's corresponding head coordinates are compared; no cross-band
    comparisons (different projection2 rows would be disjoint by construction).
    """
    sf,sd,tf,td=(x.detach().float() for x in
                  (source_features,source_deltas,target_features,target_deltas))
    if (sf.ndim!=2 or tf.ndim!=2 or sd.ndim!=2 or td.ndim!=2 or
        sf.shape[1]!=tf.shape[1] or sf.shape[0]!=sd.shape[0] or
        tf.shape[0]!=td.shape[0] or sd.shape[1]!=36 or td.shape[1]!=36):
        raise ValueError("Wrong head gradient shapes")
    if not bool(all(torch.isfinite(x).all() for x in (sf,sd,tf,td))):
        raise ValueError("Nonfinite head gradients")
    scores=[]
    for a,b in BANDS:
        target_matrix=torch.einsum("nh,nf->hf",td[:,a:b],tf)
        dot=torch.einsum("nh,nf,hf->n",sd[:,a:b],sf,target_matrix)
        denom=(torch.linalg.vector_norm(sd[:,a:b],dim=1)*
               torch.linalg.vector_norm(sf,dim=1)*
               torch.linalg.vector_norm(target_matrix))
        cosine=torch.where(denom>1e-12,dot/denom.clamp_min(1e-12),
                           torch.zeros_like(dot))
        scores.append(cosine.clamp(-1.0,1.0))
    return torch.stack(scores,dim=1).detach()


@torch.no_grad()
def stratified_shuffle(capability, alpha, seed):
    """Shuffle entire band profiles within F2-alpha quartiles, not across cities."""
    if capability.ndim!=2 or capability.shape[1]!=3 or alpha.ndim!=1 or len(alpha)!=len(capability):
        raise ValueError("Bad shuffle inputs")
    n=len(alpha)
    order=torch.argsort(alpha.detach().cpu(),stable=True)
    cpu=capability.detach().cpu()
    mixed=cpu.clone()
    generator=torch.Generator(device="cpu").manual_seed(int(seed))
    for part in range(4):
        idx=order[(part*n)//4:((part+1)*n)//4]
        if len(idx)>1:
            perm=idx[torch.randperm(len(idx),generator=generator)]
            mixed[idx]=cpu[perm]
    return mixed.to(device=capability.device,dtype=capability.dtype)


@torch.no_grad()
def h5_candidate_alphas(f2_alpha, prc_quality, capability, band_cosine,
                        demand, seed, strength=0.5):
    """Six fixed counterfactual marginals and their diagnostics.

    All methods reuse identical Sinkhorn cost, support indices and target beta.
    No candidate may inspect target PROBE labels or performance.
    """
    if not math.isfinite(strength) or not 0<=strength<=1:
        raise ValueError("Bad strength")
    alpha=f2_alpha.detach().reshape(-1)
    n=len(alpha)
    if (alpha.ndim!=1 or n<8 or capability.shape!=(n,3) or
        band_cosine.shape!=(n,3) or demand.shape!=(3,) or
        prc_quality.numel()!=n):
        raise ValueError("Bad candidate input shapes")
    if not bool(torch.isfinite(alpha).all() and torch.isfinite(capability).all()
                and torch.isfinite(band_cosine).all()
                and torch.isfinite(demand).all()):
        raise ValueError("Nonfinite scores")
    if bool((alpha<0).any()) or not torch.isclose(alpha.sum(),alpha.new_tensor(1),atol=1e-5):
        raise ValueError("Non-normalized alpha")
    if bool((capability<0).any() or (capability>1).any() or
            (band_cosine.abs()>1.00001).any() or
            (demand<0).any()) or not torch.isclose(demand.sum(),demand.new_tensor(1),atol=1e-5):
        raise ValueError("Invalid score range")
    d=demand.to(device=alpha.device,dtype=alpha.dtype).detach()
    cap=capability.to(device=alpha.device,dtype=alpha.dtype).detach()
    align=((band_cosine.to(device=alpha.device,dtype=alpha.dtype).detach()+1)/2).clamp(0,1)
    quality=prc_quality.reshape(-1).to(device=alpha.device,dtype=alpha.dtype).detach()
    shuffled=stratified_shuffle(cap,alpha,seed)
    raw={
      "prc":quality,
      "demand_skill":(cap*d).sum(1),
      "demand_gradient":(align*d).sum(1),
      "joint":(cap*align*d).sum(1),
      "rotated_demand":(cap*align*d.roll(1)).sum(1),
      "shuffled_skill":(shuffled*align*d).sum(1),
    }
    output={};records={}
    for name in METHODS:
        w,meta=calibrate_source_alpha(alpha,raw[name].clamp(0,1),
                                       strength=strength)
        output[name]=w
        records[name]={
            **meta,
            "alpha_l1_shift":float((w-alpha).abs().sum()),
            "score_mean":float(raw[name].mean()),
            "score_std":float(raw[name].std(unbiased=False)),
        }
    records["head_cosine_mean_by_band"]=band_cosine.mean(0).tolist()
    return output,records
