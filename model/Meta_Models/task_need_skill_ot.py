"""TDSM-OT: target-demand/source-skill matched bilateral Sinkhorn marginals.

Actual Stage-I training module. Unlike PRC, a source pattern is emphasized
only when it forecasts reliably AND matches an error-needing target SUPPORT
pattern. Target beta is also corrected using matched source skill.

All quantities are detached training predictions/labels, not test data.
Uses O((Ns+Nt)*K*3) descriptor-prototype matching. No full NxM pair matrix,
new learnable parameters, changes to GeomLoss ground cost, SOFTS or Stage-II.
Not a claim of previously unreported novelty or verified accuracy gain.
"""
from __future__ import annotations
import math
import torch


@torch.no_grad()
def dynamics_and_quality(prediction, truth, history, mean, std, role="source"):
    """Return [B*N,3,3] history-to-future response, [B*N,3] skill/need.

    prediction,truth [B,N,36] physical speed; history [B,N,H,>=1] with
    normalized speed at channel 0. Labels are ONLY Stage-I training data.
    """
    if role not in ("source","target"):
        raise ValueError("Expected source or target")
    if (prediction.ndim!=3 or prediction.shape!=truth.shape or
        prediction.shape[-1]!=36 or history.ndim!=4 or
        history.shape[:2]!=prediction.shape[:2] or
        history.shape[-2]<12 or history.shape[-1]<1):
        raise ValueError("Expected forecast/label [B,N,36] and history [B,N,H,C]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(truth).all():
        raise ValueError("Nonfinite forecast or label")
    device=prediction.device
    pred=prediction.detach().float()
    actual=truth.detach().float()
    hist=history.detach()[...,0].to(device=device,dtype=torch.float32)
    mean=torch.as_tensor(mean,device=device,dtype=torch.float32).reshape(-1)
    std=torch.as_tensor(std,device=device,dtype=torch.float32).reshape(-1)
    if (mean.numel()!=1 or std.numel()!=1 or
        not bool(torch.isfinite(mean).all()) or
        not bool(torch.isfinite(std).all()) or float(std[0])<=0):
        raise ValueError("City mean/std must be finite scalars with std>0")
    future=(actual-mean[0])/std[0]
    forecast=(pred-mean[0])/std[0]
    baseline=hist[...,-12:].mean(-1)
    variability=hist.std(-1,unbiased=False).clamp_min(0.25)
    last=hist[...,-1]
    descriptors=[]
    qualities=[]
    for start in (0,12,24):
        y=future[...,start:start+12]
        yp=forecast[...,start:start+12]
        rel=((y-baseline.unsqueeze(-1))/variability.unsqueeze(-1)).clamp(-8,8)
        descriptors.append(torch.stack((
            rel.mean(-1),rel[...,-4:].mean(-1)-rel[...,:4].mean(-1),
            rel.std(-1,unbiased=False)
        ),dim=-1))
        error=(yp-y).abs().mean(-1)
        if role=="source":
            persistence=(last.unsqueeze(-1)-y).abs().mean(-1)
            absolute=torch.exp(-error/error.median().clamp_min(0.1))
            skill=torch.sigmoid(((persistence-error)/
                                 persistence.median().clamp_min(0.1)).clamp(-20,20))
            score=(absolute*skill).clamp(0,1)
        else:
            scale=error.median().clamp_min(0.1)
            score=torch.sigmoid(((error-scale)/scale).clamp(-20,20))
        qualities.append(score)
    response=torch.stack(descriptors,dim=-2).reshape(-1,3,3).detach()
    quality=torch.stack(qualities,dim=-1).reshape(-1,3).detach()
    if not bool(torch.isfinite(response).all()) or not bool(torch.isfinite(quality).all()):
        raise FloatingPointError("Nonfinite dynamics/quality")
    return response,quality


@torch.no_grad()
def _anchors(mass,quality,budget):
    """Include both high-value and spread-out coverage prototypes."""
    n=len(mass)
    k=min(budget,n)
    top=(mass*quality).topk(max(1,k//2)).indices
    spaced=torch.linspace(0,n-1,steps=k+2,device=mass.device)[1:-1].round().long()
    # Preserve high-value anchors first, then add regular coverage.
    result=[]
    visited=set()
    for idx in torch.cat((top,spaced)).tolist():
        if idx not in visited:
            result.append(idx);visited.add(idx)
        if len(result)>=k:
            break
    return torch.tensor(result,device=mass.device,dtype=torch.long)


@torch.no_grad()
def calibrate_bilateral(alpha_prior,beta_prior,source_response,source_skill,
                        target_response,target_need,*,strength=0.35,
                        prototypes=32,temperature=1.0):
    """Two-way prediction-conditioned OT mass for *one city/target group*.

    source utility: skill(source) x similarity(source,target-proto)
                    x target forecast need.
    target utility: need(target) x similarity(target,source-proto)
                    x source forecast skill.
    The conditional interaction is the new mechanism, not simply PRC+BIOT.
    """
    if (not math.isfinite(strength) or not 0<=strength<=1 or
        not isinstance(prototypes,int) or prototypes<2 or
        not math.isfinite(temperature) or temperature<=0):
        raise ValueError("Invalid TDSM strength/prototype/temperature")
    if (alpha_prior.ndim!=1 or beta_prior.ndim!=1 or
        len(alpha_prior)<2 or len(beta_prior)<2 or
        source_response.shape!=(len(alpha_prior),3,3) or
        target_response.shape!=(len(beta_prior),3,3) or
        source_skill.shape!=(len(alpha_prior),3) or
        target_need.shape!=(len(beta_prior),3)):
        raise ValueError("Expected flat matching source/target OT sample dimensions")
    if any(not bool(torch.isfinite(x).all()) for x in (
        alpha_prior,beta_prior,source_response,source_skill,
        target_response,target_need)):
        raise ValueError("Nonfinite TDSM inputs")
    if (bool((alpha_prior<0).any()) or bool((beta_prior<0).any()) or
        not bool(torch.isclose(alpha_prior.sum(),alpha_prior.new_tensor(1.),atol=1e-5)) or
        not bool(torch.isclose(beta_prior.sum(),beta_prior.new_tensor(1.),atol=1e-5)) or
        bool((source_skill<0).any()) or bool((source_skill>1).any()) or
        bool((target_need<0).any()) or bool((target_need>1).any())):
        raise ValueError("Marginals must sum to one, qualities in [0,1]")
    if strength==0:
        return alpha_prior,beta_prior,{"strength":0.,"source_l1_shift":0.,
            "target_l1_shift":0.,"source_prototypes":0,"target_prototypes":0}
    a=alpha_prior.detach().float()
    b=beta_prior.detach().float()
    sr=source_response.detach().float()
    tr=target_response.detach().float()
    sq=source_skill.detach().float()
    tn=target_need.detach().float()
    src_idx=_anchors(a,sq.mean(-1),prototypes)
    tgt_idx=_anchors(b,tn.mean(-1),prototypes)
    src_value=torch.zeros_like(a)
    tgt_value=torch.zeros_like(b)
    for band in range(3):
        # Independent short/mid/long shape comparisons; pair affinities never
        # contain the raw full NxM GeomLoss transport matrix.
        ss=torch.exp(-torch.cdist(sr[:,band],tr[tgt_idx,band])/temperature)
        tt=torch.exp(-torch.cdist(tr[:,band],sr[src_idx,band])/temperature)
        src_value+=(sq[:,band]*(ss*tn[tgt_idx,band].unsqueeze(0)).mean(1))/3
        tgt_value+=(tn[:,band]*(tt*sq[src_idx,band].unsqueeze(0)).mean(1))/3

    def reweight(prior,value):
        avg=(prior*value).sum()
        if float(avg)<=1e-10:
            return prior.clone()
        conditional=(value/avg).clamp(0,2)
        proposed=prior*conditional
        proposed=proposed/proposed.sum().clamp_min(1e-12)
        adjusted=(1-strength)*prior+strength*proposed
        return adjusted/adjusted.sum().clamp_min(1e-12)

    adjusted_a=reweight(a,src_value)
    adjusted_b=reweight(b,tgt_value)
    if any(not bool(torch.isfinite(x).all()) for x in (adjusted_a,adjusted_b)):
        raise FloatingPointError("Nonfinite TDSM marginal")
    diagnostics={
        "strength":float(strength),
        "source_prototypes":int(len(src_idx)),
        "target_prototypes":int(len(tgt_idx)),
        "source_skill_mean":float(sq.mean()),
        "target_need_mean":float(tn.mean()),
        "source_matching_utility_mean":float(src_value.mean()),
        "target_matching_utility_mean":float(tgt_value.mean()),
        "source_l1_shift":float((adjusted_a-a).abs().sum()),
        "target_l1_shift":float((adjusted_b-b).abs().sum()),
        "source_ess_frac":float(1/(adjusted_a.square().sum()*len(a))),
        "target_ess_frac":float(1/(adjusted_b.square().sum()*len(b))),
        "source_relative_min":float((adjusted_a/a.clamp_min(1e-30)).min()),
        "target_relative_min":float((adjusted_b/b.clamp_min(1e-30)).min()),
        "support_labels_only":True,
    }
    return (adjusted_a.to(dtype=alpha_prior.dtype).detach(),
            adjusted_b.to(dtype=beta_prior.dtype).detach(),diagnostics)
