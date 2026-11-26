"""H9: independent multi-step adaptation of original F2 checkpoint CLONES.

Two modes share this pipeline:
 H9-A --max-updates 1: causal one-step adaptation evaluated on a chronological
     post-support *development* period not in the checkpoint's 3-day training.
 H9-B --max-updates 20: 1/5/10/20 steps with freshly recomputed original
     GeomLoss OT, alpha/beta and target FIT gradients at EVERY update.

All policy runs replay identical support/source batches and stochastic draws,
each from its own F2 model/AdamW checkpoint copy. This is not official Stage-II.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random

import numpy as np
import torch

try:
    from .h1_shadow_audit import _grads, projected_directions
    from .h3_city_counterfactual import masked_support_split
    from .h8_direction_norm_audit import build_h8_policies, gradient_norm
    from .h9_temporal_holdout import score_development
    from .transfer_utility import compute_weights
except ImportError:
    from h1_shadow_audit import _grads, projected_directions
    from h3_city_counterfactual import masked_support_split
    from h8_direction_norm_audit import build_h8_policies, gradient_norm
    from h9_temporal_holdout import score_development
    from transfer_utility import compute_weights


POLICIES=("f2","global_half","projected_half","norm_matched_raw","all_off")
MILESTONES=(0,1,5,10,20)


def reproducible_step_seed(step, base=20271127):
    if not isinstance(step,int) or step<1:
        raise ValueError("Update number must be positive")
    return int(base)+step*971


def _rng_seed(step):
    seed=reproducible_step_seed(step)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@torch.no_grad()
def _fingerprint(samples):
    """Data identity for policy replay (metadata + selected tensor values)."""
    digest=hashlib.sha256()
    for x in samples:
        digest.update(str(x.data_name).encode("utf-8"))
        digest.update(str(tuple(x.x.shape)).encode("utf-8"))
        flat=x.x.detach().float().cpu().reshape(-1)
        if len(flat):
            indices=torch.linspace(0,len(flat)-1,min(256,len(flat))).long()
            digest.update(flat.index_select(0,indices).numpy().tobytes())
    return digest.hexdigest()


def clone_f2(model,optimizer):
    """Clone original model and original AdamW state; no writes to either."""
    replica=copy.deepcopy(model)
    # The original STRep projection2 capture hook closes over the ORIGINAL
    # wrapper. A deepcopy would retain that closure. Remove it only on clone;
    # register a local hook per update below.
    replica.forecasting_model.projection2._forward_hooks.clear()
    groups=optimizer.param_groups
    if len(groups)!=1:
        raise ValueError("Expected original F2 one-group AdamW")
    state=optimizer.state_dict()
    group=groups[0]
    cloned_opt=torch.optim.AdamW(
        replica.parameters(),
        lr=float(group["lr"]),
        betas=tuple(group["betas"]),
        eps=float(group["eps"]),
        weight_decay=float(group["weight_decay"]),
    )
    cloned_opt.load_state_dict(copy.deepcopy(state))
    return replica,cloned_opt


def _source_batch(source_dataset,target_dataset,step):
    """Identical target/source input selection across independently run policies."""
    try:
        from audit_f2_checkpoint import source_target_batch
    except ImportError as err:
        raise RuntimeError("Run H9 from the repository root") from err
    seed=_rng_seed(step)
    batch,_=source_target_batch(source_dataset,target_dataset,3)
    if len(batch)!=4 or len({x.data_name for x in batch})!=4:
        raise ValueError("Expected 3 distinct sources + PEMS-BAY target")
    return batch,seed,_fingerprint(batch)


def original_f2_gradient_bank(
    model, samples, *, loss_criterion, sinkhorn,source_points,target_points,
    ot_scale=.01,step=1
):
    """Compute exact original F2 loss gradient (or a logged point cap).

    Excludes the existing-support modulo-5 probe from FIT and OT alpha/beta.
    Never reads or references future chronological validation labels.
    """
    if len(samples)!=4 or not (source_points==0 or source_points>=32) or not (
        target_points==0 or target_points>=16
    ):
        raise ValueError("H9 expected 4 batches, source 0/>=32, target 0/>=16")
    named=list(model.named_parameters())
    params=[x for _,x in named]
    transfers=[]
    def record(_module,input_tensor,output):
        transfers.append((input_tensor[0],output))
    handle=model.forecasting_model.projection2.register_forward_hook(record)
    forecasts=[];truths=[];representations=[];futures=[]
    try:
        torch.manual_seed(352000+int(step))
        for sample in samples:
            pred,true,rep,norm_future,_=model(sample,stage="train")
            forecasts.append(pred);truths.append(true)
            representations.append(rep);futures.append(norm_future)
    finally:
        handle.remove()
    if len(transfers)!=4:
        raise RuntimeError("Missing projection2 capture from one or more cities")
    n=forecasts[0].shape[0]
    if n!=46:
        raise ValueError("Expected fixed 46 target-support windows like H8")
    fit,unused_probe=masked_support_split(n,forecasts[0].device)
    t_loss=loss_criterion(forecasts[0][fit],truths[0][fit])
    s_losses=[
        loss_criterion(forecasts[i],truths[i]) for i in (1,2,3)
    ]
    source_grad=_grads(sum(s_losses),params)
    target_grad=_grads(t_loss,params)
    target_features,target_out=transfers[0]
    target_delta=torch.autograd.grad(t_loss,target_out,retain_graph=True)[0].detach()
    split_sizes=(15,15,n-30)
    t_rep=torch.split(representations[0],split_sizes,dim=0)
    t_y=torch.split(futures[0],split_sizes,dim=0)
    t_f=torch.split(target_features,split_sizes,dim=0)
    t_d=torch.split(target_delta,split_sizes,dim=0)
    t_fit=torch.split(fit,split_sizes,dim=0)
    entries=[]
    ot_summaries=[]
    for j in range(3):
        source_city=samples[j+1].data_name
        f_s,out_s=transfers[j+1]
        delta_s=torch.autograd.grad(s_losses[j],out_s,retain_graph=True)[0].detach()
        sx=representations[j+1].reshape(-1,representations[j+1].shape[-1])
        sy=futures[j+1].reshape(-1,36)
        sf=f_s.reshape(-1,f_s.shape[-1])
        sd=delta_s.reshape(-1,36)
        tx=t_rep[j][t_fit[j]].reshape(-1,sx.shape[-1])
        ty=t_y[j][t_fit[j]].reshape(-1,36)
        tf=t_f[j][t_fit[j]].reshape(-1,sf.shape[-1])
        td=t_d[j][t_fit[j]].reshape(-1,36)
        from audit_h3_city_checkpoint import sampled_flat_indices
        sids=sampled_flat_indices(len(sx),source_points,sx.device)
        tids=sampled_flat_indices(len(tx),target_points,tx.device)
        alpha,beta,stats=compute_weights(
            sf.index_select(0,sids).unsqueeze(0).detach(),
            sd.index_select(0,sids).unsqueeze(0),
            tf.index_select(0,tids).unsqueeze(0).detach(),
            td.index_select(0,tids).unsqueeze(0),
            variant="gradient",temperature=8.0,
        )
        source_geometry=torch.cat((
            sx.index_select(0,sids),
            sy.detach().index_select(0,sids)*10.,
        ),dim=-1)
        target_geometry=torch.cat((
            tx.index_select(0,tids),
            ty.detach().index_select(0,tids)*10.,
        ),dim=-1)
        loss_ot=sinkhorn(alpha,source_geometry,beta,target_geometry).sum()*ot_scale
        g_ot=_grads(loss_ot,params)
        entries.append({"city":str(source_city),"f2":g_ot})
        with torch.no_grad():
            a=alpha.detach()
            b=beta.detach()
            top,loc=torch.topk(a,min(24,a.numel()))
            ot_summaries.append({
                "source_city":str(source_city),
                "target_group":j,
                "source_ot_points":int(len(sids)),
                "target_fit_ot_points":int(len(tids)),
                "source_point_ids_sha256":hashlib.sha256(
                    sids.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "target_point_ids_sha256":hashlib.sha256(
                    tids.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "source_alpha_sum":float(a.sum()),
                "source_alpha_ess":float(1/a.square().sum()),
                "source_alpha_min":float(a.min()),
                "source_alpha_max":float(a.max()),
                "source_alpha_top24_values":top.cpu().tolist(),
                "source_alpha_top24_flat_indices":
                    sids.index_select(0,loc).cpu().tolist(),
                "target_beta_sum":float(b.sum()),
                "target_beta_ess":float(1/b.square().sum()),
                "target_beta_uniform":bool(torch.allclose(
                    b,torch.full_like(b,1.0/len(b)),atol=1e-6
                )),
                "ot_loss":float(loss_ot.detach()),
                "raw_gradient_norm":gradient_norm(g_ot),
            })
        del loss_ot,source_geometry,target_geometry
    if len({e["city"] for e in entries})!=3:
        raise RuntimeError("Duplicate source-city OT gradient")
    return {
        "source_gradient":source_grad,
        "target_fit_gradient":target_grad,
        "city_entries":entries,
        "ot_metadata":ot_summaries,
        "target_fit_count":int(fit.sum()),
        "unused_support_probe_count":int(unused_probe.sum()),
    }


def apply_original_pcgrad_adamw(model,optimizer,bank,policy):
    """ACTUAL optimizer step, but ONLY on an independent model+AdamW copy."""
    if policy not in POLICIES:
        raise ValueError("Unknown H9 policy")
    g,stats=build_h8_policies(
        bank["city_entries"],bank["target_fit_gradient"]
    )
    _,direction=projected_directions(
        bank["source_gradient"],bank["target_fit_gradient"],g[policy]
    )
    if any(not bool(torch.isfinite(x).all()) for x in direction):
        raise FloatingPointError("Nonfinite H9 optimizer direction")
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for param,grad in zip(model.parameters(),direction):
            # Keep original F2 convention: even unused parameters get 0 grads
            # and thus participate in AdamW weight decay/update.
            param.grad=grad.detach().clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "policy":policy,
        "raw_ot_norm":stats["raw_gradient_norm"],
        "projected_ot_norm":stats["projected_gradient_norm"],
        "conflict_removed_norm":stats["removed_conflict_norm"],
        "raw_target_cos":stats["raw_target_gradient_cos"],
        "projection_triggered":stats["projection_triggered"],
        "actual_policy_ot_norm":gradient_norm(g[policy]),
        "target_fit_gradient_norm":gradient_norm(bank["target_fit_gradient"]),
        "source_forecast_gradient_norm":gradient_norm(bank["source_gradient"]),
        "pcgrad_update_norm":gradient_norm(direction),
        "optimizer_lr":float(optimizer.param_groups[0]["lr"]),
    }


def run_h9_adaptation(*,original_model,original_optimizer,
                      source_dataset,target_dataset,validation_batches,
                      sinkhorn,loss_criterion,source_points,target_points,
                      max_updates,eval_steps,seed=1):
    """Single H9-A step / H9-B checkpoints, with strict shared-batch replay.

    Each model starts from the EXACT same original checkpoint; each optimizer
    begins with its original F2 moments. Every step recomputes its OWN current
    alpha/beta/OT/target-FIT gradient, using identical chosen support/source
    examples. Only development validation metrics are inspected after steps,
    never used to decide the per-step OT gradient update.
    """
    if max_updates<1 or max_updates>20 or seed!=1:
        raise ValueError("H9 requires 1<=max_updates<=20, original seed=1")
    milestones=tuple(sorted(set(eval_steps)))
    if not milestones or milestones[0]!=0 or milestones[-1]!=max_updates:
        raise ValueError("Milestones must start at 0 and end at max_updates")
    if any(x<0 or x>max_updates for x in milestones):
        raise ValueError("Invalid milestone")
    scores={};step_logs={};source_manifests={}
    params_before={
        name:p.detach().clone() for name,p in original_model.named_parameters()
    }
    optimizer_before=copy.deepcopy(original_optimizer.state_dict())
    # Sequential by policy to control VRAM, NOT parallel copies of 5 models.
    # Re-init the 5 policies from identical original weights/moments.
    for policy in POLICIES:
        policy_model,policy_opt=clone_f2(original_model,original_optimizer)
        scores[policy]={0:score_development(policy_model,validation_batches)}
        logs=[]
        for step in range(1,max_updates+1):
            batch,batch_seed,fingerprint=_source_batch(
                source_dataset,target_dataset,step
            )
            if step not in source_manifests:
                source_manifests[step]=fingerprint
            elif source_manifests[step]!=fingerprint:
                raise RuntimeError("H9 policy source/target batches differ; abort")
            bank=original_f2_gradient_bank(
                policy_model,batch,loss_criterion=loss_criterion,
                sinkhorn=sinkhorn,
                source_points=source_points,target_points=target_points,
                step=step,
            )
            outcome=apply_original_pcgrad_adamw(
                policy_model,policy_opt,bank,policy
            )
            outcome.update({
                "step":step,
                "batch_seed":batch_seed,
                "support_batch_hash":fingerprint,
                "source_city_order":[str(x.data_name) for x in batch[1:]],
                "ot_source_stats":bank["ot_metadata"],
                "target_fit_count":bank["target_fit_count"],
                "support_probe_unused_in_training":
                    bank["unused_support_probe_count"],
            })
            logs.append(outcome)
            del bank,batch
            if step in milestones:
                scores[policy][step]=score_development(
                    policy_model,validation_batches
                )
                print("H9_MILESTONE",json.dumps({
                    "policy":policy,
                    "step":step,
                    "dev_mae":scores[policy][step]["mae"],
                    "dev_rmse":scores[policy][step]["rmse"],
                    "full_ot":source_points==target_points==0,
                },sort_keys=True),flush=True)
        step_logs[policy]=logs
        del policy_model,policy_opt
    for name,p in original_model.named_parameters():
        if not torch.equal(p.detach(),params_before[name]):
            raise RuntimeError("Original pretrained F2 parameters were modified")
    current=original_optimizer.state_dict()
    for k,state in optimizer_before["state"].items():
        for field,v in state.items():
            now=current["state"][k][field]
            if torch.is_tensor(v) and not torch.equal(v,now):
                raise RuntimeError("Original F2 optimizer state modified")
            if not torch.is_tensor(v) and v!=now:
                raise RuntimeError("Original F2 optimizer non-tensor state changed")
    return {
        "audit_version":"h9-v1",
        "original_f2_updates":0,
        "stage2_runs":0,
        "adaptation_update_count_per_policy":max_updates,
        "adaptation_policies":list(POLICIES),
        "evaluated_milestones":list(milestones),
        "source_points":source_points,
        "target_points":target_points,
        "full_ot_geometry":source_points==target_points==0,
        "scores":scores,
        "step_logs":step_logs,
        "shared_source_batch_hashes":{
            str(k):v for k,v in source_manifests.items()
        },
        "development_labels_in_update":False,
        "official_test_after_development_touched":False,
    }
