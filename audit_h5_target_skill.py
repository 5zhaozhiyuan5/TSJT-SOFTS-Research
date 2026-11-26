#!/usr/bin/env python3
"""H5 checkpoint-only target-conditioned source skill and gradient OT audit.

Loads the EXISTING ORIGINAL F2 Stage-I checkpoint + optimizer state.
No F2 training, no Stage-II, no true model/optimizer updates, no test labels.
Candidate ranking is specified without ever reading target PROBE futures.
The checkpoint has already trained on the support period: NOT independent
validation, even though each shadow update uses fit-only support labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
for path in (ROOT,ROOT/"model",ROOT/"model"/"Meta_Models"):
    sys.path.insert(0,str(path))

from audit_h4_error_profiles import load_model
from audit_f2_checkpoint import source_target_batch
from audit_h3_city_checkpoint import (
    sampled_flat_indices,_verify_model_invariance,_optimizer_snapshot
)
from h3_city_counterfactual import masked_support_split
from h1_shadow_audit import _grads
from h4_error_profile_counterfactual import shadow_profile_audit
from transfer_utility import compute_weights
from prediction_reliability import source_prediction_quality
from h5_target_conditioned_skill import (
    METHODS,source_band_capability,target_fit_demand,
    band_head_cosine,h5_candidate_alphas
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("experiments/h5/h5_results.jsonl"))
    p.add_argument("--probe-batches",type=int,default=8)
    p.add_argument("--source-points",type=int,default=256)
    p.add_argument("--target-points",type=int,default=128)
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--strength",type=float,default=0.5)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--config-filename",default="configs/config.yaml")
    p.add_argument("--data-list",default="chengdu_shenzhen_metr")
    p.add_argument("--test-dataset",default="pems-bay")
    p.add_argument("--daepoch",type=int,default=1500)
    p.add_argument("--ot-scale",type=float,default=0.01)
    a=p.parse_args()
    if (a.probe_batches<1 or a.source_points<0 or a.target_points<0 or
        0<a.source_points<32 or 0<a.target_points<16 or
        a.strength!=0.5 or a.ot_scale!=0.01 or a.daepoch!=1500):
        p.error("H5 uses fixed strength=0.5, OT scale=0.01, DAepoch=1500, and >=32/16 sampled points")
    if a.data_list!="chengdu_shenzhen_metr" or a.test_dataset!="pems-bay":
        p.error("H5 preregistered for Chengdu, Shenzhen, METR-LA -> PEMS-BAY")
    return a


def file_sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as fp:
        for part in iter(lambda:fp.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def main():
    args=parse_args()
    args.checkpoint=args.checkpoint.expanduser().resolve()
    args.output=args.output.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Missing EXISTING F2 checkpoint: "+str(args.checkpoint))
    os.chdir(ROOT)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.device=torch.device("cuda:0" if torch.cuda.is_available() and args.gpu>=0 else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

    wrapper,source,target,_,checkpoint_epoch=load_model(args)
    params=list(wrapper.model.parameters())
    param_snapshot={k:p.detach().cpu().clone() for k,p in wrapper.model.named_parameters()}
    optim_snapshot=_optimizer_snapshot(wrapper.main_optim)
    buffer_snapshot={k:b.detach().clone() for k,b in wrapper.model.named_buffers()}
    from geomloss.samples_loss import SamplesLoss
    criterion=SamplesLoss(loss="sinkhorn",p=1,backend="tensorized",
                           debias=False,blur=0.001)
    metadata={
        "audit_version":"h5-v1",
        "checkpoint":str(args.checkpoint),
        "checkpoint_sha256":file_sha256(args.checkpoint),
        "checkpoint_epoch":checkpoint_epoch,
        "training_updates":0,"stage2_runs":0,
        "seed":args.seed,"source_points":args.source_points,
        "target_points":args.target_points,
        "full_ot_geometry":args.source_points==args.target_points==0,
        "probes":args.probe_batches,"strength":args.strength,
        "ot_scale":args.ot_scale,"methods":["f2",*METHODS,"identity"],
        "target_support_only":True,"target_probe_seen_during_checkpoint_training":True,
        "target_fit_only_demand":True,
        "original_f2_beta_and_geometry":True,
        "optimizer":"original all-parameter one-sided PCGrad + stateful AdamW shadow",
        "candidate_change":"one source city alpha at a time",
        "target_probe_label_use":"AFTER all candidate OT gradients and shadow updates only",
        "checkpoint_warning":"NO independent holdout; correlated support windows",
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata,indent=2,sort_keys=True),encoding="utf-8")
    print("H5_META_JSON",json.dumps(metadata,sort_keys=True),flush=True)

    with args.output.open("w",encoding="utf-8") as fp:
      for batch_index in range(args.probe_batches):
        batch,_=source_target_batch(source,target,3)
        if len(batch)!=4:
            raise RuntimeError("Expected target+three distinct sources")
        city_names=[str(x.data_name) for x in batch[1:]]
        if len(set(city_names))!=3:
            raise RuntimeError("Duplicate source city")
        wrapper._transfer_records=[]
        wrapper._capture_transfer=True
        out=[];true=[];rep=[];future=[]
        try:
            for city_data in batch:
                prediction,label,feature,norm_future,_=wrapper.model(
                    city_data,stage="train"
                )
                out.append(prediction)
                true.append(label)
                rep.append(feature)
                future.append(norm_future)
        finally:
            wrapper._capture_transfer=False
        if len(wrapper._transfer_records)!=4:
            raise RuntimeError("Cannot retrieve projection2 features for H5")
        target_windows=out[0].shape[0]
        fit,probe=masked_support_split(target_windows,out[0].device)
        if bool((fit & probe).any()) or not bool(fit.any() and probe.any()):
            raise RuntimeError("Invalid target fit/probe partition")
        target_fit_loss=wrapper.loss_criterion(out[0][fit],true[0][fit])
        source_losses=[wrapper.loss_criterion(out[i],true[i]) for i in (1,2,3)]
        target_features,target_head=wrapper._transfer_records[0]
        target_delta=torch.autograd.grad(
            target_fit_loss,target_head,retain_graph=True
        )[0].detach()
        split=(15,15,target_windows-30)
        t_rep=torch.split(rep[0],split,dim=0)
        t_future=torch.split(future[0],split,dim=0)
        t_features=torch.split(target_features,split,dim=0)
        t_deltas=torch.split(target_delta,split,dim=0)
        t_preds=torch.split(out[0],split,dim=0)
        t_truths=torch.split(true[0],split,dim=0)
        t_masks=torch.split(fit,split,dim=0)
        entries=[];city_diagnostics=[]
        for i in range(3):
            city_data=batch[i+1]
            features,source_head=wrapper._transfer_records[i+1]
            source_delta=torch.autograd.grad(
                source_losses[i],source_head,retain_graph=True
            )[0].detach()
            sx=rep[i+1].reshape(-1,rep[i+1].shape[-1])
            sy=future[i+1].reshape(-1,36)
            sf=features.reshape(-1,features.shape[-1])
            sd=source_delta.reshape(-1,36)
            tx=t_rep[i][t_masks[i]].reshape(-1,sx.shape[-1])
            ty=t_future[i][t_masks[i]].reshape(-1,36)
            tf=t_features[i][t_masks[i]].reshape(-1,features.shape[-1])
            td=t_deltas[i][t_masks[i]].reshape(-1,36)
            sids=sampled_flat_indices(len(sx),args.source_points,sx.device)
            tids=sampled_flat_indices(len(tx),args.target_points,tx.device)
            alpha,beta,alpha_stats=compute_weights(
                sf.index_select(0,sids).unsqueeze(0).detach(),
                sd.index_select(0,sids).unsqueeze(0),
                tf.index_select(0,tids).unsqueeze(0).detach(),
                td.index_select(0,tids).unsqueeze(0),
                variant="gradient",temperature=8.0
            )
            ot_src=torch.cat([
                sx.index_select(0,sids),
                sy.detach().index_select(0,sids)*10.0
            ],dim=-1)
            ot_tgt=torch.cat([
                tx.index_select(0,tids),
                ty.detach().index_select(0,tids)*10.0
            ],dim=-1)
            f2_ot=criterion(alpha,ot_src,beta,ot_tgt).sum()*args.ot_scale
            f2_grad=_grads(f2_ot,params)

            history=torch.as_tensor(city_data.x)
            history_pool=history.reshape(
                -1,history.shape[-2],history.shape[-1]
            ).index_select(0,sids.to(device=history.device)).unsqueeze(0)
            src_pred=out[i+1].reshape(-1,36).index_select(0,sids)
            src_true=true[i+1].reshape(-1,36).index_select(0,sids)
            prc_quality,prc_stats=source_prediction_quality(
                src_pred.unsqueeze(0),src_true.unsqueeze(0),history_pool,
                city_data.means,city_data.stds
            )
            cap,cap_stats=source_band_capability(
                src_pred,src_true,history_pool[0],
                city_data.means,city_data.stds
            )
            demand,demand_stats=target_fit_demand(
                t_preds[i][t_masks[i]],
                t_truths[i][t_masks[i]],
                batch[0].stds
            )
            band_cos=band_head_cosine(
                sf.index_select(0,sids),sd.index_select(0,sids),
                tf.index_select(0,tids),td.index_select(0,tids)
            )
            candidates,candidate_stats=h5_candidate_alphas(
                alpha,prc_quality,cap,band_cos,demand,
                seed=args.seed*10007+batch_index*31+i,
                strength=args.strength
            )
            changed={}
            for method in METHODS:
                loss=criterion(
                    candidates[method],ot_src,beta,ot_tgt
                ).sum()*args.ot_scale
                changed[method]=_grads(loss,params)
                del loss
            # Zero-difference control on shadow optimizer+forward numerical noise.
            if i==0:
                changed["identity"]=f2_grad
            entries.append({"city":city_names[i],"f2":f2_grad,"changed":changed})
            city_diagnostics.append({
                "city":city_names[i],
                "source_points":int(len(sids)),
                "target_fit_points":int(len(tids)),
                "source_sample_sha256":hashlib.sha256(
                    sids.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "target_fit_sample_sha256":hashlib.sha256(
                    tids.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "f2_alpha_stats":alpha_stats,
                "prc_source":prc_stats,
                "source_band":cap_stats,
                "target_fit_demand":demand_stats,
                "candidate_alpha":candidate_stats,
                "head_cosine_min":band_cos.min(0).values.tolist(),
                "head_cosine_max":band_cos.max(0).values.tolist(),
                "source_full_points":int(len(sx)),
                "target_fit_full_points":int(len(tx))
            })
            del candidates,ot_src,ot_tgt,f2_ot
        report=shadow_profile_audit(
            model=wrapper.model,optimizer=wrapper.main_optim,
            source_task_loss=sum(source_losses),
            target_fit_task_loss=target_fit_loss,entries=entries,
            target_data=batch[0],probe_mask=probe,
            sample_index=batch_index,checkpoint_epoch=checkpoint_epoch,
            beta=wrapper.loss_criterion.beta
        )
        # H4's internal label is down:<city>:<method>; these are general
        # policies rather than downweighting groups. Rename for correct reading.
        def label(key):
            return ("policy:"+key[5:]) if key.startswith("down:") else key
        report["candidate_scores"]={
            label(k):v for k,v in report["candidate_scores"].items()
        }
        report["gains"]={
            label(k):{
                "gain_mae":v["suppress_gain_mae"],
                "gain_smooth_l1":v["suppress_gain_smooth_l1"],
                "gain_bands":v["suppress_gain_bands"],
                "gain_at_minutes":v["suppress_gain_at_minutes"]
            }
            for k,v in report["gains"].items()
        }
        report["audit_version"]="h5-v1"
        report["city_diagnostics"]=city_diagnostics
        report["full_ot_geometry"]=metadata["full_ot_geometry"]
        fp.write(json.dumps(report,sort_keys=True,allow_nan=False)+"\n")
        fp.flush()
        print("H5_RECORD",json.dumps({
            "batch":batch_index,
            "candidates":report["n_shadow_candidates"],
            "gain_vs_f2":{
                k:round(v["gain_mae"],8) for k,v in report["gains"].items()
            }
        },sort_keys=True),flush=True)
        del entries,report
        with torch.no_grad():
            for name,buffer in wrapper.model.named_buffers():
                buffer.copy_(buffer_snapshot[name])
        _verify_model_invariance(
            wrapper.model,param_snapshot,wrapper.main_optim,optim_snapshot
        )
        if any(p.grad is not None for p in params):
            raise RuntimeError("H5 modified real model .grad")
    print("H5_AUDIT_FINISHED",args.probe_batches,
          "training_updates",0,"output",args.output,flush=True)


if __name__=="__main__":
    main()
