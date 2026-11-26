#!/usr/bin/env python3
"""H6: checkpoint-only F2 OT gradient contribution and strength/city audit.

No training, no Stage-II, no target TEST loader, no optimizer.step(), and no
candidate selection from target probe labels. Shadow probes are PREVIOUSLY-SEEN
target support: this is diagnostic evidence only, not independent validation.
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
for directory in (ROOT,ROOT/"model",ROOT/"model"/"Meta_Models"):
    sys.path.insert(0,str(directory))

from audit_h4_error_profiles import load_model
from audit_f2_checkpoint import source_target_batch
from audit_h3_city_checkpoint import (
    sampled_flat_indices,_verify_model_invariance,_optimizer_snapshot
)
from h3_city_counterfactual import masked_support_split
from transfer_utility import compute_weights
from h1_shadow_audit import _grads
from h6_ot_gradient_effect import shadow_h6_audit


OFFICIAL_EXISTING_F2_SHA = (
    "f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf"
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("experiments/h6/h6_results.jsonl"))
    p.add_argument("--probe-batches",type=int,default=8)
    p.add_argument("--source-points",type=int,default=256,
                   help="0=full source batch (official OT geometry)")
    p.add_argument("--target-points",type=int,default=128,
                   help="0=all target fit points (official OT geometry)")
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--config-filename",default="configs/config.yaml")
    p.add_argument("--data-list",default="chengdu_shenzhen_metr")
    p.add_argument("--test-dataset",default="pems-bay")
    p.add_argument("--daepoch",type=int,default=1500)
    p.add_argument("--ot-scale",type=float,default=0.01)
    p.add_argument("--expected-checkpoint-sha256",
                   default=OFFICIAL_EXISTING_F2_SHA,
                   help="Required exact original-F2 checkpoint SHA; no bypass")
    args=p.parse_args()
    if (args.probe_batches<1 or args.source_points<0 or args.target_points<0 or
        (0<args.source_points<32) or (0<args.target_points<16) or
        args.daepoch!=1500 or args.ot_scale!=0.01 or
        args.data_list!="chengdu_shenzhen_metr" or
        args.test_dataset!="pems-bay" or args.seed!=1):
        p.error("H6 uses fixed original F2 geometry/config/seed; "
                "source_points=0 or >=32, target_points=0 or >=16")
    if len(args.expected_checkpoint_sha256)!=64:
        p.error("Expected SHA256 must have 64 hex characters")
    return args


def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda:fp.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def sha256_tensor_ids(indices):
    return hashlib.sha256(
        indices.detach().cpu().numpy().tobytes()
    ).hexdigest()


def main():
    args=parse_args()
    args.checkpoint=args.checkpoint.expanduser().resolve()
    args.output=args.output.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Original F2 checkpoint missing; refusing to train it")
    sha=sha256_file(args.checkpoint)
    if sha.lower()!=args.expected_checkpoint_sha256.lower():
        raise ValueError("F2 checkpoint SHA256 mismatch. Expected "+args.expected_checkpoint_sha256+
                         " got "+sha+". Do not substitute PRC or another model.")
    os.chdir(ROOT)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.device=torch.device(
        "cuda:0" if torch.cuda.is_available() and args.gpu>=0 else "cpu"
    )
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    wrapper,source,target,city_list,epoch=load_model(args)
    params=list(wrapper.model.parameters())
    param_snapshot={n:p.detach().cpu().clone()
                    for n,p in wrapper.model.named_parameters()}
    optim_snapshot=_optimizer_snapshot(wrapper.main_optim)
    buffer_snapshot={n:b.detach().clone()
                     for n,b in wrapper.model.named_buffers()}
    from geomloss.samples_loss import SamplesLoss
    criterion=SamplesLoss(loss="sinkhorn",p=1,backend="tensorized",
                          blur=0.001,debias=False)
    metadata={
        "audit_version":"h6-v1","checkpoint":str(args.checkpoint),
        "checkpoint_sha256":sha,"checkpoint_epoch":int(epoch),
        "seed":args.seed,"source_cities":city_list,
        "target_city":args.test_dataset,"source_points":args.source_points,
        "target_points":args.target_points,
        "full_ot_geometry":args.source_points==args.target_points==0,
        "probe_batches":args.probe_batches,
        "ot_cost":"original GeomLoss(p=1,blur=.001,debias=False); "
                  "concat(SOFTS_rep,10*RevIN_true_future)",
        "original_ot_scale":0.01,
        "policy_grid":{"global_scales":[0,0.25,0.5,1],
                       "city_factors":[0,0.5,1]},
        "training_updates":0,"stage2_runs":0,
        "target_support_only":True,
        "target_probe_previously_seen_by_checkpoint":True,
        "probe_for_policy_selection":False,
        "optimizer":"all-parameter one-sided PCGrad, original AdamW state",
        "transport_matrix_explicitly_materialized":False,
        "warning":"Support windows overlap in time; no independent test MAE"
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata,indent=2,sort_keys=True),encoding="utf-8"
    )
    print("H6_META_JSON",json.dumps(metadata,sort_keys=True),flush=True)

    with args.output.open("w",encoding="utf-8") as fp:
      for batch_index in range(args.probe_batches):
        data,_=source_target_batch(source,target,3)
        if len(data)!=4:raise RuntimeError("Expected target+three source cities")
        wrapper._transfer_records=[]
        wrapper._capture_transfer=True
        forecast=[];truth=[];rep=[];future=[]
        try:
            for sample in data:
                pred,y,z,y_norm,_=wrapper.model(sample,stage="train")
                forecast.append(pred);truth.append(y);rep.append(z);future.append(y_norm)
        finally:
            wrapper._capture_transfer=False
        if len(wrapper._transfer_records)!=4:
            raise RuntimeError("Original F2 projection2 feature capture failed")
        b=forecast[0].shape[0]
        fit,probe=masked_support_split(b,forecast[0].device)
        if bool((fit & probe).any()) or not bool(fit.any() and probe.any()):
            raise RuntimeError("Target fit/probe overlap or empty")
        loss_fit=wrapper.loss_criterion(forecast[0][fit],truth[0][fit])
        losses_source=[wrapper.loss_criterion(forecast[i],truth[i])
                       for i in (1,2,3)]
        ft,head_target=wrapper._transfer_records[0]
        delta_t=torch.autograd.grad(loss_fit,head_target,retain_graph=True)[0].detach()
        slices=(15,15,b-30)
        rep_t=torch.split(rep[0],slices,dim=0)
        future_t=torch.split(future[0],slices,dim=0)
        feature_t=torch.split(ft,slices,dim=0)
        grad_t=torch.split(delta_t,slices,dim=0)
        fit_t=torch.split(fit,slices,dim=0)
        entries=[];city_metadata=[]
        for j in range(3):
            city=str(data[j+1].data_name)
            fs,head_source=wrapper._transfer_records[j+1]
            ds=torch.autograd.grad(losses_source[j],head_source,retain_graph=True)[0].detach()
            sx=rep[j+1].reshape(-1,rep[j+1].shape[-1])
            sy=future[j+1].reshape(-1,36)
            sf=fs.reshape(-1,fs.shape[-1])
            sd=ds.reshape(-1,36)
            tx=rep_t[j][fit_t[j]].reshape(-1,sx.shape[-1])
            ty=future_t[j][fit_t[j]].reshape(-1,36)
            tf=feature_t[j][fit_t[j]].reshape(-1,sf.shape[-1])
            td=grad_t[j][fit_t[j]].reshape(-1,36)
            src_ids=sampled_flat_indices(len(sx),args.source_points,sx.device)
            target_ids=sampled_flat_indices(len(tx),args.target_points,tx.device)
            alpha,beta,stats=compute_weights(
                sf.index_select(0,src_ids).unsqueeze(0).detach(),
                sd.index_select(0,src_ids).unsqueeze(0),
                tf.index_select(0,target_ids).unsqueeze(0).detach(),
                td.index_select(0,target_ids).unsqueeze(0),
                variant="gradient",temperature=8.0
            )
            z_source=torch.cat([
                sx.index_select(0,src_ids),
                sy.detach().index_select(0,src_ids)*10.0
            ],dim=-1)
            z_target=torch.cat([
                tx.index_select(0,target_ids),
                ty.detach().index_select(0,target_ids)*10.0
            ],dim=-1)
            real_ot=criterion(alpha,z_source,beta,z_target).sum()*args.ot_scale
            ot_grad=_grads(real_ot,params)
            entries.append({"city":city,"f2":ot_grad})
            with torch.no_grad():
                top,loc=torch.topk(alpha.detach(),min(24,len(alpha)))
                city_metadata.append({
                    "city":city,"target_group":j,
                    "source_full_points":int(len(sx)),
                    "target_fit_full_points":int(len(tx)),
                    "source_ot_points":int(len(src_ids)),
                    "target_fit_ot_points":int(len(target_ids)),
                    "source_sample_ids_sha256":sha256_tensor_ids(src_ids),
                    "target_fit_sample_ids_sha256":sha256_tensor_ids(target_ids),
                    "f2_alpha_ess":float(1/alpha.square().sum()),
                    "f2_alpha_min":float(alpha.min()),
                    "f2_alpha_max":float(alpha.max()),
                    "f2_alpha_sum":float(alpha.sum()),
                    "f2_top24_flat_indices":
                        src_ids.index_select(0,loc).cpu().tolist(),
                    "f2_top24_alpha":top.cpu().tolist(),
                    "target_beta_uniform":bool(torch.allclose(
                        beta,torch.full_like(beta,1.0/len(beta)),atol=1e-6)),
                    "target_beta_ess":float(1/beta.square().sum()),
                    "target_beta_sum":float(beta.sum()),
                    "f2_weight_diagnostics":stats,
                    "ot_loss_value":float(real_ot.detach()),
                })
            del real_ot,z_source,z_target
        report=shadow_h6_audit(
            model=wrapper.model,optimizer=wrapper.main_optim,
            source_task_loss=sum(losses_source),
            target_fit_task_loss=loss_fit,
            city_entries=entries,
            target_data=data[0],probe_mask=probe,
            sample_index=batch_index,checkpoint_epoch=epoch,
            beta=wrapper.loss_criterion.beta
        )
        report["city_metadata"]=city_metadata
        report["full_ot_geometry"]=metadata["full_ot_geometry"]
        fp.write(json.dumps(report,allow_nan=False,sort_keys=True)+"\n")
        fp.flush()
        print("H6_RECORD",json.dumps({
            "batch":batch_index,
            "shadows":report["n_shadow_candidates"],
            "gain_vs_f2":{
                k:round(v["gain_mae"],9) for k,v in report["gains"].items()
            },
            "raw_ot_norm":round(report["gradient_report"]["raw_ot_norm"],6),
            "pcgrad_effect_norm":round(report["gradient_report"][
                "effective_ot_after_pcgrad_norm"],6),
        },sort_keys=True),flush=True)
        del entries,report
        with torch.no_grad():
            for name,buf in wrapper.model.named_buffers():
                buf.copy_(buffer_snapshot[name])
        _verify_model_invariance(
            wrapper.model,param_snapshot,wrapper.main_optim,optim_snapshot
        )
        if any(p.grad is not None for p in params):
            raise RuntimeError("Read-only H6 populated real .grad")
    print("H6_AUDIT_FINISHED",args.probe_batches,
          "training_updates 0","stage2_runs 0",
          "output",args.output,flush=True)


if __name__=="__main__":
    main()
