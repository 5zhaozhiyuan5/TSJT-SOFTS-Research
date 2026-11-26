#!/usr/bin/env python3
"""H4 checkpoint-only read-only F2 error-profile counterfactual audit.

Requires ORIGINAL F2 Stage-I checkpoint with optimizer state. This does not
train a new baseline, run Stage-II, or access target test labels. Probe windows
were already visible to the trained checkpoint; NOT a holdout generalization
study. Scripts should label capped OT points as approximation (smoke/pilot).
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
import yaml

ROOT=Path(__file__).resolve().parent
for path in (ROOT,ROOT/"model",ROOT/"model"/"Meta_Models"):
    sys.path.insert(0,str(path))

from datasets import traffic_dataset
from utils import get_data_list
from PMME import STRep
from transfer_utility import compute_weights
from h1_shadow_audit import _grads
from h3_city_counterfactual import masked_support_split
from audit_h3_city_checkpoint import (
    sampled_flat_indices,_verify_model_invariance,_optimizer_snapshot
)
from audit_f2_checkpoint import source_target_batch
from h4_error_profile_counterfactual import (
    GROUP_NAMES,error_profile_groups,downweight_group,shadow_profile_audit
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--output",default="experiments/h4-profile/h4_results.jsonl",
                   type=Path)
    p.add_argument("--probe-batches",type=int,default=4)
    p.add_argument("--source-points",type=int,default=256)
    p.add_argument("--target-points",type=int,default=128)
    p.add_argument("--min-group-points",type=int,default=12)
    p.add_argument("--min-alpha-mass",type=float,default=0.02)
    p.add_argument("--keep-fraction",type=float,default=0.5)
    p.add_argument("--groups",default="accurate,inaccurate,short_weak,mid_weak,long_weak,balanced")
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--config-filename",default="configs/config.yaml")
    p.add_argument("--data-list",default="chengdu_shenzhen_metr")
    p.add_argument("--test-dataset",default="pems-bay")
    p.add_argument("--daepoch",type=int,default=1500)
    p.add_argument("--ot-scale",type=float,default=0.01)
    a=p.parse_args()
    a.groups=tuple(x.strip() for x in a.groups.split(",") if x.strip())
    if len(set(a.groups))!=len(a.groups) or not set(a.groups).issubset(GROUP_NAMES):
        p.error("Unknown or duplicate error-profile group")
    if (a.probe_batches<1 or a.min_group_points<2 or
        a.source_points<0 or a.target_points<0 or
        (a.source_points and a.source_points<32) or
        (a.target_points and a.target_points<16) or
        not 0<a.keep_fraction<1 or
        not 0<=a.min_alpha_mass<0.5 or a.ot_scale<=0):
        p.error("Invalid H4 budget or perturbation")
    return a


def load_model(args):
    os.environ.update({
        "PMME_TRANSFER_VARIANT":"gradient",
        "PMME_F2_METHOD":"original",
        "PMME_PAIR_VARIANT":"f2",
        "PMME_PAIR_STRENGTH":"0",
        "PMME_FEEDBACK_MODE":"g0",
        "PMME_F2_ACCURACY_STRENGTH":"0",
        "PMME_F2_OT_SCHEDULE":"t0",
        "PMME_F2_DIAGNOSTIC":"off",
        "PMME_F2_DIAG_READONLY":"0",
        "PMME_TRANSFER_OT_SCALE":str(args.ot_scale),
        "PMME_TRANSFER_TEMPERATURE":"8.0",
    })
    with open(args.config_filename,encoding="utf-8") as fp:
        config=yaml.safe_load(fp)
    if int(config["task"]["source"]["batch_size"])!=16:
        raise ValueError("H4 requires original source batch size 16")
    data_args,task_args,model_args=(
        config["data"],config["task"],config["model"]
    )
    cities=get_data_list(args.data_list)
    if len(cities)!=3:raise ValueError("Exactly three source cities required")
    data_cfg={"data_list":args.data_list,"base_dir":ROOT,"device":args.device}
    source=traffic_dataset(
        data_args,task_args["source"],cities,"source",
        test_data=args.test_dataset
    )
    target=traffic_dataset(
        data_args,task_args["target_few_shot"],cities,"target_maml",
        test_data=args.test_dataset
    )
    args.save_dir=args.output.parent
    args.STmodel="SOFTS"
    args.new=1
    model=STRep(
        0.5,args.daepoch,1,100,0,1,"SOFTS","OT",
        data_args,task_args,model_args,data_cfg,args
    )
    checkpoint=torch.load(
        str(args.checkpoint),map_location=args.device,weights_only=True
    )
    if not isinstance(checkpoint,dict) or not all(
        x in checkpoint for x in ("model_state_dict","optimizer_state_dict","epoch")
    ):
        raise ValueError("Need full ORIGINAL F2 Stage-I checkpoint with AdamW state")
    epoch=int(checkpoint["epoch"])
    if epoch<args.daepoch:
        raise ValueError("Checkpoint was saved before F2 OT began")
    model.load_state_dict(checkpoint["model_state_dict"],strict=True)
    model.main_optim.load_state_dict(checkpoint["optimizer_state_dict"])
    return model,source,target,cities,epoch


def main():
    args=parse_args()
    args.checkpoint=args.checkpoint.expanduser().resolve()
    args.output=args.output.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Missing original F2 checkpoint: "+str(args.checkpoint))
    os.chdir(ROOT)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.device=torch.device(
        "cuda:0" if torch.cuda.is_available() and args.gpu>=0 else "cpu"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

    model,source,target,cities,checkpoint_epoch=load_model(args)
    params=list(model.model.parameters())
    param_snapshot={k:p.detach().cpu().clone()
                    for k,p in model.model.named_parameters()}
    optim_snapshot=_optimizer_snapshot(model.main_optim)
    buffer_snapshot={k:b.detach().clone()
                     for k,b in model.model.named_buffers()}
    from geomloss.samples_loss import SamplesLoss
    criterion=SamplesLoss(
        loss="sinkhorn",p=1,backend="tensorized",debias=False,blur=0.001
    )
    h=hashlib.sha256()
    with args.checkpoint.open("rb") as fp:
        for part in iter(lambda:fp.read(1024*1024),b""):
            h.update(part)
    metadata={
        "audit_version":"h4-v1",
        "source_checkpoint":str(args.checkpoint),
        "checkpoint_sha256":h.hexdigest(),
        "checkpoint_epoch":checkpoint_epoch,
        "training_updates":0,
        "target_split":"3-day SUPPORT already used by checkpoint; "
                       "modulo-5 fit/probe; in-support exploration ONLY",
        "seed":args.seed,
        "data_list":args.data_list,
        "target_city":args.test_dataset,
        "source_cities":cities,
        "groups":args.groups,
        "source_points":args.source_points,
        "target_points":args.target_points,
        "full_ot_geometry":args.source_points==args.target_points==0,
        "min_group_points":args.min_group_points,
        "min_alpha_mass":args.min_alpha_mass,
        "keep_fraction":args.keep_fraction,
        "ot_scale":args.ot_scale,
        "ot_cost":"Original F2 GeomLoss(p=1,blur=.001,debias=False), "
                  "concat(SOFTS_rep,10*RevIN_future)",
        "source_profiles":"physical-speed source TRAIN forecasting errors "
                          "normalized by source-city std, 3x12-step bands",
        "replay":"all-parameter one-sided PCGrad + checkpoint AdamW state",
        "label_leakage_guard":"No probe future in OT alpha/beta, target-fit "
                              "gradient or virtual optimizer update; shadow "
                              "target.y forced zero",
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata,sort_keys=True,indent=2),encoding="utf-8"
    )
    print("H4_META_JSON",json.dumps(metadata,sort_keys=True),flush=True)
    with args.output.open("w",encoding="utf-8") as fp:
      for batch_index in range(args.probe_batches):
        data,_=source_target_batch(source,target,3)
        if len(data)!=4:raise ValueError("Expected one target and 3 sources")
        model._transfer_records=[]
        model._capture_transfer=True
        out=[];true=[];rep=[];future=[]
        try:
            for city_data in data:
                forecast,label,feature,norm_future,_=model.model(
                    city_data,stage="train"
                )
                out.append(forecast)
                true.append(label)
                rep.append(feature)
                future.append(norm_future)
        finally:
            model._capture_transfer=False
        if len(model._transfer_records)!=4:
            raise RuntimeError("Could not obtain 4 projection2 features")
        b=out[0].shape[0]
        fit,probe=masked_support_split(b,out[0].device)
        target_fit_loss=model.loss_criterion(
            out[0][fit],true[0][fit]
        )
        source_losses=[model.loss_criterion(out[i],true[i]) for i in (1,2,3)]
        target_features,target_head=model._transfer_records[0]
        target_delta=torch.autograd.grad(
            target_fit_loss,target_head,retain_graph=True
        )[0].detach()
        split=(15,15,b-30)
        t_rep=torch.split(rep[0],split,dim=0)
        t_future=torch.split(future[0],split,dim=0)
        t_features=torch.split(target_features,split,dim=0)
        t_deltas=torch.split(target_delta,split,dim=0)
        t_masks=torch.split(fit,split,dim=0)
        entries=[];city_info=[]
        for index in range(3):
            city_data=data[index+1]
            features,source_head=model._transfer_records[index+1]
            source_delta=torch.autograd.grad(
                source_losses[index],source_head,retain_graph=True
            )[0].detach()
            group=index
            tf=t_features[group][t_masks[group]].reshape(-1,features.shape[-1])
            td=t_deltas[group][t_masks[group]].reshape(-1,36)
            sx=rep[index+1].reshape(-1,rep[index+1].shape[-1])
            sy=future[index+1].reshape(-1,36)
            tx=t_rep[group][t_masks[group]].reshape(-1,sx.shape[-1])
            ty=t_future[group][t_masks[group]].reshape(-1,36)
            sf=features.reshape(-1,features.shape[-1])
            sd=source_delta.reshape(-1,36)
            sids=sampled_flat_indices(len(sx),args.source_points,sx.device)
            tids=sampled_flat_indices(len(tx),args.target_points,tx.device)
            alpha,beta,alpha_stats=compute_weights(
                sf.index_select(0,sids).unsqueeze(0).detach(),
                sd.index_select(0,sids).unsqueeze(0),
                tf.index_select(0,tids).unsqueeze(0).detach(),
                td.index_select(0,tids).unsqueeze(0),
                variant="gradient",temperature=8.0
            )
            x=torch.cat([sx.index_select(0,sids),
                         sy.detach().index_select(0,sids)*10.0],dim=-1)
            y=torch.cat([tx.index_select(0,tids),
                         ty.detach().index_select(0,tids)*10.0],dim=-1)
            loss_f2=criterion(alpha,x,beta,y).sum()*args.ot_scale
            f2_grad=_grads(loss_f2,params)
            src_shape=out[index+1].shape
            input_history=torch.as_tensor(city_data.x)
            pred_pool=out[index+1].reshape(-1,36).index_select(0,sids)
            truth_pool=true[index+1].reshape(-1,36).index_select(0,sids)
            history_pool=input_history.reshape(
                -1,input_history.shape[-2],input_history.shape[-1]
            ).index_select(0,sids.cpu()).unsqueeze(0)
            # n points x one synthetic node dimension; groups flatten [n].
            masks,profile_stats=error_profile_groups(
                pred_pool.unsqueeze(0),truth_pool.unsqueeze(0),
                history_pool,city_data.means,city_data.stds
            )
            changed={};perturb_stats={};skipped={}
            for profile in args.groups:
                mask=masks[profile]
                count=int(mask.sum().item())
                mass=float(alpha[mask].sum().item())
                if count<args.min_group_points or mass<args.min_alpha_mass:
                    skipped[profile]={"count":count,"alpha_mass":mass,
                                      "reason":"insufficient_support_or_mass"}
                    continue
                weighted,details=downweight_group(
                    alpha,mask,keep_fraction=args.keep_fraction
                )
                group_loss=criterion(weighted,x,beta,y).sum()*args.ot_scale
                changed[profile]=_grads(group_loss,params)
                perturb_stats[profile]=details
                del group_loss
            entries.append({
                "city":str(city_data.data_name),
                "f2":f2_grad,"changed":changed
            })
            city_info.append({
                "city":str(city_data.data_name),
                "target_group":group,
                "source_points":len(sids),"target_fit_points":len(tids),
                "source_full_points":len(sx),"target_full_fit_points":len(tx),
                "source_batch_shape":list(src_shape),
                "source_selected_ids_sha256":hashlib.sha256(
                    sids.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "source_group_profile":profile_stats,
                "f2_alpha_stats":alpha_stats,
                "interventions":perturb_stats,
                "skipped":skipped,
            })
            del x,y,loss_f2,changed
        report=shadow_profile_audit(
            model=model.model,optimizer=model.main_optim,
            source_task_loss=sum(source_losses),
            target_fit_task_loss=target_fit_loss,
            entries=entries,target_data=data[0],probe_mask=probe,
            sample_index=batch_index,checkpoint_epoch=checkpoint_epoch,
            beta=model.loss_criterion.beta
        )
        report["city_profiles"]=city_info
        report["full_ot_geometry"]=metadata["full_ot_geometry"]
        fp.write(json.dumps(report,sort_keys=True,allow_nan=False)+"\n")
        fp.flush()
        print("H4_RECORD",json.dumps({
            "batch":batch_index,
            "candidates":report["n_shadow_candidates"],
            "gain_vs_f2":{
                key:round(v["suppress_gain_mae"],8)
                for key,v in report["gains"].items()
            }
        },sort_keys=True),flush=True)
        del entries,report
        with torch.no_grad():
            for name,buffer in model.model.named_buffers():
                buffer.copy_(buffer_snapshot[name])
        _verify_model_invariance(
            model.model,param_snapshot,model.main_optim,optim_snapshot
        )
        if any(p.grad is not None for p in params):
            raise RuntimeError("Read-only H4 populated actual model .grad")
    print("H4_AUDIT_FINISHED",args.probe_batches,
          "training_updates",0,"output",args.output,flush=True)


if __name__=="__main__":
    main()
