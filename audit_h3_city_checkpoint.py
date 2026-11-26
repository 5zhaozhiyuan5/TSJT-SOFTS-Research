#!/usr/bin/env python3
"""Run H3 city-level read-only counterfactual from an EXISTING original-F2
Stage-I checkpoint. No Stage-II, no train loop, no optimizer.step, NO target
test dataset access. This is a three-day-SUPPORT shadow diagnostic only.

Run:
python -u audit_h3_city_checkpoint.py --checkpoint /existing/F2/stage1.pth \
    --probe-batches 8 --output experiments/h3/h3_results.jsonl

For an inexpensive real-GeomLoss smoke, add:
    --source-points 256 --target-points 128 --probe-batches 1
Default source/target-points=0 means full original city batches.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
for directory in (ROOT, ROOT/"model", ROOT/"model"/"Meta_Models"):
    sys.path.insert(0,str(directory))

from datasets import traffic_dataset
from utils import get_data_list
from PMME import STRep
from transfer_utility import compute_weights
from h1_shadow_audit import _grads
from h3_city_counterfactual import masked_support_split, shadow_city_audit
from audit_f2_checkpoint import source_target_batch


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("experiments/h3/h3_results.jsonl"))
    p.add_argument("--config-filename",default="configs/config.yaml")
    p.add_argument("--data-list",default="chengdu_shenzhen_metr")
    p.add_argument("--test-dataset",default="pems-bay")
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--probe-batches",type=int,default=8)
    p.add_argument("--source-points",type=int,default=0,
                   help="0 is exact full source batch; positive is SMOKE only")
    p.add_argument("--target-points",type=int,default=0,
                   help="0 is exact full target-fit batch; positive is SMOKE only")
    p.add_argument("--daepoch",type=int,default=1500)
    p.add_argument("--ot-scale",type=float,default=0.01)
    p.add_argument("--gpu",type=int,default=0)
    args=p.parse_args()
    if args.probe_batches < 1 or args.source_points < 0 or args.target_points < 0:
        p.error("Bad probe/budget settings")
    if args.daepoch < 0 or args.ot_scale <= 0:
        p.error("Positive OT scale and nonnegative DA epoch required")
    return args


def sampled_flat_indices(n, cap, device):
    if cap == 0 or cap >= n:
        return torch.arange(n,device=device)
    if cap < 2:
        raise ValueError("At least 2 OT points required when subsampling")
    return torch.linspace(0,n-1,cap,device=device).round().long().unique()


def _verify_model_invariance(model, model_snapshot, optimizer, optim_snapshot):
    with torch.no_grad():
        for name,param in model.named_parameters():
            if not torch.equal(param.detach().cpu(),model_snapshot[name]):
                raise RuntimeError("H3 mutated model parameter "+name)
        for p,fields in optimizer.state.items():
            initial=optim_snapshot[id(p)]
            for key,value in fields.items():
                old=initial[key]
                if torch.is_tensor(value):
                    if not torch.equal(value.detach().cpu(),old):
                        raise RuntimeError("H3 mutated optimizer state "+key)
                elif value!=old:
                    raise RuntimeError("H3 mutated optimizer scalar state "+key)


def _optimizer_snapshot(optimizer):
    out={}
    for p,fields in optimizer.state.items():
        out[id(p)] = {
            k:(v.detach().cpu().clone() if torch.is_tensor(v) else v)
            for k,v in fields.items()
        }
    return out


def main():
    args=arguments()
    args.checkpoint=args.checkpoint.expanduser().resolve()
    args.output=args.output.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Missing original F2 Stage-I checkpoint: "+str(args.checkpoint))
    os.chdir(ROOT)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.device=torch.device(
        "cuda:0" if torch.cuda.is_available() and args.gpu >=0 else "cpu"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    # Disable all older diagnostics and F2 enhancement variants.
    os.environ.update({
        "PMME_TRANSFER_VARIANT":"gradient",
        "PMME_F2_METHOD":"original",
        "PMME_PAIR_VARIANT":"f2",
        "PMME_PAIR_STRENGTH":"0",
        "PMME_FEEDBACK_MODE":"g0",
        "PMME_F2_DIAGNOSTIC":"off",
        "PMME_F2_DIAG_READONLY":"0",
        "PMME_TRANSFER_OT_SCALE":str(args.ot_scale),
        "PMME_TRANSFER_TEMPERATURE":"8.0",
    })
    with open(args.config_filename,encoding="utf-8") as stream:
        config=yaml.safe_load(stream)
    if int(config["task"]["source"]["batch_size"]) != 16:
        raise ValueError("H3 needs matched original-F2 source batch=16")
    data_args,task_args,model_args = (
        config["data"],config["task"],config["model"]
    )
    city_list=get_data_list(args.data_list)
    if len(city_list)!=3:
        raise ValueError("Need exactly three source cities")
    data_config={"data_list":args.data_list,"base_dir":ROOT,"device":args.device}
    source_dataset=traffic_dataset(
        data_args,task_args["source"],city_list,"source",
        test_data=args.test_dataset
    )
    target_dataset=traffic_dataset(
        data_args,task_args["target_few_shot"],city_list,"target_maml",
        test_data=args.test_dataset
    )
    args.save_dir=args.output.parent
    args.STmodel="SOFTS"
    args.new=1
    model=STRep(
        0.5,args.daepoch,1,100,0,1,"SOFTS","OT",
        data_args,task_args,model_args,data_config,args
    )
    checkpoint=torch.load(
        str(args.checkpoint),map_location=args.device,weights_only=True
    )
    if not isinstance(checkpoint,dict) or not all(
        key in checkpoint for key in
        ("model_state_dict","optimizer_state_dict","epoch")
    ):
        raise ValueError(
            "H3 requires a complete Stage-I F2 checkpoint INCLUDING AdamW state"
        )
    checkpoint_epoch=int(checkpoint["epoch"])
    if checkpoint_epoch < args.daepoch:
        raise ValueError("Checkpoint was saved before OT became active")
    model.load_state_dict(checkpoint["model_state_dict"],strict=True)
    model.main_optim.load_state_dict(checkpoint["optimizer_state_dict"])
    del checkpoint
    parameter_snapshot={
        name:p.detach().cpu().clone()
        for name,p in model.model.named_parameters()
    }
    optimizer_snapshot=_optimizer_snapshot(model.main_optim)
    buffer_snapshot={
        name:b.detach().clone() for name,b in model.model.named_buffers()
    }
    from geomloss.samples_loss import SamplesLoss
    criterion=SamplesLoss(
        loss="sinkhorn",p=1,backend="tensorized",debias=False,blur=0.001
    )
    digest=hashlib.sha256()
    with args.checkpoint.open("rb") as fp:
        for chunk in iter(lambda:fp.read(1024*1024),b""):
            digest.update(chunk)
    metadata={
        "audit_version":"h3-v1",
        "checkpoint_path":str(args.checkpoint),
        "checkpoint_sha256":digest.hexdigest(),
        "checkpoint_epoch":checkpoint_epoch,
        "seed":args.seed,
        "target_city":args.test_dataset,
        "target_split":"three-day support; modulo-5 probe; not independent",
        "source_cities":city_list,
        "batch_size_source":16,
        "target_windows":int(target_dataset.x_list[args.test_dataset].shape[0]),
        "source_points":args.source_points,
        "target_points":args.target_points,
        "full_ot_geometry":args.source_points==0 and args.target_points==0,
        "ot_cost":"original concat(rep, sqrt(100)*normalized_future), "
                   "GeomLoss(p=1, blur=0.001, debias=False)",
        "f2_reference":"ORIGINAL F2 cosine formula, recomputed using "
                       "fit-only target windows (not all-support training alpha)",
        "ot_scale":args.ot_scale,
        "pcgrad":"one-sided full-parameter source_plus_OT against target FIT",
        "optimizer":"loaded original AdamW state, virtual step only",
        "training_updates":0,
    }
    metadata_path=args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata,sort_keys=True,indent=2),
        encoding="utf-8"
    )
    print("H3_META_JSON",json.dumps(metadata,sort_keys=True),flush=True)
    print("H3_AUDIT_BEGIN",args.probe_batches,"probe batches",flush=True)
    with args.output.open("w",encoding="utf-8") as stream:
      for batch_index in range(args.probe_batches):
        data,adj=source_target_batch(source_dataset,target_dataset,3)
        if len(data)!=4:
            raise ValueError("Expected 1 target and 3 source batches")
        model._transfer_records=[]
        model._capture_transfer=True
        output=[]
        truth=[]
        reps=[]
        futures=[]
        try:
            for sample in data:
                pred,y,rep,y_norm,_=model.model(sample,stage="train")
                output.append(pred)
                truth.append(y)
                reps.append(rep)
                futures.append(y_norm)
        finally:
            model._capture_transfer=False
        if len(model._transfer_records)!=4:
            raise RuntimeError("F2 projection2 hook did not capture 4 city forwards")

        target_windows=output[0].shape[0]
        fit,probe=masked_support_split(target_windows,output[0].device)
        target_fit_loss=model.loss_criterion(
            output[0][fit],truth[0][fit]
        )
        source_losses=[
            model.loss_criterion(output[j],truth[j])
            for j in (1,2,3)
        ]
        target_features,target_head_output=model._transfer_records[0]
        # No probe labels enter the target head-grad scoring.
        target_delta=torch.autograd.grad(
            target_fit_loss,target_head_output,retain_graph=True
        )[0].detach()
        target_split=(15,15,target_windows-30)
        target_rep_groups=torch.split(reps[0],target_split,dim=0)
        target_y_groups=torch.split(futures[0],target_split,dim=0)
        target_feature_groups=torch.split(target_features,target_split,dim=0)
        target_delta_groups=torch.split(target_delta,target_split,dim=0)
        target_fit_groups=torch.split(fit,target_split,dim=0)

        entries=[]
        city_metadata=[]
        params=list(model.model.parameters())
        for j in (1,2,3):
            city=str(data[j].data_name)
            features,head_output=model._transfer_records[j]
            src_delta=torch.autograd.grad(
                source_losses[j-1],head_output,retain_graph=True
            )[0].detach()
            group=j-1
            tmask=target_fit_groups[group]
            ft=target_feature_groups[group][tmask]
            dt=target_delta_groups[group][tmask]
            src_feat=features.reshape(-1,features.shape[-1])
            src_delta_flat=src_delta.reshape(-1,src_delta.shape[-1])
            target_feat=ft.reshape(-1,ft.shape[-1])
            target_delta_flat=dt.reshape(-1,dt.shape[-1])
            src_rep=reps[j].reshape(-1,reps[j].shape[-1])
            src_future=futures[j].reshape(-1,futures[j].shape[-1])
            tgt_rep=target_rep_groups[group][tmask].reshape(
                -1,target_rep_groups[group].shape[-1]
            )
            tgt_future=target_y_groups[group][tmask].reshape(
                -1,target_y_groups[group].shape[-1]
            )
            source_ids=sampled_flat_indices(
                len(src_rep),args.source_points,src_rep.device
            )
            target_ids=sampled_flat_indices(
                len(tgt_rep),args.target_points,tgt_rep.device
            )
            alpha,beta,weight_diagnostics=compute_weights(
                src_feat.index_select(0,source_ids).unsqueeze(0).detach(),
                src_delta_flat.index_select(0,source_ids).unsqueeze(0),
                target_feat.index_select(0,target_ids).unsqueeze(0).detach(),
                target_delta_flat.index_select(0,target_ids).unsqueeze(0),
                variant="gradient",temperature=8.0
            )
            x=torch.cat((
                src_rep.index_select(0,source_ids),
                src_future.detach().index_select(0,source_ids)*10.0
            ),dim=-1)
            y=torch.cat((
                tgt_rep.index_select(0,target_ids),
                tgt_future.detach().index_select(0,target_ids)*10.0
            ),dim=-1)
            uniform=torch.full_like(alpha,1.0/len(alpha))
            coeff=args.ot_scale
            loss_f2=(criterion(alpha,x,beta,y).sum()*coeff)
            gradients_f2=_grads(loss_f2,params)
            loss_uniform=(criterion(uniform,x,beta,y).sum()*coeff)
            gradients_uniform=_grads(loss_uniform,params)
            entries.append({
                "city":city,"f2":gradients_f2,"uniform":gradients_uniform
            })
            top_count=min(16,alpha.numel())
            top_values,top_positions=torch.topk(alpha.detach(),k=top_count)
            top_full_ids=source_ids.index_select(0,top_positions)
            city_metadata.append({
                "city":city,
                "target_group":int(group),
                "source_batch_windows":int(reps[j].shape[0]),
                "source_nodes":int(reps[j].shape[1]),
                "source_point_total":len(src_rep),
                "source_OT_points":len(source_ids),
                "target_fit_point_total":len(tgt_rep),
                "target_OT_points":len(target_ids),
                "target_fit_window_indices":tmask.nonzero().flatten().cpu().tolist(),
                "f2_source_ess_fraction":weight_diagnostics["ess_frac"],
                "f2_source_top_indices":top_full_ids.cpu().tolist(),
                "f2_source_top_masses":top_values.cpu().tolist(),
                "target_beta_type":"uniform-fit-only",
                "source_sample_selection":(
                    "all" if args.source_points==0 else
                    source_ids.detach().cpu().tolist()
                ),
                "target_fit_sample_selection":(
                    "all" if args.target_points==0 else
                    target_ids.detach().cpu().tolist()
                ),
            })
            del loss_f2,loss_uniform,x,y

        report=shadow_city_audit(
            model=model.model,
            optimizer=model.main_optim,
            source_task_loss=sum(source_losses),
            target_fit_task_loss=target_fit_loss,
            city_entries=entries,target_data=data[0],
            probe_mask=probe,epoch=args.daepoch+batch_index,
            sample_index=batch_index,beta=model.loss_criterion.beta
        )
        report["city_batches"]=city_metadata
        report["checkpoint_training_epoch"]=checkpoint_epoch
        report["full_ot_geometry"]=metadata["full_ot_geometry"]
        stream.write(json.dumps(report,allow_nan=False,sort_keys=True)+"\n")
        stream.flush()
        print("H3_RECORD_JSON",json.dumps({
            "probe_batch_index":batch_index,
            "best":report["best_by_probe_smooth_l1"],
            "gain_vs_f2":{
                name:round(v["smooth_l1"],9)
                for name,v in report["gain_vs_f2"].items()
            }
        },sort_keys=True),flush=True)
        # Shadow forwards can alter train-time buffer caches, reset them.
        with torch.no_grad():
            for name,b in model.model.named_buffers():
                b.copy_(buffer_snapshot[name])
        _verify_model_invariance(
            model.model,parameter_snapshot,model.main_optim,optimizer_snapshot
        )
        if any(p.grad is not None for p in model.model.parameters()):
            raise RuntimeError("H3 unexpectedly populated .grad")
        del entries,report
    print("H3_AUDIT_FINISHED",args.probe_batches,
          "probe batches","training_updates",0,
          "output",args.output,flush=True)


if __name__=="__main__":
    main()
