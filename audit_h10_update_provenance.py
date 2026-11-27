#!/usr/bin/env python3
"""H10 official Stage-I fidelity gate and gradient-source update ablations.

A: official meta_train_revise(projection=1, F2, ALL 46 target support) vs
   manual independent gradient bank + identical original PCGrad + AdamW.
   FULL GeomLoss OT mandatory for numerical parity. Hard gate.
B: gated T0/T0-F/T1..T7 on 4-day post-support DEVELOPMENT, 0/1/5
   steps. Each policy independent checkpoint+optimizer clone.

NO original checkpoint writes, F2 retraining, Stage-II, or final test scoring.
"""
from __future__ import annotations

import argparse
import copy
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
for folder in (ROOT,ROOT/"model",ROOT/"model"/"Meta_Models"):
    sys.path.insert(0,str(folder))

from audit_h4_error_profiles import load_model
from h9_checkpoint_adaptation import clone_f2,_source_batch,reproducible_step_seed
from h9_temporal_holdout import create_development_batches,score_development
from h10_original_f2_bank import original_f2_full_support_bank
from h10_update_provenance import (
    POLICIES,UPDATE_POLICIES,h10_direction,apply_directions,
    clone_official_wrapper,snapshot_tensors,compare_snapshots
)

PINNED_SHA="f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf"
DEFAULT_CHECKPOINT=Path(
    "/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/"
    "prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth"
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("a","b","both"),default="a")
    p.add_argument("--checkpoint",type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument("--output",type=Path,default=Path("experiments/h10/h10.json"))
    p.add_argument("--parity-report",type=Path)
    p.add_argument("--config-filename",default="configs/config.yaml")
    p.add_argument("--data-list",default="chengdu_shenzhen_metr")
    p.add_argument("--test-dataset",default="pems-bay")
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--daepoch",type=int,default=1500)
    p.add_argument("--ot-scale",type=float,default=.01)
    p.add_argument("--source-points",type=int,default=0)
    p.add_argument("--target-points",type=int,default=0)
    p.add_argument("--max-updates",type=int,choices=(1,5),default=1)
    p.add_argument("--validation-days",type=int,default=4)
    p.add_argument("--validation-batch-size",type=int,default=8)
    p.add_argument("--validation-max-windows",type=int,default=0)
    p.add_argument("--parity-atol",type=float,default=2e-5)
    a=p.parse_args()
    if (a.seed!=1 or a.test_dataset!="pems-bay" or
        a.data_list!="chengdu_shenzhen_metr" or
        a.daepoch!=1500 or a.ot_scale!=.01 or
        a.source_points<0 or a.target_points<0 or
        0<a.source_points<32 or 0<a.target_points<16 or
        a.validation_days!=4 or a.validation_batch_size<1 or
        a.validation_max_windows<0 or a.validation_max_windows>256 or
        a.parity_atol>2e-5 or a.parity_atol<=0):
        p.error("H10 fixed original-F2 configuration / too large tolerance")
    if a.phase=="a" and (a.source_points!=0 or a.target_points!=0):
        p.error("H10-A requires FULL GeomLoss OT for original-parity gate")
    if a.phase=="both" and (a.source_points!=0 or a.target_points!=0):
        p.error("H10 A+B combined requires FULL OT for parity")
    if a.phase=="b" and a.parity_report is None:
        p.error("H10-B requires verified --parity-report from H10-A")
    return a


def hash_checkpoint(path):
    h=hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda:fp.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def seed_step(step):
    random.seed(reproducible_step_seed(step))
    np.random.seed(reproducible_step_seed(step))
    torch.manual_seed(reproducible_step_seed(step))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(reproducible_step_seed(step))


def official_step_fidelity(wrapper,source,target,sinkhorn,*,atol,sha):
    """Execute two FULL-OT cloned updates with exact same batch/RNG.

    The paper's original train log uses projection_args=1.0. Therefore:
       original = OneSidePCGrad(grad_source + grad_OT, grad_target_full46).
    Do NOT compare to projection=0 direct backprop, a different baseline.
    """
    if float(wrapper.projection_args)!=1.0:
        raise RuntimeError("Pinned 3000-epoch log uses projection=1.0")
    batch,seed,batch_hash=_source_batch(source,target,step=1)
    # Matching adjacency list used only by official wrapper meta_train.
    matrices=[target.A_list[target.data_list[0]]]
    for obj in batch[1:]:
        matrices.append(source.A_list[str(obj.data_name)])
    original_before=snapshot_tensors(wrapper.model,wrapper.main_optim)
    official=clone_official_wrapper(wrapper)
    torch.manual_seed(352001)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(352001)
    official.meta_train_revise(batch,matrices,2899)
    official_state=snapshot_tensors(official.model,official.main_optim)
    del official
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    manual_model,manual_opt=clone_f2(wrapper.model,wrapper.main_optim)
    bank=original_f2_full_support_bank(
        manual_model,batch,loss_criterion=wrapper.loss_criterion,
        sinkhorn=sinkhorn,source_points=0,target_points=0,
        ot_scale=.01,step=1
    )
    if bank["target_fit_count"]!=46 or bank["unused_support_probe_count"]!=0:
        raise RuntimeError("H10-A manual target split not original-F2 full46")
    direction,mask,trace=h10_direction(bank,"t5_official_f2")
    displacement=apply_directions(manual_model,manual_opt,direction,mask)
    manual_state=snapshot_tensors(manual_model,manual_opt)
    comparison=compare_snapshots(official_state,manual_state,atol=atol)
    unchanged=compare_snapshots(
        original_before,snapshot_tensors(wrapper.model,wrapper.main_optim),
        atol=0.
    )
    if not unchanged["passed"]:
        raise RuntimeError("H10-A accidentally modified original F2/AdamW")
    report={
        "audit_version":"h10-a-v1",
        "original_checkpoint_sha256":sha,
        "original_checkpoint_epoch":2899,
        "original_projection_args":1.0,
        "official_module":"STRep.meta_train_revise",
        "official_train_epoch_replay":2899,
        "seed":1,"source_batch_seed":seed,
        "shared_source_target_batch_hash":batch_hash,
        "original_checkpoint_updates":0,
        "official_clone_updates":1,
        "manual_clone_updates":1,
        "target_support_windows_in_ot":46,
        "source_points":0,"target_points":0,
        "full_ot_geometry":True,
        "manual_model_update_norm":displacement,
        "manual_update_trace":trace,
        "official_vs_manual":comparison,
        "original_unchanged":unchanged["passed"],
        "passed":comparison["passed"] and unchanged["passed"],
    }
    del bank,manual_model,manual_opt,batch
    return report


def run_ablation(wrapper,source,target,dev_batches,sinkhorn,*,updates,
                 source_points,target_points):
    """Run 9 policies, every policy has its own from-checkpoint AdamW clone."""
    original_before=snapshot_tensors(wrapper.model,wrapper.main_optim)
    score={};traces={};fixed_batch_hashes={}
    for policy in POLICIES:
        model,optim=clone_f2(wrapper.model,wrapper.main_optim)
        score[policy]={0:score_development(model,dev_batches)}
        logs=[]
        for step in range(1,updates+1):
            batch,step_seed,batch_hash=_source_batch(source,target,step)
            old=fixed_batch_hashes.setdefault(step,batch_hash)
            if old!=batch_hash:
                raise RuntimeError("Different source/target batches across H10 policies")
            if policy=="t0_no_update":
                trace={"step":step,"kind":"no_train_forward_no_step",
                       "parameter_displacement_norm":0.}
            elif policy=="t0f_train_forward_only":
                before=snapshot_tensors(model,optim)
                torch.manual_seed(352000+step)
                with torch.no_grad():
                    for item in batch:
                        model(item,stage="train")
                after=snapshot_tensors(model,optim)
                comparison=compare_snapshots(before,after,atol=0.)
                trace={"step":step,"kind":"train_forward_only",
                       "forward_state_diff":comparison,
                       "parameter_displacement_norm":0.}
            elif policy=="t1_zero_grad_adamw":
                # Zero gradients, but AdamW may move due to saved moments/WD.
                blank=[torch.zeros_like(p) for p in model.parameters()]
                displacement=apply_directions(
                    model,optim,blank,[True]*len(blank)
                )
                trace={"step":step,"kind":"adamw_zero_grad_with_history",
                       "parameter_displacement_norm":displacement}
            else:
                bank=original_f2_full_support_bank(
                    model,batch,loss_criterion=wrapper.loss_criterion,
                    sinkhorn=sinkhorn,source_points=source_points,
                    target_points=target_points,ot_scale=.01,step=step
                )
                direction,present,diagnostics=h10_direction(bank,policy)
                displacement=apply_directions(model,optim,direction,present)
                trace={
                    "step":step,"kind":"true_gradient_update",
                    "gradient_diagnostics":diagnostics,
                    "parameter_displacement_norm":displacement,
                    "ot_city_stats":bank["ot_metadata"],
                    "full_support_windows":bank["target_fit_count"],
                    "unused_target_support_probes":
                        bank["unused_support_probe_count"],
                    "original_F2_projection_flag":1.0,
                }
                del bank,direction,present
            trace["step_seed"]=step_seed
            trace["support_source_batch_hash"]=batch_hash
            trace["ordered_sources"]=[str(x.data_name) for x in batch[1:]]
            logs.append(trace)
            del batch
            if step in (1,5):
                score[policy][step]=score_development(model,dev_batches)
                print("H10_POLICY_STEP",json.dumps({
                    "policy":policy,"step":step,
                    "dev_mae":score[policy][step]["mae"],
                    "dev_rmse":score[policy][step]["rmse"]
                },sort_keys=True),flush=True)
        traces[policy]=logs
        del model,optim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    unchanged=compare_snapshots(
        original_before,snapshot_tensors(wrapper.model,wrapper.main_optim),
        atol=0.
    )
    if not unchanged["passed"]:
        raise RuntimeError("H10-B modified the original checkpoint optimizer")
    return {
        "audit_version":"h10-b-v1",
        "policies":list(POLICIES),
        "max_updates":updates,
        "evaluated_milestones":[0,1] if updates==1 else [0,1,5],
        "scores":score,
        "step_traces":traces,
        "matched_batch_fingerprints":{
            str(k):v for k,v in fixed_batch_hashes.items()
        },
        "original_model_and_optimizer_unchanged":True,
        "original_F2_training_updates":0,
        "official_StageII_runs":0,
        "full_ot_geometry":source_points==target_points==0,
        "source_ot_points_cap":source_points,
        "target_ot_points_cap":target_points,
        "dev_labels_in_update":False,
        "future_after_dev_scored":False,
    }


def main():
    a=parse_args()
    a.checkpoint=a.checkpoint.expanduser().resolve()
    a.output=a.output.expanduser().resolve()
    if not a.checkpoint.is_file():
        raise FileNotFoundError("No original F2 checkpoint, refusing retrain")
    sha=hash_checkpoint(a.checkpoint)
    if sha!=PINNED_SHA:
        raise ValueError("Unrecognized F2 checkpoint sha256; no training")
    if a.parity_report is not None:
        a.parity_report=a.parity_report.expanduser().resolve()
    os.chdir(ROOT)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.device=torch.device(
        "cuda:0" if a.gpu>=0 and torch.cuda.is_available() else "cpu"
    )
    random.seed(a.seed);np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    # Needs args.train_epochs for faithful official meta_train logging.
    a.train_epochs=3000
    a.projection=1.0
    with open(a.config_filename,encoding="utf-8") as fp:
        config=yaml.safe_load(fp)
    if (int(config["task"]["source"]["batch_size"])!=16 or
        int(config["task"]["target_few_shot"]["batch_size"])!=46 or
        int(config["task"]["source"]["his_num"])!=288 or
        int(config["task"]["source"]["pred_num"])!=36):
        raise ValueError("Not original F2 batch16/target46/H288/F36")
    wrapper,source,target,cities,epoch=load_model(a)
    if (epoch!=2899 or float(wrapper.projection_args)!=1.0 or
        wrapper.transfer_variant!="gradient" or wrapper.f2_method!="original"):
        raise RuntimeError("H10 must load real Stage-I F2 with projection=1.0")
    from geomloss.samples_loss import SamplesLoss
    sinkhorn=SamplesLoss(
        loss="sinkhorn",p=1,backend="tensorized",debias=False,blur=.001
    )
    baseline_meta={
        "audit_version":"h10-v1",
        "checkpoint_sha256":sha,"checkpoint_epoch":epoch,
        "original_f2_projection_args":1.0,
        "checkpoint_original_train_log":"projection_args 1.0 for all 3000 epochs",
        "target_city":a.test_dataset,"source_cities":cities,
        "target_support_windows":46,
        "source_batch":16,
        "training_input":"original first three days only; same source batches",
        "normalization_warning":"Original F2 full-series target scaler inherited",
        "historical_test_exposure":"Official original training logged post-support test metrics; H10 dev is not a virgin test",
        "original_checkpoint_written":False,
        "stage2_runs":0,
    }
    result={"metadata":baseline_meta}
    if a.phase in ("a","both"):
        parity=official_step_fidelity(
            wrapper,source,target,sinkhorn,atol=a.parity_atol,sha=sha
        )
        result["parity"]=parity
        print("H10_OFFICIAL_PARITY",json.dumps({
            "passed":parity["passed"],
            **parity["official_vs_manual"],
        },sort_keys=True),flush=True)
        # Persist evidence EVEN IF parity fails, but do not run B.
        a.output.write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False),
            encoding="utf-8"
        )
        if not parity["passed"]:
            raise RuntimeError(
                "H10 official-vs-manual mismatch. Stop! Fix gradient/AdamW "
                "parity BEFORE interpreting any attribution result."
            )
    else:
        if not a.parity_report or not a.parity_report.is_file():
            raise FileNotFoundError("H10-B requires prior FULL-OT parity JSON")
        trusted=json.loads(a.parity_report.read_text(encoding="utf-8"))
        parity=trusted.get("parity",trusted)
        if (not parity.get("passed") or
            parity.get("original_checkpoint_sha256")!=sha or
            parity.get("original_projection_args")!=1.0 or
            parity.get("target_support_windows_in_ot")!=46 or
            not parity.get("full_ot_geometry") or
            not parity.get("official_vs_manual",{}).get("passed")):
            raise RuntimeError("Supplied parity report not a passed H10-A check")
        result["parity"]=parity
    if a.phase in ("b","both"):
        dev,bounds=create_development_batches(
            config["data"][a.test_dataset]["dataset_path"],
            normalization_mean=target.means_list[a.test_dataset],
            normalization_std=target.stds_list[a.test_dataset],
            support_days=3,validation_days=a.validation_days,
            history=288,horizon=36,stride=12,steps_per_day=288,
            batch_size=a.validation_batch_size,
            max_windows=a.validation_max_windows,
        )
        if (bounds["development_start_inclusive"]!=864 or
            bounds["development_end_exclusive"]!=2016):
            raise RuntimeError("Unexpected H10 development interval")
        result["development_bounds"]=bounds
        result["ablation"]=run_ablation(
            wrapper,source,target,dev,sinkhorn,
            updates=a.max_updates,
            source_points=a.source_points,target_points=a.target_points
        )
    a.output.write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False),
        encoding="utf-8"
    )
    print("H10_COMPLETED phase",a.phase,
          "parity_passed",True,
          "original_f2_updates",0,
          "stage2_runs",0,
          "output",a.output,flush=True)


if __name__=="__main__":
    main()
