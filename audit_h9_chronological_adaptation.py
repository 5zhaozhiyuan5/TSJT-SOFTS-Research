#!/usr/bin/env python3
"""H9 chronological dev validation and controlled cloned-F2 adaptation.

H9-A: --max-updates 1, evaluates the 5 policies after one REAL clone update.
H9-B: --max-updates 5/10/20, recomputes F2 OT every step and evaluates
      at 0,1,5,10,20 as applicable.
Never loads the official target 'test' traffic_dataset; never touches time
beyond predeclared early post-support DEV slice except the ORIGINAL full-series
normalization inherited from PMME. Not a paper test-set metric.
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
for folder in (ROOT,ROOT/"model",ROOT/"model"/"Meta_Models"):
    sys.path.insert(0,str(folder))

from audit_h4_error_profiles import load_model
from h9_temporal_holdout import create_development_batches
from h9_checkpoint_adaptation import POLICIES,run_h9_adaptation


EXISTING_F2_SHA256=(
    "f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf"
)
DEFAULT_CKPT=Path(
    "/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/"
    "prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth"
)


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,default=DEFAULT_CKPT)
    p.add_argument("--output",type=Path,
                   default=Path("experiments/h9/h9_results.json"))
    p.add_argument("--config-filename",default="configs/config.yaml")
    p.add_argument("--data-list",default="chengdu_shenzhen_metr")
    p.add_argument("--test-dataset",default="pems-bay")
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--daepoch",type=int,default=1500)
    p.add_argument("--ot-scale",type=float,default=.01)
    p.add_argument("--source-points",type=int,default=256)
    p.add_argument("--target-points",type=int,default=128)
    p.add_argument("--max-updates",type=int,choices=(1,5,10,20),default=1)
    p.add_argument("--validation-days",type=int,default=4)
    p.add_argument("--validation-batch-size",type=int,default=8)
    p.add_argument("--validation-max-windows",type=int,default=0)
    p.add_argument("--expected-sha256",default=EXISTING_F2_SHA256)
    args=p.parse_args()
    if (
        args.seed!=1 or args.daepoch!=1500 or args.ot_scale!=.01 or
        args.data_list!="chengdu_shenzhen_metr" or
        args.test_dataset!="pems-bay" or
        args.source_points<0 or args.target_points<0 or
        (0<args.source_points<32) or (0<args.target_points<16) or
        args.validation_days<2 or args.validation_days>7 or
        args.validation_batch_size<1 or args.validation_batch_size>32 or
        args.validation_max_windows<0 or args.validation_max_windows>256 or
        args.expected_sha256!=EXISTING_F2_SHA256
    ):
        p.error("H9 locks original F2 seed/geometry and dev validation days <=7")
    return args


def digest_file(path):
    h=hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda:fp.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    args=arguments()
    args.checkpoint=args.checkpoint.expanduser().resolve()
    args.output=args.output.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            "Original F2 Stage-I checkpoint absent. Refusing to retrain: "+
            str(args.checkpoint)
        )
    sha=digest_file(args.checkpoint)
    if sha!=EXISTING_F2_SHA256:
        raise ValueError("Wrong original-F2 checkpoint SHA. Refusing to continue")
    os.chdir(ROOT)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.device=torch.device(
        "cuda:0" if args.gpu>=0 and torch.cuda.is_available() else "cpu"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    with open(args.config_filename,encoding="utf-8") as fp:
        config=yaml.safe_load(fp)
    if (config["task"]["source"]["his_num"]!=288 or
        config["task"]["source"]["pred_num"]!=36 or
        config["task"]["target_few_shot"]["his_num"]!=288 or
        config["task"]["target_few_shot"]["pred_num"]!=36 or
        config["task"]["target_few_shot"]["batch_size"]!=46):
        raise ValueError("H9 requires original SOFTS H=288 F=36, target batch46")
    wrapper,source,target,cities,epoch=load_model(args)
    if epoch!=2899:
        raise ValueError("Expected known original F2 epoch2899")
    means=target.means_list[args.test_dataset]
    stds=target.stds_list[args.test_dataset]
    dev_batches,bounds=create_development_batches(
        config["data"][args.test_dataset]["dataset_path"],
        normalization_mean=means,normalization_std=stds,
        support_days=3,validation_days=args.validation_days,
        steps_per_day=288,history=288,horizon=36,stride=12,
        batch_size=args.validation_batch_size,
        max_windows=args.validation_max_windows,
    )
    # Development input history begins AFTER the entire 3-day support. The
    # entire later time axis is untouched by H9's validation batches.
    if bounds["development_start_inclusive"]<864 or (
        bounds["forecast_end_max_exclusive"]>
        bounds["development_end_exclusive"]
    ):
        raise RuntimeError("Temporal dev split crossed fixed data boundary")
    from geomloss.samples_loss import SamplesLoss
    sinkhorn=SamplesLoss(
        loss="sinkhorn",p=1,backend="tensorized",debias=False,blur=.001
    )
    milestones=tuple(x for x in (0,1,5,10,20) if x<=args.max_updates)
    metadata={
        "audit_version":"h9-v1",
        "experiment_mode":"H9-A" if args.max_updates==1 else "H9-B",
        "checkpoint":str(args.checkpoint),"checkpoint_sha256":sha,
        "checkpoint_epoch":int(epoch),
        "source_cities":cities[:3],
        "target_city":args.test_dataset,
        "seed":args.seed,
        "max_updates_per_cloned_policy":args.max_updates,
        "milestones":milestones,
        "policies":list(POLICIES),
        "full_ot_geometry":args.source_points==args.target_points==0,
        "ot_scale":args.ot_scale,
        "source_points":args.source_points,
        "target_points":args.target_points,
        "source_target_fit":"Only original target 3-day SUPPORT; 36 FIT windows "
                            "plus 10 unchanged support PROBE (unused by adaptation)",
        "development_chronology":bounds,
        "normalization_warning":"Original F2 full-series scaler reused; not train-only",
        "benchmark_warning":"Post-support development interval was part of original "
                            "PMME test range and cannot be reused as final untouched test",
        "final_future_touched":False,
        "original_f2_training_updates":0,
        "stage2_runs":0,
        "cloned_optimizer_updates_per_policy":args.max_updates,
        "original_ot_cost":"GeomLoss(p=1,blur=.001,debias=False) on "
                           "concat(SOFTS_rep,10*RevIN_true_future)",
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata,sort_keys=True,indent=2),encoding="utf-8"
    )
    print("H9_METADATA "+json.dumps(metadata,sort_keys=True),flush=True)
    result=run_h9_adaptation(
        original_model=wrapper.model,
        original_optimizer=wrapper.main_optim,
        source_dataset=source,target_dataset=target,
        validation_batches=dev_batches,sinkhorn=sinkhorn,
        loss_criterion=wrapper.loss_criterion,
        source_points=args.source_points,target_points=args.target_points,
        max_updates=args.max_updates,eval_steps=milestones,seed=args.seed,
    )
    result["metadata"]=metadata
    result["development_time_bounds"]=bounds
    args.output.write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False),
        encoding="utf-8"
    )
    print(
        "H9_COMPLETED",
        "max_updates",args.max_updates,
        "original_f2_updates",0,
        "stage2_runs",0,
        "dev_windows",bounds["window_count"],
        "output",args.output,flush=True
    )


if __name__=="__main__":
    main()
