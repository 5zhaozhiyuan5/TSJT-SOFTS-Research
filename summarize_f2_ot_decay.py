#!/usr/bin/env python3
"""Parse original PMME verbose logs by the LAST support-validation BEST epoch.

Report target test MAE/RMSE at 10/60/120/180 minutes, plus all-36 MAE.
Never pick a favorable test epoch: checkpoint selection happens in
train_DA_OT.py using the TARGET SUPPORT validation score only.
Stage II remains the original PMME_2ndStage implementation.

Usage: python summarize_f2_ot_decay.py --run-root experiments/f2-late-ot-decay
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


HORIZON=re.compile(
    r"^\s*Horizon\s+(\d+)\s*:\s*Unnormed MSE\s*:\s*"
    r"[-+0-9.eE]+,\s*RMSE\s*:\s*([-+0-9.eE]+),\s*"
    r"MAE\s*:\s*([-+0-9.eE]+)"
)
EPOCH_START=re.compile(r"^epoch:\s*(\d+)\s*$")
EPOCH_BEST=re.compile(r"^epoch:\s*(\d+)\s+best!\s*$")
HORIZON_MINUTES={10:1,60:11,120:23,180:35}


def parse_checkpoint_selected_metrics(log):
    """Return best-by-support selected test metric blocks for stages 1/2."""
    stage="stage1"
    current_epoch=None
    collecting=False
    by_horizon={}
    last_test=None
    best={"stage1":None,"stage2":None}
    counts={"stage1":0,"stage2":0}
    for line in log.splitlines():
        if line.strip()=="Source training Ending!!!":
            stage="stage2"
            current_epoch=None
            collecting=False
            last_test=None
            continue
        if "Enter test phase" in line:
            collecting=True
            by_horizon={}
            last_test=None
            continue
        epoch_start=EPOCH_START.match(line)
        if epoch_start:
            current_epoch=int(epoch_start.group(1))
        if collecting:
            found=HORIZON.match(line)
            if found:
                h=int(found.group(1))
                by_horizon[h]={
                    "rmse":float(found.group(2)),
                    "mae":float(found.group(3))
                }
                if len(by_horizon)==36 and set(by_horizon)==set(range(36)):
                    collecting=False
                    last_test=dict(by_horizon)
        selected=EPOCH_BEST.match(line)
        if selected:
            epoch=int(selected.group(1))
            if current_epoch != epoch:
                raise ValueError("Best-checkpoint marker not paired with epoch")
            if last_test is None or set(last_test)!=set(range(36)):
                raise ValueError(
                    "Missing complete 36-horizon test block for selected epoch"
                )
            best[stage]=(epoch,last_test)
            counts[stage]+=1
    return best,counts


def summarize_selected(stage,selected,mode,seed,commit):
    epoch, horizons=selected
    result={
        "mode":mode,"seed":str(seed),"git_commit":commit,"stage":stage,
        "best_support_epoch":epoch,
        "mean_36_horizon_mae":sum(v["mae"] for v in horizons.values())/36,
        "mean_36_horizon_rmse":sum(v["rmse"] for v in horizons.values())/36,
        "sum_mae_at_18_even_horizons":sum(
            horizons[i]["mae"] for i in range(1,36,2)
        )
    }
    for minute,index in HORIZON_MINUTES.items():
        result["mae_%dmin"%minute]=horizons[index]["mae"]
        result["rmse_%dmin"%minute]=horizons[index]["rmse"]
    return result


def parse_config(path):
    fields={}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k,v=line.split("=",1)
            fields[k]=v
    return fields


def report_differences(rows):
    grouped={(r["seed"],r["stage"],r["mode"]):r for r in rows}
    differences=[]
    for r in rows:
        if r["mode"]=="t0":
            continue
        original=grouped.get((r["seed"],r["stage"],"t0"))
        if original is None:
            continue
        pair={
            "mode":r["mode"],"seed":r["seed"],"stage":r["stage"],
            "compared_with":"t0",
            "delta_mean_36_horizon_mae":(
                r["mean_36_horizon_mae"]-original["mean_36_horizon_mae"]
            )
        }
        for minute in HORIZON_MINUTES:
            pair["delta_mae_%dmin"%minute] = (
                r["mae_%dmin"%minute]-original["mae_%dmin"%minute]
            )
            pair["delta_rmse_%dmin"%minute] = (
                r["rmse_%dmin"%minute]-original["rmse_%dmin"%minute]
            )
        differences.append(pair)
    return differences


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-root",type=Path,required=True)
    args=p.parse_args()
    if not args.run_root.is_dir():
        p.error("Missing root "+str(args.run_root))
    rows=[]
    for config_path in sorted(args.run_root.glob("t*/seed*/run_config.txt")):
        work=config_path.parent
        if not (work/"COMPLETED").exists():
            print("[INCOMPLETE] Skipping",work)
            continue
        cfg=parse_config(config_path)
        best,counts=parse_checkpoint_selected_metrics(
            (work/"train.out").read_text(encoding="utf-8",errors="replace")
        )
        if best["stage1"] is None:
            raise ValueError("Stage-I support-selected best missing: "+str(work))
        if int(cfg["stage2_epochs"])>0 and best["stage2"] is None:
            raise ValueError("Stage-II support-selected best missing: "+str(work))
        for stage in ("stage1","stage2"):
            if best[stage] is not None:
                rows.append(summarize_selected(
                    stage,best[stage],cfg["mode"],cfg["seed"],
                    cfg["git_commit"]
                ))
    if not rows:
        print("No completed T0/T1/T2 runs to summarize")
        return
    # Fail rather than combine different git commits in one comparison.
    if len({r["git_commit"] for r in rows})!=1:
        raise ValueError("Different code commits are mixed in results")
    summary=args.run_root/"selected_checkpoint_metrics.csv"
    with summary.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    differences=report_differences(rows)
    if differences:
        with (args.run_root/"matched_deltas_vs_t0.csv").open(
            "w",newline="",encoding="utf-8"
        ) as file:
            writer=csv.DictWriter(file,fieldnames=list(differences[0]))
            writer.writeheader()
            writer.writerows(differences)
    print("Target test metrics from checkpoints SELECTED BY SUPPORT validation.")
    print("Do not retune T1/T2 using target test results.")
    for row in sorted(rows,key=lambda x:(x["seed"],x["stage"],x["mode"])):
        print("%s/%s/%s best=%d  mean36MAE=%.5f "
              "MAE(10/60/120/180)=%.5f/%.5f/%.5f/%.5f" % (
                  row["mode"],row["seed"],row["stage"],
                  row["best_support_epoch"],row["mean_36_horizon_mae"],
                  *(row["mae_%dmin"%m] for m in (10,60,120,180))
              ))
    for delta in sorted(
        differences,key=lambda x:(x["seed"],x["stage"],x["mode"])
    ):
        print("Delta %s vs T0 seed%s %s: 36MAE=%+.5f "
              "MAE(10/60/120/180)=%+.5f/%+.5f/%+.5f/%+.5f" % (
                  delta["mode"],delta["seed"],delta["stage"],
                  delta["delta_mean_36_horizon_mae"],
                  *(delta["delta_mae_%dmin"%m] for m in (10,60,120,180))
              ))
    print("Summaries saved:",summary)


if __name__=="__main__":
    main()
