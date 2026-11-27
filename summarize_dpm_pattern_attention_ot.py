#!/usr/bin/env python3
"""DPM actual Stage-I / original Stage-II metrics, support-selected checkpoints.

Compares against historical F2/T0 seed1 only as NON-PAIRED context; never
asserts an improvement from experiments run at different code commits.
"""
import argparse
import csv
from pathlib import Path

from summarize_f2_ot_decay import (
    parse_checkpoint_selected_metrics, summarize_selected
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-root",required=True,type=Path)
    args=parser.parse_args()
    results=[]
    for config_path in sorted(args.run_root.glob("*/seed*/run_config.txt")):
        directory=config_path.parent
        if not (directory/"COMPLETED").is_file():
            print("[INCOMPLETE]",directory)
            continue
        cfg={}
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k,v=line.split("=",1)
                cfg[k]=v
        trainlog=(directory/"train.out").read_text(encoding="utf-8",errors="replace")
        if "dpm_bank_fitted" not in trainlog or "dpm_pair_cost" not in trainlog:
            raise ValueError("DPM bank/attention missing from active OT run "+str(directory))
        selected,_=parse_checkpoint_selected_metrics(trainlog)
        for stage in ("stage1","stage2"):
            if stage=="stage2" and int(cfg["stage2_epochs"])==0:
                continue
            if selected[stage] is None:
                raise ValueError("Missing support-selected checkpoint: "+str(directory)+"/"+stage)
            results.append(summarize_selected(
                stage,selected[stage],cfg["mode"],cfg["seed"],cfg["commit"]
            ))
    if not results:
        print("No completed F2+DPM runs")
        return
    if len({r["git_commit"] for r in results})!=1:
        raise ValueError("Never combine different code commits in one results CSV")
    destination=args.run_root/"selected_checkpoint_metrics.csv"
    with destination.open("w",newline="",encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for r in sorted(results,key=lambda v:(v["seed"],v["stage"],v["mode"])):
        print("DPM",r["mode"],"seed",r["seed"],"stage",r["stage"],
              "best_support_epoch",r["best_support_epoch"],
              "mean36MAE",r["mean_36_horizon_mae"],
              "mean36RMSE",r["mean_36_horizon_rmse"],
              "MAE10/60/120/180",
              *(r["mae_%dmin"%h] for h in (10,60,120,180)))
    print("HISTORICAL-REFERENCE-ONLY: original F2/T0 seed1 Stage-II "
          "mean36MAE=2.372713333333333, from a DIFFERENT commit.")
    print("HISTORICAL-REFERENCE-ONLY: F2+TDSM seed1 Stage-II "
          "mean36MAE=2.395861111111111, from a DIFFERENT commit.")
    print("DPM is a NEW code path. Cross-commit comparisons are not paired.")
    print("Saved:",destination)


if __name__=="__main__":
    main()
