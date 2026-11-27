#!/usr/bin/env python3
"""Summarize TDSM Stage-I/unchanged Stage-II by target-support-selected ckpt."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from summarize_f2_ot_decay import (
    parse_checkpoint_selected_metrics,summarize_selected
)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    args=ap.parse_args()
    rows=[]
    for cfgpath in sorted(args.run_root.glob("tdsm/seed*/run_config.txt")):
        root=cfgpath.parent
        if not (root/"COMPLETED").is_file():
            print("[INCOMPLETE]",root)
            continue
        conf={}
        for line in cfgpath.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k,v=line.split("=",1);conf[k]=v
        trainout=(root/"train.out").read_text(encoding="utf-8",errors="replace")
        if not any("f2_tdsm_matching" in line for line in trainout.splitlines()):
            raise ValueError("TDSM never activated for "+str(root))
        selected,_=parse_checkpoint_selected_metrics(trainout)
        for stage in ("stage1","stage2"):
            if stage=="stage2" and int(conf["stage2_epochs"])==0:
                continue
            if selected[stage] is None:
                raise ValueError("No support-selected "+stage+" checkpoint for "+str(root))
            rows.append(summarize_selected(
                stage,selected[stage],"tdsm",conf["seed"],conf["commit"]
            ))
    if not rows:
        print("No completed F2+TDSM stage available")
        return
    if len({r["git_commit"] for r in rows})!=1:
        raise ValueError("Must not combine metrics across commits")
    output=args.run_root/"selected_checkpoint_metrics.csv"
    with output.open("w",newline="",encoding="utf-8") as out:
        wr=csv.DictWriter(out,fieldnames=list(rows[0]))
        wr.writeheader();wr.writerows(rows)
    for row in rows:
        print("TDSM", "seed",row["seed"],"stage",row["stage"],
              "best_support_epoch",row["best_support_epoch"],
              "mean_36_horizon_mae",row["mean_36_horizon_mae"],
              "mean_36_horizon_rmse",row["mean_36_horizon_rmse"],
              "MAE 10/60/120/180",*[row["mae_%dmin"%k] for k in (10,60,120,180)])
    print("Historical F2/T0 seed1 Stage-II 36-step MAE 2.372713 is a "
          "DIFFERENT-code historical reference only, NOT paired control.")
    print("PRC-only seed1 Stage-II 36-step MAE 2.394411 was worse than "
          "historical F2; TDSM is a distinct target-conditioned mechanism.")
    print("Selected-checkpoint results saved:",output)


if __name__=="__main__":
    main()
