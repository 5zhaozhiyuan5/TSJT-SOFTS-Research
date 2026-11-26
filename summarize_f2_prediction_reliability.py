#!/usr/bin/env python3
"""Summarize PRC checkpoints selected by target-support validation.

Defaults to PRC-only runs. If a matching same-commit F2 is present it can be
compared, but historical F2/T0 seed1 is a separate-commit contextual reference.
"""
import argparse
import csv
from pathlib import Path
from summarize_f2_ot_decay import (
    parse_checkpoint_selected_metrics, summarize_selected
)


def load_config(path):
    result={}
    for l in path.read_text(encoding="utf-8").splitlines():
        if "=" in l:
            k,v=l.split("=",1)
            result[k]=v
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-root",required=True,type=Path)
    args=p.parse_args()
    results=[]
    for path in sorted(args.run_root.glob("*/seed*/run_config.txt")):
        directory=path.parent
        if not (directory/"COMPLETED").is_file():
            print("[INCOMPLETE] Not reporting",directory)
            continue
        cfg=load_config(path)
        selected,_=parse_checkpoint_selected_metrics(
            (directory/"train.out").read_text(encoding="utf-8",errors="replace")
        )
        for stage in ("stage1","stage2"):
            if stage=="stage2" and int(cfg["stage2_epochs"])==0:
                continue
            if selected[stage] is None:
                raise ValueError("Missing checkpoint selected by support validation: "+str(directory))
            results.append(summarize_selected(
                stage,selected[stage],cfg["mode"],cfg["seed"],cfg["git_commit"]
            ))
    if not results:
        print("No complete F2/PRC comparison yet")
        return
    if len({x["git_commit"] for x in results})!=1:
        raise ValueError("Different commits were used; do not merge these runs")
    rows_path=args.run_root/"selected_checkpoint_metrics.csv"
    with rows_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    by_key={(r["mode"],r["stage"],r["seed"]):r for r in results}
    deltas=[]
    for r in results:
        if r["mode"]!="prc":
            continue
        base=by_key.get(("f2",r["stage"],r["seed"]))
        if base is None:
            # PRC-only experiment intentionally does not rerun the F2 baseline.
            continue
        item={"seed":r["seed"],"stage":r["stage"],
              "prc_best_epoch":r["best_support_epoch"],
              "f2_best_epoch":base["best_support_epoch"]}
        for name in ("mean_36_horizon_mae","mean_36_horizon_rmse",
                     "mae_10min","rmse_10min","mae_60min","rmse_60min",
                     "mae_120min","rmse_120min","mae_180min","rmse_180min"):
            item["delta_"+name]=r[name]-base[name]
        deltas.append(item)
    if deltas:
        with (args.run_root/"prc_minus_f2.csv").open(
            "w",newline="",encoding="utf-8"
        ) as f:
            w=csv.DictWriter(f,fieldnames=list(deltas[0].keys()))
            w.writeheader()
            w.writerows(deltas)
    for r in sorted(results,key=lambda z:(z["seed"],z["stage"],z["mode"])):
        print("%s seed%s %s best=%d 36MAE=%.5f 36RMSE=%.5f "
              "MAE10/60/120/180=%.5f/%.5f/%.5f/%.5f" % (
                  r["mode"],r["seed"],r["stage"],
                  r["best_support_epoch"],r["mean_36_horizon_mae"],
                  r["mean_36_horizon_rmse"],
                  *(r["mae_%dmin"%k] for k in (10,60,120,180))))
    for d in deltas:
        print("PRC-F2 seed%s %s delta36MAE=%+.6f delta36RMSE=%+.6f" % (
            d["seed"],d["stage"],d["delta_mean_36_horizon_mae"],
            d["delta_mean_36_horizon_rmse"]))
    historical_f2_seed1_stage2_mae=2.372713333333333
    for r in results:
        if r["mode"]=="prc" and r["seed"]=="1" and r["stage"]=="stage2":
            print("HISTORICAL_REFERENCE_ONLY F2/T0 seed1 Stage-II 36MAE=%.6f "
                  "PRC-minus-history=%+.6f (different code commit; NOT a "
                  "same-run paired comparison)" % (
                  historical_f2_seed1_stage2_mae,
                  r["mean_36_horizon_mae"]-historical_f2_seed1_stage2_mae
            ))
    print("Saved",rows_path)


if __name__=="__main__":
    main()
