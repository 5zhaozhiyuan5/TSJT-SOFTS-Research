#!/usr/bin/env python3
"""Summarize H3 city-wise F2 shadow update experiment (paired probe batches).

A positive gain_vs_f2 is a LOWER shadow target probe loss after one virtual
PCGrad+AdamW step. This is within-training-support only, NOT target test MAE
and NOT an estimate with independent replicate sampling.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def mean(xs):
    return sum(xs)/len(xs) if xs else float("nan")


def summarize(rows):
    if not rows:
        raise ValueError("No H3 diagnostic records")
    policies=set(rows[0]["candidate_scores"])
    if len(policies)!=8 or "f2" not in policies or "all_off" not in policies:
        raise ValueError("Expected seven main H3 policies + all_off reference")
    if any(set(x["candidate_scores"])!=policies for x in rows):
        raise ValueError("Inconsistent H3 candidate sets")
    if any(x.get("training_updates")!=0 or
           x.get("audit_version")!="h3-v1" for x in rows):
        raise ValueError("Invalid non-mutating H3 audit records")
    results=[]
    for policy in sorted(policies):
        gains=[float(x["gain_vs_f2"][policy]["smooth_l1"]) for x in rows]
        mae_gains=[float(x["gain_vs_f2"][policy]["mae"]) for x in rows]
        bands=[
            mean([float(x["gain_vs_f2"][policy]["bands"][i]) for x in rows])
            for i in range(3)
        ]
        result={
            "policy":policy,
            "n_probe_batches":len(rows),
            "mean_smooth_l1_gain_vs_f2":mean(gains),
            "positive_smooth_l1_fraction":mean([u>0 for u in gains]),
            "mean_mae_gain_vs_f2":mean(mae_gains),
            "positive_mae_fraction":mean([u>0 for u in mae_gains]),
            "mean_short_band_gain":bands[0],
            "mean_mid_band_gain":bands[1],
            "mean_long_band_gain":bands[2],
            "all_bands_positive_fraction":mean([
                all(float(v)>0 for v in x["gain_vs_f2"][policy]["bands"])
                for x in rows
            ]),
            "best_policy_frequency":mean([
                x["best_by_probe_smooth_l1"]==policy for x in rows
            ]),
        }
        for minute in ("10","60","120","180"):
            scores=[
                float(x["gain_vs_f2"][policy][
                    "horizons_10_60_120_180_mae"
                ][minute]) for x in rows
            ]
            result["mae_gain_"+minute+"min"]=mean(scores)
        results.append(result)
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--jsonl",required=True,type=Path)
    p.add_argument("--csv",type=Path)
    args=p.parse_args()
    if not args.jsonl.is_file():
        p.error("Missing H3 input JSONL")
    rows=[json.loads(line) for line in args.jsonl.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    report=summarize(rows)
    if args.csv:
        args.csv.parent.mkdir(parents=True,exist_ok=True)
        with args.csv.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(report[0]))
            writer.writeheader()
            writer.writerows(report)
    print("H3 one-step PCGrad+AdamW SHADOW; within three-day support only.")
    print("Positive gain = less probe loss than all-three-city F2; no forecast "
          "test, no independent generalization claim.")
    print("Records:",len(rows),"; full geometry:",
          all(r.get("full_ot_geometry",False) for r in rows))
    if not all(r.get("full_ot_geometry",False) for r in rows):
        print("WARNING: SUBSAMPLED OT - smoke/proxy only, not full-batch evidence")
    for r in sorted(report,key=lambda x:x["mean_smooth_l1_gain_vs_f2"],
                    reverse=True):
        print(
            "%-28s mean_gain=%+.8g positive=%4.1f%% "
            "mean_mae_gain=%+.8g "
            "bands=%+.6g/%+.6g/%+.6g "
            "MAE(10,60,120,180)=%+.6g/%+.6g/%+.6g/%+.6g" % (
                r["policy"],
                r["mean_smooth_l1_gain_vs_f2"],
                100*r["positive_smooth_l1_fraction"],
                r["mean_mae_gain_vs_f2"],
                r["mean_short_band_gain"],r["mean_mid_band_gain"],
                r["mean_long_band_gain"],
                r["mae_gain_10min"],r["mae_gain_60min"],
                r["mae_gain_120min"],r["mae_gain_180min"]
            )
        )


if __name__=="__main__":
    main()
