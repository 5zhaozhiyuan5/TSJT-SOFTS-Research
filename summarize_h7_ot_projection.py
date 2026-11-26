#!/usr/bin/env python3
"""H7: paired read-only OT projection summary (primary contrast vs H6 half).

This is an on-SUPPORT virtual-update diagnostic. The target checkpoint has
previously seen the probe period; no final forecasting/test improvement claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

POLICIES=(
    "f2","global_half","city_off:shenzhen","global_projection",
    "city_projection","city_projection_half","identity",
)


def avg(seq):
    return statistics.mean(seq)


def median(seq):
    return statistics.median(seq)


def analyze(rows):
    if not rows:
        raise ValueError("No H7 records")
    for i,r in enumerate(rows):
        if (r.get("audit_version")!="h7-v1" or r.get("batch_index")!=i or
            r.get("training_updates")!=0 or r.get("stage2_runs")!=0 or
            r.get("n_shadow_candidates")!=7 or
            tuple(r.get("candidate_scores",{}))!=POLICIES):
            raise ValueError("Invalid, incomplete or reordered H7 record "+str(i))
        if set(r["gains_vs_half"])!=set(POLICIES):
            raise ValueError("Missing H7 gains")
    identity=max(abs(r["gains_vs_f2"]["identity"]["gain_mae"]) for r in rows)
    if identity>1e-4:
        raise ValueError("F2 identity control mismatch >1e-4")
    policy_rows=[]
    for p in POLICIES:
        d0=[float(r["gains_vs_f2"][p]["gain_mae"]) for r in rows]
        dh=[float(r["gains_vs_half"][p]["gain_mae"]) for r in rows]
        bandh=[r["gains_vs_half"][p]["gain_bands"] for r in rows]
        policy_rows.append({
            "policy":p,"n":len(rows),
            "mean_gain_vs_f2":avg(d0),
            "median_gain_vs_f2":median(d0),
            "mean_gain_vs_half":avg(dh),
            "median_gain_vs_half":median(dh),
            "positive_vs_half":sum(v>0 for v in dh),
            "negative_vs_half":sum(v<0 for v in dh),
            "first4_mean_vs_half":avg(dh[:min(4,len(dh))]),
            "last4_mean_vs_half":avg(dh[4:]) if len(rows)>4 else "",
            "largest_batch_share_vs_half":
                max(abs(v) for v in dh)/max(sum(abs(v) for v in dh),1e-12),
            "short_gain_vs_half":avg(x[0] for x in bandh),
            "mid_gain_vs_half":avg(x[1] for x in bandh),
            "long_gain_vs_half":avg(x[2] for x in bandh),
            "identity_noise_max":identity,
        })
    grad_rows=[];city_rows=[]
    for r in rows:
        d=r["projection_diagnostics"]
        t=r["gradient_traces"]
        opt=r["optimizer_traces"]
        grad_rows.append({
            "batch_index":r["batch_index"],
            "all_three_conflicting":int(d["all_cities_conflicting"]),
            "global_raw_target_cos":d["global"]["raw_vs_target"]["cos"],
            "global_projected_target_cos":d["global"]["projected_vs_target"]["cos"],
            "global_raw_ot_norm":d["global"]["raw_ot_norm"],
            "global_proj_ot_norm":d["global"]["projected_ot_norm"],
            "global_projection_removed_norm":d["global"]["removed_norm"],
            "global_city_projection_difference_norm":
                d["global_vs_city_projection_difference_norm"],
            "global_city_projection_relative_difference":
                d["global_vs_city_projection_relative_difference"],
            "f2_effective_ot_after_existing_pcgrad_norm":
                t["f2"]["effective_after_existing_pcgrad_norm"],
            "projected_ot_after_existing_pcgrad_norm":
                t["global_projection"]["effective_after_existing_pcgrad_norm"],
            "projected_optimizer_delta_from_half_norm":
                opt["global_projection"]["delta_from_half_norm"],
            "projected_optimizer_delta_from_half_target_cos":
                opt["global_projection"]["delta_from_half_vs_target_descent"]["cos"],
            "projected_vs_half_gain_mae":
                r["gains_vs_half"]["global_projection"]["gain_mae"],
            "city_projected_vs_half_gain_mae":
                r["gains_vs_half"]["city_projection"]["gain_mae"],
            "city_projected_half_vs_half_gain_mae":
                r["gains_vs_half"]["city_projection_half"]["gain_mae"],
        })
        for city,info in sorted(d["city"].items()):
            city_rows.append({
                "batch_index":r["batch_index"],"city":city,
                "is_conflicting":int(info["is_conflicting"]),
                "raw_dot":info["raw_dot"],
                "raw_target_cos":info["raw_vs_target"]["cos"],
                "projected_target_cos":info["projected_vs_target"]["cos"],
                "raw_norm":info["raw_ot_norm"],
                "projected_norm":info["projected_ot_norm"],
                "removed_norm":info["removed_norm"],
            })
    return policy_rows,grad_rows,city_rows,identity


def validate_h6_replay(h7_rows,h6_path):
    """Recompute whether F2 and global-half still match H6's exact shadow setup."""
    h6=[json.loads(x) for x in Path(h6_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(h6)!=len(h7_rows):
        return {"matched":False,"reason":"different number of probes"}
    differences={}
    for p in ("f2","global_half"):
        diffs=[]
        for r7,r6 in zip(h7_rows,h6):
            if r7["batch_index"]!=r6.get("batch_index"):
                return {"matched":False,"reason":"batch IDs mismatch"}
            if not (r7.get("full_ot_geometry") and r6.get("full_ot_geometry")):
                return {"matched":False,"reason":"not both full-OT"}
            diffs.append(abs(
                r7["candidate_scores"][p]["mae"]-
                r6["candidate_scores"][p]["mae"]
            ))
        differences[p]=max(diffs)
    return {
        "matched":all(x<=1e-5 for x in differences.values()),
        "max_absolute_mae_difference_by_reference":differences,
        "tolerance":1e-5,
    }


def write_csv(path,values):
    if not values:
        raise ValueError("Cannot write empty CSV")
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(values[0]))
        writer.writeheader();writer.writerows(values)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--jsonl",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--gradients",type=Path,required=True)
    p.add_argument("--city-gradients",type=Path,required=True)
    p.add_argument("--h6-jsonl",type=Path)
    p.add_argument("--h6-check",type=Path)
    args=p.parse_args()
    rows=[json.loads(x) for x in args.jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
    stats,grad,city,noise=analyze(rows)
    for path,values in (
        (args.summary,stats),(args.gradients,grad),(args.city_gradients,city)
    ):
        write_csv(path,values)
    print("H7: existing original-F2 checkpoint, real OT, no training, no stage II.")
    print("Primary comparison is H6 global_half; positive means lower shadow MAE.")
    print("NOT independent target validation, because checkpoint saw all support probes.")
    print(f"Identity F2 numerical noise max: {noise:.10f}")
    for x in stats:
        print(
            f"{x['policy']:23s} "
            f"vsF2={x['mean_gain_vs_f2']:+.9f} "
            f"vsHalf={x['mean_gain_vs_half']:+.9f} "
            f"medVsHalf={x['median_gain_vs_half']:+.9f} "
            f"pos={x['positive_vs_half']}/{x['n']} "
            f"outlier={x['largest_batch_share_vs_half']:.3f}"
        )
    print("All-three-city conflicts:",sum(x["all_three_conflicting"] for x in grad),
          "of",len(grad))
    print("Max relative global-vs-city projection difference:",
          max(x["global_city_projection_relative_difference"] for x in grad))
    if args.h6_jsonl:
        check=validate_h6_replay(rows,args.h6_jsonl)
        print("H6 exact checkpoint/shadow comparison:",json.dumps(check,sort_keys=True))
        if args.h6_check:
            args.h6_check.write_text(json.dumps(check,indent=2),encoding="utf-8")
    if len(rows)>=8:
        print("Screen vs global_half: positive in >=6/8, positive first/last 4,")
        print("median > max(5e-5,10x identity), no single batch >50% abs gain;")
        print("cross-city projection must improve beyond global projection,")
        print("otherwise it has no incremental evidence. All screen-only.")


if __name__=="__main__":
    main()
