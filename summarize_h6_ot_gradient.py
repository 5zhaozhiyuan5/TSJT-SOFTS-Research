#!/usr/bin/env python3
"""Summarize preregistered H6 gradients, paired one-step shadow effects.

No final test accuracy or statistical significance is implied by this report.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean,median


def records_summary(rows):
    if not rows:
        raise ValueError("No H6 JSONL records")
    if any(row.get("audit_version")!="h6-v1" or
           row.get("training_updates")!=0 for row in rows):
        raise ValueError("Incorrect audit version or non-readonly record")
    batches=[int(r["batch_index"]) for r in rows]
    if batches != list(range(len(batches))):
        raise ValueError("Expected ordered, unique, consecutive batch IDs")
    names=set(rows[0]["gains"])
    if any(set(r["gains"])!=names for r in rows):
        raise ValueError("Different candidate names between batches")
    expected={"f2","identity","all_off","global_quarter","global_half"}
    if not expected.issubset(names):
        raise ValueError("Missing global H6 controls")
    if len([x for x in names if x.startswith("city_off:")])!=3:
        raise ValueError("Missing city-off controls")
    if len([x for x in names if x.startswith("city_half:")])!=3:
        raise ValueError("Missing city-half controls")
    replay_noise=max(abs(float(r["gains"]["identity"]["gain_mae"]))
                     for r in rows)
    if replay_noise>1e-4:
        raise ValueError("Identical F2 shadow replay exceeds 1e-4 MAE")
    summary=[]
    for name in sorted(names):
        v=[float(r["gains"][name]["gain_mae"]) for r in rows]
        l=[float(r["gains"][name]["gain_smooth_l1"]) for r in rows]
        row={
            "policy":name,"n":len(v),
            "mean_gain_mae":mean(v),
            "median_gain_mae":median(v),
            "positive_batches":sum(x>0 for x in v),
            "negative_batches":sum(x<0 for x in v),
            "mean_gain_smooth_l1":mean(l),
            "first_half_mean":mean(v[:min(len(v),4)]),
            "second_half_mean":mean(v[4:]) if len(v)>4 else "",
            "max_abs_share":max(abs(x) for x in v)/max(sum(abs(x) for x in v),1e-12),
            "mean_short_gain":mean(r["gains"][name]["gain_bands"][0] for r in rows),
            "mean_mid_gain":mean(r["gains"][name]["gain_bands"][1] for r in rows),
            "mean_long_gain":mean(r["gains"][name]["gain_bands"][2] for r in rows),
            "identity_noise_max_mae":replay_noise,
        }
        summary.append(row)
    traces=[]
    for r in rows:
        g=r["gradient_report"]
        step=r["step_diagnostics"]
        traces.append({
            "batch_index":r["batch_index"],
            "ot_norm":g["raw_ot_norm"],
            "source_grad_norm":g["source_gradient_norm"],
            "target_fit_grad_norm":g["target_fit_gradient_norm"],
            "effective_ot_after_pcgrad_norm":g["effective_ot_after_pcgrad_norm"],
            "pcgrad_effect_over_raw":g["effective_ot_after_pcgrad_norm"]/
                                     max(g["raw_ot_norm"],1e-15),
            "raw_ot_target_fit_cos":g["total_ot_to_target_fit"]["cos"],
            "effective_ot_target_fit_cos":
                g["effective_ot_after_pcgrad_vs_target"]["cos"],
            "raw_plus_source_target_fit_cos":
                g["raw_source_plus_ot_to_target_fit"]["cos"],
            "projected_plus_ot_target_fit_cos":
                g["projected_source_plus_ot_to_target_fit"]["cos"],
            "adamw_ot_vs_noot_step_cos_target_descent":
                step["f2_vs_no_ot"]["parameter_delta"]["cos"],
            "adamw_ot_vs_noot_step_norm":
                step["f2_vs_no_ot"]["parameter_delta"]["norm_a"],
            "global_half_shadow_gain_mae":r["gains"]["global_half"]["gain_mae"],
            "noot_shadow_gain_mae":r["gains"]["all_off"]["gain_mae"],
        })
    city_traces=[]
    for r in rows:
        for city,v in r["gradient_report"]["city"].items():
            off=r["step_diagnostics"][f"city_off:{city}"]
            city_traces.append({
                "batch_index":r["batch_index"],"city":city,
                "raw_ot_norm":v["raw_ot_norm"],
                "raw_ot_target_fit_cos":v["raw_ot_vs_target_fit"]["cos"],
                "pcgrad_conditional_norm":v["conditional_pcgrad_effect_norm"],
                "pcgrad_conditional_vs_target_cos":
                    v["conditional_after_pcgrad_vs_target_fit"]["cos"],
                "adamw_conditional_step_vs_target_descent_cos":
                    off["city_f2_minus_city_off_step"]["cos"],
                "adamw_conditional_step_norm":
                    off["city_f2_minus_city_off_step"]["norm_a"],
                "city_off_gain_mae":r["gains"][f"city_off:{city}"]["gain_mae"],
                "city_half_gain_mae":r["gains"][f"city_half:{city}"]["gain_mae"],
            })
    return summary,traces,city_traces,replay_noise


def write_csv(path,items):
    if not items:raise ValueError("Cannot write empty CSV")
    with Path(path).open("w",newline="",encoding="utf-8") as fp:
        w=csv.DictWriter(fp,fieldnames=list(items[0].keys()))
        w.writeheader();w.writerows(items)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--jsonl",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--gradients",type=Path,required=True)
    p.add_argument("--city-gradients",type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(s) for s in a.jsonl.read_text(encoding="utf-8").splitlines()
          if s.strip()]
    s,g,c,noise=records_summary(rows)
    for f,items in ((a.summary,s),(a.gradients,g),(a.city_gradients,c)):
        f.parent.mkdir(parents=True,exist_ok=True)
        write_csv(f,items)
    print("H6 existing ORIGINAL F2 checkpoint only; training updates 0.")
    print("All gains are one-step target SUPPORT-probe, NOT independent test metrics.")
    print("Identical F2 shadow replay noise max:",f"{noise:.10f}")
    for x in s:
        print(f"{x['policy']:26s} mean={x['mean_gain_mae']:+.9f} "
              f"median={x['median_gain_mae']:+.9f} "
              f"positive={x['positive_batches']}/{x['n']} "
              f"first={x['first_half_mean']:+.9f} "
              f"last={x['second_half_mean']} "
              f"max_abs_share={x['max_abs_share']:.3f}")
    print("Gradient/AdamW traces written to",a.gradients,"and",a.city_gradients)
    if len(rows)>=8:
        print("Exploratory stop conditions: at least 6/8 direction-consistent, "
              "positive first/last-four means, median > max(5e-5,10x replay "
              "noise), and max_abs_share <= .5; no support-only result proves generalization.")
    else:
        print("Smoke run: no conclusions should be drawn from a single probe.")


if __name__=="__main__":
    main()
