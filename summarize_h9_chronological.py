#!/usr/bin/env python3
"""Summarize H9-A/ H9-B CHRONOLOGICAL DEVELOPMENT, never official test.

Values come from non-overlapping *time intervals* relative to original support,
but evaluation windows within the dev interval overlap; infer no significance
or official final-test performance from this screening.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

POLICIES=("f2","global_half","projected_half","norm_matched_raw","all_off")
MILESTONES=(0,1,5,10,20)


def summarize(result):
    if result.get("audit_version")!="h9-v1":
        raise ValueError("Not an H9 audit")
    if result.get("original_f2_updates")!=0 or result.get("stage2_runs")!=0:
        raise ValueError("H9 original F2 or Stage-II was modified")
    if result.get("development_labels_in_update") is not False or (
        result.get("official_test_after_development_touched") is not False
    ):
        raise ValueError("H9 development or reserved-future leakage")
    bounds=result["development_time_bounds"]
    if (bounds["development_start_inclusive"] !=
        bounds["support_end_exclusive"] or
        bounds["forecast_start_min"] <
        bounds["support_end_exclusive"] or
        bounds["forecast_end_max_exclusive"] >
        bounds["development_end_exclusive"] or
        bounds["untouched_future_start_inclusive"] !=
        bounds["development_end_exclusive"]):
        raise ValueError("H9 chronological validation bounds violated")
    steps=result["evaluated_milestones"]
    if not steps or steps[0]!=0 or steps[-1]!=result[
        "adaptation_update_count_per_policy"
    ] or any(x not in MILESTONES for x in steps):
        raise ValueError("Unexpected H9 milestone grid")
    scores=result["scores"]
    if set(scores)!=set(POLICIES):
        raise ValueError("Missing H9 policies")
    rows=[];day_rows=[]
    for step in steps:
        ref=scores["f2"][str(step)]
        winner=scores["projected_half"][str(step)]
        half=scores["global_half"][str(step)]
        control=scores["norm_matched_raw"][str(step)]
        reference_mae={
            "f2":ref["mae"],"global_half":half["mae"],
            "projected_half":winner["mae"],"norm_matched_raw":control["mae"],
            "all_off":scores["all_off"][str(step)]["mae"],
        }
        for policy in POLICIES:
            x=scores[policy][str(step)]
            if x["window_count"]!=bounds["window_count"]:
                raise ValueError("Development window count mismatch")
            values={
                "step":step,
                "policy":policy,
                "dev_mae":x["mae"],
                "dev_rmse":x["rmse"],
                "gain_vs_f2":ref["mae"]-x["mae"],
                "gain_vs_half":half["mae"]-x["mae"],
                "gain_vs_norm_matched_raw":control["mae"]-x["mae"],
                "gain_projected_half_vs_this":x["mae"]-winner["mae"],
                "short_mae":x["horizon_band_mae"]["short"],
                "mid_mae":x["horizon_band_mae"]["mid"],
                "long_mae":x["horizon_band_mae"]["long"],
                "mae_at_10_min":x["mae_at_minutes"]["10"],
                "mae_at_60_min":x["mae_at_minutes"]["60"],
                "mae_at_120_min":x["mae_at_minutes"]["120"],
                "mae_at_180_min":x["mae_at_minutes"]["180"],
                "dev_windows":x["window_count"],
                "support_cutoff_raw_step":bounds["support_end_exclusive"],
                "development_end_raw_step":bounds["development_end_exclusive"],
                "reserved_future_start_raw_step":
                    bounds["untouched_future_start_inclusive"],
            }
            rows.append(values)
            for day,score in sorted(
                x["forecast_days"].items(),key=lambda y:int(y[0])
            ):
                day_rows.append({
                    "step":step,"policy":policy,"forecast_dev_day":int(day),
                    "mae":score["mae"],"rmse":score["rmse"],
                    "value_count":score["value_count"],
                })
    if len(set(scores[p]["0"]["mae"] for p in POLICIES))>1:
        if max(scores[p]["0"]["mae"] for p in POLICIES)-min(
            scores[p]["0"]["mae"] for p in POLICIES
        )>1e-5:
            raise ValueError("All policies must have identical zero-update MAE")
    return rows,day_rows


def write_csv(path,rows):
    if not rows:
        raise ValueError("Empty CSV")
    with Path(path).open("w",encoding="utf-8",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--json",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--days",type=Path,required=True)
    p.add_argument("--decision",type=Path,required=True)
    a=p.parse_args()
    result=json.loads(a.json.read_text(encoding="utf-8"))
    rows,days=summarize(result)
    write_csv(a.summary,rows)
    write_csv(a.days,days)
    last=max(result["evaluated_milestones"])
    at_last={r["policy"]:r for r in rows if r["step"]==last}
    differences={
        "versus_f2":at_last["projected_half"]["gain_vs_f2"],
        "versus_global_half":at_last["projected_half"]["gain_vs_half"],
        "versus_norm_matched_raw":
            at_last["projected_half"]["gain_vs_norm_matched_raw"],
    }
    all_steps={
        str(step):{
            "gain_projected_vs_half":
                next(r for r in rows if r["step"]==step and r["policy"]=="projected_half")["gain_vs_half"],
            "gain_projected_vs_same_norm":
                next(r for r in rows if r["step"]==step and r["policy"]=="projected_half")["gain_vs_norm_matched_raw"],
        }
        for step in result["evaluated_milestones"] if step>0
    }
    decision={
        "audit_version":"h9-v1",
        "max_adaptation_steps":last,
        "development_not_paper_test":True,
        "whole_dev_mae_contrasts_at_last_step":differences,
        "stepwise_development_contrasts":all_steps,
        "original_f2_updates":0,
        "stage2_runs":0,
        "no_development_labels_in_adaptation":True,
        "future_after_dev_reserved":True,
        "normalization_warning":"Original F2 full-series scaler; this is not a strictly train-only-normalized experiment",
        "policy_selection_warning":"After comparing H9 development results, this early post-support time is DEVELOPMENT, not a future independent benchmark test",
        "recommendation_rule":"Do NOT start formal Stage-II if projected_half fails to improve vs global_half and same-norm raw on chronologically disjoint development at steps 1 and 5; confirm later time blocks before finalizing",
    }
    a.decision.write_text(
        json.dumps(decision,sort_keys=True,indent=2),encoding="utf-8"
    )
    print("H9 chronological DEVELOPMENT. This is NOT official PMME test.")
    print("No F2 retraining, no Stage-II; these are clone updates.")
    for step in result["evaluated_milestones"]:
        print("step",step)
        for r in (x for x in rows if x["step"]==step):
            print(
                "  "+r["policy"],
                "devMAE="+f"{r['dev_mae']:.7f}",
                "devRMSE="+f"{r['dev_rmse']:.7f}",
                "vsF2="+f"{r['gain_vs_f2']:+.7f}",
                "vsHalf="+f"{r['gain_vs_half']:+.7f}",
                "vsSameNorm="+f"{r['gain_vs_norm_matched_raw']:+.7f}",
            )
    print("H9_DECISION "+json.dumps(decision,sort_keys=True))


if __name__=="__main__":
    main()
