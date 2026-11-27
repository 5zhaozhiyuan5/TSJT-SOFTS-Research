#!/usr/bin/env python3
"""Summarize H10 author-equivalence and source/target/OT ablation attribution.

H10 is chronological DEVELOPMENT, not official final paper testing. The
original F2 full-series scaler and previous post-support log exposure remain.
"""
import argparse
import csv
import json
from pathlib import Path

POLICIES=(
    "t0_no_update","t0f_train_forward_only","t1_zero_grad_adamw",
    "t2_source_only","t3_target_only","t4_source_pcgrad",
    "t5_official_f2","t6_no_pcgrad","t7_project_ot_half"
)


def summarize(obj):
    parity=obj.get("parity")
    if not parity or not parity.get("passed") or not parity.get(
        "official_vs_manual",{}
    ).get("passed"):
        raise ValueError("Refusing to summarize H10-B without passed H10-A")
    if obj.get("metadata",{}).get("checkpoint_sha256")!=parity.get(
        "original_checkpoint_sha256"
    ):
        raise ValueError("H10 checkpoint/parity SHA mismatch")
    if obj.get("ablation") is None:
        return [],[],{
            "audit_version":"h10-a-v1",
            "official_f2_parity":"passed",
            "target_support_windows":46,
            "original_checkpoint_updates":0,
            "stage2_runs":0,
        }
    abl=obj["ablation"]
    if (
        abl.get("original_model_and_optimizer_unchanged") is not True or
        abl.get("original_F2_training_updates")!=0 or
        abl.get("official_StageII_runs")!=0 or
        abl.get("dev_labels_in_update") is not False or
        set(abl.get("policies",()))!=set(POLICIES)
    ):
        raise ValueError("H10 ablation state/leakage/completeness check failed")
    bounds=obj["development_bounds"]
    if (bounds["development_start_inclusive"]!=864 or
        bounds["development_end_exclusive"]!=2016 or
        bounds["forecast_end_max_exclusive"]>2016):
        raise ValueError("Development time boundaries changed")
    scores=abl["scores"]
    steps=abl["evaluated_milestones"]
    rows=[];days=[]
    for step in steps:
        all_score={
            key:scores[key][str(step)] for key in POLICIES
        }
        for policy in POLICIES:
            x=all_score[policy]
            if x["window_count"]!=bounds["window_count"]:
                raise ValueError("Window count differs among policies")
            row={
                "step":step,"policy":policy,
                "dev_mae":x["mae"],"dev_rmse":x["rmse"],
                "vs_no_update_mae":all_score["t0_no_update"]["mae"]-x["mae"],
                "vs_official_f2_mae":all_score["t5_official_f2"]["mae"]-x["mae"],
                "vs_source_only_mae":all_score["t2_source_only"]["mae"]-x["mae"],
                "vs_source_pcgrad_mae":all_score["t4_source_pcgrad"]["mae"]-x["mae"],
                "vs_full_pcgrad_mae":all_score["t6_no_pcgrad"]["mae"]-x["mae"],
                "short_mae":x["horizon_band_mae"]["short"],
                "mid_mae":x["horizon_band_mae"]["mid"],
                "long_mae":x["horizon_band_mae"]["long"],
                "mae_10min":x["mae_at_minutes"]["10"],
                "mae_60min":x["mae_at_minutes"]["60"],
                "mae_120min":x["mae_at_minutes"]["120"],
                "mae_180min":x["mae_at_minutes"]["180"],
                "dev_windows":x["window_count"],
                "full_ot":abl["full_ot_geometry"],
            }
            rows.append(row)
            for day,metrics in sorted(
                x["forecast_days"].items(),key=lambda q:int(q[0])
            ):
                days.append({
                    "step":step,"policy":policy,"dev_forecast_day":int(day),
                    "day_mae":metrics["mae"],"day_rmse":metrics["rmse"],
                    "values":metrics["value_count"],
                })
    first={x["policy"]:x for x in rows if x["step"]==1}
    last={x["policy"]:x for x in rows if x["step"]==steps[-1]}
    diffs={
        "official_F2_vs_no_update":
            first["t0_no_update"]["dev_mae"]-
            first["t5_official_f2"]["dev_mae"],
        "source_only_vs_no_update":
            first["t0_no_update"]["dev_mae"]-
            first["t2_source_only"]["dev_mae"],
        "target_only_vs_no_update":
            first["t0_no_update"]["dev_mae"]-
            first["t3_target_only"]["dev_mae"],
        "zero_adamw_vs_no_update":
            first["t0_no_update"]["dev_mae"]-
            first["t1_zero_grad_adamw"]["dev_mae"],
        "forward_only_vs_no_update":
            first["t0_no_update"]["dev_mae"]-
            first["t0f_train_forward_only"]["dev_mae"],
        "ot_value_official_f2_vs_source_pcgrad":
            first["t4_source_pcgrad"]["dev_mae"]-
            first["t5_official_f2"]["dev_mae"],
        "pcgrad_value_official_f2_vs_no_pcgrad":
            first["t6_no_pcgrad"]["dev_mae"]-
            first["t5_official_f2"]["dev_mae"],
        "projected_ot_vs_official_f2":
            first["t5_official_f2"]["dev_mae"]-
            first["t7_project_ot_half"]["dev_mae"],
    }
    decision={
        "audit_version":"h10-b-v1",
        "parity_verified":True,
        "original_checkpoint_updates":0,
        "stage2_runs":0,
        "full_ot":abl["full_ot_geometry"],
        "update_count":steps[-1],
        "zero_update_mae":scores["t0_no_update"]["0"]["mae"],
        "step1_component_differences_positive_is_better":diffs,
        "last_step_best_development_policy":min(
            last,key=lambda key:last[key]["dev_mae"]
        ),
        "all_final_updates_vs_initial_checkpoint":{
            name:scores["t0_no_update"]["0"]["mae"]-
                 last[name]["dev_mae"] for name in POLICIES
        },
        "notes":"Diagnostic only. T5 is the true projection=1 F2, T6 disables PCGrad. Dev is not never-inspected independent paper test; full-series scaler inherited. Capped OT results do not establish full OT multi-step trends.",
    }
    return rows,days,decision


def csv_out(path,rows):
    if not rows:
        return
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--json",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--daywise",type=Path,required=True)
    p.add_argument("--decision",type=Path,required=True)
    a=p.parse_args()
    obj=json.loads(a.json.read_text(encoding="utf-8"))
    rows,days,decision=summarize(obj)
    csv_out(a.summary,rows);csv_out(a.daywise,days)
    a.decision.write_text(
        json.dumps(decision,indent=2,sort_keys=True),encoding="utf-8"
    )
    print("H10 ORIGINAL PARITY: PASS",
          json.dumps(obj["parity"]["official_vs_manual"],sort_keys=True))
    if not rows:
        print("H10-A only; run H10-B after parity gate.")
        return
    print("H10 chronological DEV; original F2 and Stage-II unchanged")
    for step in sorted({r["step"] for r in rows}):
        print("Step",step)
        for r in rows:
            if r["step"]==step:
                print(f"  {r['policy']:26} MAE={r['dev_mae']:.7f} "
                      f"RMSE={r['dev_rmse']:.7f} "
                      f"gain_vs_T0={r['vs_no_update_mae']:+.7f} "
                      f"gain_vs_officialF2={r['vs_official_f2_mae']:+.7f}")
    print("H10_DECISION",json.dumps(decision,sort_keys=True))


if __name__=="__main__":
    main()
