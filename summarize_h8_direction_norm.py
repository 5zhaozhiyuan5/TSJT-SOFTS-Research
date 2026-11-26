#!/usr/bin/env python3
"""H8 results: directional benefit versus BOTH magnitude-matched controls.

All gains come from SAME already-seen 3-day target support under one virtual
PCGrad + optimizer-state AdamW update. They are NOT held-out test scores.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean,median


POLICY_SET={
    "f2","global_half","projected_half","norm_matched_raw",
    "direction_norm_matched_half","adamw_step_norm_matched_raw",
    "all_off","identity"
}


def _average(xs):
    return float(mean(xs))


def _max_abs_share(xs):
    return max(abs(x) for x in xs)/max(sum(abs(x) for x in xs),1e-12)


def summarize(rows):
    if not rows:
        raise ValueError("No H8 results")
    for i,r in enumerate(rows):
        if (r.get("audit_version")!="h8-v1" or r.get("batch_index")!=i or
            r.get("training_updates")!=0 or r.get("stage2_runs")!=0 or
            set(r.get("candidate_scores",{}))!=POLICY_SET or
            set(r.get("gains_vs_half",{}))!=POLICY_SET or
            r.get("n_shadow_candidates")!=len(POLICY_SET)):
            raise ValueError("Invalid H8 batch "+str(i))
    identity=max(
        abs(r["gains_vs_f2"]["identity"]["mae"]) for r in rows
    )
    if identity>1e-4:
        raise ValueError("H8 F2 identity replay exceeds 1e-4")
    fields={
        "vs_f2":"f2",
        "vs_half":"global_half",
        "vs_projected_half":"projected_half",
        "vs_same_norm":"norm_matched_raw",
        "vs_adamw_match":"adamw_step_norm_matched_raw",
    }
    summaries=[]
    for name in sorted(POLICY_SET):
        report={"policy":name,"n":len(rows),"identity_noise_max":identity}
        for prefix,reference in fields.items():
            ds=[
                r["candidate_scores"][reference]["mae"]-
                r["candidate_scores"][name]["mae"] for r in rows
            ]
            report.update({
                prefix+"_mean":_average(ds),
                prefix+"_median":median(ds),
                prefix+"_positive":sum(x>0 for x in ds),
                prefix+"_first4_mean":_average(ds[:min(4,len(ds))]),
                prefix+"_last4_mean":_average(ds[4:]) if len(ds)>4 else "",
                prefix+"_largest_abs_share":_max_abs_share(ds),
            })
        bands=[r["candidate_scores"]["norm_matched_raw"]["band_smooth_l1"][i]-
               r["candidate_scores"][name]["band_smooth_l1"][i]
               for r in rows for i in range(3)]
        for i,group in enumerate(("short","mid","long")):
            report["vs_same_norm_"+group+"_mean"]=_average(bands[i::3])
        summaries.append(report)

    traces=[]
    cities=[]
    for r in rows:
        g=r["gradient_stats"]
        c=r["adamw_calibration"]
        x=r["adamw_traces"]
        traces.append({
            "batch":r["batch_index"],
            "raw_ot_norm":g["raw_gradient_norm"],
            "projected_ot_norm":g["projected_gradient_norm"],
            "removed_conflict_norm":g["removed_conflict_norm"],
            "raw_target_cos":g["raw_target_gradient_cos"],
            "projected_target_cos":g["projected_target_gradient_cos"],
            "same_grad_norm_abs_error":
                g["projected_vs_norm_matched_raw_gradient_norm_error"],
            "same_half_grad_norm_abs_error":
                g["global_half_vs_equal_norm_projected_error"],
            "norm_matched_raw_factor":g["norm_matched_raw_factor"],
            "adamw_step_norm_matched_raw_factor":
                g["adamw_step_match_raw_factor"],
            "projected_half_ot_adamw_effect_norm":
                x["projected_half"]["ot_induced_step_norm"],
            "same_norm_raw_ot_adamw_effect_norm":
                x["norm_matched_raw"]["ot_induced_step_norm"],
            "step_matched_raw_ot_adamw_effect_norm":
                x["adamw_step_norm_matched_raw"]["ot_induced_step_norm"],
            "adamw_calibration_relative_error":c["relative_error"],
            "adamw_calibration_within_one_percent":int(c["within_one_percent"]),
            "adamw_calibration_monotonic":int(c["monotonic"]),
            "adamw_calibration_limited":int(c["calibration_limited"]),
            "adamw_calibration_evaluations":c["calibration_evaluations"],
            "projected_half_gain_vs_same_grad_norm":
                r["gains_vs_same_norm_raw"]["projected_half"]["mae"],
            "projected_half_gain_vs_matched_adamw_step":
                r["gains_vs_step_norm_raw"]["projected_half"]["mae"],
            "projected_half_gain_vs_global_half":
                r["gains_vs_half"]["projected_half"]["mae"],
            "same_norm_raw_gain_vs_global_half":
                r["gains_vs_half"]["norm_matched_raw"]["mae"],
        })
        for city,raw_norm in sorted(g["source_city_grad_norms"].items()):
            cities.append({
                "batch":r["batch_index"],"city":city,
                "original_ot_norm":raw_norm,
            })

    return summaries,traces,cities,identity


def reference_check(rows,prior_file,ref_policies,limit=1e-5):
    """Separate reproducibility from policy-value conclusions."""
    old=[json.loads(line) for line in Path(prior_file).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(old)!=len(rows):
        return {"matched":False,"reason":"different batch count"}
    checks={}
    for new,previous in ref_policies.items():
        ds=[]
        for row,old_row in zip(rows,old):
            if not row.get("full_ot_geometry") or not old_row.get("full_ot_geometry"):
                return {"matched":False,"reason":"both runs must use full OT"}
            if row["batch_index"]!=old_row.get("batch_index"):
                return {"matched":False,"reason":"batch index mismatch"}
            ds.append(abs(row["candidate_scores"][new]["mae"]-
                          old_row["candidate_scores"][previous]["mae"]))
        checks[new+"__"+previous]=max(ds)
    return {"matched":all(x<=limit for x in checks.values()),
            "tolerance":limit,"max_mae_differences":checks}


def save_csv(filepath,records):
    if not records:
        raise ValueError("Cannot write empty CSV")
    p=Path(filepath)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def decide(rows,identity):
    """Predeclared screening gates, never a significance test."""
    def pass_rule(candidate,reference):
        ds=[r["candidate_scores"][reference]["mae"]-
            r["candidate_scores"][candidate]["mae"] for r in rows]
        if len(ds)<8:
            return {"pass":False,"reason":"fewer than 8 correlated probes"}
        threshold=max(5e-5,10*identity)
        return {
            "pass":(
                sum(v>0 for v in ds)>=6 and
                mean(ds[:4])>0 and mean(ds[4:])>0 and
                median(ds)>threshold and _max_abs_share(ds)<=.5
            ),
            "mean":_average(ds),"median":median(ds),
            "positive":sum(v>0 for v in ds),
            "first4":mean(ds[:4]),"last4":mean(ds[4:]),
            "outlier_share":_max_abs_share(ds),
            "median_threshold":threshold,
        }
    return {
        "primary_direction_vs_same_gradient_norm":
            pass_rule("projected_half","norm_matched_raw"),
        "secondary_direction_vs_same_adamw_step_norm":
            pass_rule("projected_half","adamw_step_norm_matched_raw"),
        "prior_h7_winner_vs_global_half":
            pass_rule("projected_half","global_half"),
        "adamw_control_calibration_usable":all(
            r["adamw_calibration"]["within_one_percent"] for r in rows
        ),
        "readme":"H8 probes are all within original F2 checkpoint's already-seen support; no generalization or statistical significance.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--jsonl",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--traces",type=Path,required=True)
    p.add_argument("--city-traces",type=Path,required=True)
    p.add_argument("--decision",type=Path,required=True)
    p.add_argument("--h7-reference",type=Path)
    p.add_argument("--h7-replay-check",type=Path)
    args=p.parse_args()
    rows=[json.loads(line) for line in args.jsonl.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    summaries,traces,cities,identity=summarize(rows)
    save_csv(args.summary,summaries)
    save_csv(args.traces,traces)
    save_csv(args.city_traces,cities)
    d=decide(rows,identity)
    if args.h7_reference:
        check=reference_check(
            rows,args.h7_reference,
            {"f2":"f2","global_half":"global_half",
             "projected_half":"city_projection_half"}
        )
        d["h7_reference_replay"]=check
        if args.h7_replay_check:
            args.h7_replay_check.write_text(
                json.dumps(check,sort_keys=True,indent=2),encoding="utf-8"
            )
    args.decision.write_text(
        json.dumps(d,sort_keys=True,indent=2),encoding="utf-8"
    )
    print("H8: checkpoint-only, ZERO F2/Stage-II training; SUPPORT-only shadow.")
    print("H8 primary: projected_half better than SAME-GRADIENT-NORM raw OT?")
    print("H8 secondary: projected_half better than SAME-AdamW-STEP-NORM raw OT?")
    print("Identity F2 replay numerical noise max",identity)
    for s in summaries:
        print(s["policy"],
              "gain_vs_F2="+f"{s['vs_f2_mean']:+.9f}",
              "gain_vs_half="+f"{s['vs_half_mean']:+.9f}",
              "gain_vs_norm_matched="+f"{s['vs_same_norm_mean']:+.9f}",
              "gain_vs_adamw_matched="+f"{s['vs_adamw_match_mean']:+.9f}")
    print("All optimizer-calibrations within 1%:",
          d["adamw_control_calibration_usable"])
    print("DECISIONS "+json.dumps(d,sort_keys=True))
    print("No result constitutes an independent target validation or final model improvement.")


if __name__=="__main__":
    main()
