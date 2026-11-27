#!/usr/bin/env python3
"""Analyze DPM V2-A/V2-B Stage-I real-cost and Q/R feedback logs.

Reads train.out logs from source GPU server or a copied experiment bundle.
Does not rerun F2, fit patterns, alter checkpoints, or claim causal gain.
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_events(log_path):
    records = []
    for lineno, line in enumerate(log_path.open(encoding="utf-8", errors="replace"),1):
        for marker in ("DPM_V2_DIAG ", "DPM_V2_BANK_UPDATE "):
            pos = line.find(marker)
            if pos >= 0:
                try:
                    entry=json.loads(line[pos+len(marker):])
                except json.JSONDecodeError as e:
                    raise ValueError(f"Malformed JSON at {log_path}:{lineno}") from e
                if not isinstance(entry,dict) or "event" not in entry:
                    raise ValueError(f"Incomplete DPM log at {log_path}:{lineno}")
                records.append(entry)
                break
    return records


def flatten_event(event, run):
    base={"run":run,"event":event["event"],"epoch":event["epoch"]}
    if event["event"]=="dpm_v2_attention_cost":
        wanted=[
            "city","target_group","mode","strength","source_count",
            "target_count","bank_feedback_steps","cost_reference",
            "sample_bonus_mean","sample_bonus_std","sample_extra_mean",
            "sample_extra_std","sample_cost_original_mean",
            "sample_cost_modified_mean","sample_source_rank_change",
            "sample_target_rank_change","candidate_bonus_median",
            "candidate_bonus_iqr_median","candidate_base_cost_iqr_median",
            "candidate_base_cost_iqr_p90","candidate_nonzero_margin_fraction",
            "absolute_source_trust_mean",
        ]
        base.update({k:event.get(k) for k in wanted})
        for name in ("q_band_mean","r_band_mean","source_skill_band_mean",
                     "target_need_band_mean","source_attention_top_mass_by_band",
                     "target_attention_top_mass_by_band"):
            a=event.get(name)
            if isinstance(a,list) and len(a)==3:
                for i,band in enumerate(("short","mid","long")):
                    base[name+"_"+band]=a[i]
        audit=event.get("subset_sinkhorn") or {}
        for k in ("epsilon","transport_l1_change","transport_linf_change",
                  "original_pair_bonus_mass","v2_pair_bonus_mass",
                  "original_pair_base_cost","v2_pair_base_cost",
                  "max_marginal_error"):
            base["subset_pi_"+k]=audit.get(k)
    else:
        for key in (
            "step","cities","momentum","q_min","q_max","r_min","r_max",
            "q_active_prototypes","r_active_prototypes"
        ):
            base[key]=event.get(key)
        for name in ("q_band_before","q_band_observed","q_band_after",
                     "r_band_before","r_band_observed","r_band_after",
                     "q_band_delta_l1","r_band_delta_l1"):
            a=event.get(name)
            if isinstance(a,list) and len(a)==3:
                for i,band in enumerate(("short","mid","long")):
                    base[name+"_"+band]=a[i]
    return base


def write_csv(path,rows):
    if not rows:
        return
    cols=[]
    for row in rows:
        for key in row:
            if key not in cols: cols.append(key)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-root",required=True,type=Path)
    p.add_argument("--allow-partial",action="store_true")
    args=p.parse_args()
    root=args.run_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    pairs=[]; updates=[]; group=defaultdict(list); runs=[]
    for log in sorted(root.glob("*/seed*/train.out")):
        run=log.parent
        if not args.allow_partial and not (run/"COMPLETED").exists():
            print("SKIP incomplete",run)
            continue
        events=read_events(log)
        if not events:
            print("SKIP no V2 events",run)
            continue
        label=f"{run.parent.name}/{run.name}"
        config={}
        cfg=run/"run_config.txt"
        if cfg.is_file():
            config=dict(
                line.split("=",1) for line in cfg.read_text().splitlines()
                if "=" in line
            )
        if config.get("mode") not in ("v2a","v2b"):
            raise ValueError("Not a V2-A/B run: "+str(run))
        costs=[e for e in events if e["event"]=="dpm_v2_attention_cost"]
        bank=[e for e in events if e["event"]=="dpm_v2_bank_update"]
        if not costs or not bank:
            raise ValueError("V2 run lacks pair cost or bank update evidence: "+str(run))
        for event in costs:
            row=flatten_event(event,label)
            pairs.append(row)
            group[(label,event["city"])].append(row)
        for event in bank:
            updates.append(flatten_event(event,label))
        runs.append({
            "run":label,
            "code_commit":config.get("commit"),
            "mode":config.get("mode"),
            "cost_strength":config.get("dpm_strength"),
            "pair_cost_records":len(costs),
            "bank_update_records":len(bank),
            "first_bank_step":min(x["step"] for x in bank),
            "last_logged_bank_step":max(x["step"] for x in bank),
            "last_logged_pattern_q":bank[-1].get("q_band_after"),
            "last_logged_pattern_r":bank[-1].get("r_band_after"),
            "completed":(run/"COMPLETED").is_file(),
        })
    write_csv(root/"dpm_v2_pair_cost_diagnostics.csv",pairs)
    write_csv(root/"dpm_v2_bank_update_history.csv",updates)
    overview=[]
    for (name,city),rows in sorted(group.items()):
        valid_pi=[r["subset_pi_transport_l1_change"] for r in rows
                  if r.get("subset_pi_transport_l1_change") is not None]
        summary={
            "run":name,"city":city,"records":len(rows),
            "cost_add_mean":statistics.mean(r["sample_extra_mean"] for r in rows),
            "benefit_std_mean":statistics.mean(r["sample_bonus_std"] for r in rows),
            "small_pair_rank_change_mean":statistics.mean(
                r["sample_source_rank_change"] for r in rows),
            "subset_pi_l1_change_mean":(
                statistics.mean(valid_pi) if valid_pi else None
            ),
            "subset_pi_log_count":len(valid_pi),
        }
        overview.append(summary)
    result={"runs":runs,"by_source_city":overview,
            "caveats":[
                "64x64 subset Sinkhorn uses an explicit diagnostic epsilon; it is NOT the full GeomLoss Pi.",
                "Training rows contain target three-day SUPPORT labels only; no target validation/test labels.",
                "Improvements require matched-code F2 control, full Stage-I+official Stage-II and repeated seeds.",
            ]}
    destination=root/"dpm_v2_analysis.json"
    destination.write_text(
        json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print("Saved:",destination)


if __name__=="__main__":
    main()
