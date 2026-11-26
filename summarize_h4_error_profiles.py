#!/usr/bin/env python3
"""H4 pre-registered source group suppression audit summary.

Positive suppress_gain_mae means downweighting a particular source group
improved target support-probe MAE under ONE virtual update. Results are
correlated, in-support, and do not show independent generalization.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    result=[]
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            result.append(json.loads(raw))
    if not result:
        raise ValueError("No H4 results found")
    return result


def summarize(rows):
    by=defaultdict(list)
    for row in rows:
        if row.get("audit_version")!="h4-v1" or row.get("training_updates")!=0:
            raise ValueError("Unknown audit version or unexpected model updates")
        for key,metric in row["gains"].items():
            # Name is down:city:group. City includes hyphens but not colons.
            mode,city,group=key.split(":",2)
            if mode!="down":raise ValueError("Invalid intervention name")
            cityinfo=next(x for x in row["city_profiles"] if x["city"]==city)
            strength=cityinfo["interventions"][group]
            by[(city,group)].append({
                "batch":row["batch_index"],
                "delta_mae":metric["suppress_gain_mae"],
                "delta_smooth_l1":metric["suppress_gain_smooth_l1"],
                "band":metric["suppress_gain_bands"],
                "mass":strength["group_alpha_mass_before"],
                "count":strength["group_count"],
            })
    output=[]
    for (city,group),items in sorted(by.items()):
        def mean(v):return sum(v)/len(v)
        gains=[x["delta_mae"] for x in items]
        bands=[mean([x["band"][j] for x in items]) for j in range(3)]
        output.append({
            "city":city,"group":group,"n_batches":len(items),
            "mean_suppress_gain_mae":mean(gains),
            "median_suppress_gain_mae":sorted(gains)[len(gains)//2],
            "positive_fraction":sum(x>0 for x in gains)/len(gains),
            "negative_fraction":sum(x<0 for x in gains)/len(gains),
            "mean_suppress_gain_smooth_l1":mean(
                [x["delta_smooth_l1"] for x in items]
            ),
            "short_gain":bands[0],"mid_gain":bands[1],"long_gain":bands[2],
            "mean_alpha_mass":mean([x["mass"] for x in items]),
            "mean_group_count":mean([x["count"] for x in items]),
        })
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--jsonl",required=True,type=Path)
    parser.add_argument("--csv",required=True,type=Path)
    args=parser.parse_args()
    summary=summarize(read_rows(args.jsonl))
    if not summary:
        raise ValueError("All group experiments were skipped; increase OT budget")
    args.csv.parent.mkdir(parents=True,exist_ok=True)
    with args.csv.open("w",newline="",encoding="utf-8") as fp:
        w=csv.DictWriter(fp,fieldnames=list(summary[0]))
        w.writeheader();w.writerows(summary)
    print("H4: source profile group suppression; positive means suppressing "
          "the group improved one-step target support-probe MAE")
    for x in summary:
        print(
            "%s %-12s N=%d mean=%+.8f sign+=%.2f "
            "short/mid/long=%+.8f/%+.8f/%+.8f mass=%.4f" % (
                x["city"],x["group"],x["n_batches"],
                x["mean_suppress_gain_mae"],x["positive_fraction"],
                x["short_gain"],x["mid_gain"],x["long_gain"],
                x["mean_alpha_mass"]
            )
        )
    print("These are correlated, checkpoint-in-support ONE-STEP diagnostics; "
          "no independent test accuracy or final training improvement.")
    print("Saved",args.csv)


if __name__=="__main__":
    main()
