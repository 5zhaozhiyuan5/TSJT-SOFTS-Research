#!/usr/bin/env python3
"""Aggregate read-only D1/D2 OT/probe alignment diagnostic JSONL.

Each row corresponds to one resampled three-day-support batch and one source
city. These rows are correlated; positive rates are descriptive, not p-values.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def mean(values):
    return sum(values)/len(values) if values else float("nan")


def aggregate(rows):
    if not rows:
        raise ValueError("No audit records")
    grouped = defaultdict(list)
    for row in rows:
        if row.get("audit_version") != "d1d2-v1":
            raise ValueError("Unexpected D1/D2 audit version")
        if row.get("training_updates") != 0:
            raise ValueError("Training updates detected in diagnostic artifact")
        grouped[row["city"]].append(row)
        grouped["ALL"].append(row)
    metrics = [
        "D1_cos_high_minus_low_gain", "D1_f2_minus_uniform_gain",
        "D2_sign_low_minus_high_gain_given_mid_cos",
        "D2_sign_minus_f2_gain",
    ]
    results = []
    for city, group in sorted(grouped.items()):
        result = {"city":city, "n_records":len(group),
                  "n_unique_probe_batches":len(set(
                      x["probe_batch_index"] for x in group
                  )) }
        for name in metrics:
            v = [float(x["contrasts"][name]) for x in group]
            result[name+"_mean"] = mean(v)
            result[name+"_positive_fraction"] = mean([x>0 for x in v])
            result[name+"_negative_fraction"] = mean([x<0 for x in v])
        result["cos_conflict_pearson_mean"] = mean(
            [x["cos_conflict_pearson"] for x in group]
        )
        result["d2_cosine_gap_mean"] = mean([
            x["candidate_groups"]["sign_low_cosmatched"]["mean_cos"] -
            x["candidate_groups"]["sign_high_cosmatched"]["mean_cos"]
            for x in group
        ])
        for method in ("uniform","f2","sign"):
            result[method+"_ot_probe_cos_mean"] = mean([
                x["candidate_OT"][method]["ot_probe_cos"] for x in group
            ])
        results.append(result)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()
    if not args.jsonl.is_file():
        p.error("Missing JSONL: "+str(args.jsonl))
    rows = []
    with args.jsonl.open(encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            if line.strip():
                row = json.loads(line)
                if row["audit_version"] != "d1d2-v1":
                    raise ValueError("Bad audit version at line "+str(i))
                rows.append(row)
    report = aggregate(rows)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=list(report[0]))
            writer.writeheader()
            writer.writerows(report)
    print("D1/D2 original-F2 diagnostic; positive dot means favorable "
          "raw first-order OT direction on PROBE prediction loss.")
    print("These are IN-SUPPORT counterfactuals; NOT forecast test MAE, "
          "full PCGrad/AdamW gain or statistical significance.")
    for row in report:
        print("CITY",row["city"],"samples",row["n_records"],
              "distinct_probe_batches",row["n_unique_probe_batches"])
        for key in (
            "D1_cos_high_minus_low_gain",
            "D1_f2_minus_uniform_gain",
            "D2_sign_low_minus_high_gain_given_mid_cos",
            "D2_sign_minus_f2_gain"
        ):
            print(" ", key, "mean=%.7g" % row[key+"_mean"],
                  "positive_fraction=%.3f" % row[key+"_positive_fraction"])
        print("  cosine_conflict_correlation=%.4f" %
              row["cos_conflict_pearson_mean"],
              "cosine_gap_under_D2_matched=%.4f" %
              row["d2_cosine_gap_mean"])
        print("  OT/probe cosine uniform=%.4f F2=%.4f sign=%.4f" %
              (row["uniform_ot_probe_cos_mean"],
               row["f2_ot_probe_cos_mean"],
               row["sign_ot_probe_cos_mean"]))


if __name__ == "__main__":
    main()
