#!/usr/bin/env python3
"""Summarize H1 full-shadow vs corrected readout-only predictions.

Usage:
  python summarize_h1_shadow.py PATH/TO/train.out
  python summarize_h1_shadow.py PATH/TO/train.out --csv records.csv --summary summary.txt

The 'full' quantity is a one-step in-support AdamW shadow counterfactual,
NOT a measured future test-set change and NOT multi-step generalization.
"""
import argparse
import ast
import csv
import math
import statistics
from pathlib import Path


KEYS = [
    "epoch", "full_no_ot", "full_f2", "full_gain", "head_no_ot",
    "head_f2", "head_gain", "gain_sign_match", "ot_to_source_norm",
    "ot_to_target_norm", "ot_target_cos",
]
for k in ("full_band_gains", "head_band_gains"):
    for j in range(3):
        KEYS.append(f"{k}_{j}")


def read_records(log_path):
    records = []
    marker = "h1_shadow_audit "
    for line_number, line in enumerate(Path(log_path).open(
            encoding="utf-8", errors="replace"), start=1):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            row = ast.literal_eval(payload)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid H1 record at {log_path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected dict in H1 record at line {line_number}")
        if row.get("epoch") is None or any(k not in row for k in
                                              ("full_gain", "head_gain",
                                               "gain_sign_match")):
            raise ValueError(f"Incomplete H1 record at line {line_number}")
        if any(not math.isfinite(float(row[k])) for k in
               ("full_gain", "head_gain", "full_no_ot", "full_f2")):
            raise ValueError(f"Non-finite H1 record at line {line_number}")
        records.append(row)
    records.sort(key=lambda d: d["epoch"])
    return records


def summarize(records):
    if not records:
        raise ValueError("No h1_shadow_audit records found")
    full = [float(r["full_gain"]) for r in records]
    head = [float(r["head_gain"]) for r in records]
    errors = [a-b for a,b in zip(full,head)]
    lines = [
        f"number_of_audits={len(records)}",
        f"epochs={','.join(str(r['epoch']) for r in records)}",
        f"mean_full_shadow_gain={statistics.mean(full):.9g}",
        f"median_full_shadow_gain={statistics.median(full):.9g}",
        f"mean_head_proxy_gain={statistics.mean(head):.9g}",
        f"median_head_proxy_gain={statistics.median(head):.9g}",
        f"mean_abs_head_full_disagreement={statistics.mean(abs(x) for x in errors):.9g}",
        f"sign_agreement_rate={statistics.mean(float(r['gain_sign_match']) for r in records):.6f}",
        f"full_gain_positive_fraction={statistics.mean(x > 0 for x in full):.6f}",
        f"head_gain_positive_fraction={statistics.mean(x > 0 for x in head):.6f}",
        f"median_ot_to_target_norm={statistics.median(float(r['ot_to_target_norm']) for r in records):.9g}",
        "NOTE: full_shadow_gain is full-parameter ONE-STEP F2-vs-no-OT with identical stochastic draws.",
        "NOTE: head_proxy_gain updates ONLY projection1, holding encoder/prediction-head fixed.",
        "NOTE: probe windows are training-support windows; not independent validation or test data.",
        "NOTE: a positive one-step gain is not proof of improved 180-min generalization.",
    ]
    if len(records)>2 and statistics.pstdev(full)>1e-12 and statistics.pstdev(head)>1e-12:
        lines.append(f"pearson_full_head_gain={statistics.correlation(full,head):.6f}")
    else:
        lines.append("pearson_full_head_gain=undefined")
    return "\n".join(lines)+"\n"


def export_csv(records, path):
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=KEYS, extrasaction="ignore")
        writer.writeheader()
        for item in records:
            row = {k:v for k,v in item.items() if k in KEYS}
            for k in ("full_band_gains", "head_band_gains"):
                values = item.get(k, [float("nan")]*3)
                for j in range(3):
                    row[f"{k}_{j}"] = values[j]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_log")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()
    records = read_records(args.train_log)
    message = summarize(records)
    print(message, end="")
    export_csv(records, args.csv)
    if args.summary:
        Path(args.summary).write_text(message, encoding="utf-8")


if __name__ == "__main__":
    main()
