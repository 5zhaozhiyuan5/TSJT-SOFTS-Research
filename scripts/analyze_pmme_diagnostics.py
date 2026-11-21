#!/usr/bin/env python3
"""Summarize read-only PMME target-transfer diagnostics.

No model changes, no GPU, no test-set access. The script cannot establish
causal benefit from associations between weight, error and source similarity.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


MODE_NAMES = {-1: "future_falling", 0: "future_stable", 1: "future_rising"}


def _mean(values):
    good = np.asarray([x for x in values if x is not None and np.isfinite(x)])
    return float(good.mean()) if good.size else None


def _write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze(folder, output):
    summary_file = folder / "pair_summary.jsonl"
    if not summary_file.exists():
        raise SystemExit("No pair_summary.jsonl found under " + str(folder))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for line in summary_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise SystemExit("No pair summaries have been generated yet.")

    mode_rows = []
    difficulty_rows = []
    horizon_rows = []
    source_rows = defaultdict(list)
    grouped = defaultdict(list)

    for record in records:
        filename = folder / record["sample_file"]
        if not filename.exists():
            raise SystemExit("Missing per-node diagnostic file: " + str(filename))
        epoch, src, pair = int(record["epoch"]), record["source_city"], int(record["pair"])
        key = (epoch, src, pair)
        if key in grouped:
            # Multiple independent runs must use different logged_files directories.
            raise SystemExit("Repeated epoch/source/pair: " + str(key))
        grouped[key].append(record)
        source_rows[src].append(record)
        with np.load(filename, allow_pickle=False) as data:
            err = np.asarray(data["eval_mae"], dtype=np.float64)
            weight = np.asarray(data["beta"], dtype=np.float64)
            gp = np.asarray(data["gp_target_probability"], dtype=np.float64)
            support = np.asarray(data["approx_source_distance"], dtype=np.float64)
            modes = np.asarray(data["mode"])
            active = bool(record["ot_active"])
            mass_sum = weight.sum()

            for m, label in MODE_NAMES.items():
                chosen = modes == m
                n = int(chosen.sum())
                if not n:
                    continue
                mass = weight[chosen].sum()
                fraction = n / len(err)
                share = (mass / mass_sum if mass_sum > 0 else np.nan)
                row = {
                    "epoch": epoch, "source_city": src, "pair": pair,
                    "mode": label, "count": n, "count_fraction": fraction,
                    "gp_beta_mass_share": share,
                    "mass_over_count_ratio": (share / fraction if mass_sum > 0 else np.nan),
                    "gp_filtered_fraction": 1.0 - float((weight[chosen] > 0).mean()),
                    "mean_gp_probability": (float(np.nanmean(gp[chosen]))
                                             if np.isfinite(gp[chosen]).any() else np.nan),
                    "eval_mae": float(err[chosen].mean()),
                    "approx_source_distance": float(support[chosen].mean()),
                    "ot_active": int(active),
                }
                mode_rows.append(row)

            quantiles = np.quantile(err, [0.25, 0.50, 0.75])
            difficulty = np.searchsorted(quantiles, err, side="right")
            for q in range(4):
                chosen = difficulty == q
                if not chosen.any():
                    continue
                fraction = float(chosen.mean())
                share = float(weight[chosen].sum() / mass_sum) if mass_sum > 0 else np.nan
                difficulty_rows.append({
                    "epoch": epoch, "source_city": src, "pair": pair,
                    "error_quartile": q + 1, "count": int(chosen.sum()),
                    "mean_eval_mae": float(err[chosen].mean()),
                    "weight_mass_share": share,
                    "count_fraction": fraction,
                    "mass_over_count_ratio": share / fraction,
                    "mean_support_distance": float(support[chosen].mean()),
                    "gp_filtered_fraction": 1.0 - float((weight[chosen] > 0).mean()),
                    "ot_active": int(active),
                })

            for minutes in (10, 60, 120, 180):
                val = np.asarray(data["eval_mae_{}min".format(minutes)], dtype=np.float64)
                horizon_rows.append({
                    "epoch": epoch, "source_city": src, "pair": pair,
                    "horizon_minutes": minutes, "mean_eval_mae": float(val.mean()),
                    "ot_active": int(active),
                })

    _write_csv(output / "mode_by_pair.csv", mode_rows)
    _write_csv(output / "difficulty_by_pair.csv", difficulty_rows)
    _write_csv(output / "horizon_by_pair.csv", horizon_rows)

    source_summary = []
    for src, rs in sorted(source_rows.items()):
        source_summary.append({
            "source_city": src,
            "pair_count": len(rs),
            "ot_active_pair_count": sum(bool(r["ot_active"]) for r in rs),
            "mean_target_mae_eval": _mean([r["eval_mae_mean"] for r in rs]),
            "mean_target_mae_train": _mean([r["train_mae_mean"] for r in rs]),
            "mean_gp_accept_fraction": _mean([
                r["gp_target_nonzero_fraction"] for r in rs]),
            "mean_effective_beta_count": _mean([
                r["beta_effective_count"] for r in rs]),
            "mean_gp_prob_error_corr": _mean([
                r["gp_probability_eval_mae_corr"] for r in rs]),
            "mean_beta_error_corr": _mean([
                r["beta_eval_mae_corr"] for r in rs]),
            "mean_beta_source_dist_corr": _mean([
                r["beta_support_dist_corr"] for r in rs]),
            "mean_approx_support_distance": _mean([
                r["approx_support_distance_mean"] for r in rs]),
        })
    _write_csv(output / "source_summary.csv", source_summary)

    results = {
        "pair_count": len(records),
        "ot_active_pairs": sum(bool(r["ot_active"]) for r in records),
        "sources": sorted(source_rows),
        "mean_gp_beta_eval_error_correlation": _mean([
            r["beta_eval_mae_corr"] for r in records]),
        "mean_gp_prob_eval_error_correlation": _mean([
            r["gp_probability_eval_mae_corr"] for r in records]),
        "interpretation_limits": [
            "GP beta is normalized independently within each source/target pair.",
            "The transfer-support nearest neighbor is approximated over <=N source "
            "node-windows and uses known TRAIN future labels; not a causal benefit estimate.",
            "The training target windows overlap and are repeatedly reused.",
            "These are in-sample target errors; independent chronologically held-out "
            "evaluation and controlled weight interventions are necessary for causal claims.",
            "future_falling/stable/rising are speed-trend bins, NOT observed congestion events.",
            "Upstream normalization and model-selection leakage must be separately audited.",
        ],
    }
    (output / "diagnostic_readout.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("CSV and JSON summaries saved in", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="The run's logged_files/.../diagnostics folder")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    analyze(args.folder, args.output or (args.folder / "analysis"))
