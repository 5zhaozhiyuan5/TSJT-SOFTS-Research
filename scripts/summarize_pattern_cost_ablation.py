#!/usr/bin/env python3
"""Audit the strict two-arm experiment and summarize FINAL (not cherry-picked) test scores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_ckpt(path: Path):
    # Only locally generated experiment files should be passed to torch.load.
    return torch.load(str(path), map_location="cpu", weights_only=False)


def check_pre_alignment(first: Path, second: Path) -> float:
    left = load_ckpt(first)
    right = load_ckpt(second)
    if left["epoch"] != right["epoch"]:
        raise RuntimeError("Pre-alignment checkpoint epochs differ")
    a, b = left["backbone_state_dict"], right["backbone_state_dict"]
    if set(a) != set(b):
        raise RuntimeError("Pre-alignment backbone parameter sets differ")
    maximum = 0.0
    for key in a:
        av, bv = a[key], b[key]
        if av.shape != bv.shape:
            raise RuntimeError(f"Pre-alignment parameter shape mismatch: {key}")
        if not torch.is_floating_point(av):
            if not torch.equal(av, bv):
                raise RuntimeError(f"Non-float pre-alignment value mismatch: {key}")
            continue
        maximum = max(maximum, float((av.float() - bv.float()).abs().max()))
    return maximum


def get_run(root: Path, arm: str, seed: int):
    directory = root / f"{arm}_seed{seed}"
    meta = load_json(directory / "run_metadata.json")["metadata"]
    final = load_json(directory / "final_metrics.json")
    expected_weight = {"no_pattern": 0.0, "uniform": 0.5}[arm]
    if abs(final["pattern_weight"] - expected_weight) > 1e-9:
        raise RuntimeError(f"{arm} has wrong pattern_weight: {final['pattern_weight']}")
    if final["router_mode"] != "uniform":
        raise RuntimeError(f"{arm} has wrong router mode: {final['router_mode']}")
    if final["protocol"] != "fixed_final_checkpoint_no_test_selection":
        raise RuntimeError(f"{arm} is not a strict final-checkpoint run")
    stage1 = load_ckpt(Path(meta["stage1_checkpoint_path"]))
    stage2 = load_ckpt(Path(meta["stage2_checkpoint_path"]))
    if stage1["epoch"] != final["stage1_checkpoint_epoch"]:
        raise RuntimeError(f"{arm} Stage-I checkpoint epoch mismatch")
    if stage2["epoch"] != final["stage2_checkpoint_epoch"]:
        raise RuntimeError(f"{arm} Stage-II checkpoint epoch mismatch")
    return directory, meta, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    records = {arm: get_run(args.root, arm, args.seed) for arm in ("no_pattern", "uniform")}
    left_dir, left_meta, left_report = records["no_pattern"]
    right_dir, right_meta, right_report = records["uniform"]
    for field in ("target_dataset", "source_dataset_selector", "target_days",
                  "processed_interval_minutes", "history_steps", "prediction_steps",
                  "target_normalization_scope", "source_train_epochs", "stage2_finetune_epochs"):
        if left_meta.get(field) != right_meta.get(field):
            raise RuntimeError(f"Protocol mismatch: {field}")

    pre_maxdiff = check_pre_alignment(
        left_dir / "stage1_pre_alignment_backbone.pth",
        right_dir / "stage1_pre_alignment_backbone.pth",
    )
    if pre_maxdiff > 1e-7:
        raise RuntimeError(
            f"Stage-I pre-alignment backbone weights differ (max diff={pre_maxdiff:.3g}). "
            "Rerun with a shared initial checkpoint or investigate nondeterminism."
        )
    rows = []
    print("Pre-OT backbone maximum parameter diff:", pre_maxdiff)
    print("Uniform minus NoPattern: negative change means a better forecast.")
    print(f"{'Horizon':>8} {'NoPattern RMSE':>15} {'NoPattern MAE':>14} {'Uniform RMSE':>14} {'Uniform MAE':>13} {'dRMSE':>10} {'dMAE':>10}")
    for minutes in (10, 60, 120, 180):
        key = str(minutes)
        no = left_report["results"][key]
        uni = right_report["results"][key]
        row = {
            "horizon_minutes": minutes,
            "no_pattern_rmse": no["rmse"],
            "no_pattern_mae": no["mae"],
            "uniform_rmse": uni["rmse"],
            "uniform_mae": uni["mae"],
            "delta_rmse_uniform_minus_no_pattern": uni["rmse"] - no["rmse"],
            "delta_mae_uniform_minus_no_pattern": uni["mae"] - no["mae"],
        }
        rows.append(row)
        print(f"{minutes:>8} {no['rmse']:>15.5f} {no['mae']:>14.5f} {uni['rmse']:>14.5f} {uni['mae']:>13.5f} {row['delta_rmse_uniform_minus_no_pattern']:>+10.5f} {row['delta_mae_uniform_minus_no_pattern']:>+10.5f}")
    with (args.root / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.root / "comparison_audit.json").open("w", encoding="utf-8") as f:
        json.dump({
            "protocol": "same_pre_alignment_backbone_fixed_final_checkpoints",
            "pre_alignment_max_parameter_diff": pre_maxdiff,
            "stage1_checkpoint_epoch": left_report["stage1_checkpoint_epoch"],
            "stage2_checkpoint_epoch": left_report["stage2_checkpoint_epoch"],
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print("Audit passed. Wrote comparison.csv and comparison_audit.json")


if __name__ == "__main__":
    main()
