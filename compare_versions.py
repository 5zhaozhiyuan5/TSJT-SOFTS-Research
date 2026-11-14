"""Compare two official-Stage-II result JSON files at paper horizons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    rows = []
    wins = {"mae": 0, "rmse": 0}
    for baseline_item, candidate_item in zip(
        baseline["paper_metrics"], candidate["paper_metrics"]
    ):
        row = {
            "horizon_index": baseline_item["horizon_index"],
            "baseline_mae": baseline_item["mae"],
            "candidate_mae": candidate_item["mae"],
            "delta_mae_candidate_minus_baseline": candidate_item["mae"] - baseline_item["mae"],
            "baseline_rmse": baseline_item["rmse"],
            "candidate_rmse": candidate_item["rmse"],
            "delta_rmse_candidate_minus_baseline": candidate_item["rmse"] - baseline_item["rmse"],
        }
        if row["candidate_mae"] < row["baseline_mae"]:
            wins["mae"] += 1
        if row["candidate_rmse"] < row["baseline_rmse"]:
            wins["rmse"] += 1
        rows.append(row)
    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "paper_horizon_count": len(rows),
        "mae_wins": wins["mae"],
        "rmse_wins": wins["rmse"],
        "majority_metric_condition": (
            wins["mae"] > len(rows) / 2 and wins["rmse"] > len(rows) / 2
        ),
        "rows": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
