#!/usr/bin/env python3
"""Parse H2-A1/A2/A3 full-batch shadow diagnostics into three CSV files."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import statistics


def read_records(log_path):
    records = []
    marker = "h2_candidate_audit "
    for line_number, line in enumerate(Path(log_path).open(
            encoding="utf-8", errors="replace"), start=1):
        if marker not in line:
            continue
        try:
            value = ast.literal_eval(line.split(marker, 1)[1].strip())
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Unreadable H2 record on line {line_number}"
            ) from exc
        if not isinstance(value, dict) or not all(
            k in value for k in ("epoch", "total", "folds", "cities")
        ):
            raise ValueError(f"Malformed H2 record on line {line_number}")
        if any(not math.isfinite(float(x["gain_vs_off"]))
               for x in value["total"]["scores"].values()):
            raise ValueError(f"Nonfinite H2 gain on line {line_number}")
        records.append(value)
    if not records:
        raise ValueError("No H2 candidate audit records found")
    epochs = [r["epoch"] for r in records]
    if len(set(epochs)) != len(epochs):
        raise ValueError("Duplicate H2 audit epochs in one training log")
    return sorted(records, key=lambda r: r["epoch"])


def write_table(path, fieldnames, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_rows, city_rows, fold_rows = [], [], []
    winners, fold_agreements = [], []
    margins = []
    city_scores = defaultdict(list)
    city_group_scores = defaultdict(list)
    for r in records:
        epoch = r["epoch"]
        winner = r["total"]["best_global"]
        winners.append(winner)
        fold_agreements.append(int(r["fold_consensus"]))
        gains = []
        for name in ("off", "uniform", "mixed", "f2"):
            row = r["total"]["scores"][name]
            gains.append(float(row["gain_vs_off"]))
            global_rows.append({
                "epoch":epoch,"candidate":name,"loss":row["loss"],
                "gain_vs_off":row["gain_vs_off"],
                "gain_short":row["band_gains_vs_off"][0],
                "gain_mid":row["band_gains_vs_off"][1],
                "gain_long":row["band_gains_vs_off"][2],
                "best_global":winner,
            })
        gains.sort(reverse=True)
        margins.append(gains[0] - gains[1])
        for name, row in r["total"]["scores"].items():
            if name.startswith("city_uniform:"):
                city = name.split(":",1)[1]
                score = float(row["gain_vs_f2"])
                city_scores[city].append(score)
                group = r.get("city_target_groups",{}).get(city,-1)
                city_group_scores[(city,group)].append(score)
                city_rows.append({
                    "epoch":epoch, "city":city,
                    "target_group":r.get("city_target_groups",{}).get(city,-1),
                    "gain_uniform_city_vs_f2":score,
                    "gain_short":row["band_gains_vs_f2"][0],
                    "gain_mid":row["band_gains_vs_f2"][1],
                    "gain_long":row["band_gains_vs_f2"][2],
                    "improved_over_f2":int(score>0),
                })
        for fold, report in enumerate(r["folds"]):
            for name in ("off","uniform","mixed","f2"):
                row = report["scores"][name]
                fold_rows.append({
                    "epoch":epoch, "fold":fold,
                    "candidate":name,"gain_vs_off":row["gain_vs_off"],
                    "gain_short":row["band_gains_vs_off"][0],
                    "gain_mid":row["band_gains_vs_off"][1],
                    "gain_long":row["band_gains_vs_off"][2],
                    "best_global_in_fold":report["best_global"],
                })
    write_table(out_dir/"h2_a1_global.csv",
        ["epoch","candidate","loss","gain_vs_off",
         "gain_short","gain_mid","gain_long","best_global"],global_rows)
    write_table(out_dir/"h2_a2_citywise.csv",
        ["epoch","city","target_group","gain_uniform_city_vs_f2","gain_short",
         "gain_mid","gain_long","improved_over_f2"],city_rows)
    write_table(out_dir/"h2_a3_folds.csv",
        ["epoch","fold","candidate","gain_vs_off",
         "gain_short","gain_mid","gain_long","best_global_in_fold"],fold_rows)
    counts = Counter(winners)
    lines = [
        f"audits={len(records)}",
        f"epochs={','.join(str(r['epoch']) for r in records)}",
        f"full_winner_counts={dict(sorted(counts.items()))}",
        f"median_best_vs_runnerup_gain_margin={statistics.median(margins):.9g}",
        f"fold_winner_consensus_rate={statistics.mean(fold_agreements):.6f}",
        f"fold_winner_matches_full_rate={statistics.mean(int(r['overall_winner_is_fold_consensus']) for r in records):.6f}",
    ]
    for city, scores in sorted(city_scores.items()):
        lines += [
            f"city_{city}_uniform_over_f2_mean_gain={statistics.mean(scores):.9g}",
            f"city_{city}_uniform_over_f2_positive_frac={statistics.mean(x>0 for x in scores):.6f}",
        ]
    for (city,group), scores in sorted(city_group_scores.items()):
        lines.append(
            f"city_{city}_targetgroup{group}_uniform_over_f2_mean_gain="
            f"{statistics.mean(scores):.9g}"
        )
    for candidate in ("uniform","mixed","f2"):
        scores = [
            r["total"]["scores"][candidate]["gain_vs_off"]
            for r in records
        ]
        all_folds_pos = [
            r["global_positive_in_all_folds"][candidate]
            for r in records
        ]
        lines += [
            f"{candidate}_mean_gain_vs_no_ot={statistics.mean(scores):.9g}",
            f"{candidate}_positive_frac={statistics.mean(x>0 for x in scores):.6f}",
            f"{candidate}_positive_in_all_folds_frac={statistics.mean(all_folds_pos):.6f}",
        ]
    lines += [
        "INTERPRETATION: Gains are single-step, in-support shadow Huber differences.",
        "A1: full-size OT weights differ; no-OT reference uses same PCGrad and AdamW.",
        "A2: source-specific uniform swap keeps OTHER TWO source cities at F2.",
        "A3: temporal folds share support and candidate OT gradients; NOT external validation.",
        "H2-A diagnostics NEVER change real F2 gradients or final model training.",
    ]
    summary = "\n".join(lines)+"\n"
    (out_dir/"h2_summary.txt").write_text(summary,encoding="utf-8")
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("train_log")
    p.add_argument("--output-dir",required=True)
    args=p.parse_args()
    print(summarize(read_records(args.train_log),args.output_dir),end="")


if __name__=="__main__":
    main()
