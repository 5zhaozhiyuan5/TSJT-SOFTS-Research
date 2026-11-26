#!/usr/bin/env python3
"""Summarize preregistered H5 one-step paired *in-support* shadow diagnostics.

The mean/median and 6/8 criteria are NOT evidence of independent validation.
Every candidate uses the exact same F2 checkpoint and target support period.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean,median

METHODS=("prc","demand_skill","demand_gradient","joint",
         "rotated_demand","shuffled_skill")


def analyze(rows):
    if len(rows)<1:
        raise ValueError("Empty H5 results")
    table=defaultdict(list)
    by_batch={}
    noise=[]
    demand_ranges=defaultdict(list)
    ess_ratios=defaultdict(list)
    for row in rows:
        if row.get("audit_version")!="h5-v1" or row.get("training_updates")!=0:
            raise ValueError("Not a valid read-only H5 record")
        b=int(row["batch_index"])
        if b in by_batch:
            raise ValueError("Duplicate batch index")
        by_batch[b]=row
        cities=row["city_diagnostics"]
        if len(cities)!=3:
            raise ValueError("Expected 3 cities")
        for cd in cities:
            city=cd["city"]
            demand_ranges[city].append(cd["target_fit_demand"]["demand_range"])
            for name in METHODS:
                key=f"policy:{city}:{name}"
                if key not in row["gains"]:
                    raise ValueError("Missing preregistered candidate "+key)
                gain=float(row["gains"][key]["gain_mae"])
                table[(city,name)].append((b,gain,row["gains"][key]["gain_bands"]))
                if name=="joint":
                    v=cd["candidate_alpha"]["joint"]
                    ess_ratios[city].append(v["ess_after"]/max(v["ess_f2"],1e-8))
        identity=f"policy:{cities[0]['city']}:identity"
        if identity not in row["gains"]:
            raise ValueError("Missing identical-F2 replay")
        noise.append(abs(float(row["gains"][identity]["gain_mae"])))
    if sorted(by_batch)!=list(range(len(rows))):
        raise ValueError("Expected contiguous probe batch indices")
    summaries=[]
    for (city,name),points in sorted(table.items()):
        xs=[x[1] for x in points]
        pcount=sum(x>0 for x in xs)
        ncount=sum(x<0 for x in xs)
        absolute_sum=sum(abs(x) for x in xs)
        summaries.append({
            "city":city,"method":name,"n":len(xs),
            "mean_gain_mae":mean(xs),"median_gain_mae":median(xs),
            "positive_batches":pcount,"negative_batches":ncount,
            "max_absolute_share":max(abs(x) for x in xs)/max(absolute_sum,1e-12),
            "band0_mean":mean(x[2][0] for x in points),
            "band1_mean":mean(x[2][1] for x in points),
            "band2_mean":mean(x[2][2] for x in points),
            "discovery_mean_first4":mean(xs[:min(len(xs),4)]),
            "confirmation_mean_last4":mean(xs[4:]) if len(xs)>4 else float("nan"),
            "target_demand_range_mean":mean(demand_ranges[city]),
            "joint_ess_ratio_mean":mean(ess_ratios[city]),
            "identity_shadow_noise_max":max(noise)
        })
    contrasts=[]
    for city in sorted(demand_ranges):
        joint={b:g for b,g,_ in table[(city,"joint")]}
        for other in ("prc","demand_skill","demand_gradient",
                      "rotated_demand","shuffled_skill"):
            baseline={b:g for b,g,_ in table[(city,other)]}
            differences=[joint[b]-baseline[b] for b in sorted(joint)]
            contrasts.append({
                "city":city,"joint_minus":other,"n":len(differences),
                "mean_paired_gain":mean(differences),
                "median_paired_gain":median(differences),
                "positive_batches":sum(x>0 for x in differences),
                "discovery_mean_first4":mean(differences[:min(4,len(differences))]),
                "confirmation_mean_last4":mean(differences[4:]) if len(differences)>4 else float("nan"),
                "max_absolute_share":max(abs(x) for x in differences)/
                                      max(sum(abs(x) for x in differences),1e-12)
            })
    return summaries,contrasts,max(noise)


def write_table(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--jsonl",type=Path,required=True)
    p.add_argument("--csv",type=Path,required=True)
    p.add_argument("--contrasts",type=Path,required=True)
    args=p.parse_args()
    rows=[json.loads(x) for x in args.jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
    stats,contrasts,noise=analyze(rows)
    write_table(args.csv,stats)
    write_table(args.contrasts,contrasts)
    print("H5 target-conditioned counterfactual: positive gain means better than F2")
    print("NO F2 TRAINING; NO independent validation; one-step support-only proxies")
    print("Identity numerical noise max:",f"{noise:.10f}")
    for v in stats:
        print(v["city"],v["method"],
              f"mean={v['mean_gain_mae']:+.8f}",
              f"median={v['median_gain_mae']:+.8f}",
              f"positive={v['positive_batches']}/{v['n']}",
              f"outlier_share={v['max_absolute_share']:.2f}")
    print("Paired contrasts: joint minus alternative (positive better)")
    for v in contrasts:
        print(v["city"],"vs",v["joint_minus"],
              f"mean={v['mean_paired_gain']:+.8f}",
              f"median={v['median_paired_gain']:+.8f}",
              f"positive={v['positive_batches']}/{v['n']}")
    print("Interpret only after comparing first four vs last four probes,")
    print("and never claim statistical significance across correlated support windows.")


if __name__=="__main__":
    main()
