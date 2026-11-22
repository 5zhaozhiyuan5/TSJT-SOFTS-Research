"""Independent, training-only local-operator validation; no SOFTS or OT training.

Usage:
  python scripts/diagnose_mlcea_operator.py \
      --config configs/config_mlcea_batch8.yaml \
      --target pems-bay --device cuda \
      --mode strict --seed 1

Reports target local ridge vs source-only W0 and source-city ridge transfer
onto held-out target TRAIN-window responses. Never accesses target TEST
windows; no target-model checkpoint is chosen using this script.
"""
import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model" / "Meta_Models"))
from datasets import traffic_dataset
from mlcea import MLCEA


def make_snap(model, x_full, y_full, means, stds, indices, device):
    """Same official global normalisation + RevIN as PatchFSL.forward."""
    h = x_full.to(device=device, dtype=torch.float32)[..., 0]
    y = y_full.to(device=device, dtype=torch.float32)
    mu = h.mean(-1, keepdim=True)
    sd = (h.var(-1, unbiased=False, keepdim=True) + 1e-9).sqrt()
    # The official loader computes citywide means/stds BEFORE splitting;
    # this diagnostic reproduces its normalisation, not a new protocol.
    global_mu = float(np.asarray(means).reshape(-1)[0])
    global_sd = float(np.asarray(stds).reshape(-1)[0])
    y_revin = ((y - global_mu) / global_sd - mu) / sd
    return model.make_snapshot(h, y_revin, window_indices=indices)


@torch.no_grad()
def compare_target(model, source, target, w0, limit=512):
    """History-only nearest source operator; compare on target TRAIN futures."""
    source_fit = model.fit_local(source, w0)
    ids = torch.linspace(0, target["x"].shape[0] - 1,
                         min(limit, target["x"].shape[0]),
                         device=target["x"].device).long()
    tx, tr = target["x"][ids], target["r"][ids]
    mx = model.metric_weights().sqrt()
    sx = source["x"][:, 1:] * mx
    zx = tx[:, 1:] * mx
    dist = (zx.square().sum(-1, keepdim=True) +
            sx.square().sum(-1)[None, :] - 2 * zx @ sx.T).clamp_min(0)
    idx = dist.argmin(-1)  # histories ONLY; future is evaluation label
    chosen = source_fit["W"][idx]
    pred = torch.bmm(chosen, tx.unsqueeze(-1)).squeeze(-1)
    base = tx @ w0.T
    e_local = (pred - tr).square()
    e_base = (base - tr).square()
    return {
        "target_samples": int(tx.shape[0]),
        "local_mse": float(e_local.mean()),
        "shared_mse": float(e_base.mean()),
        "local_better_rate": float((e_local.mean(-1) < e_base.mean(-1)).float().mean()),
        "gain_mse": float((e_base - e_local).mean()),
        "local_horizon_mse": json.dumps(e_local.mean(0).cpu().tolist()),
        "shared_horizon_mse": json.dumps(e_base.mean(0).cpu().tolist()),
        "source_time_eligible_zero_rate": source_fit["time_eligible_zero_rate"],
        "source_strict_fallback_rate": source_fit["strict_fallback_rate"],
        "source_effective_k_mean": float(source_fit["effective_k"].mean()),
    }


def diagnostic_run(cfg, target_name, sources, seed, device,
                   mode="strict", anchors=128, neighbors=16, mix=0.35,
                   ridge=2.0, source_windows=8, target_limit=512):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    with open(cfg, encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    data_args, task = settings["data"], settings["task"]
    # These metadata-only diagnostic datasets still follow the original
    # first-three-days restriction for target_maml.
    source_args = copy.deepcopy(task["source"])
    source_args["batch_size"] = int(source_windows)
    source_args["add_target"] = False
    target_args = copy.deepcopy(task["target_few_shot"])
    target_args["add_target"] = False
    source_ds = traffic_dataset(
        data_args, source_args, data_list=list(sources),
        stage="source", test_data=target_name, add_target=False, target_days=3,
    )
    target_ds = traffic_dataset(
        data_args, target_args, data_list=[target_name],
        stage="target_maml", test_data=target_name, add_target=False,
        target_days=3,
    )
    model = MLCEA(
        anchors=anchors, neighbors=neighbors, ridge=ridge,
        neighbor_uniform_mix=mix, adaptive_ridge=True,
        time_block_mode=mode,
    ).to(device)
    all_source = []
    for city in sources:
        n = source_ds.x_list[city].shape[0]
        sample_ids = torch.linspace(0, n-1, min(n, source_windows)).long()
        snap = make_snap(
            model, source_ds.x_list[city][sample_ids],
            source_ds.y_list[city][sample_ids],
            source_ds.means_list[city], source_ds.stds_list[city],
            sample_ids, device,
        )
        all_source.append(snap)
    target_len = target_ds.x_list[target_name].shape[0]
    target_idx = torch.arange(target_len)
    target_snap = make_snap(
        model, target_ds.x_list[target_name],
        target_ds.y_list[target_name],
        target_ds.means_list[target_name], target_ds.stds_list[target_name],
        target_idx, device,
    )
    w0 = model.fit_reference(all_source)
    # This evaluates target-local performance in a strictly blocked manner
    # when mode=strict; fallback occurs when no time-disjoint examples exist.
    with torch.no_grad():
        fit_target = model.fit_local(target_snap, w0)
        pred_target = torch.bmm(
            fit_target["W"], target_snap["x"].unsqueeze(-1)
        ).squeeze(-1)
        base_target = target_snap["x"] @ w0.T
        e_target = (pred_target-target_snap["r"]).square().mean(-1)
        e_shared = (base_target-target_snap["r"]).square().mean(-1)
        target_diag = {
            "target_local_mse": float(e_target.mean()),
            "target_shared_mse": float(e_shared.mean()),
            "target_local_better_rate": float((e_target < e_shared).float().mean()),
            "target_strict_fallback_rate": fit_target["strict_fallback_rate"],
            "target_time_eligible_zero_rate": fit_target["time_eligible_zero_rate"],
            "target_time_eligible_mean": fit_target["time_eligible_mean"],
            "target_effective_k_mean": float(fit_target["effective_k"].mean()),
        }
    result = []
    for city, source_snap in zip(sources, all_source):
        record = compare_target(model, source_snap, target_snap,
                                w0, limit=target_limit)
        record.update(target_diag)
        record.update(source=city, target=target_name, seed=seed,
                      mode=mode, mix=mix, ridge=ridge,
                      source_windows=source_windows)
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_mlcea_batch8.yaml")
    parser.add_argument("--target", default="pems-bay")
    parser.add_argument("--sources", default="chengdu_m,metr-la,shenzhen")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["diagnostic", "strict"], default="strict")
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--neighbor-uniform-mix", type=float, default=0.35)
    parser.add_argument("--ridge", type=float, default=2.0)
    parser.add_argument("--source-windows", type=int, default=8)
    parser.add_argument("--target-limit", type=int, default=512)
    parser.add_argument("--output", default="mlcea_operator_validation.csv")
    args = parser.parse_args()
    rows = diagnostic_run(
        args.config, args.target, args.sources.split(","), args.seed,
        args.device, mode=args.mode, anchors=args.anchors,
        neighbors=args.neighbors, mix=args.neighbor_uniform_mix,
        ridge=args.ridge, source_windows=args.source_windows,
        target_limit=args.target_limit,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print("{source}: local_mse={local_mse:.4f} "
              "shared_mse={shared_mse:.4f} gain={gain_mse:.4f} "
              "target_time_no_support={target_time_eligible_zero_rate:.1%}"
              .format(**row))
    print("saved", out)


if __name__ == "__main__":
    main()
