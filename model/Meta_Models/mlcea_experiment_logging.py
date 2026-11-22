"""Structured Stage-I run/epoch/evaluation logging; no model changes."""
import csv
import json
import os
from pathlib import Path
import sys
import subprocess

import numpy as np
import torch


EPOCH_FIELDS = [
    "epoch", "seed", "mode", "ot_status", "mlcea_enabled", "mlcea_active",
    "ot_active", "train_source_loss", "train_target_loss", "da_loss",
    "mlcea_aux_loss", "mlcea_aux_weighted_loss", "total_loss",
    "train_source_mae", "train_source_rmse", "train_source_mse",
    "train_source_mape", "active_source_pairs", "skipped_source_pairs",
    "metric_grad_norm", "future_grad_norm",
    "metric_parameter_step_norm", "future_parameter_step_norm",
    "main_lr", "mlcea_lr", "epoch_wall_seconds", "peak_cuda_memory_mb",
]
EVAL_FIELDS = [
    "epoch", "seed", "horizon_step", "horizon_minutes", "mode",
    "target_fewshot_train_mae", "target_heldout_test_mae",
    "target_fewshot_train_rmse", "target_heldout_test_rmse",
    "target_fewshot_train_mape", "target_heldout_test_mape",
    "target_fewshot_train_mean_mae", "target_heldout_test_mean_mae",
    "train_score_used_for_checkpoint", "is_selected_by_train_mae",
]


def _append_csv(folder, filename, fields, rows):
    path = Path(folder) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="raise")
        if out.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
    return path


def write_run_manifest(args, upstream="serre20/PMME (PMME_and_Others)"):
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True,
            text=True, timeout=3, check=False,
        ).stdout.strip() or "unavailable"
        git_branch = subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True,
            text=True, timeout=3, check=False,
        ).stdout.strip() or "detached_or_unavailable"
    except (OSError, subprocess.TimeoutExpired):
        git_sha, git_branch = "unavailable", "unavailable"
    output = {
        "upstream": upstream,
        "git_commit": git_sha,
        "git_branch": git_branch,
        "branch_purpose": "Stage-I MLCEA vs author baseline",
        "stage2_source_changed": False,
        "seed": int(args.seed),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "python_executable": sys.executable,
        "working_directory": os.getcwd(),
        "command": list(sys.argv),
        "args": {key: str(value) for key, value in vars(args).items()},
        "schema_version": 2,
    }
    path = Path(args.save_dir) / "mlcea_run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return path


def log_stage1_epoch(args, model, epoch, model_loss,
                     mse, rmse, mae, mape, elapsed):
    stat = getattr(model, "last_mlcea_epoch", {}) or {}
    mlcea_flag = bool(getattr(args, "mlcea_enable", 0))
    ot_active = args.DA == "OT" and epoch >= args.DAepoch
    status = (
        "not_OT" if args.DA != "OT"
        else "waiting_for_DAepoch" if epoch < args.DAepoch
        else "active" if stat.get("active_source_pairs", 0) > 0
        else "all_pairs_skipped" if mlcea_flag else "official_OT_active"
    )
    stats = {
        "epoch": int(epoch),
        "seed": int(args.seed),
        "mode": "mlcea" if mlcea_flag else "official",
        "ot_status": status,
        "mlcea_enabled": int(mlcea_flag),
        "mlcea_active": int(stat.get("mlcea_active", 0)),
        "ot_active": int(ot_active),
        "train_source_loss": stat.get("source_task_loss", ""),
        "train_target_loss": stat.get("target_task_loss", ""),
        "da_loss": stat.get("da_loss", ""),
        "mlcea_aux_loss": stat.get("mlcea_aux_loss", ""),
        "mlcea_aux_weighted_loss": stat.get("mlcea_aux_weighted_loss", ""),
        "total_loss": float(model_loss),
        "train_source_mae": float(mae),
        "train_source_rmse": float(rmse),
        "train_source_mse": float(mse),
        "train_source_mape": float(mape),
        "active_source_pairs": int(stat.get("active_source_pairs", 0)),
        "skipped_source_pairs": int(stat.get("skipped_source_pairs", 0)),
        "metric_grad_norm": stat.get("metric_grad_norm", ""),
        "future_grad_norm": stat.get("future_grad_norm", ""),
        "metric_parameter_step_norm": stat.get("metric_parameter_step_norm", ""),
        "future_parameter_step_norm": stat.get("future_parameter_step_norm", ""),
        "main_lr": float(model.main_optim.param_groups[0]["lr"]),
        "mlcea_lr": float(model.mlcea_optim.param_groups[0]["lr"])
                    if mlcea_flag else "",
        "epoch_wall_seconds": float(elapsed),
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated() / 1024**2)
            if torch.cuda.is_available() else 0.0
        ),
    }
    _append_csv(args.save_dir, "stage1_epoch_metrics.csv", EPOCH_FIELDS, [stats])


def log_stage1_evaluation(args, epoch, train_scores, test_scores,
                          previous_best_train_score, details=None):
    """Official so-called validation_score is target few-shot TRAIN, not validation."""
    details = details or {}
    train = np.asarray(train_scores, dtype=float).reshape(-1)
    test = np.asarray(test_scores, dtype=float).reshape(-1)
    if train.size != test.size or train.size == 0:
        raise ValueError("Unexpected MAE horizon score lengths")
    idx = list(range(1, len(train), 2))
    train_selected_score = float(train[idx].sum())
    is_best = train_selected_score < float(previous_best_train_score)
    values = [{
        "epoch": int(epoch),
        "seed": int(args.seed),
        "horizon_step": int(k + 1),
        "horizon_minutes": int(5 * (k + 1)),
        "mode": "mlcea" if getattr(args, "mlcea_enable", 0) else "official",
        "target_fewshot_train_mae": float(t),
        "target_heldout_test_mae": float(s),
        "target_fewshot_train_rmse": float(details["target_train_rmse"][k])
            if "target_train_rmse" in details else "",
        "target_heldout_test_rmse": float(details["target_test_rmse"][k])
            if "target_test_rmse" in details else "",
        "target_fewshot_train_mape": float(details["target_train_mape"][k])
            if "target_train_mape" in details else "",
        "target_heldout_test_mape": float(details["target_test_mape"][k])
            if "target_test_mape" in details else "",
        "target_fewshot_train_mean_mae": float(train.mean()),
        "target_heldout_test_mean_mae": float(test.mean()),
        "train_score_used_for_checkpoint": train_selected_score,
        "is_selected_by_train_mae": int(is_best),
    } for k, (t, s) in enumerate(zip(train, test))]
    _append_csv(args.save_dir, "stage1_eval_metrics.csv", EVAL_FIELDS, values)


STAGE2_EVAL_FIELDS = [
    "seed", "phase", "horizon_step", "horizon_minutes",
    "test_mae", "test_mean_mae", "mode",
]


def log_stage2_evaluation(args, phase, test_scores):
    """Observe outputs of original Stage-II evaluator without modifying it."""
    values = np.asarray(test_scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Empty Stage-II test MAE scores")
    rows = [{
        "seed": int(args.seed),
        "phase": str(phase),
        "horizon_step": int(k + 1),
        "horizon_minutes": int(5 * (k + 1)),
        "test_mae": float(v),
        "test_mean_mae": float(values.mean()),
        "mode": "mlcea" if getattr(args, "mlcea_enable", 0) else "official",
    } for k, v in enumerate(values)]
    _append_csv(args.save_dir, "stage2_eval_metrics.csv",
                STAGE2_EVAL_FIELDS, rows)


OT_HEALTH_FIELDS = [
    "epoch", "seed", "source_city", "target_city", "mode", "status",
    "source_mass_keep", "target_mass_keep", "ot_loss_scale", "ot_loss",
    "base_cost_probe_min", "base_cost_probe_p50", "base_cost_probe_p95",
    "base_cost_probe_p99", "base_cost_probe_max",
    "source_y_revin_abs_max", "target_y_revin_abs_max",
    "source_y_revin_abs_p99", "target_y_revin_abs_p99",
    "source_bad_node", "target_bad_node",
    "source_bad_window", "target_bad_window",
    "source_hist_std_min", "target_hist_std_min",
    "source_hist_std_p01", "target_hist_std_p01",
    "source_hist_std_lt_0p001", "target_hist_std_lt_0p001",
    "source_huge_y_rate", "target_huge_y_rate",
]


def log_ot_health(args, epoch, source_city, target_city, source_joint,
                  target_joint, source_data, target_data, ratio1,
                  ratio2, ot_loss_scale, ot_loss):
    """Common official/MLCEA anomaly audit, without modifying OT.

    Joint feature vectors contain last 36 dimensions of RevIN future,
    multiplied by sqrt(xySquaredRatio) in official code.
    Only call every 25 epochs; all reductions are detached.
    """
    with torch.no_grad():
        scale = float(args.xySquaredRatio) ** 0.5
        if not scale:
            raise ValueError("xySquaredRatio must be positive")
        source_revin = source_joint[:, -36:] / scale
        target_revin = target_joint[:, -36:] / scale
        source_bad = int(source_revin.abs().amax(-1).argmax())
        target_bad = int(target_revin.abs().amax(-1).argmax())

        def location(data, flat_idx):
            n = int(data.x.shape[1])
            row = int(flat_idx // n)
            col = int(flat_idx % n)
            window = int(data.mlcea_window_index[row])
            return col, window

        s_node, s_window = location(source_data, source_bad)
        t_node, t_window = location(target_data, target_bad)
        sh = torch.as_tensor(source_data.x[..., 0])
        th = torch.as_tensor(target_data.x[..., 0])
        s_std = sh.var(-1, unbiased=False).add(1e-9).sqrt()
        t_std = th.var(-1, unbiased=False).add(1e-9).sqrt()
        ix = torch.linspace(0, len(source_joint)-1,
                            steps=min(48,len(source_joint)),
                            device=source_joint.device).long()
        iy = torch.linspace(0, len(target_joint)-1,
                            steps=min(48,len(target_joint)),
                            device=target_joint.device).long()
        probe = torch.cdist(source_joint[ix], target_joint[iy], p=2)
        valid = torch.isfinite(probe)
        vals = probe[valid]
        finite = vals.numel() > 0
        def percentile(data, q):
            return float(torch.quantile(data, q)) if data.numel() else float("nan")
        out = dict(
            epoch=int(epoch), seed=int(args.seed),
            source_city=str(source_city), target_city=str(target_city),
            mode="mlcea" if getattr(args,"mlcea_enable",0) else "official",
            status="active" if ratio1 > 0 and ratio2 > 0 else "DVGP_skipped",
            source_mass_keep=float(ratio1), target_mass_keep=float(ratio2),
            ot_loss_scale=float(ot_loss_scale),
            ot_loss=float(ot_loss.detach()) if torch.is_tensor(ot_loss) else float(ot_loss),
            base_cost_probe_min=float(vals.min()) if finite else float("nan"),
            base_cost_probe_p50=percentile(vals,0.50),
            base_cost_probe_p95=percentile(vals,0.95),
            base_cost_probe_p99=percentile(vals,0.99),
            base_cost_probe_max=float(vals.max()) if finite else float("nan"),
            source_y_revin_abs_max=float(source_revin.abs().max()),
            target_y_revin_abs_max=float(target_revin.abs().max()),
            source_y_revin_abs_p99=percentile(source_revin.abs().reshape(-1),0.99),
            target_y_revin_abs_p99=percentile(target_revin.abs().reshape(-1),0.99),
            source_bad_node=s_node, target_bad_node=t_node,
            source_bad_window=s_window, target_bad_window=t_window,
            source_hist_std_min=float(s_std.min()),
            target_hist_std_min=float(t_std.min()),
            source_hist_std_p01=percentile(s_std,0.01),
            target_hist_std_p01=percentile(t_std,0.01),
            source_hist_std_lt_0p001=float((s_std < 0.001).float().mean()),
            target_hist_std_lt_0p001=float((t_std < 0.001).float().mean()),
            source_huge_y_rate=float((source_revin.abs()>100).float().mean()),
            target_huge_y_rate=float((target_revin.abs()>100).float().mean()),
        )
    _append_csv(args.save_dir,"ot_numeric_health.csv",OT_HEALTH_FIELDS,[out])
    return out
