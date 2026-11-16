"""Structured, append-only diagnostics for PMME Stage-I experiments.

This module is deliberately side-effect-only: it records detached scalar/statistical
summaries and never changes model losses, gradients, optimizer state, or predictions.
"""
import csv
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch


def _plain(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return json.dumps(value.tolist(), ensure_ascii=False)
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return json.dumps(value.tolist(), ensure_ascii=False)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class ExperimentDiagnostics:
    """Write structured CSV/JSON diagnostics into one unique run directory."""

    FILES = {
        "epoch": "epoch_metrics.csv",
        "stage2_epoch": "stage2_metrics.csv",
        "horizon": "horizon_metrics.csv",
        "router": "moe_router.csv",
        "pattern": "pattern_bank.csv",
        "pattern_usage": "pattern_usage.csv",
        "ot": "ot_diagnostics.csv",
        "gradient": "gradient_diagnostics.csv",
        "runtime": "runtime.csv",
        "evaluation": "evaluation_metrics.csv",
    }

    def __init__(self, save_dir, metadata=None):
        self.save_dir = os.path.abspath(str(save_dir))
        os.makedirs(self.save_dir, exist_ok=True)
        self.started_at = time.time()
        self._headers = {}
        payload = {
            "run_directory": self.save_dir,
            "started_at": datetime.now().astimezone().isoformat(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "metadata": metadata or {},
            "diagnostic_schema_version": 1,
        }
        with open(os.path.join(self.save_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    def _append(self, kind, row):
        filename = self.FILES[kind]
        path = os.path.join(self.save_dir, filename)
        clean = {str(k): _plain(v) for k, v in row.items()}
        fields = list(clean.keys())
        previous = self._headers.get(kind)
        if previous is None:
            if os.path.exists(path) and os.path.getsize(path):
                with open(path, "r", newline="", encoding="utf-8") as f:
                    previous = next(csv.reader(f), fields)
            else:
                previous = fields
            self._headers[kind] = previous
        # Keep a stable schema. New fields are represented in a companion JSONL
        # file rather than silently shifting columns in an existing CSV.
        if fields != previous:
            extra_path = os.path.join(self.save_dir, filename.replace(".csv", ".jsonl"))
            with open(extra_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(clean, ensure_ascii=False, default=str) + "\n")
            return
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=previous)
            if write_header:
                writer.writeheader()
            writer.writerow(clean)
            f.flush()

    def write(self, kind, row):
        if kind not in self.FILES:
            raise ValueError("unknown diagnostics stream: " + str(kind))
        self._append(kind, row)

    def log_exception(self, exc_type, exc_value, exc_tb):
        path = os.path.join(self.save_dir, "exception_traceback.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n===== " + datetime.now().astimezone().isoformat() + " =====\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
            f.flush()

    def install_exception_hook(self):
        previous_hook = sys.excepthook
        logger = self
        def _hook(exc_type, exc_value, exc_tb):
            logger.log_exception(exc_type, exc_value, exc_tb)
            previous_hook(exc_type, exc_value, exc_tb)
        sys.excepthook = _hook

    def update_metadata(self, metadata):
        path = os.path.join(self.save_dir, "run_metadata.json")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload.setdefault("metadata", {}).update(metadata or {})
        payload["updated_at"] = datetime.now().astimezone().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    def log_pattern_bank(self, pattern_bank, scales, city_names=None):
        bank = torch.as_tensor(pattern_bank).detach().float().cpu()
        for scale_index in range(bank.shape[0]):
            values = bank[scale_index]
            self.write("pattern", {
                "stage": "initial_bank",
                "scale_index": scale_index,
                "scale_steps": scales[scale_index],
                "prototype_count": values.shape[0],
                "descriptor_bins": values.shape[-1],
                "prototype_mean": values.mean(),
                "prototype_std": values.std(unbiased=False),
                "prototype_min": values.min(),
                "prototype_max": values.max(),
                "source_cities": city_names or [],
            })

    def log_epoch(self, row):
        self.write("epoch", row)

    def log_horizon(self, epoch, horizon_steps, minutes_per_step, mae, rmse, mse=None, mape=None):
        mae = np.asarray(mae).reshape(-1)
        rmse = np.asarray(rmse).reshape(-1)
        mse = None if mse is None else np.asarray(mse).reshape(-1)
        mape = None if mape is None else np.asarray(mape).reshape(-1)
        for index in range(min(len(mae), len(rmse))):
            row = {
                "epoch": epoch,
                "horizon_index": index,
                "horizon_step": index + 1,
                "horizon_minutes": (index + 1) * minutes_per_step,
                "mae": mae[index],
                "rmse": rmse[index],
            }
            if mse is not None and index < len(mse):
                row["mse"] = mse[index]
            if mape is not None and index < len(mape):
                row["mape"] = mape[index]
            self.write("horizon", row)

    def log_evaluation(self, phase, epoch, horizon_index, minutes_per_step, mae, rmse, mse, mape, sample_count):
        self.write("evaluation", {
            "phase": phase,
            "epoch": epoch,
            "horizon_index": horizon_index,
            "horizon_step": horizon_index + 1,
            "horizon_minutes": (horizon_index + 1) * minutes_per_step,
            "mae": mae,
            "rmse": rmse,
            "mse": mse,
            "mape": mape,
            "sample_count": sample_count,
        })

    def log_alignment(self, epoch, source_index, diagnostics, anchor_steps, minutes_per_step):
        if not diagnostics:
            return
        router = diagnostics.get("router_weights")
        expert_costs = diagnostics.get("expert_cost_means")
        horizon_losses = diagnostics.get("horizon_losses")
        row_errors = diagnostics.get("row_marginal_errors")
        col_errors = diagnostics.get("col_marginal_errors")
        pair_std = diagnostics.get("router_pair_std")
        top1 = diagnostics.get("router_top1_fraction")
        for h_index, step in enumerate(anchor_steps):
            row = {
                "epoch": epoch,
                "source_index": source_index,
                "horizon_index": h_index,
                "horizon_step": step,
                "horizon_minutes": step * minutes_per_step,
                "conditional_ot_loss": horizon_losses[h_index] if horizon_losses is not None else None,
                "router_entropy": diagnostics.get("router_entropy"),
                "plan_diversity": diagnostics.get("plan_diversity"),
                "max_row_marginal_error": row_errors[h_index] if row_errors is not None else diagnostics.get("max_row_marginal_error"),
                "max_col_marginal_error": col_errors[h_index] if col_errors is not None else diagnostics.get("max_col_marginal_error"),
                "router_weights": router[h_index] if router is not None else None,
                "router_pair_std": pair_std[h_index] if pair_std is not None else None,
                "router_top1_fraction": top1[h_index] if top1 is not None else None,
                "expert_cost_means": expert_costs,
                "router_smoothness_loss": diagnostics.get("router_smoothness_loss"),
                "film_smoothness": diagnostics.get("smoothness"),
                "sampled_source": diagnostics.get("sampled_source"),
                "sampled_target": diagnostics.get("sampled_target"),
            }
            self.write("router", row)
            self.write("ot", row)
        usage = diagnostics.get("pattern_usage")
        entropy = diagnostics.get("pattern_entropy")
        top1_usage = diagnostics.get("pattern_top1_fraction")
        scale_costs = diagnostics.get("per_scale_cost_means")
        if usage is not None:
            for scale_index in range(len(usage)):
                self.write("pattern_usage", {
                    "stage": "training_query",
                    "epoch": epoch,
                    "source_index": source_index,
                    "scale_index": scale_index,
                    "scale_steps": diagnostics.get("scale_steps", [])[scale_index]
                    if diagnostics.get("scale_steps") else None,
                    "assignment_entropy": entropy[scale_index] if entropy is not None else None,
                    "top1_fraction": top1_usage[scale_index] if top1_usage is not None else None,
                    "prototype_usage": usage[scale_index],
                    "mean_pairwise_cost": scale_costs[scale_index] if scale_costs is not None else None,
                })

    def log_gradient(self, epoch, module_name, grad_norm, parameter_norm, grad_none_fraction,
                     nonfinite_grad_count, parameter_norm_after=None, stage="stage1"):
        self.write("gradient", {
            "stage": stage,
            "epoch": epoch,
            "module": module_name,
            "gradient_norm": grad_norm,
            "parameter_norm_before": parameter_norm,
            "parameter_norm_after": parameter_norm_after,
            "gradient_none_fraction": grad_none_fraction,
            "nonfinite_gradient_count": nonfinite_grad_count,
        })

    def log_runtime(self, epoch, elapsed_seconds, optimizer_lr, device, stage="stage1"):
        cuda_allocated = torch.cuda.memory_allocated(device) if torch.cuda.is_available() and str(device).startswith("cuda") else 0
        cuda_reserved = torch.cuda.memory_reserved(device) if torch.cuda.is_available() and str(device).startswith("cuda") else 0
        self.write("runtime", {
            "stage": stage,
            "epoch": epoch,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_total_seconds": time.time() - self.started_at,
            "learning_rate": optimizer_lr,
            "device": str(device),
            "cuda_allocated_bytes": cuda_allocated,
            "cuda_reserved_bytes": cuda_reserved,
        })
