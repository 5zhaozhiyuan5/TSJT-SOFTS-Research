import csv
import json
import math
import os
import platform
import threading
import time
from datetime import datetime
from pathlib import Path

import torch


def _json_default(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def tensor_summary(value, prefix):
    if not torch.is_tensor(value):
        return {}
    x = value.detach()
    if x.numel() == 0:
        return {f"{prefix}_shape": str(tuple(x.shape)), f"{prefix}_empty": 1}
    xf = x.float()
    finite = torch.isfinite(xf)
    valid = xf[finite]
    if valid.numel() == 0:
        return {
            f"{prefix}_shape": str(tuple(x.shape)),
            f"{prefix}_finite_ratio": 0.0,
            f"{prefix}_nan_count": int(torch.isnan(xf).sum().cpu()),
            f"{prefix}_inf_count": int(torch.isinf(xf).sum().cpu()),
        }
    return {
        f"{prefix}_shape": str(tuple(x.shape)),
        f"{prefix}_dtype": str(x.dtype),
        f"{prefix}_mean": float(valid.mean().cpu()),
        f"{prefix}_std": float(valid.std(unbiased=False).cpu()),
        f"{prefix}_min": float(valid.min().cpu()),
        f"{prefix}_max": float(valid.max().cpu()),
        f"{prefix}_l2": float(torch.linalg.vector_norm(valid).cpu()),
        f"{prefix}_finite_ratio": float(finite.float().mean().cpu()),
        f"{prefix}_zero_ratio": float((xf == 0).float().mean().cpu()),
        f"{prefix}_nan_count": int(torch.isnan(xf).sum().cpu()),
        f"{prefix}_inf_count": int(torch.isinf(xf).sum().cpu()),
    }


class DiagnosticRecorder:
    def __init__(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = os.getenv("STUNET_RUN_NAME", stamp)
        root = Path(os.getenv("STUNET_DIAG_DIR", "runs/stunet_diagnostics"))
        self.rank = int(os.getenv("RANK", "0"))
        self.run_dir = root / run_name / f"rank{self.rank}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.context = {"stage": "startup", "epoch": -1, "step": 0, "batch_idx": -1}
        self.batch_every = max(1, int(os.getenv("STUNET_DIAG_EVERY", "20")))
        self.attention_every = max(1, int(os.getenv("STUNET_DIAG_ATTENTION_EVERY", "100")))
        self.gradient_every = max(1, int(os.getenv("STUNET_DIAG_GRAD_EVERY", "50")))
        self.token_every = max(1, int(os.getenv("STUNET_DIAG_TOKEN_EVERY", "50")))
        self.prediction_every = max(1, int(os.getenv("STUNET_DIAG_PRED_EVERY", "20")))
        self._batch_started = None
        self._event_path = self.run_dir / "events.jsonl"
        self.write(
            "run_start",
            python=platform.python_version(),
            platform=platform.platform(),
            torch=torch.__version__,
            cuda=torch.cuda.is_available(),
            cuda_version=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            intervals={"batch": self.batch_every, "attention": self.attention_every,
                       "gradient": self.gradient_every, "token": self.token_every, "prediction": self.prediction_every},
        )

    def set_context(self, stage=None, epoch=None, step=None, batch_idx=None):
        if stage is not None:
            self.context["stage"] = str(stage)
        if epoch is not None:
            self.context["epoch"] = int(epoch)
        if step is not None:
            self.context["step"] = int(step)
        if batch_idx is not None:
            self.context["batch_idx"] = int(batch_idx)
        self._batch_started = time.perf_counter()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def due(self, kind):
        stage = str(self.context.get("stage", ""))
        if "validation" in stage or "test" in stage:
            step = max(0, int(self.context.get("batch_idx", 0)))
        else:
            step = max(0, int(self.context.get("step", 0)))
        interval = {"batch": self.batch_every, "attention": self.attention_every,
                    "gradient": self.gradient_every, "token": self.token_every, "prediction": self.prediction_every}.get(kind, self.batch_every)
        return step % interval == 0

    def write(self, event, **payload):
        row = {"time": datetime.now().isoformat(timespec="milliseconds"),
               "event": event, **self.context, **payload}
        with self._lock:
            with self._event_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

    def csv(self, filename, row):
        path = self.run_dir / filename
        safe = {}
        for key, value in row.items():
            safe[key] = json.dumps(value, ensure_ascii=False, default=_json_default) if isinstance(value, (dict, list, tuple)) else _json_default(value)
        with self._lock:
            exists = path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(safe.keys()), extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                writer.writerow(safe)

    def log_batch(self, batch):
        if not isinstance(batch, dict):
            return
        x, y, adj = batch.get("x"), batch.get("y"), batch.get("adj_matrix")
        row = {**self.context, "x_shape": str(tuple(x.shape)) if torch.is_tensor(x) else None,
               "y_shape": str(tuple(y.shape)) if torch.is_tensor(y) else None}
        x_sample = x[:min(16, x.shape[0])] if torch.is_tensor(x) and x.ndim > 0 else x
        y_sample = y[:min(16, y.shape[0])] if torch.is_tensor(y) and y.ndim > 0 else y
        row["stats_sample_size"] = int(x_sample.shape[0]) if torch.is_tensor(x_sample) and x_sample.ndim > 0 else 0
        row.update(tensor_summary(x_sample, "x"))
        row.update(tensor_summary(y_sample, "y"))
        if torch.is_tensor(x_sample) and x_sample.ndim == 4:
            for channel in range(min(3, x_sample.shape[2])):
                row.update(tensor_summary(x_sample[:, :, channel, :], f"x_channel{channel}"))
        if torch.is_tensor(adj):
            a = adj.detach()
            if a.ndim == 3:
                a = a[0]
            if a.ndim == 2 and a.shape[0] == a.shape[1]:
                af = a.float()
                nz = af.abs() > 0
                degree = nz.sum(dim=-1).float()
                row.update({
                    "adj_shape": str(tuple(adj.shape)),
                    "adj_density": float(nz.float().mean().cpu()),
                    "adj_nonzero": int(nz.sum().cpu()),
                    "adj_diag_ratio": float(torch.diagonal(nz).float().mean().cpu()),
                    "adj_symmetry_mae": float((af - af.T).abs().mean().cpu()),
                    "degree_mean": float(degree.mean().cpu()),
                    "degree_std": float(degree.std(unbiased=False).cpu()),
                    "degree_min": float(degree.min().cpu()),
                    "degree_max": float(degree.max().cpu()),
                    "isolated_node_ratio": float((degree == 0).float().mean().cpu()),
                    "adj_weight_mean": float(af[nz].mean().cpu()) if nz.any() else 0.0,
                    "adj_weight_std": float(af[nz].std(unbiased=False).cpu()) if nz.any() else 0.0,
                })
        n_id = batch.get("n_id")
        if torch.is_tensor(n_id):
            row["n_id_count"] = int(n_id.numel())
            row["n_id_unique"] = int(torch.unique(n_id).numel())
        target_mask = batch.get("target_node_mask")
        if torch.is_tensor(target_mask):
            row["target_node_count"] = int(target_mask.sum().cpu())
            row["target_node_ratio"] = float(target_mask.float().mean().cpu())
        self.csv("batch_stats.csv", row)
        self.write("batch_stats", **row)

    def log_tokens(self, node_tokens, graph_tokens, adj_matrix=None):
        row = {**self.context}
        node_sample = node_tokens[:min(16, node_tokens.shape[0])] if torch.is_tensor(node_tokens) else node_tokens
        graph_sample = graph_tokens[:min(16, graph_tokens.shape[0])] if torch.is_tensor(graph_tokens) else graph_tokens
        row.update(tensor_summary(node_sample, "node_token"))
        row.update(tensor_summary(graph_sample, "graph_token"))
        if torch.is_tensor(node_sample) and torch.is_tensor(graph_sample):
            a = node_sample.detach().float().mean(dim=(0, 1))
            b = graph_sample.detach().float().mean(dim=(0, 1))
            row["node_graph_mean_cosine"] = float(torch.nn.functional.cosine_similarity(a, b, dim=0).cpu())
            row["node_token_count"] = int(node_tokens.shape[1])
            row["graph_token_count"] = int(graph_tokens.shape[1])
        if torch.is_tensor(adj_matrix):
            adj_sample = adj_matrix[:1] if adj_matrix.ndim == 3 else adj_matrix
            row.update(tensor_summary(adj_sample, "tokenizer_adj"))
        self.csv("token_stats.csv", row)
        self.write("token_stats", **row)

    def log_attention(self, layer, kind, weights):
        if not torch.is_tensor(weights) or weights.ndim != 4:
            return
        p = weights[:min(8, weights.shape[0])].detach().float().clamp_min(1e-12)
        entropy = -(p * p.log()).sum(dim=-1)
        support = entropy.exp()
        top1 = p.max(dim=-1).values
        topk = p.topk(min(5, p.shape[-1]), dim=-1).values.sum(dim=-1)
        row = {
            **self.context, "layer": int(layer), "kind": kind, "shape": str(tuple(p.shape)),
            "entropy_mean": float(entropy.mean().cpu()),
            "entropy_std": float(entropy.std(unbiased=False).cpu()),
            "normalized_entropy": float((entropy / max(math.log(max(2, p.shape[-1])), 1e-12)).mean().cpu()),
            "effective_support": float(support.mean().cpu()),
            "top1_mass": float(top1.mean().cpu()), "top5_mass": float(topk.mean().cpu()),
            "weight_max": float(p.max().cpu()), "weight_min": float(p.min().cpu()),
            "head_entropy": [float(v.cpu()) for v in entropy.mean(dim=(0, 2, 3))],
            "diagonal_mass": None,
        }
        if kind == "aggregate" and p.shape[-2] == p.shape[-1]:
            row["diagonal_mass"] = float(p.diagonal(dim1=-2, dim2=-1).mean().cpu())
        self.csv("attention_stats.csv", row)
        self.write("attention_stats", **row)

    def log_predictions(self, y_pred, y_true, batch=None, stage=None):
        if not torch.is_tensor(y_pred) or not torch.is_tensor(y_true):
            return
        pred, true = y_pred.detach().float(), y_true.detach().float()
        if pred.shape != true.shape or pred.ndim != 3:
            self.write("prediction_shape_warning", pred_shape=str(tuple(pred.shape)), true_shape=str(tuple(true.shape)))
            return
        finite_pred, finite_true = torch.isfinite(pred), torch.isfinite(true)
        valid = (true != 0) & finite_pred & finite_true
        err = torch.zeros_like(pred)
        err[valid] = pred[valid] - true[valid]
        abs_err = err.abs()
        valid_count = int(valid.sum().cpu())
        overall = {
            **self.context, "stage": stage or self.context.get("stage"),
            "valid_count": valid_count, "valid_ratio": float(valid.float().mean().cpu()),
            "zero_target_ratio": float((true == 0).float().mean().cpu()),
            "nonfinite_prediction_count": int((~finite_pred).sum().cpu()),
            "nonfinite_target_count": int((~finite_true).sum().cpu()),
        }
        if valid_count:
            e, ae, t, p = err[valid], abs_err[valid], true[valid], pred[valid]
            overall.update({
                "mae": float(ae.mean().cpu()), "rmse": float(e.square().mean().sqrt().cpu()),
                "mape": float((ae / t.abs().clamp_min(1e-6)).mean().mul(100).cpu()),
                "bias": float(e.mean().cpu()), "true_mean": float(t.mean().cpu()),
                "pred_mean": float(p.mean().cpu()),
                "abs_error_p90": float(torch.quantile(ae, 0.90).cpu()),
                "abs_error_p99": float(torch.quantile(ae, 0.99).cpu()),
            })
        else:
            overall.update({key: None for key in ("mae", "rmse", "mape", "bias", "true_mean", "pred_mean",
                                                   "abs_error_p90", "abs_error_p99")})
        self.write("prediction_metrics", **overall)
        self.csv("prediction_metrics.csv", overall)
        for h in range(pred.shape[1]):
            m = valid[:, h, :]
            count = int(m.sum().cpu())
            if count == 0:
                continue
            e, t, p = err[:, h, :][m], true[:, h, :][m], pred[:, h, :][m]
            ae = e.abs()
            self.csv("horizon_metrics.csv", {
                **self.context, "stage": stage or self.context.get("stage"), "horizon": h + 1,
                "valid_count": count, "mae": float(ae.mean().cpu()),
                "rmse": float(e.square().mean().sqrt().cpu()),
                "mape": float((ae / t.abs().clamp_min(1e-6)).mean().mul(100).cpu()),
                "bias": float(e.mean().cpu()), "true_mean": float(t.mean().cpu()),
                "pred_mean": float(p.mean().cpu()), "true_std": float(t.std(unbiased=False).cpu()),
                "abs_error_p90": float(torch.quantile(ae, 0.90).cpu()),
            })
        values = true[valid]
        if values.numel() > 2:
            q50, q90 = torch.quantile(values, torch.tensor([0.50, 0.90], device=values.device))
            bins = {"low": valid & (true <= q50), "middle": valid & (true > q50) & (true <= q90),
                    "high": valid & (true > q90)}
            for name, mask in bins.items():
                if mask.any():
                    e = err[mask]
                    self.csv("regime_metrics.csv", {
                        **self.context, "stage": stage or self.context.get("stage"), "regime": name,
                        "count": int(mask.sum().cpu()), "threshold_q50": float(q50.cpu()),
                        "threshold_q90": float(q90.cpu()), "mae": float(e.abs().mean().cpu()),
                        "rmse": float(e.square().mean().sqrt().cpu()), "bias": float(e.mean().cpu()),
                    })
        node_count = pred.shape[-1]
        node_ids = torch.arange(node_count, device=pred.device)
        if isinstance(batch, dict):
            n_id, target_mask = batch.get("n_id"), batch.get("target_node_mask")
            if torch.is_tensor(n_id):
                n_id = n_id.to(pred.device).reshape(-1)
                if n_id.numel() == node_count:
                    node_ids = n_id
                elif torch.is_tensor(target_mask):
                    tm = target_mask.to(pred.device).reshape(-1).bool()
                    if tm.numel() == n_id.numel() and int(tm.sum()) == node_count:
                        node_ids = n_id[tm]
        counts = valid.sum(dim=(0, 1))
        node_mae = abs_err.sum(dim=(0, 1)) / counts.clamp_min(1)
        node_mae = node_mae.masked_fill(counts == 0, -torch.inf)
        available = int((counts > 0).sum().cpu())
        if available:
            vals, idx = torch.topk(node_mae, k=min(10, available))
            for value, local_idx in zip(vals, idx):
                j = int(local_idx.cpu())
                node_valid = valid[:, :, j]
                self.csv("node_metrics.csv", {
                    **self.context, "stage": stage or self.context.get("stage"), "local_node": j,
                    "global_node": int(node_ids[j].cpu()) if j < node_ids.numel() else j,
                    "mae": float(value.cpu()), "valid_count": int(counts[j].cpu()),
                    "true_mean": float(true[:, :, j][node_valid].mean().cpu()),
                })
        if isinstance(batch, dict):
            target_mask = batch.get("target_node_mask")
            if torch.is_tensor(target_mask) and target_mask.numel() == node_count:
                tm = target_mask.to(pred.device).bool().view(1, 1, -1).expand_as(valid)
                for name, subset in (("target", valid & tm), ("context", valid & ~tm)):
                    if subset.any():
                        e = err[subset]
                        self.write("target_context_metrics", subset=name, count=int(subset.sum().cpu()),
                                   mae=float(e.abs().mean().cpu()), rmse=float(e.square().mean().sqrt().cpu()),
                                   bias=float(e.mean().cpu()))

    def log_gradients(self, module):
        if not self.due("gradient"):
            return
        groups, param_rows = {}, []
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if "node_encoder" in name or "temporal" in name:
                group = "temporal_tokenizer"
            elif "graph_encoder" in name:
                group = "spatial_tokenizer"
            elif "graph_generator" in name:
                group = "graph_generator"
            elif "backbone" in name and "tokenizer" not in name:
                group = "query_aggregate_attention"
            elif "head" in name or "forecaster" in name:
                group = "prediction_head"
            else:
                group = "other"
            item = groups.setdefault(group, {"params": 0, "missing": 0, "zero": 0,
                                             "nonfinite": 0, "grad_sq": 0.0, "param_sq": 0.0})
            item["params"] += 1
            pnorm = float(torch.linalg.vector_norm(parameter.detach().float()).cpu())
            item["param_sq"] += pnorm * pnorm
            grad = parameter.grad
            if grad is None:
                item["missing"] += 1
                continue
            g = grad.detach().float()
            finite = torch.isfinite(g).all()
            if not finite:
                item["nonfinite"] += 1
            if torch.count_nonzero(g) == 0:
                item["zero"] += 1
            gnorm = float(torch.linalg.vector_norm(torch.nan_to_num(g)).cpu())
            item["grad_sq"] += gnorm * gnorm
            param_rows.append({**self.context, "group": group, "parameter": name,
                               "param_norm": pnorm, "grad_norm": gnorm,
                               "grad_param_ratio": gnorm / max(pnorm, 1e-12),
                               "grad_finite": int(finite.cpu())})
        for group, item in groups.items():
            grad_norm, param_norm = item["grad_sq"] ** 0.5, item["param_sq"] ** 0.5
            self.csv("gradient_stats.csv", {
                **self.context, "group": group, "parameter_tensors": item["params"],
                "missing_grad": item["missing"], "zero_grad": item["zero"],
                "nonfinite_grad": item["nonfinite"], "grad_norm": grad_norm,
                "param_norm": param_norm, "grad_param_ratio": grad_norm / max(param_norm, 1e-12),
            })
        for row in param_rows:
            self.csv("parameter_gradients.csv", row)
        self.write("gradient_summary", groups=groups)

    def log_step_end(self, output, batch=None):
        loss = None
        if torch.is_tensor(output) and output.numel() == 1:
            loss = float(output.detach().cpu())
        elif isinstance(output, dict) and torch.is_tensor(output.get("loss")):
            loss = float(output["loss"].detach().cpu())
        elapsed = time.perf_counter() - self._batch_started if self._batch_started else None
        x = batch.get("x") if isinstance(batch, dict) else None
        batch_size = int(x.shape[0]) if torch.is_tensor(x) and x.ndim > 0 else None
        node_count = int(x.shape[-1]) if torch.is_tensor(x) and x.ndim >= 1 else None
        gpu_allocated = gpu_reserved = peak_allocated = peak_reserved = None
        if torch.cuda.is_available():
            gpu_allocated = float(torch.cuda.memory_allocated() / (1024 ** 2))
            gpu_reserved = float(torch.cuda.memory_reserved() / (1024 ** 2))
            peak_allocated = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            peak_reserved = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
        row = {
            **self.context, "loss": loss, "batch_seconds": elapsed,
            "batch_size": batch_size, "node_count": node_count,
            "samples_per_second": batch_size / elapsed if batch_size and elapsed else None,
            "node_timesteps_per_second": batch_size * node_count / elapsed if batch_size and node_count and elapsed else None,
            "gpu_allocated_mb": gpu_allocated, "gpu_reserved_mb": gpu_reserved,
            "gpu_peak_allocated_mb": peak_allocated, "gpu_peak_reserved_mb": peak_reserved,
        }
        self.csv("step_metrics.csv", row)
        self.write("batch_end", **row)

    def log_model(self, module):
        total = trainable = frozen = 0
        groups = {}
        for name, parameter in module.named_parameters():
            count = parameter.numel()
            total += count
            if parameter.requires_grad:
                trainable += count
            else:
                frozen += count
            if "graph_encoder" in name:
                group = "spatial_tokenizer"
            elif "node_encoder" in name:
                group = "temporal_tokenizer"
            elif "graph_generator" in name:
                group = "graph_generator"
            elif "backbone" in name and "tokenizer" not in name:
                group = "query_aggregate_attention"
            elif "head" in name or "forecaster" in name:
                group = "prediction_head"
            else:
                group = "other"
            item = groups.setdefault(group, {"total": 0, "trainable": 0, "frozen": 0})
            item["total"] += count
            item["trainable" if parameter.requires_grad else "frozen"] += count
        self.write("model_parameters", total=total, trainable=trainable, frozen=frozen, groups=groups)

    def log_epoch(self, module, stage):
        metrics, lrs = {}, []
        try:
            for key, value in module.trainer.callback_metrics.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    metrics[str(key)] = float(value.detach().cpu())
                elif isinstance(value, (float, int)):
                    metrics[str(key)] = float(value)
        except Exception:
            pass
        try:
            for optimizer in module.trainer.optimizers:
                lrs.extend([float(group["lr"]) for group in optimizer.param_groups])
        except Exception:
            pass
        row = {**self.context, "stage": stage, "learning_rates": lrs, "metrics": metrics}
        self.csv("epoch_summary.csv", row)
        self.write("epoch_summary", **row)


RECORDER = DiagnosticRecorder()
