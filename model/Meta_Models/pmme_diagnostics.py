"""Read-only audit of target samples and DVGP/Sinkhorn weights in official PMME Stage-I.

The audit stores detached CPU values only. It does not change optimization,
sample selection, the OT objective, or the forecasting model.
"""
import json
from pathlib import Path

import numpy as np
import torch


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _float(value):
    if value is None:
        return None
    x = _numpy(value)
    return float(np.asarray(x).reshape(-1)[0])


def _corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) < 1e-12 or np.std(y[valid]) < 1e-12:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


class TargetTransferAudit:
    """Log target-node/window records only at user-selected iterations."""

    def __init__(self, args):
        self.every = int(getattr(args, "diagnostic_every", 0))
        self.source_cap = max(1, int(getattr(args, "diagnostic_source_cap", 256)))
        self.daepoch = int(getattr(args, "DAepoch", 50000))
        self.folder = Path(args.save_dir) / "diagnostics"
        self.target = None
        if self.every > 0:
            self.folder.mkdir(parents=True, exist_ok=True)
            with (self.folder / "audit_config.json").open("w", encoding="utf-8") as fp:
                json.dump({
                    "diagnostic_every": self.every,
                    "source_cap": self.source_cap,
                    "daepoch": self.daepoch,
                    "seed": int(args.seed),
                    "target": str(args.test_dataset),
                    "definition": "Raw GP probability, thresholded beta and prediction "
                                  "errors are logged separately. A low GP weight is "
                                  "not proof of negative transfer.",
                    "mode_note": "future_falling/stable/rising are speed-trend proxies, "
                                 "NOT verified congestion classes",
                    "normalization_warning": "Upstream datasets.py computes per-city "
                                             "statistics before target time slicing.",
                }, fp, indent=2, ensure_ascii=False)

    def active(self, epoch):
        return self.every > 0 and (epoch % self.every == 0 or epoch == self.daepoch)

    def capture_target(self, data, train_pred, eval_pred, truth):
        """Store one copy of 46-window target diagnostics before optimizer update."""
        y = _numpy(truth).astype(np.float32, copy=False)
        pred_train = _numpy(train_pred).astype(np.float32, copy=False)
        pred_eval = _numpy(eval_pred).astype(np.float32, copy=False)
        x = _numpy(data.x).astype(np.float32, copy=False)
        mean = float(np.asarray(data.means).reshape(-1)[0])
        std = float(np.asarray(data.stds).reshape(-1)[0])
        B, N, H = y.shape
        widx = _numpy(data.window_indices).astype(np.int64).reshape(-1)
        assert widx.size == B
        err_eval = np.abs(pred_eval - y)
        err_train = np.abs(pred_train - y)
        hist = x[..., 0] * std + mean
        recent = min(12, hist.shape[-1] - 1)
        future_delta = y[..., -1] - y[..., 0]
        threshold = max(0.1, float(y.std()) * 0.5)
        mode = np.where(future_delta < -threshold, -1,
                        np.where(future_delta > threshold, 1, 0)).astype(np.int8)
        self.target = {
            "N": N,
            "B": B,
            "window_idx": np.broadcast_to(widx[:, None], (B, N)).copy(),
            "node_idx": np.broadcast_to(np.arange(N)[None, :], (B, N)).copy(),
            "future_delta": future_delta,
            "future_std": y.std(axis=-1),
            "history_last": hist[..., -1],
            "history_recent_change": hist[..., -1] - hist[..., -1 - recent],
            "eval_mae": err_eval.mean(axis=-1),
            "train_mae": err_train.mean(axis=-1),
            "eval_mae_10min": err_eval[..., 1],
            "eval_mae_60min": err_eval[..., 11],
            "eval_mae_120min": err_eval[..., 23],
            "eval_mae_180min": err_eval[..., 35],
            "mode": mode,
            "trend_threshold": threshold,
        }

    @staticmethod
    def _nearest_source(source_rep, target_rep, source_future, target_future, cap):
        """Deterministic approximate support: nearest of <=cap source node-windows.

        This uses *training labels* for diagnosis, never a deployable predictor.
        The subset is evenly spaced rather than randomly sampled.
        """
        with torch.no_grad():
            ns = source_rep.shape[0]
            select = torch.linspace(0, ns - 1, min(cap, ns),
                                    device=source_rep.device).round().long()
            src = torch.cat((source_rep[select], source_future[select]), dim=-1).detach()
            tgt = torch.cat((target_rep, target_future), dim=-1).detach()
            distances, closest = [], []
            for part in tgt.split(512, dim=0):
                d, idx = torch.cdist(part, src, p=2).min(dim=1)
                distances.append(d / (src.shape[-1] ** 0.5))
                closest.append(select[idx])
            return (torch.cat(distances).cpu().numpy(),
                    torch.cat(closest).cpu().numpy())

    def capture_pair(self, epoch, index, source_data, source_rep, target_rep,
                     source_future, target_future, gp_source, gp_target, beta,
                     ot_active, ot_loss, source_task_loss):
        if self.target is None:
            return

        n_nodes = self.target["N"]
        split_sizes = [15, 15, self.target["B"] - 30]
        first = sum(split_sizes[:index])
        count = split_sizes[index]
        target_rows = slice(first, first + count)
        target_count = count * n_nodes
        if target_rep.shape[0] != target_count:
            raise ValueError("Diagnostic target group does not match PMME split.")

        arr = {key: value[target_rows].reshape(-1) for key, value
               in self.target.items() if isinstance(value, np.ndarray)
               and value.shape[:2] == (self.target["B"], n_nodes)}
        arr["gp_target_probability"] = (_numpy(gp_target).reshape(-1).astype(np.float32)
                                         if gp_target is not None
                                         else np.full(target_count, np.nan, np.float32))
        arr["beta"] = (_numpy(beta).reshape(-1).astype(np.float32)
                       if beta is not None
                       else np.zeros(target_count, np.float32))
        arr["accepted"] = (arr["beta"] > 0).astype(np.int8)

        distance, nearest = self._nearest_source(
            source_rep, target_rep, source_future, target_future, self.source_cap)
        arr["approx_source_distance"] = distance.astype(np.float32)
        src_n = int(source_data.x.shape[1])
        source_window_idx = _numpy(source_data.window_indices).astype(np.int64)
        arr["approx_nearest_source_window"] = source_window_idx[nearest // src_n]
        arr["approx_nearest_source_node"] = (nearest % src_n).astype(np.int32)

        source_name = str(source_data.data_name)
        file_name = "epoch_{:06d}_pair_{}_{}.npz".format(
            epoch, index, source_name.replace("/", "_"))
        np.savez_compressed(str(self.folder / file_name), **arr)

        beta_sum = float(arr["beta"].sum())
        mode_stats = {}
        for val, label in [(-1, "future_falling"), (0, "future_stable"),
                           (1, "future_rising")]:
            mask = arr["mode"] == val
            n = int(mask.sum())
            mass = float(arr["beta"][mask].sum())
            share = mass / beta_sum if beta_sum > 0 else None
            fraction = n / target_count
            mode_stats[label] = {
                "count": n, "fraction": fraction, "beta_mass_share": share,
                "mass_over_count_ratio": (share / fraction if share is not None
                                           and fraction > 0 else None),
                "eval_mae_mean": float(arr["eval_mae"][mask].mean()) if n else None,
                "mean_gp_probability": (float(np.nanmean(arr["gp_target_probability"][mask]))
                                        if n and np.isfinite(
                                            arr["gp_target_probability"][mask]).any()
                                        else None),
                "accepted_fraction": float(arr["accepted"][mask].mean()) if n else None,
                "approx_support_dist_mean": (
                    float(arr["approx_source_distance"][mask].mean()) if n else None),
            }

        beta_nonzero = arr["beta"][arr["beta"] > 0]
        effective_mass = (float(beta_sum ** 2 / np.square(arr["beta"]).sum())
                          if beta_nonzero.size else 0.0)
        payload = {
            "epoch": int(epoch), "pair": int(index),
            "source_city": source_name, "target_size": target_count,
            "source_size": int(source_rep.shape[0]),
            "target_window_indices": self.target["window_idx"][target_rows, 0].tolist(),
            "source_window_indices": source_window_idx.tolist(),
            "ot_active": bool(ot_active), "ot_loss": _float(ot_loss),
            "source_task_loss": _float(source_task_loss),
            "gp_source_probability_mean": (_float(_numpy(gp_source).mean())
                                           if gp_source is not None else None),
            "gp_target_probability_mean": (
                float(np.nanmean(arr["gp_target_probability"]))
                if np.isfinite(arr["gp_target_probability"]).any() else None),
            "gp_target_nonzero_fraction": float(arr["accepted"].mean()),
            "beta_defined": bool(beta is not None),
            "beta_sum": beta_sum,
            "beta_effective_count": effective_mass,
            "eval_mae_mean": float(arr["eval_mae"].mean()),
            "train_mae_mean": float(arr["train_mae"].mean()),
            "gp_probability_eval_mae_corr": _corr(
                arr["gp_target_probability"], arr["eval_mae"]),
            "beta_eval_mae_corr": _corr(arr["beta"], arr["eval_mae"]),
            "beta_support_dist_corr": _corr(
                arr["beta"], arr["approx_source_distance"]),
            "approx_support_distance_mean": float(arr["approx_source_distance"].mean()),
            "trend_threshold_speed_units": self.target["trend_threshold"],
            "mode_summary": mode_stats,
            "sample_file": file_name,
        }
        with (self.folder / "pair_summary.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        print("[PMME-DIAG] epoch={} source={} pair={} GP accepted={:.2%} "
              "eval_MAE={:.4f} OT={}".format(
                  epoch, source_name, index, payload["gp_target_nonzero_fraction"],
                  payload["eval_mae_mean"], payload["ot_active"]))

    def finish_epoch(self, epoch, task_loss, da_loss, target_task_loss):
        if self.target is None:
            return
        payload = {
            "epoch": int(epoch), "source_task_loss": _float(task_loss),
            "domain_alignment_loss": _float(da_loss),
            "target_task_loss_diagnostic_only": _float(target_task_loss),
            "target_eval_mae": float(self.target["eval_mae"].mean()),
            "target_train_mae": float(self.target["train_mae"].mean()),
            "target_window_count": int(self.target["B"]),
        }
        with (self.folder / "epoch_summary.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        self.target = None
