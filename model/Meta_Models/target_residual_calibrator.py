"""Horizon-wise robust residual calibration for PMME Stage-II.

The frozen Stage-I forecaster is corrected directly in the same physical
output space as its predictions. A robust per-node/per-horizon bias is
estimated on an early support segment, bounded by target scale, and gated
using a later, temporally separated support-calibration segment.
"""
import torch
import torch.nn as nn


class HorizonWiseResidualCalibrator(nn.Module):
    def __init__(
        self,
        pred_len: int = 36,
        horizon_bands: int = 3,
        max_correction_ratio: float = 0.25,
        purge_windows: int = 3,
        min_calibration_windows: int = 4,
        gate_step: float = 0.05,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.horizon_bands = int(horizon_bands)
        self.max_correction_ratio = float(max_correction_ratio)
        self.purge_windows = int(purge_windows)
        self.min_calibration_windows = int(min_calibration_windows)
        self.gate_step = float(gate_step)
        if self.pred_len % self.horizon_bands != 0:
            raise ValueError("pred_len must be divisible by horizon_bands")
        if not (0.0 <= self.max_correction_ratio <= 1.0):
            raise ValueError("max_correction_ratio must be in [0, 1]")
        if not (0.0 < self.gate_step <= 1.0):
            raise ValueError("gate_step must be in (0, 1]")

        self.register_buffer("node_bias", torch.empty(0, self.pred_len))
        self.register_buffer("horizon_gate", torch.zeros(self.horizon_bands))
        self.register_buffer("node_scale", torch.empty(0))
        self.register_buffer("_is_fitted", torch.tensor(False, dtype=torch.bool))
        self.fit_summary = {}

    @property
    def is_fitted(self):
        return bool(self._is_fitted.item())

    def fit_from_residuals(self, residual_targets, target_values):
        """Fit robust bias on early support windows and gate it on later windows.

        Args:
            residual_targets: [time_windows, nodes, pred_len], target - base.
            target_values: [time_windows, nodes, pred_len], physical targets.
        """
        if residual_targets.ndim != 3 or target_values.ndim != 3:
            raise ValueError("Residuals and targets must have shape [T, N, H].")
        if residual_targets.shape != target_values.shape:
            raise ValueError("Residual and target tensor shapes must match.")
        if residual_targets.shape[-1] != self.pred_len:
            raise ValueError(
                f"Expected horizon {self.pred_len}, got {residual_targets.shape[-1]}"
            )

        residual = residual_targets.detach().to(device="cpu", dtype=torch.float64)
        target = target_values.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(residual).all() or not torch.isfinite(target).all():
            raise ValueError("Non-finite target support residuals/values.")

        n_windows, n_nodes, _ = residual.shape
        fit_end = max(1, int(n_windows * 0.65))
        calibration_start = fit_end + self.purge_windows
        calibration = residual[calibration_start:]

        # Robust node-wise, horizon-wise residual center.
        bias = residual[:fit_end].median(dim=0).values
        node_scale = target[:fit_end].std(dim=(0, 2), unbiased=False).clamp_min(1e-3)
        correction_limit = self.max_correction_ratio * node_scale[:, None]
        bounded_bias = torch.maximum(
            torch.minimum(bias, correction_limit), -correction_limit
        )

        gates = torch.zeros(self.horizon_bands, dtype=torch.float64)
        base_cal_mae = float("nan")
        corrected_cal_mae = float("nan")
        if calibration.shape[0] >= self.min_calibration_windows:
            base_cal_mae = calibration.abs().mean().item()
            candidates = torch.arange(
                0.0, 1.0 + self.gate_step / 2, self.gate_step,
                dtype=torch.float64,
            ).clamp_max(1.0)
            band_width = self.pred_len // self.horizon_bands
            for band in range(self.horizon_bands):
                start = band * band_width
                end = start + band_width
                errors = []
                for gate in candidates:
                    estimate = gate * bounded_bias[:, start:end]
                    errors.append(
                        (calibration[:, :, start:end] - estimate.unsqueeze(0))
                        .abs().mean()
                    )
                best_index = int(torch.stack(errors).argmin().item())
                gates[band] = candidates[best_index]

            corrected = calibration.clone()
            for band in range(self.horizon_bands):
                start = band * band_width
                end = start + band_width
                corrected[:, :, start:end] -= (
                    gates[band] * bounded_bias[:, start:end]
                ).unsqueeze(0)
            corrected_cal_mae = corrected.abs().mean().item()

        device = self.node_bias.device
        dtype = self.node_bias.dtype if self.node_bias.numel() else torch.float32
        self.node_bias = bounded_bias.to(device=device, dtype=dtype)
        self.horizon_gate.copy_(gates.to(device=self.horizon_gate.device,
                                         dtype=self.horizon_gate.dtype))
        self.node_scale = node_scale.to(device=device, dtype=dtype)
        self._is_fitted.fill_(True)

        band_width = self.pred_len // self.horizon_bands
        applied_gate = torch.repeat_interleave(self.horizon_gate, band_width)
        final_correction = self.node_bias * applied_gate.unsqueeze(0)
        self.fit_summary = {
            "support_windows": int(n_windows),
            "fit_windows": int(fit_end),
            "purged_windows": int(min(self.purge_windows, max(0, n_windows - fit_end))),
            "calibration_windows": int(calibration.shape[0]),
            "nodes": int(n_nodes),
            "pred_len": int(self.pred_len),
            "horizon_gates": [float(v) for v in self.horizon_gate.detach().cpu()],
            "max_correction_ratio": float(self.max_correction_ratio),
            "base_calibration_mae": float(base_cal_mae),
            "corrected_calibration_mae": float(corrected_cal_mae),
            "correction_abs_max": float(final_correction.abs().max().item()),
            "correction_abs_mean": float(final_correction.abs().mean().item()),
            "median_node_scale": float(node_scale.median().item()),
        }
        return dict(self.fit_summary)

    def forward(self, base_prediction):
        if not self.is_fitted:
            raise RuntimeError("HorizonWiseResidualCalibrator must be fitted first.")
        if base_prediction.ndim != 3:
            raise ValueError("base_prediction must have shape [B, N, H].")
        if base_prediction.shape[1] != self.node_bias.shape[0]:
            raise ValueError(
                f"Node count mismatch: fitted {self.node_bias.shape[0]}, "
                f"received {base_prediction.shape[1]}"
            )
        if base_prediction.shape[-1] != self.pred_len:
            raise ValueError("Prediction horizon does not match fitted calibrator.")

        band_width = self.pred_len // self.horizon_bands
        gates = torch.repeat_interleave(self.horizon_gate, band_width)
        correction = self.node_bias * gates.unsqueeze(0)
        return correction.to(
            device=base_prediction.device, dtype=base_prediction.dtype
        ).unsqueeze(0).expand(base_prediction.shape[0], -1, -1)
