"""Lightweight target-only low-rank residual adapter for PMME Stage-II.

The Stage-I forecasting backbone is frozen. Support-set residual trajectories
are represented by a small mixed basis (low-frequency DCT + data-driven
detail directions), while a ridge map predicts basis coefficients from
compact history and base-forecast summaries. Fitting is closed-form: no
backpropagation or epoch loop is required.
"""
import torch
import torch.nn as nn


class TargetResidualLowRankAdapter(nn.Module):
    """Closed-form target-city residual adapter with no trainable parameters."""

    def __init__(
        self,
        history_len: int = 288,
        pred_len: int = 36,
        feature_bins: int = 12,
        rank: int = 6,
        smooth_rank: int = 3,
        ridge_alpha: float = 10.0,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.feature_bins = int(feature_bins)
        self.rank = int(rank)
        self.smooth_rank = int(smooth_rank)
        self.ridge_alpha = float(ridge_alpha)
        self.feature_dim = self.feature_bins * 3 + 4

        if self.history_len % self.feature_bins != 0:
            raise ValueError("history_len must be divisible by feature_bins.")
        if not (1 <= self.smooth_rank < self.rank <= self.pred_len):
            raise ValueError(
                "Require 1 <= smooth_rank < rank <= pred_len."
            )
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive.")

        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_scale", torch.ones(self.feature_dim))
        self.register_buffer("residual_mean", torch.zeros(self.pred_len))
        self.register_buffer("residual_basis", torch.zeros(self.pred_len, self.rank))
        self.register_buffer(
            "ridge_weight", torch.zeros(self.feature_dim + 1, self.rank)
        )
        self.register_buffer("_is_fitted", torch.tensor(False, dtype=torch.bool))
        self.fit_summary = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self._is_fitted.item())

    def extract_features(
        self, history: torch.Tensor, base_prediction: torch.Tensor
    ) -> torch.Tensor:
        """Build 40 compact features per node/window (12 means, stds, slopes + 4 forecast stats)."""
        if history.ndim != 3:
            raise ValueError(
                f"history must be [B,N,L], got {tuple(history.shape)}"
            )
        if base_prediction.ndim != 3:
            raise ValueError(
                "base_prediction must be [B,N,pred_len], "
                f"got {tuple(base_prediction.shape)}"
            )
        if history.shape[-1] != self.history_len:
            raise ValueError(
                f"Expected history length {self.history_len}, got {history.shape[-1]}"
            )
        if base_prediction.shape[-1] != self.pred_len:
            raise ValueError(
                f"Expected prediction length {self.pred_len}, got {base_prediction.shape[-1]}"
            )
        if history.shape[:2] != base_prediction.shape[:2]:
            raise ValueError("History and base forecast batch/node dimensions differ.")

        segment_len = self.history_len // self.feature_bins
        segments = history.reshape(
            *history.shape[:2], self.feature_bins, segment_len
        )
        segment_mean = segments.mean(dim=-1)
        segment_std = segments.std(dim=-1, unbiased=False)
        segment_slope = (
            segments[..., -1] - segments[..., 0]
        ) / max(segment_len - 1, 1)

        first_part = base_prediction[..., : max(1, self.pred_len // 3)]
        last_part = base_prediction[..., -max(1, self.pred_len // 3):]
        forecast_summary = torch.stack(
            [
                base_prediction.mean(dim=-1),
                base_prediction.std(dim=-1, unbiased=False),
                first_part.mean(dim=-1),
                last_part.mean(dim=-1),
            ],
            dim=-1,
        )
        return torch.cat(
            [segment_mean, segment_std, segment_slope, forecast_summary],
            dim=-1,
        )

    @staticmethod
    def _low_frequency_basis(length: int, count: int) -> torch.Tensor:
        """Orthonormal low-frequency DCT-II basis."""
        positions = torch.arange(length, dtype=torch.float64) + 0.5
        frequencies = torch.arange(count, dtype=torch.float64)
        basis = torch.cos(
            torch.pi * positions[:, None] * frequencies[None, :] / length
        )
        basis[:, 0] *= (1.0 / length) ** 0.5
        if count > 1:
            basis[:, 1:] *= (2.0 / length) ** 0.5
        return basis

    def fit_from_features(
        self, features: torch.Tensor, residual_targets: torch.Tensor
    ) -> dict:
        """Fit the residual basis and ridge coefficient map using support data only."""
        if features.ndim < 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must end in {self.feature_dim}, got {tuple(features.shape)}"
            )
        if residual_targets.ndim < 2 or residual_targets.shape[-1] != self.pred_len:
            raise ValueError(
                f"residual_targets must end in {self.pred_len}, "
                f"got {tuple(residual_targets.shape)}"
            )
        if features.shape[:-1] != residual_targets.shape[:-1]:
            raise ValueError("Feature and residual sample dimensions do not match.")

        x = features.detach().reshape(-1, self.feature_dim).to(
            device="cpu", dtype=torch.float64
        )
        residual = residual_targets.detach().reshape(-1, self.pred_len).to(
            device="cpu", dtype=torch.float64
        )
        if x.shape[0] < self.rank:
            raise ValueError(
                f"Need at least {self.rank} support rows to fit rank-{self.rank} residual basis; "
                f"received {x.shape[0]}."
            )
        if not torch.isfinite(x).all() or not torch.isfinite(residual).all():
            raise ValueError("Non-finite support features or residual targets.")

        x_mean = x.mean(dim=0)
        x_scale = x.std(dim=0, unbiased=False).clamp_min(1e-4)
        x_normalized = (x - x_mean) / x_scale

        residual_mean = residual.mean(dim=0)
        centered_residual = residual - residual_mean

        smooth_basis = self._low_frequency_basis(
            self.pred_len, self.smooth_rank
        )
        smooth_coefficients = centered_residual @ smooth_basis
        detail_residual = (
            centered_residual - smooth_coefficients @ smooth_basis.T
        )

        _, singular_values, vh = torch.linalg.svd(
            detail_residual, full_matrices=False
        )
        detail_rank = self.rank - self.smooth_rank
        if vh.shape[0] < detail_rank:
            raise ValueError(
                "Insufficient support samples for the requested detail basis."
            )
        detail_basis = vh[:detail_rank].T.contiguous()
        basis = torch.cat([smooth_basis, detail_basis], dim=1)

        coefficients = centered_residual @ basis
        design = torch.cat(
            [x_normalized, torch.ones(x_normalized.shape[0], 1, dtype=x.dtype)],
            dim=1,
        )
        gram = design.T @ design
        regularizer = torch.eye(gram.shape[0], dtype=gram.dtype) * self.ridge_alpha
        regularizer[-1, -1] = 0.0  # Do not penalize the intercept.
        rhs = design.T @ coefficients
        ridge_weight = torch.linalg.solve(gram + regularizer, rhs)

        total_energy = centered_residual.square().sum().clamp_min(1e-12)
        smooth_energy = smooth_coefficients.square().sum()
        detail_energy = detail_residual.square().sum()
        captured_detail_energy = singular_values[:detail_rank].square().sum()
        captured_energy = smooth_energy + captured_detail_energy

        device = self.feature_mean.device
        self.feature_mean.copy_(x_mean.to(device=device, dtype=self.feature_mean.dtype))
        self.feature_scale.copy_(x_scale.to(device=device, dtype=self.feature_scale.dtype))
        self.residual_mean.copy_(
            residual_mean.to(device=device, dtype=self.residual_mean.dtype)
        )
        self.residual_basis.copy_(
            basis.to(device=device, dtype=self.residual_basis.dtype)
        )
        self.ridge_weight.copy_(
            ridge_weight.to(device=device, dtype=self.ridge_weight.dtype)
        )
        self._is_fitted.fill_(True)

        self.fit_summary = {
            "support_rows": int(x.shape[0]),
            "feature_dim": int(self.feature_dim),
            "rank": int(self.rank),
            "smooth_rank": int(self.smooth_rank),
            "detail_rank": int(detail_rank),
            "ridge_alpha": float(self.ridge_alpha),
            "smooth_energy_ratio": float((smooth_energy / total_energy).item()),
            "detail_energy_ratio": float((detail_energy / total_energy).item()),
            "captured_energy_ratio": float((captured_energy / total_energy).item()),
        }
        return dict(self.fit_summary)

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError(
                "TargetResidualLowRankAdapter must be fitted on target support data first."
            )
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} features, got {features.shape[-1]}"
            )
        shape = features.shape[:-1]
        x = features.reshape(-1, self.feature_dim)
        x = (x - self.feature_mean) / self.feature_scale
        design = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=-1)
        coefficients = design @ self.ridge_weight
        residual = self.residual_mean + coefficients @ self.residual_basis.T
        return residual.reshape(*shape, self.pred_len)

    def forward(
        self, history: torch.Tensor, base_prediction: torch.Tensor
    ) -> torch.Tensor:
        features = self.extract_features(history, base_prediction)
        return self.predict_from_features(features)
