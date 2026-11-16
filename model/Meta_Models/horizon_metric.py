"""Horizon-conditioned low-rank metric features for PMME Stage-I.

The module consumes SOFTS/iTransformer window representations [B, N, 128].
It creates a compact relation embedding by applying shared FiLM-style
conditioning and horizon-generated low-rank projections at a small set of
physical-time anchors. Concatenating the anchor projections means the OT
solver still receives one feature vector and computes one transport plan.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizonConditionedLowRankMetric(nn.Module):
    """Build a compact, horizon-conditioned metric embedding from backbone rep."""

    def __init__(
        self,
        input_dim: int = 128,
        rank: int = 8,
        basis_dim: int = 8,
        anchor_minutes=(10, 60, 120, 180),
        max_minutes: float = 180.0,
        condition_hidden: int = 64,
    ):
        super().__init__()
        anchors = tuple(float(v) for v in anchor_minutes)
        if input_dim <= 0 or rank <= 0 or basis_dim <= 0:
            raise ValueError("input_dim, rank and basis_dim must be positive")
        if not anchors or any(v <= 0 for v in anchors):
            raise ValueError("anchor_minutes must contain positive durations")
        if tuple(sorted(anchors)) != anchors:
            raise ValueError("anchor_minutes must be sorted in ascending order")
        if max_minutes < max(anchors):
            raise ValueError("max_minutes must be >= the largest anchor")

        self.input_dim = int(input_dim)
        self.rank = int(rank)
        self.basis_dim = int(basis_dim)
        self.anchor_minutes = anchors
        self.max_minutes = float(max_minutes)
        self.output_dim = self.rank * len(self.anchor_minutes)

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.conditioner = nn.Sequential(
            nn.Linear(1, condition_hidden),
            nn.GELU(),
            nn.Linear(condition_hidden, condition_hidden),
            nn.GELU(),
            nn.Linear(condition_hidden, 2 * self.input_dim + self.basis_dim),
        )

        # P_h = P_0 + (A * a_h) B^T. The low-rank bases are shared by all horizons.
        self.base_projection = nn.Parameter(torch.empty(self.rank, self.input_dim))
        self.left_basis = nn.Parameter(torch.empty(self.rank, self.basis_dim))
        self.right_basis = nn.Parameter(torch.empty(self.input_dim, self.basis_dim))
        nn.init.orthogonal_(self.base_projection)
        nn.init.normal_(self.left_basis, mean=0.0, std=0.01)
        nn.init.normal_(self.right_basis, mean=0.0, std=0.01)

    def _condition_for_horizon(self, minutes: float):
        # Log-time encoding makes the conditioning less sensitive to the scale
        # difference between short and long horizons.
        scaled_time = math.log1p(minutes) / math.log1p(self.max_minutes)
        time_input = self.base_projection.new_tensor([[scaled_time]])
        condition = self.conditioner(time_input).squeeze(0)

        gamma, beta, coefficients = torch.split(
            condition, [self.input_dim, self.input_dim, self.basis_dim], dim=-1
        )
        # Bounded modulation keeps early training stable.
        gamma = 0.5 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)

        projection = self.base_projection + (
            self.left_basis * coefficients.unsqueeze(0)
        ) @ self.right_basis.transpose(0, 1)
        # Row normalization controls the scale of the learned ground metric.
        projection = F.normalize(projection, p=2, dim=-1, eps=1e-6)
        return gamma, beta, coefficients, projection

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        """Return [..., rank * number_of_anchors] relation features."""
        if rep.ndim < 2 or rep.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected rep with last dimension {self.input_dim}; "
                f"received shape {tuple(rep.shape)}"
            )

        normalized = self.input_norm(rep)
        anchor_features = []
        anchor_scale = 1.0 / math.sqrt(len(self.anchor_minutes))
        for minutes in self.anchor_minutes:
            gamma, beta, _coefficients, projection = self._condition_for_horizon(minutes)
            conditioned = F.gelu((1.0 + gamma) * normalized + beta)
            projected = F.linear(conditioned, projection)
            anchor_features.append(projected * anchor_scale)
        return torch.cat(anchor_features, dim=-1)

    def smoothness_loss(self) -> torch.Tensor:
        """Mildly discourage abrupt parameter changes between adjacent anchors."""
        if len(self.anchor_minutes) < 2:
            return self.base_projection.new_zeros(())

        gamma_values = []
        beta_values = []
        coefficient_values = []
        for minutes in self.anchor_minutes:
            gamma, beta, coefficients, _projection = self._condition_for_horizon(minutes)
            gamma_values.append(gamma)
            beta_values.append(beta)
            coefficient_values.append(coefficients)

        def adjacent_mean(values):
            stacked = torch.stack(values, dim=0)
            return (stacked[1:] - stacked[:-1]).pow(2).mean()

        return (
            adjacent_mean(gamma_values)
            + adjacent_mean(beta_values)
            + adjacent_mean(coefficient_values)
        )
