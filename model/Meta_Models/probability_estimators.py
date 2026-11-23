"""Learnable drop-in domain-probability estimator for PMME Stage-I.

The official GP is a Bernoulli *domain classifier* trained on normalized
ground-truth future vectors (36 steps): source=0, target=1.
Its source weight is P(target|source), and its target weight is
P(source|target).  Do not confuse these with pairwise transfer probabilities.

This module deliberately preserves that contract and does not alter OT.
Only Stage-I training labels are passed into this estimator.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DomainProbabilityNet(nn.Module):
    """Output P(domain=target | normalized future sequence), shape [N].

    Variants:
      mlp       : future-vector MLP (architecture/control ablation).
      multiscale: 1D convolution with kernels 3, 5 and 7.
      dynamics  : multiscale + first/second temporal differences.
      horizon   : dynamics + shared short/middle/long conditional statistics.

    The three variants after MLP are nested so an ablation changes one
    mechanism at a time.  All transformations use the same 36-step inputs
    as the official DVGP; there is no access to target test labels.
    """
    VALID_VARIANTS = ("mlp", "multiscale", "dynamics", "horizon")

    def __init__(self, horizon: int = 36, hidden: int = 32, variant: str = "mlp"):
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown probability variant: {variant}")
        if horizon < 6 or horizon % 3 != 0:
            raise ValueError("horizon must be >= 6 and divisible by 3")
        self.horizon = horizon
        self.variant = variant
        self.mlp = nn.Sequential(
            nn.Linear(horizon, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        if variant != "mlp":
            self.convs = nn.ModuleList([
                nn.Conv1d(1, 8, kernel_size=k, padding=k // 2)
                for k in (3, 5, 7)
            ])
            self.conv_projection = nn.Sequential(
                nn.Linear(24, hidden), nn.LayerNorm(hidden), nn.GELU()
            )
        if variant in ("dynamics", "horizon"):
            self.dynamic_projection = nn.Sequential(
                nn.Linear(8, hidden), nn.GELU()
            )
        if variant == "horizon":
            self.horizon_projection = nn.Sequential(
                nn.Linear(9, hidden), nn.GELU()
            )
            self.horizon_gate = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.Sigmoid()
            )
        self.output = nn.Linear(hidden, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 2 or y.size(-1) != self.horizon:
            raise ValueError(
                f"Expected [N,{self.horizon}] normalized futures, got {tuple(y.shape)}"
            )
        if not torch.isfinite(y).all():
            raise ValueError("Nonfinite sequence passed to probability estimator")
        z = self.mlp(y)
        if self.variant != "mlp":
            seq = y.unsqueeze(1)
            conv = torch.cat(
                [F.adaptive_avg_pool1d(F.gelu(layer(seq)), 1).squeeze(-1)
                 for layer in self.convs], dim=-1
            )
            z = z + self.conv_projection(conv)
        if self.variant in ("dynamics", "horizon"):
            d1 = y[:, 1:] - y[:, :-1]
            d2 = d1[:, 1:] - d1[:, :-1]
            summary = torch.stack([
                y.mean(-1), y.std(-1, unbiased=False),
                d1.mean(-1), d1.std(-1, unbiased=False),
                d1.abs().mean(-1), d1.abs().amax(-1),
                d2.abs().mean(-1), d2.square().mean(-1)
            ], dim=-1)
            z = z + self.dynamic_projection(summary)
        if self.variant == "horizon":
            thirds = y.reshape(y.size(0), 3, self.horizon // 3)
            d = thirds[:, :, 1:] - thirds[:, :, :-1]
            features = torch.cat([
                thirds.mean(-1), thirds.std(-1, unbiased=False),
                d.abs().mean(-1)
            ], dim=-1)
            h = self.horizon_projection(features)
            z = z + self.horizon_gate(torch.cat([z, h], -1)) * h
        return self.output(z).squeeze(-1)


class DomainProbabilityTrainer:
    """Own an independent classifier/optimizer for one PMME source city."""

    def __init__(
        self, horizon=36, variant="mlp", hidden=32, lr=1e-3,
        device="cpu", grad_clip=5.0,
    ):
        self.model = DomainProbabilityNet(horizon, hidden, variant).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-3
        )
        self.grad_clip = float(grad_clip)

    def update(self, source_y: torch.Tensor, target_y: torch.Tensor):
        # Domain labels preserve the official DVGP convention.
        n_source, n_target = source_y.size(0), target_y.size(0)
        if n_source == 0 or n_target == 0:
            raise ValueError("Source and target probabilities need non-empty batches")
        x = torch.cat([source_y.detach(), target_y.detach()], dim=0)
        labels = torch.cat([
            x.new_zeros(n_source), x.new_ones(n_target)
        ])
        # Fit the same probability event as BernoulliLikelihood(GP).
        self.model.train()
        logits = self.model(x)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        # Match official code's use of pre-update probabilities.
        p_target_domain = torch.sigmoid(logits.detach())
        source_weight = p_target_domain[:n_source]
        target_weight = 1.0 - p_target_domain[n_source:]
        return source_weight, target_weight, {
            "domain_bce": float(loss.detach().cpu()),
            "domain_accuracy": float(
                ((p_target_domain >= .5) == labels.bool()).float().mean().cpu()
            ),
        }
