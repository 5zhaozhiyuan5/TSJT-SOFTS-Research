"""Continuous horizon-conditioned transfer alignment for PMME Stage-I.

The forecasting backbone is intentionally external to this module. Inputs are
its existing window-level representations [samples, 128] and training-only
future-trajectory descriptors. The module creates a shared base transport
plan and a small number of horizon-conditioned plans by KL-reweighting that
base plan; it never creates one Sinkhorn problem per prediction step.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_future_change_descriptors(
    future: torch.Tensor,
    last_history: torch.Tensor,
    anchor_steps: Sequence[int] = (2, 12, 24, 36),
) -> torch.Tensor:
    """Summarize future change relative to the last normalized history value.

    Args:
        future: [B, N, H], RevIN-normalized future targets (training only).
        last_history: [B, N], last history value under the same RevIN transform.
        anchor_steps: Prefix lengths used as a small quadrature set.

    Returns:
        [B, N, K, 4] with mean change, std, linear slope, and endpoint change.
    """
    if future.ndim != 3:
        raise ValueError(f"future must be [B,N,H], got {tuple(future.shape)}")
    if last_history.shape != future.shape[:2]:
        raise ValueError(
            f"last_history must have shape {tuple(future.shape[:2])}, "
            f"got {tuple(last_history.shape)}"
        )
    if not anchor_steps or any(step < 2 or step > future.shape[-1] for step in anchor_steps):
        raise ValueError("anchor_steps must be within [2, future horizon]")

    change = future - last_history.unsqueeze(-1)
    descriptors = []
    for step in anchor_steps:
        prefix = change[..., :step]
        mean = prefix.mean(dim=-1)
        std = prefix.std(dim=-1, unbiased=False)
        t = torch.linspace(-1.0, 1.0, step, device=future.device, dtype=future.dtype)
        t = t - t.mean()
        denom = t.square().sum().clamp_min(1e-6)
        slope = (prefix * t).sum(dim=-1) / denom
        endpoint = prefix[..., -1]
        descriptors.append(torch.stack((mean, std, slope, endpoint), dim=-1))
    return torch.stack(descriptors, dim=2)


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.clamp_min(1e-12)
    total = weights.sum()
    if not torch.isfinite(total) or float(total.detach()) <= 1e-10:
        return torch.full_like(weights, 1.0 / max(weights.numel(), 1))
    return weights / total


def _sinkhorn_from_log_kernel(
    log_kernel: torch.Tensor,
    source_marginal: torch.Tensor,
    target_marginal: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """Differentiable log-domain Sinkhorn scaling for a small sampled matrix."""
    a = _normalize_weights(source_marginal)
    b = _normalize_weights(target_marginal)
    log_a = torch.log(a.clamp_min(1e-12))
    log_b = torch.log(b.clamp_min(1e-12))
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    return torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))


def _pairwise_squared_cost(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cost = torch.cdist(source, target, p=2).square()
    scale = cost.detach().mean().clamp_min(1e-6)
    return cost / scale


class ContinuousHorizonTransferAlignment(nn.Module):
    """Continuous-time-conditioned feature metric and low-cost conditional OT."""

    def __init__(
        self,
        input_dim: int = 128,
        horizon_steps: int = 36,
        minutes_per_step: float = 5.0,
        anchor_steps: Sequence[int] = (2, 12, 24, 36),
        time_dim: int = 16,
        hidden_dim: int = 64,
        relation_dim: int = 32,
        future_dim: int = 4,
        epsilon: float = 0.08,
        sinkhorn_iterations: int = 20,
        max_samples: int = 128,
        future_weight: float = 1.0,
        prior_kl_weight: float = 1.0,
    ):
        super().__init__()
        anchors = tuple(int(v) for v in anchor_steps)
        if input_dim <= 0 or horizon_steps <= 0 or minutes_per_step <= 0:
            raise ValueError("input_dim, horizon_steps and minutes_per_step must be positive")
        if not anchors or tuple(sorted(anchors)) != anchors:
            raise ValueError("anchor_steps must be non-empty and sorted")
        if anchors[0] < 2 or anchors[-1] > horizon_steps:
            raise ValueError("anchor_steps must lie within the prediction horizon")
        if epsilon <= 0 or sinkhorn_iterations <= 0 or max_samples < 2:
            raise ValueError("epsilon/iterations must be positive and max_samples >= 2")

        self.input_dim = int(input_dim)
        self.horizon_steps = int(horizon_steps)
        self.minutes_per_step = float(minutes_per_step)
        self.anchor_steps = anchors
        self.time_dim = int(time_dim)
        self.epsilon = float(epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.max_samples = int(max_samples)
        self.future_weight = float(future_weight)
        self.prior_kl_weight = float(prior_kl_weight)

        # One continuous function encodes all horizons; no per-step embedding table.
        self.time_encoder = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, time_dim),
            nn.GELU(),
        )
        self.film_generator = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * input_dim),
        )
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        self.history_norm = nn.LayerNorm(input_dim)
        self.history_projection = nn.Linear(input_dim, relation_dim, bias=False)
        self.future_norm = nn.LayerNorm(future_dim)
        self.future_projection = nn.Linear(future_dim, relation_dim, bias=False)

    def encode_horizons(self, steps: Iterable[int], device=None, dtype=None) -> torch.Tensor:
        steps = tuple(int(v) for v in steps)
        if not steps or any(v < 1 or v > self.horizon_steps for v in steps):
            raise ValueError("requested horizons must lie within [1, horizon_steps]")
        if device is None:
            device = self.history_projection.weight.device
        if dtype is None:
            dtype = self.history_projection.weight.dtype
        physical_minutes = torch.tensor(steps, device=device, dtype=dtype) * self.minutes_per_step
        max_minutes = float(self.horizon_steps) * self.minutes_per_step
        u = physical_minutes / max_minutes
        features = [u.unsqueeze(-1)]
        for frequency in (1.0, 2.0, 4.0, 8.0):
            phase = 2.0 * math.pi * frequency * u
            features.extend((torch.sin(phase).unsqueeze(-1), torch.cos(phase).unsqueeze(-1)))
        return self.time_encoder(torch.cat(features, dim=-1))

    def horizon_marginals(
        self,
        base_source_weights: torch.Tensor,
        base_target_weights: torch.Tensor,
        source_descriptors: torch.Tensor,
        target_descriptors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Make DVGP marginals horizon-aware using future-change compatibility.

        The original DVGP overlap probabilities remain the base weights. A
        bounded [0.5, 1.5] compatibility gate changes their relative mass for
        each anchor without allowing a single sample to dominate unchecked.
        """
        if source_descriptors.ndim != 3 or target_descriptors.ndim != 3:
            raise ValueError("descriptors must have shape [samples, anchors, features]")
        if source_descriptors.shape[1] != len(self.anchor_steps) or target_descriptors.shape[1] != len(self.anchor_steps):
            raise ValueError("descriptor anchor dimension does not match anchor_steps")
        source_base = _normalize_weights(base_source_weights.detach())
        target_base = _normalize_weights(base_target_weights.detach())
        source_rows, target_rows = [], []
        for k in range(len(self.anchor_steps)):
            source_desc = source_descriptors[:, k, :]
            target_desc = target_descriptors[:, k, :]
            source_center = target_desc.mean(dim=0, keepdim=True)
            target_center = source_desc.mean(dim=0, keepdim=True)
            source_distance = torch.linalg.vector_norm(source_desc - source_center, dim=-1)
            target_distance = torch.linalg.vector_norm(target_desc - target_center, dim=-1)
            scale = torch.cat((source_distance, target_distance)).median().detach().clamp_min(1e-3)
            source_gate = 0.5 + torch.exp(-source_distance / scale)
            target_gate = 0.5 + torch.exp(-target_distance / scale)
            source_rows.append(_normalize_weights(source_base * source_gate))
            target_rows.append(_normalize_weights(target_base * target_gate))
        return torch.stack(source_rows, dim=0), torch.stack(target_rows, dim=0)

    def _sample_indices(self, count: int, device: torch.device) -> torch.Tensor:
        if count <= self.max_samples:
            return torch.arange(count, device=device)
        # Deterministic stratified selection avoids extra RNG state and keeps
        # source/target node coverage reproducible across runs.
        return torch.linspace(0, count - 1, self.max_samples, device=device).round().long()

    def compute_alignment_loss(
        self,
        source_rep: torch.Tensor,
        target_rep: torch.Tensor,
        source_descriptors: torch.Tensor,
        target_descriptors: torch.Tensor,
        source_marginals: torch.Tensor,
        target_marginals: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute one base plan plus K low-cost horizon-conditioned plans.

        Representations are flattened to [samples, input_dim]. Descriptors are
        [samples, K, 4], marginals are [K, samples]. Each conditional plan is
        obtained by KL-reweighting the shared base plan and rescaling it to the
        corresponding horizon-specific marginals.
        """
        if source_rep.ndim != 2 or target_rep.ndim != 2:
            raise ValueError("source_rep and target_rep must be flattened [samples, dim]")
        if source_rep.shape[-1] != self.input_dim or target_rep.shape[-1] != self.input_dim:
            raise ValueError(f"representation dimension must be {self.input_dim}")
        if source_descriptors.shape != (source_rep.shape[0], len(self.anchor_steps), 4):
            raise ValueError("source_descriptors have an incompatible shape")
        if target_descriptors.shape != (target_rep.shape[0], len(self.anchor_steps), 4):
            raise ValueError("target_descriptors have an incompatible shape")
        if source_marginals.shape != (len(self.anchor_steps), source_rep.shape[0]):
            raise ValueError("source_marginals have an incompatible shape")
        if target_marginals.shape != (len(self.anchor_steps), target_rep.shape[0]):
            raise ValueError("target_marginals have an incompatible shape")

        src_idx = self._sample_indices(source_rep.shape[0], source_rep.device)
        tgt_idx = self._sample_indices(target_rep.shape[0], target_rep.device)
        src = source_rep.index_select(0, src_idx)
        tgt = target_rep.index_select(0, tgt_idx)
        src_desc = source_descriptors.index_select(0, src_idx)
        tgt_desc = target_descriptors.index_select(0, tgt_idx)
        src_m = source_marginals.index_select(1, src_idx)
        tgt_m = target_marginals.index_select(1, tgt_idx)

        source_base_feature = F.normalize(
            self.history_projection(self.history_norm(src)), dim=-1, eps=1e-6
        )
        target_base_feature = F.normalize(
            self.history_projection(self.history_norm(tgt)), dim=-1, eps=1e-6
        )
        base_cost = _pairwise_squared_cost(source_base_feature, target_base_feature)
        base_source_marginal = _normalize_weights(src_m.mean(dim=0))
        base_target_marginal = _normalize_weights(tgt_m.mean(dim=0))
        base_plan = _sinkhorn_from_log_kernel(
            -base_cost / self.epsilon,
            base_source_marginal,
            base_target_marginal,
            self.sinkhorn_iterations,
        )

        time_codes = self.encode_horizons(self.anchor_steps, src.device, src.dtype)
        normalized_source = self.history_norm(src)
        normalized_target = self.history_norm(tgt)
        conditional_losses = []
        conditional_plans = []
        film_states = []
        row_errors, col_errors = [], []

        for k, time_code in enumerate(time_codes):
            gamma, beta = self.film_generator(time_code).chunk(2, dim=-1)
            gamma = 0.25 * torch.tanh(gamma)
            beta = 0.10 * torch.tanh(beta)
            film_states.append(torch.cat((gamma, beta), dim=-1))

            source_conditioned = (1.0 + gamma) * normalized_source + beta
            target_conditioned = (1.0 + gamma) * normalized_target + beta
            source_history = F.normalize(
                self.history_projection(source_conditioned), dim=-1, eps=1e-6
            )
            target_history = F.normalize(
                self.history_projection(target_conditioned), dim=-1, eps=1e-6
            )
            source_future = F.normalize(
                self.future_projection(self.future_norm(src_desc[:, k, :])), dim=-1, eps=1e-6
            )
            target_future = F.normalize(
                self.future_projection(self.future_norm(tgt_desc[:, k, :])), dim=-1, eps=1e-6
            )

            history_cost = _pairwise_squared_cost(source_history, target_history)
            future_cost = _pairwise_squared_cost(source_future, target_future)
            delta_cost = (history_cost + self.future_weight * future_cost) / (1.0 + self.future_weight)

            prior = base_plan.clamp_min(1e-12)
            log_kernel = torch.log(prior) - delta_cost / self.epsilon
            plan = _sinkhorn_from_log_kernel(
                log_kernel, src_m[k], tgt_m[k], self.sinkhorn_iterations
            )
            kl = (plan * (torch.log(plan.clamp_min(1e-12)) - torch.log(prior))).sum()
            conditional_losses.append((plan * delta_cost).sum() + self.epsilon * self.prior_kl_weight * kl)
            conditional_plans.append(plan)
            row_errors.append((plan.sum(dim=1) - _normalize_weights(src_m[k])).abs().max())
            col_errors.append((plan.sum(dim=0) - _normalize_weights(tgt_m[k])).abs().max())

        loss = torch.stack(conditional_losses).mean()
        if len(film_states) > 1:
            smooth_terms = []
            for left, right, h0, h1 in zip(
                film_states[:-1], film_states[1:], self.anchor_steps[:-1], self.anchor_steps[1:]
            ):
                delta_u = float(h1 - h0) / float(self.horizon_steps)
                smooth_terms.append((right - left).square().mean() / max(delta_u, 1e-6))
            smoothness = torch.stack(smooth_terms).mean()
        else:
            smoothness = loss.new_zeros(())

        diversity = []
        for i in range(len(conditional_plans)):
            for j in range(i + 1, len(conditional_plans)):
                diversity.append(0.5 * (conditional_plans[i] - conditional_plans[j]).abs().sum())
        plan_diversity = torch.stack(diversity).mean() if diversity else loss.new_zeros(())
        diagnostics = {
            "plan_diversity": plan_diversity.detach(),
            "max_row_marginal_error": torch.stack(row_errors).max().detach(),
            "max_col_marginal_error": torch.stack(col_errors).max().detach(),
            "smoothness": smoothness.detach(),
            "sampled_source": loss.new_tensor(float(src.shape[0])).detach(),
            "sampled_target": loss.new_tensor(float(tgt.shape[0])).detach(),
        }
        return loss, diagnostics

    def smoothness_loss(self) -> torch.Tensor:
        """Smooth the continuous FiLM function over the configured anchors."""
        codes = self.encode_horizons(self.anchor_steps)
        states = []
        for code in codes:
            gamma, beta = self.film_generator(code).chunk(2, dim=-1)
            states.append(torch.cat((0.25 * torch.tanh(gamma), 0.10 * torch.tanh(beta)), dim=-1))
        if len(states) < 2:
            return self.history_projection.weight.new_zeros(())
        terms = []
        for left, right, h0, h1 in zip(states[:-1], states[1:], self.anchor_steps[:-1], self.anchor_steps[1:]):
            delta_u = float(h1 - h0) / float(self.horizon_steps)
            terms.append((right - left).square().mean() / max(delta_u, 1e-6))
        return torch.stack(terms).mean()
