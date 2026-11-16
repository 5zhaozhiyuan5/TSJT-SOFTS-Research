"""MTPB-inspired multi-scale pattern bank and horizon-conditioned MoE-OT.

This is a lightweight Stage-I transfer module, not a copy of the full MTPB
predictor. It keeps five physical history scales (1/3/6/12/24 hours), groups
them into three interpretable experts, and lets a horizon/context router mix
their pairwise pattern costs. The resulting cost is used by PMME's conditional
OT objective. No graph topology, extra forecasting head, or Stage-II change is
required.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .horizon_transfer import (
        _normalize_weights,
        _pairwise_squared_cost,
        _sinkhorn_from_log_kernel,
    )
except ImportError:
    # PMME adds model/Meta_Models to sys.path and imports this module directly.
    from horizon_transfer import (
        _normalize_weights,
        _pairwise_squared_cost,
        _sinkhorn_from_log_kernel,
    )


DEFAULT_SCALE_MINUTES = (60, 180, 360, 720, 1440)
DEFAULT_SCALE_STEPS = (12, 36, 72, 144, 288)  # default 5-minute grid


def physical_minutes_to_steps(
    durations_minutes: Sequence[int],
    minutes_per_step: float,
) -> tuple[int, ...]:
    """Convert physical durations to exact steps on the preprocessed time grid."""
    interval = float(minutes_per_step)
    if interval <= 0:
        raise ValueError("minutes_per_step must be positive")
    steps = []
    for duration in durations_minutes:
        minutes = float(duration)
        if minutes <= 0:
            raise ValueError("all physical durations must be positive")
        raw_steps = minutes / interval
        rounded_steps = int(round(raw_steps))
        if rounded_steps < 1 or abs(raw_steps - rounded_steps) > 1e-6:
            raise ValueError(
                f"{minutes:g} minutes cannot be represented exactly on a "
                f"{interval:g}-minute grid"
            )
        steps.append(rounded_steps)
    if tuple(sorted(steps)) != tuple(steps) or len(set(steps)) != len(steps):
        raise ValueError("physical durations must be strictly increasing")
    return tuple(steps)


def _as_speed_windows(value) -> np.ndarray:
    """Convert a dataset x_list entry to [windows, nodes, history] speed data."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        # Dataset convention: [windows, nodes, history, channels].
        array = array[..., 0]
    elif array.ndim == 2:
        array = array[:, None, :]
    if array.ndim != 3:
        raise ValueError(
            "source windows must have shape [W,N,L,1], [W,N,L], or [W,L]; "
            f"received {array.shape}"
        )
    return np.asarray(array, dtype=np.float32)


def _resample_patch(patch: np.ndarray, bins: int) -> np.ndarray:
    positions = np.linspace(0.0, float(len(patch) - 1), num=bins)
    return np.interp(positions, np.arange(len(patch), dtype=np.float32), patch).astype(
        np.float32
    )


def _deterministic_kmeans(
    samples: np.ndarray,
    num_prototypes: int,
    seed: int,
    iterations: int = 20,
) -> np.ndarray:
    """Small deterministic NumPy k-means used only for offline bank creation."""
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("k-means requires a non-empty [samples, features] array")
    rng = np.random.default_rng(seed)
    if samples.shape[0] < num_prototypes:
        extra = rng.choice(samples.shape[0], num_prototypes - samples.shape[0], replace=True)
        samples = np.concatenate((samples, samples[extra]), axis=0)
    init_ids = rng.choice(samples.shape[0], num_prototypes, replace=False)
    centers = samples[init_ids].copy()
    for _ in range(iterations):
        distances = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = distances.argmin(axis=1)
        for index in range(num_prototypes):
            selected = samples[labels == index]
            if selected.shape[0] > 0:
                centers[index] = selected.mean(axis=0)
    return centers.astype(np.float32)


def build_multiscale_pattern_bank(
    source_arrays: Mapping[str, object],
    scale_steps: Sequence[int] = DEFAULT_SCALE_STEPS,
    num_prototypes: int = 10,
    descriptor_bins: int = 16,
    max_windows_per_city: int = 24,
    max_nodes_per_city: int = 24,
    max_patches_per_scale: int = 8000,
    seed: int = 7,
) -> torch.Tensor:
    """Build source-only MTPB-style prototypes from historical training windows.

    The input mapping must contain source-city training windows only. For each
    city, a deterministic subset of windows/nodes is sampled. Every history is
    normalized sample-wise and split into non-overlapping patches at each
    physical scale. K-means produces one shared prototype bank per scale.

    Returns:
        Float tensor [5, num_prototypes, descriptor_bins].
    """
    scales = tuple(int(step) for step in scale_steps)
    if len(scales) != 5 or tuple(sorted(scales)) != scales or min(scales) < 1:
        raise ValueError("scale_steps must contain five positive increasing scales")
    if num_prototypes < 2 or descriptor_bins < 4:
        raise ValueError("num_prototypes >= 2 and descriptor_bins >= 4 are required")
    if max_windows_per_city < 1 or max_nodes_per_city < 1 or max_patches_per_scale < 1:
        raise ValueError("pattern-bank sampling limits must all be positive")
    if not source_arrays:
        raise ValueError("at least one source-city array is required")

    rng = np.random.default_rng(seed)
    descriptors_by_scale = [[] for _ in scales]

    for city_index, city_name in enumerate(sorted(source_arrays)):
        windows = _as_speed_windows(source_arrays[city_name])
        window_count, node_count, history_len = windows.shape
        if window_count < 1 or node_count < 1:
            raise ValueError(f"{city_name} contains no usable history windows or nodes")
        if max(scales) > history_len:
            raise ValueError(
                f"{city_name} history length is {history_len} steps, but the largest "
                f"configured physical scale requires {max(scales)} steps. Check the "
                "preprocessed sampling interval and his_num."
            )

        selected_windows = rng.choice(
            window_count,
            size=min(window_count, max_windows_per_city),
            replace=False,
        )
        selected_nodes = rng.choice(
            node_count,
            size=min(node_count, max_nodes_per_city),
            replace=False,
        )

        for window_index in selected_windows:
            for node_index in selected_nodes:
                sequence = np.nan_to_num(
                    windows[window_index, node_index],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32, copy=False)
                for scale_index, scale in enumerate(scales):
                    # Extract historical patches at the configured physical scale.
                    # Normalize each patch independently, matching the query-time
                    # normalization in _pattern_assignments exactly.
                    starts = range(0, history_len - scale + 1, scale)
                    for start in starts:
                        patch = sequence[start : start + scale]
                        patch_mean = float(patch.mean())
                        patch_std = float(patch.std())
                        patch = (patch - patch_mean) / max(patch_std, 1e-4)
                        descriptors_by_scale[scale_index].append(
                            _resample_patch(patch, descriptor_bins)
                        )

        print(
            f"[PatternBank] {city_name}: sampled "
            f"{len(selected_windows)} windows x {len(selected_nodes)} nodes"
        )

    banks = []
    for scale_index, scale in enumerate(scales):
        descriptors = np.asarray(descriptors_by_scale[scale_index], dtype=np.float32)
        if descriptors.shape[0] == 0:
            raise RuntimeError(f"no source patches were generated for scale={scale}")
        if descriptors.shape[0] > max_patches_per_scale:
            selected = rng.choice(
                descriptors.shape[0], size=max_patches_per_scale, replace=False
            )
            descriptors = descriptors[selected]
        centers = _deterministic_kmeans(
            descriptors,
            num_prototypes=num_prototypes,
            seed=seed + 101 * (scale_index + 1),
        )
        banks.append(centers)
        print(
            f"[PatternBank] scale={scale} steps, patches={len(descriptors)}, "
            f"prototypes={num_prototypes}"
        )

    return torch.from_numpy(np.stack(banks, axis=0)).float()


class HorizonPatternMoEAlignment(nn.Module):
    """Three-expert, horizon-conditioned pattern cost for PMME conditional OT."""

    def __init__(
        self,
        pattern_bank: torch.Tensor,
        input_dim: int = 128,
        horizon_steps: int = 36,
        minutes_per_step: float = 5.0,
        anchor_steps: Sequence[int] = (2, 12, 24, 36),
        scale_steps: Sequence[int] = DEFAULT_SCALE_STEPS,
        time_dim: int = 16,
        hidden_dim: int = 64,
        relation_dim: int = 32,
        epsilon: float = 0.08,
        sinkhorn_iterations: int = 20,
        max_samples: int = 128,
        pattern_temperature: float = 0.15,
        router_temperature: float = 1.0,
        future_weight: float = 1.0,
        pattern_weight: float = 0.5,
        prior_kl_weight: float = 1.0,
    ):
        super().__init__()
        anchors = tuple(int(v) for v in anchor_steps)
        scales = tuple(int(v) for v in scale_steps)
        if pattern_bank.ndim != 3 or pattern_bank.shape[0] != len(scales):
            raise ValueError(
                "pattern_bank must have shape [number_of_scales, prototypes, bins]"
            )
        if anchors != tuple(sorted(anchors)) or not anchors:
            raise ValueError("anchor_steps must be non-empty and sorted")
        if anchors[0] < 2 or anchors[-1] > horizon_steps:
            raise ValueError("anchor_steps must lie within [2, horizon_steps]")
        if epsilon <= 0 or sinkhorn_iterations <= 0 or max_samples < 2:
            raise ValueError("epsilon/iterations must be positive and max_samples >= 2")
        if pattern_temperature <= 0 or router_temperature <= 0:
            raise ValueError("pattern and router temperatures must be positive")

        self.input_dim = int(input_dim)
        self.horizon_steps = int(horizon_steps)
        self.minutes_per_step = float(minutes_per_step)
        self.anchor_steps = anchors
        self.scale_steps = scales
        self.descriptor_bins = int(pattern_bank.shape[-1])
        self.epsilon = float(epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.max_samples = int(max_samples)
        self.pattern_temperature = float(pattern_temperature)
        self.router_temperature = float(router_temperature)
        self.future_weight = float(future_weight)
        self.pattern_weight = float(pattern_weight)
        self.prior_kl_weight = float(prior_kl_weight)

        # Prototypes are fixed source-only knowledge, not target-test statistics.
        self.register_buffer("pattern_prototypes", pattern_bank.detach().float().clone())

        self.time_encoder = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, time_dim),
            nn.GELU(),
        )
        self.history_norm = nn.LayerNorm(input_dim)
        self.history_projection = nn.Linear(input_dim, relation_dim, bias=False)
        self.future_norm = nn.LayerNorm(4)
        self.future_projection = nn.Linear(4, relation_dim, bias=False)
        self.film_generator = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * input_dim),
        )
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        context_dim = 32
        self.router_context = nn.Linear(input_dim, context_dim)
        # A factorized pair router: score source and target samples separately,
        # then add their three expert logits for each source-target pair. This
        # preserves pair-conditioned routing without materializing a large
        # [source, target, hidden] MLP input tensor.
        self.router = nn.Sequential(
            nn.Linear(context_dim + 4 + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.source_router_bias = nn.Parameter(torch.zeros(3))
        self.target_router_bias = nn.Parameter(torch.zeros(3))

    def encode_horizons(self, steps, device=None, dtype=None):
        steps = tuple(int(value) for value in steps)
        if not steps or any(value < 1 or value > self.horizon_steps for value in steps):
            raise ValueError("requested horizons must lie within [1, horizon_steps]")
        if device is None:
            device = self.history_projection.weight.device
        if dtype is None:
            dtype = self.history_projection.weight.dtype
        physical_minutes = (
            torch.tensor(steps, device=device, dtype=dtype) * self.minutes_per_step
        )
        max_minutes = float(self.horizon_steps) * self.minutes_per_step
        unit_time = physical_minutes / max_minutes
        features = [unit_time.unsqueeze(-1)]
        for frequency in (1.0, 2.0, 4.0, 8.0):
            phase = 2.0 * torch.pi * frequency * unit_time
            features.extend((torch.sin(phase).unsqueeze(-1), torch.cos(phase).unsqueeze(-1)))
        return self.time_encoder(torch.cat(features, dim=-1))

    def horizon_marginals(
        self,
        base_source_weights: torch.Tensor,
        base_target_weights: torch.Tensor,
        source_descriptors: torch.Tensor,
        target_descriptors: torch.Tensor,
    ):
        """Retain DVGP weights and apply bounded horizon-specific compatibility."""
        if source_descriptors.ndim != 3 or target_descriptors.ndim != 3:
            raise ValueError("descriptors must have shape [samples, anchors, features]")
        if source_descriptors.shape[1] != len(self.anchor_steps):
            raise ValueError("source descriptor anchor count does not match anchor_steps")
        if target_descriptors.shape[1] != len(self.anchor_steps):
            raise ValueError("target descriptor anchor count does not match anchor_steps")

        source_base = _normalize_weights(base_source_weights.detach())
        target_base = _normalize_weights(base_target_weights.detach())
        source_rows, target_rows, source_scores = [], [], []
        for horizon_index in range(len(self.anchor_steps)):
            source_desc = source_descriptors[:, horizon_index, :]
            target_desc = target_descriptors[:, horizon_index, :]
            source_center = target_desc.mean(dim=0, keepdim=True)
            target_center = source_desc.mean(dim=0, keepdim=True)
            source_distance = torch.linalg.vector_norm(source_desc - source_center, dim=-1)
            target_distance = torch.linalg.vector_norm(target_desc - target_center, dim=-1)
            scale = torch.cat((source_distance, target_distance)).median().detach().clamp_min(1e-3)
            source_gate = 0.5 + torch.exp(-source_distance / scale)
            target_gate = 0.5 + torch.exp(-target_distance / scale)
            source_rows.append(_normalize_weights(source_base * source_gate))
            target_rows.append(_normalize_weights(target_base * target_gate))
            source_scores.append((source_base * source_gate).sum())

        return (
            torch.stack(source_rows, dim=0),
            torch.stack(target_rows, dim=0),
            torch.stack(source_scores, dim=0),
        )

    def _sample_indices(self, count: int, device: torch.device) -> torch.Tensor:
        if count <= self.max_samples:
            return torch.arange(count, device=device)
        return torch.linspace(0, count - 1, self.max_samples, device=device).round().long()

    def _pattern_assignments(self, history: torch.Tensor):
        """Return soft prototype assignments [samples, scales, prototypes]."""
        if history.ndim != 2 or history.shape[-1] < max(self.scale_steps):
            raise ValueError(
                f"history must be [samples, >= {max(self.scale_steps)}], "
                f"received {tuple(history.shape)}"
            )
        history = history.float()
        assignments = []
        for scale_index, scale in enumerate(self.scale_steps):
            patch = history[:, -scale:]
            patch_mean = patch.mean(dim=-1, keepdim=True)
            patch_std = patch.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
            patch = (patch - patch_mean) / patch_std
            patch = F.interpolate(
                patch.unsqueeze(1),
                size=self.descriptor_bins,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            prototypes = self.pattern_prototypes[scale_index].to(dtype=patch.dtype)
            distances = torch.cdist(patch, prototypes, p=2).square()
            assignments.append(
                torch.softmax(-distances / self.pattern_temperature, dim=-1)
            )
        return torch.stack(assignments, dim=1)

    @staticmethod
    def _assignment_cost(source_assignments, target_assignments):
        source_root = source_assignments.clamp_min(1e-8).sqrt()
        target_root = target_assignments.clamp_min(1e-8).sqrt()
        cost = 0.5 * torch.cdist(source_root, target_root, p=2).square()
        return cost / cost.detach().mean().clamp_min(1e-6)

    def compute_alignment_loss(
        self,
        source_rep: torch.Tensor,
        target_rep: torch.Tensor,
        source_history: torch.Tensor,
        target_history: torch.Tensor,
        source_descriptors: torch.Tensor,
        target_descriptors: torch.Tensor,
        source_marginals: torch.Tensor,
        target_marginals: torch.Tensor,
        base_source_weights: torch.Tensor,
        base_target_weights: torch.Tensor,
    ):
        """Compute shared-base and horizon-conditioned MoE transport losses."""
        if source_rep.ndim != 2 or target_rep.ndim != 2:
            raise ValueError("source_rep and target_rep must be flattened [samples, dim]")
        if source_rep.shape[-1] != self.input_dim or target_rep.shape[-1] != self.input_dim:
            raise ValueError(f"representation dimension must be {self.input_dim}")
        if source_history.ndim != 2 or target_history.ndim != 2:
            raise ValueError("history windows must be flattened [samples, history]")
        if source_history.shape[0] != source_rep.shape[0]:
            raise ValueError("source history and representation sample counts differ")
        if target_history.shape[0] != target_rep.shape[0]:
            raise ValueError("target history and representation sample counts differ")
        expected_source = (source_rep.shape[0], len(self.anchor_steps), 4)
        expected_target = (target_rep.shape[0], len(self.anchor_steps), 4)
        if tuple(source_descriptors.shape) != expected_source:
            raise ValueError(f"source_descriptors must have shape {expected_source}")
        if tuple(target_descriptors.shape) != expected_target:
            raise ValueError(f"target_descriptors must have shape {expected_target}")

        source_indices = self._sample_indices(source_rep.shape[0], source_rep.device)
        target_indices = self._sample_indices(target_rep.shape[0], target_rep.device)
        source = source_rep.index_select(0, source_indices)
        target = target_rep.index_select(0, target_indices)
        source_hist = source_history.index_select(0, source_indices)
        target_hist = target_history.index_select(0, target_indices)
        source_desc = source_descriptors.index_select(0, source_indices)
        target_desc = target_descriptors.index_select(0, target_indices)
        source_m = source_marginals.index_select(1, source_indices)
        target_m = target_marginals.index_select(1, target_indices)
        source_base = base_source_weights.index_select(0, source_indices)
        target_base = base_target_weights.index_select(0, target_indices)

        normalized_source = self.history_norm(source)
        normalized_target = self.history_norm(target)
        source_base_feature = F.normalize(
            self.history_projection(normalized_source), dim=-1, eps=1e-6
        )
        target_base_feature = F.normalize(
            self.history_projection(normalized_target), dim=-1, eps=1e-6
        )
        base_cost = _pairwise_squared_cost(source_base_feature, target_base_feature)
        base_plan = _sinkhorn_from_log_kernel(
            -base_cost / self.epsilon,
            _normalize_weights(source_base),
            _normalize_weights(target_base),
            self.sinkhorn_iterations,
        ).detach()

        source_pattern = self._pattern_assignments(source_hist)
        target_pattern = self._pattern_assignments(target_hist)
        per_scale_costs = [
            self._assignment_cost(source_pattern[:, scale_index], target_pattern[:, scale_index])
            for scale_index in range(len(self.scale_steps))
        ]
        expert_costs = torch.stack(
            (
                0.5 * (per_scale_costs[0] + per_scale_costs[1]),
                0.5 * (per_scale_costs[2] + per_scale_costs[3]),
                per_scale_costs[4],
            ),
            dim=0,
        )

        # Detached diagnostics for checking prototype collapse/coverage. These
        # statistics do not participate in the loss or gradient computation.
        combined_assignments = torch.cat((source_pattern, target_pattern), dim=0).detach()
        pattern_usage = combined_assignments.mean(dim=0)
        source_pattern_usage = source_pattern.detach().mean(dim=0)
        target_pattern_usage = target_pattern.detach().mean(dim=0)
        source_target_pattern_gap = (source_pattern_usage - target_pattern_usage).abs().mean(dim=-1)
        pattern_entropy = -(
            pattern_usage.clamp_min(1e-8) * pattern_usage.clamp_min(1e-8).log()
        ).sum(dim=-1)
        source_pattern_entropy = -(
            source_pattern_usage.clamp_min(1e-8) * source_pattern_usage.clamp_min(1e-8).log()
        ).sum(dim=-1)
        target_pattern_entropy = -(
            target_pattern_usage.clamp_min(1e-8) * target_pattern_usage.clamp_min(1e-8).log()
        ).sum(dim=-1)
        pattern_top1_fraction = combined_assignments.argmax(dim=-1).float().mean(dim=0)
        per_scale_cost_means = torch.stack(
            [cost.detach().mean() for cost in per_scale_costs]
        )

        time_codes = self.encode_horizons(self.anchor_steps, source.device, source.dtype)
        # Pairwise contexts let the router choose experts for each source/target
        # sample pair, rather than assigning one global gate to the whole batch.
        source_context = self.router_context(normalized_source)
        target_context = self.router_context(normalized_target)
        conditional_losses = []
        conditional_plans = []
        router_weights = []
        router_entropies = []
        expert_cost_means = []
        film_states = []
        row_errors, col_errors = [], []
        plan_entropies, normalized_plan_entropies = [], []
        mean_row_entropies, plan_max_weights, row_top1_masses = [], [], []
        router_pair_stds, router_top1_fractions = [], []

        for horizon_index, time_code in enumerate(time_codes):
            gamma, beta = self.film_generator(time_code).chunk(2, dim=-1)
            gamma = 0.25 * torch.tanh(gamma)
            beta = 0.10 * torch.tanh(beta)
            film_states.append(torch.cat((gamma, beta), dim=-1))

            source_conditioned = (1.0 + gamma) * normalized_source + beta
            target_conditioned = (1.0 + gamma) * normalized_target + beta
            source_feature = F.normalize(
                self.history_projection(source_conditioned), dim=-1, eps=1e-6
            )
            target_feature = F.normalize(
                self.history_projection(target_conditioned), dim=-1, eps=1e-6
            )
            source_future = F.normalize(
                self.future_projection(self.future_norm(source_desc[:, horizon_index, :])),
                dim=-1,
                eps=1e-6,
            )
            target_future = F.normalize(
                self.future_projection(self.future_norm(target_desc[:, horizon_index, :])),
                dim=-1,
                eps=1e-6,
            )

            history_cost = _pairwise_squared_cost(source_feature, target_feature)
            future_cost = _pairwise_squared_cost(source_future, target_future)
            source_time = time_code.unsqueeze(0).expand(source.shape[0], -1)
            target_time = time_code.unsqueeze(0).expand(target.shape[0], -1)
            source_router_input = torch.cat(
                (source_context, source_desc[:, horizon_index, :], source_time),
                dim=-1,
            )
            target_router_input = torch.cat(
                (target_context, target_desc[:, horizon_index, :], target_time),
                dim=-1,
            )
            source_logits = self.router(source_router_input) + self.source_router_bias
            target_logits = self.router(target_router_input) + self.target_router_bias
            pair_logits = source_logits[:, None, :] + target_logits[None, :, :]
            gate = torch.softmax(pair_logits / self.router_temperature, dim=-1)
            router_entropies.append(
                -(
                    gate.clamp_min(1e-8) * gate.clamp_min(1e-8).log()
                ).sum(dim=-1).mean()
            )
            router_pair_stds.append(gate.detach().std(dim=(0, 1), unbiased=False))
            router_top1_fractions.append(
                F.one_hot(gate.detach().argmax(dim=-1), num_classes=3).float().mean(dim=(0, 1))
            )
            pattern_cost = (expert_costs.permute(1, 2, 0) * gate).sum(dim=-1)
            combined_cost = (
                history_cost
                + self.future_weight * future_cost
                + self.pattern_weight * pattern_cost
            ) / (1.0 + self.future_weight + self.pattern_weight)

            prior = base_plan.clamp_min(1e-12)
            plan = _sinkhorn_from_log_kernel(
                torch.log(prior) - combined_cost / self.epsilon,
                source_m[horizon_index],
                target_m[horizon_index],
                self.sinkhorn_iterations,
            )
            kl = (
                plan
                * (
                    torch.log(plan.clamp_min(1e-12))
                    - torch.log(prior)
                )
            ).sum()
            conditional_losses.append(
                (plan * combined_cost).sum()
                + self.epsilon * self.prior_kl_weight * kl
            )
            conditional_plans.append(plan)
            safe_plan = plan.detach().clamp_min(1e-12)
            plan_entropy = -(safe_plan * safe_plan.log()).sum()
            plan_entropies.append(plan_entropy)
            normalized_plan_entropies.append(
                plan_entropy / math.log(max(safe_plan.numel(), 2))
            )
            row_distribution = safe_plan / safe_plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
            row_entropy = -(
                row_distribution * row_distribution.clamp_min(1e-12).log()
            ).sum(dim=1)
            mean_row_entropies.append(row_entropy.mean())
            plan_max_weights.append(safe_plan.max())
            row_top1_masses.append(safe_plan.max(dim=1).values.sum())
            router_weights.append(gate.mean(dim=(0, 1)))
            expert_cost_means.append(expert_costs.mean(dim=(1, 2)))
            row_errors.append(
                (plan.sum(dim=1) - _normalize_weights(source_m[horizon_index])).abs().max()
            )
            col_errors.append(
                (plan.sum(dim=0) - _normalize_weights(target_m[horizon_index])).abs().max()
            )

        horizon_losses = torch.stack(conditional_losses)
        loss = horizon_losses.mean()
        if len(film_states) > 1:
            smooth_terms = []
            for left, right, step0, step1 in zip(
                film_states[:-1],
                film_states[1:],
                self.anchor_steps[:-1],
                self.anchor_steps[1:],
            ):
                interval = float(step1 - step0) / float(self.horizon_steps)
                smooth_terms.append((right - left).square().mean() / max(interval, 1e-6))
            smoothness = torch.stack(smooth_terms).mean()
        else:
            smoothness = loss.new_zeros(())

        diversity_terms = []
        for left_index in range(len(conditional_plans)):
            for right_index in range(left_index + 1, len(conditional_plans)):
                diversity_terms.append(
                    0.5
                    * (
                        conditional_plans[left_index] - conditional_plans[right_index]
                    ).abs().sum()
                )
        plan_diversity = (
            torch.stack(diversity_terms).mean()
            if diversity_terms
            else loss.new_zeros(())
        )
        router_tensor = torch.stack(router_weights, dim=0)
        router_entropy = torch.stack(router_entropies).mean()
        router_smoothness_terms = []
        for left, right, step0, step1 in zip(
            router_tensor[:-1],
            router_tensor[1:],
            self.anchor_steps[:-1],
            self.anchor_steps[1:],
        ):
            interval = float(step1 - step0) / float(self.horizon_steps)
            router_smoothness_terms.append(
                (right - left).square().mean() / max(interval, 1e-6)
            )
        router_smoothness = (
            torch.stack(router_smoothness_terms).mean()
            if router_smoothness_terms
            else loss.new_zeros(())
        )
        diagnostics = {
            "horizon_losses": horizon_losses,
            "router_weights": router_tensor.detach(),
            "router_entropy": router_entropy.detach(),
            "router_smoothness_loss": router_smoothness,
            "expert_cost_means": torch.stack(expert_cost_means).detach(),
            "plan_diversity": plan_diversity.detach(),
            "max_row_marginal_error": torch.stack(row_errors).max().detach(),
            "max_col_marginal_error": torch.stack(col_errors).max().detach(),
            "row_marginal_errors": torch.stack(row_errors).detach(),
            "col_marginal_errors": torch.stack(col_errors).detach(),
            "router_pair_std": torch.stack(router_pair_stds).detach(),
            "router_top1_fraction": torch.stack(router_top1_fractions).detach(),
            "plan_entropy": torch.stack(plan_entropies).detach(),
            "normalized_plan_entropy": torch.stack(normalized_plan_entropies).detach(),
            "mean_row_entropy": torch.stack(mean_row_entropies).detach(),
            "plan_max_weight": torch.stack(plan_max_weights).detach(),
            "row_top1_mass": torch.stack(row_top1_masses).detach(),
            "pattern_usage": pattern_usage.detach(),
            "source_pattern_usage": source_pattern_usage.detach(),
            "target_pattern_usage": target_pattern_usage.detach(),
            "source_target_pattern_gap": source_target_pattern_gap.detach(),
            "pattern_entropy": pattern_entropy.detach(),
            "source_pattern_entropy": source_pattern_entropy.detach(),
            "target_pattern_entropy": target_pattern_entropy.detach(),
            "pattern_top1_fraction": pattern_top1_fraction.detach(),
            "per_scale_cost_means": per_scale_cost_means.detach(),
            "scale_steps": list(self.scale_steps),
            "smoothness": smoothness.detach(),
            "sampled_source": loss.new_tensor(float(source.shape[0])).detach(),
            "sampled_target": loss.new_tensor(float(target.shape[0])).detach(),
        }
        return loss, diagnostics

    def smoothness_loss(self) -> torch.Tensor:
        states = []
        for time_code in self.encode_horizons(self.anchor_steps):
            gamma, beta = self.film_generator(time_code).chunk(2, dim=-1)
            states.append(torch.cat((0.25 * torch.tanh(gamma), 0.10 * torch.tanh(beta))))
        if len(states) < 2:
            return self.history_projection.weight.new_zeros(())
        terms = []
        for left, right, step0, step1 in zip(
            states[:-1], states[1:], self.anchor_steps[:-1], self.anchor_steps[1:]
        ):
            interval = float(step1 - step0) / float(self.horizon_steps)
            terms.append((right - left).square().mean() / max(interval, 1e-6))
        return torch.stack(terms).mean()
