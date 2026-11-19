"""Source-only history-to-future pattern bank and target-train reliability calibration.

Only historical training windows and paired training targets may be used here.
The future response table is fixed knowledge; test-time lookup is history-only.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .pattern_moe import _deterministic_kmeans
    from .horizon_transfer import build_future_change_descriptors
except ImportError:
    from pattern_moe import _deterministic_kmeans
    from horizon_transfer import build_future_change_descriptors


def patch_descriptors(histories: torch.Tensor, scale: int, bins: int) -> torch.Tensor:
    """Exactly the same trailing-patch transform used by online lookup."""
    if histories.ndim != 2 or histories.shape[-1] < scale:
        raise ValueError("histories must be [samples, history >= scale]")
    patch = histories[:, -scale:].float()
    patch = (patch - patch.mean(-1, keepdim=True)) / patch.std(
        -1, keepdim=True, unbiased=False
    ).clamp_min(1e-4)
    return F.interpolate(
        patch.unsqueeze(1), size=bins, mode="linear", align_corners=False
    ).squeeze(1)


def paired_training_examples(
    histories, futures, mean, std, anchor_steps: Sequence[int],
    max_windows: int, max_nodes: int, rng: np.random.Generator,
):
    """Sample paired source or target TRAIN examples in the existing RevIN frame.

    histories: [W,N,288,2], channel 0 is city-normalized speed.
    futures: [W,N,36], physical speed; never forecast/test labels.
    """
    x = torch.as_tensor(histories)
    y = torch.as_tensor(futures)
    if x.ndim != 4 or y.ndim != 3 or x.shape[:2] != y.shape[:2]:
        raise ValueError("paired windows require x=[W,N,L,C], y=[W,N,H]")
    if x.shape[-1] < 1 or y.shape[-1] < max(anchor_steps):
        raise ValueError("training examples have incompatible history/horizon")
    windows = np.sort(rng.choice(x.shape[0], min(x.shape[0], max_windows), replace=False))
    nodes = np.sort(rng.choice(x.shape[1], min(x.shape[1], max_nodes), replace=False))
    x = x[windows][:, nodes, :, 0].reshape(-1, x.shape[-2]).float()
    y = y[windows][:, nodes, :].reshape(-1, y.shape[-1]).float()
    if not x.numel() or not y.numel():
        raise ValueError("no paired training examples")
    mu = x.mean(-1, keepdim=True)
    sigma = x.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4)
    standard_y = (y - float(np.asarray(mean).reshape(-1)[0])) / float(
        np.asarray(std).reshape(-1)[0]
    )
    rev_y = (standard_y - mu) / sigma
    rev_last = ((x[:, -1:] - mu) / sigma).squeeze(-1)
    response = build_future_change_descriptors(
        rev_y.unsqueeze(1), rev_last.unsqueeze(1), anchor_steps
    ).squeeze(1)
    if not torch.isfinite(x).all() or not torch.isfinite(response).all():
        raise ValueError("nonfinite values in paired training examples")
    return x, response


@torch.no_grad()
def build_joint_pattern_bank(
    source_histories: Mapping[str, object],
    source_futures: Mapping[str, object],
    source_means: Mapping[str, object],
    source_stds: Mapping[str, object],
    scale_steps: Sequence[int], anchor_steps: Sequence[int],
    num_prototypes: int = 10, descriptor_bins: int = 16,
    pattern_temperature: float = 0.15,
    max_windows_per_city: int = 24, max_nodes_per_city: int = 24,
    max_patches_per_scale: int = 8000, support_prior: float = 8.0, seed: int = 1,
) -> dict[str, torch.Tensor]:
    """Return P[5,K,16], R[5,K,4,4], variance[5,K,4,4], support[5,K]."""
    if set(source_histories) != set(source_futures):
        raise ValueError("source history and future city sets must be identical")
    if not source_histories or len(scale_steps) != 5:
        raise ValueError("expected nonempty source cities and five history scales")
    if pattern_temperature <= 0 or support_prior < 0:
        raise ValueError("pattern_temperature must be positive; support_prior nonnegative")
    rng = np.random.default_rng(seed)
    all_history, all_response = [], []
    for city in sorted(source_histories):
        history, response = paired_training_examples(
            source_histories[city], source_futures[city],
            source_means[city], source_stds[city], anchor_steps,
            max_windows_per_city, max_nodes_per_city, rng,
        )
        if history.shape[-1] < max(scale_steps):
            raise ValueError(f"{city}: insufficient input length for pattern scales")
        all_history.append(history)
        all_response.append(response)
    x = torch.cat(all_history)
    y = torch.cat(all_response)
    prototypes, responses, variances, supports = [], [], [], []
    global_response = y.mean(dim=0)
    for index, scale in enumerate(scale_steps):
        patch = patch_descriptors(x, int(scale), descriptor_bins)
        if patch.shape[0] > max_patches_per_scale:
            ids = np.sort(rng.choice(patch.shape[0], max_patches_per_scale, replace=False))
            patch_fit = patch[ids]
            y_fit = y[ids]
        else:
            patch_fit, y_fit = patch, y
        centers = torch.from_numpy(_deterministic_kmeans(
            patch_fit.cpu().numpy(), num_prototypes=num_prototypes,
            seed=seed + 101 * (index + 1),
        ))
        assignment = torch.softmax(
            -torch.cdist(patch_fit, centers).square() / pattern_temperature, dim=-1
        )
        support = assignment.sum(dim=0)
        weighted_sum = torch.einsum("nk,nhd->khd", assignment, y_fit)
        response = (
            weighted_sum + support_prior * global_response.unsqueeze(0)
        ) / (support[:, None, None] + support_prior).clamp_min(1e-6)
        deviation = y_fit[:, None, :, :] - response[None, :, :, :]
        variance = torch.einsum(
            "nk,nkhd->khd", assignment, deviation.square()
        ) / support[:, None, None].clamp_min(1e-6)
        prototypes.append(centers)
        responses.append(response)
        variances.append(variance)
        supports.append(support)
    return {
        "prototypes": torch.stack(prototypes).float(),
        "responses": torch.stack(responses).float(),
        "uncertainty": torch.stack(variances).float(),
        "support": torch.stack(supports).float(),
    }


@torch.no_grad()
def calibrate_target_scales(
    bank: dict[str, torch.Tensor], target_histories, target_futures,
    target_mean, target_std, scale_steps: Sequence[int], anchor_steps: Sequence[int],
    pattern_temperature: float = 0.15, calibration_temperature: float = 0.5,
    shrink_to_uniform: float = 0.5, max_windows: int = 46,
    max_nodes: int = 64, seed: int = 1,
) -> dict[str, torch.Tensor]:
    """Estimate horizon-specific scale weights without touching target TEST data.

    Weighted reliability is zero when no source response improves on the
    persistence (zero-change) descriptor on target TRAIN examples.
    """
    if not 0 <= shrink_to_uniform <= 1 or calibration_temperature <= 0:
        raise ValueError("invalid calibration temperature/shrinkage")
    rng = np.random.default_rng(seed + 100003)
    x, truth = paired_training_examples(
        target_histories, target_futures, target_mean, target_std,
        anchor_steps, max_windows, max_nodes, rng,
    )
    response_bank = bank["responses"]
    error_rows = []
    normalizer = truth.abs().mean(dim=0).clamp_min(0.05)  # [horizon,feature]
    persistence_error = (truth.abs() / normalizer).mean(dim=(0, 2)).clamp_min(1e-6)
    for scale_index, scale in enumerate(scale_steps):
        patch = patch_descriptors(x, int(scale), bank["prototypes"].shape[-1])
        assignment = torch.softmax(
            -torch.cdist(patch, bank["prototypes"][scale_index]).square()
            / pattern_temperature, dim=-1
        )
        estimate = torch.einsum(
            "nk,khd->nhd", assignment, response_bank[scale_index]
        )
        normalized_error = ((estimate - truth).abs() / normalizer).mean(dim=(0, 2))
        error_rows.append(normalized_error / persistence_error)
    errors = torch.stack(error_rows)  # [scale,horizon]
    probabilities = torch.softmax(-errors / calibration_temperature, dim=0)
    weights = (
        shrink_to_uniform / len(scale_steps)
        + (1.0 - shrink_to_uniform) * probabilities
    )
    # Conservative gate: activate only if a source scale beats persistence.
    # The factor is bounded and calibration uses training futures only.
    reliability = ((1.0 - errors.min(dim=0).values) / 0.25).clamp(0.0, 1.0)
    if not torch.isfinite(errors).all() or not torch.isfinite(weights).all():
        raise ValueError("nonfinite target calibration")
    return {
        "scale_weights": weights.float(),
        "reliability": reliability.float(),
        "relative_errors": errors.float(),
        "target_samples": torch.tensor(int(x.shape[0])),
    }
