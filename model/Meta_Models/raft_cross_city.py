"""Cross-city multi-scale retrieval with reliability-aware horizon fusion.

The source-only retrieval bank and multi-scale analog matching follow RAFT
(ICML 2025). The fixed dense forecast mixer is replaced by a small,
query-conditioned horizon gate inspired by SARAF (KDD 2026) and TimeFuse
(ICML 2025): retrieval influence depends on query stationarity, top-k match
quality, future-candidate disagreement, the base/retrieval forecasts, and
forecast horizon. The SOFTS forecaster remains frozen in Stage-II.
"""
import math
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scalar(value, device):
    value = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
    if value.numel() == 0:
        raise ValueError("City normalization statistics must contain a scalar.")
    return value[0]


def _pool_sequence(sequence: torch.Tensor, period: int) -> torch.Tensor:
    """Average-pool [B,T] sequences into non-overlapping period patches."""
    if sequence.ndim != 2:
        raise ValueError(f"Expected [B,T] sequence, received {tuple(sequence.shape)}")
    if sequence.shape[-1] % period != 0:
        raise ValueError(
            f"Sequence length {sequence.shape[-1]} must be divisible by period {period}."
        )
    return F.avg_pool1d(sequence.unsqueeze(1), kernel_size=period, stride=period).squeeze(1)


def configure_raft_stage2_parameters(model: nn.Module):
    """Freeze the forecasting backbone and optimize only the retrieval adapter."""
    adapter = getattr(model, "raft_adapter", None)
    if adapter is None:
        raise RuntimeError("Stage-II requires a MultiScaleRetrievalAdapter.")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable RAFT Stage-II parameters were found.")
    return trainable


class ReliabilityAwareHorizonGate(nn.Module):
    """Bounded per-query/per-horizon gate between SOFTS and retrieved futures.

    Features are computed without target-test labels:
      1. normalized SOFTS forecast relative to the last observation;
      2. normalized retrieved forecast relative to the last observation;
      3. their absolute disagreement;
      4. weighted top-k history similarity;
      5. dispersion among retrieved future candidates;
      6. stationarity of the target history window;
      7. normalized forecast-horizon position.

    The gate starts close to the identity SOFTS forecast and can contribute at
    most max_retrieval_weight of the retrieval branch.
    """

    def __init__(
        self,
        pred_len: int,
        hidden_dim: int = 16,
        max_retrieval_weight: float = 0.5,
        initial_retrieval_weight: float = 0.05,
    ):
        super().__init__()
        if not 0.0 < initial_retrieval_weight < max_retrieval_weight <= 1.0:
            raise ValueError("Require 0 < initial gate < max gate <= 1.")
        self.pred_len = int(pred_len)
        self.max_retrieval_weight = float(max_retrieval_weight)
        self.network = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        initial_probability = initial_retrieval_weight / max_retrieval_weight
        initial_bias = math.log(initial_probability / (1.0 - initial_probability))
        nn.init.constant_(final.bias, initial_bias)

    def forward(
        self,
        base_prediction: torch.Tensor,
        retrieved_prediction: torch.Tensor,
        last_observation: torch.Tensor,
        local_scale: torch.Tensor,
        similarity_confidence: torch.Tensor,
        future_disagreement: torch.Tensor,
        stationarity: torch.Tensor,
    ):
        if base_prediction.ndim != 2 or base_prediction.shape[-1] != self.pred_len:
            raise ValueError(
                f"Expected flattened base prediction [Q,{self.pred_len}], "
                f"got {tuple(base_prediction.shape)}."
            )
        if retrieved_prediction.shape != base_prediction.shape:
            raise ValueError("Base and retrieved forecasts must have identical shapes.")

        scale = local_scale.clamp_min(1e-4)
        base_relative = ((base_prediction - last_observation) / scale).clamp(-8.0, 8.0)
        retrieved_relative = (
            (retrieved_prediction - last_observation) / scale
        ).clamp(-8.0, 8.0)
        forecast_gap = (base_relative - retrieved_relative).abs()

        query_count, horizon = base_prediction.shape
        similarity = similarity_confidence.reshape(query_count, 1).expand(-1, horizon)
        disagreement = torch.log1p(
            future_disagreement.reshape(query_count, 1).clamp_min(0.0)
        ).expand(-1, horizon)
        stationary = stationarity.reshape(query_count, 1).expand(-1, horizon)
        horizon_position = torch.linspace(
            0.0,
            1.0,
            steps=horizon,
            device=base_prediction.device,
            dtype=base_prediction.dtype,
        ).unsqueeze(0).expand(query_count, -1)

        features = torch.stack(
            [
                base_relative,
                retrieved_relative,
                forecast_gap,
                similarity,
                disagreement,
                stationary,
                horizon_position,
            ],
            dim=-1,
        )
        gate = self.max_retrieval_weight * torch.sigmoid(
            self.network(features).squeeze(-1)
        )
        fused = base_prediction + gate * (retrieved_prediction - base_prediction)
        return fused, gate


class MultiScaleRetrievalAdapter(nn.Module):
    """RAFT-style source-only analog retrieval adapted to cross-city traffic.

    Each source memory item contains:
      * a normalized historical speed window, represented at multiple periods;
      * the future trajectory relative to the source window's last value/scale.

    Query histories are normalized independently, so cities with different
    speed levels can share a retrieval space. Retrieved futures are mapped
    back to the target query's level and scale, then fused with frozen SOFTS
    through a reliability-aware horizon gate.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        max_memory_size: int,
        source_city_names: Sequence[str],
        periods: Sequence[int] = (4, 2, 1),
        topm: int = 20,
        temperature: float = 0.1,
        query_chunk: int = 256,
        max_retrieval_weight: float = 0.5,
        initial_retrieval_weight: float = 0.05,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.max_memory_size = max(1, int(max_memory_size))
        self.source_city_names = list(source_city_names)
        self.city_to_id = {name: index for index, name in enumerate(self.source_city_names)}
        self.periods = tuple(int(period) for period in periods)
        self.topm = max(1, int(topm))
        self.temperature = max(float(temperature), 1e-4)
        self.query_chunk = max(1, int(query_chunk))

        if not self.source_city_names:
            raise ValueError("At least one source city is required for the retrieval bank.")
        if len(set(self.periods)) != len(self.periods) or not self.periods:
            raise ValueError("periods must be a non-empty sequence of unique integers.")
        for period in self.periods:
            if self.seq_len % period != 0 or self.pred_len % period != 0:
                raise ValueError(
                    f"period={period} must divide seq_len={self.seq_len} "
                    f"and pred_len={self.pred_len}."
                )

        self.scale_projection = nn.ModuleDict()
        for period in self.periods:
            projection = nn.Linear(self.pred_len // period, self.pred_len)
            with torch.no_grad():
                projection.weight.zero_()
                projection.bias.zero_()
                for horizon_index in range(self.pred_len):
                    projection.weight[horizon_index, horizon_index // period] = 1.0
            self.scale_projection[str(period)] = projection

        self.reliability_gate = ReliabilityAwareHorizonGate(
            pred_len=self.pred_len,
            max_retrieval_weight=max_retrieval_weight,
            initial_retrieval_weight=initial_retrieval_weight,
        )

        self.register_buffer("memory_city_ids", torch.empty(0, dtype=torch.long))
        for period in self.periods:
            self.register_buffer(
                f"memory_keys_{period}",
                torch.empty(0, self.seq_len // period, dtype=torch.float32),
            )
            self.register_buffer(
                f"memory_future_{period}",
                torch.empty(0, self.pred_len // period, dtype=torch.float32),
            )

    @property
    def num_entries(self) -> int:
        return int(self.memory_city_ids.numel())

    def city_counts(self) -> Dict[str, int]:
        if self.num_entries == 0:
            return {name: 0 for name in self.source_city_names}
        counts = torch.bincount(
            self.memory_city_ids.detach().cpu(),
            minlength=len(self.source_city_names),
        ).tolist()
        return {
            name: int(counts[index])
            for index, name in enumerate(self.source_city_names)
        }

    def _dynamic_buffer_names(self):
        names = ["memory_city_ids"]
        for period in self.periods:
            names.extend([f"memory_keys_{period}", f"memory_future_{period}"])
        return names

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in self._dynamic_buffer_names():
            incoming = state_dict.get(prefix + name)
            if incoming is not None and self._buffers[name].shape != incoming.shape:
                self._buffers[name] = torch.empty_like(
                    incoming, device=self._buffers[name].device
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.no_grad()
    def build_memory(
        self,
        x_list,
        y_list,
        means_list,
        stds_list,
        city_names: Optional[Iterable[str]] = None,
        target_city: Optional[str] = None,
    ) -> int:
        """Build a balanced source-only bank from cached windows.

        x_list[city]: [samples,nodes,seq_len,channels], channel 0 is globally
                      standardized speed.
        y_list[city]: [samples,nodes,pred_len], raw speed values.
        """
        cities = list(city_names) if city_names is not None else list(self.source_city_names)
        if target_city is not None and target_city in cities:
            raise RuntimeError(
                f"Target city {target_city!r} is present in the Stage-II source bank."
            )

        device = self.memory_city_ids.device
        per_city_capacity = max(1, self.max_memory_size // len(self.source_city_names))
        all_city_ids = []
        scale_keys = {period: [] for period in self.periods}
        scale_futures = {period: [] for period in self.periods}

        for city in cities:
            if city not in self.city_to_id:
                continue
            if city not in x_list or city not in y_list:
                raise KeyError(f"Missing cached source windows for city {city!r}.")

            x_city = torch.as_tensor(x_list[city]).detach().cpu().float()
            y_city = torch.as_tensor(y_list[city]).detach().cpu().float()
            if x_city.ndim != 4 or x_city.shape[-2] != self.seq_len:
                raise ValueError(
                    f"{city}: expected x=[S,N,{self.seq_len},C], got {tuple(x_city.shape)}"
                )
            if y_city.ndim != 3 or y_city.shape[-1] != self.pred_len:
                raise ValueError(
                    f"{city}: expected y=[S,N,{self.pred_len}], got {tuple(y_city.shape)}"
                )
            if x_city.shape[:2] != y_city.shape[:2]:
                raise ValueError(f"{city}: source input and future window counts do not match.")

            history = x_city[..., 0].reshape(-1, self.seq_len)
            future_raw = y_city.reshape(-1, self.pred_len)
            city_mean = _scalar(means_list[city], torch.device("cpu"))
            city_std = _scalar(stds_list[city], torch.device("cpu")).clamp_min(1e-6)
            future_global = (future_raw - city_mean) / city_std

            history_std = history.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
            future_delta = (future_global - history[:, -1:]) / history_std

            valid = (
                torch.isfinite(history).all(dim=1)
                & torch.isfinite(future_delta).all(dim=1)
            )
            history = history[valid]
            future_delta = future_delta[valid]
            if history.shape[0] == 0:
                continue

            if history.shape[0] > per_city_capacity:
                selected = torch.linspace(
                    0, history.shape[0] - 1, steps=per_city_capacity
                ).round().long()
                history = history[selected]
                future_delta = future_delta[selected]

            history_mean = history.mean(dim=-1, keepdim=True)
            history_std = history.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
            normalized_history = (history - history_mean) / history_std

            city_id = self.city_to_id[city]
            all_city_ids.append(
                torch.full((history.shape[0],), city_id, dtype=torch.long, device=device)
            )

            for period in self.periods:
                key = _pool_sequence(normalized_history, period)
                key = key - key.mean(dim=-1, keepdim=True)
                key = F.normalize(key, p=2, dim=-1, eps=1e-8)
                future_scale = _pool_sequence(future_delta, period)
                scale_keys[period].append(key.to(device))
                scale_futures[period].append(future_scale.to(device))

        if not all_city_ids:
            raise RuntimeError(
                "No valid source windows were added to the RAFT bank. "
                "Check source_city_names and cached source data."
            )

        self.memory_city_ids = torch.cat(all_city_ids, dim=0)
        for period in self.periods:
            setattr(self, f"memory_keys_{period}", torch.cat(scale_keys[period], dim=0))
            setattr(self, f"memory_future_{period}", torch.cat(scale_futures[period], dim=0))

        return self.num_entries

    def _retrieve_scale(self, query_key: torch.Tensor, period: int):
        memory_keys = getattr(self, f"memory_keys_{period}")
        memory_future = getattr(self, f"memory_future_{period}")
        if memory_keys.shape[0] == 0:
            raise RuntimeError(f"RAFT memory for period {period} has not been built.")

        topm = min(self.topm, memory_keys.shape[0])
        retrieved_chunks = []
        confidence_chunks = []
        disagreement_chunks = []
        for start in range(0, query_key.shape[0], self.query_chunk):
            query = query_key[start : start + self.query_chunk]
            similarity = torch.matmul(query, memory_keys.transpose(0, 1))
            scores, indices = torch.topk(similarity, k=topm, dim=1)
            weights = torch.softmax(scores / self.temperature, dim=1)
            candidates = memory_future[indices]
            retrieved = (weights.unsqueeze(-1) * candidates).sum(dim=1)
            disagreement = (
                weights.unsqueeze(-1) * (candidates - retrieved.unsqueeze(1)).square()
            ).sum(dim=1).mean(dim=-1).clamp_min(0.0).sqrt()
            confidence = (weights * scores).sum(dim=1)

            retrieved_chunks.append(retrieved)
            confidence_chunks.append(confidence)
            disagreement_chunks.append(disagreement)

        return (
            torch.cat(retrieved_chunks, dim=0),
            torch.cat(confidence_chunks, dim=0),
            torch.cat(disagreement_chunks, dim=0),
        )

    @staticmethod
    def _query_stationarity(normalized_history: torch.Tensor) -> torch.Tensor:
        """SARAF-style scale-invariant local mean/variance stability score."""
        query_count, seq_len = normalized_history.shape
        window_count = 6
        if seq_len % window_count != 0:
            raise ValueError("History length must be divisible into six stationarity windows.")
        windows = normalized_history.reshape(query_count, window_count, seq_len // window_count)
        window_means = windows.mean(dim=-1)
        window_stds = windows.std(dim=-1, unbiased=False)
        mean_variation = window_means.std(dim=1, unbiased=False)
        std_variation = window_stds.std(dim=1, unbiased=False)
        global_std = normalized_history.std(dim=1, unbiased=False).clamp_min(1e-4)
        mean_score = 1.0 - (mean_variation / global_std).clamp(0.0, 1.0)
        std_score = 1.0 - (std_variation / global_std).clamp(0.0, 1.0)
        return (0.5 * mean_score + 0.5 * std_score).clamp(0.0, 1.0)

    def forward(
        self,
        history: torch.Tensor,
        base_prediction: torch.Tensor,
        means,
        stds,
    ):
        """Return fused forecast, retrieved analog forecast, and horizon gate."""
        if self.num_entries == 0:
            raise RuntimeError("RAFT source memory is empty; call build_memory() first.")
        if history.ndim != 3 or history.shape[-1] != self.seq_len:
            raise ValueError(
                f"Expected history=[B,N,{self.seq_len}], got {tuple(history.shape)}"
            )
        if base_prediction.shape != (*history.shape[:2], self.pred_len):
            raise ValueError(
                f"Base forecast shape {tuple(base_prediction.shape)} is incompatible "
                f"with history {tuple(history.shape)}."
            )

        batch_size, node_num, _ = history.shape
        flat_history = history.reshape(-1, self.seq_len)
        history_mean = flat_history.mean(dim=-1, keepdim=True)
        history_std = flat_history.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        normalized_history = (flat_history - history_mean) / history_std

        retrieved_by_scale = []
        confidence_by_scale = []
        disagreement_by_scale = []
        with torch.no_grad():
            for period in self.periods:
                query_key = _pool_sequence(normalized_history, period)
                query_key = query_key - query_key.mean(dim=-1, keepdim=True)
                query_key = F.normalize(query_key, p=2, dim=-1, eps=1e-8)
                retrieved, confidence, disagreement = self._retrieve_scale(query_key, period)
                retrieved_by_scale.append(retrieved)
                confidence_by_scale.append(confidence)
                disagreement_by_scale.append(disagreement)

        projected_scales = [
            self.scale_projection[str(period)](retrieved)
            for period, retrieved in zip(self.periods, retrieved_by_scale)
        ]
        retrieved_delta = torch.stack(projected_scales, dim=0).mean(dim=0)
        similarity_confidence = torch.stack(confidence_by_scale, dim=0).mean(dim=0)
        future_disagreement = torch.stack(disagreement_by_scale, dim=0).mean(dim=0)
        stationarity = self._query_stationarity(normalized_history)

        target_mean = _scalar(means, history.device)
        target_std = _scalar(stds, history.device).clamp_min(1e-6)
        retrieved_global = flat_history[:, -1:] + history_std * retrieved_delta
        retrieved_raw = retrieved_global * target_std + target_mean

        last_observation_raw = flat_history[:, -1:] * target_std + target_mean
        local_scale_raw = history_std * target_std
        flat_base = base_prediction.reshape(-1, self.pred_len)
        fused, gate = self.reliability_gate(
            flat_base,
            retrieved_raw,
            last_observation_raw,
            local_scale_raw,
            similarity_confidence,
            future_disagreement,
            stationarity,
        )
        return (
            fused.reshape(batch_size, node_num, self.pred_len),
            retrieved_raw.reshape(batch_size, node_num, self.pred_len),
            gate.reshape(batch_size, node_num, self.pred_len),
        )
