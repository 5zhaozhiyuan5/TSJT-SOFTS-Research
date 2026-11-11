"""Cross-city multi-scale retrieval adapter based on RAFT (ICML 2025).

The source bank stores normalized historical windows and their future trajectories.
It does not store forecasting residuals. Target-city windows query the source-only
bank; retrieved future trajectories are fused with a frozen SOFTS prediction.
"""
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
    """Freeze the forecasting backbone and optimize only the RAFT adapter."""
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


class MultiScaleRetrievalAdapter(nn.Module):
    """RAFT-style multi-scale analog retrieval adapted to cross-city traffic.

    Each source memory item contains:
      * a normalized historical speed window, represented at multiple periods;
      * the future trajectory relative to the source window's last value and scale.

    Query histories are normalized independently, so sensors and cities with
    different speed levels can share a retrieval space. The retrieved future
    deltas are mapped back to the target query's scale and fused with the
    frozen SOFTS forecast through RAFT's forecast/retrieval linear fusion idea.
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
            # Stable nearest-repeat initialization; the layer remains learnable.
            with torch.no_grad():
                projection.weight.zero_()
                projection.bias.zero_()
                for horizon_index in range(self.pred_len):
                    projection.weight[horizon_index, horizon_index // period] = 1.0
            self.scale_projection[str(period)] = projection

        # RAFT combines the base forecast and retrieved forecast with a learned
        # linear fusion. Initialize as an exact identity on the base forecast,
        # so an untrained adapter cannot damage the PMME Stage-I prediction.
        self.fusion = nn.Linear(2 * self.pred_len, self.pred_len)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.bias.zero_()
            self.fusion.weight[:, : self.pred_len].copy_(
                torch.eye(self.pred_len, dtype=self.fusion.weight.dtype)
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
        return {name: int(counts[index]) for index, name in enumerate(self.source_city_names)}

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
        # Memory buffers are populated dynamically from the source datasets.
        # Resize them before loading a saved Stage-II checkpoint.
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
        """Build a balanced source-only bank from the dataset's cached windows.

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

            history_mean = history.mean(dim=-1, keepdim=True)
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

            # Uniformly cover the chronological/node index range instead of
            # keeping only the earliest windows or refreshing a random reservoir.
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
            setattr(
                self,
                f"memory_keys_{period}",
                torch.cat(scale_keys[period], dim=0),
            )
            setattr(
                self,
                f"memory_future_{period}",
                torch.cat(scale_futures[period], dim=0),
            )

        return self.num_entries

    def _retrieve_scale(self, query_key: torch.Tensor, period: int) -> torch.Tensor:
        memory_keys = getattr(self, f"memory_keys_{period}")
        memory_future = getattr(self, f"memory_future_{period}")
        if memory_keys.shape[0] == 0:
            raise RuntimeError(f"RAFT memory for period {period} has not been built.")

        topm = min(self.topm, memory_keys.shape[0])
        retrieved_chunks = []
        for start in range(0, query_key.shape[0], self.query_chunk):
            query = query_key[start : start + self.query_chunk]
            similarity = torch.matmul(query, memory_keys.transpose(0, 1))
            scores, indices = torch.topk(similarity, k=topm, dim=1)
            weights = torch.softmax(scores / self.temperature, dim=1)
            candidates = memory_future[indices]
            retrieved = (weights.unsqueeze(-1) * candidates).sum(dim=1)
            retrieved_chunks.append(retrieved)
        return torch.cat(retrieved_chunks, dim=0)

    def forward(
        self,
        history: torch.Tensor,
        base_prediction: torch.Tensor,
        means,
        stds,
    ):
        """Return fused forecast and the retrieved analog forecast.

        history:         [B,N,L], globally standardized target-city speed.
        base_prediction: [B,N,H], frozen SOFTS forecast in raw speed units.
        """
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
        with torch.no_grad():
            for period in self.periods:
                query_key = _pool_sequence(normalized_history, period)
                query_key = query_key - query_key.mean(dim=-1, keepdim=True)
                query_key = F.normalize(query_key, p=2, dim=-1, eps=1e-8)
                retrieved_by_scale.append(self._retrieve_scale(query_key, period))

        projected_scales = [
            self.scale_projection[str(period)](retrieved)
            for period, retrieved in zip(self.periods, retrieved_by_scale)
        ]
        retrieved_delta = torch.stack(projected_scales, dim=0).mean(dim=0)

        target_mean = _scalar(means, history.device)
        target_std = _scalar(stds, history.device).clamp_min(1e-6)
        retrieved_global = flat_history[:, -1:] + history_std * retrieved_delta
        retrieved_raw = retrieved_global * target_std + target_mean
        retrieved_raw = retrieved_raw.reshape(batch_size, node_num, self.pred_len)

        fused = self.fusion(
            torch.cat([base_prediction, retrieved_raw], dim=-1)
        )
        return fused, retrieved_raw
