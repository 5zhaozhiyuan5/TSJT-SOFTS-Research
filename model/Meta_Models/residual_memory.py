import torch
import torch.nn as nn
import torch.nn.functional as F


def should_skip_stage2_optimizer_step(memory_was_empty, memory_entries):
    """Skip the first Stage-II optimizer step while the source residual bank warms up.

    The first forward pass is computed with an empty bank, so its detached
    backbone output has no trainable path. Source residuals are inserted after
    that forward pass; the already-computed loss must not be backpropagated.
    """
    if not memory_was_empty:
        return False
    if int(memory_entries) <= 0:
        raise RuntimeError(
            "Stage-II residual-memory warm-up failed: no source-city residuals "
            "were inserted. Check source_city_names, data_name, and source batches."
        )
    return True


class StationarityAwareResidualMemory(nn.Module):
    """
    Cross-city residual memory with SARAF-inspired retrieval.

    Keys are normalized historical speed windows; values are forecasting
    residuals from the frozen PMME backbone. The bank is updated online from
    source-city training batches only.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        d_model,
        max_memory_size,
        source_city_names,
        topk=8,
        candidate_pool=50,
        overlap_window=27,
        query_chunk=256,
        key_dim=32,
        sigma_min=0.05,
        sigma_max=0.30,
        lambda_min=0.30,
        lambda_max=0.90,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)
        self.max_memory_size = max(1, int(max_memory_size))
        self.source_city_names = list(source_city_names)
        self.city_to_id = {name: idx for idx, name in enumerate(self.source_city_names)}
        self.topk = max(1, int(topk))
        self.candidate_pool = max(self.topk, int(candidate_pool))
        self.overlap_window = max(0, int(overlap_window))
        self.query_chunk = max(1, int(query_chunk))
        self.key_dim = int(key_dim)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)

        self.residual_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.pred_len),
            nn.Sigmoid(),
        )
        # Start conservatively: retrieved residuals should not overwhelm PMME.
        nn.init.constant_(self.residual_gate[-2].bias, -2.0)

        self.register_buffer("memory_keys", torch.empty(0, self.key_dim))
        self.register_buffer("memory_values", torch.empty(0, self.pred_len))
        self.register_buffer("memory_city_ids", torch.empty(0, dtype=torch.long))
        self.register_buffer("memory_sample_ids", torch.empty(0, dtype=torch.long))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        # The online bank grows dynamically, so resize registered buffers
        # before PyTorch copies a saved Stage-II checkpoint into them.
        for name in (
            "memory_keys", "memory_values", "memory_city_ids", "memory_sample_ids"
        ):
            incoming = state_dict.get(prefix + name)
            if incoming is not None and self._buffers[name].shape != incoming.shape:
                self._buffers[name] = torch.empty_like(
                    incoming, device=self._buffers[name].device
                )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    @property
    def num_entries(self):
        return int(self.memory_keys.shape[0])

    def _encode_history(self, history):
        """Multi-scale pooled representation of [Q, seq_len] histories."""
        history = history.reshape(-1, self.seq_len)
        mean = history.mean(dim=-1, keepdim=True)
        std = history.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        normalized = (history - mean) / std
        signal = normalized.unsqueeze(1)

        scale_1 = F.adaptive_avg_pool1d(signal, self.key_dim)
        scale_2 = F.adaptive_avg_pool1d(signal, max(1, self.key_dim // 2))
        scale_2 = F.interpolate(scale_2, size=self.key_dim, mode="linear", align_corners=False)
        scale_3 = F.adaptive_avg_pool1d(signal, max(1, self.key_dim // 4))
        scale_3 = F.interpolate(scale_3, size=self.key_dim, mode="linear", align_corners=False)

        encoded = ((scale_1 + scale_2 + scale_3) / 3.0).squeeze(1)
        encoded = encoded - encoded.mean(dim=-1, keepdim=True)
        return F.normalize(encoded, p=2, dim=-1, eps=1e-8)

    def _stationarity_score(self, history):
        """Per-query stationarity from rolling-window mean/variance changes."""
        history = history.reshape(-1, self.seq_len)
        n_windows = 6
        window_size = self.seq_len // n_windows
        usable = history[:, :window_size * n_windows]
        windows = usable.reshape(-1, n_windows, window_size)

        window_means = windows.mean(dim=-1)
        window_stds = windows.std(dim=-1, unbiased=False)
        global_std = history.std(dim=-1, unbiased=False).clamp_min(1e-6)

        mean_variation = window_means.std(dim=1, unbiased=False)
        std_variation = window_stds.std(dim=1, unbiased=False)
        mean_stability = 1.0 - (mean_variation / global_std).clamp(0.0, 1.0)
        std_stability = 1.0 - (std_variation / global_std).clamp(0.0, 1.0)
        return (0.5 * mean_stability + 0.5 * std_stability).clamp(0.0, 1.0)

    @torch.no_grad()
    def update(self, data_i, base_prediction):
        """Insert source-domain residuals after the current batch has queried the bank."""
        if not self.source_city_names:
            return
        city_name = getattr(data_i, "data_name", None)
        if city_name not in self.city_to_id:
            # The target city is deliberately never inserted into the source bank.
            return

        device = self.memory_keys.device
        x = torch.as_tensor(data_i.x, dtype=torch.float32, device=device)
        y = torch.as_tensor(data_i.y, dtype=torch.float32, device=device)
        base_prediction = base_prediction.detach().to(device=device, dtype=torch.float32)
        if x.ndim != 4 or y.ndim != 3 or base_prediction.shape != y.shape:
            raise ValueError(
                "Residual memory expects x=[B,N,L,C], y/base_prediction=[B,N,H]; "
                f"received x={tuple(x.shape)}, y={tuple(y.shape)}, "
                f"base={tuple(base_prediction.shape)}"
            )

        batch_size, node_num, _, _ = x.shape
        history = x[..., 0].reshape(-1, self.seq_len)
        keys = self._encode_history(history)
        residuals = (y - base_prediction).reshape(-1, self.pred_len)

        sample_ids = getattr(data_i, "sample_index", None)
        if sample_ids is None:
            sample_ids = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        else:
            sample_ids = torch.as_tensor(sample_ids, dtype=torch.long, device=device).reshape(-1)
            if sample_ids.numel() != batch_size:
                sample_ids = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        sample_ids = sample_ids.repeat_interleave(node_num)

        valid = torch.isfinite(keys).all(dim=1) & torch.isfinite(residuals).all(dim=1)
        keys = keys[valid].detach()
        residuals = residuals[valid].detach()
        sample_ids = sample_ids[valid]
        if keys.numel() == 0:
            return

        city_id = self.city_to_id[city_name]
        city_ids = torch.full((keys.shape[0],), city_id, dtype=torch.long, device=device)
        per_city_capacity = max(1, self.max_memory_size // max(1, len(self.source_city_names)))
        incoming_capacity = max(1, per_city_capacity // 2)

        # Keep a balanced, slowly refreshed reservoir for each source city.
        if keys.shape[0] > incoming_capacity:
            chosen = torch.randperm(keys.shape[0], device=device)[:incoming_capacity]
            keys, residuals, sample_ids = keys[chosen], residuals[chosen], sample_ids[chosen]

        other_mask = self.memory_city_ids != city_id
        old_city_mask = self.memory_city_ids == city_id
        city_keys = torch.cat([self.memory_keys[old_city_mask], keys], dim=0)
        city_values = torch.cat([self.memory_values[old_city_mask], residuals], dim=0)
        city_samples = torch.cat([self.memory_sample_ids[old_city_mask], sample_ids], dim=0)
        city_ids_all = torch.full((city_keys.shape[0],), city_id, dtype=torch.long, device=device)

        if city_keys.shape[0] > per_city_capacity:
            keep = torch.randperm(city_keys.shape[0], device=device)[:per_city_capacity]
            city_keys = city_keys[keep]
            city_values = city_values[keep]
            city_samples = city_samples[keep]
            city_ids_all = city_ids_all[keep]

        self.memory_keys = torch.cat([self.memory_keys[other_mask], city_keys], dim=0)
        self.memory_values = torch.cat([self.memory_values[other_mask], city_values], dim=0)
        self.memory_city_ids = torch.cat([self.memory_city_ids[other_mask], city_ids_all], dim=0)
        self.memory_sample_ids = torch.cat([self.memory_sample_ids[other_mask], city_samples], dim=0)

    def retrieve(self, history, representation, data_i):
        """
        Retrieve diverse historical residuals for each node.
        history: [B,N,L], representation: [B,N,d_model].
        """
        batch_size, node_num, seq_len = history.shape
        device = history.device
        query_count = batch_size * node_num
        if self.num_entries == 0:
            zeros = torch.zeros(batch_size, node_num, self.pred_len, device=device, dtype=history.dtype)
            return zeros, zeros

        flat_history = history.reshape(query_count, seq_len)
        query_keys = self._encode_history(flat_history)
        stationarity = self._stationarity_score(flat_history)

        city_name = getattr(data_i, "data_name", None)
        query_city_id = self.city_to_id.get(city_name, -1)
        query_city_ids = torch.full((batch_size,), query_city_id, dtype=torch.long, device=device)
        sample_ids = getattr(data_i, "sample_index", None)
        if sample_ids is None:
            sample_ids = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        else:
            sample_ids = torch.as_tensor(sample_ids, dtype=torch.long, device=device).reshape(-1)
            if sample_ids.numel() != batch_size:
                sample_ids = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        query_city_ids = query_city_ids.repeat_interleave(node_num)
        query_sample_ids = sample_ids.repeat_interleave(node_num)

        retrieved_chunks = []
        with torch.no_grad():
            for start in range(0, query_count, self.query_chunk):
                end = min(start + self.query_chunk, query_count)
                q = query_keys[start:end]
                raw_similarity = torch.matmul(q, self.memory_keys.t())
                similarity = raw_similarity.clone()

                same_city = query_city_ids[start:end, None] == self.memory_city_ids[None, :]
                nearby_window = (
                    (query_sample_ids[start:end, None] >= 0)
                    & (self.memory_sample_ids[None, :] >= 0)
                    & (
                        (query_sample_ids[start:end, None] - self.memory_sample_ids[None, :]).abs()
                        <= self.overlap_window
                    )
                )
                similarity = similarity.masked_fill(same_city & nearby_window, float("-inf"))

                pool = min(self.candidate_pool, self.num_entries)
                k = min(self.topk, pool)
                valid_counts = torch.isfinite(similarity).sum(dim=1)
                fallback_rows = valid_counts < k
                if fallback_rows.any():
                    similarity[fallback_rows] = raw_similarity[fallback_rows]

                candidate_scores, candidate_indices = torch.topk(similarity, k=pool, dim=1)
                candidate_valid = torch.isfinite(candidate_scores)
                candidate_keys = self.memory_keys[candidate_indices]
                pair_similarity = torch.bmm(
                    candidate_keys, candidate_keys.transpose(1, 2)
                ).clamp(-1.0, 1.0)
                pair_relevance = (pair_similarity + 1.0) * 0.5
                relevance = ((candidate_scores + 1.0) * 0.5).clamp(0.0, 1.0)

                query_stationarity = stationarity[start:end]
                balance = self.lambda_min + query_stationarity * (
                    self.lambda_max - self.lambda_min
                )
                selected_mask = torch.zeros_like(relevance, dtype=torch.bool)
                max_diversity = torch.zeros_like(relevance)
                selected_positions = []

                for _ in range(k):
                    mmr_score = (
                        balance[:, None] * relevance
                        - (1.0 - balance[:, None]) * max_diversity
                    )
                    mmr_score = mmr_score.masked_fill(
                        selected_mask | ~candidate_valid, float("-inf")
                    )
                    chosen = mmr_score.argmax(dim=1)
                    selected_positions.append(chosen)
                    selected_mask.scatter_(1, chosen[:, None], True)
                    chosen_pair_similarity = pair_relevance.gather(
                        2, chosen[:, None, None].expand(-1, pool, 1)
                    ).squeeze(-1)
                    max_diversity = torch.maximum(max_diversity, chosen_pair_similarity)

                selected_positions = torch.stack(selected_positions, dim=1)
                selected_indices = candidate_indices.gather(1, selected_positions)
                selected_relevance = relevance.gather(1, selected_positions)

                sigma = self.sigma_min + (1.0 - query_stationarity) * (
                    self.sigma_max - self.sigma_min
                )
                distance = (1.0 - selected_relevance).clamp_min(0.0)
                weights = torch.exp(
                    -(distance ** 2) / (2.0 * sigma[:, None].clamp_min(1e-4) ** 2)
                )
                weight_sum = weights.sum(dim=1, keepdim=True)
                normalized_weights = weights / weight_sum.clamp_min(1e-8)
                fallback_weights = torch.softmax(
                    selected_relevance / sigma[:, None].clamp_min(1e-4), dim=1
                )
                weights = torch.where(weight_sum > 1e-8, normalized_weights, fallback_weights)
                values = self.memory_values[selected_indices]
                retrieved = (weights.unsqueeze(-1) * values).sum(dim=1)
                retrieved_chunks.append(retrieved)

            retrieved_values = torch.cat(retrieved_chunks, dim=0)

        gate = self.residual_gate(representation.reshape(query_count, self.d_model))
        correction = retrieved_values * gate
        return (
            correction.reshape(batch_size, node_num, self.pred_len),
            retrieved_values.reshape(batch_size, node_num, self.pred_len),
        )
