"""Hierarchical multi-scale residual memory for PMME Stage-II.

This module deliberately replaces the original single-bank / single-query /
linear-bias path with a three-level memory system:
  1) a global traffic-pattern bank,
  2) short-, medium-, and long-horizon residual banks,
  3) a horizon-interaction Transformer decoder.

The total memory capacity is kept close to the official PMME setting (10,000
slots). The Stage-I forecaster is frozen; all memory, retrieval, temporal
interaction, residual-generation, and gating parameters are trainable.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadMemoryReader(nn.Module):
    def __init__(self, dim=128, heads=4, dropout=0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout), nn.LayerNorm(dim))

    def forward(self, query, memory_keys, memory_values):
        # query: [B*N,D], memory: [M,D]
        batch = query.shape[0]
        q = self.query(query).view(batch, self.heads, self.head_dim)
        k = self.key(memory_keys).view(-1, self.heads, self.head_dim)
        v = self.value(memory_values).view(-1, self.heads, self.head_dim)
        scores = torch.einsum("bhd,mhd->bhm", q, k) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        context = torch.einsum("bhm,mhd->bhd", attention, v).reshape(batch, self.dim)
        return self.output(context), attention


class MultiScaleHorizonResidualMemory(nn.Module):
    def __init__(
        self,
        rep_dim=128,
        pred_len=36,
        memory_size=10000,
        horizon_dim=128,
        decoder_dim=256,
        heads=4,
        decoder_layers=2,
        dropout=0.1,
        max_correction_ratio=0.5,
    ):
        super().__init__()
        self.rep_dim = int(rep_dim)
        self.pred_len = int(pred_len)
        self.memory_size = int(memory_size)
        self.horizon_dim = int(horizon_dim)
        self.decoder_dim = int(decoder_dim)
        self.max_correction_ratio = float(max_correction_ratio)

        # Preserve approximately the official 10k-slot memory capacity.
        self.global_slots = max(1, int(round(self.memory_size * 0.4)))
        remaining = self.memory_size - self.global_slots
        self.band_slots = max(1, remaining // 3)
        self.total_slots = self.global_slots + 3 * self.band_slots

        self.memory_keys = nn.Parameter(
            torch.randn(self.total_slots, self.rep_dim) / math.sqrt(self.rep_dim)
        )
        self.memory_values = nn.Parameter(
            torch.randn(self.total_slots, self.rep_dim) * 0.02
        )

        self.global_reader = MultiHeadMemoryReader(self.rep_dim, heads, dropout)
        self.band_readers = nn.ModuleList([
            MultiHeadMemoryReader(self.rep_dim, heads, dropout) for _ in range(3)
        ])

        self.global_context_encoder = nn.Sequential(
            nn.Linear(4, self.rep_dim), nn.GELU(),
            nn.Linear(self.rep_dim, self.rep_dim), nn.LayerNorm(self.rep_dim)
        )
        self.band_context_encoder = nn.Sequential(
            nn.Linear(4, self.rep_dim), nn.GELU(),
            nn.Linear(self.rep_dim, self.rep_dim), nn.LayerNorm(self.rep_dim)
        )
        self.band_embedding = nn.Embedding(3, self.rep_dim)
        self.horizon_embedding = nn.Embedding(self.pred_len, self.horizon_dim)

        decoder_input_dim = self.rep_dim * 4 + self.horizon_dim + 2
        self.decoder_input = nn.Sequential(
            nn.Linear(decoder_input_dim, self.decoder_dim),
            nn.LayerNorm(self.decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_dim,
            nhead=8,
            dim_feedforward=self.decoder_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.horizon_interaction = nn.TransformerEncoder(
            layer, num_layers=int(decoder_layers),
            norm=nn.LayerNorm(self.decoder_dim)
        )
        self.residual_head = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_dim, self.decoder_dim // 2),
            nn.GELU(),
            nn.Linear(self.decoder_dim // 2, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim // 2),
            nn.GELU(),
            nn.Linear(self.decoder_dim // 2, 1),
        )
        nn.init.constant_(self.gate_head[-1].bias, -1.0)

        self.register_buffer(
            "horizon_to_band",
            torch.div(
                torch.arange(self.pred_len) * 3,
                self.pred_len,
                rounding_mode="floor",
            ).clamp_max(2),
            persistent=False,
        )

    @staticmethod
    def _node_scale(node_scale, batch, nodes, reference):
        scale = torch.as_tensor(
            node_scale, device=reference.device, dtype=reference.dtype
        )
        scale = scale.squeeze()
        if scale.ndim == 0:
            scale = scale.view(1, 1, 1)
        elif scale.ndim == 1:
            if scale.numel() == nodes:
                scale = scale.view(1, nodes, 1)
            elif scale.numel() == batch:
                scale = scale.view(batch, 1, 1)
            else:
                scale = scale.mean().view(1, 1, 1)
        elif scale.ndim == 2:
            if scale.shape == (batch, nodes):
                scale = scale.unsqueeze(-1)
            elif scale.shape[0] == nodes:
                scale = scale.unsqueeze(0)
            else:
                scale = scale.reshape(1, nodes, 1)
        elif scale.ndim > 3:
            scale = scale.reshape(scale.shape[0], scale.shape[1], -1).mean(-1, keepdim=True)
        if scale.shape[0] == 1 and batch > 1:
            scale = scale.expand(batch, -1, -1)
        if scale.shape[1] == 1 and nodes > 1:
            scale = scale.expand(-1, nodes, -1)
        if scale.shape[1] != nodes:
            raise ValueError(
                f"Node scale cannot broadcast to [B={batch},N={nodes}]: {tuple(scale.shape)}"
            )
        return scale.abs().clamp_min(1e-3)

    @staticmethod
    def _trajectory_stats(values):
        # values: [BN,H], use stable summary statistics for memory queries.
        mean = values.mean(dim=-1)
        std = values.std(dim=-1, unbiased=False)
        slope = (values[:, -1] - values[:, 0]) / max(1, values.shape[-1] - 1)
        curvature = (
            values[:, 2:] - 2 * values[:, 1:-1] + values[:, :-2]
        ).abs().mean(dim=-1) if values.shape[-1] > 2 else torch.zeros_like(mean)
        return torch.stack((mean, std, slope, curvature), dim=-1)

    def forward(self, representation, base_prediction, node_scale, return_aux=False):
        if representation.ndim != 3 or base_prediction.ndim != 3:
            raise ValueError("Expected representation [B,N,D] and prediction [B,N,H].")
        batch, nodes, dim = representation.shape
        if dim != self.rep_dim or base_prediction.shape != (batch, nodes, self.pred_len):
            raise ValueError("Representation or prediction shape is incompatible.")

        scale = self._node_scale(node_scale, batch, nodes, base_prediction)
        base_scaled = base_prediction / scale
        flat_rep = representation.reshape(batch * nodes, dim)
        flat_base = base_scaled.reshape(batch * nodes, self.pred_len)

        # Level 1: global memory retrieval from the full predicted trajectory.
        global_stats = self.global_context_encoder(self._trajectory_stats(flat_base))
        global_query = flat_rep + global_stats
        global_context, global_attention = self.global_reader(
            global_query,
            self.memory_keys[:self.global_slots],
            self.memory_values[:self.global_slots],
        )

        # Level 2: separate residual memories for short / medium / long ranges.
        band_contexts = []
        band_attentions = []
        band_width = self.pred_len // 3
        for band in range(3):
            start = band * band_width
            end = self.pred_len if band == 2 else (band + 1) * band_width
            band_values = flat_base[:, start:end]
            band_stats = self.band_context_encoder(self._trajectory_stats(band_values))
            band_id = torch.full(
                (batch * nodes,), band, device=representation.device, dtype=torch.long
            )
            band_query = flat_rep + band_stats + self.band_embedding(band_id)
            memory_start = self.global_slots + band * self.band_slots
            memory_end = memory_start + self.band_slots
            context, attention = self.band_readers[band](
                band_query,
                self.memory_keys[memory_start:memory_end],
                self.memory_values[memory_start:memory_end],
            )
            band_contexts.append(context)
            band_attentions.append(attention)

        band_contexts = torch.stack(band_contexts, dim=1)  # [BN,3,D]
        horizon_ids = torch.arange(self.pred_len, device=representation.device)
        band_ids = self.horizon_to_band.to(representation.device)
        horizon_emb = self.horizon_embedding(horizon_ids)
        selected_band_context = band_contexts[:, band_ids, :]
        rep_tokens = flat_rep.unsqueeze(1).expand(-1, self.pred_len, -1)
        global_tokens = global_context.unsqueeze(1).expand(-1, self.pred_len, -1)
        base_tokens = flat_base.unsqueeze(-1)
        delta_tokens = torch.cat(
            (torch.zeros_like(base_tokens[:, :1]), base_tokens[:, 1:] - base_tokens[:, :-1]),
            dim=1,
        )
        horizon_tokens = horizon_emb.unsqueeze(0).expand(batch * nodes, -1, -1)

        # Level 3: horizon-wise interaction before residual decoding.
        decoder_tokens = torch.cat(
            (
                rep_tokens,
                global_tokens,
                selected_band_context,
                rep_tokens + selected_band_context,
                horizon_tokens,
                base_tokens,
                delta_tokens,
            ),
            dim=-1,
        )
        hidden = self.decoder_input(decoder_tokens)
        hidden = self.horizon_interaction(hidden)
        raw_residual = self.residual_head(hidden).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(hidden).squeeze(-1))
        correction = (
            self.max_correction_ratio
            * scale
            * torch.tanh(raw_residual.reshape(batch, nodes, self.pred_len))
            * gate.reshape(batch, nodes, self.pred_len)
        )

        if return_aux:
            aux = {
                "gate_mean": gate.detach().mean(),
                "global_attention_entropy": -(
                    global_attention.clamp_min(1e-9)
                    * global_attention.clamp_min(1e-9).log()
                ).sum(-1).mean().detach(),
                "band_attention_entropy": torch.stack([
                    -(a.clamp_min(1e-9) * a.clamp_min(1e-9).log()).sum(-1).mean()
                    for a in band_attentions
                ]).mean().detach(),
            }
            return correction, aux
        return correction

    def regularization_loss(self):
        # Sampled key decorrelation avoids a prohibitively large 10k x 10k Gram matrix.
        sample_count = min(128, self.total_slots)
        indices = torch.randperm(self.total_slots, device=self.memory_keys.device)[:sample_count]
        sampled = F.normalize(self.memory_keys[indices], dim=-1)
        gram = sampled @ sampled.transpose(0, 1)
        identity = torch.eye(sample_count, device=gram.device, dtype=gram.dtype)
        key_diversity = (gram - identity).square().mean()
        value_energy = self.memory_values.square().mean()
        return key_diversity + 1e-4 * value_energy
