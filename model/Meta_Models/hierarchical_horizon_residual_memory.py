"""Hierarchical multi-scale residual memory for PMME Stage-II.

This module deliberately replaces the original single-bank / single-query /
linear-bias path with a three-level memory system:
  1) a global traffic-pattern bank,
  2) short-, medium-, and long-horizon residual banks,
  3) a bidirectional selective state-space (Mamba-style) horizon decoder.

The total memory capacity is kept close to the official PMME setting (10,000
slots). The Stage-I forecaster is frozen; all memory, retrieval, temporal
interaction, residual-generation, and gating parameters are trainable.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadMemoryReader(nn.Module):
    def __init__(self, dim=128, heads=4, dropout=0.1, query_chunk_size=512):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.query_chunk_size = int(query_chunk_size)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout), nn.LayerNorm(dim))

    def _read_context(self, q, k, v):
        scores = torch.einsum("bhd,mhd->bhm", q, k) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        context = torch.einsum("bhm,mhd->bhd", attention, v)
        return context.reshape(q.shape[0], self.dim)

    def forward(self, query, memory_keys, memory_values, return_attention=False):
        # Chunk queries to avoid materializing [B*N, heads, memory_slots] for
        # every city node at once. Checkpointing recomputes attention on backward.
        batch = query.shape[0]
        q = self.query(query).view(batch, self.heads, self.head_dim)
        k = self.key(memory_keys).view(-1, self.heads, self.head_dim)
        v = self.value(memory_values).view(-1, self.heads, self.head_dim)

        attention = None
        if return_attention:
            scores = torch.einsum("bhd,mhd->bhm", q, k) / math.sqrt(self.head_dim)
            attention = torch.softmax(scores, dim=-1)
            context = torch.einsum("bhm,mhd->bhd", attention, v).reshape(batch, self.dim)
        else:
            from torch.utils.checkpoint import checkpoint
            contexts = []
            for q_chunk in q.split(self.query_chunk_size, dim=0):
                if self.training and torch.is_grad_enabled():
                    context_chunk = checkpoint(
                        self._read_context, q_chunk, k, v, use_reentrant=False
                    )
                else:
                    context_chunk = self._read_context(q_chunk, k, v)
                contexts.append(context_chunk)
            context = torch.cat(contexts, dim=0)

        return self.output(context), attention


class SelectiveStateSpaceDirection(nn.Module):
    """Narrow input-selective state-space scan for one sequence direction."""
    def __init__(self, model_dim=256, inner_dim=64, state_dim=8, conv_kernel=3):
        super().__init__()
        self.inner_dim = int(inner_dim)
        self.state_dim = int(state_dim)
        self.dt_rank = max(1, self.inner_dim // 16)
        self.in_proj = nn.Linear(model_dim, self.inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.inner_dim, self.inner_dim, kernel_size=conv_kernel,
            groups=self.inner_dim, padding=conv_kernel - 1, bias=True
        )
        self.x_proj = nn.Linear(
            self.inner_dim, self.dt_rank + 2 * self.state_dim, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)
        base = torch.arange(1, self.state_dim + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(base.log().view(1, -1).repeat(self.inner_dim, 1))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, model_dim, bias=False)

    def forward(self, sequence):
        batch, length, _ = sequence.shape
        xz = self.in_proj(sequence)
        x, z = xz.chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[..., :length].transpose(1, 2)
        x = F.silu(x)
        params = self.x_proj(x)
        dt_low, b_all, c_all = torch.split(
            params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_low)).clamp(max=5.0)
        A = -torch.exp(self.A_log).to(dtype=x.dtype)
        state = x.new_zeros(batch, self.inner_dim, self.state_dim)
        outputs = []
        for step in range(length):
            dt = delta[:, step, :]
            b_t = b_all[:, step, :]
            c_t = c_all[:, step, :]
            decay = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            injection = (
                dt.unsqueeze(-1) * x[:, step, :].unsqueeze(-1)
                * b_t.unsqueeze(1)
            )
            state = decay * state + injection
            y_t = (state * c_t.unsqueeze(1)).sum(dim=-1)
            y_t = y_t + self.D.to(dtype=x.dtype) * x[:, step, :]
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        return self.out_proj(y * F.silu(z))


class BidirectionalHorizonMamba(nn.Module):
    """One narrow forward/backward selective-SSM block with residual fusion."""
    def __init__(self, model_dim=256, inner_dim=64, state_dim=8, dropout=0.05):
        super().__init__()
        self.forward_ssm = SelectiveStateSpaceDirection(
            model_dim, inner_dim, state_dim
        )
        self.backward_ssm = SelectiveStateSpaceDirection(
            model_dim, inner_dim, state_dim
        )
        self.fusion = nn.Linear(model_dim * 2, model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        forward_features = self.forward_ssm(sequence)
        backward_features = self.backward_ssm(
            sequence.flip(dims=[1])
        ).flip(dims=[1])
        fused = self.fusion(torch.cat((forward_features, backward_features), dim=-1))
        return self.norm(sequence + self.dropout(fused))


class MultiScaleHorizonResidualMemory(nn.Module):
    def __init__(
        self, rep_dim=128, pred_len=36, memory_size=10000,
        horizon_dim=128, decoder_dim=256, heads=4, dropout=0.1,
        max_correction_ratio=0.5, mamba_inner_dim=64, mamba_state_dim=8,
    ):
        super().__init__()
        self.rep_dim = int(rep_dim)
        self.pred_len = int(pred_len)
        self.memory_size = int(memory_size)
        self.horizon_dim = int(horizon_dim)
        self.decoder_dim = int(decoder_dim)
        self.max_correction_ratio = float(max_correction_ratio)
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
            nn.LayerNorm(self.decoder_dim), nn.GELU(), nn.Dropout(dropout),
        )
        # One narrow bidirectional Mamba-style selective SSM, sized for H=36.
        self.horizon_interaction = BidirectionalHorizonMamba(
            model_dim=self.decoder_dim, inner_dim=mamba_inner_dim,
            state_dim=mamba_state_dim, dropout=dropout,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.decoder_dim, self.decoder_dim // 2),
            nn.GELU(), nn.Linear(self.decoder_dim // 2, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim // 2),
            nn.GELU(), nn.Linear(self.decoder_dim // 2, 1),
        )
        nn.init.constant_(self.gate_head[-1].bias, -1.0)
        self.register_buffer(
            "horizon_to_band",
            torch.div(
                torch.arange(self.pred_len) * 3, self.pred_len,
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
            return_attention=return_aux,
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
                return_attention=return_aux,
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
