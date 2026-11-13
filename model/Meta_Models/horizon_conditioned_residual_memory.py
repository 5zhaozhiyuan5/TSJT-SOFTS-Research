"""Horizon-conditioned trainable residual memory for PMME Stage-II.

The Stage-I forecaster is frozen. A compact memory bank learns transferable
residual patterns from source and target tasks, while horizon embeddings let
each prediction step retrieve and decode its own correction.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizonConditionedResidualMemory(nn.Module):
    def __init__(
        self,
        rep_dim: int = 128,
        pred_len: int = 36,
        memory_size: int = 256,
        horizon_dim: int = 16,
        hidden_dim: int = 64,
        max_correction_ratio: float = 0.25,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.rep_dim = int(rep_dim)
        self.pred_len = int(pred_len)
        self.memory_size = int(memory_size)
        self.horizon_dim = int(horizon_dim)
        self.max_correction_ratio = float(max_correction_ratio)

        self.memory_keys = nn.Parameter(
            torch.randn(self.memory_size, self.rep_dim) / math.sqrt(self.rep_dim)
        )
        self.memory_values = nn.Parameter(
            torch.randn(self.memory_size, self.rep_dim) * 0.02
        )
        self.query_proj = nn.Linear(self.rep_dim, self.rep_dim, bias=False)
        self.key_proj = nn.Linear(self.rep_dim, self.rep_dim, bias=False)
        self.value_proj = nn.Linear(self.rep_dim, self.rep_dim, bias=False)
        self.horizon_embedding = nn.Embedding(self.pred_len, self.horizon_dim)
        self.horizon_query = nn.Linear(self.horizon_dim, self.rep_dim, bias=False)

        decoder_input = self.rep_dim * 2 + self.horizon_dim + 1
        self.residual_decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Start conservatively; training can increase the correction when useful.
        nn.init.constant_(self.gate_decoder[-1].bias, -2.0)

    def forward(self, representation, base_prediction, node_scale):
        if representation.ndim != 3 or base_prediction.ndim != 3:
            raise ValueError("Expected representation [B,N,D] and prediction [B,N,H].")
        batch, nodes, dim = representation.shape
        if dim != self.rep_dim or base_prediction.shape != (batch, nodes, self.pred_len):
            raise ValueError("Representation or prediction shape is incompatible with HRMN.")

        scale = torch.as_tensor(
            node_scale, device=base_prediction.device, dtype=base_prediction.dtype
        )
        if scale.ndim == 1:
            scale = scale.view(1, nodes, 1)
        elif scale.ndim == 2:
            scale = scale.unsqueeze(-1)
        if scale.shape[0] == 1 and batch > 1:
            scale = scale.expand(batch, -1, -1)
        if scale.shape[1] != nodes:
            raise ValueError(
                f"Node-scale mismatch: expected {nodes} nodes, got {scale.shape}."
            )
        scale = scale.abs().clamp_min(1e-3)

        horizon_ids = torch.arange(self.pred_len, device=representation.device)
        horizon_emb = self.horizon_embedding(horizon_ids)  # [H,E]
        query = self.query_proj(representation).unsqueeze(2)
        query = query + self.horizon_query(horizon_emb).view(1, 1, self.pred_len, dim)

        keys = self.key_proj(self.memory_keys)
        values = self.value_proj(self.memory_values)
        scores = torch.einsum("bnhd,md->bnhm", query, keys) / math.sqrt(dim)
        attention = torch.softmax(scores, dim=-1)
        retrieved = torch.einsum("bnhm,md->bnhd", attention, values)

        rep_h = representation.unsqueeze(2).expand(-1, -1, self.pred_len, -1)
        horizon_h = horizon_emb.view(1, 1, self.pred_len, self.horizon_dim).expand(
            batch, nodes, -1, -1
        )
        base_scaled = (base_prediction / scale).unsqueeze(-1)
        decoder_input = torch.cat((rep_h, retrieved, horizon_h, base_scaled), dim=-1)

        raw_residual = self.residual_decoder(decoder_input).squeeze(-1)
        gate = torch.sigmoid(self.gate_decoder(decoder_input).squeeze(-1))
        correction = (
            self.max_correction_ratio
            * scale
            * torch.tanh(raw_residual)
            * gate
        )
        return correction

    def regularization_loss(self):
        # Keeps the memory values and gates from growing without bound.
        memory_penalty = self.memory_values.square().mean()
        gate_bias_penalty = self.gate_decoder[-1].bias.square().mean()
        return memory_penalty + 0.01 * gate_bias_penalty
