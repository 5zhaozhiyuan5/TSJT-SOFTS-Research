"""SOFTS traffic forecaster, adapted from the official SOFTS implementation.

Original SOFTS: https://github.com/Secilia-Cxy/SOFTS
Core operations retained: inverted embedding; STAR aggregate-redistribute;
EncoderLayer's residual, LayerNorm, Conv1d feed-forward; linear forecast head;
per-instance normalization/de-normalization.

Interface adapts existing TSJT city loaders: [B, T, N, 3] -> [B, H, N, 1].
All cities share the exact same trainable parameters; no graph adjacency, city-
specific node embeddings, or horizon-specific heads are introduced.

Temporal alignment is necessary because this dataset has both 5-minute 12/12
cities and 10-minute 6/6 cities. Source 6-step inputs are converted to the
canonical 12-step 5-minute grid using historical-only linear interpolation.
The first canonical slot (t=-55min) uses the earliest observed source value
(t=-50min). Forecasts of a 6-step 10-minute city are selected at +10,+20,...,
+60min from the 12-step canonical 5-minute output.

Official STAR has stochastic multinomial selection in train mode. Since PGrad
and gradient meta-optimisation repeatedly evaluate virtual parameter states,
``star_pooling='soft'`` uses the official SOFTS evaluation aggregation in both
train and eval to suppress Monte Carlo noise; ``'stochastic'`` reproduces the
original train-time selection and is provided for strict backbone ablation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DataEmbeddingInverted(nn.Module):
    """Official SOFTS inverted embedding (variable tokens, time as channels)."""
    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, marks: torch.Tensor | None):
        x = x.permute(0, 2, 1)
        if marks is not None:
            x = torch.cat([x, marks.permute(0, 2, 1)], dim=1)
        return self.dropout(self.value_embedding(x))


class STAR(nn.Module):
    """SOFTS STar Aggregate-Redistribute layer."""
    def __init__(self, d_model: int, d_core: int, pooling: str = 'soft'):
        super().__init__()
        if pooling not in {'soft', 'stochastic'}:
            raise ValueError(f'bad STAR pooling: {pooling!r}')
        self.pooling = pooling
        self.gen1 = nn.Linear(d_model, d_model)
        self.gen2 = nn.Linear(d_model, d_core)
        self.gen3 = nn.Linear(d_model + d_core, d_model)
        self.gen4 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        core = self.gen2(F.gelu(self.gen1(x)))
        if self.training and self.pooling == 'stochastic':
            weights = torch.softmax(core, dim=1).permute(0, 2, 1)
            indices = torch.multinomial(weights.reshape(-1, tokens), 1)
            indices = indices.view(batch, -1, 1).permute(0, 2, 1)
            core = torch.gather(core, 1, indices).repeat(1, tokens, 1)
        else:
            weights = torch.softmax(core, dim=1)
            core = (core * weights).sum(dim=1, keepdim=True).repeat(1, tokens, 1)
        return self.gen4(F.gelu(self.gen3(torch.cat([x, core], dim=-1))))


class SoftsEncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_core: int, d_ff: int,
                 dropout: float, activation: str, pooling: str):
        super().__init__()
        self.star = STAR(d_model, d_core, pooling)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.star(x)))
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class SOFTSBackbone(nn.Module):
    """Shared-node-count-agnostic SOFTS with existing TSJT trainer signature."""
    def __init__(self, *, input_dim: int = 3, canonical_steps: int = 12,
                 supported_horizons: Sequence[int] = (6, 12),
                 d_model: int = 128, d_core: int = 64, d_ff: int = 256,
                 e_layers: int = 2, dropout: float = 0.1,
                 activation: str = 'gelu', use_norm: bool = True,
                 use_time_features: bool = True, star_pooling: str = 'soft'):
        super().__init__()
        self.input_dim = int(input_dim)
        self.canonical_steps = int(canonical_steps)
        self.supported_horizons = tuple(int(x) for x in supported_horizons)
        if len(set(self.supported_horizons)) != len(self.supported_horizons):
            raise ValueError('supported_horizons must be unique')
        if not self.supported_horizons or any(x <= 0 for x in self.supported_horizons):
            raise ValueError('supported_horizons must be positive')
        if any(self.canonical_steps % h != 0 for h in self.supported_horizons):
            raise ValueError('SOFTS alignment currently requires canonical_steps divisible by every horizon')
        if self.input_dim < (3 if use_time_features else 1):
            raise ValueError('input_dim too small for chosen temporal covariates')
        self.default_horizon = self.canonical_steps
        self.use_norm = bool(use_norm)
        self.use_time_features = bool(use_time_features)
        self.embedding = DataEmbeddingInverted(self.canonical_steps, d_model, dropout)
        self.encoder = nn.ModuleList([
            SoftsEncoderLayer(d_model, d_core, d_ff, dropout, activation, star_pooling)
            for _ in range(e_layers)
        ])
        self.projection = nn.Linear(d_model, self.canonical_steps, bias=True)

    def align_history(self, x: torch.Tensor) -> torch.Tensor:
        """Historical-only 6 x 10min -> 12 x 5min conversion, no future access."""
        steps = x.shape[1]
        if steps == self.canonical_steps:
            return x
        if steps not in self.supported_horizons:
            raise ValueError(f'input history {steps} is not supported: {self.supported_horizons}')
        # Observations at (...,-50,-40,...,0) for 6x10min, target grid
        # (...,-55,-50,-45,...,0) for 12x5min. The -55 slot is edge-held.
        ratio = self.canonical_steps / steps
        positions = (torch.arange(self.canonical_steps, device=x.device, dtype=x.dtype)
                     - (self.canonical_steps - 1)) / ratio + (steps - 1)
        positions = positions.clamp(0, steps - 1)
        left = positions.floor().long()
        right = (left + 1).clamp(max=steps - 1)
        w = (positions - left).reshape(1, self.canonical_steps, *([1] * (x.ndim - 2)))
        return x.index_select(1, left) * (1 - w) + x.index_select(1, right) * w

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None,
                horizon: int | None = None, return_aux: bool = False):
        """Accept adjacency for TSJT trainer compatibility, but DO NOT use it."""
        if x.ndim != 4 or x.shape[-1] < self.input_dim:
            raise ValueError(f'expect [B,T,N,F>=input_dim], got {tuple(x.shape)}')
        original_steps = int(x.shape[1])
        if horizon is None:
            horizon = original_steps
        horizon = int(horizon)
        if horizon not in self.supported_horizons:
            raise ValueError(f'unsupported horizon {horizon}; supported: {self.supported_horizons}')
        x = self.align_history(x)
        n_nodes = x.shape[2]
        # Speed is already normalized with the per-city train-only scaler in
        # tsjt_c1's data loader. Re-centering is per-window instance norm.
        speed = x[..., 0]
        marks = (x[:, :, 0, 1:3] if self.use_time_features else None)
        if self.use_norm:
            means = speed.mean(dim=1, keepdim=True).detach()
            centered = speed - means
            stdev = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            speed = centered / stdev
        h = self.embedding(speed, marks)
        for layer in self.encoder:
            h = layer(h)
        out = self.projection(h).permute(0, 2, 1)[:, :, :n_nodes]
        if self.use_norm:
            out = out * stdev + means
        if horizon != self.canonical_steps:
            # Forecast future physical times +10,+20,...,+60 min.
            stride = self.canonical_steps // horizon
            out = out[:, stride - 1::stride, :]
        prediction = out.unsqueeze(-1)
        if return_aux:
            return {'prediction': prediction}
        return prediction


__all__ = ['SOFTSBackbone', 'STAR', 'DataEmbeddingInverted', 'SoftsEncoderLayer']
