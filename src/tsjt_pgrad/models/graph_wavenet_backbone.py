"""Node-count agnostic Graph WaveNet-style traffic forecasting backbone.

Temporal gated dilated convolutions, residual/skip connections and diffusion
graph convolutions. Dynamic graph supports are inferred from *historical* traffic
signals only, so no fixed city-specific adjacency or node embeddings are needed.
This is a Graph WaveNet adaptation, not a bitwise reproduction of its original
learned node-embedding adaptive adjacency.
"""
from __future__ import annotations

from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F


class GraphWaveNetBackbone(nn.Module):
    def __init__(self, *, input_dim: int = 3, canonical_steps: int = 12,
                 supported_horizons: Sequence[int] = (6, 12),
                 residual_channels: int = 32, dilation_channels: int = 32,
                 skip_channels: int = 64, end_channels: int = 128,
                 blocks: int = 2, layers: int = 2, kernel_size: int = 2,
                 diffusion_order: int = 2, dropout: float = 0.1,
                 use_reverse_support: bool = True, use_knowledge_graph: bool = False,
                 **_ignored):
        super().__init__()
        self.input_dim = int(input_dim)
        self.canonical_steps = int(canonical_steps)
        self.supported_horizons = tuple(int(h) for h in supported_horizons)
        self.default_horizon = self.canonical_steps
        if self.canonical_steps <= 0 or not self.supported_horizons or any(
                h <= 0 or self.canonical_steps % h for h in self.supported_horizons):
            raise ValueError("invalid canonical_steps / supported_horizons")
        if min(residual_channels, dilation_channels, skip_channels, end_channels) <= 0:
            raise ValueError("channel sizes must be positive")
        if blocks <= 0 or layers <= 0 or kernel_size < 2 or diffusion_order < 1:
            raise ValueError("invalid Graph WaveNet depth / kernel / diffusion order")
        self.use_reverse_support = bool(use_reverse_support)
        self.use_knowledge_graph = bool(use_knowledge_graph)
        self.diffusion_order = int(diffusion_order)
        self.dropout = float(dropout)
        self.start_conv = nn.Conv2d(input_dim, residual_channels, kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.receptive_field = 1
        support_count = 2 if self.use_reverse_support else 1
        for _ in range(blocks):
            for layer in range(layers):
                dilation = 2 ** layer
                self.receptive_field += (kernel_size - 1) * dilation
                self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                    (1, kernel_size), dilation=(1, dilation)))
                self.gate_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                    (1, kernel_size), dilation=(1, dilation)))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, (1, 1)))
                self.graph_convs.append(nn.Conv2d(
                    dilation_channels * (1 + support_count * diffusion_order),
                    residual_channels, (1, 1)))
                self.residual_convs.append(nn.Conv2d(residual_channels, residual_channels, (1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, (1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, canonical_steps, (1, 1))

    def align_history(self, x: torch.Tensor) -> torch.Tensor:
        steps = x.shape[1]
        if steps == self.canonical_steps:
            return x
        if steps not in self.supported_horizons:
            raise ValueError(f"unsupported history length: {steps}")
        ratio = self.canonical_steps / steps
        positions = (torch.arange(self.canonical_steps, device=x.device, dtype=x.dtype)
                     - self.canonical_steps + 1) / ratio + steps - 1
        positions = positions.clamp(0, steps - 1)
        left = positions.floor().long()
        right = (left + 1).clamp(max=steps - 1)
        alpha = (positions - left).reshape(1, -1, 1, 1)
        return x.index_select(1, left) * (1 - alpha) + x.index_select(1, right) * alpha

    @staticmethod
    def _historical_support(x: torch.Tensor) -> torch.Tensor:
        # [B,T,N,F] -> row-stochastic [B,N,N], no future data or node IDs.
        values = x[..., 0].transpose(1, 2)
        values = values - values.mean(dim=-1, keepdim=True)
        values = F.normalize(values, p=2, dim=-1, eps=1e-6)
        similarities = torch.matmul(values, values.transpose(-1, -2))
        return torch.softmax(similarities, dim=-1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None,
                horizon: int | None = None, return_aux: bool = False):
        if x.ndim != 4 or x.shape[-1] < self.input_dim:
            raise ValueError(f"expected [B,T,N,F>={self.input_dim}]")
        original_steps = int(x.shape[1])
        if horizon is None:
            horizon = original_steps
        if int(horizon) not in self.supported_horizons:
            raise ValueError(f"unsupported horizon: {horizon}")
        x = self.align_history(x)
        batch, _, nodes, _ = x.shape
        support = self._historical_support(x)
        if self.use_knowledge_graph and adjacency is not None:
            if adjacency.shape != (nodes, nodes):
                raise ValueError("adjacency node count mismatch")
            support = torch.softmax(adjacency.to(device=x.device, dtype=x.dtype), dim=-1)
            support = support.unsqueeze(0).expand(batch, -1, -1)
        supports = [support]
        if self.use_reverse_support:
            reverse = support.transpose(-1, -2)
            supports.append(reverse / reverse.sum(dim=-1, keepdim=True).clamp_min(1e-8))
        h = self.start_conv(x.permute(0, 3, 2, 1))
        if h.shape[-1] < self.receptive_field:
            h = F.pad(h, (self.receptive_field - h.shape[-1], 0))
        skip = None
        for filt, gate, residual_conv, skip_conv, graph_conv, bn in zip(
                self.filter_convs, self.gate_convs, self.residual_convs,
                self.skip_convs, self.graph_convs, self.bn):
            residual = h
            h = torch.tanh(filt(h)) * torch.sigmoid(gate(h))
            s = skip_conv(h)
            skip = s if skip is None else skip[..., -s.shape[-1]:] + s
            graph_inputs = [h]
            for adj in supports:
                propagated = h
                for _ in range(self.diffusion_order):
                    propagated = torch.einsum("bcnt,bnm->bcmt", propagated, adj)
                    graph_inputs.append(propagated)
            h = graph_conv(F.dropout(torch.cat(graph_inputs, dim=1),
                                      p=self.dropout, training=self.training))
            h = bn(residual_conv(h) + residual[..., -h.shape[-1]:])
        out = self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip))))[..., -1]
        out = out.permute(0, 2, 1)
        if int(horizon) != self.canonical_steps:
            stride = self.canonical_steps // int(horizon)
            out = out[:, stride - 1::stride, :]
        prediction = out.unsqueeze(-1)
        return {"prediction": prediction} if return_aux else prediction


__all__ = ["GraphWaveNetBackbone"]
