"""TSJT-PGrad with a shared, node-count-agnostic Graph WaveNet backbone.

The target-guided angle controller and downstream PGrad/rotation/fusion logic
are unchanged. Older SOFTS checkpoints are not compatible.
"""
from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import torch
from torch import nn
from ..pgrad.rotation import AngleController
from .graph_wavenet_backbone import GraphWaveNetBackbone


class TSJTPGrad(nn.Module):
    def __init__(self, input_dim: int = 3, canonical_steps: int = 12,
                 supported_horizons: Sequence[int] = (6, 12),
                 residual_channels: int = 32, dilation_channels: int = 32,
                 skip_channels: int = 64, end_channels: int = 128,
                 blocks: int = 2, layers: int = 2, kernel_size: int = 2,
                 diffusion_order: int = 2, dropout: float = 0.1,
                 use_reverse_support: bool = True,
                 use_knowledge_graph: bool = False,
                 angle_controller_hidden: int = 16,
                 angle_max: float = 0.7853981633974483,
                 **_ignored: Any) -> None:
        super().__init__()
        self.backbone = GraphWaveNetBackbone(
            input_dim=input_dim, canonical_steps=canonical_steps,
            supported_horizons=supported_horizons,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels, skip_channels=skip_channels,
            end_channels=end_channels, blocks=blocks, layers=layers,
            kernel_size=kernel_size, diffusion_order=diffusion_order,
            dropout=dropout, use_reverse_support=use_reverse_support,
            use_knowledge_graph=use_knowledge_graph)
        self.angle_controller = AngleController(
            input_dim=3, hidden_dim=angle_controller_hidden, phi_max=angle_max)

    @property
    def supported_horizons(self) -> tuple[int, ...]:
        return self.backbone.supported_horizons

    @property
    def default_horizon(self) -> int | None:
        return self.backbone.default_horizon

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None,
                horizon: int | None = None, return_aux: bool = False):
        return self.backbone(x, adjacency, horizon=horizon, return_aux=return_aux)


__all__ = ["TSJTPGrad"]
