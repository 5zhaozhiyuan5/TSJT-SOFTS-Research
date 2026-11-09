"""SOFTS forecaster + the UNCHANGED learnable gradient-angle controller.

Replaces only the previous TSJTPaper Graph WaveNet prediction backbone.
The source interaction (TV-SGI), gradient rotation, meta-learning and
residual fusion are the original delivered implementations.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from ..pgrad.rotation import AngleController
from .softs_backbone import SOFTSBackbone


class TSJTPGrad(nn.Module):
    """Compatibility wrapper: SOFTS backbone, unchanged angle controller."""
    def __init__(
        self,
        input_dim: int = 3,
        canonical_steps: int = 12,
        supported_horizons: Sequence[int] = (6, 12),
        softs_d_model: int = 128,
        softs_d_core: int = 64,
        softs_d_ff: int = 256,
        softs_e_layers: int = 2,
        softs_use_norm: bool = True,
        softs_use_time_features: bool = True,
        softs_star_pooling: str = 'soft',
        dropout: float = 0.1,
        angle_controller_hidden: int = 16,
        angle_max: float = 0.7853981633974483,
        **_ignored: Any,
    ) -> None:
        super().__init__()
        self.backbone = SOFTSBackbone(
            input_dim=input_dim,
            canonical_steps=canonical_steps,
            supported_horizons=supported_horizons,
            d_model=softs_d_model,
            d_core=softs_d_core,
            d_ff=softs_d_ff,
            e_layers=softs_e_layers,
            dropout=dropout,
            use_norm=softs_use_norm,
            use_time_features=softs_use_time_features,
            star_pooling=softs_star_pooling,
        )
        self.angle_controller = AngleController(
            input_dim=3, hidden_dim=angle_controller_hidden, phi_max=angle_max,
        )

    @property
    def supported_horizons(self) -> tuple[int, ...]:
        return self.backbone.supported_horizons

    @property
    def default_horizon(self) -> int | None:
        return self.backbone.default_horizon

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor | None,
        horizon: int | None = None, return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        return self.backbone(x, adjacency, horizon=horizon, return_aux=return_aux)


__all__ = ['TSJTPGrad']
