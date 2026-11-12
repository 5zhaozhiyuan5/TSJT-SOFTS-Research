"""Target-only graph-temporal residual forecasting network for PMME Stage-II.

Stage-I remains frozen.  This module learns a target-city residual predictor
from the target support windows only.  It replaces memory retrieval with a
causal temporal encoder, target-graph propagation, and an N-HiTS-inspired
backcast/forecast decoder with hierarchical interpolation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_target_residual_parameters(model: nn.Module):
    """Freeze the Stage-I forecaster and train only the Stage-II residual net."""
    residual_net = getattr(model, "residual_net", None)
    if residual_net is None:
        raise RuntimeError("Target residual network is disabled.")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in residual_net.parameters():
        parameter.requires_grad_(True)

    trainable = [p for p in residual_net.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable Stage-II residual parameters were found.")
    return trainable


class CausalTemporalBlock(nn.Module):
    """Dilated causal temporal block shared across all target-city nodes."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.05):
        super().__init__()
        self.dilation = int(dilation)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=self.dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=self.dilation)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = 2 * self.dilation
        y = self.conv1(F.pad(x, (pad, 0)))
        y = F.gelu(self.norm1(y))
        y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.conv2(F.pad(y, (pad, 0)))
        y = self.norm2(y)
        return F.gelu(x + y)


class BackcastForecastBlock(nn.Module):
    """N-HiTS/N-BEATS-inspired block: explain history, then forecast a residual."""

    def __init__(
        self,
        history_len: int,
        pred_len: int,
        cond_dim: int,
        hidden_dim: int,
        forecast_knots: int,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.forecast_knots = int(forecast_knots)

        self.history_projection = nn.Linear(self.history_len, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim + hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.backcast_head = nn.Linear(hidden_dim, self.history_len)
        self.forecast_head = nn.Linear(hidden_dim, self.forecast_knots)

        # Keep both paths small initially so Stage-II starts close to Stage-I.
        nn.init.normal_(self.backcast_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.backcast_head.bias)
        nn.init.normal_(self.forecast_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.forecast_head.bias)

    def forward(self, history_remainder: torch.Tensor, condition: torch.Tensor):
        history_features = self.history_projection(history_remainder)
        hidden = self.mlp(torch.cat([condition, history_features], dim=-1))

        backcast = self.backcast_head(hidden)
        coefficients = self.forecast_head(hidden)
        forecast = F.interpolate(
            coefficients.unsqueeze(1),
            size=self.pred_len,
            mode="linear",
            align_corners=True,
        ).squeeze(1)
        return backcast, forecast


class TargetResidualGraphNet(nn.Module):
    """Target-city-only spatiotemporal residual predictor.

    Inputs:
      history:       [B, N, history_len], target-city RevIN-normalized history
      representation:[B, N, d_model], frozen Stage-I hidden representation
      base_prediction:[B, N, pred_len], frozen Stage-I RevIN-normalized forecast
      adjacency:     [N, N] normalized target-city adjacency

    Output:
      correction:    [B, N, pred_len], residual correction in RevIN space
    """

    def __init__(
        self,
        d_model: int = 128,
        history_len: int = 288,
        pred_len: int = 36,
        hidden_dim: int = 32,
        forecast_knots=(3, 9, 36),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.hidden_dim = int(hidden_dim)

        self.input_projection = nn.Conv1d(2, hidden_dim, kernel_size=5)
        self.temporal_blocks = nn.ModuleList(
            [CausalTemporalBlock(hidden_dim, dilation=d, dropout=dropout)
             for d in (1, 2, 4, 8)]
        )
        self.temporal_summary = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.representation_projection = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.GELU(),
        )
        self.base_projection = nn.Sequential(
            nn.Linear(self.pred_len, hidden_dim),
            nn.GELU(),
        )

        # Local, one-hop, and two-hop target-city graph context.
        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.blocks = nn.ModuleList([
            BackcastForecastBlock(
                history_len=self.history_len,
                pred_len=self.pred_len,
                cond_dim=hidden_dim,
                hidden_dim=hidden_dim,
                forecast_knots=int(knots),
                dropout=dropout,
            )
            for knots in forecast_knots
        ])

    @staticmethod
    def _graph_propagate(adjacency: torch.Tensor, features: torch.Tensor):
        # features: [B, N, C]; adjacency can be [N, N] or [B, N, N].
        if adjacency is None:
            return features
        adjacency = torch.as_tensor(
            adjacency, device=features.device, dtype=features.dtype
        )
        if adjacency.ndim == 2:
            return torch.einsum("nm,bmc->bnc", adjacency, features)
        if adjacency.ndim == 3:
            return torch.bmm(adjacency, features)
        raise ValueError(
            f"Expected adjacency [N,N] or [B,N,N], got {tuple(adjacency.shape)}"
        )

    def forward(
        self,
        history: torch.Tensor,
        representation: torch.Tensor,
        base_prediction: torch.Tensor,
        adjacency: torch.Tensor = None,
    ) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError(f"history must be [B,N,L], got {tuple(history.shape)}")
        if representation.ndim != 3 or representation.shape[-1] != self.d_model:
            raise ValueError(
                f"representation must be [B,N,{self.d_model}], "
                f"got {tuple(representation.shape)}"
            )
        if base_prediction.ndim != 3 or base_prediction.shape[-1] != self.pred_len:
            raise ValueError(
                f"base_prediction must be [B,N,{self.pred_len}], "
                f"got {tuple(base_prediction.shape)}"
            )
        if history.shape[-1] != self.history_len:
            raise ValueError(
                f"Expected history length {self.history_len}, got {history.shape[-1]}"
            )

        batch_size, node_num, _ = history.shape
        # Preserve both the absolute traffic level and its first-order change.
        difference = torch.diff(history, dim=-1, prepend=history[..., :1])
        temporal_input = torch.stack([history, difference], dim=2)
        temporal_input = temporal_input.reshape(
            batch_size * node_num, 2, self.history_len
        )
        temporal_features = self.input_projection(F.pad(temporal_input, (4, 0)))
        for block in self.temporal_blocks:
            temporal_features = block(temporal_features)

        temporal_mean = temporal_features.mean(dim=-1)
        temporal_last = temporal_features[..., -1]
        temporal = self.temporal_summary(
            torch.cat([temporal_mean, temporal_last], dim=-1)
        ).reshape(batch_size, node_num, self.hidden_dim)

        graph_one = self._graph_propagate(adjacency, temporal)
        graph_two = self._graph_propagate(adjacency, graph_one)
        representation_features = self.representation_projection(representation.detach())
        base_features = self.base_projection(base_prediction.detach())

        condition = self.condition_fusion(torch.cat(
            [temporal, graph_one, graph_two, representation_features, base_features],
            dim=-1,
        ))

        history_remainder = history.reshape(batch_size * node_num, self.history_len)
        condition = condition.reshape(batch_size * node_num, self.hidden_dim)
        correction = history_remainder.new_zeros(
            batch_size * node_num, self.pred_len
        )

        # Backcast links successively remove explainable history components;
        # forecast links add coarse-to-fine residual trajectories.
        for block in self.blocks:
            backcast, forecast = block(history_remainder, condition)
            history_remainder = history_remainder - backcast
            correction = correction + forecast

        return correction.reshape(batch_size, node_num, self.pred_len)
