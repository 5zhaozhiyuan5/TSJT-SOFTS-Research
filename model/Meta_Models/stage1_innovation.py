"""Stage-I-only innovation for the PMME research branch.

The module combines three ideas from the related literature without copying a
single existing method:

* frequency-shape supervision inspired by frequency-aware pre-training;
* a small multi-scale prototype bank for target-conditioned source routing;
* conflict-aware source/target gradient aggregation.

The wrapped SOFTS backbone keeps the original PMME tensor contract.  The
router, prototype bank, and auxiliary losses are Stage-I training machinery;
the official PMME Stage-II only receives ``model.*`` backbone parameters.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyPrototypeStage1(nn.Module):
    """Train the original SOFTS backbone with a non-OT transfer objective."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 128,
        horizon: int = 36,
        prototype_count: int = 16,
        route_temperature: float = 2.0,
        uniform_route_mix: float = 0.35,
        target_weight: float = 0.10,
        frequency_weight: float = 0.05,
        trend_weight: float = 0.03,
        prototype_weight: float = 0.01,
        route_smooth_weight: float = 0.05,
        grad_clip: float = 5.0,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.model = backbone
        self.feature_dim = int(feature_dim)
        self.horizon = int(horizon)
        self.route_temperature = max(float(route_temperature), 0.5)
        self.uniform_route_mix = min(max(float(uniform_route_mix), 0.0), 1.0)
        self.target_weight = max(float(target_weight), 0.0)
        self.frequency_weight = max(float(frequency_weight), 0.0)
        self.trend_weight = max(float(trend_weight), 0.0)
        self.prototype_weight = max(float(prototype_weight), 0.0)
        self.route_smooth_weight = max(float(route_smooth_weight), 0.0)
        self.grad_clip = max(float(grad_clip), 0.0)

        descriptor_dim = self.feature_dim * 2 + 8
        self.descriptor = nn.Sequential(
            nn.Linear(descriptor_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
        )
        self.route_head = nn.Sequential(
            nn.Linear(32 * 4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, self.horizon),
        )
        self.prototypes = nn.Parameter(
            torch.randn(int(prototype_count), self.feature_dim) * 0.02
        )
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(learning_rate), weight_decay=1e-2
        )
        self.loss = nn.SmoothL1Loss(reduction="mean", beta=5e-4)
        self.last_diagnostics: Dict[str, object] = {}

    @staticmethod
    def _frequency_profile(forecast: torch.Tensor) -> torch.Tensor:
        """Return eight normalized log-power bands for [samples, horizon]."""
        leading_shape = forecast.shape[:-1]
        flat_forecast = forecast.reshape(-1, forecast.shape[-1])
        centered = flat_forecast - flat_forecast.mean(dim=-1, keepdim=True)
        power = torch.fft.rfft(centered, dim=-1).abs()[..., 1:]
        bands = F.adaptive_avg_pool1d(power.unsqueeze(1), 8).squeeze(1)
        bands = torch.log1p(bands)
        bands = bands / bands.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return bands.reshape(*leading_shape, 8)

    @staticmethod
    def _trend_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape[-1] < 2:
            return prediction.new_zeros(())
        return F.smooth_l1_loss(
            prediction[..., 1:] - prediction[..., :-1],
            target[..., 1:] - target[..., :-1],
            beta=5e-4,
        )

    def _descriptor(
        self, representation: torch.Tensor, forecast: torch.Tensor
    ) -> torch.Tensor:
        representation = representation.reshape(-1, self.feature_dim)
        forecast = forecast.reshape(-1, self.horizon)
        spectral = self._frequency_profile(forecast).mean(dim=0)
        summary = torch.cat(
            [
                representation.mean(dim=0),
                representation.std(dim=0, unbiased=False),
                spectral,
            ],
            dim=-1,
        )
        return self.descriptor(summary.unsqueeze(0))

    def _prototype_loss(self, representation: torch.Tensor) -> torch.Tensor:
        features = F.normalize(
            representation.reshape(-1, self.feature_dim), dim=-1
        )
        prototypes = F.normalize(self.prototypes, dim=-1)
        assignment = torch.softmax(
            features.matmul(prototypes.t()) / 0.2, dim=-1
        )
        reconstruction = assignment.matmul(self.prototypes)
        reconstruction = F.normalize(reconstruction, dim=-1)
        return (1.0 - (reconstruction * features.detach()).sum(dim=-1)).mean()

    def _route_weights(
        self, target_descriptor: torch.Tensor, source_descriptors: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = []
        for source_descriptor in source_descriptors:
            difference = (target_descriptor - source_descriptor).abs()
            interaction = target_descriptor * source_descriptor
            pair = torch.cat(
                [target_descriptor, source_descriptor, difference, interaction], dim=-1
            )
            logits.append(self.route_head(pair))
        logits = torch.stack(logits, dim=0)
        weights = torch.softmax(logits / self.route_temperature, dim=0)
        if self.uniform_route_mix > 0.0:
            uniform = torch.full_like(weights, 1.0 / weights.shape[0])
            weights = (1.0 - self.uniform_route_mix) * weights + (
                self.uniform_route_mix * uniform
            )
        return weights, logits

    @staticmethod
    def _flat_gradients(
        parameters: Iterable[nn.Parameter],
        loss: torch.Tensor,
        retain_graph: bool,
    ) -> List[torch.Tensor | None]:
        parameters = list(parameters)
        if not loss.requires_grad:
            return [None for _ in parameters]
        return list(
            torch.autograd.grad(
                loss,
                parameters,
                retain_graph=retain_graph,
                allow_unused=True,
            )
        )

    def _write_conflict_aware_gradients(
        self,
        parameters: Sequence[nn.Parameter],
        domain_gradients: Sequence[torch.Tensor | None],
        target_gradients: Sequence[torch.Tensor | None],
    ) -> float:
        conflict_count = 0
        for parameter, domain_gradient, target_gradient in zip(
            parameters, domain_gradients, target_gradients
        ):
            if domain_gradient is None and target_gradient is None:
                parameter.grad = None
                continue
            domain_gradient = (
                torch.zeros_like(parameter)
                if domain_gradient is None
                else domain_gradient
            )
            target_gradient = (
                torch.zeros_like(parameter)
                if target_gradient is None
                else target_gradient
            )
            dot = torch.sum(domain_gradient * target_gradient)
            target_norm = torch.sum(target_gradient.square()).clamp_min(1e-8)
            if dot.item() < 0.0:
                domain_gradient = domain_gradient - (dot / target_norm) * target_gradient
                conflict_count += 1
            parameter.grad = domain_gradient + self.target_weight * target_gradient
        return float(conflict_count)

    def train_step(self, batches: Sequence[object], epoch: int) -> Dict[str, float]:
        if len(batches) < 2:
            raise ValueError("frequency-prototype Stage-I needs one target and one source")

        records = []
        for batch in batches:
            output, target, representation, _target_unscale, _memory = self.model(
                batch, stage="train"
            )
            records.append(
                {
                    "output": output,
                    "target": target,
                    "representation": representation,
                }
            )

        target_record = records[0]
        source_records = records[1:]
        target_descriptor = self._descriptor(
            target_record["representation"], target_record["target"]
        )
        source_descriptors = [
            self._descriptor(item["representation"], item["target"])
            for item in source_records
        ]
        route_weights, route_logits = self._route_weights(
            target_descriptor, source_descriptors
        )

        horizon_losses = []
        source_frequency_losses = []
        source_trend_losses = []
        source_prototype_losses = []
        source_task_losses = []
        for item in source_records:
            pointwise = F.smooth_l1_loss(
                item["output"], item["target"], reduction="none", beta=5e-4
            )
            horizon_losses.append(pointwise.mean(dim=(0, 1)))
            source_task_losses.append(pointwise.mean())
            source_frequency_losses.append(
                F.smooth_l1_loss(
                    self._frequency_profile(item["output"]),
                    self._frequency_profile(item["target"]).detach(),
                )
            )
            source_trend_losses.append(
                self._trend_loss(item["output"], item["target"])
            )
            source_prototype_losses.append(
                self._prototype_loss(item["representation"])
            )

        horizon_losses = torch.stack(horizon_losses, dim=0)
        source_loss = len(source_records) * (route_weights.squeeze(1) * horizon_losses).mean()
        source_frequency_loss = torch.stack(source_frequency_losses).mean()
        source_trend_loss = torch.stack(source_trend_losses).mean()
        prototype_loss = torch.stack(source_prototype_losses).mean()
        if route_weights.shape[-1] > 1:
            route_smoothness = (
                route_weights[..., 1:] - route_weights[..., :-1]
            ).square().mean()
        else:
            route_smoothness = route_weights.new_zeros(())

        target_loss = self.loss(
            target_record["output"], target_record["target"]
        )
        target_frequency_loss = F.smooth_l1_loss(
            self._frequency_profile(target_record["output"]),
            self._frequency_profile(target_record["target"]).detach(),
        )
        target_trend_loss = self._trend_loss(
            target_record["output"], target_record["target"]
        )

        domain_loss = (
            source_loss
            + self.frequency_weight * source_frequency_loss
            + self.trend_weight * source_trend_loss
            + self.prototype_weight * prototype_loss
            + self.route_smooth_weight * route_smoothness
        )
        target_objective = (
            target_loss
            + self.frequency_weight * target_frequency_loss
            + self.trend_weight * target_trend_loss
        )

        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        self.optimizer.zero_grad(set_to_none=True)
        domain_gradients = self._flat_gradients(parameters, domain_loss, retain_graph=True)
        target_gradients = self._flat_gradients(parameters, target_objective, retain_graph=False)
        conflicts = self._write_conflict_aware_gradients(
            parameters, domain_gradients, target_gradients
        )
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            source_mae = torch.stack(
                [torch.mean(torch.abs(item["output"] - item["target"])) for item in source_records]
            ).mean()
            mean_weights = route_weights.squeeze(1).mean(dim=-1)
            self.last_diagnostics = {
                "epoch": int(epoch),
                "domain_loss": float(domain_loss.detach().cpu()),
                "target_loss": float(target_objective.detach().cpu()),
                "source_mae": float(source_mae.detach().cpu()),
                "route_entropy": float(
                    (-(route_weights.clamp_min(1e-8) * route_weights.clamp_min(1e-8).log())
                     .sum(dim=0).mean()).detach().cpu()
                ),
                "route_weights": mean_weights.detach().cpu().tolist(),
                "route_smoothness": float(route_smoothness.detach().cpu()),
                "frequency_loss": float(source_frequency_loss.detach().cpu()),
                "prototype_loss": float(prototype_loss.detach().cpu()),
                "gradient_conflicts": conflicts,
            }
        return {
            "loss": float((domain_loss + self.target_weight * target_objective).detach().cpu()),
            "source_mae": float(source_mae.detach().cpu()),
            "target_mae": float(torch.mean(torch.abs(target_record["output"] - target_record["target"])).detach().cpu()),
        }
