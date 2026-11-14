"""Regime-Horizon Source Routing for cross-city Stage-I transfer.

This module deliberately does not implement PMME's DVGP-conditioned OT path.
It learns (1) a sample-wise shared/private factorization, (2) soft traffic
regime summaries, and (3) target-conditioned, horizon-specific source routing.
The router is used to weight supervised source forecasting losses rather than
to construct an OT transport plan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeHorizonSourceRouter(nn.Module):
    """Learn transferable factors and source weights for every forecast step."""

    def __init__(
        self,
        input_dim=128,
        horizon=36,
        num_regimes=4,
        hidden_dim=96,
        route_temperature=1.5,
        router_uniform_mix=0.0,
        route_logit_clip=0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.horizon = int(horizon)
        self.num_regimes = int(num_regimes)
        self.profile_dim = 9
        self.route_temperature = max(float(route_temperature), 0.5)
        self.router_uniform_mix = min(
            max(float(router_uniform_mix), 0.0), 1.0
        )
        self.route_logit_clip = max(float(route_logit_clip), 0.0)

        gate_input_dim = self.input_dim + self.profile_dim
        self.shared_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.input_dim),
            nn.Sigmoid(),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(self.profile_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.num_regimes),
        )
        self.reconstruct = nn.Sequential(
            nn.Linear(self.input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

        self.summary_dim = (
            self.input_dim * 2 + self.num_regimes + self.profile_dim
        )
        self.router = nn.Sequential(
            nn.Linear(self.summary_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, self.horizon),
        )

    def forecast_profile(self, forecasts):
        """Create a scale-stable temporal profile from [samples, horizon]."""
        if forecasts.ndim != 2 or forecasts.shape[-1] != self.horizon:
            raise ValueError(
                f"expected forecasts [samples, {self.horizon}], got {tuple(forecasts.shape)}"
            )
        mean = forecasts.mean(dim=-1, keepdim=True)
        std = forecasts.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        centered = (forecasts - mean) / std
        pooled = F.adaptive_avg_pool1d(
            centered.unsqueeze(1), 6
        ).squeeze(1)
        slope = ((forecasts[:, -1:] - forecasts[:, :1]) / std).clamp(-5.0, 5.0)
        profile = torch.cat(
            [mean, std.log(), pooled, slope],
            dim=-1,
        )
        return profile

    def encode(self, features, forecasts):
        """Return shared/private factors, regime probabilities and a summary."""
        profile = self.forecast_profile(forecasts.detach())
        gate_input = torch.cat([features, profile], dim=-1)
        shared_gate = self.shared_gate(gate_input)
        shared = shared_gate * features
        private = (1.0 - shared_gate) * features
        reconstructed = self.reconstruct(torch.cat([shared, private], dim=-1))

        regime_logits = self.regime_head(profile)
        regime_prob = torch.softmax(regime_logits, dim=-1)
        summary = torch.cat(
            [
                shared.mean(dim=0),
                shared.std(dim=0, unbiased=False),
                regime_prob.mean(dim=0),
                profile.mean(dim=0),
            ],
            dim=-1,
        )

        reconstruction = F.smooth_l1_loss(reconstructed, features.detach())
        if features.shape[0] > 1:
            shared_centered = shared - shared.mean(dim=0, keepdim=True)
            private_centered = private - private.mean(dim=0, keepdim=True)
            cross = shared_centered.transpose(0, 1).matmul(private_centered)
            cross = cross / float(features.shape[0] - 1)
            shared_scale = shared_centered.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
            private_scale = private_centered.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
            cross = cross / (shared_scale[:, None] * private_scale[None, :])
            orthogonality = cross.pow(2).mean()
        else:
            orthogonality = features.new_zeros(())
        decomposition_loss = reconstruction + 0.05 * orthogonality

        return {
            "shared": shared,
            "private": private,
            "regime_prob": regime_prob,
            "summary": summary,
            "decomposition_loss": decomposition_loss,
        }

    def route(self, target_summary, source_summary):
        """Return one routing logit for each prediction horizon."""
        difference = (target_summary - source_summary).abs()
        interaction = target_summary * source_summary
        pair = torch.cat(
            [target_summary, source_summary, difference, interaction], dim=-1
        )
        return self.router(pair)

    def route_weights(self, target_summary, source_summaries):
        """Normalize source logits across sources for every horizon."""
        raw_logits = torch.stack(
            [self.route(target_summary, summary) for summary in source_summaries],
            dim=0,
        )
        # The unconstrained router can increase logits without improving the
        # source task objective, eventually turning the source selector into a
        # one-city hard gate. Optional tanh clipping keeps the routing signal
        # differentiable while preserving the non-OT, horizon-wise design.
        if self.route_logit_clip > 0.0:
            logits = self.route_logit_clip * torch.tanh(
                raw_logits / self.route_logit_clip
            )
        else:
            logits = raw_logits
        weights = torch.softmax(logits / self.route_temperature, dim=0)
        if self.router_uniform_mix > 0.0:
            uniform = torch.full_like(weights, 1.0 / weights.shape[0])
            weights = (
                (1.0 - self.router_uniform_mix) * weights
                + self.router_uniform_mix * uniform
            )
        return weights, logits

    @staticmethod
    def symmetric_kl(prob_a, prob_b):
        """Symmetric KL between two probability vectors."""
        prob_a = prob_a.clamp_min(1e-6)
        prob_b = prob_b.clamp_min(1e-6)
        return 0.5 * (
            F.kl_div(prob_a.log(), prob_b, reduction="batchmean")
            + F.kl_div(prob_b.log(), prob_a, reduction="batchmean")
        )
