"""City-pattern predictive-reliability constraints for DPM-V2 OT.

This component is deliberately separate from F2 and from DPM-V2's pair-cost
attention. F2 still produces the prior source marginal alpha. DPM-V2 still
produces the pairwise ground-cost correction. CPRC adds a city-conditioned
pattern reliability table and uses it to constrain source row mass and the
effective transport budget contributed by each source city.

The shared pattern prototypes remain those of SourceEvolutionPatternBank.
Only the reliability statistics are city-conditioned:
    Q_city[c, horizon_band, pattern].

No target validation/test labels are used. The reliability snapshots and
online updates use source training forecasts and source training labels only.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch


class CityPatternReliabilityConstraint:
    """City x horizon-band x pattern predictive-reliability controller."""

    def __init__(
        self,
        shrinkage: float = 8.0,
        ema: float = 0.05,
        sample_power: float = 1.0,
        city_power: float = 1.0,
        ratio_min: float = 0.25,
        ratio_max: float = 4.0,
        city_weight_min: float = 0.50,
        city_weight_max: float = 1.50,
    ):
        values = (shrinkage, ema, sample_power, city_power,
                  ratio_min, ratio_max, city_weight_min, city_weight_max)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("CPRC parameters must be finite")
        if (shrinkage <= 0 or not 0 < ema <= 1 or
                sample_power <= 0 or city_power <= 0 or
                not 0 < ratio_min < 1 < ratio_max or
                not 0 < city_weight_min <= 1 <= city_weight_max):
            raise ValueError("Invalid CPRC parameter range")
        self.shrinkage = float(shrinkage)
        self.ema = float(ema)
        self.sample_power = float(sample_power)
        self.city_power = float(city_power)
        self.ratio_min = float(ratio_min)
        self.ratio_max = float(ratio_max)
        self.city_weight_min = float(city_weight_min)
        self.city_weight_max = float(city_weight_max)

        # city -> list[(response [n,3,3], quality [n,3])]
        self._snapshots = defaultdict(list)
        self.city_quality = {}
        self.city_occupancy = {}
        self.city_support = {}
        self.city_scores = {}
        self.city_weights = {}
        self.update_steps = 0

    @torch.no_grad()
    def observe(self, response, quality, city_name, max_per_city=128):
        """Collect source-only pre-OT forecasts separately for each city."""
        city = str(city_name)
        if not city:
            raise ValueError("CPRC requires a non-empty source city name")
        if (response.ndim != 3 or response.shape[1:] != (3, 3) or
                quality.shape != (len(response), 3)):
            raise ValueError("CPRC expects response [N,3,3] and quality [N,3]")
        if len(response) < 1 or max_per_city < 1:
            raise ValueError("CPRC observation batch/budget must be positive")
        if (not torch.isfinite(response).all() or
                not torch.isfinite(quality).all() or
                bool(((quality < 0) | (quality > 1)).any())):
            raise ValueError("Invalid CPRC source forecast observations")
        take = min(len(response), int(max_per_city))
        ids = torch.linspace(
            0, len(response) - 1, steps=take, device=response.device
        ).round().long()
        self._snapshots[city].append((
            response.detach().index_select(0, ids).float().cpu().contiguous(),
            quality.detach().index_select(0, ids).float().cpu().contiguous(),
        ))

    @torch.no_grad()
    def fit(self, bank, temperature=0.75):
        """Initialize each city's pattern skill, shrunk toward global bank Q."""
        if not bank.fitted or bank.quality is None:
            raise RuntimeError("Fit the shared source pattern bank before CPRC")
        if not self._snapshots:
            raise RuntimeError("CPRC has no source-city forecast snapshots")
        global_q = bank.quality.detach().float().cpu()
        for city, batches in sorted(self._snapshots.items()):
            response = torch.cat([x for x, _ in batches], dim=0)
            quality = torch.cat([q for _, q in batches], dim=0)
            attention = bank.attention(response, temperature=temperature).cpu()
            denominator = attention.sum(dim=0)  # [3,K]
            numerator = (attention * quality.unsqueeze(-1)).sum(dim=0)
            # Empirical-Bayes shrinkage prevents tiny clusters from receiving
            # extreme reliability estimates.
            city_q = (
                numerator + self.shrinkage * global_q
            ) / (denominator + self.shrinkage)
            occupancy = denominator / denominator.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            self.city_quality[city] = city_q.clamp(0, 1).contiguous()
            self.city_occupancy[city] = occupancy.clamp_min(0).contiguous()
            self.city_support[city] = denominator.clamp_min(0).contiguous()
        self._snapshots.clear()
        self._refresh_city_weights()
        return self.report()

    @torch.no_grad()
    def _refresh_city_weights(self):
        if not self.city_quality:
            self.city_scores = {}
            self.city_weights = {}
            return
        scores = {}
        for city, quality in self.city_quality.items():
            occupancy = self.city_occupancy[city]
            # Equal short/mid/long importance; each band's pattern occupancy
            # sums to one, so cities are not rewarded for having more windows.
            scores[city] = float((occupancy * quality).sum(-1).mean())
        score_values = torch.tensor(list(scores.values()), dtype=torch.float64)
        mean_score = float(score_values.mean().clamp_min(1e-8))
        raw = {
            city: max(1e-8, score / mean_score) ** self.city_power
            for city, score in scores.items()
        }
        # Project to bounded mean-one city weights. A bisection scale factor
        # keeps every weight inside [min,max] while preserving the total
        # multi-source OT contribution of the original equal-city setting.
        low, high = 0.0, 100.0
        for _ in range(60):
            scale = 0.5 * (low + high)
            mean_scaled = sum(
                min(self.city_weight_max,
                    max(self.city_weight_min, value * scale))
                for value in raw.values()
            ) / max(1, len(raw))
            if mean_scaled < 1.0:
                low = scale
            else:
                high = scale
        scale = 0.5 * (low + high)
        self.city_scores = scores
        self.city_weights = {
            city: float(min(
                self.city_weight_max,
                max(self.city_weight_min, value * scale)
            ))
            for city, value in raw.items()
        }

    @torch.no_grad()
    def update(self, feedback_records, momentum=None):
        """Update city-specific Q after the optimizer step, once per city."""
        if not self.city_quality:
            raise RuntimeError("CPRC must be fitted before online updates")
        rate = self.ema if momentum is None else float(momentum)
        if not math.isfinite(rate) or not 0 < rate <= 1:
            raise ValueError("CPRC update momentum must lie in (0,1]")
        if not feedback_records:
            raise ValueError("CPRC requires source-city feedback records")

        updates = []
        for record in feedback_records:
            summary = record.get("summary", {})
            city = str(summary.get("city", ""))
            if city not in self.city_quality:
                raise KeyError("CPRC received unknown source city: " + city)
            q_num = record["q_num"].detach().float().cpu()
            q_den = record["q_den"].detach().float().cpu()
            if (q_num.shape != self.city_quality[city].shape or
                    q_den.shape != self.city_quality[city].shape or
                    not torch.isfinite(q_num).all() or
                    not torch.isfinite(q_den).all() or
                    bool((q_den < 0).any()) or bool((q_num < 0).any())):
                raise ValueError("Invalid CPRC city feedback tensor")
            observed = (q_num / q_den.clamp_min(1e-8)).clamp(0, 1)
            # Low-support patterns move slowly; well-observed patterns adapt
            # more quickly. This avoids a single mini-batch flipping a mode.
            eta = rate * q_den / (q_den + self.shrinkage)
            before = self.city_quality[city].clone()
            self.city_quality[city] = (
                (1.0 - eta) * before + eta * observed
            ).clamp(0, 1)
            if float(q_den.sum()) > 0:
                observed_occupancy = q_den / q_den.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                self.city_occupancy[city] = (
                    (1.0 - rate) * self.city_occupancy[city] +
                    rate * observed_occupancy
                ).clamp_min(0)
            updates.append({
                "city": city,
                "quality_before_mean": float(before.mean()),
                "quality_observed_mean": float(observed.mean()),
                "quality_after_mean": float(self.city_quality[city].mean()),
                "quality_delta_l1": float(
                    (self.city_quality[city] - before).abs().sum()
                ),
                "active_patterns": int((q_den > 0).sum()),
                "support_mass": float(q_den.sum()),
            })
        self.update_steps += 1
        self._refresh_city_weights()
        return {
            "step": self.update_steps,
            "momentum": rate,
            "city_weights": dict(self.city_weights),
            "city_scores": dict(self.city_scores),
            "updates": updates,
        }

    @torch.no_grad()
    def constrain(self, alpha, source_response, bank, city_name,
                  strength=0.5, temperature=0.75):
        """Constrain F2 row mass by the reliability of each source sample's
        city-specific short/mid/long pattern mixture.

        F2 alpha is treated as the prior, not recomputed or replaced. The
        resulting alpha is a normalized reliability-constrained row marginal.
        The returned city_weight scales the effective city transport plan:
            Pi_eff_city = city_weight * Pi(alpha_constrained, beta, C_DPM-V2).
        """
        city = str(city_name)
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("CPRC strength must lie in [0,1]")
        if strength == 0:
            return alpha, float(self.city_weights.get(city, 1.0)), {
                "active": False, "strength": 0.0, "city": city,
                "alpha_l1_shift": 0.0,
            }
        if city not in self.city_quality:
            raise KeyError("No CPRC reliability profile for source city: " + city)
        if (alpha.ndim != 1 or len(alpha) != len(source_response) or
                bool((alpha < 0).any()) or
                not torch.isfinite(alpha).all()):
            raise ValueError("CPRC alpha/source response dimensions are invalid")

        attention = bank.attention(
            source_response, temperature=temperature
        ).float()
        q_city = self.city_quality[city].to(
            device=attention.device, dtype=attention.dtype
        )
        # Per-window quality of its soft pattern assignments in each band.
        band_quality = (attention * q_city.unsqueeze(0)).sum(-1)
        sample_quality = band_quality.mean(-1).clamp_min(1e-6)

        prior = alpha.detach().float()
        prior = prior / prior.sum().clamp_min(1e-12)
        prior_mean_quality = (prior * sample_quality).sum().clamp_min(1e-8)
        relative = (sample_quality / prior_mean_quality).clamp(
            self.ratio_min, self.ratio_max
        )
        gate = (1.0 - strength) + strength * relative.pow(self.sample_power)
        adjusted = prior * gate
        adjusted = adjusted / adjusted.sum().clamp_min(1e-12)

        mass_before = torch.einsum("n,nhk->hk", prior, attention)
        mass_after = torch.einsum("n,nhk->hk", adjusted, attention)
        quality_mass_before = float((mass_before * q_city).sum() / 3.0)
        quality_mass_after = float((mass_after * q_city).sum() / 3.0)
        low_quality_mask = q_city <= torch.quantile(q_city.flatten(), 0.25)
        low_mass_before = float(
            (mass_before * low_quality_mask).sum() / 3.0
        )
        low_mass_after = float(
            (mass_after * low_quality_mask).sum() / 3.0
        )
        ess_before = float(1.0 / prior.square().sum().clamp_min(1e-12))
        ess_after = float(1.0 / adjusted.square().sum().clamp_min(1e-12))
        city_weight = float(self.city_weights.get(city, 1.0))
        stats = {
            "active": True,
            "city": city,
            "strength": float(strength),
            "city_score": float(self.city_scores.get(city, 0.0)),
            "city_weight": city_weight,
            "source_count": int(len(prior)),
            "alpha_l1_shift": float((adjusted - prior).abs().sum()),
            "alpha_ess_before": ess_before,
            "alpha_ess_after": ess_after,
            "alpha_ess_fraction_before": ess_before / len(prior),
            "alpha_ess_fraction_after": ess_after / len(prior),
            "sample_quality_mean": float(sample_quality.mean()),
            "sample_quality_std": float(sample_quality.std(unbiased=False)),
            "pattern_quality_mass_before": quality_mass_before,
            "pattern_quality_mass_after": quality_mass_after,
            "low_quality_pattern_mass_before": low_mass_before,
            "low_quality_pattern_mass_after": low_mass_after,
            "pattern_mass_before": mass_before.detach().cpu().tolist(),
            "pattern_mass_after": mass_after.detach().cpu().tolist(),
            "city_pattern_quality": q_city.detach().cpu().tolist(),
            "city_pattern_occupancy": self.city_occupancy[city].tolist(),
        }
        return adjusted.to(device=alpha.device, dtype=alpha.dtype), city_weight, stats

    def state_dict(self):
        return {
            "city_quality": {
                k: v.clone() for k, v in self.city_quality.items()
            },
            "city_occupancy": {
                k: v.clone() for k, v in self.city_occupancy.items()
            },
            "city_support": {
                k: v.clone() for k, v in self.city_support.items()
            },
            "city_scores": dict(self.city_scores),
            "city_weights": dict(self.city_weights),
            "update_steps": int(self.update_steps),
            "shrinkage": self.shrinkage,
            "ema": self.ema,
            "sample_power": self.sample_power,
            "city_power": self.city_power,
        }

    def report(self):
        return {
            "cities": sorted(self.city_quality),
            "patterns_per_band": (
                int(next(iter(self.city_quality.values())).shape[-1])
                if self.city_quality else 0
            ),
            "city_quality_mean": {
                city: float(q.mean()) for city, q in self.city_quality.items()
            },
            "city_weights": dict(self.city_weights),
            "city_scores": dict(self.city_scores),
            "update_steps": int(self.update_steps),
            "source_training_labels_only": True,
        }
