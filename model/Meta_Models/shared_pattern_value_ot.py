"""Shared Pattern-Value OT for the original F2 Stage-I baseline.

One shared source-trained pattern bank is queried by every source/target sample.
Each pattern stores a Value: its historical source-forecast reliability.
Values are updated online from source training forecasts after each optimizer
step. The readout calibrates F2's source marginal; attention overlap shapes
the pairwise OT ground cost. Target labels are used only from the three-day
Stage-I training support to estimate target demand. No evaluation labels,
new predictor parameters, SOFTS changes, PCGrad changes, or Stage-II changes.
"""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def extract_pattern_response(prediction, truth, history, mean, std):
    """Return response [B*N,3,3], source skill [B*N,3], target need [B*N,3].

    Forecasts/truth are physical speed [B,N,Horizon]; history channel zero is
    city-standardized speed [B,N,History,C]. Horizon is divided into 3 bands.
    The response coordinates are normalized level, temporal change and
    within-band volatility. Skill compares the forecast with persistence;
    need measures relative forecast error within the current training support.
    """
    if (prediction.ndim != 3 or prediction.shape != truth.shape or
            prediction.shape[-1] < 3 or prediction.shape[-1] % 3 != 0):
        raise ValueError("prediction/truth must be matching [B,N,H] with H divisible by 3")
    if (history.ndim != 4 or history.shape[:2] != prediction.shape[:2] or
            history.shape[-2] < 1 or history.shape[-1] < 1):
        raise ValueError("history must be [B,N,History,C]")
    if (not torch.isfinite(prediction).all() or
            not torch.isfinite(truth).all() or
            not torch.isfinite(history).all()):
        raise ValueError("Nonfinite pattern descriptor input")

    device = prediction.device
    dtype = torch.float32
    mean = torch.as_tensor(mean, device=device, dtype=dtype).reshape(-1)
    std = torch.as_tensor(std, device=device, dtype=dtype).reshape(-1)
    if (mean.numel() != 1 or std.numel() != 1 or
            not torch.isfinite(mean).all() or
            not torch.isfinite(std).all() or float(std[0]) <= 0):
        raise ValueError("mean/std must be finite scalar values with std > 0")

    pred = prediction.detach().to(dtype)
    actual = truth.detach().to(dtype)
    hist = history.detach()[..., 0].to(device=device, dtype=dtype)
    pred = (pred - mean[0]) / std[0]
    actual = (actual - mean[0]) / std[0]

    last = hist[..., -1]
    history_scale = hist.std(dim=-1, unbiased=False).clamp_min(0.25)
    band_width = prediction.shape[-1] // 3
    responses, skills, needs = [], [], []

    for band in range(3):
        start = band * band_width
        stop = start + band_width
        y = actual[..., start:stop]
        yhat = pred[..., start:stop]
        relative = ((y - last.unsqueeze(-1)) /
                    history_scale.unsqueeze(-1)).clamp(-8, 8)

        response = torch.stack((
            relative.mean(dim=-1),
            relative[..., -1] - relative[..., 0],
            relative.std(dim=-1, unbiased=False),
        ), dim=-1)
        model_error = (yhat - y).abs().mean(dim=-1)
        persistence_error = (last.unsqueeze(-1) - y).abs().mean(dim=-1)
        error_scale = persistence_error.detach().median().clamp_min(0.1)
        skill = torch.sigmoid(
            ((persistence_error - model_error) / error_scale).clamp(-20, 20)
        )
        need_scale = model_error.detach().median().clamp_min(0.1)
        need = torch.sigmoid(
            ((model_error - need_scale) / need_scale).clamp(-20, 20)
        )
        responses.append(response)
        skills.append(skill)
        needs.append(need)

    response = torch.stack(responses, dim=2).reshape(-1, 3, 3).detach()
    skill = torch.stack(skills, dim=-1).reshape(-1, 3).detach().clamp(0, 1)
    need = torch.stack(needs, dim=-1).reshape(-1, 3).detach().clamp(0, 1)
    if (not torch.isfinite(response).all() or
            not torch.isfinite(skill).all() or
            not torch.isfinite(need).all()):
        raise FloatingPointError("Nonfinite pattern response/skill/need")
    return response, skill, need


class SharedTrafficPatternValueBank:
    """A single shared Key/Value bank, never one bank per source city."""

    def __init__(self, patterns=32, kmeans_iters=20):
        if not isinstance(patterns, int) or patterns < 2:
            raise ValueError("patterns must be an integer >= 2")
        if not isinstance(kmeans_iters, int) or kmeans_iters < 1:
            raise ValueError("kmeans_iters must be >= 1")
        self.patterns = patterns
        self.kmeans_iters = kmeans_iters
        self._snapshots = []
        self.fitted = False
        self.centers = None
        self.location = None
        self.spread = None
        self.values = None
        self.value_updates = None
        self.n_source_snapshots = 0
        self.n_source_patterns = 0
        self.feedback_steps = 0

    @torch.no_grad()
    def observe(self, response, max_per_city=128):
        """Collect an equal-budget sample from each source city before OT."""
        if self.fitted:
            raise RuntimeError("The shared pattern Key bank is frozen after fit")
        if (response.ndim != 3 or response.shape[1:] != (3, 3) or
                response.shape[0] < 1 or max_per_city < 1):
            raise ValueError("Expected response [N,3,3] and positive sample budget")
        response = response.detach().float().cpu()
        if not torch.isfinite(response).all():
            raise ValueError("Nonfinite source pattern response")
        count = min(len(response), max_per_city)
        ids = torch.linspace(0, len(response) - 1, steps=count).round().long()
        self._snapshots.append(response[ids].clone())
        self.n_source_snapshots += 1
        self.n_source_patterns += int(count)

    @torch.no_grad()
    def fit(self):
        if self.fitted:
            raise RuntimeError("The shared pattern bank can only be fitted once")
        if not self._snapshots:
            raise RuntimeError("No source training patterns were observed")
        samples = torch.cat(self._snapshots, dim=0).float()
        locations, spreads, centers = [], [], []
        for band in range(3):
            z = samples[:, band, :]
            loc = z.mean(dim=0)
            scale = z.std(dim=0, unbiased=False).clamp_min(0.25)
            zn = ((z - loc) / scale).clamp(-8, 8)
            k = self.patterns
            if len(zn) < k:
                reps = math.ceil(k / len(zn))
                init_pool = zn.repeat(reps, 1)[:k]
            else:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(20261127 + band)
                ids = torch.randperm(len(zn), generator=generator)[:k]
                init_pool = zn[ids].clone()
            c = init_pool.clone()
            for _ in range(self.kmeans_iters):
                assignment = torch.cdist(zn, c).argmin(dim=1)
                sums = torch.zeros_like(c)
                counts = torch.zeros(k, dtype=zn.dtype)
                sums.index_add_(0, assignment, zn)
                counts.index_add_(0, assignment, torch.ones_like(assignment, dtype=zn.dtype))
                updated = sums / counts.clamp_min(1).unsqueeze(-1)
                c = torch.where((counts > 0).unsqueeze(-1), updated, c)
            locations.append(loc)
            spreads.append(scale)
            centers.append(c)
        self.location = torch.stack(locations, dim=0)
        self.spread = torch.stack(spreads, dim=0)
        self.centers = torch.stack(centers, dim=0)
        # Neutral prior; evidence from real source forecasts updates Values.
        self.values = torch.full((3, self.patterns), 0.5, dtype=torch.float32)
        self.value_updates = torch.zeros((3, self.patterns), dtype=torch.float32)
        self.fitted = True
        self._snapshots.clear()
        return self.report()

    @torch.no_grad()
    def attention(self, response, temperature=0.75):
        """Query the shared Key bank; return [N,3,K] attention distributions."""
        if not self.fitted:
            raise RuntimeError("Fit the source-only shared bank before attention")
        if (response.ndim != 3 or response.shape[1:] != (3, 3) or
                not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("Invalid response shape or attention temperature")
        if not torch.isfinite(response).all():
            raise ValueError("Nonfinite attention query")
        device = response.device
        r = response.detach().float()
        loc = self.location.to(device)
        scale = self.spread.to(device)
        normalized = ((r - loc.unsqueeze(0)) / scale.unsqueeze(0)).clamp(-8, 8)
        centers = self.centers.to(device)
        logits = [
            -torch.cdist(normalized[:, h, :].contiguous(),
                         centers[h].contiguous()) / temperature
            for h in range(3)
        ]
        return torch.stack([torch.softmax(v, dim=-1) for v in logits], dim=1).detach()

    @torch.no_grad()
    def update_values(self, records, momentum=0.08):
        """Update pattern Values once per optimizer step, averaging cities equally."""
        if not self.fitted or self.values is None:
            raise RuntimeError("Fit the shared bank before updating Values")
        if not records or not math.isfinite(momentum) or not 0 < momentum <= 1:
            raise ValueError("Require city feedback and momentum in (0,1]")
        old = self.values.clone()
        city_quality, city_mass = [], []
        for record in records:
            if set(("q_num", "q_den")) - set(record):
                raise ValueError("Incomplete shared pattern Value feedback")
            qn = record["q_num"].detach().float().cpu()
            qd = record["q_den"].detach().float().cpu()
            if qn.shape != old.shape or qd.shape != old.shape:
                raise ValueError("Bad shared pattern feedback shape")
            if (not torch.isfinite(qn).all() or not torch.isfinite(qd).all() or
                    bool((qn < 0).any()) or bool((qd < 0).any())):
                raise ValueError("Invalid shared pattern feedback")
            active = qd > 1e-8
            observed = qn / qd.clamp_min(1e-8)
            mass = qd / qd.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            city_quality.append(observed.clamp(0, 1))
            city_mass.append(mass)
        quality = torch.stack(city_quality, dim=0)
        mass = torch.stack(city_mass, dim=0)
        active = mass > 0
        active_count = active.float().sum(dim=0)
        observed_mean = (
            (quality * active.float()).sum(dim=0) /
            active_count.clamp_min(1.0)
        )
        city_coverage = (
            (mass * active.float()).sum(dim=0) /
            active_count.clamp_min(1.0)
        )
        coverage = (city_coverage * self.patterns).clamp(0, 1)
        city_fraction = (active_count / len(records)).clamp(0, 1)
        eta = momentum * coverage * city_fraction
        updated = (1.0 - eta) * old + eta * observed_mean
        self.values = torch.where(active_count > 0, updated, old).clamp(0, 1)
        self.value_updates += active_count
        self.feedback_steps += 1
        return {
            "step": self.feedback_steps,
            "cities": len(records),
            "momentum": float(momentum),
            "value_band_before": old.mean(-1).tolist(),
            "value_band_observed": observed_mean.mean(-1).tolist(),
            "value_band_after": self.values.mean(-1).tolist(),
            "value_band_delta_l1": (self.values - old).abs().sum(-1).tolist(),
            "value_min": float(self.values.min()),
            "value_max": float(self.values.max()),
            "active_patterns": int((active_count > 0).sum()),
            "value_by_band_pattern": self.values.tolist(),
            "feedback_mass_by_band_pattern": active_count.tolist(),
            "city_pattern_coverage": city_coverage.tolist(),
        }

    def state_dict(self):
        if not self.fitted:
            raise RuntimeError("Cannot serialize an unfitted shared pattern bank")
        return {
            "patterns": self.patterns,
            "centers": self.centers.clone(),
            "location": self.location.clone(),
            "spread": self.spread.clone(),
            "values": self.values.clone(),
            "value_updates": self.value_updates.clone(),
            "n_source_snapshots": self.n_source_snapshots,
            "n_source_patterns": self.n_source_patterns,
            "feedback_steps": self.feedback_steps,
        }

    def report(self):
        return {
            "source_only": True,
            "shared_bank": True,
            "n_patterns_per_band": self.patterns,
            "n_source_snapshots": self.n_source_snapshots,
            "n_source_patterns": self.n_source_patterns,
            "value_prior": 0.5,
            "value_band_mean": (
                self.values.mean(-1).tolist() if self.values is not None else None
            ),
        }


def _ess_fraction(mass):
    return float(1.0 / (mass.detach().float().square().sum() * len(mass)).clamp_min(1e-12))


@torch.no_grad()
def build_pattern_value_ot(
    x, y, source_response, source_skill, target_response, target_need,
    alpha, bank, *, temperature=0.75, mass_strength=0.35,
    cost_strength=0.05, block_size=128, diagnostics=False,
):
    """Return (pairwise cost callback, adjusted F2 alpha, diagnostics, feedback).

    Value readout reweights the original F2 alpha. Attention overlap plus
    source reliability and target need modifies the OT pair cost. F2 beta stays
    unchanged. No NxM attention matrix is retained; pair costs are blocked.
    """
    if (not math.isfinite(temperature) or temperature <= 0 or
            not math.isfinite(mass_strength) or not 0 <= mass_strength <= 1 or
            not math.isfinite(cost_strength) or not 0 <= cost_strength <= 1 or
            not isinstance(block_size, int) or block_size < 1):
        raise ValueError("Invalid shared pattern OT settings")
    ns, nt = len(x), len(y)
    if (not bank.fitted or alpha.ndim != 1 or len(alpha) != ns or
            source_response.shape != (ns, 3, 3) or
            source_skill.shape != (ns, 3) or
            target_response.shape != (nt, 3, 3) or
            target_need.shape != (nt, 3) or ns < 2 or nt < 2):
        raise ValueError("Shared pattern OT input shapes do not match OT samples")
    tensors = (x.detach(), y.detach(), source_response, source_skill,
               target_response, target_need, alpha)
    if any(not torch.isfinite(t).all() for t in tensors):
        raise ValueError("Nonfinite shared pattern OT input")
    if (bool((source_skill < 0).any()) or bool((source_skill > 1).any()) or
            bool((target_need < 0).any()) or bool((target_need > 1).any()) or
            bool((alpha < 0).any()) or
            not bool(torch.isclose(alpha.sum(), alpha.new_tensor(1.0), atol=1e-5))):
        raise ValueError("Skill/need must be in [0,1] and F2 alpha must sum to one")

    source_attention = bank.attention(source_response, temperature)
    target_attention = bank.attention(target_response, temperature)
    values = bank.values.to(device=x.device, dtype=torch.float32).detach()
    source_skill = source_skill.detach().float()
    target_need = target_need.detach().float()
    source_value = (source_attention * values.unsqueeze(0)).sum(-1)
    source_trust = (0.5 * source_skill + 0.5 * source_value).clamp(0, 1)

    # Pattern Value readout calibrates F2's source marginal. A floor prevents
    # the feedback from deleting a source sample after a noisy mini-batch.
    source_gate = (0.15 + 0.85 * source_trust.mean(-1)).clamp(0.15, 1.0)
    multiplier = (1.0 - mass_strength) + mass_strength * source_gate
    alpha_new = (alpha.detach().float() * multiplier)
    alpha_new = (alpha_new / alpha_new.sum().clamp_min(1e-12)).to(alpha.dtype).detach()

    # Each source city contributes equally to the shared Value update.
    q_num = (source_attention * source_skill.unsqueeze(-1)).sum(dim=0)
    q_den = source_attention.sum(dim=0)
    feedback = {
        "q_num": q_num.detach().cpu(),
        "q_den": q_den.detach().cpu(),
        "summary": {
            "source_samples": ns,
            "source_skill_mean": float(source_skill.mean()),
            "source_value_mean": float(source_value.mean()),
            "source_trust_mean": float(source_trust.mean()),
            "target_need_mean": float(target_need.mean()),
        },
    }

    def bonus_rows(start, stop, target_ids=None):
        att_t = target_attention if target_ids is None else target_attention[target_ids]
        need_t = target_need if target_ids is None else target_need[target_ids]
        result = source_attention.new_zeros((stop - start, len(att_t)))
        for band in range(3):
            overlap = (
                source_attention[start:stop, band, :].sqrt() @
                att_t[:, band, :].sqrt().T
            ).clamp(0, 1)
            relevance = 0.25 + 0.75 * need_t[:, band]
            result += (
                overlap *
                source_trust[start:stop, band].unsqueeze(1) *
                relevance.unsqueeze(0)
            ) / 3.0
        return result.clamp(0, 1)

    src_ref_ids = torch.linspace(
        0, ns - 1, steps=min(48, ns), device=x.device
    ).round().long()
    tgt_ref_ids = torch.linspace(
        0, nt - 1, steps=min(48, nt), device=y.device
    ).round().long()
    cost_reference = float(torch.cdist(
        x.detach()[src_ref_ids].float(),
        y.detach()[tgt_ref_ids].float()
    ).median().clamp_min(1e-3))

    def cost(xx, yy):
        if (xx.ndim != 3 or yy.ndim != 3 or xx.shape[0] != 1 or
                yy.shape[0] != 1 or xx.shape[-1] != yy.shape[-1]):
            raise ValueError("Unexpected GeomLoss shared-pattern cost rank")
        sf = xx.shape[1] == ns and yy.shape[1] == nt
        tf = xx.shape[1] == nt and yy.shape[1] == ns
        if sf and tf:
            if xx.data_ptr() == x.data_ptr():
                sf = True
            elif xx.data_ptr() == y.data_ptr():
                sf = False
            else:
                raise ValueError("Ambiguous shared-pattern OT cost orientation")
        if not sf and not tf:
            raise ValueError("Unexpected shared-pattern OT source/target dimensions")
        pieces = []
        if sf:
            for start in range(0, ns, block_size):
                stop = min(ns, start + block_size)
                base = torch.cdist(xx[:, start:stop, :], yy, p=2)
                if cost_strength > 0:
                    bonus = bonus_rows(start, stop)
                    extra = cost_strength * cost_reference * (1.0 - bonus)
                    base = base + extra.to(dtype=base.dtype).unsqueeze(0)
                pieces.append(base)
        else:
            for start in range(0, nt, block_size):
                stop = min(nt, start + block_size)
                base = torch.cdist(xx[:, start:stop, :], yy, p=2)
                if cost_strength > 0:
                    target_ids = torch.arange(
                        start, stop, device=target_attention.device
                    )
                    bonus = bonus_rows(0, ns, target_ids).T
                    extra = cost_strength * cost_reference * (1.0 - bonus)
                    base = base + extra.to(dtype=base.dtype).unsqueeze(0)
                pieces.append(base)
        return torch.cat(pieces, dim=1)

    # Small fixed diagnostic view; this is not the full GeomLoss C matrix.
    sd, td = min(32, ns), min(32, nt)
    si = torch.linspace(0, ns - 1, sd, device=x.device).round().long()
    ti = torch.linspace(0, nt - 1, td, device=y.device).round().long()
    base_diag = torch.cdist(x.detach()[si].float(), y.detach()[ti].float())
    bonus_diag = source_attention.new_zeros((sd, td))
    for band in range(3):
        overlap = (
            source_attention[si, band, :].sqrt() @
            target_attention[ti, band, :].sqrt().T
        ).clamp(0, 1)
        relevance = 0.25 + 0.75 * target_need[ti, band]
        bonus_diag += (
            overlap * source_trust[si, band].unsqueeze(1) *
            relevance.unsqueeze(0)
        ) / 3.0
    extra_diag = cost_strength * cost_reference * (1.0 - bonus_diag)
    new_diag = base_diag + extra_diag
    diag = {
        "event": "shared_pattern_value_ot",
        "source_count": ns,
        "target_count": nt,
        "patterns": bank.patterns,
        "bank_feedback_steps": bank.feedback_steps,
        "cost_strength": float(cost_strength),
        "mass_strength": float(mass_strength),
        "cost_reference": cost_reference,
        "value_band_mean": values.mean(-1).tolist(),
        "value_band_min": values.min(-1).values.tolist(),
        "value_band_max": values.max(-1).values.tolist(),
        "source_skill_band_mean": source_skill.mean(0).tolist(),
        "source_value_band_mean": source_value.mean(0).tolist(),
        "source_trust_band_mean": source_trust.mean(0).tolist(),
        "target_need_band_mean": target_need.mean(0).tolist(),
        "source_attention_top_mass_by_band": source_attention.amax(-1).mean(0).tolist(),
        "target_attention_top_mass_by_band": target_attention.amax(-1).mean(0).tolist(),
        "pair_bonus_mean": float(bonus_diag.mean()),
        "pair_bonus_std": float(bonus_diag.std(unbiased=False)),
        "pair_cost_extra_mean": float(extra_diag.mean()),
        "pair_cost_extra_std": float(extra_diag.std(unbiased=False)),
        "base_cost_mean": float(base_diag.mean()),
        "modified_cost_mean": float(new_diag.mean()),
        "source_nearest_target_rank_change": float(
            (new_diag.argmin(-1) != base_diag.argmin(-1)).float().mean()
        ),
        "target_nearest_source_rank_change": float(
            (new_diag.argmin(0) != base_diag.argmin(0)).float().mean()
        ),
        "alpha_l1_shift": float((alpha_new.float() - alpha.detach().float()).abs().sum()),
        "alpha_ess_before": _ess_fraction(alpha),
        "alpha_ess_after": _ess_fraction(alpha_new),
        "alpha_top_ids_before": alpha.detach().topk(min(8, ns)).indices.cpu().tolist(),
        "alpha_top_ids_after": alpha_new.topk(min(8, ns)).indices.cpu().tolist(),
        "source_gate_quantiles": [
            float(v) for v in torch.quantile(
                source_gate, torch.tensor([0.1, 0.5, 0.9], device=x.device)
            )
        ],
        "feedback": feedback["summary"],
        "diagnostic_subset_only": True,
    }
    if diagnostics:
        top = bonus_diag.flatten().topk(min(8, bonus_diag.numel())).indices
        diag["top_pattern_pairs"] = [
            {
                "source_flat_id": int(si[k // td]),
                "target_flat_id": int(ti[k % td]),
                "pair_bonus": float(bonus_diag.flatten()[k]),
                "base_cost": float(base_diag.flatten()[k]),
                "modified_cost": float(new_diag.flatten()[k]),
                "source_pattern_by_band": source_attention[si[k // td]].argmax(-1).cpu().tolist(),
                "target_pattern_by_band": target_attention[ti[k % td]].argmax(-1).cpu().tolist(),
                "source_value_readout": source_value[si[k // td]].cpu().tolist(),
                "source_skill": source_skill[si[k // td]].cpu().tolist(),
                "target_need": target_need[ti[k % td]].cpu().tolist(),
            }
            for k in top.tolist()
        ]
    return (cost if cost_strength > 0 else None), alpha_new, diag, feedback
