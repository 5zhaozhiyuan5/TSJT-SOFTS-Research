"""DPM-OT: source-trained evolution pattern memory + attention-conditioned OT cost.

This is an OPTIONAL, actual Stage-I OT component on top of unchanged F2:
- Bank: robust per-horizon K-means on *source training* history->future
  descriptors, sampled over the pre-OT training epochs, then frozen.
- Attention: source/target training-support examples query the SAME bank.
- Pair utility: pattern affinity times relative source forecast skill and
  relative target forecast need (short/mid/long kept separate until cost).
- Cost: C_new(i,j) = ||[z_s,10Y_s]-[z_t,10Y_t]||_2
                    + lambda * C_ref * (1 - B(i,j)/4).
F2 alpha and uniform beta are preserved BITWISE; no new predictor parameters,
no edits to SOFTS, PCGrad, Sinkhorn hyperparameters or official Stage-II.

Target future labels are permissible only from the 3-day training SUPPORT
inside Stage-I (as in the existing PMME OT). Bank NEVER sees target labels.
No test/validation future label is read by this component.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


class SourceEvolutionPatternBank:
    """One frozen source-trained bank for each of 3 forecast bands.

    observe() is called at a fixed interval before DAepoch, after each source
    forward; each city contributes the same <=max_per_city entries. The bank
    is fitted once at DAepoch, not rebuilt from the current OT batch.
    """

    def __init__(self, patterns=32, max_snapshots=120, kmeans_iters=12):
        if patterns < 2 or max_snapshots < 3 or kmeans_iters < 1:
            raise ValueError("Invalid bank size / snapshot count / iteration count")
        self.patterns = int(patterns)
        self.max_snapshots = int(max_snapshots)
        self.kmeans_iters = int(kmeans_iters)
        self._snapshots = []
        self._quality_snapshots = []
        self.quality = None     # [3,K] CPU EMA of source forecast skill
        self.reliability = None # [3,K] CPU EMA of cross-city compatibility
        self.feedback_steps = 0
        self.cumulative_q_mass = None
        self.cumulative_r_mass = None
        self.centers = None   # CPU tensor [3, K, 3] in normalized descriptor space
        self.location = None  # CPU tensor [3, 3]
        self.spread = None    # CPU tensor [3, 3]
        self.fit_counts = None

    @property
    def fitted(self):
        return self.centers is not None

    @torch.no_grad()
    def observe(self, source_response, max_per_city=128, quality=None):
        if self.fitted:
            raise RuntimeError("Pattern bank is frozen; cannot observe more data")
        if source_response.ndim != 3 or source_response.shape[1:] != (3, 3):
            raise ValueError("Expected source response [N,3,3]")
        if max_per_city < 1 or len(self._snapshots) >= self.max_snapshots:
            if max_per_city < 1:
                raise ValueError("max_per_city must be positive")
            return
        n = len(source_response)
        if n < 1 or not torch.isfinite(source_response).all():
            raise ValueError("Invalid source response samples")
        if quality is not None:
            if (quality.shape != (n, 3) or not torch.isfinite(quality).all()
                    or bool((quality < 0).any()) or bool((quality > 1).any())):
                raise ValueError("Quality must be [N,3] finite values in [0,1]")
        take = min(n, max_per_city)
        # Sample indices across windows/sensors with deterministic spacing.
        idx = torch.linspace(0, n-1, steps=take, device=source_response.device)
        selected = idx.round().long()
        self._snapshots.append(
            source_response.detach().index_select(0, selected)
            .to(device="cpu", dtype=torch.float32).contiguous()
        )
        self._quality_snapshots.append(
            None if quality is None else
            quality.detach().index_select(0, selected)
            .to(device="cpu", dtype=torch.float32).contiguous()
        )

    @torch.no_grad()
    def fit(self):
        if self.fitted:
            return self.report()
        if not self._snapshots:
            raise RuntimeError("Empty source bank: observe source training windows before DAepoch")
        X = torch.cat(self._snapshots, dim=0)
        if X.shape[0] < self.patterns:
            raise ValueError("Not enough source training patterns for requested K")
        centers, locations, spreads, counts = [], [], [], []
        for h in range(3):
            raw = X[:, h, :].float()
            mid = raw.median(dim=0).values
            quantiles = torch.quantile(raw, torch.tensor([.25,.75]), dim=0)
            scale = (quantiles[1] - quantiles[0]).clamp_min(0.2)
            normalized = ((raw - mid) / scale).clamp(-8.0, 8.0)
            # Deterministic farthest-point initialization (more coverage
            # than simply taking the first 32 samples).
            first = torch.cdist(normalized, torch.zeros(1,3)).flatten().argmin()
            selected = [int(first)]
            nearest = torch.cdist(normalized, normalized[first:first+1]).flatten().square()
            for _ in range(1, self.patterns):
                nxt = int(nearest.argmax())
                selected.append(nxt)
                dist = (normalized - normalized[nxt]).square().sum(-1)
                nearest = torch.minimum(nearest, dist)
            c = normalized[selected].clone()
            for _ in range(self.kmeans_iters):
                d = torch.cdist(normalized, c)
                assigned = d.argmin(dim=1)
                group_n = torch.bincount(assigned, minlength=self.patterns)
                accum = torch.zeros_like(c)
                accum.index_add_(0, assigned, normalized)
                valid = group_n > 0
                c[valid] = accum[valid] / group_n[valid, None]
            assigned = torch.cdist(normalized, c).argmin(dim=1)
            centers.append(c.contiguous())
            locations.append(mid)
            spreads.append(scale)
            counts.append(torch.bincount(assigned, minlength=self.patterns))
        self.centers = torch.stack(centers, dim=0)
        self.location = torch.stack(locations, dim=0)
        self.spread = torch.stack(spreads, dim=0)
        self.fit_counts = torch.stack(counts, dim=0)
        self.n_training_patterns = len(X)
        self.n_snapshots = len(self._snapshots)
        # Initialize Q from SOURCE-only pre-OT prediction quality. R starts
        # neutral at 1, until actual source/target-support comparisons occur.
        if all(q is not None for q in self._quality_snapshots):
            q_all = torch.cat(self._quality_snapshots, dim=0)
            a_all = self.attention(X, temperature=0.75)
            self.quality = torch.stack([
                (a_all[:, h].T @ q_all[:, h]) /
                a_all[:, h].sum(0).clamp_min(1e-8)
                for h in range(3)
            ], dim=0).clamp(0, 1).cpu()
        else:
            self.quality = torch.full((3, self.patterns), 0.5)
        self.reliability = torch.ones((3, self.patterns))
        self.cumulative_q_mass = torch.zeros((3, self.patterns))
        self.cumulative_r_mass = torch.zeros((3, self.patterns))
        self._snapshots.clear()
        self._quality_snapshots.clear()
        return self.report()

    def report(self):
        return {
            "patterns_per_band": self.patterns,
            "n_training_patterns": int(self.n_training_patterns),
            "n_source_snapshots": int(self.n_snapshots),
            "band_nonempty_clusters": [
                int((row > 0).sum()) for row in self.fit_counts
            ],
            "band_max_cluster_fraction": [
                float(row.max() / row.sum().clamp_min(1)) for row in self.fit_counts
            ],
            "source_only": True,
            "frozen": self.fitted,
            "center_frozen": self.fitted,
            "feedback_steps": self.feedback_steps,
            "quality_band_mean": (
                [float(x) for x in self.quality.mean(-1)]
                if self.quality is not None else None
            ),
            "reliability_band_mean": (
                [float(x) for x in self.reliability.mean(-1)]
                if self.reliability is not None else None
            ),
        }

    def state_dict(self):
        if not self.fitted:
            raise RuntimeError("Pattern bank not fitted")
        return {
            "centers": self.centers.clone(),
            "location": self.location.clone(),
            "spread": self.spread.clone(),
            "fit_counts": self.fit_counts.clone(),
            "n_training_patterns": self.n_training_patterns,
            "n_snapshots": self.n_snapshots,
            "patterns": self.patterns,
            "quality": self.quality.clone(),
            "reliability": self.reliability.clone(),
            "feedback_steps": self.feedback_steps,
            "cumulative_q_mass": self.cumulative_q_mass.clone(),
            "cumulative_r_mass": self.cumulative_r_mass.clone(),
        }


    @torch.no_grad()
    def update_reliability(self, records, momentum=0.08):
        """ONE update per optimizer step, after *all three* source cities.

        Each city contributes equally (not proportional to node count).
        Low-occupancy prototypes use a smaller effective EMA rate.
        This operation is forbidden inside the GeomLoss cost callback.
        """
        if not self.fitted or self.quality is None:
            raise RuntimeError("Source bank must be fitted before online feedback")
        if not records or not 0 < momentum <= 1:
            raise ValueError("Require feedback from source cities and valid EMA rate")
        old_q = self.quality.clone()
        old_r = self.reliability.clone()
        q_num = torch.zeros_like(old_q)
        q_den = torch.zeros_like(old_q)
        r_num = torch.zeros_like(old_r)
        r_den = torch.zeros_like(old_r)
        city_summaries = []
        for record in records:
            if set(("q_num","q_den","r_num","r_den")) - set(record):
                raise ValueError("Incomplete bank feedback record")
            qn = record["q_num"].detach().float().cpu()
            qd = record["q_den"].detach().float().cpu()
            rn = record["r_num"].detach().float().cpu()
            rd = record["r_den"].detach().float().cpu()
            if any(t.shape != old_q.shape for t in (qn, qd, rn, rd)):
                raise ValueError("Bad pattern bank feedback shape")
            if any(not torch.isfinite(t).all() for t in (qn, qd, rn, rd)):
                raise ValueError("Nonfinite Q/R feedback")
            if any(bool((t < 0).any()) for t in (qn, qd, rn, rd)):
                raise ValueError("Negative Q/R feedback")
            # Normalize by total membership per city and horizon.
            # This avoids large-node cities dominating small-node cities.
            for numerator, denominator, total_num, total_den in (
                    (qn,qd,q_num,q_den),(rn,rd,r_num,r_den)):
                size = denominator.sum(-1,keepdim=True).clamp_min(1e-9)
                total_num.add_(numerator/size)
                total_den.add_(denominator/size)
            city_summaries.append(record.get("summary",{}))
        q_mean = (q_num / q_den.clamp_min(1e-9)).clamp(0,1)
        r_mean = (r_num / r_den.clamp_min(1e-9)).clamp(0,1)
        # At average assignment mass 1/K, update with full EMA;
        # rare patterns contribute proportionally less to prevent drift.
        q_eta = momentum * (q_den * self.patterns / len(records)).clamp(0,1)
        r_eta = momentum * (r_den * self.patterns / len(records)).clamp(0,1)
        self.quality = ((1-q_eta)*old_q + q_eta*q_mean).clamp(0,1)
        self.reliability = ((1-r_eta)*old_r + r_eta*r_mean).clamp(0,1)
        self.cumulative_q_mass += q_den
        self.cumulative_r_mass += r_den
        self.feedback_steps += 1
        return {
            "step": self.feedback_steps,
            "cities": len(records),
            "momentum": momentum,
            "q_band_before": old_q.mean(-1).tolist(),
            "q_band_observed": q_mean.mean(-1).tolist(),
            "q_band_after": self.quality.mean(-1).tolist(),
            "r_band_before": old_r.mean(-1).tolist(),
            "r_band_observed": r_mean.mean(-1).tolist(),
            "r_band_after": self.reliability.mean(-1).tolist(),
            "q_band_delta_l1": (self.quality-old_q).abs().sum(-1).tolist(),
            "r_band_delta_l1": (self.reliability-old_r).abs().sum(-1).tolist(),
            "q_min": float(self.quality.min()), "q_max": float(self.quality.max()),
            "r_min": float(self.reliability.min()), "r_max": float(self.reliability.max()),
            "q_active_prototypes": int((q_den>0).sum()),
            "r_active_prototypes": int((r_den>0).sum()),
            "city_summaries": city_summaries,
        }

    @torch.no_grad()
    def attention(self, responses, temperature=0.75):
        """Return [N,3,K] pattern attention distributions (all detached)."""
        if not self.fitted:
            raise RuntimeError("Fit the source bank before attention")
        if (responses.ndim != 3 or responses.shape[1:] != (3,3) or
                not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("Bad descriptor shapes or attention temperature")
        if not torch.isfinite(responses).all():
            raise ValueError("Nonfinite response")
        device = responses.device
        r = responses.detach().float()
        mean = self.location.to(device)
        scale = self.spread.to(device)
        normalized = ((r - mean.unsqueeze(0))/scale.unsqueeze(0)).clamp(-8,8)
        c = self.centers.to(device)
        logits = [
            -torch.cdist(normalized[:,h,:].contiguous(),
                         c[h].contiguous()) / temperature
            for h in range(3)
        ]
        return torch.stack([torch.softmax(z,dim=-1) for z in logits],dim=1)


@torch.no_grad()
def _relative_gate(values):
    """Domain- and horizon-relative skill/need, with bounded amplification."""
    if values.ndim != 2 or values.shape[-1] != 3:
        raise ValueError("Expect quality [N,3]")
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid skill/need")
    means = values.float().mean(0,keepdim=True)
    return (values.float() / means.clamp_min(1e-6)).clamp(0,2)


def build_pattern_attention_cost(
        x, y, source_response, source_skill, target_response, target_need,
        bank, strength=0.02, temperature=0.75, block_size=128,
        mode="full"):
    """Build GeomLoss tensorized cost callback and diagnostic info.

    x,y: [Ns,D], [Nt,D], ORIGINAL PMME concat(learned SOFTS rep,10*Y).
    Original F2 alpha and beta are NOT modified by this function.

    If strength==0, callback is None and caller MUST use original SamplesLoss.
    Cost uses detached attention / skill / need, but computes the base cdist
    from GeomLoss's xx,yy arguments so SOFTS still receives OT gradients.
    """
    if (not math.isfinite(strength) or not 0 <= strength <= 1 or
            not math.isfinite(temperature) or temperature <= 0 or
            not isinstance(block_size,int) or block_size < 1 or
            mode not in ("full","pattern")):
        raise ValueError("Invalid DPM-OT cost parameters")
    if strength == 0:
        return None, {"active": False, "strength": 0.0}
    if not bank.fitted:
        raise RuntimeError("Pattern library must be fitted before active OT")
    if (x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or
            source_response.shape!=(len(x),3,3) or
            target_response.shape!=(len(y),3,3) or
            source_skill.shape!=(len(x),3) or
            target_need.shape!=(len(y),3)):
        raise ValueError("Incompatible OT points / pattern descriptor shapes")
    with torch.no_grad():
        attention_s = bank.attention(source_response,temperature).sqrt()
        attention_t = bank.attention(target_response,temperature).sqrt()
        if mode == "full":
            gate_s = _relative_gate(source_skill)
            gate_t = _relative_gate(target_need)
        else:
            gate_s = torch.ones_like(source_skill).float()
            gate_t = torch.ones_like(target_need).float()
        source_keys = attention_s * gate_s.unsqueeze(-1)
        target_keys = attention_t * gate_t.unsqueeze(-1)
        # C_ref is from ORIGINAL OT points and always detached.
        src_ids = torch.linspace(0, len(x)-1, steps=min(48,len(x)),
                                 device=x.device).round().long()
        tgt_ids = torch.linspace(0, len(y)-1, steps=min(48,len(y)),
                                 device=y.device).round().long()
        ref = torch.cdist(x.detach()[src_ids].float(),
                          y.detach()[tgt_ids].float()).median().clamp_min(1e-3)
        coefficient = float(ref) * strength

        def pair_bonus(start, stop):
            # B_ij = sum_h (q_rel_i,h * d_rel_j,h *
            #               Bhattacharyya(att_s_i,h, att_t_j,h)) / 3
            # Range [0,4]. Different pair choices receive different costs.
            bonus = source_keys.new_zeros((stop-start,len(y)))
            for h in range(3):
                bonus += source_keys[start:stop,h,:] @ target_keys[:,h,:].T / 3.0
            return bonus.clamp(0,4)

        sample_b = pair_bonus(0,min(32,len(x)))[:,:min(32,len(y))]
        penalty = 1.0 - sample_b / 4.0
        # This 32x32 *cost-ranking diagnostic* checks whether DPM changes
        # actual candidate preferences. It is NOT the full GeomLoss Pi.
        sampled_base = torch.cdist(
            x.detach()[:len(sample_b)].float(),
            y.detach()[:sample_b.shape[1]].float()
        )
        sampled_new = sampled_base + coefficient * penalty
        changed_source_to_target = (
            sampled_base.argmin(dim=1) != sampled_new.argmin(dim=1)
        ).float().mean()
        changed_target_to_source = (
            sampled_base.argmin(dim=0) != sampled_new.argmin(dim=0)
        ).float().mean()
        diagnostic = {
            "active": True,
            "strength": float(strength),
            "mode": mode,
            "bank": bank.report(),
            "source_count": len(x),
            "target_count": len(y),
            "pattern_count": bank.patterns,
            "attention_temperature": float(temperature),
            "cost_reference": float(ref),
            "penalty_coeff": coefficient,
            "sample_bonus_mean": float(sample_b.mean()),
            "sample_bonus_std": float(sample_b.std(unbiased=False)),
            "sample_penalty_std": float(penalty.std(unbiased=False)),
            "sample_added_cost_mean": float((penalty*coefficient).mean()),
            "sample_original_cost_mean": float(sampled_base.mean()),
            "sample_pair_nearest_target_changed_frac": float(changed_source_to_target),
            "sample_pair_nearest_source_changed_frac": float(changed_target_to_source),
            "pair_change_note": "32x32 cost ranking proxy, not full OT plan Pi",
            "source_skill_mean": float(source_skill.float().mean()),
            "target_need_mean": float(target_need.float().mean()),
            "source_attention_top_mass": float(attention_s.square().amax(-1).mean()),
            "target_attention_top_mass": float(attention_t.square().amax(-1).mean()),
            "uses_target_support_labels_only": True,
            "marginals_unchanged_from_original_f2": True,
        }

    def cost(xx, yy):
        if (xx.ndim!=3 or yy.ndim!=3 or xx.shape[0]!=1 or yy.shape[0]!=1 or
                xx.shape[-1]!=yy.shape[-1]):
            raise ValueError(
                "Unexpected GeomLoss tensorized OT cost ranks/batch/features: "
                f"xx={tuple(xx.shape)}, yy={tuple(yy.shape)}"
            )
        # GeomLoss's tensorized Sinkhorn backend requests cost(y, x.detach())
        # as well as cost(x, y.detach()). Accept either orientation and return
        # the same pairwise DPM penalty transposed for the reverse request.
        source_first = xx.shape[1]==len(x) and yy.shape[1]==len(y)
        target_first = xx.shape[1]==len(y) and yy.shape[1]==len(x)
        if not source_first and not target_first:
            raise ValueError(
                "Unexpected GeomLoss tensorized OT sample counts: "
                f"xx={tuple(xx.shape)}, yy={tuple(yy.shape)}, "
                f"source={len(x)}, target={len(y)}"
            )
        if source_first and target_first:
            # Equal source/target counts make dimensions ambiguous. GeomLoss
            # preserves storage for its x/y inputs, including detach().
            if xx.data_ptr()==x.data_ptr():
                source_first=True
            elif xx.data_ptr()==y.data_ptr():
                source_first=False
            else:
                raise ValueError("Cannot identify GeomLoss OT cost orientation")
        chunks=[]
        if source_first:
            for start in range(0,len(x),block_size):
                stop=min(len(x),start+block_size)
                base=torch.cdist(xx[:,start:stop,:],yy,p=2)
                # Detached utility changes the transport plan; the base cost
                # remains differentiable with respect to the SOFTS features.
                extra=(1.0-pair_bonus(start,stop)/4.0).to(
                    device=base.device,dtype=base.dtype
                )
                chunks.append(base + coefficient * extra.unsqueeze(0))
        else:
            for start in range(0,len(y),block_size):
                stop=min(len(y),start+block_size)
                base=torch.cdist(xx[:,start:stop,:],yy,p=2)
                bonus=target_keys.new_zeros((stop-start,len(x)))
                for h in range(3):
                    bonus += target_keys[start:stop,h,:] @ source_keys[:,h,:].T / 3.0
                extra=(1.0-bonus.clamp(0,4)/4.0).to(
                    device=base.device,dtype=base.dtype
                )
                chunks.append(base + coefficient * extra.unsqueeze(0))
        return torch.cat(chunks,dim=1)

    return cost, diagnostic
