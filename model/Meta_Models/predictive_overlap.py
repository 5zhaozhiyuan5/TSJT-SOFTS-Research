"""Predictive overlap weighting for PMME Stage-I.

Replaces the three variational GP domain classifiers, NOT the conditional
weighted-Sinkhorn transport. A historical representation predicts both a
domain-overlap score and a future-trajectory compatibility score.

The teacher combines soft-DTW shape distortion with the expected temporal
alignment penalty (DILATE-style) using a forward dynamic program. This is a
lightweight teacher score, not the forecasting loss in the original DILATE.
All teacher targets come solely from labeled Stage-I training batches.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PredictiveOverlapWeighter(nn.Module):
    def __init__(self, num_domains=3, hidden_dim=64, teacher_steps=12,
                 prototype_count=3, soft_dtw_gamma=0.08,
                 time_penalty=0.20, teacher_temperature=0.6):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LazyLinear(hidden_dim), nn.GELU(),
                          nn.Linear(hidden_dim, 2))
            for _ in range(num_domains)
        ])
        self.teacher_steps = teacher_steps
        self.prototype_count = prototype_count
        self.gamma = soft_dtw_gamma
        self.time_penalty = time_penalty
        self.teacher_temperature = teacher_temperature

    @staticmethod
    def _shape(y, steps):
        """Relative future evolution with a small level component.

        Center at the *first future value* (not an unknown test value);
        normalise the change by its own magnitude, and retain an attenuated
        level feature so cross-city scaling is not ignored entirely.
        """
        y = F.adaptive_avg_pool1d(y.unsqueeze(1).float(), steps).squeeze(1)
        delta = y - y[:, :1]
        scale = delta.abs().mean(dim=1, keepdim=True).clamp_min(0.25)
        shape = (delta / scale).clamp(-6.0, 6.0)
        level = (y[:, :1] / 10.0).clamp(-3.0, 3.0)
        return shape + 0.10 * level

    def _prototypes(self, y):
        """Deterministic low/median/high slope representatives."""
        trend = y[:, -1] - y[:, 0]
        idx = []
        for q in torch.linspace(0, 1, self.prototype_count, device=y.device):
            pivot = torch.quantile(trend.float(), q)
            idx.append((trend - pivot).abs().argmin())
        return y[torch.stack(idx)]

    def _dilate_distance(self, query, refs):
        """Soft-DTW shape cost plus expected time distortion, batched.

        The time expectation is propagated along the same soft alignment
        distribution as the Soft-DTW recurrence (not added to local costs).
        Complexity: O(N*K*steps^2) with reduced teacher steps.
        """
        n, t = query.shape
        k = refs.shape[0]
        local = (query[:, None, :, None] - refs[None, :, None, :]).square()
        indices = torch.arange(t, device=query.device, dtype=query.dtype)
        temporal = ((indices[:, None] - indices[None, :]) /
                    max(1, t - 1)).square()
        inf = torch.full((n, k), float("inf"), device=query.device,
                         dtype=query.dtype)
        zero = torch.zeros_like(inf)
        prev = [zero] + [inf] * t
        # Expected temporal cost of an unreachable predecessor is ignored
        # by its zero softmax probability.
        prev_time = [zero] + [zero] * t
        gamma = max(self.gamma, 1e-4)
        for i in range(1, t + 1):
            curr, curr_time = [inf], [zero]
            for j in range(1, t + 1):
                paths = torch.stack((prev[j], curr[j - 1], prev[j - 1]), dim=-1)
                log_weights = -paths / gamma
                weights = torch.softmax(log_weights, dim=-1)
                curr.append(local[:, :, i - 1, j - 1] -
                            gamma * torch.logsumexp(log_weights, dim=-1))
                former_time = torch.stack((prev_time[j], curr_time[j - 1],
                                           prev_time[j - 1]), dim=-1)
                curr_time.append((weights * former_time).sum(dim=-1) +
                                 self.time_penalty * temporal[i - 1, j - 1])
            prev, prev_time = curr, curr_time
        # Shape term may be slightly negative due to soft-min entropy.
        shape_cost = prev[t].clamp_min(0) / t
        temporal_cost = prev_time[t] / t
        return 0.75 * shape_cost + 0.25 * temporal_cost

    @torch.no_grad()
    def _teacher(self, source_future, target_future):
        source = self._shape(source_future.detach(), self.teacher_steps)
        target = self._shape(target_future.detach(), self.teacher_steps)
        target_refs = self._prototypes(target)
        source_refs = self._prototypes(source)
        source_dist = self._dilate_distance(source, target_refs).min(dim=1).values
        target_dist = self._dilate_distance(target, source_refs).min(dim=1).values
        source_score = torch.exp(-source_dist.clamp_min(0) /
                                 self.teacher_temperature).clamp(0.05, 1.0)
        target_score = torch.exp(-target_dist.clamp_min(0) /
                                 self.teacher_temperature).clamp(0.05, 1.0)
        return source_score, target_score

    def training_weights(self, domain_index, source_rep, target_rep,
                         source_future, target_future, optimizer):
        # Do not backprop into the forecasting backbone through the
        # classifier; the original DVGP masses were detached as well.
        s = source_rep.detach()
        t = target_rep.detach()
        domain_logits, compatibility_logits = self.heads[domain_index](
            torch.cat((s, t), dim=0)).unbind(dim=-1)
        ns = len(s)
        labels = torch.cat((torch.zeros(ns, device=s.device),
                            torch.ones(len(t), device=t.device)))
        future_source, future_target = self._teacher(source_future, target_future)
        teacher = torch.cat((future_source, future_target)).to(s.dtype)
        loss_domain = F.binary_cross_entropy_with_logits(domain_logits, labels)
        loss_future = F.mse_loss(torch.sigmoid(compatibility_logits), teacher)
        loss = loss_domain + 0.5 * loss_future
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Source: probability of target-like domain membership.
            # Target: probability of source-like domain membership.
            # Multiply by learned future compatibility and apply a mass
            # floor to avoid hard zero-weight transport or empty marginals.
            d = torch.sigmoid(domain_logits)
            c = torch.sigmoid(compatibility_logits)
            source_mass = 0.10 + 0.90 * torch.sqrt((d[:ns] *
                                                    c[:ns]).clamp_min(1e-8))
            target_mass = 0.10 + 0.90 * torch.sqrt(((1-d[ns:]) *
                                                    c[ns:]).clamp_min(1e-8))
            return source_mass.detach(), target_mass.detach(), {
                "domain_loss": loss_domain.detach(),
                "future_loss": loss_future.detach(),
                "source_teacher": future_source.mean(),
                "target_teacher": future_target.mean(),
            }
