"""Stage-I predictive conditional evolution alignment.

The original three domain-specific DVGP heads are replaced by small history-
only heads. Their output remains two strictly positive marginal vectors for
weighted Sinkhorn. A *separate* pairwise historical embedding is supervised
using both historical conditions and anchored future evolution, and supplies
an additive OT pairwise cost. No target test future is ever used.

DILATE-style teacher = Soft-DTW shape term plus expected temporal offset.
It is not a verbatim implementation of the original published DILATE loss.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PredictiveOverlapWeighter(nn.Module):
    def __init__(self, num_domains=3, hidden_dim=64, teacher_steps=12,
                 prototype_count=3, soft_dtw_gamma=0.08,
                 time_penalty=0.20, teacher_temperature=0.6,
                 pair_samples=48, pair_dim=32):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LazyLinear(hidden_dim), nn.GELU(),
                          nn.Linear(hidden_dim, 2))
            for _ in range(num_domains)
        ])
        self.pair_heads = nn.ModuleList([
            nn.Sequential(nn.LazyLinear(hidden_dim), nn.GELU(),
                          nn.Linear(hidden_dim, pair_dim))
            for _ in range(num_domains)
        ])
        self.teacher_steps = teacher_steps
        self.prototype_count = prototype_count
        self.gamma = soft_dtw_gamma
        self.time_penalty = time_penalty
        self.teacher_temperature = teacher_temperature
        self.pair_samples = pair_samples

    def _conditional_sequences(self, history, future):
        """Keep history and future distinct; anchor future at last history.

        Both inputs MUST use the identical per-window, train-only RevIN space.
        [N,L] historical values and [N,H] known Stage-I future labels.
        """
        hist = F.adaptive_avg_pool1d(
            history.detach().float().unsqueeze(1), self.teacher_steps
        ).squeeze(1)
        fut = F.adaptive_avg_pool1d(
            future.detach().float().unsqueeze(1), self.teacher_steps
        ).squeeze(1)
        # RevIN has already standardised levels; bound extreme outliers.
        hist = hist.clamp(-6, 6)
        future_change = (fut - history[:, -1:].detach().float()).clamp(-8, 8)
        return hist, future_change

    def _prototypes(self, future):
        trend = future[:, -1] - future[:, 0]
        ids = []
        for q in torch.linspace(0, 1, self.prototype_count, device=future.device):
            pivot = torch.quantile(trend.float(), q)
            ids.append((trend - pivot).abs().argmin())
        return torch.stack(ids)

    def _dilate_pairs(self, query, refs):
        """Vectorised one-to-one DILATE-style distance; O(N*steps^2).

        Evaluates distinct history and anchored-future sequences separately:
        the dynamic programming path can never align history with future.
        """
        n, steps = query.shape
        if query.shape != refs.shape:
            raise ValueError("Paired teacher inputs must have the same shape")
        local = (query[:, :, None] - refs[:, None, :]).square()
        index = torch.arange(steps, device=query.device, dtype=query.dtype)
        temporal = ((index[:, None] - index[None, :]) /
                    max(1, steps - 1)).square()
        inf = torch.full((n,), float("inf"), device=query.device, dtype=query.dtype)
        zero = torch.zeros_like(inf)
        prev = [zero] + [inf] * steps
        prev_time = [zero] + [zero] * steps
        gamma = max(self.gamma, 1e-4)
        for i in range(1, steps + 1):
            curr, curr_time = [inf], [zero]
            for j in range(1, steps + 1):
                paths = torch.stack((prev[j], curr[j - 1], prev[j - 1]), -1)
                logits = -paths / gamma
                probs = torch.softmax(logits, dim=-1)
                curr.append(local[:, i - 1, j - 1] -
                            gamma * torch.logsumexp(logits, dim=-1))
                times = torch.stack(
                    (prev_time[j], curr_time[j - 1], prev_time[j - 1]), -1)
                curr_time.append((probs * times).sum(-1) +
                                 self.time_penalty * temporal[i - 1, j - 1])
            prev, prev_time = curr, curr_time
        return (0.75 * prev[steps].clamp_min(0) +
                0.25 * prev_time[steps]) / steps

    def _distance(self, hist_a, future_a, hist_b, future_b):
        hist_distance = self._dilate_pairs(hist_a, hist_b)
        evolution_distance = self._dilate_pairs(future_a, future_b)
        # Explicitly compare the first future transition and historical
        # approach velocity, preventing the same future from being treated as
        # equivalent under falling versus rising historical conditions.
        velocity_a = hist_a[:, -1] - hist_a[:, -2]
        velocity_b = hist_b[:, -1] - hist_b[:, -2]
        transition_penalty = (velocity_a - velocity_b).abs().clamp_max(3)
        return (0.35 * hist_distance + 0.60 * evolution_distance +
                0.05 * transition_penalty)

    @torch.no_grad()
    def _teacher(self, source_history, target_history,
                 source_future, target_future):
        sh, sf = self._conditional_sequences(source_history, source_future)
        th, tf = self._conditional_sequences(target_history, target_future)
        tid = self._prototypes(tf)
        sid = self._prototypes(sf)
        source_distances = torch.stack([
            self._distance(sh, sf,
                           th[j:j+1].expand(len(sh), -1),
                           tf[j:j+1].expand(len(sh), -1))
            for j in tid.tolist()
        ], dim=1)
        target_distances = torch.stack([
            self._distance(th, tf,
                           sh[j:j+1].expand(len(th), -1),
                           sf[j:j+1].expand(len(th), -1))
            for j in sid.tolist()
        ], dim=1)
        source_score = torch.exp(-source_distances.min(1).values /
                                 self.teacher_temperature).clamp(0.05, 1.0)
        target_score = torch.exp(-target_distances.min(1).values /
                                 self.teacher_temperature).clamp(0.05, 1.0)
        return source_score, target_score

    @torch.no_grad()
    def _pair_teacher(self, source_history, target_history,
                      source_future, target_future, src_ids, tgt_ids):
        sh, sf = self._conditional_sequences(
            source_history[src_ids], source_future[src_ids])
        th, tf = self._conditional_sequences(
            target_history[tgt_ids], target_future[tgt_ids])
        distance = self._distance(sh, sf, th, tf)
        return torch.exp(-distance / self.teacher_temperature).clamp(0.05, 1.0)

    def training_weights(self, domain_index, source_rep, target_rep,
                         source_future, target_future,
                         source_history, target_history, optimizer):
        s, t = source_rep.detach(), target_rep.detach()
        domain_logits, compatibility_logits = self.heads[domain_index](
            torch.cat((s, t), dim=0)).unbind(-1)
        ns, nt = len(s), len(t)
        labels = torch.cat((torch.zeros(ns, device=s.device),
                            torch.ones(nt, device=t.device)))
        future_source, future_target = self._teacher(
            source_history, target_history, source_future, target_future)
        teacher = torch.cat((future_source, future_target)).to(s.dtype)
        loss_domain = F.binary_cross_entropy_with_logits(domain_logits, labels)
        loss_overlap = F.mse_loss(torch.sigmoid(compatibility_logits), teacher)
        # Pair teacher and pair student are trained on the SAME sampled pairs.
        ids_s = torch.randint(ns, (self.pair_samples,), device=s.device)
        ids_t = torch.randint(nt, (self.pair_samples,), device=t.device)
        target_pairs = self._pair_teacher(
            source_history, target_history, source_future, target_future,
            ids_s, ids_t).to(s.dtype)
        ps = F.normalize(self.pair_heads[domain_index](s[ids_s]), dim=-1)
        pt = F.normalize(self.pair_heads[domain_index](t[ids_t]), dim=-1)
        predicted_pairs = (ps * pt).sum(-1).add(1).mul(0.5)
        loss_pair = F.smooth_l1_loss(predicted_pairs, target_pairs)
        loss = loss_domain + 0.5 * loss_overlap + loss_pair
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Recompute after optimizer.step to avoid returning stale activations
        # for which module parameters have already been updated.
        with torch.no_grad():
            domain, compatibility = self.heads[domain_index](
                torch.cat((s, t), dim=0)).unbind(-1)
            d, c = torch.sigmoid(domain), torch.sigmoid(compatibility)
            source_mass = 0.10 + 0.90 * torch.sqrt(
                (d[:ns] * c[:ns]).clamp_min(1e-8))
            target_mass = 0.10 + 0.90 * torch.sqrt(
                ((1-d[ns:]) * c[ns:]).clamp_min(1e-8))
        return source_mass.detach(), target_mass.detach(), {
            "domain_loss": loss_domain.detach(),
            "future_loss": loss_overlap.detach(),
            "pair_loss": loss_pair.detach(),
            "source_teacher": future_source.mean(),
            "target_teacher": future_target.mean(),
        }

    @torch.no_grad()
    def pair_embeddings(self, domain_index, source_rep, target_rep):
        """Return detached, unit-normalised student embeddings for OT cost.

        The original joint-feature cost still supplies gradients to SOFTS.
        """
        head = self.pair_heads[domain_index]
        source = F.normalize(head(source_rep.detach()), dim=-1)
        target = F.normalize(head(target_rep.detach()), dim=-1)
        return source.detach(), target.detach()


def conditional_ot_cost(base_dim, predictive_weight=1.0):
    """GeomLoss tensorized cost: original Euclidean p=1 + learned pair cost.

    GeomLoss passes x:[B,N,D], y:[B,M,D] or unbatched equivalents.
    The projected embeddings appended after base_dim are unit normalised.
    """
    def cost(x, y):
        original = (x[..., :, None, :base_dim] -
                    y[..., None, :, :base_dim]).square().sum(-1).clamp_min(1e-12).sqrt()
        if predictive_weight <= 0:
            return original
        a = x[..., :, base_dim:]
        b = y[..., :, base_dim:]
        cosine = (a[..., :, None, :] * b[..., None, :, :]).sum(-1)
        return original + predictive_weight * (1.0 - cosine).clamp(0, 2)
    return cost
