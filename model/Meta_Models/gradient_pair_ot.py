"""P1/P2/P3 pairwise OT modifiers and P4 task-aware city/group assignment.

All variants keep the original F2 source marginal and GeomLoss Sinkhorn.
A zero strength MUST use the original unmodified SamplesLoss call.
Gradient utilities are detached: they guide pair selection but do not
backpropagate through second-order task gradients.
"""
import itertools
import math

import torch
import torch.nn.functional as F


VARIANTS = ("f2", "p1", "p2", "p3", "p4")


def _flat(v):
    return v.reshape(-1, v.shape[-1])


def _normalized(v):
    return F.normalize(v.float(), dim=-1, eps=1e-8)


def _head_pair_cos(s_feat, s_delta, t_feat, t_delta):
    """Exact cosine of rank-one projection2 weight gradients, no 4608-D expansion."""
    return (_normalized(s_feat) @ _normalized(t_feat).T) * (
        _normalized(s_delta) @ _normalized(t_delta).T
    )


def _anchor_descriptors(s_rep, t_rep, s_delta, t_delta, alpha, anchor_count):
    """Find mutual high-gradient-agreement anchors in small candidate pools.

    Candidate sources are prioritized using unchanged F2 weights. Target
    candidates use target loss-gradient magnitude. Only K anchor pairs are
    selected; all sample descriptors are distances to same-index anchors.
    """
    with torch.no_grad():
        ns, nt = s_rep.shape[0], t_rep.shape[0]
        candidates = min(max(4 * anchor_count, 16), ns, nt)
        s_ids = alpha.detach().topk(candidates).indices
        t_scores = t_delta.detach().norm(dim=-1) * t_rep.detach().norm(dim=-1)
        t_ids = t_scores.topk(candidates).indices
        sr = s_rep.detach().index_select(0, s_ids)
        tr = t_rep.detach().index_select(0, t_ids)
        gd = _head_pair_cos(sr, s_delta.detach()[s_ids],
                            tr, t_delta.detach()[t_ids])
        dist = torch.cdist(sr.float(), tr.float())
        dist = dist / dist.detach().median().clamp_min(1e-6)
        score = 0.8 * gd - 0.2 * dist
        # Prefer reciprocal best-matches; greedily fill the remaining anchors.
        reciprocal = [
            (i, int(score[i].argmax().item()))
            for i in range(candidates)
            if int(score[:, int(score[i].argmax().item())].argmax().item()) == i
        ]
        reciprocal.sort(key=lambda pair: float(score[pair[0], pair[1]]),
                        reverse=True)
        matches, seen_s, seen_t = [], set(), set()
        for i, j in reciprocal:
            if i not in seen_s and j not in seen_t:
                matches.append((i, j))
                seen_s.add(i)
                seen_t.add(j)
            if len(matches) >= anchor_count:
                break
        if len(matches) < anchor_count:
            ranking = torch.argsort(score.reshape(-1), descending=True)
            for pos in ranking.tolist():
                i, j = divmod(pos, candidates)
                if i not in seen_s and j not in seen_t:
                    matches.append((i, j))
                    seen_s.add(i)
                    seen_t.add(j)
                if len(matches) >= anchor_count:
                    break
        src_anchors = sr[[i for i, _ in matches]]
        tgt_anchors = tr[[j for _, j in matches]]
    # Important: gradients DO flow through distances of every representation
    # to detached anchors, but not through anchor selection itself.
    rs = torch.cdist(s_rep.float(), src_anchors.float())
    rt = torch.cdist(t_rep.float(), tgt_anchors.float())
    with torch.no_grad():
        denom = torch.cat([rs.detach(), rt.detach()], dim=0).median(dim=0).values
        denom = denom.clamp_min(1e-3)
    sample_indices = [
        (int(s_ids[i].item()), int(t_ids[j].item())) for i, j in matches
    ]
    return (torch.log1p(rs / denom),
            torch.log1p(rt / denom), len(matches), sample_indices)


def _response(history, future):
    """3 horizons x (relative change, local slope, change volatility).

    History is globally standardized city speed [B,N,288]; future is
    standardized using the very same city mean/std [B,N,36].
    Only the original 3-day target support labels are used.
    """
    if history.shape[-1] < 12 or future.shape[-1] != 36:
        raise ValueError("Expected >=12 history and 36 future steps")
    base = history[..., -12:].float().mean(dim=-1)
    variability = history.float().std(dim=-1, unbiased=False).clamp_min(0.2)
    features = []
    for start in (0, 12, 24):
        part = future[..., start:start + 12].float()
        rel = (part - base.unsqueeze(-1)) / variability.unsqueeze(-1)
        change = rel.mean(dim=-1)
        slope = (rel[..., -4:].mean(dim=-1) -
                 rel[..., :4].mean(dim=-1))
        volatility = rel.std(dim=-1, unbiased=False)
        features.append(torch.stack([change, slope, volatility], dim=-1))
    return torch.stack(features, dim=1).detach()  # [samples, 3, 3]


def _target_band_confidence(t_feat, t_delta):
    """Target support split-half agreement, no new estimator network."""
    if t_feat.shape[0] < 4:
        return [1.0, 1.0, 1.0]
    cut = t_feat.shape[0] // 2
    values = []
    with torch.no_grad():
        for a in (0, 12, 24):
            g1 = torch.einsum("nd,nh->hd", t_feat[:cut].float(),
                              t_delta[:cut, a:a + 12].float())
            g2 = torch.einsum("nd,nh->hd", t_feat[cut:].float(),
                              t_delta[cut:, a:a + 12].float())
            sim = F.cosine_similarity(g1.flatten(), g2.flatten(), dim=0)
            values.append(float(sim.clamp(0, 1).item()))
    return values


def build_pairwise_cost(variant, x, y, source_rep, target_rep,
                        source_delta, target_delta, source_alpha,
                        source_history=None, source_future=None,
                        target_history=None, target_future=None,
                        strength=0.02, anchor_count=8, block_size=256):
    """Return (GeomLoss tensorized cost callback, diagnostics).

    x,y are original joint [representation, sqrt(100)*future] points.
    The original p=1 ground cost is Euclidean distance in this space.
    Callback adds a *dimensionless* pair penalty times a fraction of the
    detached reference Euclidean scale. The built-in original cost is
    always used when strength==0, not this callback.
    """
    if variant not in ("p1", "p2", "p3"):
        raise ValueError("Unsupported pairwise variant: " + str(variant))
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("build_pairwise_cost requires positive finite strength")
    if x.ndim != 2 or y.ndim != 2 or x.shape[-1] != y.shape[-1]:
        raise ValueError("Expected two flattened equal-dimensional point sets")
    if source_rep.shape[0] != x.shape[0] or target_rep.shape[0] != y.shape[0]:
        raise ValueError("Source/target representations do not match OT points")
    if source_delta.shape != (x.shape[0], 36) or target_delta.shape != (y.shape[0], 36):
        raise ValueError("Expected per-sample 36-step output gradients")
    if block_size < 1 or anchor_count < 1:
        raise ValueError("Invalid OT block size or anchor count")
    with torch.no_grad():
        ref_cost = torch.cdist(x[:min(48, len(x))].float(),
                               y[:min(48, len(y))].float()).median().clamp_min(1e-3)
        factor = float(ref_cost.item()) * strength
    stats = {"variant": variant, "strength": strength,
             "reference_ground_cost": float(ref_cost.item())}

    if variant == "p1":
        rs, rt, count, anchor_indices = _anchor_descriptors(
            source_rep, target_rep, source_delta, target_delta,
            source_alpha, anchor_count)
        stats["anchors"] = count
        stats["anchor_indices_source_target"] = anchor_indices

        def penalty_rows(a, b):
            return torch.cdist(rs[a:b].unsqueeze(0),
                               rt.unsqueeze(0), p=1).squeeze(0) / count

    else:
        with torch.no_grad():
            sf, tf = _normalized(source_rep.detach()), _normalized(target_rep.detach())
            sd = source_delta.detach().float()
            td = target_delta.detach().float()
            # Per-band output deltas are normalized independently.
            sband = [_normalized(sd[:, i:i + 12]) for i in (0, 12, 24)]
            tband = [_normalized(td[:, i:i + 12]) for i in (0, 12, 24)]
            if variant == "p2":
                for name, value in (("source_history", source_history),
                                    ("source_future", source_future),
                                    ("target_history", target_history),
                                    ("target_future", target_future)):
                    if value is None:
                        raise ValueError("P2 requires " + name)
                rs = _response(source_history, source_future)
                rt = _response(target_history, target_future)
                if rs.shape[0] != x.shape[0] or rt.shape[0] != y.shape[0]:
                    raise ValueError("P2 dynamic response / OT point mismatch")
            else:
                confidence = _target_band_confidence(tf, td)
                confidence = torch.tensor(confidence, device=x.device,
                                          dtype=torch.float32) + 0.15
                confidence = confidence / confidence.sum()
                stats["target_band_confidence"] = confidence.tolist()

        def penalty_rows(a, b):
            # Pairwise gradient cosine for the projection2 weight matrix:
            # cos(outer(delta_s,feat_s),outer(delta_t,feat_t))
            with torch.no_grad():
                feature_cos = sf[a:b] @ tf.T
                agreements = [
                    (feature_cos * (sband[h][a:b] @ tband[h].T)).clamp(-1, 1)
                    for h in range(3)
                ]
                if variant == "p2":
                    total = torch.zeros_like(agreements[0])
                    for h in range(3):
                        dist = torch.cdist(rs[a:b, h], rt[:, h], p=1) / 3.0
                        dist = dist / (1.0 + dist)
                        agreement = (1.0 + agreements[h]) * 0.5
                        total += ((1.0 - agreement) +
                                  agreement * dist) / 3.0
                    return total
                # Stable soft minimum; lower confidence reduces impact
                # of an unreliable horizon without discarding it entirely.
                logits = torch.stack(agreements, dim=0)
                worst = -0.15 * torch.logsumexp(
                    (-logits / 0.15) + confidence.log().view(3, 1, 1),
                    dim=0)
                return F.relu(-worst)

    # These are diagnostics of the *candidate cost*, not an explicit
    # Sinkhorn transport matrix. They are evaluated on fixed small points.
    with torch.no_grad():
        sampled = penalty_rows(0, min(32, x.shape[0]))
        sampled = sampled[:, :min(32, y.shape[0])]
        stats["penalty_sample_mean"] = float(sampled.mean().item())
        stats["penalty_sample_std"] = float(
            sampled.std(unbiased=False).item()
        )

    def cost(xx, yy):
        # GeomLoss tensorized backend invokes with [batch,N,D]/[batch,M,D].
        if xx.ndim != 3 or yy.ndim != 3 or xx.shape[0] != 1 or yy.shape[0] != 1:
            raise ValueError("P1/P2/P3 require one unbatched OT problem")
        if xx.shape[1] != x.shape[0] or yy.shape[1] != y.shape[0]:
            raise ValueError("GeomLoss point count differs from prepared descriptors")
        rows = []
        # Build [N,M] in row chunks to avoid a second N*M intermediate.
        for start in range(0, xx.shape[1], block_size):
            stop = min(xx.shape[1], start + block_size)
            base = torch.cdist(xx[:, start:stop, :], yy, p=2)
            extra = penalty_rows(start, stop).to(base.dtype).unsqueeze(0)
            rows.append(base + factor * extra)
        return torch.cat(rows, dim=1)

    return cost, stats


def assign_city_target_groups(source_features, source_deltas,
                              target_feature_groups, target_delta_groups,
                              city_keys, ema=None, counts=None,
                              momentum=0.8, repeat_penalty=0.08):
    """Balanced 3x3 assignment, source in input order -> target-group index.

    Optimizes an EMA of full-head gradient cosine with a count-based
    repeated-pair penalty. Persistent EMA/count dictionaries are caller-owned.
    No labels beyond target support are used.
    """
    if len(source_features) != 3 or len(target_feature_groups) != 3:
        raise ValueError("P4 expects exactly three source cities and target groups")
    if len(source_deltas) != 3 or len(target_delta_groups) != 3 or len(city_keys) != 3:
        raise ValueError("Inconsistent P4 source/target inputs")
    if len(set(city_keys)) != 3:
        raise ValueError("P4 requires distinct city names")
    if not (0 <= momentum < 1 and repeat_penalty >= 0):
        raise ValueError("Invalid P4 smoothing configuration")
    if ema is None:
        ema = {}
    if counts is None:
        counts = {}
    with torch.no_grad():
        source = [
            _normalized(torch.einsum("bnh,bnd->hd", d.detach().float(),
                                     f.detach().float()).reshape(1, -1))[0]
            for f, d in zip(source_features, source_deltas)
        ]
        target = [
            _normalized(torch.einsum("bnh,bnd->hd", d.detach().float(),
                                     f.detach().float()).reshape(1, -1))[0]
            for f, d in zip(target_feature_groups, target_delta_groups)
        ]
        scores = torch.stack(source) @ torch.stack(target).T
        balanced_scores = torch.empty_like(scores)
        for k, city in enumerate(city_keys):
            seen = sum(counts.get((city, g), 0) for g in range(3))
            for g in range(3):
                key = (city, g)
                current = float(scores[k, g].item())
                smoothed = (momentum * ema[key] + (1 - momentum) * current
                            if key in ema else current)
                ema[key] = smoothed
                balanced_scores[k, g] = smoothed - (
                    repeat_penalty * counts.get(key, 0) / max(1, seen)
                )
        assignments = list(itertools.permutations(range(3)))
        winner = max(assignments, key=lambda p: sum(
            float(balanced_scores[k, p[k]].item()) for k in range(3)))
        for k, g in enumerate(winner):
            key = (city_keys[k], g)
            counts[key] = counts.get(key, 0) + 1
    return list(winner), {
        "city_keys": list(city_keys),
        "assignment": list(winner),
        "raw_scores": scores.cpu().tolist(),
        "adjusted_scores": balanced_scores.cpu().tolist()
    }
