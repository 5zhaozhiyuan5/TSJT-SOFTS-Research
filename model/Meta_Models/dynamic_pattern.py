"""Dynamic-pattern components shared by both PMME training stages.

The module intentionally depends only on PyTorch.  It implements the first
proposal in a way that is cheap enough for the existing PMME mini-batches:

* node-window Top-k routing and bottleneck pattern experts;
* DVGP-weighted dynamic prototypes;
* a small partial OT problem between prototypes; and
* expert-partitioned residual memories with a fixed parameter budget.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckPatternExpert(nn.Module):
    def __init__(self, d_model=128, bottleneck=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )

    def forward(self, x):
        return self.net(x)


class DynamicPatternMoE(nn.Module):
    """Node-window router plus lightweight residual adapters.

    Future labels are deliberately excluded from the router.  Besides the
    backbone representation it sees only three statistics of the normalized
    traffic history: last state, recent slope, and volatility.
    """

    def __init__(self, d_model=128, pred_len=36, num_experts=4,
                 top_k=2, bottleneck=16, router_hidden=64, fusion_mode='add', gate_coef=0.0):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.fusion_mode = fusion_mode
        self.gate_coef = gate_coef
        self.router = nn.Sequential(
            nn.Linear(d_model + 3, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_experts),
        )
        self.experts = nn.ModuleList([
            BottleneckPatternExpert(d_model, bottleneck)
            for _ in range(num_experts)
        ])
        # The MoE must NOT directly alter the backbone forecast: a learnable
        # prediction head that adds to `out` would steal the backbone's
        # regression task and overfit the three target days.  Instead the MoE
        # only refines the *representation* (`enhanced`) that is consumed by
        # the conditional-OT alignment and the residual memory.  This keeps the
        # "dynamic pattern" idea consistent with the PM framework: the experts
        # partition the dynamics, but the forecast is still the backbone's.
        self.expert_scale = nn.Parameter(torch.tensor(0.1))
        # Learnable dynamic-mode anchors: one d_model vector per expert that
        # summarises what this dynamic pattern type looks like.  Used only in
        # Stage-I to compute the route-consistency cost between source/target
        # experts inside the conditional OT (same dynamic type -> small anchor
        # distance -> smaller transport cost -> larger OT coupling).
        self.mode_anchor = nn.Parameter(torch.randn(num_experts, d_model) / math.sqrt(d_model))

    @staticmethod
    def _history_statistics(x_node):
        # x_node: [B, N, L], already normalized by RevIN.
        last = x_node[..., -1]
        recent = x_node[..., -min(12, x_node.shape[-1]):]
        if recent.shape[-1] > 1:
            slope = (recent[..., -1] - recent[..., 0]) / (recent.shape[-1] - 1)
        else:
            slope = torch.zeros_like(last)
        volatility = recent.std(dim=-1, unbiased=False)
        return torch.stack((last, slope, volatility), dim=-1)

    def forward(self, representation, x_node):
        """Return (enhanced_representation, sparse_route, aux).

        ``enhanced`` is the dynamic-pattern-refined representation used for
        cross-city alignment; ``sparse_route`` is the per-node-window expert
        routing probability (the "dynamic pattern" assignment); ``aux`` groups
        the router regularisers (balance / diversity / adapter_l2 / entropy).
        """
        stats = self._history_statistics(x_node)
        logits = self.router(torch.cat((representation, stats), dim=-1))
        dense_prob = torch.softmax(logits, dim=-1)

        top_value, top_index = torch.topk(dense_prob, self.top_k, dim=-1)
        sparse_prob = torch.zeros_like(dense_prob).scatter(-1, top_index, top_value)
        sparse_prob = sparse_prob / sparse_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        expert_outputs = torch.stack(
            [expert(representation) for expert in self.experts], dim=-2
        )  # [B, N, K, D]
        mixed = (sparse_prob.unsqueeze(-1) * expert_outputs).sum(dim=-2)

        # FUSION variants (choose via fusion_mode):
        #  - 'add'  (v31): residual addition, enhanced = rep + scale * mixed.
        #  - 'gate' (new) : multiplicative gated modulation.  The expert mixture
        #    is passed through a sigmoid and used to *scale* each feature
        #    dimension of the backbone representation, so a dynamic pattern
        #    selectively amplifies/suppresses feature channels rather than
        #    shifting them additively.  This changes the MoE<->backbone
        #    interaction (FiLM-style modulation) while still never touching the
        #    forecast head.
        if self.fusion_mode == 'gate':
            gate = torch.sigmoid(mixed)
            enhanced = representation * (1.0 + self.expert_scale * gate)
        elif self.fusion_mode == 'hybrid':
            # Hybrid additive + bounded multiplicative: keep v31's additive
            # shift (rep + scale*mixed) AND add a small FiLM-style modulation
            # term (rep * gate), weighted by gate_coef.  gate_coef=0 recovers
            # v31 exactly.  This lets us add a *little* multiplicative rhythm
            # without abandoning the additive optimum.
            gate = torch.sigmoid(mixed)
            enhanced = representation + self.expert_scale * mixed                        + self.gate_coef * representation * gate
        else:  # 'add' / default == v31
            enhanced = representation + self.expert_scale * mixed

        # Switch-style load balancing, computed from dense probabilities so
        # every router logit receives a gradient.
        importance = dense_prob.mean(dim=(0, 1))
        dispatch = (sparse_prob > 0).to(dense_prob.dtype).mean(dim=(0, 1))
        balance_loss = self.num_experts * torch.sum(importance * dispatch)

        # Discourage all adapters from learning the same residual direction.
        signatures = expert_outputs.mean(dim=(0, 1))
        signatures = F.normalize(signatures, dim=-1)
        similarity = signatures @ signatures.t()
        off_diagonal = ~torch.eye(
            self.num_experts, dtype=torch.bool, device=similarity.device
        )
        diversity_loss = similarity[off_diagonal].abs().mean()

        # Shrinkage penalty on the adapter magnitude so the MoE never overrides
        # the backbone representation (which would implicitly change the
        # forecast).  This is the representation-space analogue of the gating
        # shrinkage in STGM / residual MoE works.
        adapter_l2 = mixed.pow(2).mean()

        aux = {
            "balance": balance_loss,
            "diversity": diversity_loss,
            "adapter_l2": adapter_l2,
            "entropy": -(dense_prob.clamp_min(1e-8).log() * dense_prob).sum(-1).mean(),
            "dense_prob": dense_prob,
        }
        return enhanced, sparse_prob, aux


def _normalise_mass(mass):
    mass = mass.clamp_min(1e-8)
    return mass / mass.sum()


def entropic_partial_transport(cost, source_mass, target_mass,
                               transported_mass=0.8, epsilon=0.05,
                               iterations=50):
    """Differentiable partial OT via a dummy source and target reservoir.

    The dummy-to-dummy edge is forbidden.  Consequently, exactly
    ``transported_mass`` is assigned between real source/target prototypes,
    while the remaining mass is absorbed by the two reservoirs.
    """
    if not 0.0 < transported_mass <= 1.0:
        raise ValueError("transported_mass must be in (0, 1]")
    source_mass = _normalise_mass(source_mass)
    target_mass = _normalise_mass(target_mass)
    k_s, k_t = cost.shape
    leftover = 1.0 - transported_mass

    if leftover < 1e-8:
        aug_cost = cost
        aug_source = source_mass
        aug_target = target_mass
    else:
        aug_cost = cost.new_zeros(k_s + 1, k_t + 1)
        aug_cost[:k_s, :k_t] = cost
        # A very expensive dummy-dummy edge enforces real-real mass m.
        aug_cost[-1, -1] = cost.detach().max() + 1000.0 * epsilon
        aug_source = torch.cat((source_mass, cost.new_tensor([leftover])))
        aug_target = torch.cat((target_mass, cost.new_tensor([leftover])))

    log_kernel = -aug_cost / max(epsilon, 1e-6)
    log_u = torch.zeros_like(aug_source)
    log_v = torch.zeros_like(aug_target)
    log_a = aug_source.clamp_min(1e-12).log()
    log_b = aug_target.clamp_min(1e-12).log()
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    plan = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))
    return plan[:k_s, :k_t]


def prototype_partial_alignment(source_rep, source_y, source_route, source_weight,
                                target_rep, target_y, target_route, target_weight,
                                transported_mass=0.8, epsilon=0.05,
                                iterations=50, label_weight=1.0,
                                mode_anchor=None, route_weight=0.0):
    """Build DVGP-weighted prototypes and return their partial coupling.

    Inputs are flattened node-window samples.  The returned mapped source
    routes live in the target-expert coordinate system and can be appended to
    the original Conditional OT features without constructing an NxM matrix.
    """
    source_weight = _normalise_mass(source_weight)
    target_weight = _normalise_mass(target_weight)

    ws = source_weight.unsqueeze(-1) * source_route
    wt = target_weight.unsqueeze(-1) * target_route
    source_expert_mass = ws.sum(0).clamp_min(1e-8)
    target_expert_mass = wt.sum(0).clamp_min(1e-8)

    source_proto = ws.t() @ source_rep / source_expert_mass.unsqueeze(-1)
    target_proto = wt.t() @ target_rep / target_expert_mass.unsqueeze(-1)
    source_y_proto = ws.t() @ source_y / source_expert_mass.unsqueeze(-1)
    target_y_proto = wt.t() @ target_y / target_expert_mass.unsqueeze(-1)

    feature_cost = torch.cdist(source_proto, target_proto, p=2).pow(2)
    feature_cost = feature_cost / max(source_rep.shape[-1], 1)
    label_cost = torch.cdist(source_y_proto, target_y_proto, p=2).pow(2)
    label_cost = label_cost / max(source_y.shape[-1], 1)
    cost = feature_cost + label_weight * label_cost

    # --- Route-consistency cost (dynamic-pattern sub-division) ---
    # Reward coupling source/target experts whose dynamic pattern type is close.
    # mode_anchor holds K learnable per-expert anchors; L1 distance between
    # anchor i (source) and anchor j (target) is 0 on the diagonal (same type),
    # growing off-diagonal.  Scaling into transport cost makes Sinkhorn give
    # larger mass to same-type pairs ("similar dynamics migrate first") without
    # hard assignment.  route_weight=0 -> original PMME partial OT.
    if mode_anchor is not None and route_weight > 0.0:
        route_cost = torch.cdist(mode_anchor, mode_anchor, p=1)  # [K, K]
        rc_scale = route_cost.detach().median().clamp_min(1e-4)
        cost = cost + route_weight * (route_cost / rc_scale)

    # Scaling makes epsilon portable across backbones and datasets.
    cost_scale = cost.detach().median().clamp_min(1e-4)
    normalised_cost = cost / cost_scale
    plan = entropic_partial_transport(
        normalised_cost,
        source_expert_mass,
        target_expert_mass,
        transported_mass=transported_mass,
        epsilon=epsilon,
        iterations=iterations,
    )
    prototype_loss = (plan * normalised_cost).sum() / max(transported_mass, 1e-6)

    # --- Dynamic-pattern / DVGP consistency (proposal "consistency") ---
    # The DVGP mass ``source_weight``/``target_weight`` measures how strongly a
    # pattern is *shared* across cities.  To make the dynamic router consistent
    # with this notion of "shareability", we re-weight each sample's routing
    # entropy by its DVGP mass: highly-shared patterns get a larger penalty for
    # a high-entropy (ambiguous) routing, which pushes the router to assign a
    # sharp, stable expert for the patterns that actually transfer.  This is the
    # routing analogue of confidence-weighted contrastive alignment.
    def _weighted_entropy(route, weight):
        route = route.clamp_min(1e-8)
        entropy = -(route * route.log()).sum(-1)
        return (entropy * weight.detach()).sum() / weight.detach().sum().clamp_min(1e-8)

    route_entropy_loss = (
        _weighted_entropy(source_route, source_weight)
        + _weighted_entropy(target_route, target_weight)
    ) * 0.5

    row_transition = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mapped_source_route = source_route @ row_transition
    return prototype_loss, plan, mapped_source_route, route_entropy_loss


class ExpertResidualMemory(nn.Module):
    """Expert memories whose total parameters do not exceed PMME's baseline.

    ``reference_slots`` is the original PMME memory size.  Since an explicit
    key/value/confidence slot uses more parameters than its key alone, the
    number of slots is reduced to stay within the original memory budget.
    """

    def __init__(self, reference_slots, num_experts=4, d_model=128, pred_len=36,
                 slots_per_expert=None, retrieval_top_k=32,
                 residual_gate_init=0.0, correction_clip=0.5,
                 horizon_decay=0.0):
        super().__init__()
        baseline_budget = reference_slots * d_model
        per_slot = d_model + pred_len + 1
        total_slots = max(num_experts, baseline_budget // per_slot)
        affordable_slots = max(1, total_slots // num_experts)
        if slots_per_expert is None or slots_per_expert <= 0:
            self.slots_per_expert = affordable_slots
        else:
            # Never exceed the original PMME memory parameter budget.
            self.slots_per_expert = min(int(slots_per_expert), affordable_slots)
        self.total_slots = self.slots_per_expert * num_experts
        self.num_experts = num_experts
        self.d_model = d_model
        self.retrieval_top_k = min(int(retrieval_top_k), self.slots_per_expert)
        # A learnable per-expert magnitude cap on the residual correction.  The
        # correction is passed through ``clip * tanh(delta / clip)`` so that it
        # stays linear for small corrections but saturates for large ones.  This
        # is the "bounded residual" prior from robust regression / residual-MoE
        # works: the memory may nudge the backbone but must not over-shoot on
        # outlier samples (which is what was inflating the 180-min RMSE).
        self.correction_clip = correction_clip
        self.horizon_decay = float(horizon_decay)

        self.keys = nn.Parameter(
            torch.randn(num_experts, self.slots_per_expert, d_model) / math.sqrt(d_model)
        )
        self.values = nn.Parameter(
            torch.zeros(num_experts, self.slots_per_expert, pred_len)
        )
        self.confidence_logits = nn.Parameter(
            torch.zeros(num_experts, self.slots_per_expert)
        )
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        # Horizon-wise expert gates prevent an unreliable source residual from
        # overwhelming the Stage-I prediction.  They start mostly closed and
        # are opened only when residual supervision supports the correction.
        # Far-horizon residuals are noisier (long-range forecast error is
        # dominated by un-predictable dynamics), so the gate bias decays with
        # horizon: near-horizon gates start around neutral, far-horizon gates
        # start clearly closed.  This is the "residual confidence decays with
        # lead time" prior and is what tames the 180-min over-shoot.
        gate_bias = residual_gate_init - 0.6 * horizon_decay * (torch.arange(pred_len, dtype=torch.float32) / max(pred_len - 1, 1))
        self.residual_gate_logits = nn.Parameter(
            gate_bias.unsqueeze(0).repeat(num_experts, 1)
        )
        # Pattern-aware residual confidence.  This is the concrete coupling
        # point between the Stage-I dynamic pattern (expert route) and the
        # Stage-II residual enhancement: *each* dynamic pattern (expert) gets
        # its own horizon-dependent confidence, initialised so far horizons are
        # trusted less, then calibrated from source-city per-pattern residual
        # statistics in ``seed_from_source``.  It acts *before* the tanh cap so
        # a low-confidence pattern cannot over-shoot the 180-min forecast.
        h = torch.arange(pred_len, dtype=torch.float32) / max(pred_len - 1, 1)
        # pattern_confidence is a *buffer*, not a Parameter: the per-pattern
        # horizon confidence is a STATISTIC of the source-city residuals (the
        # "dynamic-pattern consistency" prior), so it must NOT be updated by the
        # Stage-II source/target loss.  If it were trainable, the source
        # prediction loss would re-open the long-horizon gates (rewarding a
        # stronger 180-min correction on the noisy source data) and wash out the
        # calibration -- the exact over-shoot failure we are fixing.  Freezing it
        # makes the Stage-I dynamic pattern -> Stage-II residual-confidence
        # coupling rigid and interpretable.
        init_conf = (1.0 - 0.55 * horizon_decay * h).unsqueeze(0).repeat(num_experts, 1)
        self.register_buffer('pattern_confidence', init_conf)
        self.register_buffer('horizon_confidence', 1.0 - 0.55 * horizon_decay * h)

    def forward(self, representation, route):
        bsz, nodes, dim = representation.shape
        flat_rep = representation.reshape(-1, dim)
        flat_route = route.reshape(-1, self.num_experts)
        correction = flat_rep.new_zeros(flat_rep.shape[0], self.values.shape[-1])

        for expert_id in range(self.num_experts):
            active = torch.nonzero(flat_route[:, expert_id] > 0, as_tuple=False).squeeze(-1)
            if active.numel() == 0:
                continue
            q = self.query(flat_rep.index_select(0, active))
            k = self.key(self.keys[expert_id])
            scores = q @ k.t() / math.sqrt(self.d_model)
            confidence = torch.sigmoid(self.confidence_logits[expert_id]).clamp_min(1e-6)
            scores = scores + confidence.log().unsqueeze(0)
            top_scores, top_indices = torch.topk(
                scores, k=self.retrieval_top_k, dim=-1
            )
            attention = torch.softmax(top_scores, dim=-1)
            selected_values = self.values[expert_id][top_indices]
            expert_delta = (attention.unsqueeze(-1) * selected_values).sum(dim=1)
            gate = torch.sigmoid(self.residual_gate_logits[expert_id])
            # Pattern-aware residual confidence: each dynamic pattern (expert)
            # carries its own horizon-dependent confidence, so a pattern whose
            # long-horizon source residual was noisy is trusted less.  This is
            # the concrete Stage-I <-> Stage-II coupling: the same expert route
            # that partitions Stage-I dynamics also calibrates how strongly its
            # residual correction may act at each lead time.
            pattern_conf = self.pattern_confidence[expert_id]  # already in [0,1]
            expert_delta = expert_delta * (gate * pattern_conf)
            # Bounded residual correction: linear around zero, saturating at
            # +/- correction_clip.  This caps the squared-error contribution of
            # any single over-injected residual while leaving mild nudges intact.
            expert_delta = self.correction_clip * torch.tanh(
                expert_delta / max(self.correction_clip, 1e-4)
            )
            weight = flat_route.index_select(0, active)[:, expert_id].unsqueeze(-1)
            correction.index_add_(0, active, weight * expert_delta)

        return correction.view(bsz, nodes, -1)

    def confidence_regularisation(self):
        confidence = torch.sigmoid(self.confidence_logits)
        # Avoid immediate saturation at zero or one; retrieval utility can
        # subsequently move individual slots away from 0.5.
        return ((confidence - 0.5) ** 2).mean()

    @torch.no_grad()
    def seed_from_source(self, representation, route, residual, weight=None):
        """Initialise the learned memory bank from *source-city* Stage-I
        residuals partitioned by expert route (the dynamic pattern), and
        calibrate the per-pattern horizon confidence from those residuals.

        This is the Stage-I <-> Stage-II coupling: the SAME dynamic pattern
        (expert route) that partitions Stage-I dynamics also determines how
        trustworthy each horizon's residual correction is.  A horizon whose
        source residual is highly variable (un-predictable, e.g. 180-min) gets
        a *lower* confidence, so the memory is conservative exactly where the
        backbone's long-horizon error is dominated by irreducible noise.
        """
        flat_rep = F.normalize(representation.detach().reshape(-1, self.d_model), dim=-1)
        flat_route = route.detach().reshape(-1, self.num_experts)
        flat_res = residual.detach().reshape(-1, self.values.shape[-1])
        pred_len = self.values.shape[-1]
        if weight is not None:
            conf = weight.detach().reshape(-1, pred_len).mean(dim=-1)
        else:
            conf = flat_route.sum(dim=-1)

        # Per-pattern, per-horizon residual std (inverse of trust) over source data.
        pattern_std = flat_res.new_ones(self.num_experts, pred_len)
        pattern_count = flat_res.new_zeros(self.num_experts)
        for expert_id in range(self.num_experts):
            w = flat_route[:, expert_id]
            active = torch.nonzero(w > 0, as_tuple=False).squeeze(-1)
            if active.numel() == 0:
                continue
            pattern_count[expert_id] = active.numel()
            # Weighted std of residuals assigned to this pattern.
            resid = flat_res.index_select(0, active)
            m = resid.mean(dim=0)
            v = ((resid - m) ** 2).mean(dim=0)
            pattern_std[expert_id] = v.sqrt().clamp_min(1e-4)

            take = min(self.slots_per_expert, active.numel())
            chosen_local = torch.topk(w.index_select(0, active), k=take, sorted=False).indices
            chosen = active.index_select(0, chosen_local)
            chosen_res = flat_res.index_select(0, chosen)
            mu = chosen_res.mean(dim=0)
            sd = chosen_res.std(dim=0).clamp_min(1e-4)
            chosen_res = torch.where(chosen_res > mu + 3 * sd, mu + 3 * sd, chosen_res)
            chosen_res = torch.where(chosen_res < mu - 3 * sd, mu - 3 * sd, chosen_res)
            self.keys.data[expert_id, :take].copy_(flat_rep.index_select(0, chosen))
            self.values.data[expert_id, :take].copy_(chosen_res)
            c = conf.index_select(0, chosen)
            c = (c - c.mean()) / (c.std().clamp_min(1e-4) + 1e-6)
            self.confidence_logits.data[expert_id, :take].copy_(c)

        # Pattern confidence: keep NEUTRAL (all ones) so the trainable gate,
        # not a frozen source-std heuristic, calibrates each horizon's trust.
        # The previous 1/(1+rel) scheme wrongly shrank long-horizon confidence
        # to ~0.385 -- the root cause of the 180-min MAE shortfall -- because it
        # penalised the noisy (large-residual) horizons that in fact need the
        # STRONGEST correction.  Long-horizon residual is large not because it
        # is un-correctable, but because the Stage-I backbone errs more there.
        self.pattern_confidence.fill_(1.0)
        return {
            'seeded_slots': int(self.keys.shape[1]),
            'seed_mean': float(self.values.mean().cpu()),
            'pattern_std_mean': float(pattern_std.mean().cpu()),
            'pattern_counts': pattern_count.long().cpu().tolist(),
        }

    @torch.no_grad()
    def finalize_source_seed(self, correction_clip=None):
        """Re-scale the seeded values so their scale matches ``correction_clip``,
        keeping the per-pattern horizon calibration from ``seed_from_source``."""
        clip = correction_clip if correction_clip is not None else self.correction_clip
        scale = self.values.abs().mean().clamp_min(1e-4)
        if scale > clip:
            self.values.data.mul_(float(clip) / float(scale))


class CrossFittedPrototypeResidualMemory(nn.Module):
    """Non-parametric expert residual memory with cross-fitted calibration.

    The learned key/value memory used in the first implementation can fit the
    three target days while moving predictions in the wrong direction on the
    held-out target period.  This memory stores *observed* Stage-I residuals
    instead.  Stage-I routes partition the slots and also mix the retrieved
    corrections, so the dynamic semantics are shared by both stages.

    A small expert-by-horizon gate is estimated with K-fold out-of-fold
    predictions.  It is a registered buffer rather than a trainable parameter;
    consequently no target example is used both as a retrieval slot and as the
    calibration label in the same fold.
    """

    def __init__(self, num_experts=4, d_model=128, pred_len=36,
                 slots_per_expert=512, retrieval_top_k=16,
                 temperature=0.10, residual_clip=3.0):
        super().__init__()
        self.num_experts = int(num_experts)
        self.d_model = int(d_model)
        self.pred_len = int(pred_len)
        self.slots_per_expert = int(slots_per_expert)
        self.retrieval_top_k = int(retrieval_top_k)
        self.temperature = float(temperature)
        self.residual_clip = float(residual_clip)

        self.register_buffer(
            'keys', torch.zeros(self.num_experts, self.slots_per_expert, self.d_model)
        )
        self.register_buffer(
            'values', torch.zeros(self.num_experts, self.slots_per_expert, self.pred_len)
        )
        self.register_buffer(
            'valid', torch.zeros(self.num_experts, self.slots_per_expert, dtype=torch.bool)
        )
        self.register_buffer('gates', torch.zeros(self.num_experts, self.pred_len))
        self.register_buffer('slot_counts', torch.zeros(self.num_experts, dtype=torch.long))
        self.register_buffer('ready', torch.tensor(False, dtype=torch.bool))

    def _build_bank(self, representation, route, residual, batch_mask=None):
        """Build deterministic high-confidence expert banks."""
        if batch_mask is not None:
            representation = representation[batch_mask]
            route = route[batch_mask]
            residual = residual[batch_mask]

        flat_rep = F.normalize(
            representation.detach().reshape(-1, self.d_model), dim=-1
        )
        flat_route = route.detach().reshape(-1, self.num_experts)
        flat_residual = residual.detach().reshape(-1, self.pred_len).clamp(
            -self.residual_clip, self.residual_clip
        )

        self.keys.zero_()
        self.values.zero_()
        self.valid.zero_()
        self.slot_counts.zero_()
        for expert_id in range(self.num_experts):
            weights = flat_route[:, expert_id]
            active = torch.nonzero(weights > 0, as_tuple=False).squeeze(-1)
            if active.numel() == 0:
                continue
            take = min(self.slots_per_expert, active.numel())
            confidence = weights.index_select(0, active)
            chosen_local = torch.topk(confidence, k=take, sorted=False).indices
            chosen = active.index_select(0, chosen_local)
            self.keys[expert_id, :take].copy_(flat_rep.index_select(0, chosen))
            self.values[expert_id, :take].copy_(flat_residual.index_select(0, chosen))
            self.valid[expert_id, :take] = True
            self.slot_counts[expert_id] = take

    def expert_contributions(self, representation, route):
        """Return route-weighted retrievals before calibration gates."""
        bsz, nodes, _ = representation.shape
        flat_rep = F.normalize(representation.reshape(-1, self.d_model), dim=-1)
        flat_route = route.reshape(-1, self.num_experts)
        contributions = flat_rep.new_zeros(
            flat_rep.shape[0], self.num_experts, self.pred_len
        )

        for expert_id in range(self.num_experts):
            count = int(self.slot_counts[expert_id].item())
            if count == 0:
                continue
            active = torch.nonzero(
                flat_route[:, expert_id] > 0, as_tuple=False
            ).squeeze(-1)
            if active.numel() == 0:
                continue
            queries = flat_rep.index_select(0, active)
            similarities = queries @ self.keys[expert_id, :count].t()
            top_k = min(self.retrieval_top_k, count)
            top_scores, top_indices = torch.topk(similarities, k=top_k, dim=-1)
            attention = torch.softmax(
                top_scores / max(self.temperature, 1e-4), dim=-1
            )
            selected_values = self.values[expert_id, :count][top_indices]
            retrieved = (attention.unsqueeze(-1) * selected_values).sum(dim=1)
            route_weight = flat_route.index_select(0, active)[:, expert_id]
            contributions[active, expert_id] = route_weight.unsqueeze(-1) * retrieved

        return contributions.view(bsz, nodes, self.num_experts, self.pred_len)

    @torch.no_grad()
    def fit(self, representation, route, residual, mae_weight=None, folds=3,
            ridge=0.10, gate_prior=0.25, gate_max=1.25):
        """Fit robust expert/horizon gates from out-of-fold predictions.

        Calibration minimizes weighted absolute error rather than normalized
        MSE.  ``mae_weight`` converts a normalized residual back to its RevIN
        scale, matching the MAE used for checkpoint selection and reporting.

        The original ridge term shrank every gate back toward ``gate_prior``
        (0.25), which empirically pinned all gates to 0.125-0.25 and disabled
        the residual memory.  We replace constant-shrinkage-by-prior with two
        fixes drawn from memory-augmented calibration works:

        1. a soft *minimum* gate: zero-gating is only accepted when it beats a
           small positive offset by a clear margin (avoids the degenerate
           closed-gate solution for horizons where memory is genuinely useful);
        2. a *post-fit* shrinkage towards zero (not towards the prior), applied
           only to the negligible gates, so the large useful gates are kept.
        """
        batch_size = representation.shape[0]
        folds = max(2, min(int(folds), batch_size))
        fold_id = torch.arange(batch_size, device=representation.device) % folds
        contribution_parts = []
        residual_parts = []
        weight_parts = []
        if mae_weight is None:
            mae_weight = torch.ones_like(residual)

        for fold in range(folds):
            support_mask = fold_id != fold
            query_mask = ~support_mask
            self._build_bank(representation, route, residual, support_mask)
            contribution_parts.append(
                self.expert_contributions(
                    representation[query_mask], route[query_mask]
                ).reshape(-1, self.num_experts, self.pred_len)
            )
            residual_parts.append(
                residual[query_mask].reshape(-1, self.pred_len)
            )
            weight_parts.append(
                mae_weight[query_mask].reshape(-1, self.pred_len)
            )

        contributions = torch.cat(contribution_parts, dim=0)
        targets = torch.cat(residual_parts, dim=0)
        weights = torch.cat(weight_parts, dim=0)
        gates = targets.new_full(
            (self.num_experts, self.pred_len), float(gate_prior)
        )
        # Finer grid so the search is not forced onto coarse quantiles.
        candidates = torch.linspace(
            0.0, float(gate_max), 21,
            device=targets.device,
            dtype=targets.dtype,
        )

        for horizon in range(self.pred_len):
            design = contributions[:, :, horizon]
            target = targets[:, horizon]
            sample_weight = weights[:, horizon]
            gate = gates[:, horizon].clone()
            prediction = design @ gate

            # Coordinate descent over the four expert gates.  The ridge now
            # penalises |gate| (shrink towards zero) rather than
            # (gate-prior)^2, and is *small*: it only breaks ties among
            # near-flat solutions instead of dominating the MAE term.
            for _ in range(2):
                for expert_id in range(self.num_experts):
                    base = prediction - design[:, expert_id] * gate[expert_id]
                    candidate_prediction = (
                        base.unsqueeze(-1)
                        + design[:, expert_id].unsqueeze(-1)
                        * candidates.unsqueeze(0)
                    )
                    errors = (target.unsqueeze(-1) - candidate_prediction).abs()
                    scores = (
                        (sample_weight.unsqueeze(-1) * errors).sum(dim=0)
                        / sample_weight.sum().clamp_min(1e-6)
                    )
                    # Shrink-towards-zero regularisation (cf. ridge on |gate|).
                    # The original `(gate - prior)^2` with a large ``ridge``
                    # pinned every gate near the prior and disabled the memory.
                    # We keep only a *weak* shrinkage to break ties among
                    # near-flat solutions, and no longer bias towards the prior.
                    scores = scores + float(ridge) * (
                        candidates
                    ).pow(2)
                    best = candidates[scores.argmin()]
                    prediction = base + design[:, expert_id] * best
                    gate[expert_id] = best
            gates[:, horizon] = gate

        self.gates.copy_(gates)
        self._build_bank(representation, route, residual)
        self.ready.fill_(True)
        return {
            'slot_counts': self.slot_counts.detach().cpu().tolist(),
            'gate_mean': float(self.gates.mean().cpu()),
            'gate_min': float(self.gates.min().cpu()),
            'gate_max': float(self.gates.max().cpu()),
        }

    def forward(self, representation, route):
        if not bool(self.ready.item()):
            return representation.new_zeros(
                representation.shape[0], representation.shape[1], self.pred_len
            )
        contributions = self.expert_contributions(representation, route)
        return (contributions * self.gates.view(1, 1, self.num_experts, self.pred_len)).sum(dim=2)


class PatternPrototypeResidualMemory(nn.Module):
    """Pattern-prototype residual memory (Stage-II residual enhancement).

    Coupled to Stage-I: the per-pattern key/value reuse the SAME DVGP
    route-weighted prototype geometry as Stage-I's ``prototype_partial_alignment``,
    so a query retrieves the residual of the pattern(s) it structurally
    resembles.  The residual migrates at *pattern* granularity.

    The fix for the "correction == 0" failure of the median-based version:

    1. Store the route-weighted *mean* residual prototype (NOT the median).  The
       median centres the signed error around ~0 and erases the transferable
       systematic bias; the mean keeps the actual error scale (~0.1-0.4 in the
       RevIN-normalised space) so the correction is the right order of magnitude.

    2. Remove the MAD amplitude shrinkage: it had compressed the correction to
       ~1e-4, i.e. it silently disabled the residual enhancement entirely
       (finetune-before == finetune-after in the test metrics).

    3. A trainable per-pattern gate (initialised OPEN at ``residual_gate_init``)
       that the Stage-II few-shot loss calibrates, so the backbone decides how
       much of each pattern's residual to transfer -- rather than a frozen
       hand-picked constant.

    4. Keep the near-horizon floor (10-min correction ~0) and a tanh cap to
       bound the far-horizon over-shoot.
    """

    def __init__(self, num_experts=4, d_model=128, pred_len=36,
                 residual_gate_init=1.0, correction_clip=0.5,
                 horizon_decay=0.45, floor_cut=0.05):
        super().__init__()
        self.num_experts = int(num_experts)
        self.d_model = int(d_model)
        self.pred_len = int(pred_len)
        self.correction_clip = float(correction_clip)
        self.floor_cut = float(floor_cut)

        # Buffers seeded from source-city statistics (NOT trainable): a pure
        # transfer of Stage-I's dynamic-pattern geometry.
        self.register_buffer('keys', torch.zeros(num_experts, d_model))
        self.register_buffer('values', torch.zeros(num_experts, pred_len))   # mean residual proto
        self.register_buffer('counts', torch.zeros(num_experts))
        self.register_buffer('ready', torch.tensor(False))

        # Near-horizon floor: 0 on the accurate short horizon, rising smoothly
        # to 1 on the far horizon.
        h = torch.arange(pred_len, dtype=torch.float32) / max(pred_len - 1, 1)
        ramp = torch.clamp((h - self.floor_cut) / (1.0 - self.floor_cut + 1e-8), 0.0, 1.0)
        floor = 0.5 - 0.5 * torch.cos(ramp * math.pi)          # [P]
        self.register_buffer('horizon_floor', floor)

        # Trainable per-pattern gate, initialised OPEN so the residual can be
        # calibrated by the few-shot target loss (rather than frozen shut).
        gate_init = torch.full((num_experts,), residual_gate_init)
        self.residual_gate_logits = nn.Parameter(gate_init)

    @torch.no_grad()
    def seed_from_source(self, representation, route, residual, weight=None):
        """Build per-pattern mean-residual prototypes from source residuals."""
        flat_rep = F.normalize(representation, dim=-1)
        flat_route = route
        flat_res = residual
        if weight is not None:
            w = weight.mean(dim=-1)
            w = w / w.sum().clamp_min(1e-8)
        else:
            w = flat_route.sum(dim=-1)
            w = w / w.sum().clamp_min(1e-8)

        self.keys.zero_()
        self.values.zero_()
        self.counts.zero_()
        for expert_id in range(self.num_experts):
            rw = flat_route[:, expert_id]
            mask = rw > 0
            if mask.sum() == 0:
                continue
            wr = (rw * w)[mask]
            wr = wr / wr.sum().clamp_min(1e-8)
            self.keys[expert_id] = (flat_rep[mask] * wr.unsqueeze(-1)).sum(0)
            # MEAN residual prototype keeps the real error scale.
            self.values[expert_id] = (flat_res[mask] * wr.unsqueeze(-1)).sum(0)
            self.counts[expert_id] = mask.sum()
        self.keys = F.normalize(self.keys, dim=-1)
        self.ready.fill_(True)
        return {
            'proto_counts': self.counts.long().cpu().tolist(),
            'residual_proto_norm': float(self.values.norm(dim=-1).mean().cpu()),
        }

    @torch.no_grad()
    def finalize_source_seed(self, correction_clip=None):
        # NO aggressive re-scaling: the mean-residual prototype already lives at
        # the correct error scale.  Only guard against a degenerate huge case.
        clip = correction_clip if correction_clip is not None else self.correction_clip
        scale = self.values.abs().mean().clamp_min(1e-4)
        if scale > 5.0:
            self.values.mul_(5.0 / float(scale))

    def confidence_regularisation(self):
        return self.residual_gate_logits.new_zeros(())

    def forward(self, representation, route):
        if not bool(self.ready.item()):
            return representation.new_zeros(
                representation.shape[0], representation.shape[1], self.pred_len
            )
        bsz, nodes, dim = representation.shape
        # Pattern assignment: use the stage-I *route* (the dynamic-pattern
        # sparse routing probability) directly as the aggregation weight over the
        # per-pattern residual prototypes.  This keeps the residual at the true
        # error scale (the L2-normalised similarity was compressing it by ~100x
        # and silently disabling the enhancement) AND is the most faithful form
        # of "dynamic-pattern consistency": a sample receives the residual of the
        # pattern(s) Stage-I actually assigned it to.
        flat_route = route.reshape(-1, self.num_experts)                # [N, E]

        gate = torch.sigmoid(self.residual_gate_logits).unsqueeze(-1)   # [E,1]
        amplitude = gate * self.horizon_floor                            # [E, P]
        residual_proto = self.values * amplitude                        # [E, P]
        raw = flat_route @ residual_proto                               # [N, P]

        clip = self.correction_clip
        flat_corr = clip * torch.tanh(raw / max(clip, 1e-4))
        return flat_corr.view(bsz, nodes, self.pred_len)

