"""MLCEA: Multi-scale Local Conditional Evolution Alignment (Stage I only).

The official SOFTS, GP-based DVGP marginals, Stage-II and objective are retained.
Only a pairwise, label-supervised *training-time* OT cost is added.

Inputs: normalized historic windows [B,N,288] and the official model's
RevIN-space training labels [B,N,36]. Chengdu/Shenzhen have already been
linearly interpolated onto the 5-minute grid by the official dataset loader.
No target test labels are accessed here.

Architecture:
  (1) six multiscale history conditions + six future responses;
  (2) learned positive diagonal historical metric;
  (3) leave-node-out local differentiable weighted ridge with source prior;
  (4) bidirectional cross-city conditional explanation;
  (5) improvement over shared source prior + support reliability;
  (6) alpha/beta interaction-centering and additive OT cost.

NOTE: Batch-local neighbours may still have correlated future windows on
different traffic nodes. Leave-node-out prevents direct label self-fitting,
but is NOT a fully independent chronological cross-validation guarantee.
"""
import csv
import math
from pathlib import Path

import torch
from torch import nn


class MLCEA(nn.Module):
    """Two tiny trainable vectors; per-sample operators use ridge solves."""

    def __init__(self, anchors=128, neighbors=16, ridge=2.0,
                 metric_temperature=1.0, response_mix=0.5,
                 event_gamma=0.2, score_beta=0.25, gain_temp=0.35,
                 support_gamma=0.35, response_floor=0.02,
                 history_dim=6, horizon_dim=6,
                 neighbor_uniform_mix=0.35, min_effective_k=8.0,
                 adaptive_ridge=True, time_block_mode="diagnostic",
                 temporal_exclusion_steps=324):
        super().__init__()
        if history_dim != 6 or horizon_dim != 6:
            raise ValueError("MLCEA v1 uses six history conditions and six future anchors")
        self.metric_logits = nn.Parameter(torch.zeros(history_dim))
        self.horizon_logits = nn.Parameter(torch.zeros(3))
        self.anchors = int(anchors)
        self.neighbors = int(neighbors)
        self.ridge = float(ridge)
        self.metric_temperature = float(metric_temperature)
        self.response_mix = float(response_mix)
        self.event_gamma = float(event_gamma)
        self.score_beta = float(score_beta)
        self.gain_temp = float(gain_temp)
        self.support_gamma = float(support_gamma)
        self.response_floor = float(response_floor)
        self.history_dim = int(history_dim)
        self.horizon_dim = int(horizon_dim)
        self.neighbor_uniform_mix = float(neighbor_uniform_mix)
        self.min_effective_k = float(min_effective_k)
        self.adaptive_ridge = bool(adaptive_ridge)
        if time_block_mode not in ("diagnostic", "strict"):
            raise ValueError("time_block_mode must be diagnostic or strict")
        self.time_block_mode = time_block_mode
        self.temporal_exclusion_steps = int(temporal_exclusion_steps)
        if not 0 <= self.neighbor_uniform_mix < 1:
            raise ValueError("neighbor_uniform_mix must be in [0,1)")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")


    def metric_weights(self):
        return self.history_dim * torch.softmax(self.metric_logits, dim=0)

    def make_snapshot(self, history, future_rev, window_indices=None, window_stride=12):
        """Output node-wise x:[B*N,7], r:[B*N,6], event:[B*N,6]."""
        h = torch.as_tensor(history, device=future_rev.device,
                            dtype=future_rev.dtype)
        if h.ndim != 3 or future_rev.ndim != 3:
            raise ValueError("Expected [B,N,L] histories and [B,N,H] futures")
        if h.shape[:2] != future_rev.shape[:2] or h.shape[-1] < 192 or future_rev.shape[-1] < 36:
            raise ValueError("MLCEA expects aligned batch/node sizes, >=192 histories and >=36 futures")
        n_batch, n_nodes, _ = h.shape
        if window_indices is None:
            # Synthetic tests may lack true chronology metadata.
            windows_abs = torch.full((n_batch,), -1, dtype=torch.long,
                                     device=h.device)
        else:
            windows_abs = torch.as_tensor(window_indices, device=h.device,
                                           dtype=torch.long).reshape(-1)
            if windows_abs.numel() != n_batch:
                raise ValueError("window_indices count differs from batch size")
        stride = int(window_stride)
        if stride < 1:
            raise ValueError("window_stride must be positive")
        h = h[..., -288:]
        center = h.mean(dim=-1, keepdim=True).detach()
        stdev = h.var(dim=-1, keepdim=True, unbiased=False).add(1e-9).sqrt().detach()
        z = ((h - center) / stdev).clamp(-8, 8)
        windows = (6, 12, 24, 48, 96)
        conditions = [
            z[..., -k:].mean(-1) - z[..., -2 * k:-k].mean(-1)
            for k in windows
        ]
        conditions.append(z[..., -1] - z[..., -12:].mean(-1))
        raw = torch.stack(conditions, dim=-1).clamp(-6, 6)
        raw = raw.reshape(-1, self.history_dim).detach()
        x = torch.cat((torch.ones_like(raw[:, :1]), raw), dim=-1)
        idx = torch.as_tensor([5, 11, 17, 23, 29, 35], device=z.device)
        # future_rev has already been normalized with the same RevIN
        # statistics inside the official PatchFSL.forward.
        unbounded_response = (
            future_rev.index_select(-1, idx) - z[..., -1:]
        ).reshape(-1, 6).detach()
        r = unbounded_response.clamp(-6, 6)
        diff = torch.cat((r[:, :1], r[:, 1:] - r[:, :-1]), dim=-1)
        curvature = torch.cat((torch.zeros_like(diff[:, :1]),
                               diff[:, 1:] - diff[:, :-1]), dim=-1)
        event = (diff.abs() + curvature.abs()).detach()
        event = event / (event.mean(-1, keepdim=True) + 0.25)
        event = event.clamp(0, 4)
        return dict(
            x=x, r=r, event=event,
            node=torch.arange(n_nodes, device=z.device).repeat(n_batch),
            batch=torch.arange(n_batch, device=z.device).repeat_interleave(n_nodes),
            time_index=windows_abs.repeat_interleave(n_nodes),
            time_stride=torch.full((n_batch * n_nodes,), stride, device=z.device,
                                   dtype=torch.long),
            history_std=stdev.reshape(-1).detach(),
            raw_response_abs_max=unbounded_response.abs().amax(dim=-1),
            response_clipped_rate=(unbounded_response.abs() > 6).float().mean(-1),
        )

    def split_by_batch(self, snapshot, batch_sizes, node_count):
        """Split the flattened [B*N,...] snapshot along batch dimension."""
        lengths = [int(b) * int(node_count) for b in batch_sizes]
        if sum(lengths) != snapshot["x"].size(0):
            raise ValueError("Incorrect target source/target split dimensions")
        pieces = {k: v.split(lengths, dim=0) for k, v in snapshot.items()}
        return [{k: pieces[k][i] for k in snapshot} for i in range(len(lengths))]

    def fit_reference(self, source_snapshots):
        """Source-only global ridge prior, balanced by source-city subsampling."""
        pools = []
        for snap in source_snapshots:
            x, r = snap["x"], snap["r"]
            if x.size(0) > 512:
                ids = torch.linspace(0, x.size(0) - 1, steps=512,
                                     device=x.device).long()
                x, r = x[ids], r[ids]
            pools.append((x, r))
        if not pools:
            raise ValueError("fit_reference needs source-city snapshots")
        X = torch.cat([v[0] for v in pools], dim=0)
        R = torch.cat([v[1] for v in pools], dim=0)
        # Scale by count, otherwise prior strength is source-size dependent.
        A = X.T @ X / X.size(0)
        B = R.T @ X / X.size(0)
        eye = torch.eye(X.size(1), device=X.device, dtype=X.dtype)
        # Do not penalize the bias (first) component as aggressively.
        ridge = torch.diag(torch.tensor([0.25] + [1.0] * (X.size(1) - 1),
                                        device=X.device, dtype=X.dtype))
        return torch.linalg.solve(A + 0.5 * ridge + 1e-5 * eye, B.T).T.detach()

    def fit_local(self, snap, shared):
        """Stable ridge and optional strict temporal block-out.

        A complete query interval includes 288 history + 36 future steps.
        The target has only 46 windows: strict disjoint time fitting often
        has zero candidates and then must fall back exactly to W0.
        """
        x, r = snap["x"], snap["r"]
        n = x.size(0)
        anchors = min(self.anchors, n)
        if anchors < 3:
            raise ValueError("Not enough training samples for MLCEA")
        anchor_id = torch.randperm(n, device=x.device)[:anchors]
        ax, ar = x[anchor_id], r[anchor_id]
        metric = self.metric_weights()
        z = x[:, 1:] * metric.sqrt()
        za = ax[:, 1:] * metric.sqrt()
        dist = (
            z.square().sum(-1, keepdim=True)
            + za.square().sum(-1)[None, :]
            - 2 * (z @ za.T)
        ).clamp_min(0)

        different_node = snap["node"][:, None] != snap["node"][anchor_id][None]
        query_t = snap["time_index"][:, None]
        anchor_t = snap["time_index"][anchor_id][None, :]
        time_available = (query_t >= 0) & (anchor_t >= 0)
        time_gap = (query_t - anchor_t).abs() * snap["time_stride"][:, None]
        no_time_overlap = time_available & (
            time_gap >= self.temporal_exclusion_steps
        )
        time_valid = different_node & no_time_overlap
        temporal_eligible = time_valid.sum(-1)
        if self.time_block_mode == "strict":
            if not bool(time_available.all()):
                raise ValueError("strict mode requires true window indices")
            eligible = time_valid
        else:
            eligible = different_node

        distances = dist.masked_fill(~eligible, float("inf"))
        k = min(self.neighbors, anchors)
        ds, picks = torch.topk(distances, k, dim=-1, largest=False)
        finite = torch.isfinite(ds)
        safe_ds = torch.where(finite, ds, torch.zeros_like(ds))
        logits = (-safe_ds / max(self.metric_temperature, 1e-4)).masked_fill(
            ~finite, -1e9
        )
        soft = torch.softmax(logits, dim=-1) * finite.float()
        soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-9)
        uniform = finite.float() / finite.float().sum(-1, keepdim=True).clamp_min(1)
        mix = self.neighbor_uniform_mix
        weights = (1.0 - mix) * soft + mix * uniform
        effective_k = torch.where(
            weights.sum(-1) > 0,
            weights.square().sum(-1).clamp_min(1e-8).reciprocal(),
            torch.zeros_like(weights.sum(-1))
        )
        weak = ((self.min_effective_k - effective_k)
                / max(self.min_effective_k, 1e-6)).clamp_min(0)
        local_lambda = self.ridge * (
            1 + weak if self.adaptive_ridge else torch.ones_like(weak)
        )
        X = ax[picks]
        R = ar[picks]
        gram = torch.einsum("nk,nkd,nke->nde", weights, X, X)
        cross = torch.einsum("nk,nkf,nkd->nfd", weights, R, X)
        eye = torch.eye(x.size(1), device=x.device, dtype=x.dtype)[None]
        lhs = gram + local_lambda[:, None, None] * eye
        rhs = cross + local_lambda[:, None, None] * shared[None]
        W = torch.linalg.solve(lhs, rhs.transpose(-1, -2)).transpose(-1, -2)
        radius = (weights * safe_ds).sum(-1).detach().clamp_min(0.10)
        fitted = torch.bmm(W, x.unsqueeze(-1)).squeeze(-1)
        fit_error = (fitted - r).square().mean(-1)
        return dict(
            W=W, radius=radius, fit_error=fit_error,
            effective_k=effective_k, pool_size=anchors, neighbor_count=k,
            time_eligible_count=temporal_eligible,
            time_eligible_zero_rate=float((temporal_eligible == 0).float().mean().detach()),
            time_eligible_mean=float(temporal_eligible.float().mean().detach()),
            strict_fallback_rate=float((~finite.any(-1)).float().mean().detach()),
            ridge_effective_mean=float(local_lambda.detach().mean()),
            weight_mix=mix, time_block_mode=self.time_block_mode,
        )

    def time_weights(self, snap):
        base = self.horizon_logits.repeat_interleave(2)[None, :]
        weight = torch.softmax(base + self.event_gamma * snap["event"], dim=-1)
        return (1.0 - self.response_mix) / 6.0 + self.response_mix * weight

    def _pair_block(self, source, target, ops_s, ops_t, shared,
                    ws, wt, start, stop):
        """[B,M] outputs with no [B,M,H] materialization."""
        xs = source["x"][start:stop]
        rs = source["r"][start:stop]
        xt = target["x"]
        rt = target["r"]
        ow_s = ws[start:stop]
        ow_t = wt
        W_s = ops_s["W"][start:stop]
        W_t = ops_t["W"]
        n, m = xs.size(0), xt.size(0)
        e_s_to_t = xs.new_zeros(n, m)
        e_t_to_s = xs.new_zeros(n, m)
        bs = xs @ shared.T
        bt = xt @ shared.T
        e0_s = ((bs - rs).square() * ow_s).sum(-1)
        e0_t = ((bt - rt).square() * ow_t).sum(-1)
        for h in range(self.horizon_dim):
            pred_t = W_s[:, h, :] @ xt.T  # source local operator -> target
            pred_s = xs @ W_t[:, h, :].T  # target local operator -> source
            e_s_to_t = e_s_to_t + ow_t[None, :, h] * (pred_t - rt[None, :, h]).square()
            e_t_to_s = e_t_to_s + ow_s[:, None, h] * (pred_s - rs[:, None, h]).square()
        floor = self.response_floor
        s_st = torch.tanh(torch.log((e0_t[None] + floor) / (e_s_to_t + floor)))
        s_ts = torch.tanh(torch.log((e0_s[:, None] + floor) / (e_t_to_s + floor)))
        beta = self.score_beta
        score = -beta * torch.logsumexp(
            torch.stack((-s_st / beta, -s_ts / beta), dim=0), dim=0
        ) + beta * math.log(2.0)
        g = self.metric_weights()
        zs = xs[:, 1:] * g.sqrt()
        zt = xt[:, 1:] * g.sqrt()
        dist = (
            zs.square().sum(-1, keepdim=True)
            + zt.square().sum(-1)[None, :]
            - 2 * zs @ zt.T
        ).clamp_min(0)
        rsup = ops_s["radius"][start:stop, None]
        tsup = ops_t["radius"][None, :]
        typical = (rsup * tsup).sqrt() + 0.25
        reliability = torch.exp(
            -self.support_gamma * (dist / typical - 1.0).clamp_min(0)
        )
        A = 0.5 + reliability * (torch.sigmoid(score / self.gain_temp) - 0.5)
        # Keep only aggregated block diagnostics, not four additional
        # [source,target] matrices. Detached to avoid holding the OT graph.
        with torch.no_grad():
            diagnostic = torch.stack([
                e_s_to_t.detach().mean(),
                e_t_to_s.detach().mean(),
                e0_t.detach().mean(),
                e0_s.detach().mean(),
                s_st.detach().mean(),
                s_ts.detach().mean(),
                (s_st.detach() > 0).float().mean(),
                (s_ts.detach() > 0).float().mean(),
                (s_st.detach() * s_ts.detach() < 0).float().mean(),
                (s_st.detach() - s_ts.detach()).abs().mean(),
                (e_s_to_t.detach() < e0_t.detach()[None, :]).float().mean(),
                (e_t_to_s.detach() < e0_s.detach()[:, None]).float().mean(),
            ])
        return A, score, reliability, diagnostic

    def alignment(self, source, target, shared, alpha, beta, chunk=96):
        """Construct differentiable A, then alpha/beta-interaction-centered R."""
        os = self.fit_local(source, shared)
        ot = self.fit_local(target, shared)
        ws, wt = self.time_weights(source), self.time_weights(target)
        As, Ss, Qs, diagnostics = [], [], [], []
        for start in range(0, source["x"].size(0), chunk):
            stop = min(start + chunk, source["x"].size(0))
            a, s, q, diagnostic = self._pair_block(
                source, target, os, ot, shared, ws, wt, start, stop
            )
            As.append(a)
            Ss.append(s.detach())
            Qs.append(q.detach())
            diagnostics.append(diagnostic * a.numel())
        A = torch.cat(As, dim=0)
        # Correct only interaction information matters under fixed marginals.
        R = A - 0.5
        row = R @ beta[:, None]
        col = alpha[None, :] @ R
        grand = (alpha[:, None] * row).sum()
        interaction = R - row - col + grand
        # Fixed, unbiased self-explanation regularizer: source and target,
        # excluding same-node direct label reconstruction.
        auxiliary = 0.5 * (os["fit_error"].mean() + ot["fit_error"].mean())
        auxiliary = auxiliary + 0.01 * (
            (self.metric_weights() - 1).square().mean()
            + (torch.softmax(self.horizon_logits, dim=0) - 1 / 3).square().mean()
        )
        with torch.no_grad():
            ss = torch.cat(Ss, dim=0)
            qq = torch.cat(Qs, dim=0)
            diag = torch.stack(diagnostics, 0).sum(0) / float(A.numel())
            stats = {
                # Bidirectional explanation, measured independently.
                "err_src_to_tgt": float(diag[0]),
                "err_tgt_to_src": float(diag[1]),
                "err_shared_to_tgt": float(diag[2]),
                "err_shared_to_src": float(diag[3]),
                "gain_src_to_tgt": float(diag[4]),
                "gain_tgt_to_src": float(diag[5]),
                "gain_positive_src_to_tgt": float(diag[6]),
                "gain_positive_tgt_to_src": float(diag[7]),
                "gain_direction_disagree": float(diag[8]),
                "gain_direction_abs_gap": float(diag[9]),
                "src_better_than_shared_rate": float(diag[10]),
                "tgt_better_than_shared_rate": float(diag[11]),
                "a_min": float(A.detach().min()),
                "a_max": float(A.detach().max()),
                "a_lt_half": float((A.detach() < 0.5).float().mean()),
                "q_min": float(qq.min()),
                "q_std": float(qq.std(unbiased=False)),
                "r_uncentered_std": float(R.detach().std(unbiased=False)),
                "interaction_suppression_ratio": float(
                    interaction.detach().std(unbiased=False)
                    / R.detach().std(unbiased=False).clamp_min(1e-12)
                ),
                "a_mean": float(A.detach().mean()),
                "a_std": float(A.detach().std(unbiased=False)),
                "a_gt_half": float((A.detach() > 0.5).float().mean()),
                "s_mean": float(ss.mean()),
                "s_neg_rate": float((ss < 0).float().mean()),
                "q_mean": float(qq.mean()),
                "q_low_rate": float((qq < 0.25).float().mean()),
                "r_interaction_std": float(interaction.detach().std(unbiased=False)),
                "r_interaction_max_abs": float(interaction.detach().abs().max()),
                "operator_fit_src": float(os["fit_error"].detach().mean()),
                "operator_fit_tgt": float(ot["fit_error"].detach().mean()),
                "operator_fit_src_p90": float(torch.quantile(os["fit_error"].detach(), .90)),
                "operator_fit_tgt_p90": float(torch.quantile(ot["fit_error"].detach(), .90)),
                "operator_fit_src_max": float(os["fit_error"].detach().max()),
                "operator_fit_tgt_max": float(ot["fit_error"].detach().max()),
                "operator_norm_src": float(os["W"].detach().norm(dim=(-2, -1)).mean()),
                "operator_norm_tgt": float(ot["W"].detach().norm(dim=(-2, -1)).mean()),
                "effective_neighbors_src": float(os["effective_k"].detach().mean()),
                "effective_neighbors_tgt": float(ot["effective_k"].detach().mean()),
                "effective_neighbors_src_min": float(os["effective_k"].detach().min()),
                "effective_neighbors_tgt_min": float(ot["effective_k"].detach().min()),
                "time_eligible_src_mean": os["time_eligible_mean"],
                "time_eligible_tgt_mean": ot["time_eligible_mean"],
                "time_eligible_src_zero_rate": os["time_eligible_zero_rate"],
                "time_eligible_tgt_zero_rate": ot["time_eligible_zero_rate"],
                "strict_fallback_src_rate": os["strict_fallback_rate"],
                "strict_fallback_tgt_rate": ot["strict_fallback_rate"],
                "ridge_effective_src_mean": os["ridge_effective_mean"],
                "ridge_effective_tgt_mean": ot["ridge_effective_mean"],
                "neighbor_uniform_mix": os["weight_mix"],
                "time_block_mode": os["time_block_mode"],
                "history_std_src_min": float(source["history_std"].min()),
                "history_std_tgt_min": float(target["history_std"].min()),
                "history_std_src_p01": float(torch.quantile(source["history_std"], 0.01)),
                "history_std_tgt_p01": float(torch.quantile(target["history_std"], 0.01)),
                "history_std_src_lt_1e3": float((source["history_std"] < 0.001).float().mean()),
                "history_std_tgt_lt_1e3": float((target["history_std"] < 0.001).float().mean()),
                "raw_response_src_abs_max": float(source["raw_response_abs_max"].max()),
                "raw_response_tgt_abs_max": float(target["raw_response_abs_max"].max()),
                "response_src_clipped_rate": float(source["response_clipped_rate"].mean()),
                "response_tgt_clipped_rate": float(target["response_clipped_rate"].mean()),
                "local_support_radius_src_mean": float(os["radius"].detach().mean()),
                "local_support_radius_tgt_mean": float(ot["radius"].detach().mean()),
                "metric_weights": ",".join(f"{v:.5f}" for v in self.metric_weights().detach().cpu().tolist()),
                "future_weights": ",".join(f"{v:.5f}" for v in torch.softmax(self.horizon_logits.detach(), -1).cpu().tolist()),
                "source_count": int(A.shape[0]),
                "target_count": int(A.shape[1]),
                "pair_count": int(A.numel()),
            }
        return interaction, auxiliary, stats

    @staticmethod
    def weighted_center_probe(base, rel, alpha, beta, entropy=0.001,
                              iters=100):
        """Detached *subsample* Sinkhorn diagnostic, NOT the full GeomLoss plan."""
        with torch.no_grad():
            a = alpha / alpha.sum().clamp_min(1e-12)
            b = beta / beta.sum().clamp_min(1e-12)
            def plan(cost):
                loga = a.clamp_min(1e-20).log()
                logb = b.clamp_min(1e-20).log()
                logk = -cost / entropy
                u = torch.zeros_like(a)
                v = torch.zeros_like(b)
                for _ in range(iters):
                    u = loga - torch.logsumexp(logk + v[None, :], dim=1)
                    v = logb - torch.logsumexp(logk + u[:, None], dim=0)
                P = (u[:, None] + logk + v[None, :]).exp()
                return P / P.sum().clamp_min(1e-12)
            p0 = plan(base)
            p1 = plan(base - rel)
            entropy0 = -(p0 * p0.clamp_min(1e-20).log()).sum()
            entropy1 = -(p1 * p1.clamp_min(1e-20).log()).sum()
            return {
                "probe_plan_l1": float((p1 - p0).abs().sum()),
                "probe_entropy_base": float(entropy0),
                "probe_entropy_new": float(entropy1),
                "probe_mass_shift_abs": float((p1 - p0).abs().max()),
            }


def log_diagnostics(save_dir, rows):
    """Independent CSV, one line per source-pair per epoch; restart-safe header."""
    if not rows:
        return
    out = Path(save_dir) / "mlcea_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch", "source_pair", "source_city", "target_city", "seed", "status", "source_count", "target_count",
        "pair_count", "a_mean", "a_std", "a_gt_half", "a_lt_half", "a_min", "a_max",
        "s_mean", "s_neg_rate",
        "q_mean", "q_std", "q_min", "q_low_rate",
        "err_src_to_tgt", "err_tgt_to_src",
        "err_shared_to_tgt", "err_shared_to_src",
        "gain_src_to_tgt", "gain_tgt_to_src",
        "gain_positive_src_to_tgt", "gain_positive_tgt_to_src",
        "gain_direction_disagree", "gain_direction_abs_gap",
        "src_better_than_shared_rate", "tgt_better_than_shared_rate",
        "r_uncentered_std", "r_interaction_std", "r_interaction_max_abs",
        "interaction_suppression_ratio",
        "operator_fit_src", "operator_fit_tgt",
        "operator_fit_src_p90", "operator_fit_tgt_p90",
        "operator_fit_src_max", "operator_fit_tgt_max",
        "operator_norm_src",
        "operator_norm_tgt", "effective_neighbors_src",
        "effective_neighbors_tgt", "effective_neighbors_src_min",
        "effective_neighbors_tgt_min",
        "time_eligible_src_mean", "time_eligible_tgt_mean",
        "time_eligible_src_zero_rate", "time_eligible_tgt_zero_rate",
        "strict_fallback_src_rate", "strict_fallback_tgt_rate",
        "ridge_effective_src_mean", "ridge_effective_tgt_mean",
        "neighbor_uniform_mix", "time_block_mode",
        "history_std_src_min", "history_std_tgt_min",
        "history_std_src_p01", "history_std_tgt_p01",
        "history_std_src_lt_1e3", "history_std_tgt_lt_1e3",
        "raw_response_src_abs_max", "raw_response_tgt_abs_max",
        "response_src_clipped_rate", "response_tgt_clipped_rate",
        "local_support_radius_src_mean",
        "local_support_radius_tgt_mean",
        "metric_weights", "future_weights",
        "base_probe_min", "base_probe_mean", "base_probe_max",
        "base_probe_std", "relation_scale", "relation_probe_std",
        "relation_correction_mean", "relation_correction_std",
        "relation_correction_max_abs", "relation_vs_base_spread",
        "dvGP_source_keep_ratio", "dvGP_target_keep_ratio",
        "dvGP_prob_source_mean", "dvGP_prob_target_mean",
        "ot_loss_scale", "ot_unscaled_loss",
        "ot_blur", "relation_vs_blur", "ot_loss", "aux_loss",
        "probe_plan_l1", "probe_entropy_base", "probe_entropy_new",
        "probe_mass_shift_abs", "metric_grad_norm", "future_grad_norm",
        "metric_parameter_step_norm", "future_parameter_step_norm",
        "grad_metric_missing", "grad_future_missing",
        "epoch_task_loss", "epoch_target_task_loss", "epoch_da_loss",
        "epoch_aux_weighted_loss", "epoch_total_loss",
        "epoch_source_mae", "epoch_source_rmse",
        "ot_is_active", "mlcea_is_active",
        "source_mass_nonzero", "target_mass_nonzero",
        "source_mass_entropy", "target_mass_entropy",
    ]
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        if out.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)
