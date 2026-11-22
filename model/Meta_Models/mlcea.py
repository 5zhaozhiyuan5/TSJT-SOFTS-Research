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
from torch.nn import functional as F


class MLCEA(nn.Module):
    """Two tiny trainable vectors; per-sample operators use ridge solves."""

    def __init__(self, anchors=128, neighbors=16, ridge=2.0,
                 metric_temperature=1.0, response_mix=0.5,
                 event_gamma=0.2, score_beta=0.25, gain_temp=0.35,
                 support_gamma=0.35, response_floor=0.02,
                 history_dim=6, horizon_dim=6):
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

    def metric_weights(self):
        return self.history_dim * torch.softmax(self.metric_logits, dim=0)

    def make_snapshot(self, history, future_rev):
        """Output node-wise x:[B*N,7], r:[B*N,6], event:[B*N,6]."""
        h = torch.as_tensor(history, device=future_rev.device,
                            dtype=future_rev.dtype)
        if h.ndim != 3 or future_rev.ndim != 3:
            raise ValueError("Expected [B,N,L] histories and [B,N,H] futures")
        if h.shape[:2] != future_rev.shape[:2] or h.shape[-1] < 192 or future_rev.shape[-1] < 36:
            raise ValueError("MLCEA expects aligned batch/node sizes, >=192 histories and >=36 futures")
        n_batch, n_nodes, _ = h.shape
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
        r = (future_rev.index_select(-1, idx) - z[..., -1:]).reshape(-1, 6)
        r = r.clamp(-6, 6).detach()
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
        )

    @staticmethod
    def split_snapshot(snapshot, sizes):
        return [
            {key: val for key, val in zip(snapshot.keys(), chunks)}
            for chunks in zip(*[
                torch.split(v.reshape(-1, snapshot["x"].shape[-1])
                            if False else v.reshape(-1, v.shape[-1]) if v.ndim > 2 else v,
                            [s * (len(snapshot["node"]) // sum(sizes)) for s in sizes],
                            dim=0)
                for v in snapshot.values()
            ])
        ]

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
        """Differentiable KNN weights and local ridge operators.

        Neighbour candidates exclude every sample of the query's own road
        node, even at different (possibly overlapping) time windows.
        """
        x, r = snap["x"], snap["r"]
        n = x.size(0)
        anchors = min(self.anchors, n)
        if anchors < 3:
            raise ValueError("Not enough nodes to estimate MLCEA operators")
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
        # Node-level exclusion also covers query's own overlapping windows.
        invalid = snap["node"][:, None] == snap["node"][anchor_id][None, :]
        dist = dist.masked_fill(invalid, float("inf"))
        k = min(self.neighbors, anchors - 1)
        ds, picks = torch.topk(dist, k, dim=-1, largest=False)
        if not torch.isfinite(ds).all():
            raise RuntimeError("No legal neighbours: insufficient distinct road nodes")
        weights = torch.softmax(-ds / max(self.metric_temperature, 1e-4), dim=-1)
        X = ax[picks]
        R = ar[picks]
        gram = torch.einsum("nk,nkd,nke->nde", weights, X, X)
        cross = torch.einsum("nk,nkf,nkd->nfd", weights, R, X)
        reg_eye = torch.eye(x.size(1), device=x.device, dtype=x.dtype)[None]
        rhs = cross + self.ridge * shared[None]
        lhs = gram + self.ridge * reg_eye
        # torch.linalg.solve is differentiable through both inputs.
        W = torch.linalg.solve(lhs, rhs.transpose(-1, -2)).transpose(-1, -2)
        radius = (weights * ds).sum(-1).detach().clamp_min(0.10)
        fitted = torch.bmm(W, x.unsqueeze(-1)).squeeze(-1)
        fit_error = (fitted - r).square().mean(-1)
        effective_k = (weights.square().sum(-1) + 1e-8).reciprocal()
        return dict(W=W, radius=radius, fit_error=fit_error,
                    effective_k=effective_k, pool_size=anchors,
                    neighbor_count=k)

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
        return A, score, reliability

    def alignment(self, source, target, shared, alpha, beta, chunk=96):
        """Construct differentiable A, then alpha/beta-interaction-centered R."""
        os = self.fit_local(source, shared)
        ot = self.fit_local(target, shared)
        ws, wt = self.time_weights(source), self.time_weights(target)
        As, Ss, Qs = [], [], []
        for start in range(0, source["x"].size(0), chunk):
            stop = min(start + chunk, source["x"].size(0))
            a, s, q = self._pair_block(source, target, os, ot, shared,
                                      ws, wt, start, stop)
            As.append(a)
            Ss.append(s.detach())
            Qs.append(q.detach())
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
            stats = {
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
                "operator_norm_src": float(os["W"].detach().norm(dim=(-2, -1)).mean()),
                "operator_norm_tgt": float(ot["W"].detach().norm(dim=(-2, -1)).mean()),
                "effective_neighbors_src": float(os["effective_k"].detach().mean()),
                "effective_neighbors_tgt": float(ot["effective_k"].detach().mean()),
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
        "epoch", "source_pair", "status", "source_count", "target_count",
        "pair_count", "a_mean", "a_std", "a_gt_half", "s_mean", "s_neg_rate",
        "q_mean", "q_low_rate", "r_interaction_std", "r_interaction_max_abs",
        "operator_fit_src", "operator_fit_tgt", "operator_norm_src",
        "operator_norm_tgt", "effective_neighbors_src",
        "effective_neighbors_tgt", "metric_weights", "future_weights",
        "base_probe_std", "relation_scale", "relation_probe_std",
        "ot_blur", "relation_vs_blur", "ot_loss", "aux_loss",
        "probe_plan_l1", "probe_entropy_base", "probe_entropy_new",
        "probe_mass_shift_abs", "metric_grad_norm", "future_grad_norm",
        "source_mass_nonzero", "target_mass_nonzero",
        "source_mass_entropy", "target_mass_entropy",
    ]
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if out.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)
