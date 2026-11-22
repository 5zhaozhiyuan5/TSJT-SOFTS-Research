"""Fast, CPU-only MLCEA unit tests (no traffic datasets or gpytorch needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model" / "Meta_Models"))

import torch
from mlcea import MLCEA


def _snapshot(model, batch=3, nodes=7):
    h = torch.randn(batch, nodes, 288)
    mean = h.mean(-1, keepdim=True)
    std = (h.var(-1, unbiased=False, keepdim=True) + 1e-9).sqrt()
    anchored = (h[..., -1:] - mean) / std
    future = anchored + 0.5 * torch.randn(batch, nodes, 36)
    return model.make_snapshot(h, future)


def test_snapshot_dimensions_and_target_split():
    torch.manual_seed(10)
    model = MLCEA(anchors=12, neighbors=4)
    snap = _snapshot(model)
    assert snap["x"].shape == (21, 7)
    assert snap["r"].shape == (21, 6)
    assert snap["node"].shape == (21,)
    pieces = model.split_by_batch(snap, [1, 1, 1], 7)
    assert len(pieces) == 3
    assert all(p["x"].shape == (7, 7) for p in pieces)
    assert torch.isfinite(snap["r"]).all()


def test_interaction_centering_and_parameter_gradients():
    torch.manual_seed(11)
    model = MLCEA(anchors=12, neighbors=4)
    source, target = _snapshot(model), _snapshot(model)
    shared = model.fit_reference([source])
    assert shared.shape == (6, 7)
    alpha = torch.softmax(torch.randn(21), dim=0)
    beta = torch.softmax(torch.randn(21), dim=0)
    rel, aux, stats = model.alignment(source, target, shared, alpha, beta, chunk=5)
    assert rel.shape == (21, 21)
    assert torch.isfinite(rel).all()
    assert torch.isfinite(aux)
    assert torch.allclose(rel @ beta, torch.zeros_like(alpha), atol=1e-5)
    assert torch.allclose(alpha @ rel, torch.zeros_like(beta), atol=1e-5)
    assert stats["source_count"] == 21 and stats["target_count"] == 21
    assert 0 <= stats["q_mean"] <= 1
    (rel.square().mean() + 0.02 * aux).backward()
    assert model.metric_logits.grad is not None
    assert model.horizon_logits.grad is not None
    assert torch.isfinite(model.metric_logits.grad).all()
    assert torch.isfinite(model.horizon_logits.grad).all()


def test_probe_detects_changed_plan():
    model = MLCEA()
    base = torch.ones(2, 2)
    corr = torch.tensor([[0.25, -0.25], [-0.25, 0.25]])
    weights = torch.tensor([0.5, 0.5])
    probe = model.weighted_center_probe(base, corr, weights, weights, entropy=0.1)
    assert probe["probe_plan_l1"] > 0.2
    nochange = model.weighted_center_probe(base, torch.zeros_like(base),
                                           weights, weights, entropy=0.1)
    assert nochange["probe_plan_l1"] < 1e-6
