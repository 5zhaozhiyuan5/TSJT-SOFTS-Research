"""MLCEA temporal-support and adaptive-ridge regression tests."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model" / "Meta_Models"))
from mlcea import MLCEA


def _batch(model, positions, nodes=4):
    torch.manual_seed(123)
    h = torch.randn(len(positions), nodes, 288)
    center = h.mean(-1, keepdim=True)
    sd = (h.var(-1, keepdim=True, unbiased=False) + 1e-9).sqrt()
    y = (h[..., -1:] - center) / sd + torch.randn(len(positions), nodes, 36) * 0.25
    return model.make_snapshot(h, y, window_indices=positions, window_stride=12)


def test_abs_window_ids_and_revin_diagnostics():
    model = MLCEA(anchors=6, neighbors=2)
    snap = _batch(model, [3, 4, 40], nodes=3)
    assert snap["x"].shape == (9, 7)
    assert snap["time_index"].tolist() == [3]*3+[4]*3+[40]*3
    assert (snap["time_stride"] == 12).all()
    assert (snap["history_std"] > 0).all()
    assert (snap["raw_response_abs_max"] >= 0).all()


def test_strict_disjoint_time_no_support_uses_exact_source_reference():
    model = MLCEA(
        anchors=20, neighbors=4, time_block_mode="strict",
        neighbor_uniform_mix=0.35, ridge=2.0,
    )
    snap = _batch(model, [10, 11, 12, 13, 14], nodes=5)
    w0 = model.fit_reference([snap])
    ops = model.fit_local(snap, w0)
    assert ops["strict_fallback_rate"] == pytest.approx(1.0)
    assert ops["time_eligible_zero_rate"] == pytest.approx(1.0)
    assert torch.allclose(ops["W"], w0[None].expand_as(ops["W"]), atol=1e-5)
    assert (ops["effective_k"] == 0).all()
    assert ops["ridge_effective_mean"] > 2.0


def test_strict_disjoint_time_allows_only_distant_windows():
    model = MLCEA(
        anchors=12, neighbors=4, time_block_mode="strict",
        neighbor_uniform_mix=0.35,
    )
    snap = _batch(model, [0, 40], nodes=6)
    w0 = model.fit_reference([snap])
    ops = model.fit_local(snap, w0)
    assert ops["time_eligible_zero_rate"] == pytest.approx(0.0)
    assert ops["strict_fallback_rate"] == pytest.approx(0.0)
    assert torch.isfinite(ops["W"]).all()
    assert (ops["effective_k"] >= 1).all()
    assert (ops["effective_k"] <= 4.01).all()


def test_strict_requires_true_time_metadata():
    model = MLCEA(anchors=12, neighbors=4, time_block_mode="strict")
    h = torch.randn(2, 7, 288)
    y = torch.randn(2, 7, 36)
    snap = model.make_snapshot(h, y)
    with pytest.raises(ValueError, match="true window indices"):
        model.fit_local(snap, model.fit_reference([snap]))


def test_stable_soft_knn_backpropagates():
    model = MLCEA(anchors=15, neighbors=6, neighbor_uniform_mix=0.35,
                  adaptive_ridge=True)
    snap = _batch(model, [0, 3, 60], nodes=8)
    prior = model.fit_reference([snap])
    ops = model.fit_local(snap, prior)
    loss = ops["W"].square().sum()
    loss.backward()
    assert model.metric_logits.grad is not None
    assert torch.isfinite(model.metric_logits.grad).all()
