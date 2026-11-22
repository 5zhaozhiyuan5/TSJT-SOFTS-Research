"""Regression tests for structured MLCEA diagnostics, independent of traffic data."""
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model" / "Meta_Models"))

import torch

from mlcea import MLCEA, log_diagnostics
from mlcea_experiment_logging import (
    write_run_manifest, log_stage1_epoch, log_stage1_evaluation,
    log_stage2_evaluation,
)


def _read(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_local_operator_logs_directional_explanation_and_gradients(tmp_path):
    torch.manual_seed(74)
    model = MLCEA(anchors=15, neighbors=4)
    h = torch.randn(3, 7, 288)
    future = (h[..., -1:] - h.mean(-1, keepdim=True)) / (
        h.var(-1, keepdim=True, unbiased=False) + 1e-9
    ).sqrt() + torch.randn(3, 7, 36) * 0.3
    snap = model.make_snapshot(h, future)
    prior = model.fit_reference([snap])
    alpha = torch.softmax(torch.randn(21), -1)
    beta = torch.softmax(torch.randn(21), -1)
    rel, aux, stats = model.alignment(snap, snap, prior, alpha, beta, chunk=6)
    required = [
        "err_src_to_tgt", "err_tgt_to_src",
        "err_shared_to_tgt", "err_shared_to_src",
        "gain_src_to_tgt", "gain_tgt_to_src",
        "gain_direction_abs_gap", "interaction_suppression_ratio",
        "effective_neighbors_src_min", "operator_fit_src_p90",
    ]
    for name in required:
        assert name in stats, name
        assert isinstance(stats[name], float)
    assert torch.isfinite(rel).all()
    assert torch.isfinite(aux)
    (rel.square().mean() + aux * 0.02).backward()
    assert model.metric_logits.grad is not None
    assert model.horizon_logits.grad is not None

    stats.update(dict(epoch=3, source_pair=0, source_city="metr-la",
                      target_city="pems-bay", seed=1, status="active"))
    log_diagnostics(tmp_path, [stats])
    assert (tmp_path / "mlcea_metrics.csv").exists()
    rows = _read(tmp_path / "mlcea_metrics.csv")
    assert len(rows) == 1
    assert rows[0]["err_src_to_tgt"] != ""
    assert rows[0]["source_city"] == "metr-la"


class _DummyModel:
    def __init__(self):
        p = torch.nn.Parameter(torch.ones(1))
        self.main_optim = torch.optim.Adam([p], lr=1e-3)
        self.mlcea_optim = torch.optim.Adam([p], lr=2e-4)
        self.last_mlcea_epoch = {
            "source_task_loss": 1.25,
            "target_task_loss": 0.12,
            "da_loss": 0.40,
            "mlcea_aux_loss": 0.18,
            "mlcea_aux_weighted_loss": 0.004,
            "active_source_pairs": 3,
            "metric_grad_norm": 0.11,
            "future_grad_norm": 0.05,
            "metric_parameter_step_norm": 0.001,
            "future_parameter_step_norm": 0.002,
            "mlcea_active": 1,
        }


def test_epoch_evaluation_and_manifest_logs(tmp_path):
    args = Namespace(save_dir=tmp_path, seed=1, mlcea_enable=1,
                     DA="OT", DAepoch=3, test_dataset="pems-bay")
    manifest = write_run_manifest(args)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert doc["seed"] == 1
    model = _DummyModel()
    log_stage1_epoch(args, model, 1, 3.5, 1.0, 1.4, 1.1, 0.1, 0.25)
    log_stage1_epoch(args, model, 3, 2.1, 0.7, 1.0, 0.8, 0.08, 0.22)
    epochs = _read(tmp_path / "stage1_epoch_metrics.csv")
    assert epochs[0]["ot_status"] == "waiting_for_DAepoch"
    assert epochs[1]["ot_status"] == "active"
    assert epochs[1]["metric_parameter_step_norm"] != ""
    assert epochs[1]["train_target_loss"] == "0.12"

    train = [1.0] * 36
    test = [1.2] * 36
    details = {
        "target_train_rmse": [2.0] * 36,
        "target_test_rmse": [2.2] * 36,
        "target_train_mape": [0.1] * 36,
        "target_test_mape": [0.15] * 36,
    }
    log_stage1_evaluation(args, 3, train, test, 1e10, details=details)
    rows = _read(tmp_path / "stage1_eval_metrics.csv")
    assert len(rows) == 36
    assert rows[-1]["horizon_minutes"] == "180"
    assert rows[0]["target_heldout_test_rmse"] == "2.2"
    assert rows[0]["is_selected_by_train_mae"] == "1"

    log_stage2_evaluation(args, "without_memory", test)
    log_stage2_evaluation(args, "with_memory", train)
    rows = _read(tmp_path / "stage2_eval_metrics.csv")
    assert len(rows) == 72
    assert rows[0]["phase"] == "without_memory"
    assert rows[-1]["phase"] == "with_memory"
