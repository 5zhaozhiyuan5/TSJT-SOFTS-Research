import os
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from geomloss import SamplesLoss as GeomlossSamplesLoss

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META_MODELS = os.path.join(ROOT, "model", "Meta_Models")
if META_MODELS not in sys.path:
    sys.path.insert(0, META_MODELS)

import PMME
import PMME_2ndStage


class TinyBackbone(nn.Module):
    """Fast differentiable stand-in for the forecasting backbone in integration tests."""

    def __init__(self):
        super().__init__()
        self.representation = nn.Linear(1, 128)
        self.forecast = nn.Linear(128, 36)

    def forward(self, x):
        # PMME passes [batch, history, nodes].
        node_summary = x.mean(dim=1).unsqueeze(-1)
        rep = torch.tanh(self.representation(node_summary))
        out = self.forecast(rep)
        return rep, out


class FastSmokeSinkhorn:
    """Run the real GeomLoss Sinkhorn on a deterministic small subset."""

    def __init__(self, **kwargs):
        self.loss = GeomlossSamplesLoss(
            loss="sinkhorn", p=1, backend="tensorized", debias=False, blur=0.05
        )

    def __call__(self, alpha, x, beta, y):
        x_idx = torch.linspace(
            0, x.shape[0] - 1, steps=min(32, x.shape[0]), device=x.device
        ).long()
        y_idx = torch.linspace(
            0, y.shape[0] - 1, steps=min(32, y.shape[0]), device=y.device
        ).long()
        alpha_sample = alpha[x_idx]
        beta_sample = beta[y_idx]
        alpha_sample = alpha_sample / alpha_sample.sum()
        beta_sample = beta_sample / beta_sample.sum()
        return self.loss(alpha_sample, x[x_idx], beta_sample, y[y_idx])


def make_data(batch_size, node_count, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, node_count, 288, 1, generator=generator)
    y = 10.0 + torch.rand(batch_size, node_count, 36, generator=generator)
    return Data(
        node_num=node_count,
        x=x,
        y=y,
        means=np.asarray([0.0], dtype=np.float32),
        stds=np.asarray([1.0], dtype=np.float32),
    )


class StageITrainingIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = Namespace(save_dir=self.tmp.name, STmodel="SOFTS")
        self.model_args = {
            "STnet": {"lr": 1e-3},
            "chta": {
                "enabled": True,
                "input_dim": 128,
                "horizon_steps": 36,
                "minutes_per_step": 5,
                "anchor_steps": [2, 12, 24, 36],
                "time_dim": 16,
                "hidden_dim": 32,
                "relation_dim": 16,
                "epsilon": 0.08,
                "sinkhorn_iterations": 12,
                "max_samples": 32,
                "future_weight": 1.0,
                "prior_kl_weight": 1.0,
                "loss_weight": 0.1,
                "smoothness_weight": 1e-4,
                "target_loss_weight": 0.1,
            },
        }
        self.patch_cfg = {"device": torch.device("cpu")}

    def test_stage1_initializes_and_stage2_loads_official_backbone_weights(self):
        stage1 = PMME.STRep(
            classifier_args=0.0,
            DAepoch_args=0,
            DAcoef_args=0.01,
            xySquaredRatio_args=100.0,
            memory_size_args=0,
            projection_args=1,
            model_name="SOFTS",
            DA="OT",
            data_args={},
            task_args={},
            model_args=self.model_args,
            PatchFSL_cfg=self.patch_cfg,
            args=self.args,
        )
        self.assertIsNotNone(stage1.chta)
        self.assertTrue(
            any(p is stage1.chta.history_projection.weight for group in stage1.main_optim.param_groups for p in group["params"])
        )

        stage2 = PMME_2ndStage.STRep(
            0.0, 0, 0.01, 100.0, 0, 0, "SOFTS",
            {}, {}, self.model_args, self.patch_cfg, self.args
        )
        stage1_state = stage1.state_dict()
        official_backbone_state = {
            key[len("model."):]: value
            for key, value in stage1_state.items()
            if key.startswith("model.")
        }
        stage2_state = stage2.model.state_dict()
        compatible = {
            key: value for key, value in official_backbone_state.items()
            if key in stage2_state and stage2_state[key].shape == value.shape
        }
        self.assertGreater(len(compatible), 0)
        stage2_state.update(compatible)
        stage2.model.load_state_dict(stage2_state)

        # CHTA is a Stage-I-only module and is not part of the Stage-II model.
        self.assertFalse(any(key.startswith("chta.") for key in stage2.model.state_dict()))

    def test_one_stage1_meta_update_with_chta_and_pcgrad(self):
        stage1 = PMME.STRep(
            classifier_args=0.0,  # exercise uniform DVGP marginals without fitting GPs
            DAepoch_args=0,
            DAcoef_args=0.01,
            xySquaredRatio_args=100.0,
            memory_size_args=0,
            projection_args=1,  # exercise the PCGrad path used by the launch scripts
            model_name="SOFTS",
            DA="OT",
            data_args={},
            task_args={},
            model_args=self.model_args,
            PatchFSL_cfg=self.patch_cfg,
            args=self.args,
        )
        stage1.model.forecasting_model = TinyBackbone()
        # The test swaps the backbone after STRep construction, so rebuild the
        # optimizer to include the synthetic backbone parameters as production
        # construction would do for the official SOFTS/iTransformer parameters.
        stage1.optimized_parameters = list(stage1.model.parameters()) + list(stage1.chta.parameters())
        stage1.main_optim = torch.optim.AdamW(
            stage1.optimized_parameters, lr=stage1.lr, weight_decay=1e-2
        )

        target = make_data(46, 325, 1)
        sources = [
            make_data(1, 207, 2),
            make_data(1, 524, 3),
            make_data(1, 627, 4),
        ]
        data = [target] + sources
        matrices = [torch.eye(item.node_num) for item in data]

        before = stage1.chta.history_projection.weight.detach().clone()
        before_backbone = stage1.model.forecasting_model.representation.weight.detach().clone()

        with patch.object(PMME, "SamplesLoss", FastSmokeSinkhorn):
            result = stage1.meta_train_revise(data, matrices, epoch=0)

        self.assertEqual(len(result), 5)
        self.assertTrue(all(np.isfinite(float(value)) for value in result))
        self.assertFalse(torch.equal(before, stage1.chta.history_projection.weight.detach()))
        self.assertFalse(
            torch.equal(before_backbone, stage1.model.forecasting_model.representation.weight.detach())
        )


    def test_one_stage1_meta_update_with_pattern_moe_as_main_alignment(self):
        pattern_model_args = {
            "STnet": {"lr": 1e-3},
            "chta": {"enabled": False},
            "pattern_moe": {
                "enabled": True,
                "input_dim": 128,
                "horizon_steps": 36,
                "minutes_per_step": 5,
                "anchor_steps": [2, 12, 24, 36],
                "scale_steps": [12, 36, 72, 144, 288],
                "time_dim": 12,
                "hidden_dim": 32,
                "relation_dim": 16,
                "epsilon": 0.08,
                "sinkhorn_iterations": 12,
                "max_samples": 24,
                "pattern_temperature": 0.2,
                "router_temperature": 1.0,
                "future_weight": 1.0,
                "pattern_weight": 0.5,
                "prior_kl_weight": 1.0,
                "loss_weight": 1.0,
                "smoothness_weight": 1e-4,
                "target_loss_weight": 0.1,
            },
        }
        pattern_bank = torch.randn(5, 8, 16)
        stage1 = PMME.STRep(
            classifier_args=0.0,
            DAepoch_args=0,
            DAcoef_args=0.01,
            xySquaredRatio_args=100.0,
            memory_size_args=0,
            projection_args=1,
            model_name="SOFTS",
            DA="OT",
            data_args={},
            task_args={},
            model_args=pattern_model_args,
            PatchFSL_cfg=self.patch_cfg,
            args=self.args,
            pattern_bank=pattern_bank,
        )
        stage1.model.forecasting_model = TinyBackbone()
        stage1.optimized_parameters = (
            list(stage1.model.parameters()) + list(stage1.pattern_moe.parameters())
        )
        stage1.main_optim = torch.optim.AdamW(
            stage1.optimized_parameters, lr=stage1.lr, weight_decay=1e-2
        )

        target = make_data(46, 325, 11)
        sources = [
            make_data(1, 207, 12),
            make_data(1, 524, 13),
            make_data(1, 627, 14),
        ]
        data = [target] + sources
        matrices = [torch.eye(item.node_num) for item in data]

        before_router = stage1.pattern_moe.router[-1].weight.detach().clone()
        result = stage1.meta_train_revise(data, matrices, epoch=0)

        self.assertEqual(len(result), 5)
        self.assertTrue(all(np.isfinite(float(value)) for value in result))
        self.assertFalse(
            torch.equal(before_router, stage1.pattern_moe.router[-1].weight.detach())
        )
        self.assertIsNone(stage1.chta)


if __name__ == "__main__":
    unittest.main()
