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
    """Keeps the complete STRep update test small; real Sinkhorn is tested separately."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, alpha, x, beta, y):
        x_sample = x[:: max(1, x.shape[0] // 32)][:32]
        y_sample = y[:: max(1, y.shape[0] // 32)][:32]
        return torch.cdist(x_sample, y_sample, p=1).mean()


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
            "hclm": {
                "enabled": True,
                "input_dim": 128,
                "rank": 8,
                "basis_dim": 8,
                "anchor_minutes": [10, 60, 120, 180],
                "max_minutes": 180,
                "relation_weight": 0.25,
                "smoothness_weight": 1e-4,
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
        self.assertIsNotNone(stage1.hclm)
        self.assertTrue(
            any(p is stage1.hclm.base_projection for group in stage1.main_optim.param_groups for p in group["params"])
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

        # HCLM is a Stage-I-only module and is not part of the Stage-II model.
        self.assertFalse(any(key.startswith("hclm.") for key in stage2.model.state_dict()))

    def test_one_stage1_meta_update_with_hclm_and_pcgrad(self):
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

        target = make_data(46, 325, 1)
        sources = [
            make_data(1, 207, 2),
            make_data(1, 524, 3),
            make_data(1, 627, 4),
        ]
        data = [target] + sources
        matrices = [torch.eye(item.node_num) for item in data]

        before = stage1.hclm.base_projection.detach().clone()
        before_backbone = stage1.model.forecasting_model.representation.weight.detach().clone()

        with patch.object(PMME, "SamplesLoss", FastSmokeSinkhorn):
            result = stage1.meta_train_revise(data, matrices, epoch=0)

        self.assertEqual(len(result), 5)
        self.assertTrue(all(np.isfinite(float(value)) for value in result))
        self.assertFalse(torch.equal(before, stage1.hclm.base_projection.detach()))
        self.assertFalse(
            torch.equal(before_backbone, stage1.model.forecasting_model.representation.weight.detach())
        )


if __name__ == "__main__":
    unittest.main()
