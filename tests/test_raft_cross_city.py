import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from model.Meta_Models.raft_cross_city import (
    MultiScaleRetrievalAdapter,
    configure_raft_stage2_parameters,
)


class RaftCrossCityAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.adapter = MultiScaleRetrievalAdapter(
            seq_len=288,
            pred_len=36,
            max_memory_size=24,
            source_city_names=["source_a", "source_b"],
            periods=(4, 2, 1),
            topm=5,
            temperature=0.1,
            query_chunk=4,
        )

    @staticmethod
    def make_city_data(city_mean, city_std, samples=8, nodes=3, horizon=36):
        history_global = torch.randn(samples, nodes, 288)
        future_global = history_global[:, :, -1:] + 0.2 * torch.randn(
            samples, nodes, horizon
        )
        future_raw = future_global * city_std + city_mean
        x = torch.stack(
            [history_global, torch.zeros_like(history_global)], dim=-1
        )
        return x, future_raw

    def build_bank(self):
        x_a, y_a = self.make_city_data(60.0, 10.0)
        x_b, y_b = self.make_city_data(30.0, 8.0)
        return self.adapter.build_memory(
            {"source_a": x_a, "source_b": x_b},
            {"source_a": y_a, "source_b": y_b},
            {"source_a": np.array([60.0]), "source_b": np.array([30.0])},
            {"source_a": np.array([10.0]), "source_b": np.array([8.0])},
            city_names=["source_a", "source_b"],
            target_city="target",
        )

    def test_balanced_source_only_bank(self):
        count = self.build_bank()
        self.assertEqual(count, 24)
        self.assertEqual(
            self.adapter.city_counts(),
            {"source_a": 12, "source_b": 12},
        )
        with self.assertRaisesRegex(RuntimeError, "present in the Stage-II source bank"):
            self.adapter.build_memory(
                {"source_a": torch.randn(2, 2, 288, 2), "target": torch.randn(2, 2, 288, 2)},
                {"source_a": torch.randn(2, 2, 36), "target": torch.randn(2, 2, 36)},
                {"source_a": [60.0], "target": [50.0]},
                {"source_a": [10.0], "target": [9.0]},
                city_names=["source_a", "target"],
                target_city="target",
            )

    def test_multiscale_retrieval_shape_and_identity_initialization(self):
        self.build_bank()
        history = torch.randn(2, 4, 288)
        base = torch.randn(2, 4, 36) * 4.0 + 55.0
        fused, retrieved = self.adapter(
            history,
            base,
            means=np.array([55.0]),
            stds=np.array([11.0]),
        )
        self.assertEqual(tuple(fused.shape), (2, 4, 36))
        self.assertEqual(tuple(retrieved.shape), (2, 4, 36))
        self.assertTrue(torch.isfinite(fused).all())
        self.assertTrue(torch.isfinite(retrieved).all())
        # Before target adaptation, the linear fusion is exactly the SOFTS base.
        self.assertTrue(torch.allclose(fused, base, atol=1e-6, rtol=1e-6))

    def test_adapter_receives_gradients_without_updating_memory(self):
        self.build_bank()
        history = torch.randn(2, 3, 288)
        base = torch.randn(2, 3, 36) * 3.0 + 50.0
        target = torch.randn(2, 3, 36) * 2.0 + 50.0
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=1e-2)
        memory_before = self.adapter.memory_future_1.clone()

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            fused, _ = self.adapter(history, base, [50.0], [9.0])
            loss = torch.nn.functional.smooth_l1_loss(fused, target)
            loss.backward()
            self.assertIsNotNone(self.adapter.fusion.weight.grad)
            optimizer.step()

        self.assertIsNotNone(self.adapter.scale_projection["1"].weight.grad)
        self.assertTrue(torch.equal(memory_before, self.adapter.memory_future_1))

    def test_stage2_freezes_backbone_and_trains_only_adapter(self):
        class DummyModel(nn.Module):
            def __init__(self, adapter):
                super().__init__()
                self.forecasting_model = nn.Linear(4, 4)
                self.raft_adapter = adapter

        model = DummyModel(self.adapter)
        trainable = configure_raft_stage2_parameters(model)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.forecasting_model.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.raft_adapter.parameters())
        )
        self.assertEqual(
            {id(parameter) for parameter in trainable},
            {id(parameter) for parameter in self.adapter.parameters()},
        )

    def test_dynamic_memory_checkpoint_roundtrip(self):
        self.build_bank()
        restored = MultiScaleRetrievalAdapter(
            seq_len=288,
            pred_len=36,
            max_memory_size=24,
            source_city_names=["source_a", "source_b"],
            periods=(4, 2, 1),
            topm=5,
        )
        restored.load_state_dict(self.adapter.state_dict())
        self.assertEqual(restored.num_entries, self.adapter.num_entries)
        self.assertEqual(restored.city_counts(), self.adapter.city_counts())
        for period in (4, 2, 1):
            self.assertTrue(
                torch.equal(
                    getattr(restored, f"memory_keys_{period}"),
                    getattr(self.adapter, f"memory_keys_{period}"),
                )
            )

    def test_shorter_horizon_12_is_supported(self):
        adapter = MultiScaleRetrievalAdapter(
            seq_len=288,
            pred_len=12,
            max_memory_size=12,
            source_city_names=["source_a"],
            periods=(4, 2, 1),
            topm=3,
        )
        x, y = self.make_city_data(60.0, 10.0, horizon=12)
        adapter.build_memory(
            {"source_a": x},
            {"source_a": y},
            {"source_a": [60.0]},
            {"source_a": [10.0]},
            city_names=["source_a"],
            target_city="target",
        )
        fused, retrieved = adapter(
            torch.randn(2, 3, 288),
            torch.randn(2, 3, 12),
            [50.0],
            [9.0],
        )
        self.assertEqual(tuple(fused.shape), (2, 3, 12))
        self.assertEqual(tuple(retrieved.shape), (2, 3, 12))


if __name__ == "__main__":
    unittest.main()
