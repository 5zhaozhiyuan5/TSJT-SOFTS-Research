import os
import sys
import unittest
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from model.Meta_Models.residual_memory import (
    StationarityAwareResidualMemory,
    should_skip_stage2_optimizer_step,
)


class ResidualMemorySmokeTest(unittest.TestCase):
    def setUp(self):
        self.memory = StationarityAwareResidualMemory(
            seq_len=288,
            pred_len=36,
            d_model=128,
            max_memory_size=120,
            source_city_names=["source_a", "source_b", "source_c"],
            topk=4,
            candidate_pool=8,
            overlap_window=27,
            query_chunk=5,
        )

    @staticmethod
    def make_batch(city, sample_ids, batch_size=4, node_num=5):
        x = torch.randn(batch_size, node_num, 288, 2)
        y = torch.randn(batch_size, node_num, 36)
        base = y + 0.2 * torch.randn_like(y)
        data = SimpleNamespace(
            x=x,
            y=y,
            data_name=city,
            sample_index=torch.tensor(sample_ids, dtype=torch.long),
        )
        return data, base

    def fill_source_memory(self):
        for city in ("source_a", "source_b", "source_c"):
            data, base = self.make_batch(city, [0, 1, 2, 3])
            self.memory.update(data, base)

    def test_stage2_empty_bank_warmup_guard(self):
        # First batch starts with an empty bank; after source insertion, skip
        # its graph-free backward once. Later batches must train normally.
        self.assertTrue(should_skip_stage2_optimizer_step(True, 1))
        self.assertFalse(should_skip_stage2_optimizer_step(False, 1))
        with self.assertRaisesRegex(RuntimeError, "warm-up failed"):
            should_skip_stage2_optimizer_step(True, 0)

    def test_source_only_balanced_bank(self):
        self.fill_source_memory()
        self.assertEqual(self.memory.num_entries, 60)
        counts = torch.bincount(self.memory.memory_city_ids, minlength=3)
        self.assertTrue(torch.all(counts == 20))

        before = self.memory.num_entries
        target_data, target_base = self.make_batch("target", [0, 1])
        self.memory.update(target_data, target_base)
        self.assertEqual(self.memory.num_entries, before)

    def test_target_retrieval_shape_and_gate_gradient(self):
        self.fill_source_memory()
        data, _ = self.make_batch("target", [20, 21], batch_size=2, node_num=3)
        history = data.x[..., 0]
        representation = torch.randn(2, 3, 128)

        correction, retrieved = self.memory.retrieve(history, representation, data)
        self.assertEqual(tuple(correction.shape), (2, 3, 36))
        self.assertEqual(tuple(retrieved.shape), (2, 3, 36))
        self.assertTrue(torch.isfinite(correction).all())

        correction.square().mean().backward()
        self.assertIsNotNone(self.memory.residual_gate[-2].weight.grad)

    def test_overlap_mask_and_checkpoint_roundtrip(self):
        self.fill_source_memory()
        data, _ = self.make_batch("source_a", [0, 1], batch_size=2, node_num=3)
        correction, _ = self.memory.retrieve(
            data.x[..., 0], torch.randn(2, 3, 128), data
        )
        self.assertTrue(torch.isfinite(correction).all())

        restored = StationarityAwareResidualMemory(
            seq_len=288,
            pred_len=36,
            d_model=128,
            max_memory_size=120,
            source_city_names=["source_a", "source_b", "source_c"],
            topk=4,
            candidate_pool=8,
            overlap_window=27,
            query_chunk=5,
        )
        restored.load_state_dict(self.memory.state_dict())
        self.assertEqual(restored.num_entries, self.memory.num_entries)


if __name__ == "__main__":
    unittest.main()
