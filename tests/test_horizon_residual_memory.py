import os
import sys
import unittest

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.Meta_Models.horizon_residual_memory import (
    HorizonConditionedResidualMemory,
    configure_horizon_stage2_parameters,
)


class HorizonResidualMemoryTest(unittest.TestCase):
    def test_target_support_prototype_initialization_is_deterministic(self):
        features = torch.randn(31, 12)
        first = HorizonConditionedResidualMemory(
            d_model=12, pred_len=12, memory_size=20, query_chunk=5
        )
        second = HorizonConditionedResidualMemory(
            d_model=12, pred_len=12, memory_size=20, query_chunk=5
        )

        count_first = first.initialize_from_features(features, seed=17)
        count_second = second.initialize_from_features(features, seed=17)

        self.assertEqual(count_first, 31)
        self.assertEqual(count_second, 31)
        self.assertTrue(bool(first.memory_initialized.item()))
        self.assertEqual(int(first.initialization_feature_count.item()), 31)
        self.assertTrue(torch.equal(first.memory, second.memory))
        self.assertEqual(tuple(first.memory.shape), (20, 12))
        self.assertTrue(torch.isfinite(first.memory).all())

    def test_small_support_set_fills_all_slots_without_exact_duplicates(self):
        memory = HorizonConditionedResidualMemory(
            d_model=8, pred_len=12, memory_size=10, query_chunk=3
        )
        features = torch.randn(4, 8)
        memory.initialize_from_features(features, seed=3)

        self.assertEqual(memory.memory.shape[0], 10)
        self.assertTrue(torch.isfinite(memory.memory).all())
        self.assertFalse(torch.equal(memory.memory[0], memory.memory[4]))

    def test_three_horizon_queries_return_36_steps_and_receive_gradients(self):
        torch.manual_seed(9)
        memory = HorizonConditionedResidualMemory(
            d_model=16, pred_len=36, memory_size=24, query_chunk=4
        )
        memory.initialize_from_features(torch.randn(60, 16), seed=9)
        memory.train()

        representation = torch.randn(2, 3, 16)
        correction, entropy = memory(representation)

        self.assertEqual(tuple(correction.shape), (2, 3, 36))
        self.assertEqual(tuple(entropy.shape), (3,))
        self.assertTrue(torch.isfinite(correction).all())
        self.assertTrue(torch.isfinite(entropy).all())

        loss = correction.square().mean()
        loss.backward()

        self.assertIsNotNone(memory.memory.grad)
        self.assertIsNotNone(memory.horizon_embeddings.grad)
        self.assertIsNotNone(memory.Wq.weight.grad)
        self.assertIsNotNone(memory.Wk.weight.grad)
        self.assertIsNotNone(memory.Wv.weight.grad)
        for head in memory.bias_heads:
            self.assertIsNotNone(head.weight.grad)
            self.assertTrue(torch.isfinite(head.weight.grad).all())

    def test_freeze_stage1_and_train_only_stage2_memory(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.forecasting_model = nn.Linear(4, 4)
                self.horizon_memory = HorizonConditionedResidualMemory(
                    d_model=4, pred_len=12, memory_size=8
                )

        model = DummyModel()
        trainable = configure_horizon_stage2_parameters(model)

        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.forecasting_model.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.horizon_memory.parameters())
        )
        self.assertEqual(
            {id(parameter) for parameter in trainable},
            {id(parameter) for parameter in model.horizon_memory.parameters()},
        )


if __name__ == "__main__":
    unittest.main()
