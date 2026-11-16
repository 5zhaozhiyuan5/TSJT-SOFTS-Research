import unittest

import numpy as np
import torch

from model.Meta_Models.pattern_moe import (
    HorizonPatternMoEAlignment,
    build_multiscale_pattern_bank,
)
from model.Meta_Models.horizon_transfer import build_future_change_descriptors


class PatternMoEAlignmentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        np.random.seed(17)

    def test_source_only_pattern_bank_builder(self):
        source_arrays = {
            "city_a": np.random.randn(8, 5, 288, 1).astype(np.float32),
            "city_b": np.random.randn(7, 4, 288, 1).astype(np.float32),
        }
        bank = build_multiscale_pattern_bank(
            source_arrays,
            num_prototypes=4,
            descriptor_bins=8,
            max_windows_per_city=5,
            max_nodes_per_city=3,
            max_patches_per_scale=500,
            seed=17,
        )
        self.assertEqual(tuple(bank.shape), (5, 4, 8))
        self.assertTrue(torch.isfinite(bank).all())

    def test_horizon_moe_ot_is_differentiable_and_uses_three_experts(self):
        bank = torch.randn(5, 6, 12)
        module = HorizonPatternMoEAlignment(
            pattern_bank=bank,
            input_dim=128,
            horizon_steps=36,
            minutes_per_step=5,
            anchor_steps=(2, 12, 24, 36),
            scale_steps=(12, 36, 72, 144, 288),
            time_dim=12,
            hidden_dim=32,
            relation_dim=16,
            epsilon=0.08,
            sinkhorn_iterations=20,
            max_samples=24,
            pattern_temperature=0.2,
            router_temperature=1.0,
        )

        source_rep = torch.randn(7, 128, requires_grad=True)
        target_rep = torch.randn(9, 128, requires_grad=True)
        source_history = torch.randn(7, 288)
        target_history = torch.randn(9, 288)
        source_future = torch.randn(1, 7, 36)
        target_future = torch.randn(3, 3, 36)
        source_last = torch.randn(1, 7)
        target_last = torch.randn(3, 3)

        source_desc = build_future_change_descriptors(
            source_future, source_last, module.anchor_steps
        ).reshape(-1, 4, 4)
        target_desc = build_future_change_descriptors(
            target_future, target_last, module.anchor_steps
        ).reshape(-1, 4, 4)

        source_base = torch.rand(7) + 0.1
        target_base = torch.rand(9) + 0.1
        source_marginals, target_marginals, _ = module.horizon_marginals(
            source_base, target_base, source_desc, target_desc
        )
        loss, diagnostics = module.compute_alignment_loss(
            source_rep,
            target_rep,
            source_history,
            target_history,
            source_desc,
            target_desc,
            source_marginals,
            target_marginals,
            source_base,
            target_base,
        )
        (loss + 1e-4 * module.smoothness_loss()).backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(diagnostics["router_weights"].shape), (4, 3))
        self.assertTrue(
            torch.allclose(
                diagnostics["router_weights"].sum(dim=-1),
                torch.ones(4),
                atol=1e-6,
            )
        )
        self.assertGreater(float(source_rep.grad.abs().sum()), 0.0)
        self.assertGreater(float(target_rep.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.router[-1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.history_projection.weight.grad.abs().sum()), 0.0)
        self.assertLess(float(diagnostics["max_row_marginal_error"]), 1e-2)
        self.assertLess(float(diagnostics["max_col_marginal_error"]), 1e-2)


if __name__ == "__main__":
    unittest.main()
