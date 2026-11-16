import unittest

import numpy as np
import torch

from model.Meta_Models.pattern_moe import (
    HorizonPatternMoEAlignment,
    build_multiscale_pattern_bank,
    physical_minutes_to_steps,
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

    def test_physical_durations_resolve_on_five_and_ten_minute_grids(self):
        scale_minutes = (60, 180, 360, 720, 1440)
        anchor_minutes = (10, 60, 120, 180)
        self.assertEqual(
            physical_minutes_to_steps(scale_minutes, 5),
            (12, 36, 72, 144, 288),
        )
        self.assertEqual(
            physical_minutes_to_steps(scale_minutes, 10),
            (6, 18, 36, 72, 144),
        )
        self.assertEqual(
            physical_minutes_to_steps(anchor_minutes, 5),
            (2, 12, 24, 36),
        )
        self.assertEqual(
            physical_minutes_to_steps(anchor_minutes, 10),
            (1, 6, 12, 18),
        )
        with self.assertRaises(ValueError):
            physical_minutes_to_steps((10, 60), 6)

    def test_ten_minute_raw_grid_can_build_physical_twenty_four_hour_bank(self):
        source_arrays = {
            "city_10min": np.random.randn(5, 4, 144, 1).astype(np.float32),
        }
        scale_steps = physical_minutes_to_steps((60, 180, 360, 720, 1440), 10)
        bank = build_multiscale_pattern_bank(
            source_arrays,
            scale_steps=scale_steps,
            num_prototypes=3,
            descriptor_bins=8,
            max_windows_per_city=4,
            max_nodes_per_city=3,
            max_patches_per_scale=300,
            seed=19,
        )
        self.assertEqual(tuple(bank.shape), (5, 3, 8))
        self.assertTrue(torch.isfinite(bank).all())

    def test_pattern_bank_is_invariant_to_per_patch_affine_speed_scaling(self):
        base = np.random.randn(5, 4, 288, 1).astype(np.float32)
        shifted_scaled = 3.5 * base + 17.0
        kwargs = dict(
            scale_steps=(12, 36, 72, 144, 288),
            num_prototypes=4,
            descriptor_bins=8,
            max_windows_per_city=4,
            max_nodes_per_city=3,
            max_patches_per_scale=500,
            seed=23,
        )
        bank_a = build_multiscale_pattern_bank({"city": base}, **kwargs)
        bank_b = build_multiscale_pattern_bank({"city": shifted_scaled}, **kwargs)
        torch.testing.assert_close(bank_a, bank_b, atol=2e-5, rtol=2e-5)

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
        (
            loss
            + 1e-4 * module.smoothness_loss()
            + 1e-4 * diagnostics["router_smoothness_loss"]
        ).backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(diagnostics["router_smoothness_loss"]))
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
        # The log-domain Sinkhorn solver uses 20 alternating updates; the
        # row marginal is therefore approximate while the final column update
        # is exact up to floating-point error.
        self.assertLess(float(diagnostics["max_row_marginal_error"]), 3e-2)
        self.assertLess(float(diagnostics["max_col_marginal_error"]), 1e-2)
        for key in (
            "plan_entropy",
            "normalized_plan_entropy",
            "mean_row_entropy",
            "plan_max_weight",
            "row_top1_mass",
            "pattern_usage",
            "pattern_entropy",
            "pattern_top1_fraction",
            "router_pair_std",
            "router_top1_fraction",
        ):
            self.assertIn(key, diagnostics)
            self.assertTrue(torch.isfinite(diagnostics[key]).all())


if __name__ == "__main__":
    unittest.main()
