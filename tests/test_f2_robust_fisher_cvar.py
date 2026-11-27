"""CPU tests for the three gradient-only F2 refinements."""
import os
import sys
import unittest

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "model", "Meta_Models")
)
from transfer_utility import compute_weights
from f2_four_extensions import (
    compute_f2_extension,
    _geometric_median,
    _lower_tail_mean,
    _rankone_metric_utility,
    _robust_tanh_standardize,
    _temporal_block_gradients,
)


class RobustF2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261128)
        self.fs = torch.randn(2, 9, 128)
        self.ds = torch.randn(2, 9, 36) * 0.005
        # Fifteen target windows mirror the smallest PMME target group.
        self.ft = torch.randn(15, 10, 128)
        self.dt = torch.randn(15, 10, 36) * 0.007
        self.alpha0, self.beta0, _ = compute_weights(
            self.fs, self.ds, self.ft, self.dt, "gradient"
        )

    def run_method(self, method, ft=None, dt=None, **kwargs):
        return compute_f2_extension(
            self.fs, self.ds,
            self.ft if ft is None else ft,
            self.dt if dt is None else dt,
            self.alpha0, self.beta0, method,
            temperature=8.0, **kwargs
        )

    def test_all_new_methods_return_valid_source_marginals(self):
        for method in ("geomed", "influence", "cvar", "triple"):
            with self.subTest(method=method):
                alpha, beta, info = self.run_method(method)
                self.assertEqual(alpha.shape, self.alpha0.shape)
                self.assertEqual(beta.shape, self.beta0.shape)
                self.assertTrue(torch.isfinite(alpha).all())
                self.assertTrue(torch.isfinite(beta).all())
                self.assertGreaterEqual(float(alpha.min()), 0.0)
                self.assertAlmostEqual(float(alpha.sum()), 1.0, places=6)
                self.assertTrue(torch.equal(beta, self.beta0))
                self.assertEqual(info["method"], method)
                self.assertTrue(info["target_marginal_unchanged"])
                self.assertFalse(info["prediction_horizon_split"])
                self.assertIn("source_ess_frac", info)

    def test_geometric_median_resists_one_extreme_target_window(self):
        points = torch.tensor([
            [0.00, 0.00],
            [0.10, 0.00],
            [-0.10, 0.00],
            [0.00, 0.10],
            [100.0, 100.0],
        ])
        mean = points.mean(dim=0)
        median = _geometric_median(points, iterations=12)
        self.assertLess(float(median.norm()), float(mean.norm()) * 0.15)

    def test_fisher_influence_proxy_is_finite_with_zero_target_gradients(self):
        ft = torch.zeros_like(self.ft)
        dt = torch.zeros_like(self.dt)
        alpha0, beta0, _ = compute_weights(
            self.fs, self.ds, ft, dt, "gradient"
        )
        alpha, beta, info = compute_f2_extension(
            self.fs, self.ds, ft, dt, alpha0, beta0,
            "influence", fisher_damping=0.1
        )
        self.assertTrue(torch.isfinite(alpha).all())
        self.assertTrue(torch.equal(beta, beta0))
        self.assertAlmostEqual(float(alpha.sum()), 1.0, places=6)
        self.assertEqual(info["fisher_damping"], 0.1)

    def test_influence_utility_sign_matches_virtual_target_loss_change(self):
        # P=I, source gradient is +1 or -1 in one coordinate, target gradient +1.
        source_features = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        source_delta = torch.zeros(2, 36)
        source_delta[0, 0] = 1.0
        source_delta[1, 0] = -1.0
        target_grad = torch.zeros(36, 2)
        target_grad[0, 0] = 1.0
        precision = torch.ones_like(target_grad)
        utility = _rankone_metric_utility(
            source_features, source_delta, target_grad, precision
        )
        self.assertGreater(float(utility[0]), 0.0)
        self.assertLess(float(utility[1]), 0.0)
        standardized = _robust_tanh_standardize(utility)
        self.assertTrue(torch.isfinite(standardized).all())
        self.assertGreater(float(standardized[0]), 0.0)
        self.assertLess(float(standardized[1]), 0.0)
        self.assertTrue(bool((standardized.abs() <= 1.0).all()))

    def test_cvar_uses_lower_tail_not_mean(self):
        scores = torch.tensor([
            [0.90, 0.40],
            [0.80, 0.50],
            [0.70, -0.50],
            [0.60, 0.80],
            [0.50, 0.60],
        ])
        tail, count = _lower_tail_mean(scores, tail_fraction=0.4)
        self.assertEqual(count, 2)
        self.assertTrue(torch.all(tail < scores.mean(dim=0)))

    def test_target_windows_are_grouped_into_contiguous_blocks(self):
        windows = torch.arange(15 * 2, dtype=torch.float32).reshape(15, 2, 1)
        blocks = _temporal_block_gradients(windows, requested_blocks=5)
        self.assertEqual(tuple(blocks.shape), (3, 2, 1))
        expected = torch.stack([
            windows[:5].mean(dim=0),
            windows[5:10].mean(dim=0),
            windows[10:15].mean(dim=0),
        ])
        self.assertTrue(torch.equal(blocks, expected))

    def test_triple_is_deterministic_and_changes_original_f2_weights(self):
        alpha1, beta1, info1 = self.run_method("triple")
        alpha2, beta2, info2 = self.run_method("triple")
        self.assertTrue(torch.equal(alpha1, alpha2))
        self.assertTrue(torch.equal(beta1, beta2))
        self.assertGreater(float(info1["source_l1_shift"]), 0.0)
        self.assertEqual(info1["cvar_blocks"], 3)
        self.assertEqual(info1["cvar_tail_count"], 2)
        self.assertAlmostEqual(
            info1["final_score_mean"], info2["final_score_mean"], places=7
        )

    def test_small_target_group_falls_back_to_one_block_without_nan(self):
        ft = self.ft[:1]
        dt = self.dt[:1]
        alpha0, beta0, _ = compute_weights(
            self.fs, self.ds, ft, dt, "gradient"
        )
        for method in ("geomed", "influence", "cvar", "triple"):
            with self.subTest(method=method):
                alpha, beta, info = compute_f2_extension(
                    self.fs, self.ds, ft, dt, alpha0, beta0, method
                )
                self.assertTrue(torch.isfinite(alpha).all())
                self.assertTrue(torch.equal(beta, beta0))
                self.assertEqual(info["cvar_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
