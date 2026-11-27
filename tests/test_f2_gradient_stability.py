"""CPU tests for full-horizon target-gradient stability calibration."""
import os
import sys
import unittest

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "model", "Meta_Models")
)
from transfer_utility import compute_weights
from f2_four_extensions import compute_f2_extension


class F2GradientStabilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261128)
        self.fs = torch.randn(2, 8, 16)
        self.ds = torch.randn(2, 8, 36) * 0.01
        self.ft = torch.randn(12, 6, 16)
        self.dt = torch.randn(12, 6, 36) * 0.01
        self.alpha0, self.beta0, _ = compute_weights(
            self.fs, self.ds, self.ft, self.dt, "gradient"
        )

    def run_stability(self, **kwargs):
        return compute_f2_extension(
            self.fs, self.ds, self.ft, self.dt,
            self.alpha0, self.beta0, "stability",
            temperature=8.0, **kwargs
        )

    def test_valid_marginals_and_target_prior_unchanged(self):
        alpha, beta, info = self.run_stability(
            stability_lambda=0.5,
            stability_subsamples=8,
            stability_sample_ratio=0.8,
            stability_seed=123,
        )
        self.assertEqual(alpha.shape, self.alpha0.shape)
        self.assertEqual(beta.shape, self.beta0.shape)
        self.assertTrue(torch.isfinite(alpha).all())
        self.assertTrue(torch.isfinite(beta).all())
        self.assertTrue(bool((alpha >= 0).all()))
        self.assertAlmostEqual(float(alpha.sum()), 1.0, places=6)
        self.assertTrue(torch.equal(beta, self.beta0))
        self.assertTrue(info["target_subsample_active"])
        self.assertEqual(info["gradient_dimensions"], 36)
        self.assertFalse(info["prediction_horizon_split"])
        self.assertEqual(info["target_subset_windows"], 10)
        self.assertIn("target_cosine_dispersion_p90", info)

    def test_deterministic_for_same_seed(self):
        a1, _, _ = self.run_stability(stability_seed=99)
        a2, _, _ = self.run_stability(stability_seed=99)
        self.assertTrue(torch.equal(a1, a2))

    def test_zero_penalty_recovers_original_f2(self):
        alpha, beta, _ = self.run_stability(
            stability_lambda=0.0, stability_seed=31
        )
        self.assertTrue(torch.allclose(alpha, self.alpha0, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.equal(beta, self.beta0))

    def test_uncertain_source_gradient_is_downweighted(self):
        # Both source samples have zero cosine to the full target gradient.
        # The first sample's alignment flips with target-window subsampling;
        # the second sample is orthogonal and has zero dispersion.
        fs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        ds = torch.zeros(1, 2, 36)
        ds[..., 0] = 1.0
        ft = torch.zeros(8, 1, 2)
        ft[..., 0] = 1.0
        dt = torch.zeros(8, 1, 36)
        dt[:4, 0, 0] = 1.0
        dt[4:, 0, 0] = -1.0
        alpha0, beta0, _ = compute_weights(fs, ds, ft, dt, "gradient")
        alpha, beta, info = compute_f2_extension(
            fs, ds, ft, dt, alpha0, beta0, "stability",
            temperature=8.0, stability_lambda=1.0,
            stability_subsamples=64, stability_sample_ratio=0.5,
            stability_seed=7,
        )
        self.assertTrue(torch.allclose(alpha0, torch.tensor([0.5, 0.5])))
        self.assertLess(float(alpha[0]), float(alpha[1]))
        self.assertTrue(torch.equal(beta, beta0))
        self.assertGreater(info["target_cosine_dispersion_mean"], 0.0)

    def test_too_few_target_windows_falls_back_without_nan(self):
        ft = self.ft[:2]
        dt = self.dt[:2]
        alpha0, beta0, _ = compute_weights(self.fs, self.ds, ft, dt, "gradient")
        alpha, beta, info = compute_f2_extension(
            self.fs, self.ds, ft, dt, alpha0, beta0, "stability"
        )
        self.assertTrue(torch.isfinite(alpha).all())
        self.assertTrue(torch.equal(beta, beta0))
        self.assertFalse(info["target_subsample_active"])
        self.assertEqual(info["target_subsample_reason"], "fewer_than_3_target_windows")


if __name__ == "__main__":
    unittest.main()
