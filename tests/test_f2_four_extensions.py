"""CPU unit tests for all four F2-only marginal extensions.

This test requires torch, not the traffic dataset, GeomLoss or CUDA.
Run: python -m unittest discover -s tests -p 'test_f2_four_extensions.py'
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model", "Meta_Models"))
from transfer_utility import compute_weights
from f2_four_extensions import compute_f2_extension


class FourF2ExtensionsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(311)
        self.fs = torch.randn(2, 9, 128)
        self.ds = torch.randn(2, 9, 36) * 0.005
        self.ft = torch.randn(3, 10, 128)
        self.dt = torch.randn(3, 10, 36) * 0.007
        self.alpha0, self.beta0, _ = compute_weights(
            self.fs, self.ds, self.ft, self.dt, "gradient"
        )

    def run_method(self, method, state=None, key=0, dt=None):
        return compute_f2_extension(
            self.fs, self.ds, self.ft,
            self.dt if dt is None else dt,
            self.alpha0, self.beta0, method,
            state=state, state_key=key
        )

    def test_all_variants_produce_valid_marginals(self):
        for method in ("mag", "sign", "ema", "biot"):
            with self.subTest(method=method):
                alpha, beta, info = self.run_method(method, state={})
                self.assertEqual(alpha.shape, self.alpha0.shape)
                self.assertEqual(beta.shape, self.beta0.shape)
                self.assertTrue(torch.isfinite(alpha).all().item())
                self.assertTrue(torch.isfinite(beta).all().item())
                self.assertGreaterEqual(float(alpha.min()), 0.0)
                self.assertGreaterEqual(float(beta.min()), 0.0)
                self.assertAlmostEqual(float(alpha.sum()), 1.0, places=5)
                self.assertAlmostEqual(float(beta.sum()), 1.0, places=5)
                self.assertEqual(info["method"], method)
                self.assertIn("source_ess_frac", info)

    def test_original_is_exact_identity(self):
        alpha, beta, _ = self.run_method("original")
        self.assertTrue(torch.equal(alpha, self.alpha0))
        self.assertTrue(torch.equal(beta, self.beta0))

    def test_one_sided_methods_leave_target_uniform(self):
        for method in ("mag", "sign", "ema"):
            alpha, beta, _ = self.run_method(method, state={})
            with self.subTest(method=method):
                self.assertTrue(torch.allclose(beta, self.beta0, atol=1e-7))

    def test_biot_changes_both_marginals_and_preserves_target_floor(self):
        alpha, beta, info = self.run_method("biot")
        self.assertGreater(info["source_l1_shift"], 1e-8)
        self.assertGreater(info["target_l1_shift"], 1e-8)
        # Retain 60% of uniform target mass by the conservative blend.
        self.assertTrue(torch.all(beta >= 0.6 * self.beta0 - 1e-7))

    def test_ema_uses_persistent_target_history(self):
        state = {}
        _, _, first = self.run_method("ema", state=state, key="target-group-0")
        self.assertTrue(first["ema_initialized"])
        prior = state["target-group-0"].clone()
        changed = self.dt + 0.5 * torch.randn_like(self.dt)
        alpha, _, second = self.run_method(
            "ema", state=state, key="target-group-0", dt=changed
        )
        self.assertFalse(second["ema_initialized"])
        self.assertEqual(alpha.shape, self.alpha0.shape)
        self.assertFalse(torch.equal(state["target-group-0"], prior))

    def test_magnitude_calibration_is_scale_invariant(self):
        alpha1, _, _ = self.run_method("mag")
        new_alpha_prior, new_beta_prior, _ = compute_weights(
            self.fs, self.ds * 1000, self.ft, self.dt * 0.01, "gradient"
        )
        alpha2, _, _ = compute_f2_extension(
            self.fs, self.ds * 1000, self.ft, self.dt * 0.01,
            new_alpha_prior, new_beta_prior, "mag"
        )
        self.assertTrue(torch.allclose(alpha1, alpha2, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
