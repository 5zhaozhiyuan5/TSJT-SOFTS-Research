"""CPU-only checks for F2 internal bidirectional marginal refinement.

Run: python -m unittest discover -s tests -p 'test_f2_internal_bilateral.py'
No full traffic prediction/GeomLoss/CUDA experiment is implied by these tests.
"""
import unittest
import torch
from model.Meta_Models.transfer_utility import compute_weights


class F2InternalBilateralTests(unittest.TestCase):
    def _random_inputs(self):
        torch.manual_seed(925)
        return (torch.randn(2, 5, 128), torch.randn(2, 5, 36),
                torch.randn(3, 4, 128), torch.randn(3, 4, 36))

    def test_zero_blend_and_zero_steps_recover_original_f2(self):
        fs, ds, ft, dt = self._random_inputs()
        base_a, base_b, _ = compute_weights(fs, ds, ft, dt, "gradient")
        for kwargs in ({"coupled_blend": 0.0, "coupled_steps": 3},
                       {"coupled_blend": 0.7, "coupled_steps": 0}):
            with self.subTest(kwargs=kwargs):
                a, b, _ = compute_weights(
                    fs, ds, ft, dt, "gradient_coupled", **kwargs)
                self.assertTrue(torch.allclose(a, base_a, atol=1e-7, rtol=1e-7))
                self.assertTrue(torch.allclose(b, base_b, atol=1e-7, rtol=1e-7))

    def test_both_marginals_probability_and_minimum_mass(self):
        fs, ds, ft, dt = self._random_inputs()
        prior_a, prior_b, _ = compute_weights(fs, ds, ft, dt, "gradient")
        a, b, stats = compute_weights(
            fs, ds, ft, dt, "gradient_coupled",
            coupled_blend=0.2, coupled_steps=2)
        self.assertEqual(a.shape, prior_a.shape)
        self.assertEqual(b.shape, prior_b.shape)
        for w in (a, b):
            self.assertTrue(torch.isfinite(w).all().item())
            self.assertTrue((w >= 0).all().item())
            self.assertAlmostEqual(w.sum().item(), 1.0, places=5)
            self.assertFalse(w.requires_grad)
        self.assertTrue((a >= 0.8 * prior_a - 1e-7).all().item())
        self.assertTrue((b >= 0.8 * prior_b - 1e-7).all().item())
        self.assertGreater(stats["target_ess_frac"], 0)
        self.assertLessEqual(stats["target_ess_frac"], 1.0 + 1e-5)

    def test_target_weight_responds_to_selected_source_gradient(self):
        fs = torch.zeros(1, 4, 128)
        ft = torch.zeros(1, 4, 128)
        fs[..., 0] = 1.0
        ft[..., 0] = 1.0
        ds = torch.zeros(1, 4, 36)
        dt = torch.zeros(1, 4, 36)
        ds[..., 0] = 1.0
        dt[..., 0] = torch.tensor([1.0, 1.0, -1.0, -1.0])
        baseline_a, baseline_b, _ = compute_weights(
            fs, ds, ft, dt, "gradient")
        a, b, _ = compute_weights(
            fs, ds, ft, dt, "gradient_coupled",
            coupled_blend=0.2, coupled_steps=2)
        self.assertTrue(torch.allclose(baseline_b, torch.full_like(baseline_b, 0.25)))
        self.assertGreater(b[0].item(), b[2].item())
        self.assertGreater(b[1].item(), b[3].item())
        self.assertTrue(torch.allclose(a, baseline_a, atol=1e-6))

    def test_zero_gradients_stable_and_detached(self):
        fs, ds, ft, dt = self._random_inputs()
        fs.requires_grad_()
        for zero in ("source", "target"):
            a, b, stats = compute_weights(
                fs, torch.zeros_like(ds) if zero == "source" else ds,
                ft, torch.zeros_like(dt) if zero == "target" else dt,
                "gradient_coupled")
            self.assertTrue(torch.isfinite(a).all().item())
            self.assertTrue(torch.isfinite(b).all().item())
            self.assertFalse(a.requires_grad)
            self.assertFalse(b.requires_grad)
            self.assertAlmostEqual(a.sum().item(), 1.0, places=5)
            self.assertAlmostEqual(b.sum().item(), 1.0, places=5)

    def test_invalid_config(self):
        fs, ds, ft, dt = self._random_inputs()
        for blend in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                compute_weights(fs, ds, ft, dt, "gradient_coupled",
                                coupled_blend=blend)


if __name__ == "__main__":
    unittest.main()
