"""CPU-only mathematical smoke checks for the F1-F4 OT marginals.

Run: python -m unittest discover -s tests -p 'test_transfer_utility.py'
"""
import unittest
import torch
from model.Meta_Models.transfer_utility import VARIANTS, compute_weights


class TransferUtilityTests(unittest.TestCase):
    def test_normalization_finite_and_ess(self):
        torch.manual_seed(9)
        source_features = torch.randn(2, 7, 128)
        target_features = torch.randn(3, 5, 128)
        source_delta = torch.randn(2, 7, 36)
        target_delta = torch.randn(3, 5, 36)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                a, b, stats = compute_weights(
                    source_features, source_delta,
                    target_features, target_delta, variant
                )
                self.assertEqual(tuple(a.shape), (14,))
                self.assertEqual(tuple(b.shape), (15,))
                self.assertTrue(torch.isfinite(a).all().item())
                self.assertTrue(torch.isfinite(b).all().item())
                self.assertTrue((a >= 0).all().item())
                self.assertTrue((b >= 0).all().item())
                self.assertAlmostEqual(a.sum().item(), 1.0, places=5)
                self.assertAlmostEqual(b.sum().item(), 1.0, places=5)
                if variant == "uniform":
                    self.assertTrue(torch.allclose(a, torch.full((14,), 1 / 14)))
                if variant == "reliable":
                    self.assertGreaterEqual(stats["ess_frac"], 0.5 - 1e-5)

    def test_head_gradient_path_and_zero_gradients(self):
        torch.manual_seed(10)
        head = torch.nn.Linear(128, 36)
        a = torch.randn(2, 6, 128, requires_grad=True)
        b = torch.randn(2, 5, 128, requires_grad=True)
        ya, yb = head(a), head(b)
        la = torch.nn.functional.smooth_l1_loss(ya, torch.randn_like(ya))
        lb = torch.nn.functional.smooth_l1_loss(yb, torch.randn_like(yb))
        da = torch.autograd.grad(la, ya, retain_graph=True)[0]
        db = torch.autograd.grad(lb, yb, retain_graph=True)[0]
        for variant in ("gradient", "horizon", "reliable"):
            source, target, _ = compute_weights(a.detach(), da.detach(),
                                                 b.detach(), db.detach(), variant)
            self.assertAlmostEqual(source.sum().item(), 1.0, places=5)
            self.assertAlmostEqual(target.sum().item(), 1.0, places=5)
            source_zero, _, _ = compute_weights(
                a.detach(), torch.zeros_like(da), b.detach(), db.detach(), variant
            )
            self.assertTrue(torch.isfinite(source_zero).all().item())


if __name__ == "__main__":
    unittest.main()
