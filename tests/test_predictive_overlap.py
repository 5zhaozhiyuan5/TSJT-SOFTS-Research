"""CPU smoke tests for the Stage-I predictive-overlap DVGP replacement.

Run: python -m unittest discover -s tests -p 'test_predictive_overlap.py'
"""
import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model", "Meta_Models"))
from predictive_overlap import PredictiveOverlapWeighter


class PredictiveOverlapTests(unittest.TestCase):
    def test_future_teacher_prefers_similar_evolution(self):
        torch.manual_seed(7)
        model = PredictiveOverlapWeighter(teacher_steps=8, prototype_count=3)
        target = torch.linspace(0, 1, 36).repeat(5, 1)
        similar = torch.linspace(2, 3, 36).repeat(5, 1)
        reversed_trend = torch.linspace(1, 0, 36).repeat(5, 1)
        scores_similar, _ = model._teacher(similar, target)
        scores_opposite, _ = model._teacher(reversed_trend, target)
        self.assertGreater(scores_similar.mean().item(), scores_opposite.mean().item())

    def test_masses_are_positive_normalisable_and_detached(self):
        torch.manual_seed(9)
        model = PredictiveOverlapWeighter(teacher_steps=8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        source = torch.randn(10, 128, requires_grad=True)
        target = torch.randn(7, 128, requires_grad=True)
        future_source = torch.randn(10, 36)
        future_target = torch.randn(7, 36)
        for domain in range(3):
            alpha_raw, beta_raw, metrics = model.training_weights(
                domain, source, target, future_source, future_target, opt)
            self.assertEqual(tuple(alpha_raw.shape), (10,))
            self.assertEqual(tuple(beta_raw.shape), (7,))
            self.assertFalse(alpha_raw.requires_grad)
            self.assertFalse(beta_raw.requires_grad)
            self.assertTrue(torch.isfinite(alpha_raw).all())
            self.assertTrue(torch.isfinite(beta_raw).all())
            self.assertTrue((alpha_raw > 0).all())
            self.assertTrue((beta_raw > 0).all())
            self.assertAlmostEqual((alpha_raw / alpha_raw.sum()).sum().item(), 1.0, places=5)
            self.assertAlmostEqual((beta_raw / beta_raw.sum()).sum().item(), 1.0, places=5)
            self.assertTrue(torch.isfinite(metrics["future_loss"]).all())
        self.assertIsNone(source.grad)
        self.assertIsNone(target.grad)


if __name__ == "__main__":
    unittest.main()
