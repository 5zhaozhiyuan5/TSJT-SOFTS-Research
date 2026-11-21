"""CPU contract checks for conditional evolution + pairwise Sinkhorn cost."""
import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model", "Meta_Models"))
from predictive_overlap import PredictiveOverlapWeighter, conditional_ot_cost


class PredictiveOverlapTests(unittest.TestCase):
    def test_history_changes_teacher_even_when_future_is_identical(self):
        model = PredictiveOverlapWeighter(teacher_steps=6, prototype_count=1)
        history_rising = torch.linspace(-1, 1, 24).unsqueeze(0)
        history_falling = torch.linspace(1, -1, 24).unsqueeze(0)
        future = torch.linspace(1.1, 2, 12).unsqueeze(0)
        h_a, f_a = model._conditional_sequences(history_rising, future)
        h_b, f_b = model._conditional_sequences(history_falling, future)
        distance = model._distance(h_a, f_a, h_b, f_b)
        identical = model._distance(h_a, f_a, h_a, f_a)
        self.assertGreater(distance.item(), identical.item() + 0.01)

    def test_weights_and_pairwise_teacher_train_without_backbone_gradient(self):
        torch.manual_seed(9)
        model = PredictiveOverlapWeighter(
            teacher_steps=6, prototype_count=2, pair_samples=8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        source = torch.randn(10, 128, requires_grad=True)
        target = torch.randn(7, 128, requires_grad=True)
        source_history = torch.randn(10, 24)
        target_history = torch.randn(7, 24)
        source_future = torch.randn(10, 12)
        target_future = torch.randn(7, 12)
        for domain in range(3):
            a, b, info = model.training_weights(
                domain, source, target, source_future, target_future,
                source_history, target_history, opt)
            self.assertEqual(tuple(a.shape), (10,))
            self.assertEqual(tuple(b.shape), (7,))
            self.assertFalse(a.requires_grad)
            self.assertTrue((a > 0).all() and (b > 0).all())
            self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
            self.assertAlmostEqual((a/a.sum()).sum().item(), 1.0, places=5)
            self.assertAlmostEqual((b/b.sum()).sum().item(), 1.0, places=5)
            self.assertTrue(torch.isfinite(info["pair_loss"]).all())
            z1, z2 = model.pair_embeddings(domain, source, target)
            self.assertEqual(z1.shape, (10, 32))
            self.assertEqual(z2.shape, (7, 32))
        self.assertIsNone(source.grad)
        self.assertIsNone(target.grad)

    def test_custom_cost_has_shape_and_original_cost_at_zero_weight(self):
        torch.manual_seed(2)
        original_a = torch.randn(1, 4, 8, requires_grad=True)
        original_b = torch.randn(1, 3, 8, requires_grad=True)
        embed_a = torch.nn.functional.normalize(torch.randn(1, 4, 32), dim=-1)
        embed_b = torch.nn.functional.normalize(torch.randn(1, 3, 32), dim=-1)
        x = torch.cat([original_a, embed_a], -1)
        y = torch.cat([original_b, embed_b], -1)
        baseline = torch.cdist(original_a, original_b, p=2)
        zero = conditional_ot_cost(8, 0)(x, y)
        augmented = conditional_ot_cost(8, 0.8)(x, y)
        self.assertEqual(augmented.shape, (1, 4, 3))
        self.assertTrue(torch.allclose(zero, baseline, atol=1e-5))
        self.assertTrue(torch.all(augmented >= zero - 1e-6))
        augmented.sum().backward()
        self.assertIsNotNone(original_a.grad)
        self.assertGreater(original_a.grad.abs().sum().item(), 0)

    def test_geomloss_weighted_sinkhorn_custom_cost(self):
        try:
            from geomloss import SamplesLoss
        except ImportError:
            self.skipTest("geomloss not installed in minimal CPU environment")
        torch.manual_seed(5)
        a = torch.rand(5, 8, requires_grad=True)
        b = torch.rand(4, 8, requires_grad=True)
        xa = torch.cat((a, torch.nn.functional.normalize(
            torch.randn(5, 32), dim=-1)), dim=-1)
        xb = torch.cat((b, torch.nn.functional.normalize(
            torch.randn(4, 32), dim=-1)), dim=-1)
        alpha = torch.ones(5) / 5
        beta = torch.ones(4) / 4
        loss = SamplesLoss(loss="sinkhorn", p=1, blur=0.05,
                           debias=False, backend="tensorized",
                           cost=conditional_ot_cost(8, 0.5))
        value = loss(alpha, xa, beta, xb)
        self.assertTrue(torch.isfinite(value).all())
        value.backward()
        self.assertGreater(a.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
