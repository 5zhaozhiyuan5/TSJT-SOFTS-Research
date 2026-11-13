import unittest

import torch

from model.Meta_Models.stage1_transfer import SharedPrivateReliabilityTransfer


class SharedPrivateReliabilityTransferTest(unittest.TestCase):
    def test_identity_partition_and_shapes(self):
        module = SharedPrivateReliabilityTransfer(input_dim=36)
        x = torch.randn(64, 36, requires_grad=True)
        shared, private, reconstructed, loss = module(x)
        self.assertEqual(tuple(shared.shape), (64, 24))
        self.assertEqual(tuple(private.shape), (64, 12))
        self.assertEqual(tuple(reconstructed.shape), (64, 36))
        self.assertTrue(torch.allclose(reconstructed.detach(), x.detach(), atol=1e-6))
        self.assertTrue(torch.isfinite(loss))

    def test_reliability_is_valid_probability_mass(self):
        module = SharedPrivateReliabilityTransfer(input_dim=36)
        source = torch.randn(80, 24)
        target = torch.randn(45, 24) + 0.25
        source_weights, target_weights = module.reliability_weights(source, target)
        self.assertEqual(tuple(source_weights.shape), (80,))
        self.assertEqual(tuple(target_weights.shape), (45,))
        self.assertTrue(torch.isfinite(source_weights).all())
        self.assertTrue(torch.isfinite(target_weights).all())
        self.assertTrue((source_weights > 0).all())
        self.assertTrue((target_weights > 0).all())
        self.assertAlmostEqual(float(source_weights.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(target_weights.sum()), 1.0, places=5)

    def test_source_quality_is_target_conditioned(self):
        module = SharedPrivateReliabilityTransfer(input_dim=36)
        target = torch.randn(64, 24)
        near_source = target.mean(dim=0, keepdim=True) + 0.01 * torch.randn(40, 24)
        far_source = near_source + 4.0
        near_score = module.source_quality(near_source, target)
        far_score = module.source_quality(far_source, target)
        self.assertGreaterEqual(float(near_score), float(far_score))
        self.assertGreaterEqual(float(far_score), 0.05)
        self.assertLessEqual(float(near_score), 1.0)

    def test_gradient_flow(self):
        module = SharedPrivateReliabilityTransfer(input_dim=36)
        x = torch.randn(32, 36, requires_grad=True)
        shared, private, _, loss = module(x)
        objective = loss + shared.square().mean() + private.square().mean()
        objective.backward()
        self.assertIsNotNone(module.shared_proj.weight.grad)
        self.assertIsNotNone(module.private_proj.weight.grad)
        self.assertIsNotNone(module.reconstruct.weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
