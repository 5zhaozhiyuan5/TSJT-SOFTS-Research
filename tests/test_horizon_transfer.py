import unittest

import torch

from model.Meta_Models.horizon_transfer import (
    ContinuousHorizonTransferAlignment,
    build_future_change_descriptors,
)


class ContinuousHorizonTransferTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.module = ContinuousHorizonTransferAlignment(
            input_dim=128,
            horizon_steps=36,
            minutes_per_step=5,
            anchor_steps=(2, 12, 24, 36),
            time_dim=12,
            hidden_dim=32,
            relation_dim=16,
            epsilon=0.08,
            sinkhorn_iterations=30,
            max_samples=48,
            future_weight=1.0,
        )

    def test_continuous_horizon_encoder_and_future_descriptors(self):
        codes = self.module.encode_horizons((2, 12, 24, 36))
        self.assertEqual(tuple(codes.shape), (4, 12))
        self.assertFalse(torch.allclose(codes[0], codes[-1]))

        future = torch.randn(3, 5, 36)
        last_history = torch.randn(3, 5)
        descriptors = build_future_change_descriptors(
            future, last_history, self.module.anchor_steps
        )
        self.assertEqual(tuple(descriptors.shape), (3, 5, 4, 4))
        self.assertTrue(torch.isfinite(descriptors).all())

    def test_horizon_marginals_and_conditional_transport_are_differentiable(self):
        source_rep = torch.randn(3 * 12, 128, requires_grad=True)
        target_rep = torch.randn(4 * 11, 128, requires_grad=True)
        source_future = torch.randn(3, 12, 36)
        target_future = torch.randn(4, 11, 36)
        source_last = torch.randn(3, 12)
        target_last = torch.randn(4, 11)

        source_desc = build_future_change_descriptors(
            source_future, source_last, self.module.anchor_steps
        ).reshape(-1, 4, 4)
        target_desc = build_future_change_descriptors(
            target_future, target_last, self.module.anchor_steps
        ).reshape(-1, 4, 4)

        source_base = torch.rand(source_rep.shape[0]) + 0.1
        target_base = torch.rand(target_rep.shape[0]) + 0.1
        source_marginals, target_marginals = self.module.horizon_marginals(
            source_base, target_base, source_desc, target_desc
        )
        self.assertEqual(tuple(source_marginals.shape), (4, source_rep.shape[0]))
        self.assertEqual(tuple(target_marginals.shape), (4, target_rep.shape[0]))
        self.assertTrue(torch.allclose(source_marginals.sum(dim=1), torch.ones(4), atol=1e-6))
        self.assertTrue(torch.allclose(target_marginals.sum(dim=1), torch.ones(4), atol=1e-6))

        loss, diagnostics = self.module.compute_alignment_loss(
            source_rep,
            target_rep,
            source_desc,
            target_desc,
            source_marginals,
            target_marginals,
        )
        total = loss + 1e-4 * self.module.smoothness_loss()
        total.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(source_rep.grad.abs().sum()), 0.0)
        self.assertGreater(float(target_rep.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(self.module.history_projection.weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(self.module.film_generator[-1].weight.grad.abs().sum()), 0.0
        )
        self.assertLess(float(diagnostics["max_row_marginal_error"]), 1e-2)
        self.assertLess(float(diagnostics["max_col_marginal_error"]), 1e-2)
        self.assertGreater(float(diagnostics["plan_diversity"]), 0.0)


if __name__ == "__main__":
    unittest.main()
