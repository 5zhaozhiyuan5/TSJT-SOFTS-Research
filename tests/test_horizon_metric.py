import unittest

import torch

from model.Meta_Models.horizon_metric import HorizonConditionedLowRankMetric
from model.Meta_Models.SOFTS.SOFTS_new import Model as SOFTS


class HorizonMetricSmokeTest(unittest.TestCase):
    def test_backbone_rep_to_relation_embedding_and_gradients(self):
        torch.manual_seed(7)
        backbone = SOFTS()
        backbone.eval()
        x = torch.randn(2, 288, 5)
        rep, forecast = backbone(x)

        self.assertEqual(tuple(rep.shape), (2, 5, 128))
        self.assertEqual(tuple(forecast.shape), (2, 5, 36))

        metric = HorizonConditionedLowRankMetric(
            input_dim=128,
            rank=8,
            basis_dim=8,
            anchor_minutes=(10, 60, 120, 180),
            max_minutes=180,
        )
        rep = rep.detach().requires_grad_(True)
        relation = metric(rep)

        self.assertEqual(tuple(relation.shape), (2, 5, 32))
        self.assertTrue(torch.isfinite(relation).all())
        loss = relation.square().mean() + metric.smoothness_loss()
        loss.backward()

        self.assertIsNotNone(rep.grad)
        self.assertGreater(float(rep.grad.abs().sum()), 0.0)
        parameter_grad = sum(
            float(p.grad.abs().sum())
            for p in metric.parameters()
            if p.grad is not None
        )
        self.assertGreater(parameter_grad, 0.0)

    def test_source_target_relation_cost_is_differentiable(self):
        torch.manual_seed(11)
        metric = HorizonConditionedLowRankMetric()
        source_rep = torch.randn(2, 3, 128, requires_grad=True)
        target_rep = torch.randn(2, 4, 128, requires_grad=True)

        source_relation = metric(source_rep).reshape(-1, metric.output_dim)
        target_relation = metric(target_rep).reshape(-1, metric.output_dim)
        source_cost_features = torch.cat(
            [source_rep.reshape(-1, 128), 0.5 * source_relation], dim=-1
        )
        target_cost_features = torch.cat(
            [target_rep.reshape(-1, 128), 0.5 * target_relation], dim=-1
        )
        pairwise_cost = torch.cdist(source_cost_features, target_cost_features)
        loss = pairwise_cost.mean()
        loss.backward()

        self.assertEqual(tuple(pairwise_cost.shape), (6, 8))
        self.assertTrue(torch.isfinite(pairwise_cost).all())
        self.assertGreater(float(source_rep.grad.abs().sum()), 0.0)
        self.assertGreater(float(target_rep.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
