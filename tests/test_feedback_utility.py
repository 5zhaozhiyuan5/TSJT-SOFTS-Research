"""CPU-only G1-G3 counterfactual feedback mathematical smoke tests.

The differentiable toy OT criterion is ONLY for unit testing. The actual
training always uses the unchanged official GeomLoss Sinkhorn operator.
Run:
  python -m unittest discover -s tests -p 'test_feedback_utility.py'
"""
import unittest

import torch

from model.Meta_Models.feedback_utility import (
    _indices, _pcgrad_one_sided, _proxy_virtual_gain,
    blend_source_weights, evaluate_feedback_candidates,
    select_feedback_policy,
)


def toy_ot(alpha, x, beta, y):
    pairwise = (x[:, None, :] - y[None, :, :]).square().sum(dim=-1)
    return (pairwise * alpha[:, None] * beta[None, :]).sum()


class FeedbackUtilityTests(unittest.TestCase):
    def test_weight_blending_and_policy(self):
        f2 = torch.tensor([0.8, 0.1, 0.1])
        self.assertTrue(torch.allclose(
            blend_source_weights(f2, 1.0), f2))
        self.assertTrue(torch.allclose(
            blend_source_weights(f2, 0.0),
            torch.full((3,), 1 / 3)))
        self.assertTrue(torch.allclose(
            blend_source_weights(f2, 0.5),
            0.5 * f2 + torch.full((3,), 1 / 6)))
        diag = [
            {"rho": 0.0, "gain": 0.1, "horizon_score": -0.05},
            {"rho": 0.5, "gain": 0.2, "horizon_score": 0.05},
            {"rho": 1.0, "gain": -0.1, "horizon_score": -0.1},
            {"rho": "off", "gain": 0.0, "horizon_score": 0.0},
        ]
        self.assertEqual(select_feedback_policy(diag, "g2"), 0.5)
        self.assertEqual(select_feedback_policy(diag, "g3"), 0.5)
        bad = [{"rho": 1.0, "gain": -1.0, "horizon_score": -1.0},
               {"rho": "off", "gain": 0.0, "horizon_score": 0.0}]
        self.assertEqual(select_feedback_policy(bad, "g2"), "off")
        self.assertEqual(_indices(12, 5, torch.device("cpu")).numel(), 5)

    def test_exact_readout_virtual_change(self):
        torch.manual_seed(5)
        z = torch.randn(2, 3, 512)
        p2 = torch.randn(36, 128) * 0.01
        grad_task = torch.randn(128, 512) * 0.02
        grad_target = torch.randn(128, 512) * 0.02
        grad_ot = torch.randn(128, 512) * 0.002
        base = torch.randn(2, 3, 36)
        true = torch.randn(2, 3, 36)
        scale = torch.ones(2, 3, 1) * 8.0
        record = _proxy_virtual_gain(
            grad_ot, grad_task, grad_target,
            z, base, true, scale, p2, 0.05, 0.02
        )
        self.assertIsNotNone(record)
        self.assertEqual(len(record["bands"]), 3)
        self.assertTrue(torch.isfinite(torch.tensor(record["gain"])))

    def test_toy_ot_candidates_do_not_touch_model_or_rng(self):
        torch.manual_seed(37)
        n_s, n_t = 18, 12
        source = torch.randn(n_s, 128)
        target = torch.randn(n_t, 128)
        source_y = torch.randn(n_s, 36)
        target_y = torch.randn(n_t, 36)
        input_s = torch.randn(n_s, 512)
        input_t = torch.randn(n_t, 512)
        f2_weights = torch.softmax(torch.randn(n_s), dim=0)
        readout = torch.nn.Linear(128, 36)
        probe = torch.randn(3, 5, 512)
        pred = torch.randn(3, 5, 36)
        truth = torch.randn(3, 5, 36)
        tg = torch.randn(128, 512) * .01
        tt = torch.randn(128, 512) * .01
        rng = torch.get_rng_state().clone()
        original = readout.weight.detach().clone()
        scores = evaluate_feedback_candidates(
            source_rep=source, target_rep=target,
            source_y=source_y, target_y=target_y,
            source_pre=input_s, target_pre=input_t,
            f2_alpha=f2_weights,
            target_probe_pre=probe,
            target_probe_output=pred,
            target_probe_y=truth,
            target_probe_scale=torch.ones(3, 5, 1),
            projection2_weight=readout.weight.detach(),
            task_head_grad=tg, target_head_grad=tt,
            criterion=toy_ot, ot_coefficient=0.001,
            xy_scale=10.0, n_source=8, n_target=6
        )
        self.assertEqual([r["rho"] for r in scores], [0.0, 0.5, 1.0, "off"])
        self.assertTrue(all(torch.isfinite(torch.tensor(r["gain"]))
                            for r in scores))
        self.assertTrue(torch.equal(readout.weight.detach(), original))
        self.assertIsNone(readout.weight.grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertIn(select_feedback_policy(scores, "g2"),
                      (0.0, 0.5, 1.0, "off"))


if __name__ == "__main__":
    unittest.main()
