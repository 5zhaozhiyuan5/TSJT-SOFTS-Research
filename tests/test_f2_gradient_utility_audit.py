"""CPU tests for F2 D1/D2 probe-excluded, non-mutating counterfactuals.

The toy criterion is differentiable weighted distance, NOT GeomLoss OT.
Tests validate instrumentation only. Real GeomLoss runs must be smoke-tested
on the user's training server.
"""
import unittest
import torch
import torch.nn as nn

from model.Meta_Models.f2_gradient_utility_audit import (
    probe_window_masks, source_gradient_scores, _selected_weights,
    audit_f2_source_utility
)


class ToyWeightedDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.records = []

    def forward(self, a, x, b, y):
        self.records.append((x.detach().clone(), y.detach().clone()))
        cost = torch.cdist(x, y)
        return ((a[:, None] * b[None, :]) * cost).sum()


class GradientAuditTests(unittest.TestCase):
    def test_fit_probe_masks_are_window_disjoint(self):
        for n in (15, 16, 46):
            fit, probe = probe_window_masks(n, 5)
            self.assertFalse(torch.any(fit & probe))
            self.assertTrue(torch.all(fit | probe))
            self.assertGreater(int(probe.sum()), 0)
            self.assertGreater(int(fit.sum()), 0)
        fit, probe = probe_window_masks(15)
        self.assertEqual(probe.nonzero().flatten().tolist(), [0, 5, 10])

    def test_sign_score_extremes_and_permutation(self):
        fs = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.]]])
        ds = torch.ones(1, 3, 36)
        ft = torch.tensor([[[1., 0.]]])
        dt = torch.ones(1, 1, 36)
        cosine, conflict = source_gradient_scores(fs, ds, ft, dt)
        self.assertAlmostEqual(float(cosine[0]), 1, places=5)
        self.assertAlmostEqual(float(cosine[2]), -1, places=5)
        self.assertAlmostEqual(float(conflict[0]), 0, places=5)
        self.assertAlmostEqual(float(conflict[2]), 1, places=5)
        self.assertTrue(torch.isfinite(conflict).all())
        co2, si2 = source_gradient_scores(fs[:, [2, 0, 1]], ds[:, [2, 0, 1]], ft, dt)
        self.assertTrue(torch.allclose(co2, cosine[[2, 0, 1]]))
        self.assertTrue(torch.allclose(si2, conflict[[2, 0, 1]]))

    def test_matched_sign_groups_are_cosine_stratified(self):
        c = torch.linspace(-0.8, 0.9, 96)
        conflict = torch.rand(96)
        weights, selections = _selected_weights(c, conflict, 1.5, 8.)
        for name, w in weights.items():
            self.assertAlmostEqual(float(w.sum()), 1., places=5, msg=name)
            self.assertTrue(torch.all(w >= 0), msg=name)
        self.assertEqual(
            len(selections["sign_low_cosmatched"]),
            len(selections["sign_high_cosmatched"])
        )
        self.assertLess(
            abs(float(c[selections["sign_low_cosmatched"]].mean() -
                      c[selections["sign_high_cosmatched"]].mean())),
            0.15
        )

    def test_readout_ot_gradients_have_no_parameter_or_grad_mutation(self):
        torch.manual_seed(9)
        head = nn.Linear(6, 128)
        fs_input = torch.randn(25, 6)
        ft_input = torch.randn(21, 6)
        src = head(fs_input).reshape(1, 25, 128)
        tgt = head(ft_input).reshape(1, 21, 128)
        ds = torch.randn(1, 25, 36)
        dt = torch.randn(1, 21, 36)
        source_future = torch.randn(25, 36)
        target_future = torch.randn(21, 36)
        # A separate probe graph whose labels do NOT enter candidate OT.
        probe_prediction = head(torch.randn(5, 6)).sum(dim=1)
        probe_targets = torch.randn(5)
        probe = ((probe_prediction - probe_targets)**2).mean()
        probe_grad = torch.autograd.grad(probe, head.weight, retain_graph=True)[0]
        old_weight = head.weight.detach().clone()
        head.weight.grad = torch.ones_like(head.weight) * 3.0
        old_grad = head.weight.grad.detach().clone()
        criterion = ToyWeightedDistance()
        out = audit_f2_source_utility(
            source_rep=src.reshape(-1,128),
            source_future=source_future,
            source_features=src.detach(),
            source_delta=ds,
            target_fit_rep=tgt.reshape(-1,128),
            target_fit_future=target_future,
            target_fit_features=tgt.detach(),
            target_fit_delta=dt,
            probe_gradient=probe_grad,
            shared_parameter=head.weight,
            criterion=criterion,
            ot_scale=0.01,
            xy_scale=10.,
            epoch=1500,
            city="toy",
            target_group=0,
            source_budget=24,
            target_budget=20,
        )
        self.assertEqual(len(out["candidate_OT"]), 7)
        self.assertEqual(len(criterion.records), 7)
        self.assertEqual(out["source_subset"], 24)
        for x, y in criterion.records[1:]:
            self.assertTrue(torch.equal(x, criterion.records[0][0]))
            self.assertTrue(torch.equal(y, criterion.records[0][1]))
        self.assertTrue(torch.equal(head.weight.detach(), old_weight))
        self.assertTrue(torch.equal(head.weight.grad, old_grad))
        self.assertIn("D1_cos_high_minus_low_gain", out["contrasts"])
        self.assertIn("D2_sign_minus_f2_gain", out["contrasts"])

    def test_probe_gradient_zero_does_not_affect_candidate_scores(self):
        torch.manual_seed(3)
        fs = torch.randn(1, 20, 128)
        ds = torch.randn(1, 20, 36)
        ft = torch.randn(1, 12, 128)
        dt = torch.randn(1, 12, 36)
        c1, d1 = source_gradient_scores(fs,ds,ft,dt)
        c2, d2 = source_gradient_scores(fs,ds,ft,dt)
        self.assertTrue(torch.equal(c1, c2))
        self.assertTrue(torch.equal(d1, d2))


if __name__ == "__main__":
    unittest.main()
