"""CPU unit tests for F2 + shared pattern-value OT."""
import unittest
import torch

from model.Meta_Models.shared_pattern_value_ot import (
    SharedTrafficPatternValueBank,
    build_pattern_value_ot,
    extract_pattern_response,
)


class TestSharedPatternValueOT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)
        self.bank = SharedTrafficPatternValueBank(patterns=3, kmeans_iters=8)
        modes = torch.tensor([
            [0.0, 0.0, 0.1],
            [3.0, 1.0, 0.2],
            [-3.0, -1.0, 0.3],
        ])
        samples = modes.repeat_interleave(30, dim=0).unsqueeze(1).repeat(1, 3, 1)
        samples = samples + 0.03 * torch.randn_like(samples)
        self.bank.observe(samples, max_per_city=90)
        self.bank.fit()

    def test_source_only_shared_bank_and_attention(self):
        self.assertTrue(self.bank.fitted)
        self.assertEqual(tuple(self.bank.centers.shape), (3, 3, 3))
        self.assertTrue(self.bank.report()["source_only"])
        query = torch.randn(12, 3, 3)
        attention = self.bank.attention(query)
        self.assertEqual(tuple(attention.shape), (12, 3, 3))
        self.assertTrue(torch.allclose(
            attention.sum(-1), torch.ones(12, 3), atol=1e-6
        ))
        self.assertFalse(attention.requires_grad)

    def test_value_ema_moves_toward_observed_skill(self):
        attention = self.bank.attention(torch.zeros(6, 3, 3))
        q_num = (attention * 0.9).sum(0)
        q_den = attention.sum(0)
        before = self.bank.values.clone()
        report = self.bank.update_values(
            [{"q_num": q_num, "q_den": q_den},
             {"q_num": q_num, "q_den": q_den}],
            momentum=0.2,
        )
        self.assertGreater(report["step"], 0)
        self.assertTrue(bool((self.bank.values >= before).all()))
        self.assertTrue(bool((self.bank.values <= 1).all()))

    def test_response_skill_and_need_shapes(self):
        prediction = torch.randn(4, 5, 36)
        truth = prediction + 0.2 * torch.randn_like(prediction)
        history = torch.randn(4, 5, 12, 3)
        response, skill, need = extract_pattern_response(
            prediction, truth, history, 50.0, 10.0
        )
        self.assertEqual(tuple(response.shape), (20, 3, 3))
        self.assertEqual(tuple(skill.shape), (20, 3))
        self.assertEqual(tuple(need.shape), (20, 3))
        self.assertTrue(bool(torch.isfinite(response).all()))
        self.assertTrue(bool(((skill >= 0) & (skill <= 1)).all()))
        self.assertTrue(bool(((need >= 0) & (need <= 1)).all()))

    def test_f2_alpha_is_reweighted_and_cost_has_gradients(self):
        ns, nt = 18, 14
        sr = torch.randn(ns, 3, 3)
        tr = torch.randn(nt, 3, 3)
        skill = torch.linspace(0.05, 0.95, ns).unsqueeze(1).repeat(1, 3)
        need = torch.linspace(0.1, 0.9, nt).unsqueeze(1).repeat(1, 3)
        alpha = torch.full((ns,), 1.0 / ns)
        x = torch.randn(ns, 8, requires_grad=True)
        y = torch.randn(nt, 8, requires_grad=True)
        cost, alpha_new, diag, feedback = build_pattern_value_ot(
            x, y, sr, skill, tr, need, alpha, self.bank,
            temperature=0.8, mass_strength=0.5,
            cost_strength=0.1, block_size=5, diagnostics=True,
        )
        self.assertIsNotNone(cost)
        self.assertTrue(torch.allclose(alpha_new.sum(), torch.tensor(1.0), atol=1e-6))
        self.assertFalse(torch.allclose(alpha_new, alpha))
        self.assertGreater(diag["alpha_l1_shift"], 0)
        self.assertIn("q_num", feedback)
        c = cost(x.unsqueeze(0), y.unsqueeze(0))
        self.assertEqual(tuple(c.shape), (1, ns, nt))
        self.assertTrue(bool(torch.isfinite(c).all()))
        c.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(bool(torch.isfinite(y.grad).all()))

    def test_transposed_cost_orientation(self):
        ns, nt = 12, 10
        sr, tr = torch.randn(ns, 3, 3), torch.randn(nt, 3, 3)
        skill = torch.rand(ns, 3)
        need = torch.rand(nt, 3)
        alpha = torch.full((ns,), 1.0 / ns)
        x, y = torch.randn(ns, 4), torch.randn(nt, 4)
        cost, _, _, _ = build_pattern_value_ot(
            x, y, sr, skill, tr, need, alpha, self.bank,
            cost_strength=0.05, block_size=4,
        )
        forward = cost(x.unsqueeze(0), y.unsqueeze(0))
        reverse = cost(y.unsqueeze(0), x.unsqueeze(0))
        self.assertTrue(torch.allclose(forward, reverse.transpose(1, 2), atol=1e-5))

    def test_zero_strength_preserves_f2_mass_and_cost(self):
        ns, nt = 8, 7
        sr, tr = torch.randn(ns, 3, 3), torch.randn(nt, 3, 3)
        skill, need = torch.rand(ns, 3), torch.rand(nt, 3)
        alpha = torch.full((ns,), 1.0 / ns)
        x, y = torch.randn(ns, 4), torch.randn(nt, 4)
        cost, alpha_new, _, _ = build_pattern_value_ot(
            x, y, sr, skill, tr, need, alpha, self.bank,
            mass_strength=0.0, cost_strength=0.0,
        )
        self.assertIsNone(cost)
        self.assertTrue(torch.equal(alpha_new, alpha))


if __name__ == "__main__":
    unittest.main()
