"""Regression: Stage-I audit must match the original combined backward gradient.

No dataset labels or optimizer updates are used. This guards diagnostic validity,
not an algorithmic improvement or a real traffic forecast result.
"""
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from stage1_gradient_audit import audit_stage1_gradients


class TestJointGradientIdentity(unittest.TestCase):
    def test_joint_backward_matches_audited_decomposition(self):
        p = torch.nn.Parameter(torch.tensor([0.8, -0.3], dtype=torch.float64))
        source = (p[0] - 2 * p[1]).square()
        alignment = 0.25 * (p[0] + p[1]).square()
        target = (p[0] + 0.5 * p[1]).square()
        before = p.detach().clone()
        d = audit_stage1_gradients(source, alignment, target, [p])
        self.assertIsNone(p.grad)
        self.assertTrue(torch.equal(p.detach(), before))
        joint = torch.autograd.grad(source + alignment, p, retain_graph=True)[0]
        target_grad = torch.autograd.grad(target, p)[0]
        expected_cos = torch.dot(joint, target_grad) / (joint.norm() * target_grad.norm())
        self.assertAlmostEqual(d['combined_norm'], joint.norm().item(), places=10)
        self.assertAlmostEqual(d['combined_target_cos'], expected_cos.item(), places=10)

    def test_opposed_source_and_ot_can_hide_component_conflict(self):
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        source = -2 * p[0] + p[1]
        alignment = 3 * p[0]
        target = p[0]
        d = audit_stage1_gradients(source, alignment, target, [p])
        self.assertLess(d['source_target_cos'], 0)
        self.assertGreater(d['ot_target_cos'], 0)
        self.assertGreater(d['combined_target_cos'], 0)
        self.assertAlmostEqual(d['source_ot_dot'], -6.0)

    def test_disabled_alignment_matches_source_gradient(self):
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        source, target = p[0] * p[1], p[0].square()
        d = audit_stage1_gradients(source, torch.tensor(0.), target, [p])
        self.assertAlmostEqual(d['combined_norm'], d['source_norm'])
        self.assertAlmostEqual(d['combined_target_cos'], d['source_target_cos'])
        self.assertTrue(math.isnan(d['ot_target_cos']))


if __name__ == '__main__':
    unittest.main()
