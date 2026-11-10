"""CPU-only regression tests for read-only Stage-I gradient diagnostics."""
import math
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from stage1_gradient_audit import audit_stage1_gradients


class TestAudit(unittest.TestCase):
    def test_masking_and_unchanged_grads(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        unused = torch.nn.Parameter(torch.tensor([3.]))
        p.grad = torch.tensor([7., 8.])
        d = audit_stage1_gradients(p[0], p[1], -p[0], [p, unused])
        self.assertAlmostEqual(d['source_target_cos'], -1.)
        self.assertAlmostEqual(d['ot_target_cos'], 0.)
        self.assertAlmostEqual(d['combined_target_cos'], -1/math.sqrt(2))
        self.assertTrue(torch.equal(p.grad, torch.tensor([7., 8.])))
        self.assertIsNone(unused.grad)

    def test_zero_alignment_and_no_grad_target(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        d = audit_stage1_gradients(p[0], None, p[1], [p])
        self.assertEqual(d['ot_norm'], 0.)
        self.assertTrue(math.isnan(d['ot_target_cos']))

    def test_empty_parameters_rejected(self):
        with self.assertRaises(ValueError):
            audit_stage1_gradients(None, None, None, [])


if __name__ == '__main__':
    unittest.main()
