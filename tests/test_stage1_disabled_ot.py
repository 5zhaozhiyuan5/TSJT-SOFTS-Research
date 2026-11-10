"""CPU-only checks for disabled OT in read-only Stage-I gradient audit."""
import math
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from stage1_gradient_audit import audit_stage1_gradients


class TestDisabledOT(unittest.TestCase):
    def test_disabled_ot_variants_preserve_backward(self):
        for disabled_ot in (None, 0, 0.0, torch.tensor(0.)):
            with self.subTest(disabled_ot=disabled_ot):
                p = torch.nn.Parameter(torch.tensor([1., 2.]))
                p.grad = torch.tensor([7., 8.])
                result = audit_stage1_gradients(p[0]**2, disabled_ot, p[1]**2, [p])
                self.assertEqual(result['ot_norm'], 0.)
                self.assertTrue(math.isnan(result['ot_target_cos']))
                self.assertTrue(torch.equal(p.grad, torch.tensor([7., 8.])))
                (p[0]**2).backward()
                self.assertTrue(torch.equal(p.grad, torch.tensor([9., 8.])))

    def test_invalid_untracked_nonzero_loss_rejected(self):
        for loss in (1.0, torch.tensor(1.)):
            with self.subTest(loss=loss):
                p = torch.nn.Parameter(torch.tensor([1., 2.]))
                with self.assertRaises((TypeError, ValueError)):
                    audit_stage1_gradients(p[0]**2, loss, p[1]**2, [p])

    def test_nonscalar_loss_rejected(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        with self.assertRaisesRegex(ValueError, 'scalar'):
            audit_stage1_gradients(p**2, None, p[1]**2, [p])


if __name__ == '__main__':
    unittest.main()
