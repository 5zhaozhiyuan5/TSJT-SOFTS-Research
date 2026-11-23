"""CPU smoke tests for alternative PMME Stage-I domain classifiers.

Run from repo root:
    python -m unittest discover -s tests -p 'test_domain_probability.py' -v
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model" / "Meta_Models"))
from probability_estimators import DomainProbabilityNet, DomainProbabilityTrainer


class DomainProbabilityTests(unittest.TestCase):
    def test_all_variants_shapes_gradients_and_bounds(self):
        torch.manual_seed(7)
        for variant in DomainProbabilityNet.VALID_VARIANTS:
            with self.subTest(variant=variant):
                model = DomainProbabilityNet(horizon=36, variant=variant)
                x = torch.randn(9, 36)
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (9,))
                probs = torch.sigmoid(logits)
                self.assertTrue(torch.isfinite(probs).all().item())
                self.assertTrue(((probs > 0) & (probs < 1)).all().item())
                logits.square().mean().backward()
                self.assertTrue(any(
                    p.grad is not None and p.grad.abs().sum().item() > 0
                    for p in model.parameters()
                ))

    def test_source_target_probability_orientation(self):
        torch.manual_seed(11)
        trainer = DomainProbabilityTrainer(variant="horizon")
        source = torch.randn(8, 36) - 1
        target = torch.randn(6, 36) + 1
        source_weight, target_weight, diagnostics = trainer.update(source, target)
        self.assertEqual(tuple(source_weight.shape), (8,))
        self.assertEqual(tuple(target_weight.shape), (6,))
        self.assertTrue(((source_weight >= 0) & (source_weight <= 1)).all().item())
        self.assertTrue(((target_weight >= 0) & (target_weight <= 1)).all().item())
        self.assertIn("domain_bce", diagnostics)
        with self.assertRaises(ValueError):
            trainer.model(torch.randn(4, 12))

    def test_classifier_checkpoint_round_trip(self):
        torch.manual_seed(3)
        a = DomainProbabilityNet(variant="dynamics")
        x = torch.randn(5, 36)
        b = DomainProbabilityNet(variant="dynamics")
        b.load_state_dict(a.state_dict())
        self.assertTrue(torch.allclose(a(x), b(x)))


if __name__ == "__main__":
    unittest.main()
