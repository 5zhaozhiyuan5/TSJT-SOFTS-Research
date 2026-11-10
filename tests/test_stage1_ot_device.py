"""Regression tests for the PMME Stage-I no-classifier OT weights branch.

These tests do not import PMME.py because its GP/GeomLoss dependencies are
optional in lightweight CPU environments.
"""
import ast
from pathlib import Path
import unittest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "model" / "Meta_Models" / "PMME.py"


class TestStage1OTWeights(unittest.TestCase):
    def test_cpu_uniform_ot_weights_from_source(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        assignments = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"alpha", "beta"}:
                    text = ast.unparse(node.value)
                    if "torch.ones(" in text and "self.device" in text:
                        assignments[target.id] = node
        self.assertEqual(set(assignments), {"alpha", "beta"})
        scope = {"torch": torch, "size1": 7, "size2": 11,
                 "self": type("Device", (), {"device": torch.device("cpu")})()}
        for key in ("alpha", "beta"):
            stmt = assignments[key]
            exec(compile(ast.Module(body=[stmt], type_ignores=[]), str(SOURCE), "exec"), scope)
            self.assertEqual(scope[key].device.type, "cpu")
            self.assertAlmostEqual(scope[key].sum().item(), 1.0, places=6)

    def test_no_cuda_hardcoding_in_uniform_ot_branch(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("torch.ones(size1, device='cuda')", source)
        self.assertNotIn("torch.ones(size2, device='cuda')", source)


if __name__ == "__main__":
    unittest.main()
