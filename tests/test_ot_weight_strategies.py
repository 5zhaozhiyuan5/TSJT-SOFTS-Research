"""Smoke tests for OT weighting ablation; no GP or dataset dependency."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model" / "Meta_Models"))
from ot_weight_strategies import build_ot_weights


class OTWeightTests(unittest.TestCase):
    def test_legacy_exact_weights_and_scale(self):
        s = torch.tensor([0.1, 0.4, 0.8])
        t = torch.tensor([0.9, 0.2, 0.5])
        a,b,scale,active,diag = build_ot_weights(s,t)
        self.assertTrue(active)
        self.assertTrue(torch.allclose(a,torch.tensor([0.,1./3.,2./3.])))
        self.assertTrue(torch.allclose(b,torch.tensor([9./14.,0.,5./14.])))
        self.assertAlmostEqual(float(scale), float(((.4+.8)/3*(.9+.5)/3)**.5), places=6)
        self.assertAlmostEqual(float(a.sum()),1.0,places=6)
        self.assertAlmostEqual(float(b.sum()),1.0,places=6)

    def test_legacy_empty_mass_disables_ot(self):
        a,b,scale,active,d = build_ot_weights(
            torch.tensor([.01,.02]),torch.tensor([.05,.08]),threshold=.3
        )
        self.assertFalse(active)
        self.assertEqual(d["source_ess"],0.0)
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.isfinite(b).all())
        self.assertEqual(float(scale),0.)

    def test_uniform_fixed_preserves_full_mass(self):
        a,b,scale,active,d = build_ot_weights(
            torch.tensor([.0001,.0002]),torch.tensor([.0001,.0002,.0003]),
            strategy="uniform",scale_strategy="fixed"
        )
        self.assertTrue(active)
        self.assertTrue(torch.allclose(a,torch.full((2,),.5)))
        self.assertTrue(torch.allclose(b,torch.full((3,),1/3)))
        self.assertEqual(float(scale),1.0)
        self.assertEqual(d["source_ess_fraction"],1.0)

    def test_smooth_prevents_zero_mass_and_has_grad_free_weights(self):
        ps=torch.tensor([1e-10,.0002,.003],requires_grad=True)
        pt=torch.tensor([.0001,.01],requires_grad=True)
        a,b,scale,active,d=build_ot_weights(
            ps,pt,strategy="smooth",scale_strategy="fixed",floor=.05
        )
        self.assertTrue(active)
        self.assertTrue(torch.all(a>0))
        self.assertTrue(torch.all(b>0))
        self.assertFalse(a.requires_grad)
        self.assertFalse(b.requires_grad)
        self.assertEqual(float(scale),1.)
        self.assertGreater(d["source_ess"],1.0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            build_ot_weights(torch.tensor([float("nan")]),torch.tensor([.1]))
        with self.assertRaises(ValueError):
            build_ot_weights(torch.tensor([.1]),torch.tensor([.1]),power=0)
        with self.assertRaises(ValueError):
            build_ot_weights(torch.tensor([.1]),torch.tensor([.1]),strategy="unknown")


if __name__ == "__main__":
    unittest.main()
