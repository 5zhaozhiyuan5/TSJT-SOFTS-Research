"""CPU-only mathematical regression for single PRC module on original F2.

Run: python -m unittest discover -s tests -p 'test_prediction_reliability.py'
No GeomLoss, GPU or datasets are needed.
"""
import unittest

import torch

from model.Meta_Models.prediction_reliability import (
    source_prediction_quality, calibrate_source_alpha
)


class SourceAccuracyTests(unittest.TestCase):
    def make_example(self):
        # normalized speed history=0 -> true speed last observation = mean
        history=torch.zeros(1,4,6,2)
        truth=torch.full((1,4,36),100.0)
        pred=truth.clone()
        # sample0: correct forecast; naive persistence error=10
        # sample1: wrong forecast; naive persistence error=10
        # sample2: persistence prediction itself, not better than naive
        # sample3: reasonably good forecast
        pred[:,1]=116.0
        pred[:,2]=90.0
        pred[:,3]=102.0
        return pred,truth,history

    def test_accuracy_and_persistence_skill(self):
        pred,truth,history=self.make_example()
        q,s=source_prediction_quality(pred,truth,history,[90.0],[10.0])
        self.assertEqual(q.shape,(1,4))
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue((q>=0).all())
        self.assertTrue((q<=1).all())
        self.assertGreater(q[0,0].item(),q[0,3].item())
        self.assertGreater(q[0,3].item(),q[0,2].item())
        self.assertGreater(q[0,2].item(),q[0,1].item())
        self.assertAlmostEqual(s["source_mae"],7.0,places=4)
        self.assertAlmostEqual(s["persistence_mae"],10.0,places=4)
        self.assertAlmostEqual(s["model_beats_persistence_frac"],0.5,places=4)

    def test_city_affine_scale_invariant(self):
        pred,truth,history=self.make_example()
        q1,_=source_prediction_quality(pred,truth,history,[90.0],[10.0])
        # Same standardized traffic dynamics in a city with x2 values
        pred2=pred*2+10
        truth2=truth*2+10
        q2,_=source_prediction_quality(
            pred2,truth2,history,[190.0],[20.0]
        )
        self.assertTrue(torch.allclose(q1,q2,atol=1e-6))

    def test_zero_strength_is_exact_baseline(self):
        torch.manual_seed(99)
        alpha=torch.softmax(torch.randn(32),dim=0)
        q=torch.rand(32)
        out,s=calibrate_source_alpha(alpha,q,strength=0)
        self.assertTrue(torch.equal(out,alpha))
        self.assertEqual(s["min_multiplier"],1)
        self.assertEqual(s["max_multiplier"],1)

    def test_accuracy_increases_relative_alpha_and_is_bounded(self):
        pred,truth,history=self.make_example()
        q,_=source_prediction_quality(pred,truth,history,[90.0],[10.0])
        alpha=torch.full((4,),0.25)
        out,stats=calibrate_source_alpha(alpha,q,strength=0.5)
        self.assertAlmostEqual(out.sum().item(),1,places=6)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(out[0].item(),out[1].item())
        self.assertGreaterEqual(stats["min_multiplier"],0.5)
        self.assertLessEqual(stats["max_multiplier"],1.5)
        self.assertGreater(stats["ess_after"],0)

    def test_extreme_bad_predictions_no_nan_or_zero_mass(self):
        q=torch.tensor([1.0,1e-12,0,0])
        alpha=torch.tensor([0.25,0.25,0.25,0.25])
        for strength in (0,0.5,1):
            out,_=calibrate_source_alpha(alpha,q,strength)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(out.sum().allclose(torch.tensor(1.0),atol=1e-6))
            self.assertTrue((out>=0).all())

    def test_reject_mismatch_invalid_strength_and_nonfinite(self):
        alpha=torch.full((4,),0.25)
        with self.assertRaises(ValueError):
            calibrate_source_alpha(alpha,torch.rand(3),0.5)
        with self.assertRaises(ValueError):
            calibrate_source_alpha(alpha,torch.rand(4),-0.1)
        with self.assertRaises(ValueError):
            calibrate_source_alpha(alpha,torch.rand(4),float("nan"))
        with self.assertRaises(ValueError):
            calibrate_source_alpha(alpha,torch.tensor([float("nan"),0,0,0]),0.5)
        pred,truth,history=self.make_example()
        with self.assertRaises(ValueError):
            source_prediction_quality(pred,truth,history,[90.0],[0.0])


if __name__=="__main__":
    unittest.main()
