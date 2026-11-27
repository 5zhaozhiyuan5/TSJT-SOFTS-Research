"""CPU tests for actual PMME Stage-I target-need/source-skill matching.

python -m unittest discover -s tests -p 'test_task_need_skill_ot.py'
"""
import unittest
import torch
from model.Meta_Models.task_need_skill_ot import (
    dynamics_and_quality,calibrate_bilateral
)


class TestTrajectorySkill(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(707)
        self.x=torch.randn(3,5,288,2)
        self.y=torch.randn(3,5,36)*3+60
        self.pred=self.y+torch.randn_like(self.y)*.8

    def test_training_quality_is_detached_and_bounded(self):
        prediction=self.pred.detach().requires_grad_()
        sr,q=dynamics_and_quality(
            prediction,self.y,self.x,[60.],[8.],role="source"
        )
        tr,need=dynamics_and_quality(
            prediction+1.,self.y,self.x,[60.],[8.],role="target"
        )
        self.assertEqual(tuple(sr.shape),(15,3,3))
        self.assertEqual(tuple(q.shape),(15,3))
        self.assertEqual(tuple(tr.shape),(15,3,3))
        self.assertEqual(tuple(need.shape),(15,3))
        self.assertFalse(sr.requires_grad)
        self.assertFalse(q.requires_grad)
        self.assertTrue(bool(((q>=0)&(q<=1)).all()))
        self.assertTrue(bool(((need>=0)&(need<=1)).all()))

    def test_perfect_source_beats_corrupted(self):
        exact=self.y.clone()
        bad=self.y+8.
        _,qgood=dynamics_and_quality(exact,self.y,self.x,[60.],[8.])
        _,qbad=dynamics_and_quality(bad,self.y,self.x,[60.],[8.])
        self.assertGreater(float(qgood.mean()),float(qbad.mean()))

    def test_zero_strength_preserves_bitwise_prior(self):
        sr,sq=dynamics_and_quality(
            self.pred,self.y,self.x,[60.],[8.]
        )
        tr,tn=dynamics_and_quality(
            self.pred+1.,self.y,self.x,[60.],[8.],role="target"
        )
        a=torch.softmax(torch.randn(15),dim=0)
        b=torch.ones(15)/15
        aa,bb,meta=calibrate_bilateral(
            a,b,sr,sq,tr,tn,strength=0
        )
        self.assertIs(aa,a)
        self.assertIs(bb,b)
        self.assertEqual(meta["source_l1_shift"],0.)

    def test_both_marginals_shift_but_keep_coverage(self):
        sr,sq=dynamics_and_quality(
            self.pred,self.y,self.x,[60.],[8.]
        )
        tr,tn=dynamics_and_quality(
            self.pred+1.,self.y,self.x,[60.],[8.],role="target"
        )
        a=torch.ones(15)/15
        b=a.clone()
        aa,bb,meta=calibrate_bilateral(
            a,b,sr,sq,tr,tn,strength=.35,prototypes=7
        )
        self.assertAlmostEqual(float(aa.sum()),1.,places=5)
        self.assertAlmostEqual(float(bb.sum()),1.,places=5)
        self.assertTrue(bool((aa>0).all() and (bb>0).all()))
        self.assertGreater(meta["source_l1_shift"],0.)
        self.assertGreater(meta["target_l1_shift"],0.)
        self.assertGreaterEqual(meta["source_relative_min"],.65-1e-4)
        self.assertGreaterEqual(meta["target_relative_min"],.65-1e-4)

    def test_target_aware_source_and_source_aware_target(self):
        # Two source samples each aligned to their matching target descriptor.
        sr=torch.zeros(4,3,3)
        tr=torch.zeros(4,3,3)
        sr[1::2]=3.
        tr[1::2]=3.
        sq=torch.tensor([[1.]*3,[.01]*3,[1.]*3,[.01]*3])
        tn=torch.tensor([[.1]*3,[1.]*3,[.1]*3,[1.]*3])
        a=torch.ones(4)/4
        b=a.clone()
        aa,bb,_=calibrate_bilateral(
            a,b,sr,sq,tr,tn,strength=.6,prototypes=4,temperature=.3
        )
        # A high-quality source unable to match high-demand targets should
        # not be automatically preferred just because it forecasts well.
        self.assertTrue(torch.isfinite(aa).all())
        self.assertTrue(torch.isfinite(bb).all())
        self.assertNotEqual(float(aa[0]),float(aa[1]))
        self.assertNotEqual(float(bb[0]),float(bb[1]))

    def test_all_zero_skill_preserves_prior(self):
        a=torch.softmax(torch.randn(6),dim=0)
        b=torch.softmax(torch.randn(7),dim=0)
        sr=torch.zeros(6,3,3)
        tr=torch.zeros(7,3,3)
        q=torch.zeros(6,3)
        need=torch.ones(7,3)*.9
        aa,bb,_=calibrate_bilateral(
            a,b,sr,q,tr,need,strength=1,prototypes=4
        )
        self.assertTrue(torch.allclose(aa,a))
        self.assertTrue(torch.allclose(bb,b))

    def test_bad_shapes_and_negative_strength(self):
        a=torch.ones(4)/4
        b=torch.ones(4)/4
        x=torch.zeros(4,3,3)
        q=torch.ones(4,3)
        with self.assertRaises(ValueError):
            calibrate_bilateral(a,b,x,q,x,q,strength=-1)
        with self.assertRaises(ValueError):
            calibrate_bilateral(a,b,x[:3],q,x,q)
        with self.assertRaises(ValueError):
            dynamics_and_quality(
                self.pred,self.y[:2],self.x,[60.],[8.]
            )


if __name__=="__main__":
    unittest.main()
