"""CPU regression tests for original F2 + dynamic Q/R pattern memory + OT.

Run: python -m unittest discover -s tests -p 'test_dpm_v2.py' -v
"""
import unittest
import torch

from model.Meta_Models.dpm_pattern_attention_ot import (
    SourceEvolutionPatternBank,
    build_dynamic_pattern_attention_cost,
    _diagnostic_sinkhorn,
)


class TestDynamicPatternOT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(301)
        self.bank = SourceEvolutionPatternBank(patterns=4,max_snapshots=5,kmeans_iters=5)
        base = torch.tensor([
            [[-4., 0., .5]]*3,
            [[-1., 0., .5]]*3,
            [[1., 0., .5]]*3,
            [[4., 0., .5]]*3,
        ])
        src=(base.repeat_interleave(15,dim=0)+.07*torch.randn(60,3,3))
        source_q=torch.full((60,3),.2)
        source_q[0:15]=.85
        self.bank.observe(src,quality=source_q,max_per_city=60)
        self.bank.fit()
        self.sr=src[:12].clone()
        self.tr=src[[0,19,21,34,37,38,48,51,55,59]].clone()
        self.q=torch.linspace(.15,.95,12)[:,None].expand(-1,3).contiguous()
        self.d=torch.linspace(.25,.9,10)[:,None].expand(-1,3).contiguous()
        self.x=torch.randn(12,8,requires_grad=True)
        self.y=torch.randn(10,8,requires_grad=True)
        self.alpha=torch.softmax(torch.linspace(-2.,2.,12),dim=0)
        self.beta=torch.full((10,),.1)

    def run_cost(self,mode="v2b",**kwargs):
        args=dict(
            x=self.x,y=self.y,source_response=self.sr,source_skill=self.q,
            target_response=self.tr,target_need=self.d,bank=self.bank,
            alpha=self.alpha,beta=self.beta,strength=.3,
            mode=mode,feedback_source=10,feedback_target=8,
            top_l=4,block_size=3,temperature=.75,
            diagnostics=True,
        )
        args.update(kwargs)
        return build_dynamic_pattern_attention_cost(**args)

    def test_bank_initializes_forecast_accuracy_from_source_only(self):
        self.assertEqual(tuple(self.bank.quality.shape),(3,4))
        self.assertEqual(tuple(self.bank.reliability.shape),(3,4))
        self.assertTrue(bool((self.bank.quality>=0).all()))
        self.assertTrue(bool((self.bank.quality<=1).all()))
        self.assertGreater(float(self.bank.quality.max()-self.bank.quality.min()),.05)
        self.assertTrue(torch.equal(self.bank.reliability,torch.ones_like(self.bank.reliability)))
        self.assertEqual(self.bank.feedback_steps,0)

    def test_v2a_v2b_modify_real_pairwise_cost_in_different_ways(self):
        a,ad,_=self.run_cost(mode="v2a")
        b,bd,_=self.run_cost(mode="v2b")
        v1=torch.cdist(self.x,self.y)
        c1=a(self.x[None],self.y[None])[0]
        c2=b(self.x[None],self.y[None])[0]
        self.assertEqual(tuple(c1.shape),(12,10))
        self.assertEqual(tuple(c2.shape),(12,10))
        self.assertTrue(bool(((c1-v1)>-1e-6).all()))
        self.assertTrue(bool(((c2-v1)>-1e-6).all()))
        self.assertGreater(float((c1-c2).detach().abs().mean()),1e-5)
        self.assertTrue(bd["f2_marginals_unchanged"])
        self.assertIn("subset_sinkhorn",bd)
        self.assertTrue(torch.isfinite(c2).all())

    def test_geomloss_reverse_orientation_is_exact_transpose(self):
        for mode in ("v2a","v2b"):
            cost,_,_=self.run_cost(mode=mode)
            forward=cost(self.x[None],self.y[None])[0]
            backward=cost(self.y[None],self.x[None])[0]
            self.assertTrue(torch.allclose(forward,backward.T,atol=1e-5))

    def test_equal_source_target_count_orientation(self):
        y2=self.y[:10].detach().clone().requires_grad_()
        x2=self.x[:10].detach().clone().requires_grad_()
        cost,_,_=self.run_cost(
            x=x2,y=y2,source_response=self.sr[:10],source_skill=self.q[:10],
            target_response=self.tr,target_need=self.d,
            alpha=torch.full((10,),.1)
        )
        self.assertTrue(torch.allclose(
            cost(x2[None],y2[None])[0],
            cost(y2[None],x2[None])[0].T,atol=1e-5
        ))

    def test_ot_representation_gradient_survives_both_orientations(self):
        cost,_,_=self.run_cost()
        (cost(self.x[None],self.y[None]).sum()+
         cost(self.y[None],self.x[None]).sum()).backward()
        self.assertTrue(bool(torch.isfinite(self.x.grad).all()))
        self.assertTrue(bool(torch.isfinite(self.y.grad).all()))
        self.assertGreater(float(self.x.grad.abs().sum()),0)
        self.assertGreater(float(self.y.grad.abs().sum()),0)

    def test_bank_is_frozen_during_cost_then_updates_once(self):
        q_before=self.bank.quality.clone()
        r_before=self.bank.reliability.clone()
        cost,diag,record=self.run_cost()
        _=cost(self.x[None],self.y[None])
        _=cost(self.y[None],self.x[None])
        self.assertTrue(torch.equal(self.bank.quality,q_before))
        self.assertTrue(torch.equal(self.bank.reliability,r_before))
        self.assertEqual(self.bank.feedback_steps,0)
        stats=self.bank.update_reliability([record,record],momentum=.2)
        self.assertEqual(self.bank.feedback_steps,1)
        self.assertNotEqual(float((self.bank.quality-q_before).abs().sum()),0)
        self.assertLess(float(self.bank.reliability.mean()),float(r_before.mean()))
        self.assertEqual(stats["cities"],2)
        self.assertTrue(bool(torch.isfinite(self.bank.quality).all()))
        self.assertTrue(bool(torch.isfinite(self.bank.reliability).all()))
        self.assertTrue(bool(torch.equal(self.bank.centers,self.bank.state_dict()["centers"])))

    def test_feedback_record_from_each_city_has_qr_masses(self):
        _,_,rec=self.run_cost(diagnostics=False)
        for key in ("q_num","q_den","r_num","r_den"):
            self.assertEqual(tuple(rec[key].shape),(3,4))
            self.assertEqual(rec[key].device.type,"cpu")
            self.assertTrue(bool(torch.isfinite(rec[key]).all()))
        self.assertGreater(float(rec["r_den"].sum()),0)
        self.assertGreater(float(rec["q_den"].sum()),0)
        self.assertEqual(rec["summary"]["source_selected"],10)
        self.assertEqual(rec["summary"]["candidates_per_source"],4)

    def test_skill_mix_and_history_reliability_change_bonus(self):
        # Test that stored memory Q/R (not merely current source q) affects
        # the actual OT cost, while leaving the original F2 marginals alone.
        c1,_,_=self.run_cost(skill_mix=1)
        c2,_,_=self.run_cost(skill_mix=0)
        diff=(c1(self.x[None],self.y[None])-
              c2(self.x[None],self.y[None])).detach()
        self.assertGreater(float(diff.abs().mean()),1e-5)
        self.bank.reliability*=.25
        c3,_,_=self.run_cost(skill_mix=0)
        d=(c2(self.x[None],self.y[None])-
           c3(self.x[None],self.y[None])).detach()
        self.assertGreater(float(d.abs().mean()),1e-5)

    def test_explicit_subset_sinkhorn_plan_has_valid_marginals(self):
        cost=torch.tensor([
            [0.,2.,5.],[1.,0.,2.],[4.,1.,0.]
        ])
        a=torch.tensor([.2,.3,.5])
        b=torch.tensor([.4,.2,.4])
        pi,stats=_diagnostic_sinkhorn(a,b,cost,temperature=.8,iterations=240)
        self.assertTrue(torch.allclose(pi.sum(1),a,atol=1e-3))
        self.assertTrue(torch.allclose(pi.sum(0),b,atol=1e-3))
        self.assertLess(stats["max_marginal_error"],1e-3)

    def test_no_label_leak_or_bad_shape_fails_fast(self):
        with self.assertRaises(ValueError):
            self.run_cost(source_skill=self.q*10)
        with self.assertRaises(ValueError):
            self.run_cost(target_response=self.tr[:9])
        with self.assertRaises(ValueError):
            self.run_cost(top_l=1)
        with self.assertRaises(ValueError):
            self.bank.update_reliability([],momentum=.1)
        with self.assertRaises(ValueError):
            self.bank.update_reliability([{}],momentum=.1)

    def test_with_actual_geomloss_custom_cost_if_available(self):
        try:
            from geomloss import SamplesLoss
        except ImportError:
            self.skipTest("Optional GeomLoss not installed")
        cost,_,_=self.run_cost(mode="v2b",diagnostics=False)
        ot=SamplesLoss(loss="sinkhorn",p=1,backend="tensorized",
                       debias=False,blur=.001,cost=cost)
        loss=ot(self.alpha,self.x,self.beta,self.y)
        self.assertTrue(bool(torch.isfinite(loss).all()))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(self.x.grad).all()))
        self.assertGreater(float(self.x.grad.abs().sum()),0)


if __name__ == "__main__":
    unittest.main()
