"""Unit checks for F2 + DPM attention-ground-cost module (CPU only).

python -m unittest discover -s tests -p 'test_dpm_pattern_attention_ot.py'
"""
import unittest
import torch

from model.Meta_Models.dpm_pattern_attention_ot import (
    SourceEvolutionPatternBank, build_pattern_attention_cost,
)


class TestDpmPatternAttentionOt(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(217)
        self.bank = SourceEvolutionPatternBank(patterns=3,kmeans_iters=6)
        # Three distinct source-only evolution modes, same across 3 bands.
        base = torch.tensor([
            [[0., 0., 0.]]*3,
            [[4., 0., 0.]]*3,
            [[-4., 0., 0.]]*3,
        ])
        samples = (base.repeat_interleave(30,dim=0) +
                   torch.randn(90,3,3)*0.05)
        self.bank.observe(samples,max_per_city=90)
        self.report = self.bank.fit()
        self.sr = base[[0,0,1,1]].clone()
        self.tr = base[[0,1,0,1]].clone()
        self.q = torch.tensor([
            [0.95]*3,[0.10]*3,[0.75]*3,[0.20]*3
        ])
        self.d = torch.tensor([
            [0.9]*3,[0.2]*3,[0.3]*3,[0.8]*3
        ])
        self.x = torch.randn(4,5).requires_grad_()
        self.y = torch.randn(4,5).requires_grad_()

    def call(self, **kw):
        return build_pattern_attention_cost(
            self.x,self.y,
            self.sr,self.q,self.tr,self.d,
            self.bank,block_size=2,
            **kw
        )

    def test_source_memory_is_built_once_and_frozen(self):
        self.assertTrue(self.bank.fitted)
        self.assertEqual(tuple(self.bank.centers.shape),(3,3,3))
        self.assertEqual(self.report["n_training_patterns"],90)
        self.assertEqual(self.report["n_source_snapshots"],1)
        self.assertEqual(len(self.bank._snapshots),0)
        self.assertTrue(self.bank.report()["source_only"])
        with self.assertRaises(RuntimeError):
            self.bank.observe(self.sr)
        state=self.bank.state_dict()
        self.assertEqual(tuple(state["centers"].shape),(3,3,3))

    def test_attentions_are_soft_distributions_and_detached(self):
        src = self.sr.clone().requires_grad_()
        a=self.bank.attention(src,temperature=0.5)
        self.assertEqual(tuple(a.shape),(4,3,3))
        self.assertFalse(a.requires_grad)
        self.assertTrue(torch.allclose(a.sum(-1),torch.ones(4,3),atol=1e-6))
        self.assertTrue(bool(torch.isfinite(a).all()))

    def test_zero_strength_is_exact_original_f2_path(self):
        cost,stats=self.call(strength=0)
        self.assertIsNone(cost)
        self.assertFalse(stats["active"])
        self.assertEqual(stats["strength"],0.)

    def test_pair_cost_changes_by_shared_pattern(self):
        cost,stats=self.call(strength=.04,temperature=.5,mode="pattern")
        c=cost(self.x.unsqueeze(0),self.y.unsqueeze(0))[0]
        old=torch.cdist(self.x,self.y)
        extra=c-old
        self.assertEqual(tuple(c.shape),(4,4))
        self.assertTrue(bool((extra>=-1e-6).all()))
        # S0 and T0 share pattern; S0 and T1 have disjoint patterns.
        self.assertLess(float(extra[0,0]),float(extra[0,1]))
        self.assertGreater(float(extra.std()),0.)
        self.assertTrue(stats["marginals_unchanged_from_original_f2"])

    def test_skill_need_gates_really_alter_pair_costs(self):
        a,_=self.call(strength=.04,temperature=.5,mode="pattern")
        b,_=self.call(strength=.04,temperature=.5,mode="full")
        x,y=self.x[None],self.y[None]
        c0=a(x,y).detach()
        c1=b(x,y).detach()
        self.assertFalse(torch.allclose(c0,c1))
        self.assertNotEqual(float((c1-c0)[0,0,0]),float((c1-c0)[0,1,2]))

    def test_backbone_still_gets_finite_gradient_from_cost(self):
        cost,_=self.call(strength=.03,temperature=.75,mode="full")
        c=cost(self.x.unsqueeze(0),self.y.unsqueeze(0))
        c.sum().backward()
        self.assertIsNotNone(self.x.grad)
        self.assertIsNotNone(self.y.grad)
        self.assertGreater(float(self.x.grad.abs().sum()),0.)
        self.assertGreater(float(self.y.grad.abs().sum()),0.)
        self.assertTrue(bool(torch.isfinite(self.x.grad).all()))
        self.assertTrue(bool(torch.isfinite(self.y.grad).all()))

    def test_bad_settings_and_missing_bank_fail(self):
        empty=SourceEvolutionPatternBank(patterns=3)
        with self.assertRaises(RuntimeError):
            build_pattern_attention_cost(
                self.x,self.y,self.sr,self.q,self.tr,self.d,
                empty,strength=.1
            )
        with self.assertRaises(ValueError):
            self.call(strength=-1)
        with self.assertRaises(ValueError):
            self.call(strength=.01,mode="unknown")
        with self.assertRaises(ValueError):
            self.call(strength=.01,block_size=0)
        with self.assertRaises(ValueError):
            build_pattern_attention_cost(
                self.x,self.y,self.sr[:3],self.q,self.tr,self.d,
                self.bank,strength=.01
            )


if __name__=="__main__":
    unittest.main()
