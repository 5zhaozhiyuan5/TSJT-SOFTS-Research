"""CPU-only H10 update source and official PCGrad parity guardrails.

python -m unittest discover -s tests -p 'test_h10_update_provenance.py'
No PEMS traffic arrays, GeomLoss, GPU or original checkpoint required.
"""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from model.Meta_Models.h1_shadow_audit import projected_directions
from model.Meta_Models.h10_update_provenance import (
    POLICIES,UPDATE_POLICIES,h10_direction,
    apply_directions,clone_official_wrapper,snapshot_tensors,
    compare_snapshots
)


class MiniForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.forecasting_model=nn.Module()
        self.forecasting_model.projection1=nn.Linear(2,2)
        self.forecasting_model.projection2=nn.Linear(2,2)


def tiny_bank():
    return {
        "source_gradient":[torch.tensor([-1.,1.])],
        "target_fit_gradient":[torch.tensor([2.,0.])],
        "city_entries":[
            {"city":"chengdu_m","f2":[torch.tensor([-1.,0.])]},
            {"city":"metr-la","f2":[torch.tensor([-1.,0.])]},
            {"city":"shenzhen","f2":[torch.tensor([0.,1.])]},
        ],
        "source_present_mask":[True],
        "target_present_mask":[True],
        "ot_present_mask":[True],
    }


class TestH10GradientDirections(unittest.TestCase):
    def test_policy_coverage(self):
        self.assertEqual(len(POLICIES),9)
        self.assertEqual(len(UPDATE_POLICIES),7)
        self.assertEqual(POLICIES[0],"t0_no_update")
        self.assertEqual(POLICIES[-1],"t7_project_ot_half")

    def test_official_projection_equals_onesided_pcgrad(self):
        bank=tiny_bank()
        raw=sum((x["f2"][0] for x in bank["city_entries"]))
        g_s=bank["source_gradient"][0]
        t=bank["target_fit_gradient"][0]
        expected=projected_directions(
            [g_s],[t],[raw]
        )[1][0]
        actual,mask,trace=h10_direction(bank,"t5_official_f2")
        self.assertTrue(torch.allclose(actual[0],expected))
        self.assertEqual(mask,[True])
        self.assertLess(trace["ot_vs_target"]["cos"],0.)

    def test_t6_is_direct_update_without_pcgrad(self):
        bank=tiny_bank()
        with_pcgrad,_,_=h10_direction(bank,"t5_official_f2")
        without,mask,_=h10_direction(bank,"t6_full_pcgrad")
        raw=sum((x["f2"][0] for x in bank["city_entries"]))
        self.assertTrue(torch.equal(without[0],bank["source_gradient"][0]+raw))
        self.assertFalse(torch.equal(with_pcgrad[0],without[0]))
        self.assertEqual(mask,[True])

    def test_no_ot_projection_is_exact_t4(self):
        bank=tiny_bank()
        src_only,_,_=h10_direction(bank,"t2_source_only")
        src_pc,mask,_=h10_direction(bank,"t4_source_pcgrad")
        t=bank["target_fit_gradient"]
        expectation=projected_directions(
            bank["source_gradient"],t,
            [torch.zeros_like(x) for x in t]
        )[0]
        self.assertTrue(torch.equal(src_only[0],bank["source_gradient"][0]))
        self.assertTrue(torch.allclose(src_pc[0],expectation[0]))
        self.assertEqual(mask,[True])

    def test_t7_ot_projection_then_whole_gradient_pcgrad(self):
        bank=tiny_bank()
        projected,mask,trace=h10_direction(bank,"t7_project_ot_half")
        raw=sum((x["f2"][0] for x in bank["city_entries"]))
        t=bank["target_fit_gradient"][0]
        pc=raw-(min(0,float(raw.dot(t)))/(float(t.dot(t))+1e-8))*t
        expected=projected_directions(
            bank["source_gradient"],bank["target_fit_gradient"],
            [pc*.5]
        )[1]
        self.assertTrue(torch.allclose(projected[0],expected[0]))
        self.assertEqual(mask,[True])
        self.assertGreaterEqual(trace["update_grad_norm"],0.)

    def test_invalid_policy_fails(self):
        with self.assertRaises(ValueError):
            h10_direction(tiny_bank(),"f2")
        with self.assertRaises(ValueError):
            h10_direction(tiny_bank(),"t0_no_update")


class TestH10CloneAndParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(811)
        self.model=MiniForecaster()
        self.optim=torch.optim.AdamW(
            self.model.parameters(),lr=.01,weight_decay=.01
        )
        for p in self.model.parameters():
            p.grad=torch.randn_like(p)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

    def test_adamw_saved_moments_move_even_with_zero_grads(self):
        source=snapshot_tensors(self.model,self.optim)
        zeros=[torch.zeros_like(p) for p in self.model.parameters()]
        delta=apply_directions(
            self.model,self.optim,zeros,[True]*len(zeros)
        )
        self.assertGreater(delta,0.)
        target=snapshot_tensors(self.model,self.optim)
        comparison=compare_snapshots(source,target)
        self.assertFalse(comparison["passed"])

    def test_adamw_none_grad_does_not_change_frozen_param(self):
        before=[p.detach().clone() for p in self.model.parameters()]
        directions=[torch.ones_like(p) for p in self.model.parameters()]
        mask=[False]+[True]*(len(before)-1)
        apply_directions(self.model,self.optim,directions,mask)
        self.assertTrue(torch.equal(before[0],list(self.model.parameters())[0]))
        self.assertTrue(any(
            not torch.equal(old,p) for old,p in zip(
                before[1:],list(self.model.parameters())[1:]
            )
        ))

    def test_parity_snapshot_pass_and_fails_on_corruption(self):
        snap=snapshot_tensors(self.model,self.optim)
        twin=copy.deepcopy(self.model)
        another=torch.optim.AdamW(twin.parameters(),lr=.01,weight_decay=.01)
        another.load_state_dict(copy.deepcopy(self.optim.state_dict()))
        clone=snapshot_tensors(twin,another)
        self.assertTrue(compare_snapshots(snap,clone)["passed"])
        with torch.no_grad():
            next(twin.parameters()).add_(.1)
        self.assertFalse(compare_snapshots(
            snap,snapshot_tensors(twin,another)
        )["passed"])

    def test_official_wrapper_closure_rebinding_is_clone_local(self):
        wrapper=SimpleNamespace()
        wrapper.model=MiniForecaster()
        wrapper.main_optim=torch.optim.AdamW(wrapper.model.parameters())
        wrapper.projection_args=1.
        wrapper._capture_transfer=False
        wrapper._capture_feedback=False
        wrapper._transfer_records=[]
        def old(_m,ins,out):
            if wrapper._capture_transfer:
                wrapper._transfer_records.append((ins[0],out))
        wrapper._transfer_hook=(
            wrapper.model.forecasting_model.projection2.register_forward_hook(old)
        )
        try:
            clone=clone_official_wrapper(wrapper)
            clone._capture_transfer=True
            inputs=torch.ones(1,2)
            clone.model.forecasting_model.projection2(inputs)
            self.assertEqual(len(clone._transfer_records),1)
            self.assertEqual(len(wrapper._transfer_records),0)
            self.assertEqual(len(
                wrapper.model.forecasting_model.projection2._forward_hooks
            ),1)
        finally:
            wrapper._transfer_hook.remove()

    def test_failed_author_flag_refused(self):
        wrapper=SimpleNamespace()
        wrapper.model=MiniForecaster()
        wrapper.main_optim=torch.optim.AdamW(wrapper.model.parameters())
        wrapper.projection_args=0.
        wrapper._capture_transfer=False
        wrapper._transfer_records=[]
        with self.assertRaises(RuntimeError):
            clone_official_wrapper(wrapper)


if __name__=="__main__":
    unittest.main()
