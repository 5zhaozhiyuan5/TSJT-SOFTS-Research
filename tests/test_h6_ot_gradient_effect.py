"""CPU-only H6 tests: policies, PCGrad/AdamW diagnostics and invariance.

python -m unittest discover -s tests -p 'test_h6_ot_gradient_effect.py'
No GeomLoss, data files or GPU are required for this file.
"""
import copy
import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Meta_Models.h3_city_counterfactual import masked_support_split
from model.Meta_Models.h6_ot_gradient_effect import (
    SCALES,gradient_tuple_scaled,gradient_tuple_difference,gradient_norm,
    make_h6_policies,shadow_h6_audit
)


class MiniForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Linear(4,8)
        self.out=nn.Linear(8,36)

    def forward(self,data,stage="train"):
        z=torch.tanh(self.body(data.x))
        pred=self.out(z)
        self.RevIn_mean=data.x.mean()
        self.RevIn_std=data.x.std()
        return pred,data.y,z,data.y,None


def fake_gradient(model,factor):
    return [torch.full_like(p,factor) for p in model.parameters()]


class TestH6Policies(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        self.model=MiniForecast()
        self.entries=[
            {"city":"chengdu_m","f2":fake_gradient(self.model,0.01)},
            {"city":"metr-la","f2":fake_gradient(self.model,-0.005)},
            {"city":"shenzhen","f2":fake_gradient(self.model,0.02)}
        ]

    def test_strength_grid_and_city_interventions(self):
        policies=make_h6_policies(self.entries)
        self.assertEqual(len(policies),11)
        self.assertEqual(set(policies),{
            "f2","all_off","global_quarter","global_half","identity",
            "city_off:chengdu_m","city_off:metr-la","city_off:shenzhen",
            "city_half:chengdu_m","city_half:metr-la","city_half:shenzhen"
        })
        self.assertEqual(SCALES,(0.,.25,.5,1.))
        for x,z in zip(policies["f2"],policies["identity"]):
            self.assertTrue(torch.equal(x,z))
        for x in policies["all_off"]:
            self.assertTrue(torch.equal(x,torch.zeros_like(x)))
        for full,half,quarter in zip(
            policies["f2"],policies["global_half"],
            policies["global_quarter"]
        ):
            self.assertTrue(torch.allclose(full,half*2,atol=1e-6))
            self.assertTrue(torch.allclose(full,quarter*4,atol=1e-6))
        for e in self.entries:
            city=e["city"]
            for f,off,half,grad in zip(
                policies["f2"],policies["city_off:"+city],
                policies["city_half:"+city],e["f2"]
            ):
                self.assertTrue(torch.allclose(f-off,grad,atol=1e-6))
                self.assertTrue(torch.allclose(f-half,grad*.5,atol=1e-6))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_invalid_cities_rejected(self):
        with self.assertRaises(ValueError):
            make_h6_policies(self.entries[:2])
        dup=copy.deepcopy(self.entries)
        dup[-1]["city"]="metr-la"
        with self.assertRaises(ValueError):
            make_h6_policies(dup)
        with self.assertRaises(ValueError):
            make_h6_policies([{"city":"a","f2":[torch.ones(1)]},
                              {"city":"b","f2":[]},
                              {"city":"c","f2":[torch.ones(1)]}])

    def test_gradient_tuple_validation(self):
        a=[torch.tensor([1.,2.])]
        self.assertTrue(torch.equal(
            gradient_tuple_scaled(a,.5)[0],torch.tensor([.5,1.])))
        self.assertAlmostEqual(gradient_norm(a),math.sqrt(5),places=6)
        with self.assertRaises(ValueError):
            gradient_tuple_scaled(a,float("nan"))
        with self.assertRaises(ValueError):
            gradient_tuple_difference(a,[])

    def test_shadow_full_model_invariance_and_identity_noise(self):
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=.001,weight_decay=.01)
        target=SimpleNamespace(x=torch.randn(46,3,4),
                               y=torch.randn(46,3,36),
                               means=torch.zeros(1),stds=torch.ones(1))
        source=SimpleNamespace(x=torch.randn(5,3,4),
                               y=torch.randn(5,3,36))
        # Populate optimizer state as in a real loaded training checkpoint.
        pred=self.model(source)[0]
        loss=pred.square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        fit,probe=masked_support_split(46,torch.device("cpu"))
        src_pred=self.model(source)[0]
        target_pred=self.model(target)[0]
        src_loss=F.smooth_l1_loss(src_pred,source.y,beta=5e-4)
        fit_loss=F.smooth_l1_loss(
            target_pred[fit],target.y[fit],beta=5e-4
        )
        before_params={n:p.detach().clone() for n,p in self.model.named_parameters()}
        before_optimizer=copy.deepcopy(optimizer.state_dict())
        before_buffers={n:x.detach().clone() for n,x in self.model.named_buffers()}
        initial_training={n:m.training for n,m in self.model.named_modules()}
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        result=shadow_h6_audit(
            model=self.model,optimizer=optimizer,
            source_task_loss=src_loss,target_fit_task_loss=fit_loss,
            city_entries=self.entries,target_data=target,
            probe_mask=probe,sample_index=0,checkpoint_epoch=2899
        )
        self.assertEqual(result["audit_version"],"h6-v1")
        self.assertEqual(result["n_shadow_candidates"],11)
        self.assertEqual(len(result["candidate_scores"]),11)
        self.assertEqual(len(result["gradient_report"]["city"]),3)
        self.assertEqual(result["training_updates"],0)
        self.assertEqual(result["stage2_runs"],0)
        self.assertEqual(result["gains"]["f2"]["gain_mae"],0.0)
        self.assertLess(abs(result["gains"]["identity"]["gain_mae"]),1e-5)
        self.assertEqual(len(result["fit_window_indices"]),36)
        self.assertEqual(len(result["probe_window_indices"]),10)
        self.assertEqual(len(result["candidate_scores"]["f2"]["band_smooth_l1"]),3)
        self.assertIn("f2_vs_no_ot",result["step_diagnostics"])
        for n,p in self.model.named_parameters():
            self.assertTrue(torch.equal(p.detach(),before_params[n]))
        for n,x in self.model.named_buffers():
            self.assertTrue(torch.equal(x.detach(),before_buffers[n]))
        self.assertEqual(initial_training,
                         {n:m.training for n,m in self.model.named_modules()})
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        for key,value in optimizer.state_dict()["state"].items():
            for name,tensor in value.items():
                old=before_optimizer["state"][key][name]
                if torch.is_tensor(tensor):
                    self.assertTrue(torch.equal(tensor,old))
                else:
                    self.assertEqual(tensor,old)


if __name__=="__main__":
    unittest.main()
