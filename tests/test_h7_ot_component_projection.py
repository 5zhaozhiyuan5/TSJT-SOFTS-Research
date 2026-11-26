"""CPU-only H7 math + virtual update tests; does not import GeomLoss or data.

Run: python -m unittest discover -s tests -p 'test_h7_ot_component_projection.py'
"""
import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Meta_Models.h3_city_counterfactual import masked_support_split
from model.Meta_Models.h7_ot_component_projection import (
    POLICY_NAMES, dot_tuple, project_conflicting_ot, make_h7_policies,
    shadow_h7_audit,
)


class MiniForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Linear(4,8)
        self.output=nn.Linear(8,36)

    def forward(self,data,stage="train"):
        z=torch.tanh(self.body(data.x))
        pred=self.output(z)
        self.RevIn_mean=data.x.mean()
        self.RevIn_std=data.x.std()
        return pred,data.y,z,data.y,None


class ProjectionMathTest(unittest.TestCase):
    def test_negative_conflict_removed_without_erasing_orthogonal_part(self):
        g=[torch.tensor([-2.,3.]),torch.tensor([1.])]
        t=[torch.tensor([1.,0.]),torch.tensor([0.])]
        new,info=project_conflicting_ot(g,t)
        self.assertTrue(info["is_conflicting"])
        self.assertLess(abs(float(new[0][0])),1e-5)
        self.assertAlmostEqual(float(new[0][1]),3.,places=6)
        self.assertAlmostEqual(float(new[1][0]),1.,places=6)
        self.assertGreaterEqual(float(dot_tuple(new,t)), -1e-6)
        self.assertGreater(info["removed_norm"],0)

    def test_positive_and_zero_target_no_surgery(self):
        g=[torch.tensor([2.,3.])]
        t=[torch.tensor([1.,0.])]
        new,info=project_conflicting_ot(g,t)
        self.assertFalse(info["is_conflicting"])
        self.assertTrue(torch.equal(new[0],g[0]))
        no,_=project_conflicting_ot(g,[torch.zeros_like(t[0])])
        self.assertTrue(torch.equal(no[0],g[0]))
        with self.assertRaises(ValueError):
            project_conflicting_ot(g,t,eps=0)
        with self.assertRaises(ValueError):
            dot_tuple(g,[])

    def test_global_and_city_projection_equivalent_if_all_conflict(self):
        t=[torch.tensor([1.,0.])]
        e=[
          {"city":"chengdu_m","f2":[torch.tensor([-1.,2.])]},
          {"city":"metr-la","f2":[torch.tensor([-2.,1.])]},
          {"city":"shenzhen","f2":[torch.tensor([-3.,-1.])]}
        ]
        policies,d=make_h7_policies(e,t)
        self.assertEqual(tuple(policies),POLICY_NAMES)
        self.assertTrue(d["all_cities_conflicting"])
        self.assertLess(d["global_vs_city_projection_difference_norm"],1e-5)
        self.assertTrue(torch.allclose(policies["global_projection"][0],
                                       policies["city_projection"][0],atol=1e-5))
        self.assertTrue(torch.equal(policies["identity"][0],policies["f2"][0]))
        self.assertTrue(torch.allclose(policies["city_projection_half"][0]*2,
                                       policies["city_projection"][0],atol=1e-5))

    def test_global_city_difference_when_some_cities_help(self):
        t=[torch.tensor([1.,0.])]
        entries=[
          {"city":"chengdu_m","f2":[torch.tensor([-2.,0.])]},
          {"city":"metr-la","f2":[torch.tensor([3.,0.])]},
          {"city":"shenzhen","f2":[torch.tensor([0.,1.])]}
        ]
        policies,d=make_h7_policies(entries,t)
        self.assertFalse(d["all_cities_conflicting"])
        self.assertGreater(d["global_vs_city_projection_difference_norm"],1.5)
        self.assertTrue(torch.allclose(policies["global_projection"][0],
                                       torch.tensor([1.,1.]),atol=1e-5))
        self.assertTrue(torch.allclose(policies["city_projection"][0],
                                       torch.tensor([3.,1.]),atol=1e-5))
        self.assertTrue(torch.allclose(policies["city_off:shenzhen"][0],
                                       torch.tensor([1.,0.]),atol=1e-5))

    def test_validation_of_city_names_and_shapes(self):
        t=[torch.ones(2)]
        entries=[{"city":"chengdu_m","f2":[torch.ones(2)]},
                 {"city":"metr-la","f2":[torch.ones(2)]},
                 {"city":"shenzhen","f2":[torch.ones(2)]}]
        with self.assertRaises(ValueError):
            make_h7_policies(entries[:2],t)
        with self.assertRaises(ValueError):
            make_h7_policies(entries[:-1]+[entries[0]],t)
        with self.assertRaises(ValueError):
            make_h7_policies(entries[:-1]+[{"city":"shenzhen","f2":[torch.ones(3)]}],t)

    def test_shadows_leave_parameters_and_optimizer_untouched(self):
        torch.manual_seed(712)
        net=MiniForecast()
        optimizer=torch.optim.AdamW(net.parameters(),lr=0.001,weight_decay=0.01)
        source=SimpleNamespace(x=torch.randn(6,2,4),y=torch.randn(6,2,36))
        target=SimpleNamespace(x=torch.randn(46,2,4),y=torch.randn(46,2,36),
                               means=torch.zeros(1),stds=torch.ones(1))
        # Create a non-empty AdamW momentum/variance history.
        (net(source)[0].square().mean()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        source_pred=net(source)[0]
        target_pred=net(target)[0]
        fit,probe=masked_support_split(46,target.x.device)
        source_loss=F.smooth_l1_loss(source_pred,source.y,beta=5e-4)
        target_fit_loss=F.smooth_l1_loss(target_pred[fit],target.y[fit],beta=5e-4)
        params=list(net.parameters())
        city_entries=[
           {"city":name,"f2":[torch.full_like(p,(i+1)*0.0001) for p in params]}
           for i,name in enumerate(("chengdu_m","metr-la","shenzhen"))
        ]
        before_parameters={n:p.detach().clone() for n,p in net.named_parameters()}
        before_optimizer=copy.deepcopy(optimizer.state_dict())
        before_training={n:m.training for n,m in net.named_modules()}
        self.assertTrue(all(p.grad is None for p in params))
        record=shadow_h7_audit(
            model=net,optimizer=optimizer,source_task_loss=source_loss,
            target_fit_task_loss=target_fit_loss,city_entries=city_entries,
            target_data=target,probe_mask=probe,sample_index=0,
            checkpoint_epoch=2899
        )
        self.assertEqual(record["audit_version"],"h7-v1")
        self.assertEqual(record["n_shadow_candidates"],7)
        self.assertEqual(tuple(record["candidate_scores"]),POLICY_NAMES)
        self.assertEqual(record["training_updates"],0)
        self.assertEqual(record["stage2_runs"],0)
        self.assertEqual(len(record["fit_window_indices"]),36)
        self.assertEqual(len(record["probe_window_indices"]),10)
        self.assertEqual(record["gains_vs_f2"]["f2"]["gain_mae"],0)
        self.assertEqual(record["gains_vs_half"]["global_half"]["gain_mae"],0)
        self.assertLess(abs(record["gains_vs_f2"]["identity"]["gain_mae"]),1e-5)
        self.assertEqual(len(record["candidate_scores"]["city_projection"]["band_smooth_l1"]),3)
        self.assertIn("global_projection",record["optimizer_traces"])
        for n,p in net.named_parameters():
            self.assertTrue(torch.equal(p.detach(),before_parameters[n]))
        self.assertTrue(all(p.grad is None for p in params))
        self.assertEqual(before_training,{n:m.training for n,m in net.named_modules()})
        for k,field in optimizer.state_dict()["state"].items():
            for name,val in field.items():
                original=before_optimizer["state"][k][name]
                if torch.is_tensor(val):
                    self.assertTrue(torch.equal(val,original))
                else:
                    self.assertEqual(val,original)


if __name__=="__main__":
    unittest.main()
