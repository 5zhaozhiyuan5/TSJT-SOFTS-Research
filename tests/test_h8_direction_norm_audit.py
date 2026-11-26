"""H8 CPU tests: directional-vs-magnitude controls, AdamW calibration, safety.

Run: python -m unittest discover -s tests -p 'test_h8_direction_norm_audit.py'
No traffic dataset / GeomLoss / GPU required.
"""
import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Meta_Models.h3_city_counterfactual import masked_support_split
from model.Meta_Models.h8_direction_norm_audit import (
    POLICIES, build_h8_policies, calibrate_adamw_step_norm,
    shadow_h8_audit, gradient_norm,
)


class TinyForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=nn.Linear(4,8)
        self.b=nn.Linear(8,36)

    def forward(self,data,stage="train"):
        z=torch.tanh(self.a(data.x))
        y=self.b(z)
        self.RevIn_mean=data.x.mean()
        self.RevIn_std=data.x.std()
        return y,data.y,z,data.y,None


class H8DirectionMagnitudeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(825)
        self.net=TinyForecast()
        self.opt=torch.optim.AdamW(
            self.net.parameters(),lr=.001,weight_decay=.01
        )
        self.src=SimpleNamespace(
            x=torch.randn(6,3,4),y=torch.randn(6,3,36)
        )
        self.target=SimpleNamespace(
            x=torch.randn(46,3,4),y=torch.randn(46,3,36),
            means=torch.zeros(1),stds=torch.ones(1)
        )
        (self.net(self.src)[0].square().mean()).backward()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.source_loss=F.smooth_l1_loss(
            self.net(self.src)[0],self.src.y,beta=5e-4
        )
        pred=self.net(self.target)[0]
        self.fit,self.probe=masked_support_split(46,pred.device)
        self.target_fit_loss=F.smooth_l1_loss(
            pred[self.fit],self.target.y[self.fit],beta=5e-4
        )
        params=list(self.net.parameters())
        t=torch.autograd.grad(
            self.target_fit_loss,params,retain_graph=True
        )
        gen=torch.Generator().manual_seed(51)
        self.source_cities=[]
        for i,city in enumerate(("chengdu_m","metr-la","shenzhen")):
            raw=[
                -(0.01+i*.002)*g.detach()+
                torch.randn(x.shape,generator=gen)*.0002
                for x,g in zip(params,t)
            ]
            self.source_cities.append({"city":city,"f2":raw})

    def test_same_gradient_norm_and_direction_distinct(self):
        params=list(self.net.parameters())
        target_grad=torch.autograd.grad(
            self.target_fit_loss,params,retain_graph=True
        )
        pol,diag=build_h8_policies(self.source_cities,target_grad)
        self.assertEqual(
            set(pol),set(POLICIES)-{"adamw_step_norm_matched_raw"}
        )
        self.assertAlmostEqual(
            gradient_norm(pol["projected_half"]),
            gradient_norm(pol["norm_matched_raw"]),places=6
        )
        self.assertAlmostEqual(
            gradient_norm(pol["direction_norm_matched_half"]),
            gradient_norm(pol["global_half"]),places=6
        )
        self.assertTrue(diag["projection_triggered"])
        self.assertGreater(diag["removed_conflict_norm"],0)
        self.assertLess(abs(diag["projected_target_gradient_cos"]),1e-5)
        self.assertLess(diag["raw_target_gradient_cos"],0)

    def test_zero_gradient_is_safe(self):
        target=[torch.tensor([1.,0.])]
        sources=[
            {"city":c,"f2":[torch.zeros(2)]}
            for c in ("chengdu_m","metr-la","shenzhen")
        ]
        pol,diag=build_h8_policies(sources,target)
        self.assertEqual(diag["raw_gradient_norm"],0)
        self.assertEqual(diag["norm_matched_raw_factor"],0)
        self.assertEqual(diag["projected_equal_half_norm_factor"],0)
        self.assertTrue(all(
            torch.equal(x,torch.zeros_like(x))
            for x in pol["norm_matched_raw"]
        ))

    def test_fail_on_bad_city_tuple(self):
        t=[torch.ones(2)]
        e=[{"city":"x","f2":[torch.ones(2)]}]*3
        with self.assertRaises(ValueError):
            build_h8_policies(e,t)
        with self.assertRaises(ValueError):
            build_h8_policies(e[:2],t)

    def test_model_optimizer_unmodified_and_probe_label_not_used_in_policy(self):
        before_params={
            name:p.detach().clone() for name,p in self.net.named_parameters()
        }
        before_buffers={
            name:b.detach().clone() for name,b in self.net.named_buffers()
        }
        before_optimizer=copy.deepcopy(self.opt.state_dict())
        before_train={
            name:m.training for name,m in self.net.named_modules()
        }
        self.assertTrue(all(p.grad is None for p in self.net.parameters()))
        audit=shadow_h8_audit(
            model=self.net,optimizer=self.opt,
            source_task_loss=self.source_loss,
            target_fit_task_loss=self.target_fit_loss,
            city_entries=self.source_cities,target_data=self.target,
            probe_mask=self.probe,sample_index=0,checkpoint_epoch=2899
        )
        self.assertEqual(audit["audit_version"],"h8-v1")
        self.assertEqual(audit["n_shadow_candidates"],8)
        self.assertEqual(set(audit["candidate_scores"]),set(POLICIES))
        self.assertEqual(audit["training_updates"],0)
        self.assertEqual(audit["stage2_runs"],0)
        self.assertEqual(audit["gains_vs_f2"]["f2"]["mae"],0)
        self.assertEqual(audit["gains_vs_projected_half"]["projected_half"]["mae"],0)
        self.assertAlmostEqual(
            audit["gains_vs_f2"]["identity"]["mae"],0,places=6
        )
        self.assertEqual(len(audit["probe_window_indices"]),10)
        self.assertEqual(len(audit["fit_window_indices"]),36)
        self.assertLess(
            audit["gradient_stats"][
                "projected_vs_norm_matched_raw_gradient_norm_error"
            ],1e-6
        )
        self.assertLess(
            audit["adamw_calibration"]["relative_error"],.05
        )
        for name,p in self.net.named_parameters():
            self.assertTrue(torch.equal(p.detach(),before_params[name]))
        for name,b in self.net.named_buffers():
            self.assertTrue(torch.equal(b.detach(),before_buffers[name]))
        self.assertEqual(before_train,{
            name:m.training for name,m in self.net.named_modules()
        })
        self.assertTrue(all(p.grad is None for p in self.net.parameters()))
        for k,item in self.opt.state_dict()["state"].items():
            for n,v in item.items():
                old=before_optimizer["state"][k][n]
                if torch.is_tensor(v):
                    self.assertTrue(torch.equal(v,old))
                else:
                    self.assertEqual(v,old)

        # Alter ONLY the held-back probe labels. Fit loss was built using
        # untouched FIT labels. Intervention must be exactly unchanged.
        changed_y=self.target.y.clone()
        changed_y[self.probe]+=100
        changed=SimpleNamespace(
            x=self.target.x,y=changed_y,
            means=self.target.means,stds=self.target.stds
        )
        second=shadow_h8_audit(
            model=self.net,optimizer=self.opt,
            source_task_loss=self.source_loss,
            target_fit_task_loss=self.target_fit_loss,
            city_entries=self.source_cities,target_data=changed,
            probe_mask=self.probe,sample_index=0,checkpoint_epoch=2899
        )
        self.assertEqual(
            audit["gradient_stats"],second["gradient_stats"]
        )
        self.assertEqual(
            audit["adamw_calibration"],second["adamw_calibration"]
        )
        self.assertNotEqual(
            audit["candidate_scores"]["f2"]["mae"],
            second["candidate_scores"]["f2"]["mae"]
        )

    def test_adamw_control_bisection_has_no_dependency_on_probe(self):
        """Calibration uses FIT-only gradients and saved AdamW states."""
        from model.Meta_Models.h1_shadow_audit import (
            _grads,projected_directions,simulate_adamw_step
        )
        named=list(self.net.named_parameters())
        params=[p for _,p in named]
        src=_grads(self.source_loss,params)
        tgt=_grads(self.target_fit_loss,params)
        pol,_=build_h8_policies(self.source_cities,tgt)
        shadow=lambda ot: simulate_adamw_step(
            named,self.opt,projected_directions(src,tgt,ot)[1]
        )
        base=shadow(pol["all_off"])
        winner=shadow(pol["projected_half"])
        factor,stats=calibrate_adamw_step_norm(
            named=named,optimizer=self.opt,
            source_gradient=src,target_gradient=tgt,
            raw_ot=pol["f2"],projected_half_shadow=winner,
            no_ot_shadow=base,
        )
        self.assertGreaterEqual(factor,0.)
        self.assertLessEqual(factor,4.)
        self.assertEqual(
            stats["within_one_percent"],stats["relative_error"]<=.01
        )


if __name__=="__main__":
    unittest.main()
