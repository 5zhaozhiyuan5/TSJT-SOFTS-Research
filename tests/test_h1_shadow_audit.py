"""CPU-only tests for H1 paired-shadow counterfactual audit.

Run:
  python -m unittest discover -s tests -p 'test_h1_shadow_audit.py'
"""
import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Meta_Models.h1_shadow_audit import (
    audit_h1_shadow_update,
    projected_directions,
    simulate_adamw_step,
)


class MiniTrafficModel(nn.Module):
    """Mimic PMME forecasting_model.projection1/2 for functional_call."""
    def __init__(self):
        super().__init__()
        self.forecasting_model = nn.Module()
        self.forecasting_model.projection1 = nn.Linear(8, 128)
        self.forecasting_model.projection2 = nn.Linear(128, 36)

    def forward(self, data, stage="train"):
        z = self.forecasting_model.projection1(data.x)
        out = self.forecasting_model.projection2(z)
        self.RevIn_mean = data.x.mean(dim=-1, keepdim=True).detach()
        self.RevIn_std = torch.ones_like(out[..., :1])
        return out, data.y, z, data.y, None


class H1AuditTests(unittest.TestCase):
    def test_full_vector_projection(self):
        task = [torch.tensor([2.0, 0.0]), torch.tensor([0.0])]
        target = [torch.tensor([-1.0, 0.0]), torch.tensor([1.0])]
        ot = [torch.tensor([0.0, 1.0]), torch.tensor([1.0])]
        baseline, augmented = projected_directions(task, target, ot)
        def dot(gs):
            return sum((a*b).sum().item() for a,b in zip(gs,target))
        self.assertGreaterEqual(dot(baseline), -1e-6)
        self.assertGreaterEqual(dot(augmented), -1e-6)

    def test_shadow_adamw_matches_real_optimizer(self):
        torch.manual_seed(31)
        net = nn.Linear(7, 3)
        opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.01)
        x = torch.randn(11, 7)
        y = torch.randn(11, 3)
        for _ in range(2):
            opt.zero_grad()
            F.smooth_l1_loss(net(x), y).backward()
            opt.step()
        opt.zero_grad()
        named = list(net.named_parameters())
        params = [p for _,p in named]
        # Check stateful moment / step-count replay.
        loss_task = F.smooth_l1_loss(net(x), y)
        loss_ot = 0.01*net(x).square().mean()
        loss_target = (net(x) - y).square().mean()
        task = list(torch.autograd.grad(loss_task, params, retain_graph=True))
        ot = list(torch.autograd.grad(loss_ot, params, retain_graph=True))
        target = list(torch.autograd.grad(loss_target, params, retain_graph=True))
        base, augmented = projected_directions(task, target, ot)
        original_state = [
            (p.clone().detach(), opt.state[p]["exp_avg"].clone(),
             opt.state[p]["exp_avg_sq"].clone(),
             opt.state[p]["step"].clone()) for p in params
        ]
        expected = simulate_adamw_step(named, opt, augmented)
        for name,p in named:
            self.assertTrue(torch.isfinite(expected[name]).all())
            self.assertIsNone(p.grad)
        opt.zero_grad()
        for p,g in zip(params, augmented):
            p.grad = g.detach().clone()
        opt.step()
        for name,p in named:
            self.assertTrue(torch.allclose(
                p.detach(), expected[name], atol=1e-7, rtol=2e-6
            ), msg=name)
        for (old,old_m,old_v,old_step),p in zip(original_state,params):
            self.assertFalse(torch.equal(old, p))
        self.assertEqual(len(base), len(params))

    def test_h1_counterfactual_preserves_parameters_and_rng(self):
        torch.manual_seed(77)
        model = MiniTrafficModel()
        opt = torch.optim.AdamW(model.parameters(), lr=0.0005,
                                weight_decay=0.01)
        data = SimpleNamespace(
            x=torch.randn(10, 4, 8), y=torch.randn(10, 4, 36)
        )
        pred, _, rep, _, _ = model(data)
        source_loss = F.smooth_l1_loss(pred, data.y)
        target_loss = F.smooth_l1_loss(pred[:, :2], data.y[:, :2])
        ot_loss = rep.square().mean() * .01
        probe_idx = torch.tensor([0,5])
        probe_prediction = pred.detach().index_select(0, probe_idx)
        probe_input = data.x.index_select(0, probe_idx)
        probe_target = data.y.index_select(0, probe_idx)
        before_rng = torch.get_rng_state().clone()
        before_params = {k: v.detach().clone()
                         for k,v in model.named_parameters()}
        old_mean = model.RevIn_mean
        old_std = model.RevIn_std
        report = audit_h1_shadow_update(
            model=model, optimizer=opt,
            source_task_loss=source_loss, da_loss=ot_loss,
            target_task_loss=target_loss,
            target_data=data, probe_input=probe_input,
            probe_prediction=probe_prediction, probe_target=probe_target,
            probe_scale=torch.ones(2,4,1),
            probe_indices=probe_idx, epoch=1600
        )
        for key in ("full_gain","head_gain","full_no_ot","full_f2",
                    "head_no_ot","head_f2"):
            self.assertTrue(math.isfinite(report[key]), msg=key)
        self.assertEqual(len(report["full_band_gains"]),3)
        self.assertEqual(len(report["head_band_gains"]),3)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        self.assertIs(model.RevIn_mean, old_mean)
        self.assertIs(model.RevIn_std, old_std)
        for name,p in model.named_parameters():
            self.assertTrue(torch.equal(p.detach(), before_params[name]))
            self.assertIsNone(p.grad)
        self.assertEqual(len(opt.state), 0)


if __name__ == "__main__":
    unittest.main()
