"""H2-A1/A2/A3 CPU-only smoke and mathematical tests.

The toy OT losses below are NOT the real GeomLoss operators. Actual training
still uses the official full-batch Sinkhorn and full SOFTS/PMME graph.

Run:
  python -m unittest discover -s tests -p 'test_h2_candidate_audit.py'
"""
import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Meta_Models.h2_candidate_audit import (
    detached_model_gradients, global_and_city_candidate_gradients,
    sum_gradient_vectors, support_fold_slices,
    score_shadow_predictions, evaluate_h2_candidate_audit
)


class MiniTrafficModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.forecasting_model = nn.Module()
        self.forecasting_model.projection1 = nn.Linear(8, 128)
        self.forecasting_model.projection2 = nn.Linear(128, 36)

    def forward(self, data, stage="train"):
        z = self.forecasting_model.projection1(data.x)
        out = self.forecasting_model.projection2(z)
        self.RevIn_mean = data.x.mean(dim=-1, keepdim=True).detach()
        self.RevIn_std = out[..., :1].detach().clone()
        return out, data.y, z, data.y, None


class H2CandidateAuditTests(unittest.TestCase):
    def test_global_and_city_gradient_composition(self):
        def grad(v):
            return [torch.tensor([v, 2*v]), torch.tensor([v])]
        cities = [
            {"city":"alpha","f2":grad(1.),"uniform":grad(2.),
             "mixed":grad(1.5)},
            {"city":"beta","f2":grad(10.),"uniform":grad(20.),
             "mixed":grad(15.)},
            {"city":"gamma","f2":grad(100.),"uniform":grad(200.),
             "mixed":grad(150.)}
        ]
        result = global_and_city_candidate_gradients(cities)
        self.assertEqual(set(result), {
            "f2", "mixed", "uniform",
            "city_uniform:alpha", "city_uniform:beta",
            "city_uniform:gamma"
        })
        self.assertAlmostEqual(float(result["f2"][0][0]),111.)
        self.assertAlmostEqual(float(result["uniform"][0][0]),222.)
        self.assertAlmostEqual(float(result["mixed"][0][0]),166.5)
        self.assertAlmostEqual(float(result["city_uniform:alpha"][0][0]),112.)
        self.assertAlmostEqual(float(result["city_uniform:beta"][0][0]),121.)
        self.assertAlmostEqual(float(result["city_uniform:gamma"][0][0]),211.)
        with self.assertRaises(ValueError):
            global_and_city_candidate_gradients([cities[0], cities[0]])
        self.assertEqual(sum_gradient_vectors(
            [grad(1.),grad(2.)])[0].tolist(),[3.,6.])

    def test_contiguous_folds_and_paired_loss_signs(self):
        folds = support_fold_slices(46)
        self.assertEqual([(f.start,f.stop) for f in folds],
                         [(0,15),(15,30),(30,46)])
        truth = torch.zeros((12,3,36))
        off = torch.ones_like(truth)
        uniform = torch.zeros_like(truth)
        mixed = 0.5 * torch.ones_like(truth)
        f2 = 0.8 * torch.ones_like(truth)
        city_u = 0.3 * torch.ones_like(truth)
        report = score_shadow_predictions({
            "off":off,"uniform":uniform,"mixed":mixed,
            "f2":f2,"city_uniform:beta":city_u,
        }, truth)
        self.assertEqual(report["total"]["best_global"],"uniform")
        self.assertTrue(report["fold_consensus"])
        self.assertEqual(report["fold_best_global"],
                         ["uniform", "uniform", "uniform"])
        self.assertTrue(report["global_positive_in_all_folds"]["uniform"])
        self.assertGreater(report["total"]["scores"]
                           ["city_uniform:beta"]["gain_vs_f2"],0)
        self.assertLess(report["total"]["scores"]["off"]["gain_vs_off"]
                        + 1e-12,1e-7)

    def test_real_shadow_simulation_does_not_mutate_model(self):
        torch.manual_seed(78)
        model = MiniTrafficModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
        data = SimpleNamespace(
            x=torch.randn(9,2,8), y=torch.randn(9,2,36)
        )
        out, _, rep, _, _ = model(data)
        task_loss = F.smooth_l1_loss(out, data.y)
        target_loss = F.smooth_l1_loss(out[:4], data.y[:4])
        params = list(model.parameters())
        city_grads = []
        for i, name in enumerate(("chengdu_m","shenzhen","metr-la")):
            base = rep.square().mean() * (0.001 + i*0.0001)
            city_grads.append({
                "city":name,
                "f2":detached_model_gradients(base, model),
                "uniform":detached_model_gradients(2*base, model),
                "mixed":detached_model_gradients(1.5*base, model),
            })
        old_parameters = {
            name:p.detach().clone() for name,p in model.named_parameters()
        }
        old_rng = torch.get_rng_state().clone()
        cached_mean = model.RevIn_mean
        cached_std = model.RevIn_std
        report = evaluate_h2_candidate_audit(
            model=model, optimizer=optimizer,
            source_task_loss=task_loss, target_task_loss=target_loss,
            city_gradients=city_grads, target_data=data,
            epoch=1600, citywise=True
        )
        self.assertEqual(report["n_shadow_candidates"],7)
        self.assertEqual(report["n_windows"],9)
        self.assertEqual(set(report["total"]["scores"]),
            {"off","uniform","mixed","f2",
             "city_uniform:chengdu_m",
             "city_uniform:shenzhen","city_uniform:metr-la"})
        for row in report["total"]["scores"].values():
            self.assertTrue(math.isfinite(row["loss"]))
            self.assertTrue(math.isfinite(row["gain_vs_off"]))
        self.assertEqual(len(report["folds"]),3)
        self.assertTrue(torch.equal(torch.get_rng_state(),old_rng))
        self.assertEqual(len(optimizer.state),0)
        self.assertIs(model.RevIn_mean,cached_mean)
        self.assertIs(model.RevIn_std,cached_std)
        for name,param in model.named_parameters():
            self.assertTrue(torch.equal(param.detach(),old_parameters[name]))
            self.assertIsNone(param.grad)


if __name__ == "__main__":
    unittest.main()
