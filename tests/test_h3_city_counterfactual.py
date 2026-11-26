"""CPU tests for H3 seven-policy city shadow audit, no GeomLoss/GPU needed.

Run: python -m unittest discover -s tests -p 'test_h3_city_counterfactual.py'
"""
import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Meta_Models.h3_city_counterfactual import (
    make_policy_grads, masked_support_split, evaluate_probes,
    shadow_city_audit, vector_alignment
)


class MiniForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 16)
        self.predictor = nn.Linear(16, 36)

    def forward(self, data, stage="train"):
        z = torch.tanh(self.body(data.x))
        out = self.predictor(z)
        self.RevIn_mean = data.x.detach().mean()
        self.RevIn_std = data.x.detach().std()
        return out, data.y, z, data.y, None


def fake_gradients(model, seed):
    generator=torch.Generator().manual_seed(seed)
    return [torch.randn(p.shape,generator=generator) * 0.001
            for p in model.parameters()]


class H3CityCounterfactualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(204)
        self.net=MiniForecast()
        self.optimizer=torch.optim.AdamW(self.net.parameters(),lr=0.001,
                                         weight_decay=0.01)
        target_x=torch.randn(46,3,4)
        target_y=torch.randn(46,3,36)
        self.target=SimpleNamespace(
            x=target_x,y=target_y,means=torch.zeros(1),stds=torch.ones(1)
        )

    def entries(self):
        return [
            {"city":"chengdu_m",
             "f2":fake_gradients(self.net,11),
             "uniform":fake_gradients(self.net,12)},
            {"city":"shenzhen",
             "f2":fake_gradients(self.net,21),
             "uniform":fake_gradients(self.net,22)},
            {"city":"metr-la",
             "f2":fake_gradients(self.net,31),
             "uniform":fake_gradients(self.net,32)},
        ]

    def test_seven_real_policies_and_no_ot_reference(self):
        entries=self.entries()
        policies=make_policy_grads(entries)
        self.assertEqual(len(policies),8)
        self.assertEqual(set(policies),{
            "f2","city_uniform:chengdu_m","city_uniform:shenzhen",
            "city_uniform:metr-la","city_off:chengdu_m","city_off:shenzhen",
            "city_off:metr-la","all_off"
        })
        expected = [
            a + b + c for a,b,c in zip(
                entries[0]["f2"],entries[1]["f2"],entries[2]["f2"])
        ]
        for a,b in zip(expected,policies["f2"]):
            self.assertTrue(torch.allclose(a,b))
        for a,b,c in zip(
            policies["f2"],policies["city_off:chengdu_m"],
            entries[0]["f2"]
        ):
            self.assertTrue(torch.allclose(a-b,c))
        for a,b,c,d in zip(
            policies["f2"],policies["city_uniform:chengdu_m"],
            entries[0]["f2"],entries[0]["uniform"]
        ):
            self.assertTrue(torch.allclose(a-b,c-d))

    def test_target_fit_probe_window_separation(self):
        fit,probe=masked_support_split(46,torch.device("cpu"))
        self.assertEqual(probe.nonzero().flatten().tolist(),
                         [0,5,10,15,20,25,30,35,40,45])
        self.assertEqual(int(fit.sum()),36)
        self.assertFalse((fit & probe).any())
        self.assertTrue((fit | probe).all())

    def test_readonly_shadows_keep_model_and_optimizer_unmodified(self):
        data=self.target
        source_pred=self.net(
            SimpleNamespace(x=torch.randn(4,3,4),
                            y=torch.randn(4,3,36))
        )[0]
        source_loss=source_pred.square().mean()
        original_pred=self.net(data)[0]
        fit,probe=masked_support_split(46,original_pred.device)
        fit_loss=F.smooth_l1_loss(
            original_pred[fit],data.y[fit],beta=5e-4
        )
        parameters={n:p.detach().clone() for n,p in self.net.named_parameters()}
        optimizer_before=copy.deepcopy(self.optimizer.state_dict())
        # Model gradients remain empty: use autograd.grad, never backward.
        self.assertTrue(all(p.grad is None for p in self.net.parameters()))
        r=shadow_city_audit(
            model=self.net,optimizer=self.optimizer,
            source_task_loss=source_loss,
            target_fit_task_loss=fit_loss,
            city_entries=self.entries(),target_data=data,
            probe_mask=probe,epoch=1500,sample_index=0
        )
        self.assertEqual(len(r["candidate_scores"]),8)
        self.assertEqual(len(r["gain_vs_f2"]),8)
        self.assertEqual(r["training_updates"],0)
        self.assertEqual(r["gain_vs_f2"]["f2"]["smooth_l1"],0.0)
        for score in r["candidate_scores"].values():
            self.assertEqual(len(score["band_smooth_l1"]),3)
            self.assertEqual(set(score["mae_at_minutes"]),
                             {"10","60","120","180"})
        for n,p in self.net.named_parameters():
            self.assertTrue(torch.equal(parameters[n],p.detach()))
        self.assertEqual(self.optimizer.state_dict()["state"],
                         optimizer_before["state"])
        self.assertTrue(all(p.grad is None for p in self.net.parameters()))

    def test_scores_use_real_probe_truth_not_fit(self):
        net=self.net
        base=net(self.target)[0]
        x=base[:2]
        truth=self.target.y[:2]
        score=evaluate_probes(x,truth)
        self.assertGreaterEqual(score["smooth_l1"],0)
        self.assertGreaterEqual(score["mae"],0)
        self.assertEqual(len(score["band_smooth_l1"]),3)
        with self.assertRaises(ValueError):
            evaluate_probes(x[:,:,:12],truth[:,:,:12])

    def test_city_uniqueness_and_grad_shapes(self):
        entries=self.entries()
        entries[2]["city"]="shenzhen"
        with self.assertRaises(ValueError):
            make_policy_grads(entries)
        self.assertAlmostEqual(vector_alignment(
            [torch.ones(3)],[-torch.ones(3)])["cos"],-1.0,places=5)


if __name__=="__main__":
    unittest.main()
