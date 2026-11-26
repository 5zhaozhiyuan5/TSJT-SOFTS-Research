"""CPU regressions for H4 only. No traffic data or GeomLoss required.

python -m unittest discover -s tests -p 'test_h4_error_profiles.py'
"""
import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Meta_Models.h4_error_profile_counterfactual import (
    GROUP_NAMES,error_profile_groups,downweight_group,shadow_profile_audit
)


class ProfileGroupsTest(unittest.TestCase):
    @staticmethod
    def samples():
        n=40
        pred=torch.full((1,n,36),100.0)
        truth=torch.full((1,n,36),100.0)
        hist=torch.zeros(1,n,7,2)
        pred[:, :10] += 0.1
        pred[:, 30:] += 15
        for i in range(10,30):
            if i % 4 == 3:
                pred[:,i] += 4
            else:
                b=i%3
                pred[:,i] += 1
                pred[:,i,b*12:(b+1)*12] += 7
        return pred,truth,hist

    def test_disjoint_partition_and_accuracy(self):
        p,y,h=self.samples()
        masks,stats=error_profile_groups(p,y,h,[90.0],[10.0])
        self.assertEqual(set(masks),set(GROUP_NAMES))
        self.assertEqual(stats["n"],40)
        self.assertEqual(stats["group_counts"]["accurate"],10)
        self.assertEqual(stats["group_counts"]["inaccurate"],10)
        self.assertEqual(sum(v.sum().item() for v in masks.values()),40)
        union=torch.stack(list(masks.values())).sum(0)
        self.assertTrue(torch.equal(union,torch.ones_like(union)))
        self.assertTrue(masks["short_weak"].any())
        self.assertTrue(masks["mid_weak"].any())
        self.assertTrue(masks["long_weak"].any())
        self.assertTrue(masks["balanced"].any())
        self.assertGreater(
            stats["group_mean_normalized_mae"]["inaccurate"],
            stats["group_mean_normalized_mae"]["accurate"]
        )

    def test_city_standardization_invariance(self):
        p,y,h=self.samples()
        a,s1=error_profile_groups(p,y,h,[90.0],[10.0])
        b,s2=error_profile_groups(p*2+10,y*2+10,h,[190.0],[20.0])
        for name in GROUP_NAMES:
            self.assertTrue(torch.equal(a[name],b[name]))
        for x,z in zip(s1["mean_mae_by_band"],s2["mean_mae_by_band"]):
            self.assertAlmostEqual(x,z,places=5)

    def test_bounded_group_suppression(self):
        p,y,h=self.samples()
        masks,_=error_profile_groups(p,y,h,[90.0],[10.0])
        torch.manual_seed(7)
        a=torch.softmax(torch.randn(40),dim=0)
        b,meta=downweight_group(a,masks["inaccurate"],keep_fraction=0.5)
        self.assertAlmostEqual(b.sum().item(),1.0,places=6)
        self.assertGreater(meta["group_alpha_mass_before"],
                           meta["group_alpha_mass_after"])
        self.assertTrue((b>0).all())
        self.assertEqual(meta["group_count"],10)
        with self.assertRaises(ValueError):
            downweight_group(a,torch.zeros_like(masks["inaccurate"]))
        with self.assertRaises(ValueError):
            downweight_group(a,masks["accurate"],keep_fraction=0)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(2,36)

    def forward(self,data,stage="train"):
        out=self.linear(torch.as_tensor(data.x,dtype=self.linear.weight.dtype))
        return out,torch.as_tensor(data.y),None,None,None


class ShadowTest(unittest.TestCase):
    def test_readonly_and_horizon_outputs(self):
        torch.manual_seed(17)
        net=DummyModel()
        opt=torch.optim.AdamW(net.parameters(),lr=0.001)
        src=SimpleNamespace(
            x=torch.randn(6,2,2),y=torch.randn(6,2,36),
            means=torch.tensor([0]),stds=torch.tensor([1])
        )
        tar=SimpleNamespace(
            x=torch.randn(10,2,2),y=torch.randn(10,2,36),
            means=torch.tensor([0]),stds=torch.tensor([1])
        )
        loss=F.smooth_l1_loss(net(src)[0],src.y)
        loss.backward();opt.step();opt.zero_grad(set_to_none=True)
        src_loss=F.smooth_l1_loss(net(src)[0],src.y)
        tgt_loss=F.smooth_l1_loss(net(tar)[0][:8],tar.y[:8])
        parameters_before={k:v.detach().clone() for k,v in net.named_parameters()}
        optimizer_before=copy.deepcopy(opt.state_dict())
        f2=[torch.full_like(p,0.01) for p in net.parameters()]
        changed=[torch.full_like(p,0.007) for p in net.parameters()]
        records=[
            {"city":city,"f2":f2,"changed":{"accurate":changed}}
            for city in ("metr-la","shenzhen","chengdu_m")
        ]
        mask=torch.zeros(10,dtype=torch.bool)
        mask[-2:]=True
        row=shadow_profile_audit(
            model=net,optimizer=opt,
            source_task_loss=src_loss,target_fit_task_loss=tgt_loss,
            entries=records,target_data=tar,probe_mask=mask,
            checkpoint_epoch=2899,sample_index=0
        )
        self.assertEqual(row["training_updates"],0)
        self.assertEqual(row["n_shadow_candidates"],4)
        self.assertEqual(len(row["gains"]),3)
        self.assertEqual(len(row["f2_baseline"]["band_smooth_l1"]),3)
        for key in row["gains"]:
            self.assertEqual(len(row["gains"][key]["suppress_gain_bands"]),3)
        for k,v in net.named_parameters():
            self.assertTrue(torch.equal(v,parameters_before[k]))
            self.assertIsNone(v.grad)
        after=opt.state_dict()
        for key,value in optimizer_before["state"].items():
            for field,x in value.items():
                y=after["state"][key][field]
                if torch.is_tensor(x):
                    self.assertTrue(torch.equal(x,y))
                else:self.assertEqual(x,y)


if __name__=="__main__":
    unittest.main()
