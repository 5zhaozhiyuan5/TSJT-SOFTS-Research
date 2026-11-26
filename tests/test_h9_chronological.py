"""H9 CPU-only time/leakage, cloned optimizer and result-summary tests.

python -m unittest discover -s tests -p 'test_h9_chronological.py'
No PMME dataset, GPU, GeomLoss, or F2 checkpoint required for these tests.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.Meta_Models.h9_temporal_holdout import (
    chronological_bounds,create_development_batches,score_development,
)
from model.Meta_Models.h9_checkpoint_adaptation import (
    clone_f2,reproducible_step_seed,POLICIES
)
from summarize_h9_chronological import summarize


class BlindTinyForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(.5))
        self.forecasting_model=nn.Module()
        self.forecasting_model.projection2=nn.Linear(2,2)

    def forward(self,data,stage="test"):
        # This invariant protects the hidden future from the model, even when
        # validation itself has physical-speed labels for scoring.
        assert bool((data.y==0).all()),"Development y entered model forward"
        x=data.x.float()
        pred=self.weight*x[:,:,-1,0].unsqueeze(-1).expand(-1,-1,36)
        return pred,data.y,None,None,None


def synthetic_file(path,*,change_future=False):
    length=540
    t=np.arange(length,dtype=np.float32)
    x=np.zeros((length,3,4),dtype=np.float32)
    for node in range(3):
        x[:,node,0]=30+t*.015+node
        x[:,node,-1]=t%2016
    if change_future:
        # Support [0,180), development [180,420); untouched [420,540).
        x[420:,:,0]+=10000
    np.save(path,x)
    return x


class TemporalGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/"target.npy"
        synthetic_file(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self,**kwargs):
        return create_development_batches(
            self.path,
            normalization_mean=np.array([50.]),
            normalization_std=np.array([12.]),
            support_days=3,validation_days=4,
            steps_per_day=60,history=48,horizon=36,
            stride=12,batch_size=4,
            **kwargs
        )

    def test_time_boundaries_no_support_overlap_future_reserved(self):
        bounds=chronological_bounds(
            540,support_days=3,validation_days=4,steps_per_day=60,
            history=48,horizon=36,stride=12
        )
        self.assertEqual(bounds["support_end_exclusive"],180)
        self.assertEqual(bounds["development_start_inclusive"],180)
        self.assertEqual(bounds["development_end_exclusive"],420)
        self.assertEqual(bounds["untouched_future_start_inclusive"],420)
        self.assertEqual(bounds["starts"][0],180)
        self.assertLessEqual(bounds["starts"][-1]+48+36,420)
        self.assertEqual(len(bounds["starts"]),14)
        with self.assertRaises(ValueError):
            chronological_bounds(
                420,support_days=3,validation_days=4,
                steps_per_day=60,history=48,horizon=36,stride=12
            )
        with self.assertRaises(ValueError):
            chronological_bounds(
                540,support_days=3,validation_days=4,
                steps_per_day=60,history=200,horizon=36,stride=12
            )

    def test_only_development_is_read_and_model_is_blind(self):
        batches,bounds=self.build()
        self.assertEqual(bounds["window_count"],14)
        self.assertEqual(sum(len(d[2]) for d in batches),14)
        for blind,truth,spans,days in batches:
            self.assertTrue(torch.equal(
                blind.y,torch.zeros_like(truth)
            ))
            for span in spans:
                self.assertGreaterEqual(span["input_start"],180)
                self.assertLessEqual(span["forecast_end_exclusive"],420)
            self.assertTrue(all(i>=0 for i in days))
        scorer=BlindTinyForecast()
        out=score_development(scorer,batches)
        self.assertEqual(out["window_count"],14)
        self.assertTrue(np.isfinite(out["mae"]))
        self.assertTrue(np.isfinite(out["rmse"]))
        self.assertEqual(
            set(out["horizon_band_mae"]),{"short","mid","long"}
        )
        self.assertEqual(
            set(out["mae_at_minutes"]),{"10","60","120","180"}
        )

    def test_sentinels_after_cutoff_cannot_change_dev(self):
        before,bounds=self.build()
        before_y=torch.cat([x[1] for x in before],dim=0)
        before_x=torch.cat([x[0].x for x in before],dim=0)
        synthetic_file(self.path,change_future=True)
        after,_=self.build()
        after_y=torch.cat([x[1] for x in after],dim=0)
        after_x=torch.cat([x[0].x for x in after],dim=0)
        self.assertTrue(torch.equal(before_x,after_x))
        self.assertTrue(torch.equal(before_y,after_y))

    def test_fixed_label_independent_window_sampling(self):
        b1,d1=self.build(max_windows=7)
        self.assertEqual(d1["window_count"],7)
        self.assertEqual(len(set(d1["window_start_indices"])),7)
        # Change development true future values, but selected window indexes
        # must remain unchanged (selection uses only indices).
        raw=np.load(self.path)
        raw[265:400,:,0]+=20
        np.save(self.path,raw)
        b2,d2=self.build(max_windows=7)
        self.assertEqual(d1["window_start_indices"],d2["window_start_indices"])
        self.assertNotEqual(
            float(torch.cat([b[1] for b in b1]).sum()),
            float(torch.cat([b[1] for b in b2]).sum()),
        )


class CloneAndSummaryTests(unittest.TestCase):
    def test_cloned_optimizer_not_original(self):
        torch.manual_seed(13)
        model=BlindTinyForecast()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        for p in model.parameters():
            p.grad=torch.randn_like(p)
        opt.step()
        opt.zero_grad(set_to_none=True)
        weights={n:p.detach().clone() for n,p in model.named_parameters()}
        state=copy.deepcopy(opt.state_dict())
        captured=[]
        handle=model.forecasting_model.projection2.register_forward_hook(
            lambda *_:captured.append(1)
        )
        try:
            copy_model,copy_opt=clone_f2(model,opt)
            self.assertEqual(
                len(copy_model.forecasting_model.projection2._forward_hooks),0
            )
            for p in copy_model.parameters():
                p.grad=torch.ones_like(p)
            copy_opt.step()
            for n,p in model.named_parameters():
                self.assertTrue(torch.equal(p.detach(),weights[n]))
            for k,item in opt.state_dict()["state"].items():
                for n,x in item.items():
                    old=state["state"][k][n]
                    if torch.is_tensor(x):
                        self.assertTrue(torch.equal(x,old))
                    else:
                        self.assertEqual(x,old)
            self.assertEqual(
                reproducible_step_seed(1),reproducible_step_seed(1)
            )
            self.assertNotEqual(
                reproducible_step_seed(1),reproducible_step_seed(2)
            )
        finally:
            handle.remove()

    def test_summary_rejects_support_dev_overlap(self):
        bounds={
            "support_end_exclusive":180,
            "development_start_inclusive":180,
            "development_end_exclusive":420,
            "forecast_start_min":228,
            "forecast_end_max_exclusive":420,
            "untouched_future_start_inclusive":420,
            "window_count":14,
        }
        def score(mae):
            return {
                "mae":mae,"rmse":mae*1.2,"window_count":14,
                "horizon_band_mae":{
                    "short":mae,"mid":mae,"long":mae
                },
                "mae_at_minutes":{k:mae for k in ("10","60","120","180")},
                "forecast_days":{
                    "1":{"mae":mae,"rmse":mae*1.2,"value_count":100}
                }
            }
        rows={
            p:{"0":score(2.8),"1":score(2.78 if p=="projected_half" else 2.8)}
            for p in POLICIES
        }
        report={
            "audit_version":"h9-v1",
            "original_f2_updates":0,"stage2_runs":0,
            "development_labels_in_update":False,
            "official_test_after_development_touched":False,
            "development_time_bounds":bounds,
            "evaluated_milestones":[0,1],
            "adaptation_update_count_per_policy":1,
            "scores":rows,
        }
        summary,daywise=summarize(report)
        self.assertEqual(len(summary),len(POLICIES)*2)
        self.assertEqual(len(daywise),len(POLICIES)*2)
        winner=next(
            v for v in summary if v["step"]==1
            and v["policy"]=="projected_half"
        )
        self.assertGreater(winner["gain_vs_half"],0)
        report["development_time_bounds"]=dict(bounds)
        report["development_time_bounds"]["forecast_start_min"]=170
        with self.assertRaises(ValueError):
            summarize(report)


if __name__=="__main__":
    unittest.main()
