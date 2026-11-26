"""CPU-only checks for log guards and support-selected Stage-I/II metrics."""
import unittest

from summarize_f2_ot_decay import (
    parse_checkpoint_selected_metrics, summarize_selected, report_differences
)
from verify_f2_ot_schedule_log import verify
from model.Meta_Models.f2_ot_schedule import effective_ot_scale


def test_horizon_block(offset):
    return "\n".join(
        "Horizon %d : Unnormed MSE : 10.0, RMSE : %.5f, MAE : %.5f, MAPE: 0.0" %
        (k,2+offset+k/100,1+offset+k/100)
        for k in range(36)
    )


def test_logs():
    return "\n".join([
        "epoch: 0",
        "[INFO] Enter test phase",
        test_horizon_block(0.0),
        "epoch:0 best!",
        "epoch: 99",
        "[INFO] Enter test phase",
        test_horizon_block(10.0),
        "epoch:99 best!",
        "Source training Ending!!!",
        "epoch: 0",
        "[INFO] Enter test phase",
        test_horizon_block(20.0),
        "epoch:0 best!",
        "epoch: 99",
        "[INFO] Enter test phase",
        test_horizon_block(30.0),
        "not_best",
        "2ndStage Source training Ending!!!",
    ])


class SelectedCheckpointMetricTests(unittest.TestCase):
    def test_test_epoch_not_selected_when_support_not_best(self):
        selected,counts=parse_checkpoint_selected_metrics(test_logs())
        self.assertEqual(selected["stage1"][0],99)
        self.assertEqual(selected["stage2"][0],0)
        self.assertEqual(counts["stage1"],2)
        self.assertEqual(counts["stage2"],1)
        row=summarize_selected("stage2",selected["stage2"],"t2",1,"onecommit")
        self.assertAlmostEqual(row["mae_10min"],21.01,places=5)
        self.assertAlmostEqual(row["mae_180min"],21.35,places=5)
        self.assertAlmostEqual(row["mean_36_horizon_mae"],21.175,places=5)

    def test_pair_delta_sign_means_decrease_is_better(self):
        c={"mae_%dmin"%m:1.0 for m in (10,60,120,180)}
        t0=dict(mode="t0",seed="1",stage="stage2",mean_36_horizon_mae=2.5,**c)
        t1=dict(mode="t1",seed="1",stage="stage2",
                mean_36_horizon_mae=2.4,**{
                    "mae_%dmin"%m:0.9 for m in (10,60,120,180)
                })
        delta=report_differences([t0,t1])[0]
        self.assertAlmostEqual(delta["delta_mean_36_horizon_mae"],-0.1)
        self.assertAlmostEqual(delta["delta_mae_60min"],-0.1)


class ScheduleLogGuardTests(unittest.TestCase):
    def setUp(self):
        self.k=dict(
            stage1=4,daepoch=1,decay_start=2,decay_end=3,
            base_scale=0.01,tail_fraction=0.25
        )

    def logs(self,mode,incorrect=False):
        parts=[]
        for ep in range(4):
            scale=effective_ot_scale(
                ep,mode=mode,base_scale=0.01,
                daepoch=1,decay_start=2,decay_end=3
            )
            if incorrect and ep==3:
                scale=0.01
            parts.append(
                "f2_ot_schedule epoch %d policy %s phase test "
                "base_scale 0.01 effective_scale %.12g "
                "DAcoef 1.0 effective_ot_coefficient %.12g" %
                (ep,mode,scale,scale)
            )
        return "\n".join(parts)+"\nSource training Ending!!!"

    def test_t0_t1_t2_logs(self):
        for mode in ("t0","t1","t2"):
            with self.subTest(mode=mode):
                result=verify(self.logs(mode),mode=mode,**self.k)
                self.assertEqual(len(result),4)

    def test_guard_rejects_undecayed_t1(self):
        with self.assertRaises(ValueError):
            verify(self.logs("t1",incorrect=True),mode="t1",**self.k)


if __name__=="__main__":
    unittest.main()
