"""Synthetic test for the PMME opt-in diagnostics (no traffic dataset needed)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model" / "Meta_Models"))
sys.path.insert(0, str(ROOT / "scripts"))

from pmme_diagnostics import TargetTransferAudit
from analyze_pmme_diagnostics import analyze


class TestAudit(unittest.TestCase):
    def _args(self, folder, every):
        return SimpleNamespace(
            save_dir=folder, diagnostic_every=every, diagnostic_source_cap=8,
            DAepoch=0, seed=7, test_dataset="pems-bay",
        )

    def test_default_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = TargetTransferAudit(self._args(tmp, every=0))
            self.assertFalse(audit.active(0))
            self.assertFalse((Path(tmp) / "diagnostics").exists())

    def test_pair_and_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = TargetTransferAudit(self._args(tmp, every=1))
            self.assertTrue(audit.active(0))
            B, N, H = 46, 2, 36
            x = torch.ones((B, N, 288, 2), dtype=torch.float32)
            y = torch.ones((B, N, H), dtype=torch.float32) * 42.0
            for b in range(B):
                y[b, :, :] += (b % 3 - 1) * torch.linspace(0, 5, H)
            pred_train = y + 1.0
            pred_eval = y + 2.0
            target_data = SimpleNamespace(
                x=x, y=y, means=np.array([40.0]), stds=np.array([5.0]),
                window_indices=torch.arange(B, dtype=torch.long),
            )
            audit.capture_target(target_data, pred_train, pred_eval, y)
            self.assertEqual(audit.target["window_idx"].shape, (B, N))
            self.assertEqual(audit.target["eval_mae"].shape, (B, N))

            S_B, S_N = 16, 3
            source_data = SimpleNamespace(
                data_name="metr-la",
                x=torch.ones((S_B, S_N, 288, 2)),
                window_indices=torch.arange(S_B) * 2,
            )
            src = torch.zeros((S_B * S_N, 128))
            dst = torch.zeros((15 * N, 128))
            src_future = torch.zeros((S_B * S_N, H))
            dst_future = torch.zeros((15 * N, H))
            raw_src = torch.linspace(0.1, 0.9, S_B * S_N)
            raw_dst = torch.linspace(0.2, 0.8, 15 * N)
            beta = raw_dst / raw_dst.sum()
            audit.capture_pair(
                epoch=0, index=0, source_data=source_data,
                source_rep=src, target_rep=dst,
                source_future=src_future, target_future=dst_future,
                gp_source=raw_src, gp_target=raw_dst, beta=beta,
                ot_active=True, ot_loss=torch.tensor(0.1),
                source_task_loss=torch.tensor(1.2),
            )
            audit.finish_epoch(0, torch.tensor(2.0),
                               torch.tensor(0.1), torch.tensor(3.0))
            folder = Path(tmp) / "diagnostics"
            summaries = [json.loads(x) for x in (folder / "pair_summary.jsonl").read_text().splitlines()]
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["target_size"], 15 * N)
            self.assertAlmostEqual(summaries[0]["beta_sum"], 1.0, places=5)
            self.assertEqual(summaries[0]["source_city"], "metr-la")
            with np.load(folder / summaries[0]["sample_file"], allow_pickle=False) as data:
                self.assertEqual(len(data["beta"]), 30)
                self.assertEqual(int(data["window_idx"][0]), 0)
                self.assertAlmostEqual(float(data["eval_mae"].mean()), 2.0, places=5)
            self.assertIsNone(audit.target)
            analyze(folder, folder / "analysis")
            self.assertTrue((folder / "analysis" / "mode_by_pair.csv").exists())
            self.assertTrue((folder / "analysis" / "difficulty_by_pair.csv").exists())

    def test_invalid_beta_is_recorded_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = TargetTransferAudit(self._args(tmp, every=1))
            B, N, H = 46, 1, 36
            target = SimpleNamespace(
                x=torch.zeros(B, N, 288, 2), y=torch.zeros(B, N, H),
                means=np.array([0.0]), stds=np.array([1.0]),
                window_indices=torch.arange(B),
            )
            vals = torch.zeros(B, N, H)
            audit.capture_target(target, vals, vals, vals)
            src_meta = SimpleNamespace(data_name="chengdu_m",
                                       x=torch.zeros(16, 2, 288, 2),
                                       window_indices=torch.arange(16))
            audit.capture_pair(
                epoch=0, index=0, source_data=src_meta,
                source_rep=torch.zeros(32, 128), target_rep=torch.zeros(15, 128),
                source_future=torch.zeros(32, H), target_future=torch.zeros(15, H),
                gp_source=None, gp_target=None, beta=None,
                ot_active=False, ot_loss=None, source_task_loss=0.0,
            )
            audit.finish_epoch(0, 0, 0, 0)
            record = json.loads((Path(tmp) / "diagnostics" / "pair_summary.jsonl").read_text().strip())
            self.assertFalse(record["beta_defined"])
            self.assertFalse(record["ot_active"])
            self.assertEqual(record["beta_sum"], 0.0)


if __name__ == "__main__":
    unittest.main()
