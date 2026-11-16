import csv
import os
import tempfile
import unittest

import numpy as np
import torch

from experiment_diagnostics import ExperimentDiagnostics


class ExperimentDiagnosticsTest(unittest.TestCase):
    def test_writes_separate_structured_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = ExperimentDiagnostics(directory, metadata={"seed": 7})
            logger.log_epoch({"epoch": 0, "loss": 1.25, "finite": True})
            logger.log_horizon(
                0, 4, 5, np.asarray([1.0, 2.0, 3.0, 4.0]),
                np.asarray([2.0, 3.0, 4.0, 5.0]),
                np.asarray([4.0, 9.0, 16.0, 25.0]),
                np.asarray([0.1, 0.2, 0.3, 0.4]),
            )
            logger.log_gradient(
                0, "pattern_moe", torch.tensor(0.5), torch.tensor(3.0),
                0.0, 0, torch.tensor(3.1),
            )
            logger.log_runtime(0, 0.25, 1e-3, "cpu")

            with open(os.path.join(directory, "horizon_metrics.csv"), newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(float(rows[0]["horizon_minutes"]), 5.0)
            self.assertAlmostEqual(float(rows[-1]["mae"]), 4.0)

            with open(os.path.join(directory, "gradient_diagnostics.csv"), newline="", encoding="utf-8") as handle:
                gradient_rows = list(csv.DictReader(handle))
            self.assertEqual(gradient_rows[0]["module"], "pattern_moe")
            self.assertAlmostEqual(float(gradient_rows[0]["gradient_norm"]), 0.5)

            self.assertTrue(os.path.exists(os.path.join(directory, "epoch_metrics.csv")))
            self.assertTrue(os.path.exists(os.path.join(directory, "runtime.csv")))
            self.assertTrue(os.path.exists(os.path.join(directory, "run_metadata.json")))

    def test_alignment_diagnostics_write_router_ot_and_pattern_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = ExperimentDiagnostics(directory)
            diagnostics = {
                "horizon_losses": torch.tensor([0.4, 0.3]),
                "router_weights": torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]),
                "router_entropy": torch.tensor(0.8),
                "plan_diversity": torch.tensor(0.2),
                "row_marginal_errors": torch.tensor([0.01, 0.02]),
                "col_marginal_errors": torch.tensor([0.005, 0.006]),
                "router_pair_std": torch.ones(2, 3) * 0.1,
                "router_top1_fraction": torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]),
                "expert_cost_means": torch.tensor([0.2, 0.3, 0.4]),
                "pattern_usage": torch.ones(2, 4) / 4,
                "pattern_entropy": torch.ones(2) * 1.386,
                "pattern_top1_fraction": torch.ones(2) * 0.25,
                "per_scale_cost_means": torch.tensor([0.1, 0.2]),
                "scale_steps": [12, 36],
                "sampled_source": torch.tensor(8.0),
                "sampled_target": torch.tensor(9.0),
            }
            logger.log_alignment(3, 0, diagnostics, [2, 12], 5)
            for filename in ("moe_router.csv", "ot_diagnostics.csv", "pattern_usage.csv"):
                self.assertTrue(os.path.exists(os.path.join(directory, filename)))
                with open(os.path.join(directory, filename), newline="", encoding="utf-8") as handle:
                    self.assertGreaterEqual(len(list(csv.DictReader(handle))), 2)


if __name__ == "__main__":
    unittest.main()
