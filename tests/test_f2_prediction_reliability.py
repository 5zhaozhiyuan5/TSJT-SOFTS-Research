"""CPU unit tests for source prediction reliability calibration."""
import unittest

import torch

from model.Meta_Models.f2_prediction_reliability import (
    calibrate_source_reliability,
)


class F2PredictionReliabilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.alpha = torch.full((6,), 1.0 / 6.0)
        # Six source window x sensor items, each with a 36-step forecast.
        self.target = torch.zeros(2, 3, 36)
        errors = torch.tensor([0.1, 0.2, 0.3, 1.0, 2.0, 4.0])
        self.prediction = errors.view(2, 3, 1).expand(2, 3, 36).clone()

    def test_reliable_predictions_receive_more_source_mass(self):
        alpha_new, stats = calibrate_source_reliability(
            self.alpha, self.prediction, self.target,
            blend=0.5, sensitivity=0.5,
        )
        self.assertTrue(torch.isfinite(alpha_new).all())
        self.assertTrue((alpha_new >= 0).all())
        self.assertAlmostEqual(float(alpha_new.sum()), 1.0, places=6)
        self.assertGreater(float(alpha_new[0]), float(alpha_new[-1]))
        self.assertGreater(stats["source_l1_shift"], 0.0)
        self.assertLess(stats["error_alpha_corr_after"], 0.0)
        self.assertTrue(stats["target_marginal_unchanged"])

    def test_zero_blend_preserves_original_f2_marginal(self):
        alpha_new, stats = calibrate_source_reliability(
            self.alpha, self.prediction, self.target, blend=0.0
        )
        self.assertTrue(torch.allclose(alpha_new, self.alpha, atol=1e-7))
        self.assertAlmostEqual(stats["source_l1_shift"], 0.0, places=7)

    def test_identical_errors_do_not_change_f2_marginal(self):
        prediction = torch.full_like(self.target, 2.0)
        alpha_new, stats = calibrate_source_reliability(
            self.alpha, prediction, self.target
        )
        self.assertTrue(torch.allclose(alpha_new, self.alpha, atol=1e-6))
        self.assertAlmostEqual(stats["reliability_min"], 1.0, places=6)
        self.assertAlmostEqual(stats["reliability_max"], 1.0, places=6)

    def test_marginal_shape_must_match_prediction_items(self):
        with self.assertRaises(ValueError):
            calibrate_source_reliability(
                torch.full((5,), 0.2), self.prediction, self.target
            )

    def test_nonfinite_predictions_are_rejected(self):
        prediction = self.prediction.clone()
        prediction[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            calibrate_source_reliability(
                self.alpha, prediction, self.target
            )

    def test_invalid_blend_is_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_source_reliability(
                self.alpha, self.prediction, self.target, blend=1.5
            )


if __name__ == "__main__":
    unittest.main()
