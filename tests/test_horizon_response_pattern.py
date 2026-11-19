"""CPU smoke tests for the stage-I history/future bank and conditional transfer."""
import unittest

import numpy as np
import torch

from model.Meta_Models.response_pattern import (
    build_joint_pattern_bank,
    calibrate_target_scales,
    paired_training_examples,
    patch_descriptors,
)
from model.Meta_Models.pattern_moe import HorizonPatternMoEAlignment
from model.Meta_Models.horizon_transfer import build_future_change_descriptors


class HorizonResponsePatternTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        rng = np.random.default_rng(2)
        # Synthetic CITY-normalized histories and corresponding raw-speed futures.
        self.x = rng.normal(size=(6, 5, 288, 2)).astype(np.float32)
        last = self.x[..., 0, 0]  # only synthetic generation; not used by model
        self.y = 56.0 + 8.0 * rng.normal(size=(6, 5, 36)).astype(np.float32)
        self.mean, self.std = np.array([56.0]), np.array([8.0])
        self.scales = (12, 36, 72, 144, 288)
        self.horizons = (2, 12, 24, 36)

    def build_bank(self):
        return build_joint_pattern_bank(
            {"source_city": self.x},
            {"source_city": self.y},
            {"source_city": self.mean},
            {"source_city": self.std},
            self.scales, self.horizons,
            num_prototypes=3, descriptor_bins=8,
            max_windows_per_city=6, max_nodes_per_city=5,
            max_patches_per_scale=100, seed=5,
        )

    def test_response_bank_shapes_and_source_only(self):
        bank = self.build_bank()
        self.assertEqual(tuple(bank["prototypes"].shape), (5, 3, 8))
        self.assertEqual(tuple(bank["responses"].shape), (5, 3, 4, 4))
        self.assertEqual(tuple(bank["uncertainty"].shape), (5, 3, 4, 4))
        self.assertTrue(torch.isfinite(bank["responses"]).all())
        with self.assertRaises(ValueError):
            build_joint_pattern_bank(
                {"a": self.x}, {"b": self.y}, {"a": self.mean}, {"a": self.std},
                self.scales, self.horizons,
            )

    def test_train_only_calibration_probabilities(self):
        bank = self.build_bank()
        result = calibrate_target_scales(
            bank, self.x, self.y, self.mean, self.std,
            self.scales, self.horizons,
            max_windows=6, max_nodes=5,
        )
        self.assertEqual(tuple(result["scale_weights"].shape), (5, 4))
        self.assertEqual(tuple(result["reliability"].shape), (4,))
        self.assertTrue(torch.allclose(
            result["scale_weights"].sum(0), torch.ones(4), atol=1e-5
        ))
        self.assertTrue(((result["reliability"] >= 0) & (result["reliability"] <= 1)).all())
        self.assertTrue(torch.isfinite(result["relative_errors"]).all())

    def test_response_guidance_backprop_to_prediction(self):
        bank = self.build_bank()
        module = HorizonPatternMoEAlignment(
            bank["prototypes"], input_dim=128, scale_steps=self.scales,
            anchor_steps=self.horizons, router_mode="response",
            response_bank=bank["responses"],
            scale_weights=torch.full((5, 4), 0.2),
            reliability=torch.ones(4),
            sinkhorn_iterations=30,
        )
        hist = torch.from_numpy(self.x[..., 0].reshape(-1, 288))
        predicted = torch.randn(hist.shape[0], 36, requires_grad=True)
        loss = module.response_guidance_loss(hist, predicted)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(predicted.grad)
        self.assertGreater(float(predicted.grad.abs().sum()), 0.0)

    def test_response_mode_alignment_is_finite(self):
        bank = self.build_bank()
        module = HorizonPatternMoEAlignment(
            bank["prototypes"], input_dim=128, scale_steps=self.scales,
            anchor_steps=self.horizons, router_mode="response",
            response_bank=bank["responses"],
            scale_weights=torch.full((5, 4), 0.2),
            reliability=torch.ones(4),
            max_samples=12, sinkhorn_iterations=40,
        )
        n = 12
        x = torch.from_numpy(self.x[..., 0].reshape(-1, 288)[:n])
        future = torch.randn(n, 36)
        last = (x[:, -1] - x.mean(-1)) / x.std(-1, unbiased=False).clamp_min(1e-4)
        desc = build_future_change_descriptors(
            future.unsqueeze(1), last.unsqueeze(1), self.horizons
        ).squeeze(1)
        rep_s = torch.randn(n, 128, requires_grad=True)
        rep_t = torch.randn(n, 128, requires_grad=True)
        m = torch.full((4, n), 1.0/n)
        base = torch.full((n,), 1.0/n)
        loss, diagnostic = module.compute_alignment_loss(
            rep_s, rep_t, x, x, desc, desc, m, m, base, base
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(diagnostic["horizon_losses"].shape), (4,))
        loss.backward()
        self.assertIsNotNone(rep_s.grad)
        self.assertTrue(torch.isfinite(rep_s.grad).all())


if __name__ == "__main__":
    unittest.main()
