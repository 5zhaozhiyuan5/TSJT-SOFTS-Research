"""CPU unit tests for city-pattern predictive-reliability OT constraints."""
import unittest
import torch

from model.Meta_Models.dpm_pattern_attention_ot import SourceEvolutionPatternBank
from model.Meta_Models.city_pattern_reliability_constraint import (
    CityPatternReliabilityConstraint,
)


class TestCityPatternReliabilityConstraint(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.bank = SourceEvolutionPatternBank(patterns=3, kmeans_iters=8)
        base = torch.tensor([
            [[0.0, 0.0, 0.0]] * 3,
            [[4.0, 0.0, 0.0]] * 3,
            [[-4.0, 0.0, 0.0]] * 3,
        ])
        samples = (
            base.repeat_interleave(30, dim=0) +
            torch.randn(90, 3, 3) * 0.03
        )
        sample_quality = torch.full((90, 3), 0.7)
        self.bank.observe(
            samples, max_per_city=90, quality=sample_quality
        )
        self.bank.fit()

        self.cprc = CityPatternReliabilityConstraint(
            shrinkage=1.0, ema=0.2, sample_power=1.0, city_power=1.0
        )
        city_a_response = base.repeat_interleave(20, dim=0)
        city_a_quality = torch.tensor([
            [0.05, 0.05, 0.05],
            [0.90, 0.90, 0.90],
            [0.90, 0.90, 0.90],
        ]).repeat_interleave(20, dim=0)
        city_b_response = base.repeat_interleave(20, dim=0)
        city_b_quality = torch.full((60, 3), 0.95)
        self.cprc.observe(
            city_a_response, city_a_quality, "city_a", max_per_city=60
        )
        self.cprc.observe(
            city_b_response, city_b_quality, "city_b", max_per_city=60
        )
        self.cprc.fit(self.bank, temperature=0.25)

    def test_city_conditioned_pattern_quality_separates_cities(self):
        qa = self.cprc.city_quality["city_a"]
        qb = self.cprc.city_quality["city_b"]
        # The first prototype is poorly forecast in city_a, but not city_b.
        self.assertLess(float(qa[:, 0].mean()), float(qb[:, 0].mean()))
        self.assertLess(
            float(self.cprc.city_weights["city_a"]),
            float(self.cprc.city_weights["city_b"]),
        )

    def test_bad_pattern_receives_less_source_row_mass(self):
        base = torch.tensor([
            [[0.0, 0.0, 0.0]] * 3,
            [[4.0, 0.0, 0.0]] * 3,
            [[-4.0, 0.0, 0.0]] * 3,
        ])
        alpha = torch.full((3,), 1.0 / 3.0)
        adjusted, city_weight, stats = self.cprc.constrain(
            alpha, base, self.bank, "city_a",
            strength=0.8, temperature=0.25
        )
        self.assertAlmostEqual(float(adjusted.sum()), 1.0, places=6)
        self.assertTrue(bool((adjusted >= 0).all()))
        self.assertLess(float(adjusted[0]), float(alpha[0]))
        self.assertGreater(stats["pattern_quality_mass_after"],
                           stats["pattern_quality_mass_before"])
        self.assertGreater(stats["alpha_l1_shift"], 0.0)
        self.assertGreater(city_weight, 0.0)

    def test_zero_strength_is_exact_identity(self):
        alpha = torch.tensor([0.2, 0.3, 0.5])
        response = torch.zeros(3, 3, 3)
        adjusted, weight, stats = self.cprc.constrain(
            alpha, response, self.bank, "city_a", strength=0.0
        )
        self.assertTrue(torch.equal(adjusted, alpha))
        self.assertEqual(stats["alpha_l1_shift"], 0.0)
        self.assertGreater(weight, 0.0)

    def test_online_update_changes_only_reported_city(self):
        before_a = self.cprc.city_quality["city_a"].clone()
        before_b = self.cprc.city_quality["city_b"].clone()
        q_den = torch.ones(3, 3)
        q_num = torch.full((3, 3), 0.95)
        update = self.cprc.update([{
            "q_num": q_num,
            "q_den": q_den,
            "summary": {"city": "city_a"},
        }], momentum=0.2)
        self.assertGreater(
            float((self.cprc.city_quality["city_a"] - before_a).abs().sum()),
            0.0,
        )
        self.assertTrue(torch.equal(
            self.cprc.city_quality["city_b"], before_b
        ))
        self.assertEqual(update["step"], 1)

    def test_missing_city_and_bad_inputs_fail(self):
        alpha = torch.full((3,), 1.0 / 3.0)
        response = torch.zeros(3, 3, 3)
        with self.assertRaises(KeyError):
            self.cprc.constrain(alpha, response, self.bank, "unknown")
        with self.assertRaises(ValueError):
            self.cprc.constrain(alpha, response, self.bank, "city_a",
                                strength=1.5)


if __name__ == "__main__":
    unittest.main()
