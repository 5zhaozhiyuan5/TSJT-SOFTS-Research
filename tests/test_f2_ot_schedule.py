"""Mathematical and boundary regression tests for H3-motivated OT decay.

Run: python -m unittest discover -s tests -p 'test_f2_ot_schedule.py'
Does not require GPU, GeomLoss or traffic datasets.
"""
import math
import unittest

from model.Meta_Models.f2_ot_schedule import (
    validate_schedule, effective_ot_scale, phase
)


class TestF2OtSchedule(unittest.TestCase):
    def setUp(self):
        self.args = dict(
            base_scale=0.01, daepoch=1500,
            decay_start=2000, decay_end=2999, tail_fraction=0.25,
        )

    def scale(self, mode, epoch):
        return effective_ot_scale(epoch, mode=mode, **self.args)

    def test_t0_matches_original_F2_for_all_active_epochs(self):
        for epoch in (0, 1499, 1500, 1750, 1999, 2000, 2500, 2998, 2999):
            self.assertAlmostEqual(
                self.scale("t0", epoch),
                0.0 if epoch < 1500 else 0.01,
                places=12
            )

    def test_all_schedules_identical_up_to_decay_start(self):
        for epoch in (0, 1499, 1500, 1999, 2000):
            self.assertEqual(self.scale("t0", epoch), self.scale("t1", epoch))
            self.assertEqual(self.scale("t0", epoch), self.scale("t2", epoch))

    def test_t1_linearly_falls_to_exact_zero(self):
        self.assertEqual(self.scale("t1", 2999), 0.0)
        self.assertEqual(self.scale("t1", 3000), 0.0)
        self.assertAlmostEqual(self.scale("t1", 2499),
                               0.01 * (1 - 499 / 999), places=12)
        self.assertGreater(self.scale("t1", 2001), self.scale("t1", 2500))
        self.assertGreater(self.scale("t1", 2500), self.scale("t1", 2998))

    def test_t2_final_coefficient_is_one_quarter(self):
        self.assertAlmostEqual(self.scale("t2", 2999), 0.0025, places=12)
        self.assertAlmostEqual(self.scale("t2", 3001), 0.0025, places=12)
        self.assertAlmostEqual(self.scale("t2", 2499),
                               0.01*(1 - 0.75*499/999), places=12)
        for epoch in (2001, 2250, 2750, 2998):
            self.assertTrue(self.scale("t1", epoch) < self.scale("t2", epoch))
            self.assertTrue(self.scale("t2", epoch) < self.scale("t0", epoch))

    def test_smoke_short_epochs(self):
        kwargs=dict(base_scale=0.01,daepoch=1,decay_start=2,
                    decay_end=3,tail_fraction=0.25)
        self.assertEqual(effective_ot_scale(0,mode="t1",**kwargs),0)
        self.assertEqual(effective_ot_scale(1,mode="t1",**kwargs),0.01)
        self.assertEqual(effective_ot_scale(3,mode="t1",**kwargs),0)
        self.assertAlmostEqual(effective_ot_scale(3,mode="t2",**kwargs),0.0025)

    def test_phase_labels(self):
        a=self.args
        self.assertEqual(phase(0,mode="t0",**{
            k:a[k] for k in ("daepoch","decay_start","decay_end")
        }),"off_before_daepoch")
        self.assertEqual(phase(2000,mode="t2",**{
            k:a[k] for k in ("daepoch","decay_start","decay_end")
        }),"linear_decay")
        self.assertEqual(phase(2999,mode="t2",**{
            k:a[k] for k in ("daepoch","decay_start","decay_end")
        }),"tail")

    def test_legacy_constant_f2_does_not_fail_late_daepoch(self):
        old={**self.args,"daepoch":50000}
        self.assertEqual(
            effective_ot_scale(1000,mode="t0",**old),0.0
        )
        self.assertEqual(
            effective_ot_scale(50000,mode="t0",**old),0.01
        )
        with self.assertRaises(ValueError):
            effective_ot_scale(50000,mode="t1",**old)

    def test_invalid_ranges_and_fractions(self):
        with self.assertRaises(ValueError):
            validate_schedule("t3",**self.args)
        with self.assertRaises(ValueError):
            validate_schedule("t1",**{**self.args,"decay_end":2000})
        with self.assertRaises(ValueError):
            validate_schedule("t1",**{**self.args,"decay_start":1499})
        with self.assertRaises(ValueError):
            validate_schedule("t1",**{**self.args,"tail_fraction":1.1})
        with self.assertRaises(ValueError):
            validate_schedule("t1",**{**self.args,"base_scale":float("nan")})


if __name__ == "__main__":
    unittest.main()
