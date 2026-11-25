"""CPU regression tests for four F2-preserving Stage-I OT alternatives.

Run: python -m unittest discover -s tests -p 'test_gradient_pair_ot.py'
GeomLoss end-to-end custom-cost smoke test runs only if GeomLoss is installed.
"""
import importlib.util
import unittest

import torch

from model.Meta_Models.gradient_pair_ot import (
    VARIANTS, build_pairwise_cost, assign_city_target_groups,
    _head_pair_cos, _response,
)


class GradientPairOTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        ns, nt, width = 18, 12, 16
        self.rs = torch.randn(ns, width, requires_grad=True)
        self.rt = torch.randn(nt, width, requires_grad=True)
        self.ys = torch.randn(ns, 36)
        self.yt = torch.randn(nt, 36)
        self.x = torch.cat((self.rs, self.ys), dim=1)
        self.y = torch.cat((self.rt, self.yt), dim=1)
        self.ds = torch.randn(ns, 36)
        self.dt = torch.randn(nt, 36)
        self.alpha = torch.softmax(torch.randn(ns), dim=0)
        self.hs = torch.randn(ns, 288)
        self.ht = torch.randn(nt, 288)

    def options(self, variant):
        options = {
            "variant": variant, "x": self.x, "y": self.y,
            "source_rep": self.rs, "target_rep": self.rt,
            "source_delta": self.ds, "target_delta": self.dt,
            "source_alpha": self.alpha, "strength": 0.05,
            "anchor_count": 4, "block_size": 5,
        }
        if variant == "p2":
            options.update(
                source_history=self.hs, target_history=self.ht,
                source_future=self.ys, target_future=self.yt
            )
        return options

    def test_four_variants_including_exact_f2_baseline(self):
        self.assertEqual(VARIANTS, ("f2", "p1", "p2", "p3", "p4"))

    def test_rank_one_gradient_cosine(self):
        s = torch.randn(5, 7)
        t = torch.randn(4, 7)
        ds = torch.randn(5, 36)
        dt = torch.randn(4, 36)
        result = _head_pair_cos(s, ds, t, dt)
        outer_s = torch.einsum("nh,nd->nhd", ds, s).flatten(1)
        outer_t = torch.einsum("mh,md->mhd", dt, t).flatten(1)
        direct = torch.nn.functional.normalize(outer_s, dim=1) @ (
            torch.nn.functional.normalize(outer_t, dim=1).T
        )
        self.assertTrue(torch.allclose(result, direct, atol=1e-6))

    def test_dynamic_response_shape_and_history_conditioning(self):
        baseline = _response(self.hs, self.ys)
        altered = _response(self.hs + 2.0, self.ys)
        self.assertEqual(baseline.shape, (len(self.hs), 3, 3))
        self.assertTrue(torch.isfinite(baseline).all())
        self.assertFalse(torch.allclose(baseline, altered))

    def test_pair_cost_shapes_finite_and_backward(self):
        for variant in ("p1", "p2", "p3"):
            with self.subTest(variant=variant):
                cost, report = build_pairwise_cost(**self.options(variant))
                matrix = cost(self.x.unsqueeze(0), self.y.unsqueeze(0))
                baseline = torch.cdist(self.x.unsqueeze(0), self.y.unsqueeze(0))
                self.assertEqual(matrix.shape, (1, len(self.rs), len(self.rt)))
                self.assertTrue(torch.isfinite(matrix).all())
                self.assertGreater((matrix - baseline).abs().max().item(), 0)
                grads = torch.autograd.grad(
                    matrix.mean(), [self.rs, self.rt], retain_graph=True
                )
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertEqual(report["variant"], variant)

    def test_p3_explicit_long_horizon_conflict_penalty(self):
        rs = torch.ones(6, 8, requires_grad=True)
        rt = torch.ones(5, 8, requires_grad=True)
        ds = torch.ones(6, 36)
        dt = torch.ones(5, 36)
        dt[:, 24:] = -1.0
        x = torch.cat([rs, torch.randn(6, 36)], dim=1)
        y = torch.cat([rt, torch.randn(5, 36)], dim=1)
        cost, _ = build_pairwise_cost(
            "p3", x, y, rs, rt, ds, dt,
            torch.full((6,), 1 / 6), strength=0.05,
            block_size=3)
        adjusted = cost(x.unsqueeze(0), y.unsqueeze(0))
        native = torch.cdist(x.unsqueeze(0), y.unsqueeze(0))
        self.assertGreater(float((adjusted - native).mean()), 0)

    def test_p4_matches_known_gradient_permutation(self):
        def group(j):
            z = torch.zeros(1, 4, 3)
            z[..., j] = 1
            delta = torch.ones(1, 4, 36)
            return z, delta
        sources = [group(i) for i in (0, 1, 2)]
        targets = [group(i) for i in (2, 0, 1)]
        ema, counts = {}, {}
        assignment, report = assign_city_target_groups(
            [x[0] for x in sources],
            [x[1] for x in sources],
            [x[0] for x in targets],
            [x[1] for x in targets],
            ["chengdu_m", "shenzhen", "metr-la"],
            ema=ema, counts=counts,
        )
        self.assertEqual(assignment, [1, 2, 0])
        self.assertEqual(len(ema), 9)
        self.assertEqual(sum(counts.values()), 3)
        self.assertEqual(report["assignment"], assignment)

    def test_invalid_settings_fail_loudly(self):
        with self.assertRaises(ValueError):
            build_pairwise_cost(**{**self.options("p2"), "strength": -0.01})
        with self.assertRaises(ValueError):
            assign_city_target_groups([], [], [], [], [])

    @unittest.skipUnless(
        importlib.util.find_spec("geomloss") is not None,
        "GeomLoss is not installed in this CPU environment"
    )
    def test_real_geomloss_custom_cost_has_gradient(self):
        from geomloss import SamplesLoss
        cost, _ = build_pairwise_cost(**self.options("p2"))
        sinkhorn = SamplesLoss(loss="sinkhorn", p=1, backend="tensorized",
                               debias=False, blur=0.001, cost=cost)
        beta = torch.full((len(self.rt),), 1 / len(self.rt))
        loss = sinkhorn(self.alpha, self.x, beta, self.y)
        self.assertTrue(torch.isfinite(loss))
        ds, dt = torch.autograd.grad(loss, (self.rs, self.rt))
        self.assertTrue(torch.isfinite(ds).all())
        self.assertTrue(torch.isfinite(dt).all())


if __name__ == "__main__":
    unittest.main()
