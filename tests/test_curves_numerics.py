"""Curve construction and independent scalar numerical benchmarks."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hw_oas.curves import DiscountCurve, bootstrap_swaps, par_swap_rate, synthetic_curve
from hw_oas.numerics import brent, nelder_mead, normal_cdf, normal_pdf


class CurveTests(unittest.TestCase):
    def setUp(self):
        self.curve = synthetic_curve()

    def test_origin_discount(self):
        self.assertEqual(self.curve.discount(0), 1)

    def test_node_reconciliation(self):
        np.testing.assert_allclose(self.curve.discount(self.curve.times), self.curve.discounts, atol=1e-15)

    def test_log_linear_midpoint(self):
        expected = np.sqrt(self.curve.discount(3) * self.curve.discount(5))
        self.assertAlmostEqual(self.curve.discount(4), expected, places=15)

    def test_constant_curve(self):
        curve = DiscountCurve.from_zero_rates(np.array([1, 5, 30]), np.full(3, 0.04))
        for time in (0, 0.7, 10, 40):
            self.assertAlmostEqual(curve.discount(time), np.exp(-0.04 * time), places=14)

    def test_forward_integral(self):
        time = 4.0
        delta = 0.001
        finite_difference = -np.log(self.curve.discount(time + delta) / self.curve.discount(time)) / delta
        self.assertAlmostEqual(finite_difference, self.curve.forward(time), places=12)

    def test_terminal_extrapolation(self):
        self.assertAlmostEqual(self.curve.discount(35) / self.curve.discount(30),
                               np.exp(-5 * self.curve.forward(30)), places=14)

    def test_zero_rate_origin(self):
        self.assertEqual(self.curve.zero_rate(0), self.curve.forward(0))

    def test_seed_reproducibility(self):
        np.testing.assert_array_equal(self.curve.discounts, synthetic_curve().discounts)

    def test_different_seeds(self):
        self.assertFalse(np.array_equal(self.curve.discounts, synthetic_curve(99).discounts))

    def test_positive_forwards(self):
        self.assertTrue(np.all(self.curve.forward(np.arange(0, 40, 0.1)) > 0))

    def test_negative_rates_supported(self):
        curve = DiscountCurve.from_zero_rates(np.array([1, 2]), np.array([-0.01, -0.01]))
        self.assertGreater(curve.discount(2), 1)

    def test_parallel_shift_identity(self):
        times = np.array([0, 0.25, 4, 12, 40])
        np.testing.assert_allclose(self.curve.parallel_bump(0.001).discount(times),
                                   self.curve.discount(times) * np.exp(-0.001 * times), rtol=1e-14)

    def test_key_weights_partition(self):
        keys = np.array([1, 2, 5, 10, 30])
        logs = sum(np.log(self.curve.key_rate_bump(k, keys, 0.001).discounts / self.curve.discounts) for k in keys)
        np.testing.assert_allclose(logs, -0.001 * self.curve.times, atol=5e-16)

    def test_swap_repricing(self):
        quotes = np.array([par_swap_rate(self.curve, t) for t in range(1, 31)])
        curve = bootstrap_swaps(quotes)
        for t, quote in enumerate(quotes, 1):
            self.assertLess(abs(par_swap_rate(curve, t) - quote), 1e-10)

    def test_swap_pv_is_zero(self):
        for tenor in (1, 3, 10, 30):
            fixed = par_swap_rate(self.curve, tenor) * np.sum(self.curve.discount(np.arange(1, tenor + 1)))
            self.assertAlmostEqual(fixed - 1 + self.curve.discount(tenor), 0, places=14)

    def test_forward_swap_par(self):
        dates = 2.5 + np.arange(1, 6)
        fixed = par_swap_rate(self.curve, 5, 2.5) * np.sum(self.curve.discount(dates))
        self.assertAlmostEqual(fixed, self.curve.discount(2.5) - self.curve.discount(7.5), places=14)

    def test_csv_roundtrip(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "curve.csv"
            path.write_text("time,zero_rate\n1,0.04\n2,0.035\n", encoding="utf-8")
            curve = DiscountCurve.from_csv(path)
            np.testing.assert_allclose(curve.discount(np.array([1, 2])), np.exp(-np.array([0.04, 0.07])))

    def test_invalid_curves(self):
        for times, discounts in (([0, 1, 1], [1, 0.9, 0.8]), ([0, 1], [1, -1]),
                                 ([0, 1], [0.99, 0.9]), ([0, 1], [1, np.nan])):
            with self.subTest(times=times, discounts=discounts), self.assertRaises(ValueError):
                DiscountCurve(np.array(times), np.array(discounts))

    def test_invalid_times(self):
        for time in (-1, np.nan, np.inf):
            with self.subTest(time=time), self.assertRaises(ValueError):
                self.curve.discount(time)

    def test_inconsistent_swaps_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap_swaps(np.array([0.01, 100.0, 100.0]))

    def test_curve_arrays_immutable(self):
        with self.assertRaises(ValueError):
            self.curve.discounts[1] = 0.5


class NumericalTests(unittest.TestCase):
    def test_cdf_symmetry(self):
        for value in (0, 0.1, 1, 3, 8):
            self.assertAlmostEqual(normal_cdf(value) + normal_cdf(-value), 1, places=15)

    def test_cdf_at_zero(self):
        self.assertEqual(normal_cdf(0), 0.5)

    def test_pdf_at_zero(self):
        self.assertAlmostEqual(normal_pdf(0), 1 / np.sqrt(2 * np.pi), places=15)

    def test_brent_quadratic(self):
        self.assertAlmostEqual(brent(lambda x: x*x - 2, 0, 2), np.sqrt(2), places=11)

    def test_brent_reversed_sign(self):
        self.assertAlmostEqual(brent(lambda x: 2 - x*x, 0, 2), np.sqrt(2), places=11)

    def test_brent_flat_root(self):
        self.assertAlmostEqual(brent(lambda x: (x - 0.3)**3, -1, 1), 0.3, places=10)

    def test_brent_endpoint(self):
        self.assertEqual(brent(lambda x: x, 0, 1), 0)

    def test_brent_unbracketed(self):
        with self.assertRaises(ValueError):
            brent(lambda x: x*x + 1, -1, 1)

    def test_brent_nonfinite(self):
        with self.assertRaises(ValueError):
            brent(lambda x: np.nan, 0, 1)

    def test_brent_iteration_limit(self):
        with self.assertRaises(RuntimeError):
            brent(lambda x: x*x - 2, 0, 2, max_iter=1)

    def test_simplex_quadratic(self):
        result = nelder_mead(lambda x: float((x[0] - 2)**2 + 3 * (x[1] + 1)**2), np.array([0., 0.]))
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.x, [2, -1], atol=1e-8)

    def test_simplex_rosenbrock(self):
        result = nelder_mead(lambda x: float((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2),
                             np.array([-1.2, 1.0]))
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.x, [1, 1], atol=1e-7)
