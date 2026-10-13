"""Analytic pricing parity, deterministic limits and calibration recovery."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hw_oas.calibration import calibrate, synthetic_swaption_quotes
from hw_oas.curves import par_swap_rate, synthetic_curve
from hw_oas.model import HullWhite, bachelier_price


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.curve = synthetic_curve()
        self.model = HullWhite(self.curve, 0.075, 0.012)

    def test_initial_curve_fit(self):
        for maturity in (0.25, 1, 5, 10, 30):
            self.assertAlmostEqual(self.model.bond_price(0, maturity, self.model.phi(0)),
                                   self.curve.discount(maturity), places=14)

    def test_bond_at_maturity(self):
        self.assertEqual(self.model.bond_price(5, 5, 0.03), 1)

    def test_bond_decreasing_in_short_rate(self):
        self.assertGreater(self.model.bond_price(2, 10, 0.02), self.model.bond_price(2, 10, 0.04))

    def test_zero_bond_parity(self):
        call = self.model.zero_bond_option(2, 10, 0.8)
        put = self.model.zero_bond_option(2, 10, 0.8, "put")
        self.assertAlmostEqual(call - put, self.curve.discount(10) - 0.8 * self.curve.discount(2), places=14)

    def test_zero_bond_option_bounds(self):
        call = self.model.zero_bond_option(2, 10, 0.8)
        self.assertGreaterEqual(call, max(self.curve.discount(10) - 0.8 * self.curve.discount(2), 0))
        self.assertLessEqual(call, self.curve.discount(10))

    def test_volatility_increases_option_value(self):
        lower = HullWhite(self.curve, 0.075, 0.006)
        self.assertGreater(self.model.zero_bond_option(2, 10, 0.8), lower.zero_bond_option(2, 10, 0.8))

    def test_call_decreasing_in_strike(self):
        self.assertGreater(self.model.zero_bond_option(2, 10, 0.7), self.model.zero_bond_option(2, 10, 0.9))

    def test_zero_volatility_intrinsic(self):
        model = HullWhite(self.curve, 0.1, 0)
        self.assertEqual(model.zero_bond_option(2, 10, 0.8),
                         max(self.curve.discount(10) - 0.8 * self.curve.discount(2), 0))

    def test_zero_expiry_intrinsic(self):
        self.assertEqual(self.model.zero_bond_option(0, 10, 0.6), self.curve.discount(10) - 0.6)

    def test_option_maturity_at_expiry(self):
        self.assertAlmostEqual(self.model.zero_bond_option(2, 2, 0.9), 0.1 * self.curve.discount(2), places=14)

    def test_jamshidian_single_payment(self):
        value = self.model.coupon_bond_option(2, np.array([10.0]), np.array([100.0]), 80)
        self.assertAlmostEqual(value, 100 * self.model.zero_bond_option(2, 10, 0.8), places=10)

    def test_coupon_bond_parity(self):
        dates, amounts = np.arange(3, 11), np.full(8, 0.05)
        amounts[-1] += 1
        call = self.model.coupon_bond_option(2, dates, amounts, 1)
        put = self.model.coupon_bond_option(2, dates, amounts, 1, "put")
        self.assertAlmostEqual(call - put, np.dot(amounts, self.curve.discount(dates)) - self.curve.discount(2), places=11)

    def test_coupon_bond_scaling(self):
        dates, amounts = np.array([3, 4]), np.array([0.05, 1.05])
        unit = self.model.coupon_bond_option(2, dates, amounts, 1)
        scaled = self.model.coupon_bond_option(2, dates, 100 * amounts, 100)
        self.assertAlmostEqual(scaled, 100 * unit, places=10)

    def test_swaption_atm_parity(self):
        self.assertAlmostEqual(self.model.swaption(2, 5), self.model.swaption(2, 5, payer=False), places=11)

    def test_swaption_off_atm_parity(self):
        strike = 0.05
        payer = self.model.swaption(2, 5, strike)
        receiver = self.model.swaption(2, 5, strike, payer=False)
        annuity = np.sum(self.curve.discount(2 + np.arange(1, 6)))
        self.assertAlmostEqual(payer - receiver, annuity * (par_swap_rate(self.curve, 5, 2) - strike), places=11)

    def test_cap_floor_parity(self):
        cap = self.model.caplet(2, 2.5, 0.04)
        floor = self.model.caplet(2, 2.5, 0.04, floor=True)
        self.assertAlmostEqual(cap - floor, self.curve.discount(2) - 1.02 * self.curve.discount(2.5), places=14)

    def test_caplet_strike_monotonicity(self):
        self.assertGreater(self.model.caplet(2, 2.5, 0.02), self.model.caplet(2, 2.5, 0.05))

    def test_deterministic_conditional_discount(self):
        model = HullWhite(self.curve, 0.1, 0)
        self.assertAlmostEqual(model.bond_price(2, 10, model.phi(2)),
                               self.curve.discount(10) / self.curve.discount(2), places=14)

    def test_integrated_phi_curve_identity(self):
        for time in (0.001, 0.25, 3, 30):
            self.assertAlmostEqual(np.exp(-self.model.integrated_phi(0, time)
                                          + 0.5 * self.model.integral_variance(time)),
                                   self.curve.discount(time), places=14)

    def test_integrated_phi_additivity(self):
        self.assertAlmostEqual(self.model.integrated_phi(0, 10),
                               self.model.integrated_phi(0, 3) + self.model.integrated_phi(3, 10), places=14)

    def test_integral_variance_independent_quadrature(self):
        nodes, weights = np.polynomial.legendre.leggauss(80)
        for time in (1e-5, 0.1, 0.27, 1, 30):
            points = time * (nodes + 1) / 2
            reference = self.model.sigma**2 * time / 2 * np.dot(weights, (-np.expm1(-self.model.a * points) / self.model.a)**2)
            self.assertLess(abs(self.model.integral_variance(time) / reference - 1), 1e-10)

    def test_theta_phi_relationship_away_from_knots(self):
        time, delta = 4.0, 1e-5
        derivative = (self.model.phi(time + delta) - self.model.phi(time - delta)) / (2 * delta)
        self.assertAlmostEqual(self.model.theta(time), derivative + self.model.a * self.model.phi(time), places=10)

    def test_invalid_model_parameters(self):
        for a, sigma in ((0, 0.01), (-1, 0.01), (0.1, -1), (np.nan, 0.01)):
            with self.subTest(a=a, sigma=sigma), self.assertRaises(ValueError):
                HullWhite(self.curve, a, sigma)

    def test_invalid_jamshidian_cashflows(self):
        with self.assertRaises(ValueError):
            self.model.coupon_bond_option(2, np.array([3]), np.array([-1]), 1)

    def test_bachelier_atm(self):
        self.assertAlmostEqual(bachelier_price(0.04, 0.04, 0.01, 4, 5), 0.1 / np.sqrt(2 * np.pi), places=15)

    def test_bachelier_parity(self):
        payer = bachelier_price(0.04, 0.05, 0.01, 2, 5)
        receiver = bachelier_price(0.04, 0.05, 0.01, 2, 5, False)
        self.assertAlmostEqual(payer - receiver, -0.05, places=14)

    def test_bachelier_zero_vol(self):
        self.assertAlmostEqual(bachelier_price(0.04, 0.03, 0, 2, 5), 0.05, places=14)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curve = synthetic_curve()
        cls.exact = calibrate(cls.curve, synthetic_swaption_quotes(cls.curve, noise=0))
        cls.noisy = calibrate(cls.curve, synthetic_swaption_quotes(cls.curve))

    def test_roundtrip_converges(self):
        self.assertTrue(self.exact.converged)

    def test_roundtrip_parameter_recovery(self):
        self.assertLess(abs(self.exact.model.a - 0.075), 1e-7)
        self.assertLess(abs(self.exact.model.sigma - 0.012), 1e-8)

    def test_roundtrip_price_recovery(self):
        self.assertLess(self.exact.relative_rmse, 1e-8)

    def test_noisy_fit_converges(self):
        self.assertTrue(self.noisy.converged)
        self.assertLess(self.noisy.relative_rmse, 0.05)

    def test_residual_reconciliation(self):
        self.assertAlmostEqual(self.noisy.rmse, np.sqrt(np.mean(self.noisy.residuals**2)), places=14)

    def test_quote_bachelier_reconciliation(self):
        for q in synthetic_swaption_quotes(self.curve):
            forward = par_swap_rate(self.curve, q.tenor, q.expiry)
            annuity = np.sum(self.curve.discount(q.expiry + np.arange(1, q.tenor + 1)))
            self.assertAlmostEqual(q.price, bachelier_price(forward, forward, q.normal_vol, q.expiry, annuity), places=15)

    def test_insufficient_quotes_rejected(self):
        with self.assertRaises(ValueError):
            calibrate(self.curve, [])
