"""Mortgage conservation and Monte Carlo Gaussian and pricing benchmarks."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hw_oas.curves import synthetic_curve
from hw_oas.mbs import MBSEngine, paired_samples, simulate_rates, static_psa_price
from hw_oas.model import HullWhite
from hw_oas.oas import effective_risk
from hw_oas.prepayment import MortgagePool, RefinanceCPR, cpr_to_smm, psa_cpr, weighted_average_life


class PrepaymentTests(unittest.TestCase):
    def test_psa_endpoints(self):
        self.assertEqual(psa_cpr(1), 0.002)
        self.assertEqual(psa_cpr(30), 0.06)
        self.assertEqual(psa_cpr(360), 0.06)

    def test_psa_speed_scaling(self):
        self.assertEqual(psa_cpr(10, 200), 2 * psa_cpr(10, 100))

    def test_psa_capped_at_probability_one(self):
        self.assertEqual(psa_cpr(30, 2000), 1)

    def test_smm_cpr_identity(self):
        rates = np.array([0, 0.01, 0.06, 0.25, 1])
        np.testing.assert_allclose(1 - (1 - cpr_to_smm(rates))**12, rates, atol=1e-14)

    def test_refinance_incentive_monotonicity(self):
        rule = RefinanceCPR()
        self.assertGreater(rule.cpr(30, 0.055, 0.03), rule.cpr(30, 0.055, 0.05))

    def test_refinance_seasoning(self):
        rule = RefinanceCPR()
        self.assertAlmostEqual(rule.cpr(1, 0.055, 0.04) * 30, rule.cpr(30, 0.055, 0.04), places=14)

    def test_refinance_burnout(self):
        rule = RefinanceCPR()
        self.assertGreater(rule.cpr(30, 0.055, 0.03, 0), rule.cpr(30, 0.055, 0.03, 0.5))

    def test_no_incentive_base_only(self):
        rule = RefinanceCPR()
        self.assertEqual(rule.cpr(30, 0.055, 0.07), rule.base_cpr)

    def test_invalid_cpr(self):
        for rate in (-0.01, 1.01, np.nan):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                cpr_to_smm(rate)

    def test_invalid_refinance_parameters(self):
        with self.assertRaises(ValueError):
            RefinanceCPR(base_cpr=0.9, incentive_cpr=0.2)

    def test_amortization_total_principal(self):
        pool = MortgagePool()
        cashflows = pool.static_cashflows(0)
        self.assertAlmostEqual(cashflows.scheduled_principal.sum(), pool.balance, places=10)
        self.assertEqual(cashflows.ending_balance.iloc[-1], 0)

    def test_constant_payment_without_prepayment(self):
        pool = MortgagePool(servicing_fee=0, guarantee_fee=0)
        cashflows = pool.static_cashflows(0)
        np.testing.assert_allclose(cashflows.cashflow, cashflows.cashflow.iloc[0], rtol=1e-12)

    def test_psa_principal_conservation(self):
        pool = MortgagePool()
        cashflows = pool.static_cashflows()
        self.assertAlmostEqual((cashflows.scheduled_principal + cashflows.prepaid_principal).sum(), pool.balance, places=10)

    def test_balance_reconciliation(self):
        cashflows = MortgagePool().static_cashflows()
        np.testing.assert_allclose(cashflows.opening_balance - cashflows.scheduled_principal
                                   - cashflows.prepaid_principal, cashflows.ending_balance, atol=1e-13)
        np.testing.assert_allclose(cashflows.ending_balance[:-1], cashflows.opening_balance[1:], atol=1e-13)

    def test_zero_coupon_amortization(self):
        pool = MortgagePool(coupon=0, servicing_fee=0, guarantee_fee=0, term_months=12)
        cashflows = pool.static_cashflows(0)
        np.testing.assert_allclose(cashflows.cashflow, 100 / 12, atol=1e-12)

    def test_full_prepayment_first_month(self):
        pool = MortgagePool()
        cashflows = pool.static_cashflows(50000)
        self.assertAlmostEqual(cashflows.scheduled_principal.iloc[0] + cashflows.prepaid_principal.iloc[0], 100)
        self.assertTrue(np.all(cashflows.ending_balance == 0))

    def test_fee_strip_interest(self):
        pool = MortgagePool()
        cashflows = pool.static_cashflows()
        np.testing.assert_allclose(cashflows.interest, cashflows.opening_balance * pool.net_coupon / 12)

    def test_faster_psa_shortens_wal(self):
        pool, curve = MortgagePool(), synthetic_curve()
        self.assertLess(static_psa_price(pool, curve, speed=200)[1], static_psa_price(pool, curve, speed=100)[1])

    def test_wal_single_payment(self):
        self.assertEqual(weighted_average_life(np.array([100]), np.array([5]), 100), 5)

    def test_invalid_fee_strip(self):
        with self.assertRaises(ValueError):
            MortgagePool(coupon=0.001)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curve = synthetic_curve()
        cls.model = HullWhite(cls.curve, 0.075, 0.012)
        cls.paths = simulate_rates(cls.model, 60, 20000)

    def test_antithetic_short_rate_centering(self):
        half = len(self.paths.short_rates) // 2
        centers = (self.paths.short_rates[:half] + self.paths.short_rates[half:]) / 2
        np.testing.assert_allclose(centers, self.model.phi(self.paths.times)[None, :] + np.zeros_like(centers), atol=1e-14)

    def test_short_rate_variance(self):
        x = self.paths.short_rates[:, -1] - self.model.phi(5)
        expected = self.model.variance_x(5)
        independent = len(x) // 2
        standard_error = expected * np.sqrt(2 / independent)
        self.assertLess(abs(np.mean(x*x) - expected), 6 * standard_error)

    def test_integrated_rate_variance(self):
        integral = -np.log(self.paths.discounts[:, -1]) - self.model.integrated_phi(0, 5)
        expected = self.model.integral_variance(5)
        standard_error = expected * np.sqrt(2 / (len(integral) // 2))
        self.assertLess(abs(np.mean(integral**2) - expected), 6 * standard_error)

    def test_endpoint_integral_covariance(self):
        x = self.paths.short_rates[:, -1] - self.model.phi(5)
        integral = -np.log(self.paths.discounts[:, -1]) - self.model.integrated_phi(0, 5)
        products = paired_samples(x * integral, True)
        standard_error = products.std(ddof=1) / np.sqrt(len(products))
        self.assertLess(abs(products.mean() - self.model.covariance_x_integral(5)), 6 * standard_error)

    def test_monte_carlo_discount_curve(self):
        for month in (12, 36, 60):
            values = paired_samples(self.paths.discounts[:, month - 1], True)
            standard_error = values.std(ddof=1) / np.sqrt(len(values))
            self.assertLess(abs(values.mean() - self.curve.discount(month / 12)), 6 * standard_error)

    def test_discounted_bond_martingale(self):
        values = self.paths.discounts[:, -1] * self.model.bond_price(5, 10, self.paths.short_rates[:, -1])
        samples = paired_samples(values, True)
        standard_error = samples.std(ddof=1) / np.sqrt(len(samples))
        self.assertLess(abs(samples.mean() - self.curve.discount(10)), 6 * standard_error)

    def test_monte_carlo_zero_bond_option_closed_form(self):
        payoff = np.maximum(self.model.bond_price(5, 10, self.paths.short_rates[:, -1]) - 0.85, 0)
        values = paired_samples(self.paths.discounts[:, -1] * payoff, True)
        standard_error = values.std(ddof=1) / np.sqrt(len(values))
        analytic = self.model.zero_bond_option(5, 10, 0.85)
        self.assertLess(abs(values.mean() - analytic), 6 * standard_error)

    def test_seed_reproducibility(self):
        first = simulate_rates(self.model, 12, 20)
        second = simulate_rates(self.model, 12, 20)
        np.testing.assert_array_equal(first.discounts, second.discounts)

    def test_deterministic_simulation(self):
        paths = simulate_rates(HullWhite(self.curve, 0.1, 0), 12, 4)
        np.testing.assert_allclose(paths.discounts, np.broadcast_to(self.curve.discount(paths.times), paths.discounts.shape), atol=1e-14)

    def test_odd_antithetic_count_rejected(self):
        with self.assertRaises(ValueError):
            simulate_rates(self.model, 12, 5)

    def test_pair_standard_error_inputs(self):
        np.testing.assert_array_equal(paired_samples(np.array([1, 3, 5, 7]), True), [3, 5])


class MBSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curve = synthetic_curve()
        cls.model = HullWhite(cls.curve, 0.075, 0.012)
        cls.pool = MortgagePool(term_months=120)
        cls.engine = MBSEngine(cls.model, cls.pool, paths=2048)

    def test_path_principal_conservation(self):
        np.testing.assert_allclose(self.engine.principal.sum(axis=1), self.pool.balance, atol=1e-10)

    def test_positive_cashflows(self):
        self.assertTrue(np.all(self.engine.cashflows >= 0))

    def test_wal_bounds(self):
        self.assertTrue(np.all(self.engine.path_wal >= 1 / 12))
        self.assertTrue(np.all(self.engine.path_wal <= 10))

    def test_spread_monotonicity(self):
        self.assertGreater(self.engine.estimate(0).price, self.engine.estimate(0.005).price)

    def test_known_oas_recovery(self):
        market = self.engine.estimate(0.003).price
        result = self.engine.solve_oas(market)
        self.assertAlmostEqual(result.oas, 0.003, places=10)
        self.assertGreater(result.oas_standard_error, 0)

    def test_oas_repricing(self):
        market = self.engine.estimate(0.003).price + 0.1
        result = self.engine.solve_oas(market)
        self.assertLess(abs(self.engine.estimate(result.oas).price - market), 1e-8)

    def test_standard_error_pair_reconciliation(self):
        path_values = np.sum(self.engine.cashflows * self.engine.simulation.discounts, axis=1)
        samples = paired_samples(path_values, True)
        expected = np.std(samples, ddof=1) / np.sqrt(len(samples))
        self.assertAlmostEqual(self.engine.estimate().standard_error, expected, places=14)

    def test_independent_pair_count(self):
        self.assertEqual(self.engine.estimate().independent_samples, 1024)

    def test_zero_vol_static_psa_price(self):
        deterministic = HullWhite(self.curve, 0.075, 0)
        engine = MBSEngine(deterministic, self.pool, paths=4, psa_speed=100)
        price, wal = static_psa_price(self.pool, self.curve, 0.002)
        self.assertAlmostEqual(engine.estimate(0.002).price, price, places=10)
        self.assertAlmostEqual(engine.estimate().wal, wal, places=10)
        self.assertEqual(engine.estimate().standard_error, 0)

    def test_stochastic_static_psa_benchmark(self):
        engine = MBSEngine(self.model, self.pool, paths=8192, psa_speed=100)
        price, wal = static_psa_price(self.pool, self.curve, 0.002)
        estimate = engine.estimate(0.002)
        self.assertLess(abs(estimate.price - price), 6 * estimate.standard_error)
        self.assertAlmostEqual(estimate.wal, wal, places=10)

    def test_fee_strip_reduces_price(self):
        gross_pool = MortgagePool(term_months=120, servicing_fee=0, guarantee_fee=0)
        gross = MBSEngine(self.model, gross_pool, paths=2048)
        self.assertGreater(gross.estimate().price, self.engine.estimate().price)
        np.testing.assert_array_equal(gross.principal, self.engine.principal)

    def test_mbs_price_additivity(self):
        double = MBSEngine(self.model, MortgagePool(balance=200, term_months=120), paths=2048)
        self.assertAlmostEqual(double.estimate().price, 2 * self.engine.estimate().price, places=10)
        self.assertAlmostEqual(double.estimate().wal, self.engine.estimate().wal, places=10)

    def test_static_psa_risk_positive_convexity(self):
        risk = effective_risk(lambda c, s: static_psa_price(self.pool, c, s)[0], self.curve, 0.003)
        self.assertGreater(risk.duration, 0)
        self.assertGreater(risk.convexity, 0)

    def test_wal_independent_of_discount_spread(self):
        self.assertEqual(self.engine.estimate(0).wal, self.engine.estimate(0.01).wal)

    def test_plain_sampling_supported(self):
        engine = MBSEngine(self.model, self.pool, paths=101, antithetic=False)
        self.assertEqual(engine.estimate().independent_samples, 101)
        self.assertGreater(engine.estimate().standard_error, 0)

    def test_insufficient_pairs_rejected(self):
        with self.assertRaises(ValueError):
            MBSEngine(self.model, self.pool, paths=2)
