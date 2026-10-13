"""Tree conservation, option convergence, exercise ordering and spread risk."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hw_oas.curves import synthetic_curve
from hw_oas.instruments import FixedRateBond
from hw_oas.model import HullWhite
from hw_oas.oas import effective_risk, key_rate_durations, solve_oas, z_spread
from hw_oas.tree import HullWhiteTree


class TreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curve = synthetic_curve()
        cls.model = HullWhite(cls.curve, 0.075, 0.012)
        cls.tree = HullWhiteTree(cls.model, 10, 200)

    def test_transition_probabilities(self):
        for node in self.tree.levels:
            self.assertTrue(np.all(node.probabilities >= 0))
            np.testing.assert_allclose(node.probabilities.sum(axis=1), 1, atol=1e-15)

    def test_transition_conditional_mean(self):
        for i, node in enumerate(self.tree.levels):
            next_x = self.tree.levels[i + 1].x if i + 1 < self.tree.steps else self.tree.terminal_x
            means = np.sum(node.probabilities * next_x[node.branches], axis=1)
            np.testing.assert_allclose(means, node.x * np.exp(-self.model.a * self.tree.dt), atol=1e-15)

    def test_transition_conditional_variance(self):
        for i, node in enumerate(self.tree.levels):
            next_x = self.tree.levels[i + 1].x if i + 1 < self.tree.steps else self.tree.terminal_x
            outcomes = next_x[node.branches]
            mean = np.sum(node.probabilities * outcomes, axis=1)
            variance = np.sum(node.probabilities * (outcomes - mean[:, None])**2, axis=1)
            np.testing.assert_allclose(variance, self.model.variance_x(self.tree.dt), atol=1e-15)

    def test_state_price_conservation(self):
        for time, node in zip(self.tree.times, self.tree.levels):
            self.assertAlmostEqual(node.state_prices.sum(), self.curve.discount(time), places=13)
        self.assertAlmostEqual(self.tree.terminal_state_prices.sum(), self.curve.discount(10), places=13)

    def test_zero_bond_curve_repricing(self):
        for time in (0, 0.25, 0.5, 1, 2, 5, 10):
            self.assertLess(abs(self.tree.zero_bond(time) / self.curve.discount(time) - 1), 1e-6)

    def test_spread_discount_identity(self):
        for time in (1, 2, 10):
            self.assertAlmostEqual(self.tree.zero_bond(time, 0.003), self.curve.discount(time) * np.exp(-0.003 * time), places=13)

    def test_straight_bond_reconciliation(self):
        bond = FixedRateBond(10, 0.045)
        self.assertAlmostEqual(self.tree.bond_price(bond, 0.002), bond.straight_price(self.curve, 0.002), places=10)

    def test_callable_below_straight(self):
        bond = FixedRateBond(10, 0.045, calls={2: 100, 4: 100, 6: 100, 8: 100})
        self.assertLess(self.tree.bond_price(bond), bond.straight_price(self.curve))

    def test_putable_above_straight(self):
        bond = FixedRateBond(10, 0.025, puts={2: 100, 4: 100, 6: 100, 8: 100})
        self.assertGreater(self.tree.bond_price(bond), bond.straight_price(self.curve))

    def test_more_calls_reduce_price(self):
        first = FixedRateBond(10, 0.045, calls={2: 100})
        second = FixedRateBond(10, 0.045, calls={2: 100, 4: 100})
        self.assertLessEqual(self.tree.bond_price(second), self.tree.bond_price(first))

    def test_higher_call_redemption_increases_price(self):
        lower = FixedRateBond(10, 0.045, calls={2: 100})
        higher = FixedRateBond(10, 0.045, calls={2: 105})
        self.assertGreaterEqual(self.tree.bond_price(higher), self.tree.bond_price(lower))

    def test_one_call_equals_straight_minus_european(self):
        bond = FixedRateBond(10, 0.045, calls={2: 100})
        dates, amounts = bond.cashflows()
        future = dates > 2
        option = self.tree.coupon_bond_option(dates[future], amounts[future], 100, [2])
        self.assertAlmostEqual(self.tree.bond_price(bond), bond.straight_price(self.curve) - option, places=10)

    def test_one_put_equals_straight_plus_european(self):
        bond = FixedRateBond(10, 0.025, puts={2: 100})
        dates, amounts = bond.cashflows()
        future = dates > 2
        option = self.tree.coupon_bond_option(dates[future], amounts[future], 100, [2], "put")
        self.assertAlmostEqual(self.tree.bond_price(bond), bond.straight_price(self.curve) + option, places=10)

    def test_european_zero_option_convergence(self):
        strike = self.curve.discount(10) / self.curve.discount(2)
        analytic = self.model.zero_bond_option(2, 10, strike)
        errors = []
        for steps in (40, 640):
            tree = HullWhiteTree(self.model, 10, steps)
            price = tree.coupon_bond_option(np.array([10]), np.array([1]), strike, [2])
            errors.append(abs(price - analytic))
        self.assertLess(errors[-1], errors[0] / 4)
        self.assertLess(errors[-1], 5e-5)

    def test_european_coupon_option_convergence(self):
        dates = np.arange(3, 11, dtype=float)
        amounts = np.full(8, 0.04)
        amounts[-1] += 1
        analytic = self.model.coupon_bond_option(2, dates, amounts, 1)
        tree = HullWhiteTree(self.model, 10, 640)
        price = tree.coupon_bond_option(dates, amounts, 1, [2])
        self.assertLess(abs(price - analytic), 8e-5)

    def test_bermudan_above_european(self):
        dates, amounts = np.array([10]), np.array([1])
        european = self.tree.coupon_bond_option(dates, amounts, 0.8, [2])
        bermudan = self.tree.coupon_bond_option(dates, amounts, 0.8, [1, 2])
        self.assertGreaterEqual(bermudan, european)

    def test_bermudan_with_intervening_coupons(self):
        dates = np.arange(1, 11, dtype=float)
        amounts = np.full(10, 0.04)
        amounts[-1] += 1
        bermudan = self.tree.coupon_bond_option(dates, amounts, 1, [2, 4, 6])
        for expiry in (2, 4, 6):
            future = dates > expiry
            european = self.tree.coupon_bond_option(dates[future], amounts[future], 1, [expiry])
            self.assertGreaterEqual(bermudan, european)

    def test_exercise_excludes_current_coupon(self):
        dates = np.arange(1, 11, dtype=float)
        amounts = np.full(10, 0.04)
        amounts[-1] += 1
        supplied = self.tree.coupon_bond_option(dates, amounts, 1, [2])
        future = dates > 2
        excluded = self.tree.coupon_bond_option(dates[future], amounts[future], 1, [2])
        self.assertAlmostEqual(supplied, excluded, places=14)

    def test_tree_option_parity(self):
        dates, amounts = np.array([10]), np.array([1])
        call = self.tree.coupon_bond_option(dates, amounts, 0.8, [2])
        put = self.tree.coupon_bond_option(dates, amounts, 0.8, [2], "put")
        self.assertAlmostEqual(call - put, self.curve.discount(10) - 0.8 * self.curve.discount(2), places=12)

    def test_zero_volatility_tree(self):
        tree = HullWhiteTree(HullWhite(self.curve, 0.1, 0), 10, 100)
        self.assertAlmostEqual(tree.zero_bond(10), self.curve.discount(10), places=13)
        bond = FixedRateBond(10, 0.06, calls={2: 100})
        expected = 100 * self.curve.discount(2) + sum(3 * self.curve.discount(t) for t in (0.5, 1, 1.5, 2))
        self.assertAlmostEqual(tree.bond_price(bond), expected, places=10)

    def test_off_grid_event_rejected(self):
        with self.assertRaises(ValueError):
            self.tree.zero_bond(1.023)

    def test_invalid_bond_schedule(self):
        with self.assertRaises(ValueError):
            FixedRateBond(10, 0.04, calls={1.1: 100})

    def test_call_put_priority_required(self):
        with self.assertRaises(ValueError):
            FixedRateBond(10, 0.04, calls={2: 100}, puts={2: 100})


class OASTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curve = synthetic_curve()
        cls.model = HullWhite(cls.curve, 0.075, 0.012)
        cls.bond = FixedRateBond(10, 0.045, calls={t: 100 for t in np.arange(2, 10, 0.5)})
        cls.tree = HullWhiteTree(cls.model, 10, 200)

    def test_known_oas_recovery(self):
        market = self.tree.bond_price(self.bond, 0.0025)
        spread = solve_oas(lambda s: self.tree.bond_price(self.bond, s), market)
        self.assertAlmostEqual(spread, 0.0025, places=10)

    def test_callable_oas_less_than_z_spread(self):
        market = self.tree.bond_price(self.bond, 0.0025)
        self.assertGreater(z_spread(self.bond, self.curve, market), 0.0025)

    def test_straight_z_equals_oas(self):
        bond = FixedRateBond(10, 0.04)
        market = self.tree.bond_price(bond, 0.001)
        self.assertAlmostEqual(z_spread(bond, self.curve, market), 0.001, places=10)

    def test_price_decreasing_in_spread(self):
        self.assertGreater(self.tree.bond_price(self.bond, 0), self.tree.bond_price(self.bond, 0.005))

    def test_negative_oas_recovery(self):
        market = self.tree.bond_price(self.bond, -0.003)
        spread = solve_oas(lambda s: self.tree.bond_price(self.bond, s), market)
        self.assertAlmostEqual(spread, -0.003, places=10)

    def test_unbracketed_market_rejected(self):
        with self.assertRaises(ValueError):
            solve_oas(lambda s: self.tree.bond_price(self.bond, s), 10000)

    def test_nonpositive_market_rejected(self):
        with self.assertRaises(ValueError):
            solve_oas(lambda s: 100, 0)

    def test_zero_coupon_duration_convexity(self):
        def price(curve, spread):
            return curve.discount(10) * np.exp(-spread * 10)
        risk = effective_risk(price, self.curve, 0.003, 1e-5)
        self.assertAlmostEqual(risk.duration, 10, places=6)
        self.assertAlmostEqual(risk.convexity, 100, places=4)

    def test_callable_negative_convexity(self):
        def price(curve, spread):
            return HullWhiteTree(HullWhite(curve, self.model.a, self.model.sigma), 10, 400).bond_price(self.bond, spread)
        risk = effective_risk(price, self.curve, 0.0025)
        self.assertLess(risk.convexity, 0)
        self.assertGreater(risk.duration, 0)

    def test_key_rate_duration_additivity(self):
        bond = FixedRateBond(10, 0.04)
        def price(curve, spread):
            return bond.straight_price(curve, spread)
        keys = np.array([1, 2, 5, 10, 20, 30])
        durations = key_rate_durations(price, self.curve, 0.002, keys, 1e-5)
        parallel = effective_risk(price, self.curve, 0.002, 1e-5).duration
        self.assertLess(abs(sum(durations.values()) - parallel), 1e-7)

    def test_straight_duration_cashflow_benchmark(self):
        bond = FixedRateBond(10, 0.04)
        dates, amounts = bond.cashflows()
        present_values = amounts * self.curve.discount(dates)
        analytic = np.dot(dates, present_values) / present_values.sum()
        risk = effective_risk(lambda c, s: bond.straight_price(c, s), self.curve, 0, 1e-5)
        self.assertAlmostEqual(risk.duration, analytic, places=6)

    def test_invalid_risk_bump(self):
        with self.assertRaises(ValueError):
            effective_risk(lambda c, s: 100, self.curve, 0, 0)
