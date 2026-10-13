"""Exact joint Gaussian simulation and mortgage pass-through OAS."""

from dataclasses import dataclass

import numpy as np

from .curves import DiscountCurve
from .model import HullWhite
from .oas import solve_oas
from .prepayment import MortgagePool, RefinanceCPR, cpr_to_smm, psa_cpr, weighted_average_life


@dataclass(frozen=True)
class GaussianPaths:
    times: np.ndarray
    short_rates: np.ndarray
    discounts: np.ndarray
    antithetic: bool


def simulate_rates(model: HullWhite, months: int, paths: int, seed: int = 31415,
                    antithetic: bool = True) -> GaussianPaths:
    """Simulate OU endpoints and integrated rates jointly without Euler bias."""
    if months < 1 or paths < 2 or int(months) != months or int(paths) != paths:
        raise ValueError("Positive integer months and at least two paths required")
    if antithetic and paths % 2:
        raise ValueError("Antithetic path count must be even")
    rng = np.random.default_rng(seed)
    dt = 1 / 12
    b = model.b(0, dt)
    vx = model.variance_x(dt)
    vi = model.integral_variance(dt)
    cov = model.covariance_x_integral(dt)
    sx = np.sqrt(vx)
    loading = cov / sx if sx > 0 else 0.0
    sy = np.sqrt(max(vi - loading**2, 0))
    times = np.arange(1, months + 1) * dt
    rates, discounts = np.empty((paths, months)), np.empty((paths, months))
    x, integral = np.zeros(paths), np.zeros(paths)
    count = paths // 2 if antithetic else paths
    for i, time in enumerate(times):
        z = rng.standard_normal((count, 2))
        if antithetic:
            z = np.vstack((z, -z))
        integrated_x = b * x + loading * z[:, 0] + sy * z[:, 1]
        integral += integrated_x + model.integrated_phi(i * dt, time)
        x = np.exp(-model.a * dt) * x + sx * z[:, 0]
        rates[:, i] = x + model.phi(time)
        discounts[:, i] = np.exp(-integral)
    return GaussianPaths(times, rates, discounts, antithetic)


@dataclass(frozen=True)
class MonteCarloEstimate:
    price: float
    standard_error: float
    wal: float
    wal_standard_error: float
    paths: int
    independent_samples: int


@dataclass(frozen=True)
class MBSOASResult:
    oas: float
    price_standard_error: float
    oas_standard_error: float
    wal: float


def paired_samples(values: np.ndarray, antithetic: bool) -> np.ndarray:
    """Use pair averages as independent samples for antithetic uncertainty."""
    values = np.asarray(values)
    if antithetic:
        if len(values) % 2:
            raise ValueError("Even number of antithetic values required")
        half = len(values) // 2
        return (values[:half] + values[half:]) / 2
    return values


class MBSEngine:
    """Cache path cash flows so spread inversion uses common random numbers."""

    def __init__(self, model: HullWhite, pool: MortgagePool, paths: int = 4096,
                 seed: int = 31415, prepayment: RefinanceCPR | None = None,
                 mortgage_spread: float = 0.015, mortgage_beta: float = 1.0,
                 psa_speed: float | None = None, antithetic: bool = True) -> None:
        if antithetic and paths < 4:
            raise ValueError("At least two independent antithetic pairs required for standard errors")
        if mortgage_beta < 0 or not np.isfinite(mortgage_spread) or not np.isfinite(mortgage_beta):
            raise ValueError("Finite mortgage proxy parameters required")
        self.model, self.pool = model, pool
        self.simulation = simulate_rates(model, pool.term_months, paths, seed, antithetic)
        rule = prepayment or RefinanceCPR()
        self.cashflows = np.empty_like(self.simulation.short_rates)
        self.principal = np.empty_like(self.cashflows)
        balance, prepaid_total = np.full(paths, pool.balance), np.zeros(paths)
        for i in range(pool.term_months):
            month = i + 1
            mortgage_rate = np.maximum(mortgage_spread + mortgage_beta * self.simulation.short_rates[:, i], 0.0)
            cpr = (psa_cpr(month + pool.age_months, psa_speed) if psa_speed is not None else
                   rule.cpr(month + pool.age_months, pool.coupon, mortgage_rate, prepaid_total / pool.balance))
            scheduled = pool.scheduled_principal(balance, pool.term_months - i)
            prepaid = (balance - scheduled) * cpr_to_smm(cpr)
            if month == pool.term_months:
                scheduled, prepaid = balance.copy(), np.zeros(paths)
            principal = scheduled + prepaid
            self.principal[:, i] = principal
            self.cashflows[:, i] = principal + balance * pool.net_coupon / 12
            balance = np.maximum(balance - principal, 0.0)
            prepaid_total += prepaid
        self.path_wal = self.principal @ self.simulation.times / pool.balance

    def estimate(self, spread: float = 0.0) -> MonteCarloEstimate:
        """Mean price and WAL, with standard errors based on independent pairs."""
        if not np.isfinite(spread):
            raise ValueError("Spread must be finite")
        values = np.sum(self.cashflows * self.simulation.discounts
                        * np.exp(-spread * self.simulation.times), axis=1)
        price_samples = paired_samples(values, self.simulation.antithetic)
        wal_samples = paired_samples(self.path_wal, self.simulation.antithetic)
        n = len(price_samples)
        price_se = float(np.std(price_samples, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        wal_se = float(np.std(wal_samples, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return MonteCarloEstimate(float(np.mean(values)), price_se, float(np.mean(self.path_wal)),
                                 wal_se, len(values), n)

    def solve_oas(self, market_price: float) -> MBSOASResult:
        """Invert cached Monte Carlo price; apply a delta-method OAS error."""
        spread = solve_oas(lambda s: self.estimate(s).price, market_price)
        estimate = self.estimate(spread)
        epsilon = 0.00001
        derivative = (self.estimate(spread + epsilon).price
                      - self.estimate(spread - epsilon).price) / (2 * epsilon)
        return MBSOASResult(spread, estimate.standard_error,
                            estimate.standard_error / abs(derivative), estimate.wal)


def static_psa_price(pool: MortgagePool, curve: DiscountCurve,
                      spread: float = 0.0, speed: float = 100.0) -> tuple[float, float]:
    """Discount static PSA payments analytically, returning price and WAL."""
    cashflows = pool.static_cashflows(speed)
    times = cashflows["time"].to_numpy()
    price = float(np.dot(cashflows["cashflow"], curve.discount(times) * np.exp(-spread * times)))
    principal = (cashflows["scheduled_principal"] + cashflows["prepaid_principal"]).to_numpy()
    return price, weighted_average_life(principal, times, pool.balance)
