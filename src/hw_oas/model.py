"""Curve-fitted Hull-White model and European closed forms."""

from dataclasses import dataclass
from math import log, sqrt
from typing import Literal

import numpy as np

from .curves import DiscountCurve, par_swap_rate
from .numerics import brent, normal_cdf, normal_pdf

OptionKind = Literal["call", "put"]


@dataclass(frozen=True)
class HullWhite:
    """Short rate r(t)=x(t)+phi(t), with dx=-a*x*dt+sigma*dW."""

    curve: DiscountCurve
    a: float
    sigma: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.a) or not np.isfinite(self.sigma) or self.a <= 0 or self.sigma < 0:
            raise ValueError("Mean reversion must be positive and volatility nonnegative")

    def b(self, time: float, maturity: float) -> float:
        """Bond loading B(t,T)."""
        if time < 0 or maturity < time or not np.all(np.isfinite([time, maturity])):
            raise ValueError("Require 0 <= time <= maturity")
        return float(-np.expm1(-self.a * (maturity - time)) / self.a)

    def variance_x(self, time: float) -> float:
        """Variance of the centered short rate starting at zero."""
        if time < 0:
            raise ValueError("Time must be nonnegative")
        return float(self.sigma**2 * -np.expm1(-2 * self.a * time) / (2 * self.a))

    def covariance_x_integral(self, time: float) -> float:
        """Covariance between x(t) and its integral from zero to t."""
        if time < 0:
            raise ValueError("Time must be nonnegative")
        return float(0.5 * self.sigma**2 * (-np.expm1(-self.a * time) / self.a)**2)

    def integral_variance(self, time: float) -> float:
        """Variance of the integral of x from zero to time."""
        if time < 0:
            raise ValueError("Time must be nonnegative")
        z = self.a * time
        if z < 0.02:
            # The series avoids cancellation of nearly equal exponential terms.
            series = z**3 / 3 - z**4 / 4 + 7 * z**5 / 60 - z**6 / 24 + 31 * z**7 / 2520
            return self.sigma**2 / self.a**3 * series
        return self.sigma**2 / self.a**2 * (time + 2 * np.expm1(-z) / self.a
                                           - np.expm1(-2 * z) / (2 * self.a))

    def phi(self, time: float | np.ndarray) -> float | np.ndarray:
        """Deterministic shift that reproduces the input discount curve."""
        t = np.asarray(time, dtype=float)
        value = self.curve.forward(t) + self.sigma**2 / (2 * self.a**2) * np.expm1(-self.a * t)**2
        return float(value) if t.ndim == 0 else value

    def theta(self, time: float) -> float:
        """Regular drift away from forward-curve knots; jumps are documented."""
        return float(self.a * self.curve.forward(time)
                     + self.sigma**2 / (2 * self.a) * -np.expm1(-2 * self.a * time))

    def integrated_phi(self, start: float, end: float) -> float:
        """Exact deterministic integral, including piecewise forward jumps."""
        if start < 0 or end < start:
            raise ValueError("Require 0 <= start <= end")
        return (log(self.curve.discount(start) / self.curve.discount(end))
                + 0.5 * (self.integral_variance(end) - self.integral_variance(start)))

    def bond_price(self, time: float, maturity: float,
                   short_rate: float | np.ndarray) -> float | np.ndarray:
        """Conditional zero-coupon price given the instantaneous short rate."""
        b = self.b(time, maturity)
        x = np.asarray(short_rate) - self.phi(time)
        out = self.curve.discount(maturity) / self.curve.discount(time) * np.exp(
            -b * (x + self.covariance_x_integral(time)) - 0.5 * b * b * self.variance_x(time))
        return float(out) if np.ndim(out) == 0 else out

    def zero_bond_option(self, expiry: float, maturity: float, strike: float,
                         kind: OptionKind = "call") -> float:
        """Time-zero European option on a unit zero-coupon bond."""
        if kind not in ("call", "put") or strike <= 0 or not np.isfinite(strike):
            raise ValueError("Positive strike and valid option kind required")
        vol = self.b(expiry, maturity) * sqrt(self.variance_x(expiry))
        p, k = self.curve.discount(maturity), strike * self.curve.discount(expiry)
        if vol < 1e-15:
            return max(p - k, 0.0) if kind == "call" else max(k - p, 0.0)
        d1 = log(p / k) / vol + vol / 2
        d2 = d1 - vol
        if kind == "call":
            return p * normal_cdf(d1) - k * normal_cdf(d2)
        return k * normal_cdf(-d2) - p * normal_cdf(-d1)

    def coupon_bond_option(self, expiry: float, dates: np.ndarray, amounts: np.ndarray,
                           strike: float, kind: OptionKind = "call") -> float:
        """Jamshidian decomposition for strictly positive future cash flows."""
        dates, amounts = np.asarray(dates, dtype=float), np.asarray(amounts, dtype=float)
        if (dates.ndim != 1 or amounts.shape != dates.shape or not len(dates)
                or np.any(dates <= expiry) or np.any(amounts <= 0)
                or not np.all(np.isfinite(dates)) or not np.all(np.isfinite(amounts))
                or strike <= 0 or kind not in ("call", "put")):
            raise ValueError("Positive future cash flows and strike required")
        def objective(rate: float) -> float:
            return sum(c * self.bond_price(expiry, t, rate) for t, c in zip(dates, amounts)) - strike
        lo, hi = -0.5, 0.5
        while objective(lo) < 0:
            lo *= 2
        while objective(hi) > 0:
            hi *= 2
        rate = brent(objective, lo, hi)
        return sum(c * self.zero_bond_option(expiry, t, self.bond_price(expiry, t, rate), kind)
                   for t, c in zip(dates, amounts))

    def swaption(self, expiry: float, tenor: int, strike: float | None = None,
                 payer: bool = True, notional: float = 1.0) -> float:
        """European annual-pay physical swaption under a single OIS curve."""
        if expiry < 0 or tenor < 1 or int(tenor) != tenor or notional <= 0:
            raise ValueError("Valid expiry, integer tenor and positive notional required")
        k = par_swap_rate(self.curve, tenor, expiry) if strike is None else strike
        if k <= 0:
            raise ValueError("Jamshidian requires positive fixed coupons")
        dates = expiry + np.arange(1, tenor + 1)
        amounts = np.full(tenor, k)
        amounts[-1] += 1
        return notional * self.coupon_bond_option(expiry, dates, amounts, 1.0, "put" if payer else "call")

    def caplet(self, start: float, end: float, strike: float,
                notional: float = 1.0, floor: bool = False) -> float:
        """Simple-rate caplet or floorlet paid at end, with fixing at start."""
        if end <= start or notional <= 0 or 1 + (end - start) * strike <= 0:
            raise ValueError("Invalid caplet terms")
        factor = 1 + (end - start) * strike
        return notional * factor * self.zero_bond_option(start, end, 1 / factor,
                                                        "call" if floor else "put")


def bachelier_price(forward: float, strike: float, normal_vol: float,
                     expiry: float, annuity: float, payer: bool = True) -> float:
    """Normal-model European swaption value with decimal normal volatility."""
    if normal_vol < 0 or expiry < 0 or annuity <= 0:
        raise ValueError("Invalid Bachelier inputs")
    difference = (forward - strike) * (1 if payer else -1)
    scale = normal_vol * sqrt(expiry)
    if scale == 0:
        return annuity * max(difference, 0.0)
    d = difference / scale
    return annuity * (difference * normal_cdf(d) + scale * normal_pdf(d))
