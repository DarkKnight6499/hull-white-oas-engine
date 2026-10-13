"""Discount curves with log-linear interpolation and annual OIS bootstrap."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiscountCurve:
    """Year fractions and positive discount factors, including time zero."""

    times: np.ndarray
    discounts: np.ndarray

    def __post_init__(self) -> None:
        t = np.array(self.times, dtype=float, copy=True)
        p = np.array(self.discounts, dtype=float, copy=True)
        if (t.ndim != 1 or p.shape != t.shape or len(t) < 2
                or not np.all(np.isfinite(t)) or not np.all(np.isfinite(p))
                or t[0] != 0 or p[0] != 1 or np.any(np.diff(t) <= 0)
                or np.any(p <= 0)):
            raise ValueError("Curve requires ordered times, P(0)=1 and positive finite discounts")
        t.setflags(write=False)
        p.setflags(write=False)
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "discounts", p)

    @classmethod
    def from_zero_rates(cls, times: np.ndarray, rates: np.ndarray) -> "DiscountCurve":
        """Construct from continuously compounded zero rates at positive times."""
        t, r = np.asarray(times, dtype=float), np.asarray(rates, dtype=float)
        if t.ndim != 1 or r.shape != t.shape or np.any(t <= 0):
            raise ValueError("Positive maturities and matching zero rates required")
        return cls(np.r_[0.0, t], np.r_[1.0, np.exp(-t * r)])

    @classmethod
    def from_csv(cls, path: str | Path) -> "DiscountCurve":
        """Load columns time and zero_rate, with decimal continuous rates."""
        data = pd.read_csv(path)
        if not {"time", "zero_rate"}.issubset(data.columns):
            raise ValueError("CSV requires time and zero_rate columns")
        return cls.from_zero_rates(data["time"].to_numpy(), data["zero_rate"].to_numpy())

    def discount(self, time: float | np.ndarray) -> float | np.ndarray:
        """Interpolate log discounts, extrapolating with the terminal forward."""
        t = np.asarray(time, dtype=float)
        if np.any(t < 0) or not np.all(np.isfinite(t)):
            raise ValueError("Time must be finite and nonnegative")
        logs = np.log(self.discounts)
        result = np.interp(t, self.times, logs)
        slope = (logs[-1] - logs[-2]) / (self.times[-1] - self.times[-2])
        result = np.where(t > self.times[-1], logs[-1] + slope * (t - self.times[-1]), result)
        out = np.exp(result)
        return float(out) if t.ndim == 0 else out

    def forward(self, time: float | np.ndarray) -> float | np.ndarray:
        """Right-continuous instantaneous forward, constant between knots."""
        t = np.asarray(time, dtype=float)
        if np.any(t < 0) or not np.all(np.isfinite(t)):
            raise ValueError("Time must be finite and nonnegative")
        idx = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 2)
        slopes = -np.diff(np.log(self.discounts)) / np.diff(self.times)
        out = slopes[idx]
        return float(out) if t.ndim == 0 else out

    def zero_rate(self, time: float | np.ndarray) -> float | np.ndarray:
        """Continuous zero rate, using the initial forward at time zero."""
        t = np.asarray(time, dtype=float)
        p = np.asarray(self.discount(t))
        out = np.divide(-np.log(p), t, out=np.full_like(t, self.forward(0)), where=t != 0)
        return float(out) if t.ndim == 0 else out

    def parallel_bump(self, bump: float) -> "DiscountCurve":
        """Shift every continuous zero rate by the same decimal amount."""
        return DiscountCurve(self.times, self.discounts * np.exp(-bump * self.times))

    def key_rate_bump(self, key: float, keys: np.ndarray, bump: float) -> "DiscountCurve":
        """Piecewise-linear key weights with constant end weights."""
        keys = np.asarray(keys, dtype=float)
        if (keys.ndim != 1 or len(keys) < 2 or np.any(np.diff(keys) <= 0)
                or keys[0] <= 0 or key not in keys):
            raise ValueError("Key must belong to ordered positive key maturities")
        weights = np.interp(self.times, keys, (keys == key).astype(float))
        return DiscountCurve(self.times, self.discounts * np.exp(-bump * self.times * weights))


def par_swap_rate(curve: DiscountCurve, maturity: int, start: float = 0.0) -> float:
    """Annual fixed versus idealized compounded OIS, with no payment lag."""
    if maturity < 1 or int(maturity) != maturity or start < 0:
        raise ValueError("Positive integer tenor and nonnegative start required")
    dates = start + np.arange(1, maturity + 1)
    return float((curve.discount(start) - curve.discount(dates[-1])) / np.sum(curve.discount(dates)))


def bootstrap_swaps(rates: np.ndarray) -> DiscountCurve:
    """Bootstrap contiguous annual par swap quotes beginning at one year."""
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 1 or not len(rates) or not np.all(np.isfinite(rates)):
        raise ValueError("Finite contiguous annual swap rates required")
    discounts = [1.0]
    for rate in rates:
        if rate <= -1:
            raise ValueError("Invalid par swap rate")
        p = (1.0 - rate * sum(discounts[1:])) / (1.0 + rate)
        if p <= 0:
            raise ValueError("Swap quotes imply nonpositive discount factor")
        discounts.append(p)
    return DiscountCurve(np.arange(len(discounts), dtype=float), np.array(discounts))


def synthetic_curve(seed: int = 1729) -> DiscountCurve:
    """Seeded synthetic SOFR-style shape, not observed market data."""
    times = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
    rng = np.random.default_rng(seed)
    rates = 0.033 + 0.011 * np.exp(-times / 2.5) + 0.002 * (1 - np.exp(-times / 12))
    rates += rng.normal(0, 0.00012, len(times))
    return DiscountCurve.from_zero_rates(times, rates)
