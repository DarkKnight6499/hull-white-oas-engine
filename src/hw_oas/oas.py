"""Spread inversion and finite-difference risk at a fixed OAS."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .curves import DiscountCurve
from .instruments import FixedRateBond
from .numerics import brent


@dataclass(frozen=True)
class EffectiveRisk:
    price: float
    down_price: float
    up_price: float
    duration: float
    convexity: float


def solve_oas(price: Callable[[float], float], market_price: float,
               bracket: tuple[float, float] = (-0.10, 0.20)) -> float:
    """Solve a constant spread with Brent; reject an unbracketed target."""
    if not np.isfinite(market_price) or market_price <= 0:
        raise ValueError("Positive finite market price required")
    return brent(lambda spread: price(spread) - market_price, *bracket)


def z_spread(bond: FixedRateBond, curve: DiscountCurve, market_price: float) -> float:
    """Spread on promised payments, ignoring exercise optionality."""
    return solve_oas(lambda spread: bond.straight_price(curve, spread), market_price)


def effective_risk(price: Callable[[DiscountCurve, float], float], curve: DiscountCurve,
                   oas: float, bump: float = 0.001) -> EffectiveRisk:
    """Reprice curve bumps at constant OAS, allowing exercise to change."""
    if bump <= 0:
        raise ValueError("Positive risk bump required")
    base = price(curve, oas)
    down = price(curve.parallel_bump(-bump), oas)
    up = price(curve.parallel_bump(bump), oas)
    if base <= 0:
        raise ValueError("Positive base price required")
    return EffectiveRisk(base, down, up, (down - up) / (2 * base * bump),
                         (down + up - 2 * base) / (base * bump**2))


def key_rate_durations(price: Callable[[DiscountCurve, float], float], curve: DiscountCurve,
                       oas: float, keys: np.ndarray, bump: float = 0.0001) -> dict[float, float]:
    """Central differences for a partition of continuous zero-rate bumps."""
    if bump <= 0:
        raise ValueError("Positive risk bump required")
    base = price(curve, oas)
    if base <= 0:
        raise ValueError("Positive base price required")
    return {float(key): (price(curve.key_rate_bump(key, keys, -bump), oas)
                         - price(curve.key_rate_bump(key, keys, bump), oas)) / (2 * base * bump)
            for key in keys}
