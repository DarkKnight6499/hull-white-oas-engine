"""Positive-parameter calibration against European swaption prices."""

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from .curves import DiscountCurve, par_swap_rate
from .model import HullWhite, bachelier_price
from .numerics import nelder_mead


@dataclass(frozen=True)
class SwaptionQuote:
    expiry: float
    tenor: int
    normal_vol: float
    price: float


@dataclass(frozen=True)
class CalibrationResult:
    model: HullWhite
    rmse: float
    relative_rmse: float
    iterations: int
    converged: bool
    residuals: np.ndarray


def synthetic_swaption_quotes(curve: DiscountCurve, seed: int = 2718,
                               noise: float = 0.025) -> list[SwaptionQuote]:
    """Generate ATM normal vols from a reference model with seeded quote noise."""
    if noise < 0:
        raise ValueError("Noise must be nonnegative")
    reference = HullWhite(curve, 0.075, 0.012)
    rng = np.random.default_rng(seed)
    quotes = []
    for expiry in (1.0, 2.0, 5.0):
        for tenor in (2, 5, 10):
            forward = par_swap_rate(curve, tenor, expiry)
            annuity = float(np.sum(curve.discount(expiry + np.arange(1, tenor + 1))))
            vol = reference.swaption(expiry, tenor) * sqrt(2 * pi) / (annuity * sqrt(expiry))
            vol *= float(np.exp(rng.normal(0, noise)))
            price = bachelier_price(forward, forward, vol, expiry, annuity)
            quotes.append(SwaptionQuote(expiry, tenor, vol, price))
    return quotes


def calibrate(curve: DiscountCurve, quotes: list[SwaptionQuote],
               initial: tuple[float, float] = (0.12, 0.009)) -> CalibrationResult:
    """Fit a and sigma, minimizing equal-weight relative price residuals."""
    if (len(quotes) < 2 or min(initial) <= 0
            or any(q.expiry <= 0 or q.tenor < 1 or q.price <= 0
                   or not np.isfinite(q.price) for q in quotes)):
        raise ValueError("Positive prices, initial parameters and informative quotes required")
    target = np.array([q.price for q in quotes])
    def residuals(parameters: np.ndarray) -> np.ndarray:
        model = HullWhite(curve, *np.exp(parameters))
        return np.array([model.swaption(q.expiry, q.tenor) for q in quotes]) - target
    def objective(parameters: np.ndarray) -> float:
        if np.any(parameters < np.log([0.001, 0.00001])) or np.any(parameters > np.log([2.0, 0.15])):
            return 1e10 + float(np.sum(parameters**2))
        return float(np.mean((residuals(parameters) / target)**2))
    result = nelder_mead(objective, np.log(initial))
    errors = residuals(result.x)
    return CalibrationResult(HullWhite(curve, *np.exp(result.x)),
                             float(np.sqrt(np.mean(errors**2))),
                             float(np.sqrt(np.mean((errors / target)**2))),
                             result.iterations, result.converged, errors)
