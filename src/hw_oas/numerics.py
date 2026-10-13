"""Dependency-free scalar roots, normal probabilities and optimization."""

from dataclasses import dataclass
from math import erf, exp, isfinite, pi, sqrt
from typing import Callable

import numpy as np


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normal_pdf(x: float) -> float:
    """Standard normal density."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def brent(f: Callable[[float], float], left: float, right: float,
          xtol: float = 1e-12, max_iter: int = 200) -> float:
    """Bracketed Brent root with interpolation and bisection safeguards."""
    if not left < right or xtol <= 0 or max_iter < 1:
        raise ValueError("Invalid root solver settings")
    a, b = left, right
    fa, fb = f(a), f(b)
    if not isfinite(fa) or not isfinite(fb):
        raise ValueError("Nonfinite endpoint value")
    if fa == 0:
        return a
    if fb == 0:
        return b
    if (fa > 0) == (fb > 0):
        raise ValueError("Root is not bracketed")
    if abs(fa) < abs(fb):
        a, b, fa, fb = b, a, fb, fa
    c, fc, d, bisected = a, fa, a, True
    for _ in range(max_iter):
        if abs(b - a) <= xtol or fb == 0:
            return b
        if fa != fc and fb != fc:
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            s = b - fb * (b - a) / (fb - fa)
        lo, hi = sorted(((3 * a + b) / 4, b))
        unsafe = (not lo < s < hi or
                  (bisected and abs(s - b) >= abs(b - c) / 2) or
                  (not bisected and abs(s - b) >= abs(c - d) / 2) or
                  (bisected and abs(b - c) < xtol) or
                  (not bisected and abs(c - d) < xtol))
        if unsafe:
            s, bisected = (a + b) / 2, True
        else:
            bisected = False
        fs = f(s)
        if not isfinite(fs):
            raise ValueError("Nonfinite function value")
        d, c, fc = c, b, fb
        if (fa > 0) != (fs > 0):
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b, fa, fb = b, a, fb, fa
    raise RuntimeError("Brent solver did not converge")


@dataclass(frozen=True)
class OptimizeResult:
    x: np.ndarray
    value: float
    iterations: int
    converged: bool


def nelder_mead(f: Callable[[np.ndarray], float], start: np.ndarray,
                step: float = 0.15, xtol: float = 1e-9,
                ftol: float = 1e-16, max_iter: int = 1000) -> OptimizeResult:
    """Minimize a function using a deterministic Nelder-Mead simplex."""
    start = np.asarray(start, dtype=float)
    if start.ndim != 1 or not len(start) or step <= 0 or xtol <= 0 or ftol <= 0:
        raise ValueError("Invalid simplex settings")
    simplex = np.vstack([start, start + step * np.eye(len(start))])
    values = np.array([f(x) for x in simplex])
    for iteration in range(max_iter):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        if (np.max(np.abs(simplex[1:] - simplex[0])) < xtol
                and np.max(np.abs(values[1:] - values[0])) < ftol):
            return OptimizeResult(simplex[0], float(values[0]), iteration, True)
        centroid = simplex[:-1].mean(axis=0)
        reflected = 2 * centroid - simplex[-1]
        fr = f(reflected)
        if fr < values[0]:
            expanded = centroid + 2 * (reflected - centroid)
            fe = f(expanded)
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            outside = fr < values[-1]
            contracted = centroid + 0.5 * ((reflected if outside else simplex[-1]) - centroid)
            fc = f(contracted)
            if fc < (fr if outside else values[-1]):
                simplex[-1], values[-1] = contracted, fc
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                values[1:] = [f(x) for x in simplex[1:]]
    best = int(np.argmin(values))
    return OptimizeResult(simplex[best], float(values[best]), max_iter, False)
