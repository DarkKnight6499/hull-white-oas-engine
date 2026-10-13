"""Recombining Gaussian trinomial tree with discount-curve fitting."""

from dataclasses import dataclass

import numpy as np

from .instruments import FixedRateBond
from .model import HullWhite, OptionKind


@dataclass(frozen=True)
class TreeLevel:
    x: np.ndarray
    alpha: float
    state_prices: np.ndarray
    branches: np.ndarray
    probabilities: np.ndarray


class HullWhiteTree:
    """Exact OU moment matching, nearest-center branches and fitted alpha."""

    def __init__(self, model: HullWhite, horizon: float, steps: int) -> None:
        if horizon <= 0 or steps < 1 or int(steps) != steps:
            raise ValueError("Positive horizon and integer steps required")
        self.model, self.horizon, self.steps = model, horizon, steps
        self.dt = horizon / steps
        self.times = np.linspace(0, horizon, steps + 1)
        variance = model.variance_x(self.dt)
        self.dx = np.sqrt(3 * variance)
        indices = np.array([0], dtype=int)
        state_prices = np.array([1.0])
        self.levels: list[TreeLevel] = []
        decay = np.exp(-model.a * self.dt)
        for level in range(steps):
            x = indices * self.dx
            if self.dx == 0:
                branches = np.zeros((1, 3), dtype=int)
                probabilities = np.array([[0.0, 1.0, 0.0]])
                next_indices = indices
            else:
                center = np.rint(indices * decay).astype(int)
                error = indices * decay - center
                raw_branches = center[:, None] + np.array([-1, 0, 1])
                next_indices = np.arange(raw_branches.min(), raw_branches.max() + 1)
                branches = raw_branches - next_indices[0]
                probabilities = np.column_stack(((1 / 3 + error**2 - error) / 2,
                                                  2 / 3 - error**2,
                                                  (1 / 3 + error**2 + error) / 2))
            total = float(np.dot(state_prices, np.exp(-x * self.dt)))
            alpha = np.log(total / model.curve.discount(self.times[level + 1])) / self.dt
            self.levels.append(TreeLevel(x, float(alpha), state_prices, branches, probabilities))
            weights = state_prices[:, None] * np.exp(-(x + alpha) * self.dt)[:, None] * probabilities
            state_prices = np.bincount(branches.ravel(), weights=weights.ravel(), minlength=len(next_indices))
            indices = next_indices
        self.terminal_x = indices * self.dx
        self.terminal_state_prices = state_prices

    def index(self, time: float) -> int:
        """Map an event to the grid without silently moving its date."""
        i = round(time / self.dt)
        if i < 0 or i > self.steps or abs(i * self.dt - time) > 1e-9:
            raise ValueError("Event date must align exactly with the tree grid")
        return i

    def rollback(self, level: int, values: np.ndarray, spread: float = 0.0) -> np.ndarray:
        """One risk-neutral discounted expectation."""
        node = self.levels[level]
        expected = np.sum(node.probabilities * values[node.branches], axis=1)
        return expected * np.exp(-(node.x + node.alpha + spread) * self.dt)

    def zero_bond(self, maturity: float, spread: float = 0.0) -> float:
        """Price a unit payment at an aligned maturity."""
        end = self.index(maturity)
        size = len(self.terminal_x) if end == self.steps else len(self.levels[end].x)
        values = np.ones(size)
        for i in reversed(range(end)):
            values = self.rollback(i, values, spread)
        return float(values[0])

    def bond_price(self, bond: FixedRateBond, spread: float = 0.0) -> float:
        """Value investor cash flows with issuer calls and investor puts."""
        dates, amounts = bond.cashflows()
        cashflows = {self.index(t): c for t, c in zip(dates, amounts)}
        calls = {self.index(t): c for t, c in bond.calls.items()}
        puts = {self.index(t): c for t, c in bond.puts.items()}
        end = self.index(bond.maturity)
        size = len(self.terminal_x) if end == self.steps else len(self.levels[end].x)
        values = np.full(size, cashflows[end])
        for i in reversed(range(end)):
            values = self.rollback(i, values, spread)
            if i in calls:
                values = np.minimum(values, calls[i])
            if i in puts:
                values = np.maximum(values, puts[i])
            values += cashflows.get(i, 0.0)
        return float(values[0])

    def coupon_bond_option(self, dates: np.ndarray, amounts: np.ndarray, strike: float,
                           exercise_times: list[float], kind: OptionKind = "call",
                           spread: float = 0.0) -> float:
        """European or Bermudan option on remaining ex-coupon bond payments."""
        dates, amounts = np.asarray(dates), np.asarray(amounts)
        if (kind not in ("call", "put") or strike <= 0 or not exercise_times
                or dates.ndim != 1 or dates.shape != amounts.shape or not len(dates)
                or np.any(amounts < 0) or np.any(dates <= 0)
                or not np.all(np.isfinite(dates)) or not np.all(np.isfinite(amounts))
                or max(exercise_times) >= max(dates)):
            raise ValueError("Valid option terms and exercise before maturity required")
        cashflows: dict[int, float] = {}
        for t, c in zip(dates, amounts):
            idx = self.index(float(t))
            cashflows[idx] = cashflows.get(idx, 0.0) + c
        exercise = {self.index(t) for t in exercise_times}
        end = max(cashflows)
        size = len(self.terminal_x) if end == self.steps else len(self.levels[end].x)
        underlying = np.full(size, cashflows[end])
        option = np.zeros(size)
        sign = 1 if kind == "call" else -1
        for i in reversed(range(end)):
            underlying = self.rollback(i, underlying, spread)
            option = self.rollback(i, option, spread)
            if i in exercise:
                option = np.maximum(option, np.maximum(sign * (underlying - strike), 0))
            underlying += cashflows.get(i, 0)
        return float(option[0])
