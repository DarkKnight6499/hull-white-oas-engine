"""Monthly mortgage amortization and explicit prepayment assumptions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def cpr_to_smm(cpr: float | np.ndarray) -> float | np.ndarray:
    """Convert annual conditional prepayment probability to monthly mortality."""
    rate = np.asarray(cpr, dtype=float)
    if np.any((rate < 0) | (rate > 1)) or not np.all(np.isfinite(rate)):
        raise ValueError("CPR must lie in [0, 1]")
    result = 1 - (1 - rate)**(1 / 12)
    return float(result) if rate.ndim == 0 else result


def psa_cpr(month: int | np.ndarray, speed: float = 100.0) -> float | np.ndarray:
    """PSA ramp verified in Federal Reserve FEDS Notes, MBS Principal Payments."""
    age = np.asarray(month)
    if speed < 0 or not np.isfinite(speed) or np.any(age < 1):
        raise ValueError("Positive loan month and nonnegative PSA percentage required")
    result = np.minimum(0.002 * np.minimum(age, 30) * speed / 100, 1.0)
    return float(result) if age.ndim == 0 else result


@dataclass(frozen=True)
class RefinanceCPR:
    """Stylized assumptions, not fitted empirical or regulatory parameters."""

    base_cpr: float = 0.03
    incentive_cpr: float = 0.35
    incentive_scale: float = 0.01
    seasoning_months: int = 30
    burnout: float = 1.5

    def __post_init__(self) -> None:
        if (self.base_cpr < 0 or self.incentive_cpr < 0
                or self.base_cpr + self.incentive_cpr > 1 or self.incentive_scale <= 0
                or self.seasoning_months < 1 or self.burnout < 0
                or not np.all(np.isfinite([self.base_cpr, self.incentive_cpr,
                                           self.incentive_scale, self.burnout]))):
            raise ValueError("Invalid refinance CPR assumptions")

    def cpr(self, month: int, coupon: float, mortgage_rate: float | np.ndarray,
             cumulative_prepaid_fraction: float | np.ndarray = 0.0) -> float | np.ndarray:
        """Seasoned CPR rises with positive incentive and declines with burnout."""
        rates = np.asarray(mortgage_rate)
        prepaid = np.asarray(cumulative_prepaid_fraction)
        if (month < 1 or np.any(prepaid < 0) or np.any(prepaid > 1)
                or not np.all(np.isfinite(rates)) or not np.all(np.isfinite(prepaid))):
            raise ValueError("Invalid month, mortgage rate or prepaid fraction")
        incentive = np.maximum(np.tanh((coupon - rates) / self.incentive_scale), 0.0)
        result = min(month / self.seasoning_months, 1.0) * (
            self.base_cpr + self.incentive_cpr * incentive * np.exp(-self.burnout * prepaid))
        return float(result) if np.ndim(result) == 0 else result


@dataclass(frozen=True)
class MortgagePool:
    """Homogeneous fully amortizing pool with proportional loan removals."""

    balance: float = 100.0
    coupon: float = 0.055
    term_months: int = 360
    servicing_fee: float = 0.0025
    guarantee_fee: float = 0.0020
    age_months: int = 0

    def __post_init__(self) -> None:
        if (self.balance <= 0 or self.coupon < 0 or self.term_months < 1
                or int(self.term_months) != self.term_months or self.age_months < 0
                or int(self.age_months) != self.age_months
                or min(self.servicing_fee, self.guarantee_fee) < 0
                or self.servicing_fee + self.guarantee_fee > self.coupon
                or not np.all(np.isfinite([self.balance, self.coupon, self.servicing_fee, self.guarantee_fee]))):
            raise ValueError("Invalid mortgage pool terms")

    @property
    def net_coupon(self) -> float:
        """Annual pass-through interest rate after fee strip."""
        return self.coupon - self.servicing_fee - self.guarantee_fee

    def scheduled_principal(self, balance: float | np.ndarray, remaining: int) -> float | np.ndarray:
        """Amortize surviving loans over the unchanged remaining term."""
        if remaining < 1 or np.any(np.asarray(balance) < 0):
            raise ValueError("Nonnegative balance and positive remaining term required")
        monthly = self.coupon / 12
        if monthly == 0:
            return np.asarray(balance) / remaining
        payment_factor = monthly / -np.expm1(-remaining * np.log1p(monthly))
        return np.minimum(np.asarray(balance), np.asarray(balance) * (payment_factor - monthly))

    def static_cashflows(self, psa_speed: float = 100.0) -> pd.DataFrame:
        """Deterministic PSA cash flows with no rate-dependent optionality."""
        balance = self.balance
        rows = []
        for month in range(1, self.term_months + 1):
            cpr = psa_cpr(month + self.age_months, psa_speed)
            scheduled = float(self.scheduled_principal(balance, self.term_months - month + 1))
            prepaid = (balance - scheduled) * cpr_to_smm(cpr)
            if month == self.term_months:
                scheduled, prepaid = balance, 0.0
            interest = balance * self.net_coupon / 12
            ending = max(balance - scheduled - prepaid, 0.0)
            rows.append((month, month / 12, balance, cpr, scheduled, prepaid,
                         interest, scheduled + prepaid + interest, ending))
            balance = ending
        return pd.DataFrame(rows, columns=["month", "time", "opening_balance", "cpr",
                                          "scheduled_principal", "prepaid_principal", "interest",
                                          "cashflow", "ending_balance"])


def weighted_average_life(principal: np.ndarray, times: np.ndarray,
                           original_balance: float) -> float:
    """Principal-weighted repayment time in years."""
    principal, times = np.asarray(principal), np.asarray(times)
    if principal.shape != times.shape or original_balance <= 0 or np.any(principal < 0) or np.any(times < 0):
        raise ValueError("Invalid WAL inputs")
    return float(np.dot(principal, times) / original_balance)
