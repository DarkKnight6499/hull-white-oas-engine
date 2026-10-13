"""Fixed-rate bonds with explicit coupon and exercise schedules."""

from dataclasses import dataclass, field

import numpy as np

from .curves import DiscountCurve


@dataclass(frozen=True)
class FixedRateBond:
    """Bond issued at time zero; exercise redemption excludes the current coupon."""

    maturity: float
    coupon: float
    frequency: int = 2
    face: float = 100.0
    calls: dict[float, float] = field(default_factory=dict)
    puts: dict[float, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        periods = self.maturity * self.frequency
        if (self.maturity <= 0 or self.coupon < 0 or self.frequency < 1
                or int(self.frequency) != self.frequency or self.face <= 0
                or not np.all(np.isfinite([periods, self.coupon, self.face]))
                or abs(periods - round(periods)) > 1e-10):
            raise ValueError("Invalid regular bond terms")
        for schedule in (self.calls, self.puts):
            for time, redemption in schedule.items():
                if (time <= 0 or time >= self.maturity or redemption <= 0
                        or not np.all(np.isfinite([time, redemption]))
                        or abs(time * self.frequency - round(time * self.frequency)) > 1e-10):
                    raise ValueError("Exercise must occur on a coupon date before maturity")
        if self.calls.keys() & self.puts.keys():
            raise ValueError("Simultaneous call and put require a priority convention")

    def cashflows(self) -> tuple[np.ndarray, np.ndarray]:
        """Coupon dates and amounts including redemption at maturity."""
        dates = np.arange(1, round(self.maturity * self.frequency) + 1) / self.frequency
        amounts = np.full(len(dates), self.face * self.coupon / self.frequency)
        amounts[-1] += self.face
        return dates, amounts

    def straight_price(self, curve: DiscountCurve, spread: float = 0.0) -> float:
        """Discount promised payments with a constant continuous spread."""
        dates, amounts = self.cashflows()
        return float(np.dot(amounts, curve.discount(dates) * np.exp(-spread * dates)))
