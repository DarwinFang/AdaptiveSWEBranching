from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from adaptive_swe_branching.data.records import Cost, Outcome
from adaptive_swe_branching.oracle.records import FutureSample


class OutcomeUtility(Protocol):
    name: str

    def value(self, sample: FutureSample, extra_cost: Cost | None = None) -> float: ...


@dataclass(frozen=True)
class SuccessOnlyUtility:
    """Explicit solve-rate utility; compute is reported separately."""

    name: str = "success_only"

    def value(self, sample: FutureSample, extra_cost: Cost | None = None) -> float:
        return 1.0 if sample.outcome == Outcome.SOLVED else 0.0


@dataclass(frozen=True)
class BudgetedSuccessUtility:
    """A solved run is valuable only when total measured cost fits one budget."""

    axis: str
    budget: float

    @property
    def name(self) -> str:
        return f"budgeted_success:{self.axis}<={self.budget}"

    def value(self, sample: FutureSample, extra_cost: Cost | None = None) -> float:
        total = sample.cost_from_state + (extra_cost or Cost())
        return float(
            sample.outcome == Outcome.SOLVED
            and cost_value(total, self.axis) <= self.budget
        )


def cost_value(cost: Cost, axis: str) -> float:
    if axis == "total_tokens":
        return float(cost.total_tokens)
    if not hasattr(cost, axis):
        raise ValueError(f"unknown cost axis: {axis}")
    return float(getattr(cost, axis))
