from __future__ import annotations

from dataclasses import dataclass

from adaptive_swe_branching.data.records import Cost, Outcome
from adaptive_swe_branching.evaluation.costs import cost_value


@dataclass(frozen=True)
class StrategyTrace:
    task_id: str
    strategy: str
    outcome: Outcome
    total_cost: Cost
    first_solve_cost: Cost | None
    invalid_reason: str | None = None


@dataclass(frozen=True)
class CurvePoint:
    strategy: str
    cost_axis: str
    budget: float
    solved: int
    task_count: int
    solve_rate: float


class MatchedComputeEvaluator:
    """Compare any strategies on identical task sets and actual total cost."""

    def curve(
        self,
        traces: tuple[StrategyTrace, ...],
        *,
        cost_axis: str,
        budgets: tuple[float, ...],
    ) -> tuple[CurvePoint, ...]:
        grouped: dict[str, dict[str, StrategyTrace]] = {}
        for trace in traces:
            if trace.outcome == Outcome.INVALID:
                raise ValueError(
                    "invalid infrastructure trace must be repaired or "
                    "explicitly removed: "
                    f"{trace.task_id}/{trace.strategy}"
                )
            by_task = grouped.setdefault(trace.strategy, {})
            if trace.task_id in by_task:
                raise ValueError(
                    f"duplicate strategy/task trace: {trace.strategy}/{trace.task_id}"
                )
            by_task[trace.task_id] = trace
        task_sets = {strategy: frozenset(items) for strategy, items in grouped.items()}
        if len(set(task_sets.values())) > 1:
            raise ValueError(f"strategies do not share the same task set: {task_sets}")
        points: list[CurvePoint] = []
        for strategy, by_task in sorted(grouped.items()):
            for budget in sorted(budgets):
                solved = sum(
                    trace.first_solve_cost is not None
                    and cost_value(trace.first_solve_cost, cost_axis) <= budget
                    for trace in by_task.values()
                )
                points.append(
                    CurvePoint(
                        strategy=strategy,
                        cost_axis=cost_axis,
                        budget=budget,
                        solved=solved,
                        task_count=len(by_task),
                        solve_rate=solved / len(by_task),
                    )
                )
        return tuple(points)

    def terminal_table(
        self, traces: tuple[StrategyTrace, ...], *, cost_axis: str
    ) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[StrategyTrace]] = {}
        for trace in traces:
            grouped.setdefault(trace.strategy, []).append(trace)
        return {
            strategy: {
                "tasks": float(len(items)),
                "solve_rate": sum(item.outcome == Outcome.SOLVED for item in items)
                / len(items),
                "mean_total_compute": sum(
                    cost_value(item.total_cost, cost_axis) for item in items
                )
                / len(items),
            }
            for strategy, items in grouped.items()
        }
