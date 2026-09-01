from __future__ import annotations

import pytest

from adaptive_swe_branching.data.records import Cost, Outcome, StepRecord
from adaptive_swe_branching.evaluation.matched_compute import (
    MatchedComputeEvaluator,
    StrategyTrace,
)
from adaptive_swe_branching.evaluation.simulate import (
    best_of_n,
    oracle_a_b,
    random_branching,
    single_chain,
)
from adaptive_swe_branching.oracle.judgers import (
    OracleA,
    TrajectoryOutcomeOracle,
)
from adaptive_swe_branching.oracle.records import (
    FutureSample,
    ParentContinuationExperiment,
)


def step(index: int, tokens: int) -> StepRecord:
    return StepRecord(
        absolute_step=index,
        reasoning="",
        tool_name="terminal",
        tool_input={},
        action_text="",
        observation_text="",
        is_error=False,
        explored_files_before=(),
        cost=Cost(input_tokens=tokens),
    )


def sample(
    identity: str,
    solved: bool,
    tokens: int = 10,
    *,
    invalid: bool = False,
    cap_hit: bool = False,
    step_tokens: tuple[int, ...] = (2, 3),
) -> FutureSample:
    outcome = Outcome.INVALID if invalid else (
        Outcome.SOLVED if solved else Outcome.UNSOLVED
    )
    return FutureSample(
        trajectory_id=identity,
        seed=1,
        outcome=outcome,
        cost_from_state=Cost(input_tokens=tokens),
        final_patch="",
        termination_reason="finish",
        steps=tuple(step(index, cost) for index, cost in enumerate(step_tokens, 1)),
        cap_hit=cap_hit,
    )


def experiment(*samples: FutureSample) -> ParentContinuationExperiment:
    return ParentContinuationExperiment("t", "p", ("random",), tuple(samples))


def test_oracle_a_measures_outcome_boundary_and_excludes_invalid() -> None:
    mixed = OracleA().measure(
        experiment(
            sample("a", True),
            sample("b", False),
            sample("invalid", False, invalid=True),
        )
    )
    certain = OracleA().measure(experiment(sample("c", True), sample("d", True)))
    assert (mixed.valid_k, mixed.successes, mixed.branchability) == (2, 1, 1.0)
    assert mixed.invalid_trajectory_ids == ("invalid",)
    assert certain.branchability == 0.0
    assert OracleA().decide(
        experiment(sample("a", True), sample("b", False)), threshold=0.75
    )


def test_trajectory_outcome_oracle_prefers_realized_success() -> None:
    fast_failure = sample("failure", False, 1)
    slow_success = sample("slow", True, 20)
    fast_success = sample("fast", True, 10)
    selected = TrajectoryOutcomeOracle().select(
        (fast_failure, slow_success, fast_success)
    )
    assert selected.trajectory_id == "fast"
    tied_failures = TrajectoryOutcomeOracle().rank(
        (sample("z", False), sample("a", False))
    )
    assert [rank.trajectory_id for rank in tied_failures] == ["a", "z"]


def test_matched_compute_requires_same_tasks_and_uses_first_solve_cost() -> None:
    traces = (
        StrategyTrace("a", "single", Outcome.SOLVED, Cost(steps=10), Cost(steps=5)),
        StrategyTrace("b", "single", Outcome.UNSOLVED, Cost(steps=10), None),
        StrategyTrace("a", "oracle", Outcome.SOLVED, Cost(steps=9), Cost(steps=3)),
        StrategyTrace("b", "oracle", Outcome.SOLVED, Cost(steps=9), Cost(steps=8)),
    )
    points = MatchedComputeEvaluator().curve(traces, cost_axis="steps", budgets=(4, 8))
    oracle = [point for point in points if point.strategy == "oracle"]
    assert [point.solve_rate for point in oracle] == [0.5, 1.0]


def test_strategy_simulators_charge_all_purchased_compute() -> None:
    samples = (sample("one", False, 10), sample("two", True, 20))
    assert single_chain("t", samples[0]).total_cost.total_tokens == 10
    best = best_of_n("t", samples, n=2)
    assert best.outcome == Outcome.SOLVED
    assert best.total_cost.total_tokens == 30

    parent = experiment(*samples)
    random_trace = random_branching(parent, n=2, branch_span=1, seed=1)
    assert random_trace.total_cost.total_tokens in {12, 22}
    oracle_trace = oracle_a_b(
        parent,
        branchability_threshold=0.75,
        n=2,
        branch_span=1,
        seed=1,
        minimum_valid_k=2,
    )
    assert oracle_trace.outcome == Outcome.SOLVED
    assert oracle_trace.total_cost.total_tokens == 22


def test_oracle_a_skips_branching_for_certain_parent() -> None:
    parent = experiment(sample("one", True, 10), sample("two", True, 20))
    trace = oracle_a_b(
        parent,
        branchability_threshold=0.75,
        n=2,
        branch_span=1,
        seed=3,
        minimum_valid_k=2,
    )
    assert trace.outcome == Outcome.SOLVED
    assert trace.total_cost.total_tokens in {10, 20}


def test_oracle_a_enforces_minimum_k_and_supports_cap_hit_sensitivity() -> None:
    parent = experiment(
        sample("one", True),
        sample("two", False, cap_hit=True),
    )
    with pytest.raises(ValueError, match="at least 2 valid"):
        OracleA(minimum_valid_k=2).measure(parent, exclude_cap_hits=True)
