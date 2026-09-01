from __future__ import annotations

from adaptive_swe_branching.data.records import Cost, Outcome
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
from adaptive_swe_branching.oracle.judgers import OracleA, OracleB
from adaptive_swe_branching.oracle.records import (
    ChildFutures,
    CounterfactualExperiment,
    FutureSample,
)
from adaptive_swe_branching.oracle.utility import SuccessOnlyUtility


def sample(identity: str, solved: bool, tokens: int = 10) -> FutureSample:
    return FutureSample(
        trajectory_id=identity,
        seed=1,
        outcome=Outcome.SOLVED if solved else Outcome.UNSOLVED,
        cost_from_state=Cost(input_tokens=tokens),
        final_patch="",
        termination_reason="finish",
    )


def test_oracle_b_prefers_success_not_fast_failure() -> None:
    bad = ChildFutures(
        "bad", Cost(steps=1), (sample("b1", False, 1), sample("b2", False, 1))
    )
    good = ChildFutures(
        "good", Cost(steps=1), (sample("g1", True, 20), sample("g2", False, 1))
    )
    assert OracleB().select((bad, good)).child_checkpoint_id == "good"


def test_oracle_a_utility_is_explicit() -> None:
    experiment = CounterfactualExperiment(
        task_id="t",
        parent_checkpoint_id="p",
        no_branch_samples=(sample("n1", False), sample("n2", False)),
        children=(
            ChildFutures("c1", Cost(steps=2), (sample("a", True), sample("b", True))),
            ChildFutures("c2", Cost(steps=2), (sample("c", False), sample("d", False))),
        ),
    )
    result = OracleA().measure(experiment, utility=SuccessOnlyUtility())
    assert result.no_branch_value == 0.0
    assert result.selected_branch_value == 1.0
    assert result.branching_headroom == 1.0
    assert result.local_branch_cost.steps == 4


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
    experiment = CounterfactualExperiment(
        task_id="t",
        parent_checkpoint_id="p",
        no_branch_samples=samples,
        children=(
            ChildFutures("good", Cost(input_tokens=3), (sample("g", True, 5),)),
            ChildFutures("bad", Cost(input_tokens=4), (sample("b", False, 5),)),
        ),
    )
    random_trace = random_branching(experiment, seed=1)
    assert random_trace.total_cost.total_tokens == 12
    oracle_trace = oracle_a_b(
        experiment,
        utility=SuccessOnlyUtility(),
        headroom_threshold=-1.0,
        evaluation_seed=1,
    )
    assert oracle_trace.outcome == Outcome.SOLVED
    assert oracle_trace.total_cost.total_tokens == 12
