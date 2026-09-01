from __future__ import annotations

import random

from adaptive_swe_branching.baselines.swe_replay.types import ReplayResult
from adaptive_swe_branching.data.records import Cost, Outcome
from adaptive_swe_branching.evaluation.matched_compute import StrategyTrace
from adaptive_swe_branching.oracle.judgers import OracleA, OracleB
from adaptive_swe_branching.oracle.records import (
    FutureSample,
    ParentContinuationExperiment,
)


def single_chain(task_id: str, sample: FutureSample) -> StrategyTrace:
    return _trace(task_id, "single_chain", sample, sample.cost_from_state)


def best_of_n(
    task_id: str, samples: tuple[FutureSample, ...], *, n: int
) -> StrategyTrace:
    chosen = _valid_prefix(samples, n)
    total = sum((sample.cost_from_state for sample in chosen), Cost())
    solved = next(
        (sample for sample in chosen if sample.outcome == Outcome.SOLVED), None
    )
    representative = solved or chosen[0]
    return _trace(
        task_id,
        f"best_of_{n}",
        representative,
        total,
        outcome=Outcome.SOLVED if solved else Outcome.UNSOLVED,
    )


def random_branching(
    experiment: ParentContinuationExperiment,
    *,
    n: int,
    branch_span: int,
    seed: int,
) -> StrategyTrace:
    candidates = _sample_siblings(experiment, n=n, seed=seed)
    selected = random.Random(seed + 1).choice(candidates)
    total = _temporary_branch_cost(candidates, selected, branch_span)
    return _trace(experiment.task_id, f"random_branching_n{n}", selected, total)


def oracle_a_b(
    experiment: ParentContinuationExperiment,
    *,
    branchability_threshold: float,
    n: int,
    branch_span: int,
    seed: int,
) -> StrategyTrace:
    rng = random.Random(seed)
    if not OracleA().decide(experiment, threshold=branchability_threshold):
        selected = rng.choice(experiment.valid_samples)
        return _trace(
            experiment.task_id,
            f"oracle_a_b_n{n}",
            selected,
            selected.cost_from_state,
        )
    candidates = _sample_siblings(experiment, n=n, seed=seed)
    selected = OracleB().select(candidates)
    total = _temporary_branch_cost(candidates, selected, branch_span)
    return _trace(experiment.task_id, f"oracle_a_b_n{n}", selected, total)


def faithful_swe_replay(task_id: str, result: ReplayResult) -> StrategyTrace:
    selected = next(
        item
        for item in result.archive
        if item.trajectory.trajectory_id == result.selected_trajectory_id
    )
    total = sum((item.generation_cost for item in result.archive), Cost())
    sample = FutureSample(
        trajectory_id=selected.trajectory.trajectory_id,
        seed=selected.trajectory.seed,
        outcome=selected.trajectory.outcome,
        cost_from_state=total,
        final_patch=selected.trajectory.final_patch,
        termination_reason=selected.trajectory.termination_reason,
        steps=selected.trajectory.steps,
    )
    return _trace(task_id, "faithful_swe_replay", sample, total)


def _valid_prefix(
    samples: tuple[FutureSample, ...], n: int
) -> tuple[FutureSample, ...]:
    valid = tuple(sample for sample in samples if sample.outcome != Outcome.INVALID)
    if n < 1 or len(valid) < n:
        raise ValueError("strategy needs at least N valid independent samples")
    return valid[:n]


def _sample_siblings(
    experiment: ParentContinuationExperiment, *, n: int, seed: int
) -> tuple[FutureSample, ...]:
    valid = experiment.valid_samples
    if n < 1 or len(valid) < n:
        raise ValueError("temporary branching needs at least N valid continuations")
    return tuple(random.Random(seed).sample(valid, n))


def _temporary_branch_cost(
    candidates: tuple[FutureSample, ...],
    selected: FutureSample,
    branch_span: int,
) -> Cost:
    if branch_span < 1:
        raise ValueError("branch span must be positive")
    # The selected continuation's full cost already includes its prefix. Every
    # sibling that is discarded is charged only for the prefix actually run.
    discarded_prefixes = sum(
        (
            _prefix_cost(candidate, branch_span)
            for candidate in candidates
            if candidate.trajectory_id != selected.trajectory_id
        ),
        Cost(),
    )
    return selected.cost_from_state + discarded_prefixes


def _prefix_cost(sample: FutureSample, branch_span: int) -> Cost:
    if not sample.steps:
        raise ValueError(
            f"sample {sample.trajectory_id} has no step records for prefix costing"
        )
    return sum((step.cost for step in sample.steps[:branch_span]), Cost())


def _trace(
    task_id: str,
    strategy: str,
    sample: FutureSample,
    total: Cost,
    *,
    outcome: Outcome | None = None,
) -> StrategyTrace:
    final_outcome = outcome or sample.outcome
    return StrategyTrace(
        task_id=task_id,
        strategy=strategy,
        outcome=final_outcome,
        total_cost=total,
        first_solve_cost=total if final_outcome == Outcome.SOLVED else None,
        invalid_reason=(
            sample.termination_reason if final_outcome == Outcome.INVALID else None
        ),
    )
