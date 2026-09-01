from __future__ import annotations

import random

from adaptive_swe_branching.baselines.swe_replay.types import ReplayResult
from adaptive_swe_branching.data.records import Cost, Outcome
from adaptive_swe_branching.evaluation.matched_compute import StrategyTrace
from adaptive_swe_branching.oracle.judgers import OracleA, OracleB
from adaptive_swe_branching.oracle.records import CounterfactualExperiment, FutureSample
from adaptive_swe_branching.oracle.utility import OutcomeUtility


def single_chain(task_id: str, sample: FutureSample) -> StrategyTrace:
    return _trace(task_id, "single_chain", sample, sample.cost_from_state)


def best_of_n(
    task_id: str, samples: tuple[FutureSample, ...], *, n: int
) -> StrategyTrace:
    if n < 1 or len(samples) < n:
        raise ValueError("Best-of-N needs at least N independent samples")
    chosen = samples[:n]
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
    experiment: CounterfactualExperiment, *, seed: int
) -> StrategyTrace:
    rng = random.Random(seed)
    child = rng.choice(experiment.children)
    sample = rng.choice(child.samples)
    acquisition = sum(
        (candidate.local_branch_cost for candidate in experiment.children), Cost()
    )
    total = acquisition + sample.cost_from_state
    return _trace(experiment.task_id, "random_branching", sample, total)


def oracle_a_b(
    experiment: CounterfactualExperiment,
    *,
    utility: OutcomeUtility,
    headroom_threshold: float,
    evaluation_seed: int,
) -> StrategyTrace:
    measurement = OracleA().measure(experiment, utility=utility)
    rng = random.Random(evaluation_seed)
    if measurement.branching_headroom <= headroom_threshold:
        sample = rng.choice(experiment.no_branch_samples)
        return _trace(experiment.task_id, "oracle_a_b", sample, sample.cost_from_state)
    child = OracleB().select(experiment.children)
    sample = rng.choice(child.samples)
    total = measurement.local_branch_cost + sample.cost_from_state
    return _trace(experiment.task_id, "oracle_a_b", sample, total)


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
    )
    return _trace(task_id, "faithful_swe_replay", sample, total)


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
