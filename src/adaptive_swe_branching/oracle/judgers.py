from __future__ import annotations

from dataclasses import dataclass

from adaptive_swe_branching.data.records import Outcome
from adaptive_swe_branching.oracle.records import (
    FutureSample,
    ParentContinuationExperiment,
)


@dataclass(frozen=True)
class OracleAMeasurement:
    parent_checkpoint_id: str
    valid_k: int
    successes: int
    success_rate: float
    branchability: float
    invalid_trajectory_ids: tuple[str, ...]
    excluded_cap_hit_trajectory_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrajectoryOutcomeRank:
    trajectory_id: str
    outcome: Outcome


class OracleA:
    """Measure whether same-parent futures straddle the outcome boundary."""

    def __init__(self, *, minimum_valid_k: int = 1) -> None:
        if minimum_valid_k < 1:
            raise ValueError("minimum valid K must be positive")
        self.minimum_valid_k = minimum_valid_k

    def measure(
        self,
        experiment: ParentContinuationExperiment,
        *,
        exclude_cap_hits: bool = False,
    ) -> OracleAMeasurement:
        valid = tuple(
            sample
            for sample in experiment.valid_samples
            if not (exclude_cap_hits and sample.cap_hit)
        )
        if len(valid) < self.minimum_valid_k:
            raise ValueError(
                f"Oracle A needs at least {self.minimum_valid_k} valid "
                f"continuations; found {len(valid)}"
            )
        successes = sum(sample.outcome == Outcome.SOLVED for sample in valid)
        success_rate = successes / len(valid)
        return OracleAMeasurement(
            parent_checkpoint_id=experiment.parent_checkpoint_id,
            valid_k=len(valid),
            successes=successes,
            success_rate=success_rate,
            branchability=4.0 * success_rate * (1.0 - success_rate),
            invalid_trajectory_ids=tuple(
                sample.trajectory_id
                for sample in experiment.samples
                if sample.outcome == Outcome.INVALID
            ),
            excluded_cap_hit_trajectory_ids=tuple(
                sample.trajectory_id
                for sample in experiment.valid_samples
                if exclude_cap_hits and sample.cap_hit
            ),
        )

    def decide(
        self,
        experiment: ParentContinuationExperiment,
        *,
        threshold: float,
    ) -> bool:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("branchability threshold must be in [0, 1]")
        return self.measure(experiment).branchability >= threshold


class TrajectoryOutcomeOracle:
    """Clairvoyant upper bound using each sampled sibling's realized outcome.

    This is not an estimate of a child's true q: one non-nested continuation
    provides only one Bernoulli realization. Learned B instead applies the same
    q(state) model used by A to each child checkpoint and selects the largest q.
    """

    def rank(
        self, samples: tuple[FutureSample, ...]
    ) -> tuple[TrajectoryOutcomeRank, ...]:
        valid = tuple(
            sample for sample in samples if sample.outcome != Outcome.INVALID
        )
        if not valid:
            raise ValueError(
                "Trajectory-Outcome Oracle needs at least one valid sibling"
            )
        ranks = tuple(
            TrajectoryOutcomeRank(
                trajectory_id=sample.trajectory_id,
                outcome=sample.outcome,
            )
            for sample in valid
        )
        return tuple(
            sorted(
                ranks,
                key=lambda item: (
                    0 if item.outcome == Outcome.SOLVED else 1,
                    item.trajectory_id,
                ),
            )
        )

    def select(self, samples: tuple[FutureSample, ...]) -> FutureSample:
        best_id = self.rank(samples)[0].trajectory_id
        return next(sample for sample in samples if sample.trajectory_id == best_id)
