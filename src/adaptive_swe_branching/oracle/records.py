from __future__ import annotations

from dataclasses import dataclass

from adaptive_swe_branching.data.records import Cost, Outcome, StepRecord


@dataclass(frozen=True)
class FutureSample:
    """One full continuation restored from a parent checkpoint."""

    trajectory_id: str
    seed: int
    outcome: Outcome
    cost_from_state: Cost
    final_patch: str
    termination_reason: str
    steps: tuple[StepRecord, ...] = ()


@dataclass(frozen=True)
class ParentContinuationExperiment:
    """The single raw counterfactual unit used by both Judger A and B.

    Every sample starts from exactly the same executable parent checkpoint.
    There is deliberately no child-to-downstream second rollout level.
    """

    task_id: str
    parent_checkpoint_id: str
    candidate_source: str
    samples: tuple[FutureSample, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("a parent experiment needs at least one continuation")

    @property
    def valid_samples(self) -> tuple[FutureSample, ...]:
        return tuple(
            sample for sample in self.samples if sample.outcome != Outcome.INVALID
        )
