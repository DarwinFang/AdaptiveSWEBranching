from __future__ import annotations

from dataclasses import dataclass

from adaptive_swe_branching.data.records import (
    ContinuationRecord,
    Outcome,
    StepRecord,
    TrajectoryRecord,
)


@dataclass(frozen=True)
class ContinuationEvidence:
    record: ContinuationRecord
    trajectory: TrajectoryRecord

    def post_parent_steps(self) -> tuple[StepRecord, ...]:
        if self.record.trajectory_id != self.trajectory.trajectory_id:
            raise ValueError("continuation and trajectory identities differ")
        if (
            self.trajectory.parent_checkpoint_id is not None
            and self.trajectory.parent_checkpoint_id != self.record.source_checkpoint_id
        ):
            raise ValueError("trajectory was not restored from the recorded parent")
        start = self.record.post_parent_step_start
        count = self.record.post_parent_step_count
        end = len(self.trajectory.steps) if count is None else start + count
        steps = self.trajectory.steps[start:end]
        if count is not None and len(steps) != count:
            raise ValueError("recorded post-parent step count exceeds trajectory")
        return steps


@dataclass(frozen=True)
class SuccessProbabilityTarget:
    """One binomial target for the shared q(state) model."""

    state_id: str
    state_kind: str
    source_parent_checkpoint_id: str
    successes: int
    trials: int
    empirical_q: float
    prefix_depth: int | None = None
    continuation_id: str | None = None
    invalid_continuation_ids: tuple[str, ...] = ()
    excluded_cap_hit_continuation_ids: tuple[str, ...] = ()


def branchability_from_q(q: float) -> float:
    """Inference-time A score derived from the shared success probability."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("success probability must be in [0, 1]")
    return 4.0 * q * (1.0 - q)


def parent_q_target(
    parent_checkpoint_id: str,
    continuations: tuple[ContinuationRecord, ...],
    *,
    minimum_valid_k: int = 1,
    exclude_cap_hits: bool = False,
) -> SuccessProbabilityTarget:
    """Aggregate same-parent futures into one binomial q target.

    Main protocol counts a 60-step cap hit as unsolved. Setting
    ``exclude_cap_hits`` creates the pre-specified sensitivity analysis.
    """
    if minimum_valid_k < 1:
        raise ValueError("minimum valid K must be positive")
    _require_same_parent(parent_checkpoint_id, continuations)
    valid = [
        item
        for item in continuations
        if item.outcome != Outcome.INVALID
        and not (exclude_cap_hits and item.cap_hit)
    ]
    invalid = tuple(
        item.continuation_id
        for item in continuations
        if item.outcome == Outcome.INVALID
    )
    excluded_cap_hits = tuple(
        item.continuation_id
        for item in continuations
        if exclude_cap_hits and item.cap_hit and item.outcome != Outcome.INVALID
    )
    if len(valid) < minimum_valid_k:
        raise ValueError(
            f"parent q target needs at least {minimum_valid_k} valid "
            f"continuations; found {len(valid)}"
        )
    successes = sum(item.outcome == Outcome.SOLVED for item in valid)
    return SuccessProbabilityTarget(
        state_id=parent_checkpoint_id,
        state_kind="parent_checkpoint",
        source_parent_checkpoint_id=parent_checkpoint_id,
        successes=successes,
        trials=len(valid),
        empirical_q=successes / len(valid),
        invalid_continuation_ids=invalid,
        excluded_cap_hit_continuation_ids=excluded_cap_hits,
    )


def prefix_q_targets(
    parent_checkpoint_id: str,
    evidence: tuple[ContinuationEvidence, ...],
    *,
    depths: tuple[int, ...] = (1, 2, 4, 6),
    exclude_cap_hits: bool = False,
) -> tuple[SuccessProbabilityTarget, ...]:
    """Create Bernoulli q targets for saved child states at each depth.

    A prefix has one observed future in this non-nested design, so its target is
    one Bernoulli draw (trials=1), not an exact empirical child success rate.
    """
    if any(depth < 1 for depth in depths):
        raise ValueError("prefix depths must be positive")
    _require_same_parent(
        parent_checkpoint_id, tuple(item.record for item in evidence)
    )
    targets: list[SuccessProbabilityTarget] = []
    for item in evidence:
        if item.record.outcome == Outcome.INVALID:
            continue
        if exclude_cap_hits and item.record.cap_hit:
            continue
        step_count = len(item.post_parent_steps())
        for depth in depths:
            if step_count < depth:
                continue
            checkpoint_id = item.record.prefix_checkpoint_ids_by_depth.get(depth)
            if checkpoint_id is None:
                raise ValueError(
                    f"continuation {item.record.continuation_id} is missing "
                    f"the saved checkpoint at depth {depth}"
                )
            success = int(item.record.outcome == Outcome.SOLVED)
            targets.append(
                SuccessProbabilityTarget(
                    state_id=checkpoint_id,
                    state_kind="continuation_prefix_checkpoint",
                    source_parent_checkpoint_id=parent_checkpoint_id,
                    successes=success,
                    trials=1,
                    empirical_q=float(success),
                    prefix_depth=depth,
                    continuation_id=item.record.continuation_id,
                )
            )
    return tuple(targets)


def _require_same_parent(
    parent_checkpoint_id: str,
    continuations: tuple[ContinuationRecord, ...],
) -> None:
    mismatched = [
        item.continuation_id
        for item in continuations
        if item.source_checkpoint_id != parent_checkpoint_id
    ]
    if mismatched:
        raise ValueError(f"continuations do not share parent: {mismatched}")
