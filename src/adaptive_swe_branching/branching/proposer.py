from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from adaptive_swe_branching.data.records import CheckpointRecord, TrajectoryRecord


@dataclass(frozen=True)
class CandidateNode:
    checkpoint: CheckpointRecord
    reason: str


class CandidateProposer(Protocol):
    def propose(
        self, trajectory: TrajectoryRecord, checkpoint: CheckpointRecord
    ) -> CandidateNode | None: ...


class EveryKStepsProposer:
    """Simple non-learned proposer for controlled ablations."""

    def __init__(self, every: int):
        if every < 1:
            raise ValueError("every must be positive")
        self.every = every

    def propose(
        self, trajectory: TrajectoryRecord, checkpoint: CheckpointRecord
    ) -> CandidateNode | None:
        if checkpoint.absolute_step % self.every:
            return None
        return CandidateNode(checkpoint, f"periodic_every_{self.every}_steps")
