from __future__ import annotations

from dataclasses import dataclass

from adaptive_swe_branching.branching.engine import TemporaryBrancher
from adaptive_swe_branching.branching.gate import BranchGate, GateDecision
from adaptive_swe_branching.branching.proposer import CandidateProposer
from adaptive_swe_branching.branching.ranker import BranchRanker, RankedChild
from adaptive_swe_branching.data.records import CheckpointRecord, TrajectoryRecord


@dataclass(frozen=True)
class SchedulingDecision:
    action: str
    gate: GateDecision | None
    selected: RankedChild | None
    branch_group_id: str | None


class SelectiveBranchingScheduler:
    """Propose -> gate -> temporary branch -> rank -> collapse to one child."""

    def __init__(
        self,
        *,
        proposer: CandidateProposer,
        gate: BranchGate,
        brancher: TemporaryBrancher,
        ranker: BranchRanker,
    ) -> None:
        self.proposer = proposer
        self.gate = gate
        self.brancher = brancher
        self.ranker = ranker

    def decide(
        self, trajectory: TrajectoryRecord, checkpoint: CheckpointRecord
    ) -> SchedulingDecision:
        candidate = self.proposer.propose(trajectory, checkpoint)
        if candidate is None:
            return SchedulingDecision("continue_single", None, None, None)
        gate = self.gate.decide(candidate)
        if not gate.branch:
            return SchedulingDecision("gate_rejected", gate, None, None)
        group, children = self.brancher.branch(checkpoint)
        ranked = self.ranker.rank(checkpoint, children)
        if not ranked:
            raise RuntimeError("ranker returned no child after branching")
        # The caller resumes only this checkpoint: the population collapses.
        return SchedulingDecision(
            "collapse_to_child", gate, ranked[0], group.branch_group_id
        )
