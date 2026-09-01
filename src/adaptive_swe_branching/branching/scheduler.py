from __future__ import annotations

from dataclasses import dataclass

from adaptive_swe_branching.branching.alternatives import (
    BranchPointState,
    RankedAlternativeController,
    RankedAlternativeStore,
)
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
    branch_point_state: BranchPointState | None


class SelectiveBranchingScheduler:
    """Propose -> gate -> temporary branch -> rank -> collapse to one child."""

    def __init__(
        self,
        *,
        proposer: CandidateProposer,
        gate: BranchGate,
        brancher: TemporaryBrancher,
        ranker: BranchRanker,
        max_candidate_attempts: int,
        alternative_store: RankedAlternativeStore,
    ) -> None:
        self.proposer = proposer
        self.gate = gate
        self.brancher = brancher
        self.ranker = ranker
        self.max_candidate_attempts = max_candidate_attempts
        self.alternatives = RankedAlternativeController(store=alternative_store)

    def decide(
        self, trajectory: TrajectoryRecord, checkpoint: CheckpointRecord
    ) -> SchedulingDecision:
        candidate = self.proposer.propose(trajectory, checkpoint)
        if candidate is None:
            return SchedulingDecision("continue_single", None, None, None, None)
        gate = self.gate.decide(candidate)
        if not gate.branch:
            return SchedulingDecision("gate_rejected", gate, None, None, None)
        group, children = self.brancher.branch(checkpoint)
        ranked = self.ranker.rank(checkpoint, children)
        if not ranked:
            raise RuntimeError("ranker returned no child after branching")
        state = BranchPointState.create(
            branch_point_id=group.branch_group_id,
            parent_checkpoint_id=checkpoint.checkpoint_id,
            candidates=tuple(
                (
                    item.child.record.child_branch_id,
                    item.child.checkpoint.checkpoint_id,
                    item.score,
                )
                for item in ranked
            ),
            max_attempts_p=self.max_candidate_attempts,
            creation_seed=self.brancher.root_seed,
            creation_config={
                "children_n": self.brancher.children,
                "local_span_steps": self.brancher.local_span_steps,
                "max_attempts_p": self.max_candidate_attempts,
            },
        )
        initial = self.alternatives.start(state)
        selected = next(
            item
            for item in ranked
            if item.child.record.child_branch_id == initial.selected.candidate_id
        )
        # The caller resumes only this checkpoint: the population collapses.
        return SchedulingDecision(
            "collapse_to_child",
            gate,
            selected,
            group.branch_group_id,
            initial.state,
        )
