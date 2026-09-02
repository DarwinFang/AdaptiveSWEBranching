from __future__ import annotations

from dataclasses import dataclass

from adaptive_swe_branching.branching.alternatives import (
    AlternativeRestorer,
    BranchPointState,
    RankedAlternative,
    RankedAlternativeController,
    RankedAlternativeStore,
    RankedRetryPolicy,
)
from adaptive_swe_branching.branching.engine import TemporaryBrancher
from adaptive_swe_branching.branching.gate import BranchGate, GateDecision
from adaptive_swe_branching.branching.proposer import CandidateProposer
from adaptive_swe_branching.branching.ranker import BranchRanker, RankedChild
from adaptive_swe_branching.branching.success_probability import (
    SuccessProbabilityModel,
)
from adaptive_swe_branching.data.records import CheckpointRecord, TrajectoryRecord


@dataclass(frozen=True)
class SchedulingDecision:
    action: str
    gate: GateDecision | None
    selected: RankedChild | None
    branch_group_id: str | None
    branch_point_state: BranchPointState | None


@dataclass(frozen=True)
class ActiveBranchDecision:
    """What to do after periodically rescoring the active branch."""

    action: str
    active_q: float | None
    branch_point_state: BranchPointState
    selected_alternative: RankedAlternative | None
    termination_reason: str | None


class SelectiveBranchingScheduler:
    """Propose -> gate -> temporary branch -> rank -> collapse to one child."""

    def __init__(
        self,
        *,
        proposer: CandidateProposer,
        gate: BranchGate,
        brancher: TemporaryBrancher,
        ranker: BranchRanker,
        success_model: SuccessProbabilityModel,
        retry_policy: RankedRetryPolicy,
        alternative_store: RankedAlternativeStore,
    ) -> None:
        self.proposer = proposer
        self.gate = gate
        self.brancher = brancher
        self.ranker = ranker
        self.success_model = success_model
        self.retry_policy = retry_policy
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
            max_attempts_p=self.retry_policy.max_candidate_attempts_p,
            creation_seed=self.brancher.root_seed,
            creation_config={
                "children_n": self.brancher.children,
                "local_span_steps": self.brancher.local_span_steps,
                "max_attempts_p": self.retry_policy.max_candidate_attempts_p,
                "rollback_q_threshold": (
                    self.retry_policy.rollback_q_threshold
                ),
                "q_reassessment_interval_steps": (
                    self.retry_policy.q_reassessment_interval_steps
                ),
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

    def reassess_active_branch(
        self,
        *,
        state: BranchPointState,
        active_checkpoint: CheckpointRecord,
        steps_since_last_q_assessment: int,
        restorer: AlternativeRestorer,
    ) -> ActiveBranchDecision:
        """Continue, roll back to the next saved child, or terminate.

        The caller invokes this while following the currently selected child.
        A rollback never samples a replacement: it restores the original branch
        point and the highest-ranked child that has not already been attempted.
        """

        if steps_since_last_q_assessment < 0:
            raise ValueError("steps_since_last_q_assessment cannot be negative")
        if (
            steps_since_last_q_assessment
            < self.retry_policy.q_reassessment_interval_steps
        ):
            return ActiveBranchDecision(
                action="continue_until_q_reassessment",
                active_q=None,
                branch_point_state=state,
                selected_alternative=None,
                termination_reason=None,
            )

        estimate = self.success_model.predict(active_checkpoint)
        rollback = self.alternatives.evaluate_rollback(
            state,
            active_q=estimate.q,
            rollback_threshold=self.retry_policy.rollback_q_threshold,
            restorer=restorer,
        )
        return ActiveBranchDecision(
            action=rollback.action,
            active_q=estimate.q,
            branch_point_state=rollback.state,
            selected_alternative=rollback.selected,
            termination_reason=rollback.termination_reason,
        )
