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
    selected_child: RankedChild | None
    branch_group_id: str | None
    superseded_branch_point_state: BranchPointState | None
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
        if self.retry_policy.low_q_action == "branch_current":
            estimate = self.success_model.predict(checkpoint)
            cutoff = self.retry_policy.high_q_no_branch_threshold
            gate = GateDecision(
                branch=estimate.q < cutoff,
                score=1.0 - estimate.q,
                threshold=1.0 - cutoff,
                predicted_branch_cost=None,
                explanation=(
                    f"shared_q={estimate.q:.6f}; branch_current unless "
                    f"q>={cutoff:.6f}"
                ),
            )
        else:
            gate = self.gate.decide(candidate)
        if not gate.branch:
            return SchedulingDecision("gate_rejected", gate, None, None, None)
        group_id, selected, state = self._branch_and_collapse(checkpoint)
        return SchedulingDecision(
            "collapse_to_child", gate, selected, group_id, state
        )

    def _branch_and_collapse(
        self, checkpoint: CheckpointRecord
    ) -> tuple[str, RankedChild, BranchPointState]:
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
                "low_q_action": self.retry_policy.low_q_action,
                "low_q_threshold": self.retry_policy.low_q_threshold,
                "high_q_no_branch_threshold": (
                    self.retry_policy.high_q_no_branch_threshold
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
        return group.branch_group_id, selected, initial.state

    def reassess_active_branch(
        self,
        *,
        state: BranchPointState,
        active_checkpoint: CheckpointRecord,
        steps_since_last_q_assessment: int,
        restorer: AlternativeRestorer,
    ) -> ActiveBranchDecision:
        """Continue, branch here, roll back, or terminate.

        The caller invokes this while following the currently selected child.
        ``cold_continue`` leaves low q alone. ``ranked_rollback`` restores the
        original branch point and its highest-ranked untried child.
        ``branch_current`` creates and collapses a fresh branch group unless q is
        already at or above its high-q no-branch cutoff.
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
                selected_child=None,
                branch_group_id=None,
                superseded_branch_point_state=None,
                termination_reason=None,
            )

        estimate = self.success_model.predict(active_checkpoint)
        if self.retry_policy.low_q_action == "branch_current":
            cutoff = self.retry_policy.high_q_no_branch_threshold
            if estimate.q >= cutoff:
                updated = self.alternatives.record_high_q_continue(
                    state,
                    active_q=estimate.q,
                    high_q_no_branch_threshold=cutoff,
                )
                return ActiveBranchDecision(
                    action="continue_high_q_without_branching",
                    active_q=estimate.q,
                    branch_point_state=updated,
                    selected_alternative=None,
                    selected_child=None,
                    branch_group_id=None,
                    superseded_branch_point_state=None,
                    termination_reason=None,
                )
            group_id, selected, next_state = self._branch_and_collapse(
                active_checkpoint
            )
            superseded = self.alternatives.record_branch_current(
                state,
                active_q=estimate.q,
                high_q_no_branch_threshold=cutoff,
                spawned_branch_point_id=group_id,
                selected_candidate_id=selected.child.record.child_branch_id,
            )
            return ActiveBranchDecision(
                action="branch_current_and_collapse",
                active_q=estimate.q,
                branch_point_state=next_state,
                selected_alternative=None,
                selected_child=selected,
                branch_group_id=group_id,
                superseded_branch_point_state=superseded,
                termination_reason=None,
            )
        if (
            estimate.q < self.retry_policy.low_q_threshold
            and self.retry_policy.low_q_action == "cold_continue"
        ):
            rollback = self.alternatives.record_cold_continue(
                state,
                active_q=estimate.q,
                low_q_threshold=self.retry_policy.low_q_threshold,
            )
        else:
            rollback = self.alternatives.evaluate_rollback(
                state,
                active_q=estimate.q,
                low_q_threshold=self.retry_policy.low_q_threshold,
                restorer=restorer,
            )
        return ActiveBranchDecision(
            action=rollback.action,
            active_q=estimate.q,
            branch_point_state=rollback.state,
            selected_alternative=rollback.selected,
            selected_child=None,
            branch_group_id=None,
            superseded_branch_point_state=None,
            termination_reason=rollback.termination_reason,
        )
