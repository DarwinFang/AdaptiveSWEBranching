from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from conftest import make_trajectory

from adaptive_swe_branching.branching.alternatives import (
    BranchPointState,
    RankedAlternativeController,
    RankedAlternativeStore,
    RankedRetryPolicy,
)
from adaptive_swe_branching.branching.engine import ChildExecution, TemporaryBrancher
from adaptive_swe_branching.branching.ranker import SuccessProbabilityRanker
from adaptive_swe_branching.branching.scheduler import SelectiveBranchingScheduler
from adaptive_swe_branching.branching.success_probability import (
    SuccessProbabilityEstimate,
)
from adaptive_swe_branching.data.records import CheckpointRecord, Cost


class RecordingRestorer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def restore_parent(self, checkpoint_id: str) -> None:
        self.calls.append(("parent", checkpoint_id))

    def restore_candidate(self, checkpoint_id: str) -> None:
        self.calls.append(("candidate", checkpoint_id))


def new_state(*, max_attempts: int = 3) -> BranchPointState:
    return BranchPointState.create(
        branch_point_id="branch-1",
        parent_checkpoint_id="parent-checkpoint",
        candidates=(
            ("candidate-b", "checkpoint-b", 0.6),
            ("candidate-c", "checkpoint-c", 0.2),
            ("candidate-a", "checkpoint-a", 0.9),
            ("candidate-d", "checkpoint-d", 0.1),
        ),
        max_attempts_p=max_attempts,
        creation_seed=17,
        creation_config={"children": 4, "span": 6},
    )


def test_first_candidate_is_highest_scoring() -> None:
    decision = RankedAlternativeController().start(new_state())
    assert decision.action == "select_initial_candidate"
    assert decision.selected is not None
    assert decision.selected.candidate_id == "candidate-a"
    assert decision.state.attempted_candidate_ids == ("candidate-a",)


def test_successive_low_q_rollbacks_choose_untried_ranked_candidates() -> None:
    controller = RankedAlternativeController()
    restorer = RecordingRestorer()
    first = controller.start(new_state())
    second = controller.evaluate_rollback(
        first.state, active_q=0.1, low_q_threshold=0.2, restorer=restorer
    )
    third = controller.evaluate_rollback(
        second.state, active_q=0.1, low_q_threshold=0.2, restorer=restorer
    )
    assert second.selected is not None
    assert third.selected is not None
    assert second.selected.candidate_id == "candidate-b"
    assert third.selected.candidate_id == "candidate-c"
    assert second.action == "rollback_to_ranked_alternative"
    assert third.action == "rollback_to_ranked_alternative"
    assert third.state.attempted_candidate_ids == (
        "candidate-a",
        "candidate-b",
        "candidate-c",
    )
    assert len(set(third.state.attempted_candidate_ids)) == 3


def test_rollback_restores_parent_then_selected_executable_child() -> None:
    controller = RankedAlternativeController()
    restorer = RecordingRestorer()
    first = controller.start(new_state())
    controller.evaluate_rollback(
        first.state, active_q=0.1, low_q_threshold=0.2, restorer=restorer
    )
    assert restorer.calls == [
        ("parent", "parent-checkpoint"),
        ("candidate", "checkpoint-b"),
    ]


def test_after_p_attempts_next_low_q_terminates_explicitly() -> None:
    controller = RankedAlternativeController()
    restorer = RecordingRestorer()
    state = controller.start(new_state(max_attempts=2)).state
    state = controller.evaluate_rollback(
        state, active_q=0.0, low_q_threshold=0.2, restorer=restorer
    ).state
    decision = controller.evaluate_rollback(
        state, active_q=0.0, low_q_threshold=0.2, restorer=restorer
    )
    assert decision.action == "terminate"
    assert decision.termination_reason == "branch_candidates_exhausted"
    assert decision.state.exhausted
    assert decision.state.num_attempted == 2


def test_serialization_restart_preserves_attempted_candidates(tmp_path: Path) -> None:
    store = RankedAlternativeStore(tmp_path / "raw" / "branch_controller")
    controller = RankedAlternativeController(store=store)
    restorer = RecordingRestorer()
    state = controller.start(new_state()).state
    state = controller.evaluate_rollback(
        state, active_q=0.0, low_q_threshold=0.2, restorer=restorer
    ).state

    restored = store.load("branch-1")
    assert restored == state
    assert restored.attempted_candidate_ids == ("candidate-a", "candidate-b")
    resumed = RankedAlternativeController(store=store).evaluate_rollback(
        restored, active_q=0.0, low_q_threshold=0.2, restorer=restorer
    )
    assert resumed.selected is not None
    assert resumed.selected.candidate_id == "candidate-c"
    assert [event["action"] for event in store.events("branch-1")] == [
        "select_initial_candidate",
        "rollback_to_ranked_alternative",
        "rollback_to_ranked_alternative",
    ]


class FixedActiveQ:
    def __init__(self, q: float):
        self.q = q

    def predict(self, state: CheckpointRecord) -> SuccessProbabilityEstimate:
        return SuccessProbabilityEstimate(checkpoint_id=state.checkpoint_id, q=self.q)


def active_checkpoint() -> CheckpointRecord:
    return CheckpointRecord(
        checkpoint_id="active-checkpoint",
        task_id="task",
        parent_trajectory_id="active-trajectory",
        absolute_step=12,
        image_digest="sha256:x",
        workspace_hash="workspace",
        base_commit="base",
        git_diff="",
        git_status="",
        modified_files=(),
        history_hash="history",
        model_input_hash="model-input",
        restore_fingerprint="fingerprint",
        cost_to_checkpoint=Cost(steps=12),
        workspace_ref="workspace",
        scaffold_state_ref="agent-state",
    )


def retry_scheduler(
    tmp_path: Path,
    *,
    q: float,
    low_q_action: str,
    with_brancher: bool = False,
) -> SelectiveBranchingScheduler:
    class Backend:
        def run_child(self, *, parent, branch_index, seed, local_span_steps):
            return ChildExecution(
                local_trajectory=make_trajectory(f"child-{branch_index}"),
                child_checkpoint=active_checkpoint_with_id(
                    f"child-checkpoint-{branch_index}"
                ),
            )

    model = FixedActiveQ(q)
    return SelectiveBranchingScheduler(
        proposer=None,  # type: ignore[arg-type]
        gate=None,  # type: ignore[arg-type]
        brancher=(
            TemporaryBrancher(
                children=2, local_span_steps=3, root_seed=19, backend=Backend()
            )
            if with_brancher
            else None  # type: ignore[arg-type]
        ),
        ranker=(
            SuccessProbabilityRanker(model)
            if with_brancher
            else None  # type: ignore[arg-type]
        ),
        success_model=model,
        retry_policy=RankedRetryPolicy(
            low_q_threshold=0.2,
            high_q_no_branch_threshold=0.8,
            max_candidate_attempts_p=2,
            q_reassessment_interval_steps=3,
            low_q_action=low_q_action,
        ),
        alternative_store=RankedAlternativeStore(tmp_path / "controller"),
    )


def test_scheduler_default_cold_policy_does_not_restore_or_consume_candidate(
    tmp_path: Path,
) -> None:
    scheduler = retry_scheduler(tmp_path, q=0.1, low_q_action="cold_continue")
    restorer = RecordingRestorer()
    state = scheduler.alternatives.start(new_state(max_attempts=2)).state

    decision = scheduler.reassess_active_branch(
        state=state,
        active_checkpoint=active_checkpoint(),
        steps_since_last_q_assessment=3,
        restorer=restorer,
    )

    assert decision.action == "cold_continue_candidate"
    assert decision.active_q == 0.1
    assert decision.selected_alternative is None
    assert decision.branch_point_state.attempted_candidate_ids == ("candidate-a",)
    assert decision.branch_point_state.current_candidate_id == "candidate-a"
    assert not decision.branch_point_state.exhausted
    assert restorer.calls == []
    assert [
        event["action"]
        for event in scheduler.alternatives.store.events("branch-1")
    ] == ["select_initial_candidate", "cold_continue_active_candidate"]


def test_scheduler_ranked_rollback_policy_waits_then_rolls_back_and_exhausts(
    tmp_path: Path,
) -> None:
    scheduler = retry_scheduler(tmp_path, q=0.1, low_q_action="ranked_rollback")
    restorer = RecordingRestorer()
    state = scheduler.alternatives.start(new_state(max_attempts=2)).state

    not_due = scheduler.reassess_active_branch(
        state=state,
        active_checkpoint=active_checkpoint(),
        steps_since_last_q_assessment=2,
        restorer=restorer,
    )
    assert not_due.action == "continue_until_q_reassessment"
    assert not_due.active_q is None
    assert restorer.calls == []

    rollback = scheduler.reassess_active_branch(
        state=state,
        active_checkpoint=active_checkpoint(),
        steps_since_last_q_assessment=3,
        restorer=restorer,
    )
    assert rollback.action == "rollback_to_ranked_alternative"
    assert rollback.selected_alternative is not None
    assert rollback.selected_alternative.candidate_id == "candidate-b"
    assert rollback.branch_point_state.attempted_candidate_ids == (
        "candidate-a",
        "candidate-b",
    )

    exhausted = scheduler.reassess_active_branch(
        state=rollback.branch_point_state,
        active_checkpoint=active_checkpoint(),
        steps_since_last_q_assessment=3,
        restorer=restorer,
    )
    assert exhausted.action == "terminate"
    assert exhausted.termination_reason == "branch_candidates_exhausted"
    assert exhausted.branch_point_state.exhausted


def test_retry_policy_hyperparameters_are_validated() -> None:
    with pytest.raises(ValueError, match="low_q_threshold"):
        RankedRetryPolicy(
            low_q_threshold=1.1,
            high_q_no_branch_threshold=0.8,
            max_candidate_attempts_p=2,
            q_reassessment_interval_steps=3,
        )
    with pytest.raises(ValueError, match="low_q_action"):
        RankedRetryPolicy(
            low_q_threshold=0.2,
            high_q_no_branch_threshold=0.8,
            max_candidate_attempts_p=2,
            q_reassessment_interval_steps=3,
            low_q_action="invent_new_policy",
        )
    with pytest.raises(ValueError, match="high_q_no_branch_threshold"):
        RankedRetryPolicy(
            low_q_threshold=0.2,
            high_q_no_branch_threshold=1.1,
            max_candidate_attempts_p=2,
            q_reassessment_interval_steps=3,
        )


def active_checkpoint_with_id(checkpoint_id: str) -> CheckpointRecord:
    return replace(active_checkpoint(), checkpoint_id=checkpoint_id)


def test_branch_current_is_default_policy() -> None:
    policy = RankedRetryPolicy(
        low_q_threshold=0.2,
        high_q_no_branch_threshold=0.8,
        max_candidate_attempts_p=2,
        q_reassessment_interval_steps=3,
    )
    assert policy.low_q_action == "branch_current"


def test_branch_current_rolls_below_cutoff_and_collapses_to_new_child(
    tmp_path: Path,
) -> None:
    scheduler = retry_scheduler(
        tmp_path, q=0.1, low_q_action="branch_current", with_brancher=True
    )
    restorer = RecordingRestorer()
    old_state = scheduler.alternatives.start(new_state(max_attempts=2)).state

    decision = scheduler.reassess_active_branch(
        state=old_state,
        active_checkpoint=active_checkpoint(),
        steps_since_last_q_assessment=3,
        restorer=restorer,
    )

    assert decision.action == "branch_current_and_collapse"
    assert decision.selected_child is not None
    assert decision.branch_group_id == decision.branch_point_state.branch_point_id
    assert decision.superseded_branch_point_state is not None
    assert decision.superseded_branch_point_state.current_candidate_id is None
    assert restorer.calls == []
    assert scheduler.alternatives.store.events("branch-1")[-1]["action"] == (
        "branch_from_current_checkpoint"
    )


def test_branch_current_skips_branching_at_or_above_high_q_cutoff(
    tmp_path: Path,
) -> None:
    scheduler = retry_scheduler(
        tmp_path, q=0.8, low_q_action="branch_current", with_brancher=True
    )
    restorer = RecordingRestorer()
    state = scheduler.alternatives.start(new_state(max_attempts=2)).state

    decision = scheduler.reassess_active_branch(
        state=state,
        active_checkpoint=active_checkpoint(),
        steps_since_last_q_assessment=3,
        restorer=restorer,
    )

    assert decision.action == "continue_high_q_without_branching"
    assert decision.selected_child is None
    assert decision.branch_group_id is None
    assert restorer.calls == []
