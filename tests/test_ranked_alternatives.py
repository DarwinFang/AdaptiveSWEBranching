from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_swe_branching.branching.alternatives import (
    BranchPointState,
    RankedAlternativeController,
    RankedAlternativeStore,
    RankedRetryPolicy,
)
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
        first.state, active_q=0.1, rollback_threshold=0.2, restorer=restorer
    )
    third = controller.evaluate_rollback(
        second.state, active_q=0.1, rollback_threshold=0.2, restorer=restorer
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
        first.state, active_q=0.1, rollback_threshold=0.2, restorer=restorer
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
        state, active_q=0.0, rollback_threshold=0.2, restorer=restorer
    ).state
    decision = controller.evaluate_rollback(
        state, active_q=0.0, rollback_threshold=0.2, restorer=restorer
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
        state, active_q=0.0, rollback_threshold=0.2, restorer=restorer
    ).state

    restored = store.load("branch-1")
    assert restored == state
    assert restored.attempted_candidate_ids == ("candidate-a", "candidate-b")
    resumed = RankedAlternativeController(store=store).evaluate_rollback(
        restored, active_q=0.0, rollback_threshold=0.2, restorer=restorer
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


def retry_scheduler(tmp_path: Path, *, q: float) -> SelectiveBranchingScheduler:
    return SelectiveBranchingScheduler(
        proposer=None,  # type: ignore[arg-type]
        gate=None,  # type: ignore[arg-type]
        brancher=None,  # type: ignore[arg-type]
        ranker=None,  # type: ignore[arg-type]
        success_model=FixedActiveQ(q),
        retry_policy=RankedRetryPolicy(
            rollback_q_threshold=0.2,
            max_candidate_attempts_p=2,
            q_reassessment_interval_steps=3,
        ),
        alternative_store=RankedAlternativeStore(tmp_path / "controller"),
    )


def test_scheduler_waits_for_q_interval_then_rolls_back_and_exhausts(
    tmp_path: Path,
) -> None:
    scheduler = retry_scheduler(tmp_path, q=0.1)
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
    with pytest.raises(ValueError, match="rollback_q_threshold"):
        RankedRetryPolicy(
            rollback_q_threshold=1.1,
            max_candidate_attempts_p=2,
            q_reassessment_interval_steps=3,
        )
