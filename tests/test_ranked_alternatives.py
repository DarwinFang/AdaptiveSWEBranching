from __future__ import annotations

from pathlib import Path

from adaptive_swe_branching.branching.alternatives import (
    BranchPointState,
    RankedAlternativeController,
    RankedAlternativeStore,
)


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
