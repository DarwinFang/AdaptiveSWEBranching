from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import make_step, make_trajectory

from adaptive_swe_branching.data.records import (
    ContinuationRecord,
    Cost,
    Outcome,
)
from adaptive_swe_branching.training.labels import (
    ContinuationEvidence,
    branchability_from_q,
    parent_q_target,
    prefix_q_targets,
)


def continuation(
    identity: str,
    outcome: Outcome,
    *,
    step_count: int = 6,
) -> tuple[ContinuationRecord, ContinuationEvidence]:
    trajectory = make_trajectory(
        identity,
        solved=outcome == Outcome.SOLVED,
        steps=tuple(make_step(index) for index in range(step_count)),
    )
    trajectory = replace(
        trajectory, parent_checkpoint_id="parent", outcome=outcome
    )
    record = ContinuationRecord(
        continuation_id=f"cont-{identity}",
        task_id="task",
        source_checkpoint_id="parent",
        role="same_parent_full",
        seed=1,
        trajectory_id=identity,
        outcome=outcome,
        cost_from_source=Cost(input_tokens=1),
        final_patch="",
        termination_reason="finish",
        post_parent_step_count=step_count,
        prefix_checkpoint_ids_by_depth={
            depth: f"cp-{identity}-{depth}"
            for depth in (1, 2, 4, 6)
            if depth <= step_count
        },
    )
    return record, ContinuationEvidence(record, trajectory)


def test_parent_and_prefixes_use_the_same_q_target_type() -> None:
    solved_record, solved = continuation("yes", Outcome.SOLVED)
    failed_record, failed = continuation("no", Outcome.UNSOLVED)
    parent = parent_q_target("parent", (solved_record, failed_record))
    prefixes = prefix_q_targets("parent", (solved, failed), depths=(2,))
    assert parent.empirical_q == 0.5
    assert (parent.successes, parent.trials) == (1, 2)
    assert [item.empirical_q for item in prefixes] == [1.0, 0.0]
    assert all(type(item) is type(parent) for item in prefixes)


def test_a_is_an_inference_transform_of_q() -> None:
    assert branchability_from_q(0.5) == 1.0
    assert branchability_from_q(0.0) == 0.0
    assert branchability_from_q(1.0) == 0.0


def test_targets_support_variable_k_and_exclude_invalid() -> None:
    solved, _ = continuation("a", Outcome.SOLVED)
    failed, _ = continuation("b", Outcome.UNSOLVED)
    invalid, _ = continuation("c", Outcome.INVALID)
    target = parent_q_target("parent", (solved, failed, invalid))
    assert target.trials == 2
    assert target.invalid_continuation_ids == ("cont-c",)


def test_prefix_target_requires_saved_executable_state() -> None:
    record, evidence = continuation("short", Outcome.SOLVED, step_count=2)
    record = replace(record, prefix_checkpoint_ids_by_depth={})
    with pytest.raises(ValueError, match="missing the saved checkpoint"):
        prefix_q_targets(
            "parent", (ContinuationEvidence(record, evidence.trajectory),), depths=(2,)
        )


def test_targets_reject_cross_parent_data() -> None:
    record, _ = continuation("a", Outcome.SOLVED)
    with pytest.raises(ValueError):
        parent_q_target("different", (record,))
