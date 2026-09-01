from __future__ import annotations

from conftest import make_trajectory

from adaptive_swe_branching.branching.engine import ChildExecution, TemporaryBrancher
from adaptive_swe_branching.data.records import CheckpointRecord, Cost


def checkpoint(identity: str) -> CheckpointRecord:
    return CheckpointRecord(
        checkpoint_id=identity,
        task_id="task",
        parent_trajectory_id="source",
        absolute_step=6,
        image_digest="sha256:x",
        workspace_hash="w",
        base_commit="base",
        git_diff="",
        git_status="",
        modified_files=(),
        history_hash="h",
        model_input_hash="m",
        restore_fingerprint="f",
        cost_to_checkpoint=Cost(steps=6),
        workspace_ref="workspace",
        scaffold_state_ref="state",
    )


class Backend:
    def run_child(self, *, parent, branch_index, seed, local_span_steps):
        return ChildExecution(
            local_trajectory=make_trajectory(f"child-trajectory-{branch_index}"),
            child_checkpoint=checkpoint(f"child-checkpoint-{branch_index}"),
        )


def test_temporary_brancher_has_configurable_count_span_and_stable_seeds() -> None:
    first = TemporaryBrancher(
        children=4, local_span_steps=3, root_seed=9, backend=Backend()
    ).branch(checkpoint("parent"))
    second = TemporaryBrancher(
        children=4, local_span_steps=3, root_seed=9, backend=Backend()
    ).branch(checkpoint("parent"))
    assert first[0].child_count == 4
    assert first[0].local_span_steps == 3
    assert first[0].seeds == second[0].seeds
    assert len(set(first[0].seeds)) == 4
