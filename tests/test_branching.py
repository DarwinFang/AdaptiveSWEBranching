from __future__ import annotations

from conftest import make_trajectory

from adaptive_swe_branching.branching.engine import ChildExecution, TemporaryBrancher
from adaptive_swe_branching.branching.gate import SuccessProbabilityGate
from adaptive_swe_branching.branching.parent_candidates import (
    merge_swe_replay_candidate,
    stratified_random_parents,
)
from adaptive_swe_branching.branching.proposer import CandidateNode
from adaptive_swe_branching.branching.ranker import SuccessProbabilityRanker
from adaptive_swe_branching.branching.success_probability import (
    SuccessProbabilityEstimate,
)
from adaptive_swe_branching.data.records import CheckpointRecord, Cost


def checkpoint(identity: str, *, step: int = 6) -> CheckpointRecord:
    return CheckpointRecord(
        checkpoint_id=identity,
        task_id="task",
        parent_trajectory_id="source",
        absolute_step=step,
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


class FixedQModel:
    def __init__(self, values: dict[str, float]):
        self.values = values

    def predict(self, state: CheckpointRecord) -> SuccessProbabilityEstimate:
        return SuccessProbabilityEstimate(
            checkpoint_id=state.checkpoint_id,
            q=self.values[state.checkpoint_id],
        )


def test_gate_and_ranker_share_one_q_model_with_different_rules() -> None:
    model = FixedQModel(
        {"parent": 0.5, "child-checkpoint-0": 0.2, "child-checkpoint-1": 0.8}
    )
    gate = SuccessProbabilityGate(model=model, branchability_threshold=0.75)
    assert gate.decide(CandidateNode(checkpoint("parent"), "test")).branch

    _, children = TemporaryBrancher(
        children=2, local_span_steps=3, root_seed=9, backend=Backend()
    ).branch(checkpoint("parent"))
    ranked = SuccessProbabilityRanker(model).rank(checkpoint("parent"), children)
    assert ranked[0].child.checkpoint.checkpoint_id == "child-checkpoint-1"
    assert ranked[0].score == 0.8


def test_relative_thirds_and_swe_replay_overlap_keep_four_unique_parents() -> None:
    eligible = tuple(
        checkpoint(f"checkpoint-{step}", step=step) for step in range(1, 10)
    )
    random_candidates = stratified_random_parents(
        eligible, root_seed=11, task_id="task"
    )
    assert 1 <= random_candidates[0].checkpoint.absolute_step <= 3
    assert 4 <= random_candidates[1].checkpoint.absolute_step <= 6
    assert 7 <= random_candidates[2].checkpoint.absolute_step <= 9

    overlap_id = random_candidates[1].checkpoint.checkpoint_id
    merged = merge_swe_replay_candidate(
        eligible,
        random_candidates,
        swe_replay_checkpoint_id=overlap_id,
        root_seed=11,
        task_id="task",
    )
    assert len(merged) == 4
    assert len({item.checkpoint.checkpoint_id for item in merged}) == 4
    overlapped = next(
        item for item in merged if item.checkpoint.checkpoint_id == overlap_id
    )
    assert overlapped.candidate_sources == ("random_middle", "swe_replay")
    assert sum("random_middle" in item.candidate_sources for item in merged) == 2
