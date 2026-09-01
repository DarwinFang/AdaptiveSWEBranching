from __future__ import annotations

from conftest import make_trajectory

from adaptive_swe_branching.branching.engine import ChildExecution, TemporaryBrancher
from adaptive_swe_branching.branching.gate import SuccessProbabilityGate
from adaptive_swe_branching.branching.proposer import CandidateNode
from adaptive_swe_branching.branching.ranker import SuccessProbabilityRanker
from adaptive_swe_branching.branching.success_probability import (
    SuccessProbabilityEstimate,
)
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
