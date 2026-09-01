from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from adaptive_swe_branching.data.records import (
    BranchGroupRecord,
    CheckpointRecord,
    ChildBranchRecord,
    TrajectoryRecord,
)
from adaptive_swe_branching.data.store import RawStore, stable_sha256
from adaptive_swe_branching.seeds import derive_seed


@dataclass(frozen=True)
class ChildExecution:
    local_trajectory: TrajectoryRecord
    child_checkpoint: CheckpointRecord


@dataclass(frozen=True)
class ExecutableChild:
    record: ChildBranchRecord
    local_trajectory: TrajectoryRecord
    checkpoint: CheckpointRecord


class BranchBackend(Protocol):
    def run_child(
        self,
        *,
        parent: CheckpointRecord,
        branch_index: int,
        seed: int,
        local_span_steps: int,
    ) -> ChildExecution: ...


class TemporaryBrancher:
    """Generate real sibling states and then return them to a caller for collapse."""

    def __init__(
        self,
        *,
        children: int,
        local_span_steps: int,
        root_seed: int,
        backend: BranchBackend,
        store: RawStore | None = None,
    ) -> None:
        if children < 2:
            raise ValueError("temporary branching needs at least two children")
        if local_span_steps < 1:
            raise ValueError("local branch span must be positive")
        self.children = children
        self.local_span_steps = local_span_steps
        self.root_seed = root_seed
        self.backend = backend
        self.store = store

    def branch(
        self, parent: CheckpointRecord
    ) -> tuple[BranchGroupRecord, tuple[ExecutableChild, ...]]:
        group_id = stable_sha256(
            {
                "parent": parent.checkpoint_id,
                "children": self.children,
                "span": self.local_span_steps,
                "root_seed": self.root_seed,
            }
        )
        seeds = tuple(
            derive_seed(self.root_seed, "branch_group", group_id, index)
            for index in range(self.children)
        )
        executions = tuple(
            self.backend.run_child(
                parent=parent,
                branch_index=index,
                seed=seed,
                local_span_steps=self.local_span_steps,
            )
            for index, seed in enumerate(seeds)
        )
        children = tuple(
            ExecutableChild(
                record=ChildBranchRecord(
                    child_branch_id=stable_sha256(
                        {"branch_group": group_id, "index": index}
                    ),
                    branch_group_id=group_id,
                    branch_index=index,
                    seed=seeds[index],
                    local_trajectory_id=execution.local_trajectory.trajectory_id,
                    child_checkpoint_id=execution.child_checkpoint.checkpoint_id,
                    downstream_trajectory_id=None,
                    downstream_outcome=None,
                    downstream_cost=None,
                    final_patch=None,
                    termination_reason=None,
                ),
                local_trajectory=execution.local_trajectory,
                checkpoint=execution.child_checkpoint,
            )
            for index, execution in enumerate(executions)
        )
        group = BranchGroupRecord(
            branch_group_id=group_id,
            task_id=parent.task_id,
            parent_checkpoint_id=parent.checkpoint_id,
            child_count=self.children,
            local_span_steps=self.local_span_steps,
            seeds=seeds,
            child_trajectory_ids=tuple(
                execution.local_trajectory.trajectory_id for execution in executions
            ),
            child_checkpoint_ids=tuple(
                execution.child_checkpoint.checkpoint_id for execution in executions
            ),
        )
        if self.store is not None:
            self.store.put("branch_group", group_id, group)
            for child in children:
                self.store.put(
                    "child_branch", child.record.child_branch_id, child.record
                )
        return group, children
