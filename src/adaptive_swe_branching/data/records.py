from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Outcome(StrEnum):
    SOLVED = "solved"
    UNSOLVED = "unsolved"
    INVALID = "invalid"


@dataclass(frozen=True)
class Cost:
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    verifier_calls: int = 0
    wall_clock_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            steps=self.steps + other.steps,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            model_calls=self.model_calls + other.model_calls,
            tool_calls=self.tool_calls + other.tool_calls,
            verifier_calls=self.verifier_calls + other.verifier_calls,
            wall_clock_seconds=self.wall_clock_seconds + other.wall_clock_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepRecord:
    absolute_step: int
    reasoning: str
    tool_name: str
    tool_input: dict[str, Any]
    action_text: str
    observation_text: str
    is_error: bool
    explored_files_before: tuple[str, ...]
    cost: Cost
    raw_action: dict[str, Any] = field(default_factory=dict)
    raw_observation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    repository: str
    base_commit: str
    issue: str
    benchmark: str
    benchmark_version: str
    split: str
    image_name: str
    image_digest: str
    checkout_ref: str
    container_workdir: str
    platform: str
    dataset_row_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    task_id: str
    seed: int
    steps: tuple[StepRecord, ...]
    outcome: Outcome
    final_patch: str
    total_cost: Cost
    termination_reason: str
    parent_checkpoint_id: str | None = None
    final_answer: str | None = None
    invalid_reason: str | None = None
    verifier_record_id: str | None = None
    purpose: str = "agent_execution"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        return payload


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    task_id: str
    parent_trajectory_id: str
    absolute_step: int
    image_digest: str
    workspace_hash: str
    base_commit: str
    git_diff: str
    git_status: str
    modified_files: tuple[str, ...]
    history_hash: str
    model_input_hash: str
    restore_fingerprint: str
    cost_to_checkpoint: Cost
    workspace_ref: str
    scaffold_state_ref: str
    unsupported_runtime_state: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BranchGroupRecord:
    branch_group_id: str
    task_id: str
    parent_checkpoint_id: str
    child_count: int
    local_span_steps: int
    seeds: tuple[int, ...]
    child_trajectory_ids: tuple[str, ...]
    child_checkpoint_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChildBranchRecord:
    child_branch_id: str
    branch_group_id: str
    branch_index: int
    seed: int
    local_trajectory_id: str
    child_checkpoint_id: str
    downstream_trajectory_id: str | None
    downstream_outcome: Outcome | None
    downstream_cost: Cost | None
    final_patch: str | None
    termination_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.downstream_outcome is not None:
            payload["downstream_outcome"] = self.downstream_outcome.value
        return payload


@dataclass(frozen=True)
class ContinuationRecord:
    continuation_id: str
    task_id: str
    source_checkpoint_id: str
    role: str
    seed: int
    trajectory_id: str
    outcome: Outcome
    cost_from_source: Cost
    final_patch: str
    termination_reason: str
    post_parent_step_start: int = 0
    post_parent_step_count: int | None = None
    prefix_checkpoint_ids_by_depth: dict[int, str] = field(default_factory=dict)
    attempt_index: int = 0
    replacement_for_invalid_continuation_id: str | None = None
    cap_hit: bool = False
    branch_group_id: str | None = None
    child_checkpoint_id: str | None = None
    invalid_reason: str | None = None
    purpose: str = "counterfactual_label"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        return payload


@dataclass(frozen=True)
class ParentContinuationGroupRecord:
    """One parent checkpoint and its single layer of full continuations."""

    group_id: str
    task_id: str
    parent_checkpoint_id: str
    candidate_sources: tuple[str, ...]
    target_valid_k: int
    minimum_valid_k: int
    continuation_ids: tuple[str, ...]
    protocol: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChildQAuditGroupRecord:
    """Small, explicitly nested audit of true child-state success rates."""

    audit_group_id: str
    task_id: str
    parent_checkpoint_id: str
    parent_branchability: float
    branch_span_steps: int
    child_checkpoint_ids: tuple[str, ...]
    target_valid_k_per_child: int
    minimum_valid_k_per_child: int
    continuation_ids_by_child: dict[str, tuple[str, ...]]
    protocol: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CounterfactualGroupRecord:
    """Generic nested counterfactual structure; not used by Phase 4 A/B labels."""
    counterfactual_group_id: str
    task_id: str
    parent_checkpoint_id: str
    no_branch_continuation_ids: tuple[str, ...]
    branch_group_id: str
    downstream_continuation_ids_by_child: dict[str, tuple[str, ...]]
    protocol: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifierRecord:
    verifier_record_id: str
    task_id: str
    patch_sha256: str
    outcome: Outcome
    regression_passed: bool | None
    report_ref: str | None
    cost: Cost
    invalid_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        return payload


@dataclass(frozen=True)
class ScreeningPlanRecord:
    screen_version: str
    purpose: str
    dataset_revision: str
    split: str
    sampling_seed: int
    sampling_algorithm: str
    eligible_pool_size: int
    eligible_pool_sha256: str
    ordered_tasks: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScreeningRunRecord:
    screen_run_id: str
    screen_version: str
    purpose: str
    task_id: str
    repository: str
    sample_index: int
    attempt_index: int
    seed: int
    trajectory_id: str
    outcome: Outcome
    infrastructure_invalid: bool
    invalid_reason: str | None
    steps: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    wall_clock_seconds: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        return payload


@dataclass(frozen=True)
class ScreeningTaskRecord:
    screen_version: str
    purpose: str
    task_id: str
    repository: str
    sample_index: int
    n_valid: int
    n_success: int
    n_failure: int
    difficulty_class: str
    valid_run_ids: tuple[str, ...]
    all_attempt_run_ids: tuple[str, ...]
    screening_invalid_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
