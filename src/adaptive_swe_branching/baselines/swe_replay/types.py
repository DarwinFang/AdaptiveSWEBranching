from __future__ import annotations

from dataclasses import dataclass, field

from adaptive_swe_branching.data.records import Cost, TrajectoryRecord


@dataclass(frozen=True)
class ArchivedTrajectory:
    trajectory: TrajectoryRecord
    regression_passed: bool
    pre_step_checkpoint_ids: tuple[str, ...]
    accumulated_patches_before_step: tuple[str, ...]
    generation_cost: Cost = field(default_factory=Cost)

    def __post_init__(self) -> None:
        size = len(self.trajectory.steps)
        if len(self.pre_step_checkpoint_ids) != size:
            raise ValueError("one pre-step checkpoint id is required per step")
        if len(self.accumulated_patches_before_step) != size:
            raise ValueError("one accumulated prefix patch is required per step")


@dataclass(frozen=True)
class SelectedStep:
    trajectory_id: str
    step_index: int
    abstract_state: tuple[str, ...]
    abstract_state_frequency: int
    reasoning_paragraphs: int


@dataclass(frozen=True)
class ReplayTrial:
    trial_index: int
    mode: str
    seed: int
    trajectory_id: str
    selected_step: SelectedStep | None = None
    note: str | None = None


@dataclass(frozen=True)
class ReplayResult:
    trials: tuple[ReplayTrial, ...]
    archive: tuple[ArchivedTrajectory, ...]
    selected_patch: str
    selected_trajectory_id: str
    majority_tied: bool
