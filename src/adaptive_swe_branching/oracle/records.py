from __future__ import annotations

from dataclasses import dataclass

from adaptive_swe_branching.data.records import Cost, Outcome


@dataclass(frozen=True)
class FutureSample:
    trajectory_id: str
    seed: int
    outcome: Outcome
    cost_from_state: Cost
    final_patch: str
    termination_reason: str


@dataclass(frozen=True)
class ChildFutures:
    child_checkpoint_id: str
    local_branch_cost: Cost
    samples: tuple[FutureSample, ...]


@dataclass(frozen=True)
class CounterfactualExperiment:
    task_id: str
    parent_checkpoint_id: str
    no_branch_samples: tuple[FutureSample, ...]
    children: tuple[ChildFutures, ...]

    def __post_init__(self) -> None:
        if not self.no_branch_samples:
            raise ValueError("Oracle A needs no-branch counterfactuals")
        if len(self.children) < 2:
            raise ValueError("Oracle branching needs at least two children")
        if any(not child.samples for child in self.children):
            raise ValueError("every child needs downstream counterfactuals")
