from __future__ import annotations

import random
from typing import Protocol

from adaptive_swe_branching.baselines.swe_replay.candidate import final_candidate
from adaptive_swe_branching.baselines.swe_replay.restore import (
    RestorePlan,
    restore_plan,
)
from adaptive_swe_branching.baselines.swe_replay.selector import CriticalStepSelector
from adaptive_swe_branching.baselines.swe_replay.types import (
    ArchivedTrajectory,
    ReplayResult,
    ReplayTrial,
    SelectedStep,
)
from adaptive_swe_branching.seeds import derive_seed


class ReplayBackend(Protocol):
    def explore(self, *, trial_index: int, seed: int) -> ArchivedTrajectory: ...

    def exploit(
        self,
        *,
        source: ArchivedTrajectory,
        selected: SelectedStep,
        restore: RestorePlan,
        trial_index: int,
        seed: int,
        suffix_step_budget: int,
    ) -> ArchivedTrajectory: ...


class SWEReplayRunner:
    """Complete Appendix Algorithm 1 orchestration, independent of our gate."""

    def __init__(
        self,
        *,
        trials: int,
        explore_probability: float,
        max_steps: int,
        root_seed: int,
        backend: ReplayBackend,
        selector: CriticalStepSelector | None = None,
    ) -> None:
        if trials < 1:
            raise ValueError("SWE-Replay needs at least one trial")
        self.trials = trials
        self.explore_probability = explore_probability
        self.max_steps = max_steps
        self.root_seed = root_seed
        self.backend = backend
        self.selector = selector or CriticalStepSelector()

    def run(self) -> ReplayResult:
        archive: list[ArchivedTrajectory] = []
        trials: list[ReplayTrial] = []
        for index in range(self.trials):
            seed = derive_seed(self.root_seed, "swe_replay", index)
            rng = random.Random(seed)
            # Paper Algorithm 1 lines 1–3: first trial is always explore.
            explore = index == 0 or rng.random() < self.explore_probability
            selected: SelectedStep | None = None
            note: str | None = None
            if not explore:
                selected = self.selector.select(tuple(archive), rng)
                if selected is None:
                    explore = True
                    note = "exploit_fallback_no_eligible_trajectory"
            if explore:
                created = self.backend.explore(trial_index=index, seed=seed)
                mode = "explore"
            else:
                source = next(
                    item
                    for item in archive
                    if item.trajectory.trajectory_id == selected.trajectory_id
                )
                plan = restore_plan(source, selected.step_index)
                # Frozen interpretation: one common absolute trajectory cap.
                suffix_budget = max(self.max_steps - selected.step_index, 0)
                created = self.backend.exploit(
                    source=source,
                    selected=selected,
                    restore=plan,
                    trial_index=index,
                    seed=seed,
                    suffix_step_budget=suffix_budget,
                )
                mode = "exploit"
            # Paper §2.3: every resulting trajectory enters the archive; only
            # replay-source and final-candidate filtering inspect regression.
            archive.append(created)
            trials.append(
                ReplayTrial(
                    trial_index=index,
                    mode=mode,
                    seed=seed,
                    trajectory_id=created.trajectory.trajectory_id,
                    selected_step=selected,
                    note=note,
                )
            )
        selected_patch = final_candidate(tuple(archive))
        return ReplayResult(
            trials=tuple(trials),
            archive=tuple(archive),
            selected_patch=selected_patch.patch,
            selected_trajectory_id=selected_patch.trajectory_id,
            majority_tied=selected_patch.tied,
        )
