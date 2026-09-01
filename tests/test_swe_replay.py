from __future__ import annotations

import random

from conftest import make_step, make_trajectory

from adaptive_swe_branching.baselines.swe_replay.candidate import final_candidate
from adaptive_swe_branching.baselines.swe_replay.restore import restore_plan
from adaptive_swe_branching.baselines.swe_replay.runner import SWEReplayRunner
from adaptive_swe_branching.baselines.swe_replay.selector import (
    CriticalStepSelector,
    paragraph_count,
)
from adaptive_swe_branching.baselines.swe_replay.types import ArchivedTrajectory


def archived(identity: str, *, regression: bool = True, patch: str | None = None):
    trajectory = make_trajectory(
        identity,
        steps=(
            make_step(0, files=(), reasoning="one"),
            make_step(1, files=("a.py",), reasoning="one\n\ntwo\n\nthree"),
        ),
        patch=patch,
    )
    return ArchivedTrajectory(
        trajectory=trajectory,
        regression_passed=regression,
        pre_step_checkpoint_ids=(f"{identity}-0", f"{identity}-1"),
        accumulated_patches_before_step=("", "diff"),
    )


def test_selector_filters_regressions_and_uses_paper_abstraction() -> None:
    selector = CriticalStepSelector()
    clean = archived("clean")
    dirty = archived("dirty", regression=False)
    groups = selector.candidates((clean, dirty))
    assert set(groups) == {(), ("a.py",)}
    assert all(
        item[0].trajectory.trajectory_id == "clean"
        for group in groups.values()
        for item in group
    )
    selected = selector.select((clean,), random.Random(7))
    assert selected is not None
    assert selected.abstract_state in {(), ("a.py",)}
    assert paragraph_count("a\n\n b\n\n\n c") == 3


def test_restore_mode_detects_non_repository_mutation() -> None:
    source = archived("source")
    assert restore_plan(source, 1).mode == "git_diff"
    mutated = ArchivedTrajectory(
        trajectory=make_trajectory(
            "mutated",
            steps=(make_step(0), make_step(1)),
        ),
        regression_passed=True,
        pre_step_checkpoint_ids=("m0", "m1"),
        accumulated_patches_before_step=("", "diff"),
    )
    first = mutated.trajectory.steps[0]
    object.__setattr__(first, "action_text", "pip install package")
    assert restore_plan(mutated, 1).mode == "action_replay"


def test_majority_vote_filters_and_tie_is_earliest() -> None:
    choice = final_candidate(
        (
            archived("a", patch="same\n"),
            archived("b", patch="same   \n"),
            archived("c", patch="other\n"),
            archived("d", regression=False, patch="other\n"),
        )
    )
    assert choice.trajectory_id == "a"
    assert not choice.tied


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def explore(self, *, trial_index: int, seed: int) -> ArchivedTrajectory:
        self.calls.append("explore")
        return archived(f"t{trial_index}", regression=trial_index != 0)

    def exploit(self, **kwargs) -> ArchivedTrajectory:
        self.calls.append("exploit")
        return archived(f"t{kwargs['trial_index']}")


def test_full_runner_first_explores_and_empty_eligible_pool_falls_back() -> None:
    backend = FakeBackend()
    result = SWEReplayRunner(
        trials=3,
        explore_probability=0.0,
        max_steps=20,
        root_seed=4,
        backend=backend,
    ).run()
    assert backend.calls == ["explore", "explore", "exploit"]
    assert result.trials[1].note == "exploit_fallback_no_eligible_trajectory"
    assert len(result.archive) == 3
