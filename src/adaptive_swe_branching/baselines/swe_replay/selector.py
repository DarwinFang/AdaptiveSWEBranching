from __future__ import annotations

import math
import random
import re
from collections import defaultdict

from adaptive_swe_branching.baselines.swe_replay.types import (
    ArchivedTrajectory,
    SelectedStep,
)


def paragraph_count(reasoning: str) -> int:
    stripped = reasoning.replace("\r\n", "\n").strip()
    if not stripped:
        return 0
    return len([part for part in re.split(r"\n\s*\n+", stripped) if part.strip()])


def _sample_softmax(values: list[float], rng: random.Random) -> int:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    point = rng.random() * sum(weights)
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if point <= cumulative:
            return index
    return len(weights) - 1


class CriticalStepSelector:
    """Paper §2.1: exact file-set state, then reasoning-intensity sampling."""

    def candidates(
        self, archive: tuple[ArchivedTrajectory, ...]
    ) -> dict[tuple[str, ...], list[tuple[ArchivedTrajectory, int]]]:
        groups: dict[tuple[str, ...], list[tuple[ArchivedTrajectory, int]]] = (
            defaultdict(list)
        )
        # Paper §2.1.1: regression-failing trajectories are not replay sources.
        for archived in archive:
            if not archived.regression_passed:
                continue
            for index, step in enumerate(archived.trajectory.steps):
                groups[tuple(sorted(step.explored_files_before))].append(
                    (archived, index)
                )
        return dict(groups)

    def select(
        self, archive: tuple[ArchivedTrajectory, ...], rng: random.Random
    ) -> SelectedStep | None:
        groups = self.candidates(archive)
        if not groups:
            return None
        states = sorted(groups)
        # Paper Eq. 1: P(S_i) = softmax(1 / v_i).
        state_index = _sample_softmax(
            [1.0 / len(groups[state]) for state in states], rng
        )
        state = states[state_index]
        concrete = sorted(
            groups[state],
            key=lambda value: (value[0].trajectory.trajectory_id, value[1]),
        )
        # Paper Eq. 2: P(step) = softmax(reasoning paragraph count).
        step_index = _sample_softmax(
            [
                paragraph_count(item[0].trajectory.steps[item[1]].reasoning)
                for item in concrete
            ],
            rng,
        )
        archived, index = concrete[step_index]
        return SelectedStep(
            trajectory_id=archived.trajectory.trajectory_id,
            step_index=index,
            abstract_state=state,
            abstract_state_frequency=len(concrete),
            reasoning_paragraphs=paragraph_count(
                archived.trajectory.steps[index].reasoning
            ),
        )
