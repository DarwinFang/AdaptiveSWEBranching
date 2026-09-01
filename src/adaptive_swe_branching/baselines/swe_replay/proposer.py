from __future__ import annotations

import random

from adaptive_swe_branching.baselines.swe_replay.selector import CriticalStepSelector
from adaptive_swe_branching.baselines.swe_replay.types import (
    ArchivedTrajectory,
    SelectedStep,
)


class SWEReplayCriticalStepProposer:
    """Cheap candidate proposer only; it never decides whether to branch."""

    def __init__(self, selector: CriticalStepSelector | None = None):
        self.selector = selector or CriticalStepSelector()

    def propose(
        self, archive: tuple[ArchivedTrajectory, ...], *, seed: int
    ) -> SelectedStep | None:
        return self.selector.select(archive, random.Random(seed))
