from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from adaptive_swe_branching.baselines.swe_replay.types import ArchivedTrajectory


@dataclass(frozen=True)
class CandidateSelection:
    patch: str
    trajectory_id: str
    tied: bool
    eligible_count: int


def normalise_patch(patch: str) -> str:
    lines = [line.rstrip() for line in patch.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).rstrip() + ("\n" if any(lines) else "")


def patch_identity(patch: str) -> str:
    return hashlib.sha256(normalise_patch(patch).encode("utf-8")).hexdigest()


def final_candidate(archive: tuple[ArchivedTrajectory, ...]) -> CandidateSelection:
    # Appendix Algorithm 1: regression filter, then majority vote.
    eligible = [item for item in archive if item.regression_passed]
    if not eligible:
        raise ValueError("SWE-Replay produced no regression-clean candidate")
    identities = [patch_identity(item.trajectory.final_patch) for item in eligible]
    counts = Counter(identities)
    maximum = max(counts.values())
    winners = {identity for identity, count in counts.items() if count == maximum}
    chosen = next(
        item
        for item in eligible
        if patch_identity(item.trajectory.final_patch) in winners
    )
    return CandidateSelection(
        patch=chosen.trajectory.final_patch,
        trajectory_id=chosen.trajectory.trajectory_id,
        tied=len(winners) > 1,
        eligible_count=len(eligible),
    )
