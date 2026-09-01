from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from adaptive_swe_branching.data.records import ChildBranchRecord


@dataclass(frozen=True)
class RankedChild:
    child: ChildBranchRecord
    score: float
    explanation: str


class BranchRanker(Protocol):
    def rank(
        self, children: tuple[ChildBranchRecord, ...]
    ) -> tuple[RankedChild, ...]: ...
