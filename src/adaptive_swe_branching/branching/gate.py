from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from adaptive_swe_branching.branching.proposer import CandidateNode


@dataclass(frozen=True)
class GateDecision:
    branch: bool
    score: float
    threshold: float
    predicted_branch_cost: float | None
    explanation: str


class BranchGate(Protocol):
    def decide(self, candidate: CandidateNode) -> GateDecision: ...
