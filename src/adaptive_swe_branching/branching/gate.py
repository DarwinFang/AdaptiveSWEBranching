from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from adaptive_swe_branching.branching.proposer import CandidateNode
from adaptive_swe_branching.branching.success_probability import (
    SuccessProbabilityModel,
)
from adaptive_swe_branching.training.labels import branchability_from_q


@dataclass(frozen=True)
class GateDecision:
    branch: bool
    score: float
    threshold: float
    predicted_branch_cost: float | None
    explanation: str


class BranchGate(Protocol):
    def decide(self, candidate: CandidateNode) -> GateDecision: ...


class SuccessProbabilityGate:
    """Judger A: transform the shared model's q at the parent state."""

    def __init__(
        self, *, model: SuccessProbabilityModel, branchability_threshold: float
    ) -> None:
        if not 0.0 <= branchability_threshold <= 1.0:
            raise ValueError("branchability threshold must be in [0, 1]")
        self.model = model
        self.threshold = branchability_threshold

    def decide(self, candidate: CandidateNode) -> GateDecision:
        estimate = self.model.predict(candidate.checkpoint)
        score = branchability_from_q(estimate.q)
        return GateDecision(
            branch=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            predicted_branch_cost=None,
            explanation=(
                f"shared_q={estimate.q:.6f}; "
                f"branchability=4q(1-q)={score:.6f}"
            ),
        )
