from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from adaptive_swe_branching.branching.engine import ExecutableChild
from adaptive_swe_branching.branching.success_probability import (
    SuccessProbabilityModel,
)
from adaptive_swe_branching.data.records import CheckpointRecord


@dataclass(frozen=True)
class RankedChild:
    child: ExecutableChild
    score: float
    explanation: str


class BranchRanker(Protocol):
    def rank(
        self,
        parent: CheckpointRecord,
        children: tuple[ExecutableChild, ...],
    ) -> tuple[RankedChild, ...]: ...


class SuccessProbabilityRanker:
    """Judger B: apply the same q model to child states and maximize q."""

    def __init__(self, model: SuccessProbabilityModel) -> None:
        self.model = model

    def rank(
        self,
        parent: CheckpointRecord,
        children: tuple[ExecutableChild, ...],
    ) -> tuple[RankedChild, ...]:
        del parent  # The child checkpoint already contains its complete state.
        ranked = tuple(
            RankedChild(
                child=child,
                score=(estimate := self.model.predict(child.checkpoint)).q,
                explanation=f"shared_q={estimate.q:.6f}",
            )
            for child in children
        )
        return tuple(sorted(ranked, key=lambda item: item.score, reverse=True))
