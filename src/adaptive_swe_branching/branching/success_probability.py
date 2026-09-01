from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from adaptive_swe_branching.data.records import CheckpointRecord


@dataclass(frozen=True)
class SuccessProbabilityEstimate:
    checkpoint_id: str
    q: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.q <= 1.0:
            raise ValueError("predicted success probability must be in [0, 1]")


class SuccessProbabilityModel(Protocol):
    """The one learned model shared by gate A and ranker B."""

    def predict(self, state: CheckpointRecord) -> SuccessProbabilityEstimate: ...
