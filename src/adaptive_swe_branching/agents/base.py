from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from adaptive_swe_branching.data.records import Cost, StepRecord, TaskRecord
from adaptive_swe_branching.environments.container import DockerContainer


@dataclass(frozen=True)
class StepResult:
    step: StepRecord | None
    cost: Cost
    finished: bool
    final_answer: str | None = None


@dataclass(frozen=True)
class AgentSnapshot:
    format: str
    state_file: str
    completed_steps: int
    history_hash: str
    model_input_hash: str
    unsupported_runtime_state: tuple[str, ...]


class AgentSession(Protocol):
    @property
    def fingerprint(self) -> str: ...

    @property
    def finished(self) -> bool: ...

    @property
    def steps(self) -> tuple[StepRecord, ...]: ...

    @property
    def final_answer(self) -> str | None: ...

    @property
    def termination_reason(self) -> str | None: ...

    def start(self, task: TaskRecord, container: DockerContainer) -> None: ...

    def attach(self, task: TaskRecord, container: DockerContainer) -> None: ...

    def step(self) -> StepResult: ...

    def snapshot(self, destination: Path) -> AgentSnapshot: ...

    def restore(self, state_dir: Path, snapshot: AgentSnapshot) -> None: ...

    def history_payload(self) -> tuple[dict[str, Any], ...]: ...

    def close(self) -> None: ...
