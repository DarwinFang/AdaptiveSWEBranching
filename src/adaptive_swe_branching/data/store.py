from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_swe_branching.config import config_sha256

SCHEMA_VERSION = 4


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentManifest:
    schema_version: int
    experiment_name: str
    created_at_utc: str
    git_commit: str
    config: dict[str, Any]
    config_sha256: str
    root_seed: int
    step_semantics: str
    external_resources: dict[str, Any]
    host: str
    python: str

    @classmethod
    def create(
        cls,
        *,
        experiment_name: str,
        git_commit: str,
        config: dict[str, Any],
        root_seed: int,
        external_resources: dict[str, Any],
    ) -> ExperimentManifest:
        return cls(
            schema_version=SCHEMA_VERSION,
            experiment_name=experiment_name,
            created_at_utc=datetime.now(UTC).isoformat(),
            git_commit=git_commit,
            config=config,
            config_sha256=config_sha256(config),
            root_seed=root_seed,
            step_semantics="one model response plus its tool action/observation",
            external_resources=external_resources,
            host=socket.gethostname(),
            python=platform.python_version(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RawStore:
    """Write immutable raw entities; derived analysis lives elsewhere."""

    _folders = {
        "task": "tasks",
        "trajectory": "trajectories",
        "checkpoint": "checkpoint_records",
        "branch_group": "branch_groups",
        "child_branch": "child_branches",
        "continuation": "continuations",
        "parent_continuation_group": "parent_continuation_groups",
        "child_q_audit_group": "child_q_audit_groups",
        "counterfactual_group": "counterfactual_groups",
        "verifier": "verifier_records",
        "swe_replay_run": "swe_replay_runs",
        "screening_plan": "screening_plans",
        "screening_run": "screening_runs",
        "screening_task": "screening_tasks",
        "screening_cohort": "screening_cohorts",
    }

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def initialise(self, manifest: ExperimentManifest) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_immutable(self.root / "manifest.json", manifest.to_dict())
        (self.root / "derived").mkdir(exist_ok=True)

    def put(self, kind: str, record_id: str, payload: Any) -> Path:
        if kind not in self._folders:
            raise ValueError(f"unknown raw record kind: {kind}")
        data = payload.to_dict() if hasattr(payload, "to_dict") else payload
        path = self.root / self._folders[kind] / f"{record_id}.json"
        self._write_immutable(path, data)
        return path

    def get(self, kind: str, record_id: str) -> dict[str, Any]:
        path = self.root / self._folders[kind] / f"{record_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_immutable(path: Path, payload: Any) -> None:
        rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        if path.exists():
            current = path.read_text(encoding="utf-8")
            if current != rendered + "\n":
                raise FileExistsError(f"immutable record differs: {path}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            Path(temporary).replace(path)
        finally:
            Path(temporary).unlink(missing_ok=True)
