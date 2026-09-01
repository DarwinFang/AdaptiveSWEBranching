from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_swe_branching.data.records import Cost, Outcome, VerifierRecord
from adaptive_swe_branching.data.store import RawStore
from adaptive_swe_branching.environments.swesmith import SWESmithTask


@dataclass(frozen=True)
class Verification:
    record: VerifierRecord
    report: dict[str, Any]


class SWESmithVerifier:
    """Thin adapter around the frozen official SWE-smith evaluation harness."""

    def __init__(
        self,
        *,
        harness_path: Path,
        timeout_seconds: float,
        store: RawStore | None = None,
    ) -> None:
        self.harness_path = harness_path.expanduser().resolve()
        self.timeout_seconds = timeout_seconds
        self.store = store

    def verify(self, task: SWESmithTask, patch: str) -> Verification:
        patch_hash = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        record_id = hashlib.sha256(
            f"{task.record.task_id}\0{patch_hash}".encode()
        ).hexdigest()
        if self.store is not None:
            try:
                cached = self.store.get("verifier", record_id)
                return Verification(
                    _record_from_dict(cached), cached.get("details", {})
                )
            except FileNotFoundError:
                pass

        started = time.monotonic()
        report: dict[str, Any] = {}
        invalid: str | None = None
        with tempfile.TemporaryDirectory(prefix="asb-verify-") as temporary:
            root = Path(temporary)
            dataset_path = root / "task.json"
            predictions_path = root / "predictions.jsonl"
            dataset_path.write_text(
                json.dumps([task.verifier_row]) + "\n", encoding="utf-8"
            )
            predictions_path.write_text(
                json.dumps(
                    {
                        "instance_id": task.record.task_id,
                        "model_patch": patch,
                        "model_name_or_path": "adaptive_swe_branching",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            run_id = f"asb-{uuid.uuid4().hex[:12]}"
            command = [
                "python",
                "-m",
                "swesmith.harness.eval",
                "--dataset_path",
                str(dataset_path),
                "--predictions_path",
                str(predictions_path),
                "--run_id",
                run_id,
                "--workers",
                "1",
                "--instance_ids",
                task.record.task_id,
            ]
            try:
                environment = dict(os.environ)
                environment["PYTHONPATH"] = (
                    str(self.harness_path)
                    + os.pathsep
                    + environment.get("PYTHONPATH", "")
                )
                completed = subprocess.run(
                    command,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=environment,
                )
                report_path = (
                    root
                    / "logs"
                    / "run_evaluation"
                    / run_id
                    / task.record.task_id
                    / "report.json"
                )
                if report_path.exists():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                else:
                    invalid = (
                        f"harness produced no report (exit={completed.returncode}): "
                        f"{completed.stderr[-2000:]}"
                    )
            except subprocess.TimeoutExpired:
                invalid = f"verifier timeout after {self.timeout_seconds}s"

        if invalid is not None:
            outcome = Outcome.INVALID
            regression_passed = None
        else:
            resolved = _find_bool(report, "resolved")
            outcome = Outcome.SOLVED if resolved else Outcome.UNSOLVED
            failures = _pass_to_pass_failures(report)
            regression_passed = not failures

        elapsed = time.monotonic() - started
        record = VerifierRecord(
            verifier_record_id=record_id,
            task_id=task.record.task_id,
            patch_sha256=patch_hash,
            outcome=outcome,
            regression_passed=regression_passed,
            report_ref=None,
            cost=Cost(verifier_calls=1, wall_clock_seconds=elapsed),
            invalid_reason=invalid,
            details=report,
        )
        if self.store is not None:
            self.store.put("verifier", record_id, record)
        return Verification(record=record, report=report)


def _find_bool(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if key in value:
            return bool(value[key])
        return any(_find_bool(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_find_bool(child, key) for child in value)
    return False


def _pass_to_pass_failures(report: dict[str, Any]) -> list[Any]:
    def walk(value: Any) -> list[Any] | None:
        if isinstance(value, dict):
            status = value.get("PASS_TO_PASS")
            if isinstance(status, dict) and isinstance(status.get("failure"), list):
                return status["failure"]
            for child in value.values():
                found = walk(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = walk(child)
                if found is not None:
                    return found
        return None

    return walk(report) or []


def _record_from_dict(payload: dict[str, Any]) -> VerifierRecord:
    return VerifierRecord(
        verifier_record_id=payload["verifier_record_id"],
        task_id=payload["task_id"],
        patch_sha256=payload["patch_sha256"],
        outcome=Outcome(payload["outcome"]),
        regression_passed=payload["regression_passed"],
        report_ref=payload.get("report_ref"),
        cost=Cost(**payload["cost"]),
        invalid_reason=payload.get("invalid_reason"),
        details=payload.get("details", {}),
    )
