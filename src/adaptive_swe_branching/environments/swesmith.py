from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.dataset as ds

from adaptive_swe_branching.data.records import TaskRecord
from adaptive_swe_branching.data.store import stable_sha256
from adaptive_swe_branching.environments.container import image_digest
from adaptive_swe_branching.environments.workspace import copy_workspace, run_git

SAFE_COLUMNS = ("instance_id", "problem_statement", "repo", "image_name")


@dataclass(frozen=True)
class SWESmithTask:
    record: TaskRecord
    row: dict[str, Any]
    verifier_row: dict[str, Any]


class SWESmithDataset:
    """Read a frozen SWE-smith snapshot without exposing gold fields to the agent."""

    def __init__(
        self,
        *,
        snapshot: Path,
        revision: str,
        split_manifest: Path,
        container_workdir: str = "/testbed",
        platform: str = "linux/amd64",
    ) -> None:
        self.snapshot = snapshot.expanduser().resolve()
        self.revision = revision
        self.split_manifest = split_manifest.expanduser().resolve()
        self.container_workdir = container_workdir
        self.platform = platform
        self._splits = json.loads(self.split_manifest.read_text(encoding="utf-8"))
        raw_assignments = self._splits.get("assignments", self._splits)
        if isinstance(raw_assignments, list):
            self._assignments = {
                item["instance_id"]: item["split"] for item in raw_assignments
            }
        else:
            self._assignments = {
                task_id: (
                    value.get("split") if isinstance(value, dict) else value
                )
                for task_id, value in raw_assignments.items()
            }

    def _split_for(self, task_id: str) -> str:
        value = self._assignments.get(task_id)
        if not isinstance(value, str):
            raise KeyError(f"task is absent from split manifest: {task_id}")
        return value

    def screening_pool(self, *, split: str) -> tuple[tuple[str, str], ...]:
        """Return deterministic eligible task/repository pairs for screening."""
        data_path = self.snapshot / "data"
        dataset = ds.dataset(
            data_path if data_path.exists() else self.snapshot,
            format="parquet",
            exclude_invalid_files=True,
        )
        table = dataset.to_table(
            columns=["instance_id", "repo", "problem_statement"]
        )
        rows = table.to_pylist()
        eligible = (
            (str(row["instance_id"]), str(row["repo"]))
            for row in rows
            if self._assignments.get(str(row["instance_id"])) == split
            and str(row.get("problem_statement") or "").strip()
        )
        return tuple(sorted(eligible))

    def load(self, task_id: str, *, base_commit: str = "") -> SWESmithTask:
        data_path = self.snapshot / "data"
        dataset = ds.dataset(
            data_path if data_path.exists() else self.snapshot,
            format="parquet",
            exclude_invalid_files=True,
        )
        table = dataset.to_table(
            filter=pc.field("instance_id") == task_id,
        )
        if table.num_rows != 1:
            raise ValueError(f"expected one row for {task_id}, found {table.num_rows}")
        verifier_row = table.to_pylist()[0]
        row = {name: verifier_row[name] for name in SAFE_COLUMNS}
        digest = image_digest(row["image_name"])
        split = self._split_for(task_id)
        record = TaskRecord(
            task_id=task_id,
            repository=row["repo"],
            base_commit=base_commit,
            issue=row["problem_statement"],
            benchmark="SWE-bench/SWE-smith-py",
            benchmark_version=self.revision,
            split=split,
            image_name=row["image_name"],
            image_digest=digest,
            checkout_ref=task_id,
            container_workdir=self.container_workdir,
            platform=self.platform,
            dataset_row_sha256=stable_sha256(row),
        )
        return SWESmithTask(record=record, row=row, verifier_row=verifier_row)


class WorkspaceCache:
    """Copy an existing immutable task template or materialise it once from Docker."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve() / "swe_smith_py"

    def _template(self, task_id: str) -> Path:
        return self.root / task_id

    def prepare(self, task: SWESmithTask, destination: Path) -> SWESmithTask:
        template = self._template(task.record.task_id)
        if not template.exists():
            self._materialise(task, template)
        resolved = self.resolve(task)
        copy_workspace(template, destination)
        marker = destination / ".recov_identity.json"
        if marker.exists():
            tracked = run_git(destination, "ls-files", "--", marker.name).strip()
            if tracked:
                raise RuntimeError(
                    ".recov_identity.json is tracked by the benchmark repository; "
                    "refusing to treat it as an external cache marker"
                )
            marker.unlink()
        return resolved

    def resolve(self, task: SWESmithTask) -> SWESmithTask:
        template = self._template(task.record.task_id)
        if not template.exists():
            self._materialise(task, template)
        head = run_git(template, "rev-parse", "HEAD").strip()
        record = TaskRecord(**{**task.record.to_dict(), "base_commit": head})
        return SWESmithTask(
            record=record,
            row=task.row,
            verifier_row=task.verifier_row,
        )

    def _materialise(self, task: SWESmithTask, template: Path) -> None:
        if template.exists():
            return
        temporary = template.parent / f".{template.name}.building"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "docker",
                "create",
                "--platform",
                task.record.platform,
                "--name",
                f"asb-cache-{hashlib.sha256(task.record.task_id.encode()).hexdigest()[:12]}",
                task.record.image_name,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        name = (
            f"asb-cache-{hashlib.sha256(task.record.task_id.encode()).hexdigest()[:12]}"
        )
        try:
            subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{name}:{task.record.container_workdir}/.",
                    str(temporary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)
        run_git(temporary, "checkout", "--force", task.record.checkout_ref)
        temporary.replace(template)
