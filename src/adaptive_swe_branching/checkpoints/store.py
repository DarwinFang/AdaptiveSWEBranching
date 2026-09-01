from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from adaptive_swe_branching.agents.base import AgentSession, AgentSnapshot
from adaptive_swe_branching.data.records import CheckpointRecord, Cost, TaskRecord
from adaptive_swe_branching.data.store import RawStore, stable_sha256
from adaptive_swe_branching.environments.container import DockerContainer, image_digest
from adaptive_swe_branching.environments.workspace import (
    copy_workspace,
    git_state,
    workspace_hash,
)


class CheckpointStore:
    """Save and verify the two parts of an executable checkpoint.

    Workspace bytes and scaffold state are stored together. A restore always
    starts a new container from the pinned image and mounts a fresh workspace
    copy; running processes and sockets are intentionally not serialised.
    """

    def __init__(self, root: Path, raw_store: RawStore | None = None):
        self.root = root.expanduser().resolve()
        self.raw_store = raw_store

    def create(
        self,
        *,
        checkpoint_id: str,
        task: TaskRecord,
        parent_trajectory_id: str,
        absolute_step: int,
        cost_to_checkpoint: Cost,
        workspace: Path,
        agent: AgentSession,
    ) -> CheckpointRecord:
        target = self.root / checkpoint_id
        if target.exists():
            return self.load(checkpoint_id)
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{checkpoint_id}.", dir=self.root))
        try:
            workspace_target = staging / "workspace"
            state_target = staging / "agent_state"
            copy_workspace(workspace, workspace_target)
            snapshot = agent.snapshot(state_target)
            head, diff, status, modified = git_state(workspace_target)
            tree_hash = workspace_hash(workspace_target)
            fingerprint = stable_sha256(
                {
                    "task_id": task.task_id,
                    "step": absolute_step,
                    "workspace_hash": tree_hash,
                    "history_hash": snapshot.history_hash,
                    "model_input_hash": snapshot.model_input_hash,
                    "image_digest": task.image_digest,
                    "scaffold_fingerprint": agent.fingerprint,
                }
            )
            record = CheckpointRecord(
                checkpoint_id=checkpoint_id,
                task_id=task.task_id,
                parent_trajectory_id=parent_trajectory_id,
                absolute_step=absolute_step,
                image_digest=task.image_digest,
                workspace_hash=tree_hash,
                base_commit=head,
                git_diff=diff,
                git_status=status,
                modified_files=modified,
                history_hash=snapshot.history_hash,
                model_input_hash=snapshot.model_input_hash,
                restore_fingerprint=fingerprint,
                cost_to_checkpoint=cost_to_checkpoint,
                workspace_ref="workspace",
                scaffold_state_ref="agent_state",
                unsupported_runtime_state=snapshot.unsupported_runtime_state,
            )
            (staging / "record.json").write_text(
                json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (staging / "agent_snapshot.json").write_text(
                json.dumps(asdict(snapshot), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staging.replace(target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        if self.raw_store is not None:
            self.raw_store.put("checkpoint", checkpoint_id, record)
        return record

    def load(self, checkpoint_id: str) -> CheckpointRecord:
        payload = json.loads(
            (self.root / checkpoint_id / "record.json").read_text(encoding="utf-8")
        )
        payload["cost_to_checkpoint"] = Cost(**payload["cost_to_checkpoint"])
        payload["modified_files"] = tuple(payload["modified_files"])
        payload["unsupported_runtime_state"] = tuple(
            payload["unsupported_runtime_state"]
        )
        return CheckpointRecord(**payload)

    def restore(
        self,
        *,
        checkpoint_id: str,
        task: TaskRecord,
        destination: Path,
        agent: AgentSession,
    ) -> tuple[CheckpointRecord, DockerContainer]:
        record = self.load(checkpoint_id)
        if task.task_id != record.task_id:
            raise ValueError("checkpoint task does not match requested task")
        if task.image_digest != record.image_digest:
            raise RuntimeError("task image digest differs from checkpoint")
        actual_digest = image_digest(task.image_name)
        if actual_digest != record.image_digest:
            message = (
                f"local image drifted: expected {record.image_digest}, "
                f"got {actual_digest}"
            )
            raise RuntimeError(message)
        source = self.root / checkpoint_id
        copy_workspace(source / record.workspace_ref, destination)
        if workspace_hash(destination) != record.workspace_hash:
            raise RuntimeError("restored workspace bytes do not match checkpoint")
        container = DockerContainer(
            image=task.image_name,
            workspace=destination,
            container_root=task.container_workdir,
            platform=task.platform,
        )
        container.start()
        try:
            agent.attach(task, container)
            snapshot = _snapshot_from_json(source / "agent_snapshot.json")
            agent.restore(source / record.scaffold_state_ref, snapshot)
            head, diff, status, _ = git_state(destination)
            current_fingerprint = stable_sha256(
                {
                    "task_id": task.task_id,
                    "step": record.absolute_step,
                    "workspace_hash": workspace_hash(destination),
                    "history_hash": snapshot.history_hash,
                    "model_input_hash": snapshot.model_input_hash,
                    "image_digest": task.image_digest,
                    "scaffold_fingerprint": agent.fingerprint,
                }
            )
            checks = {
                "base_commit": head == record.base_commit,
                "git_diff": diff == record.git_diff,
                "git_status": status == record.git_status,
                "restore_fingerprint": current_fingerprint
                == record.restore_fingerprint,
            }
            if not all(checks.values()):
                raise RuntimeError(f"checkpoint restore audit failed: {checks}")
        except Exception:
            agent.close()
            container.stop()
            shutil.rmtree(destination, ignore_errors=True)
            raise
        return record, container


def _snapshot_from_json(path: Path) -> AgentSnapshot:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unsupported_runtime_state"] = tuple(payload["unsupported_runtime_state"])
    return AgentSnapshot(**payload)
