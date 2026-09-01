from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from adaptive_swe_branching.agents.base import AgentSession
from adaptive_swe_branching.agents.openhands_tools import ContainerFileEditorExecutor
from adaptive_swe_branching.baselines.swe_replay.restore import RestorePlan
from adaptive_swe_branching.baselines.swe_replay.types import (
    ArchivedTrajectory,
    SelectedStep,
)
from adaptive_swe_branching.checkpoints.store import CheckpointStore
from adaptive_swe_branching.data.records import Cost, TrajectoryRecord
from adaptive_swe_branching.data.store import RawStore, stable_sha256
from adaptive_swe_branching.environments.container import (
    DockerContainer,
    PathMap,
)
from adaptive_swe_branching.environments.swesmith import (
    SWESmithTask,
    WorkspaceCache,
)
from adaptive_swe_branching.environments.verifier import SWESmithVerifier
from adaptive_swe_branching.environments.workspace import apply_patch, git_state

AgentFactory = Callable[[int], AgentSession]


class OpenHandsReplayBackend:
    """Execute every paper explore/replay trial against SWE-smith."""

    def __init__(
        self,
        *,
        task: SWESmithTask,
        workspace_cache: WorkspaceCache,
        checkpoint_store: CheckpointStore,
        raw_store: RawStore,
        verifier: SWESmithVerifier,
        agent_factory: AgentFactory,
        run_root: Path,
        max_steps: int,
    ) -> None:
        self.task = task
        self.workspace_cache = workspace_cache
        self.checkpoints = checkpoint_store
        self.raw = raw_store
        self.verifier = verifier
        self.agent_factory = agent_factory
        self.run_root = run_root
        self.max_steps = max_steps

    def explore(self, *, trial_index: int, seed: int) -> ArchivedTrajectory:
        trajectory_id = self._trajectory_id(trial_index, seed, "explore")
        workspace = self.run_root / "workspaces" / trajectory_id
        prepared = self.workspace_cache.prepare(self.task, workspace)
        if prepared.record.base_commit != self.task.record.base_commit:
            raise RuntimeError("workspace cache base commit drifted between trials")
        container = self._container(workspace)
        agent = self.agent_factory(seed)
        container.start()
        agent.start(self.task.record, container)
        try:
            return self._continue(
                trajectory_id=trajectory_id,
                agent=agent,
                container=container,
                seed=seed,
                step_budget=self.max_steps,
                parent_checkpoint_id=None,
                pre_step_ids=[],
                prefix_patches=[],
                restore_cost=Cost(),
            )
        finally:
            agent.close()
            container.stop()

    def exploit(
        self,
        *,
        source: ArchivedTrajectory,
        selected: SelectedStep,
        restore: RestorePlan,
        trial_index: int,
        seed: int,
        suffix_step_budget: int,
    ) -> ArchivedTrajectory:
        trajectory_id = self._trajectory_id(trial_index, seed, "exploit")
        workspace = self.run_root / "workspaces" / trajectory_id
        prepared = self.workspace_cache.prepare(self.task, workspace)
        if prepared.record.base_commit != self.task.record.base_commit:
            raise RuntimeError("workspace cache base commit drifted between trials")
        container = self._container(workspace)
        restore_started = time.monotonic()
        container.start()
        restore_tool_calls = 0
        try:
            if restore.mode == "git_diff":
                apply_patch(workspace, restore.accumulated_patch)
            elif restore.mode == "action_replay":
                restore_tool_calls = self._replay_actions(
                    container, restore.prefix_actions
                )
            else:
                raise ValueError(f"unknown SWE-Replay restore mode: {restore.mode}")
            agent = self.agent_factory(seed)
            self.checkpoints.restore_prepared(
                checkpoint_id=restore.checkpoint_id,
                task=self.task.record,
                workspace=workspace,
                container=container,
                agent=agent,
            )
            restore_cost = Cost(
                tool_calls=restore_tool_calls,
                wall_clock_seconds=time.monotonic() - restore_started,
            )
            try:
                return self._continue(
                    trajectory_id=trajectory_id,
                    agent=agent,
                    container=container,
                    seed=seed,
                    step_budget=suffix_step_budget,
                    parent_checkpoint_id=restore.checkpoint_id,
                    pre_step_ids=list(
                        source.pre_step_checkpoint_ids[: selected.step_index + 1]
                    ),
                    prefix_patches=list(
                        source.accumulated_patches_before_step[
                            : selected.step_index + 1
                        ]
                    ),
                    restore_cost=restore_cost,
                )
            finally:
                agent.close()
        finally:
            container.stop()

    def _continue(
        self,
        *,
        trajectory_id: str,
        agent: AgentSession,
        container: DockerContainer,
        seed: int,
        step_budget: int,
        parent_checkpoint_id: str | None,
        pre_step_ids: list[str],
        prefix_patches: list[str],
        restore_cost: Cost,
    ) -> ArchivedTrajectory:
        new_cost = Cost()
        for _ in range(step_budget):
            if len(pre_step_ids) <= len(agent.steps):
                checkpoint_id = stable_sha256(
                    {
                        "trajectory": trajectory_id,
                        "pre_step": len(agent.steps),
                    }
                )
                _, patch, _, _ = git_state(container.workspace)
                self.checkpoints.create(
                    checkpoint_id=checkpoint_id,
                    task=self.task.record,
                    parent_trajectory_id=trajectory_id,
                    absolute_step=len(agent.steps),
                    cost_to_checkpoint=_sum_step_cost(agent.steps),
                    workspace=container.workspace,
                    agent=agent,
                )
                pre_step_ids.append(checkpoint_id)
                prefix_patches.append(patch)
            result = agent.step()
            new_cost = new_cost + result.cost
            if result.finished:
                break
        _, final_patch, _, _ = git_state(container.workspace)
        verification = self.verifier.verify(self.task, final_patch)
        full_cost = _sum_step_cost(agent.steps) + verification.record.cost
        trajectory = TrajectoryRecord(
            trajectory_id=trajectory_id,
            task_id=self.task.record.task_id,
            seed=seed,
            steps=agent.steps,
            outcome=verification.record.outcome,
            final_patch=final_patch,
            total_cost=full_cost,
            termination_reason=(
                "agent_finished" if agent.finished else "absolute_step_cap"
            ),
            parent_checkpoint_id=parent_checkpoint_id,
            final_answer=agent.final_answer,
            invalid_reason=verification.record.invalid_reason,
            verifier_record_id=verification.record.verifier_record_id,
        )
        self.raw.put("trajectory", trajectory_id, trajectory)
        return ArchivedTrajectory(
            trajectory=trajectory,
            regression_passed=verification.record.regression_passed is True,
            pre_step_checkpoint_ids=tuple(pre_step_ids),
            accumulated_patches_before_step=tuple(prefix_patches),
            generation_cost=(restore_cost + new_cost + verification.record.cost),
        )

    def _replay_actions(
        self, container: DockerContainer, actions: tuple[dict, ...]
    ) -> int:
        from openhands.tools.file_editor.definition import FileEditorAction

        editor = ContainerFileEditorExecutor(
            PathMap(container.workspace, container.container_root)
        )
        calls = 0
        for event in actions:
            name = str(event.get("tool_name", ""))
            payload = event.get("action") or event.get("tool_input") or {}
            if "terminal" in name:
                command = payload.get("command")
                if command:
                    completed = container.exec(str(command))
                    if completed.returncode:
                        raise RuntimeError(
                            f"SWE-Replay action replay failed: {completed.stderr}"
                        )
                    calls += 1
            elif "file_editor" in name:
                observation = editor(FileEditorAction.model_validate(payload))
                if observation.is_error:
                    raise RuntimeError(f"SWE-Replay file replay failed: {observation}")
                calls += 1
        return calls

    def _container(self, workspace: Path) -> DockerContainer:
        return DockerContainer(
            image=self.task.record.image_name,
            workspace=workspace,
            container_root=self.task.record.container_workdir,
            platform=self.task.record.platform,
        )

    def _trajectory_id(self, trial_index: int, seed: int, mode: str) -> str:
        return stable_sha256(
            {
                "task": self.task.record.task_id,
                "swe_replay_trial": trial_index,
                "seed": seed,
                "mode": mode,
            }
        )


def _sum_step_cost(steps) -> Cost:
    return sum((step.cost for step in steps), Cost())
