from __future__ import annotations

import asyncio
import copy
import inspect
import pickle
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from adaptive_swe_branching.agents.base import AgentSnapshot, StepResult
from adaptive_swe_branching.agents.registry import CONTAINERS
from adaptive_swe_branching.data.records import Cost, StepRecord, TaskRecord
from adaptive_swe_branching.data.store import stable_sha256
from adaptive_swe_branching.environments.container import DockerContainer

STATE_FILE = "openhands_state.pkl"
STATE_FORMAT = "openhands-sdk-pickle-v1"
WORKSPACE_TOKEN = "<ASB_WORKSPACE>"


class OpenHandsSession:
    """One OpenHands conversation whose public step is one closed-loop turn."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        timeout_seconds: float,
        retries: int,
        native_tool_calling: bool,
        tools: tuple[str, ...],
        max_iterations_per_step: int,
        seed: int,
    ) -> None:
        from adaptive_swe_branching.agents.openhands_tools import (
            register_tools,
            tool_names,
        )

        register_tools()
        self.model = model
        self.seed = seed
        self.tools = tool_names(tools)
        self.max_iterations_per_step = max_iterations_per_step
        self.llm_settings = {
            "model": model,
            "base_url": base_url,
            "api_key": "local",
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "timeout": timeout_seconds,
            "num_retries": retries,
            "native_tool_calling": native_tool_calling,
        }
        self.task: TaskRecord | None = None
        self.container: DockerContainer | None = None
        self._agent: Any = None
        self._conversation: Any = None
        self._steps: list[StepRecord] = []
        self._cost = Cost()
        self._finished = False
        self._final_answer: str | None = None
        self._explored_files: set[str] = set()

    @property
    def fingerprint(self) -> str:
        return stable_sha256(
            {
                "scaffold": "openhands",
                "model": self.model,
                "settings": {
                    k: v for k, v in self.llm_settings.items() if k != "api_key"
                },
                "tools": self.tools,
                "step_semantics": "one model response plus tool observation",
            }
        )

    @property
    def finished(self) -> bool:
        return self._finished

    def _build(self) -> None:
        from openhands.sdk import LLM, Agent, LocalConversation, Tool

        llm = LLM(
            **self.llm_settings,
            seed=self.seed,
            usage_id="adaptive_swe_branching",
            stream=False,
        )
        self._agent = Agent(llm=llm, tools=[Tool(name=name) for name in self.tools])
        self._conversation = LocalConversation(
            agent=self._agent,
            workspace=str(self.container.workspace),
            callbacks=[self._pause_after_observation],
            max_iteration_per_run=self.max_iterations_per_step,
            visualizer=None,
        )

    def start(self, task: TaskRecord, container: DockerContainer) -> None:
        if not container.started:
            raise RuntimeError("container must be started first")
        self.close()
        self.task = task
        self.container = container
        self._steps = []
        self._cost = Cost()
        self._finished = False
        self._final_answer = None
        self._explored_files = set()
        CONTAINERS.register(container)
        try:
            self._build()
            self._conversation.send_message(
                f"## Task\n{task.issue}\n\n"
                f"The repository is at `{container.container_root}`. Terminal and file "
                "editor use that same path namespace. Find the root cause, fix it, run "
                "relevant tests, and finish with a short summary."
            )
        except Exception:
            self.close()
            raise

    def attach(self, task: TaskRecord, container: DockerContainer) -> None:
        if not container.started:
            raise RuntimeError("container must be started first")
        self.close()
        self.task = task
        self.container = container
        CONTAINERS.register(container)

    def _pause_after_observation(self, event: Any) -> None:
        if self._conversation is None:
            return
        payload = _dump(event)
        if str(payload.get("tool_name", "")).casefold() in {"finish", "finishtool"}:
            return
        assistant_without_tool = (
            str(payload.get("source", "")).casefold() == "agent"
            and (payload.get("llm_message") or {}).get("role") == "assistant"
            and not (payload.get("llm_message") or {}).get("tool_calls")
        )
        if payload.get("kind") == "ObservationEvent" or assistant_without_tool:
            self._conversation.pause()

    def step(self) -> StepResult:
        if self._conversation is None:
            raise RuntimeError("session has not started or restored")
        if self._finished:
            return StepResult(None, Cost(), True, self._final_answer)
        start = len(self._conversation.state.events)
        usage_before = self._usage()
        result = self._conversation.run()
        if inspect.isawaitable(result):
            asyncio.run(_await(result))
        events = list(self._conversation.state.events)
        raw = [_dump(event) for event in events[start:]]
        action, observation = _paired_action_observation(raw)
        answer = _assistant_text(raw)
        status = getattr(self._conversation.state, "execution_status", "")
        status_name = getattr(status, "name", str(status)).casefold()
        if answer or status_name == "finished":
            self._finished = True
            self._final_answer = answer or self._final_answer
        elif status_name in {"error", "stopped", "stuck"}:
            raise RuntimeError(f"OpenHands ended with status {status_name}")
        usage_after = self._usage()
        cost = Cost(
            steps=1,
            input_tokens=max(usage_after[0] - usage_before[0], 0),
            output_tokens=max(usage_after[1] - usage_before[1], 0),
            model_calls=max(usage_after[2] - usage_before[2], 1),
            tool_calls=1 if observation else 0,
        )
        reasoning = _action_reasoning(action)
        action_text = _action_text(action)
        observation_text = _observation_text(observation)
        before = tuple(sorted(self._explored_files))
        self._explored_files.update(
            _repository_paths(
                f"{action_text}\n{observation_text}", self.container.container_root
            )
        )
        step = StepRecord(
            absolute_step=len(self._steps) + 1,
            reasoning=reasoning,
            tool_name=str((action or {}).get("tool_name", "")),
            tool_input=_tool_input(action),
            action_text=action_text,
            observation_text=observation_text,
            is_error=bool((observation or {}).get("observation", {}).get("is_error")),
            explored_files_before=before,
            cost=cost,
            raw_action=action or {},
            raw_observation=observation or {},
        )
        self._steps.append(step)
        self._cost = self._cost + cost
        return StepResult(step, cost, self._finished, self._final_answer)

    def _usage(self) -> tuple[int, int, int]:
        metrics = getattr(getattr(self._agent, "llm", None), "metrics", None)
        usage = getattr(metrics, "accumulated_token_usage", None) if metrics else None
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
            len(getattr(metrics, "token_usages", ()) or ()) if metrics else 0,
        )

    def history_payload(self) -> tuple[dict[str, Any], ...]:
        if self._conversation is None:
            return ()
        workspace = str(self.container.workspace)
        return tuple(
            _replace_text(_dump(event), workspace, WORKSPACE_TOKEN)
            for event in self._conversation.state.events
        )

    def snapshot(self, destination: Path) -> AgentSnapshot:
        if self._conversation is None or self.task is None:
            raise RuntimeError("cannot snapshot an uninitialised session")
        destination.mkdir(parents=True, exist_ok=True)
        workspace = str(self.container.workspace)
        events = [
            event.model_copy(deep=True)
            if hasattr(event, "model_copy")
            else copy.deepcopy(event)
            for event in self._conversation.state.events
        ]
        payload = _replace_text(
            {
                "steps": self._steps,
                "cost": self._cost,
                "finished": self._finished,
                "final_answer": self._final_answer,
                "explored_files": self._explored_files,
                "events": events,
                "agent_state": copy.deepcopy(self._conversation._state.agent_state),
                "activated_skills": list(
                    self._conversation._state.activated_knowledge_skills
                ),
            },
            workspace,
            WORKSPACE_TOKEN,
        )
        with (destination / STATE_FILE).open("wb") as handle:
            pickle.dump(payload, handle)
        history_hash = stable_sha256(self.history_payload())
        return AgentSnapshot(
            format=STATE_FORMAT,
            state_file=STATE_FILE,
            completed_steps=len(self._steps),
            history_hash=history_hash,
            model_input_hash=stable_sha256(
                {"task": self.task.to_dict(), "history": self.history_payload()}
            ),
            unsupported_runtime_state=(
                "background processes",
                "open sockets",
                "shell process state",
            ),
        )

    def restore(self, state_dir: Path, snapshot: AgentSnapshot) -> None:
        if self.container is None or self.task is None:
            raise RuntimeError("attach task and container before restore")
        if snapshot.format != STATE_FORMAT:
            raise ValueError(f"unsupported snapshot format: {snapshot.format}")
        with (state_dir / snapshot.state_file).open("rb") as handle:
            payload = pickle.load(handle)
        payload = _replace_text(payload, WORKSPACE_TOKEN, str(self.container.workspace))
        self._build()
        state = self._conversation._state
        for event in payload["events"]:
            state.events.append(event)
        if hasattr(state, "rebuild_view"):
            state.rebuild_view()
        state.agent_state = payload["agent_state"]
        state.activated_knowledge_skills = list(payload.get("activated_skills", []))
        idle = getattr(type(state.execution_status), "IDLE", None)
        if idle is not None:
            state.execution_status = idle
        self._steps = list(payload["steps"])
        self._cost = payload["cost"]
        self._finished = bool(payload["finished"])
        self._final_answer = payload.get("final_answer")
        self._explored_files = set(payload.get("explored_files", []))
        if stable_sha256(self.history_payload()) != snapshot.history_hash:
            raise RuntimeError("restored OpenHands history does not match checkpoint")

    def close(self) -> None:
        if self._conversation is not None:
            try:
                self._conversation.close()
            except Exception:
                pass
        self._conversation = None
        self._agent = None
        if self.container is not None:
            CONTAINERS.unregister(self.container)


async def _await(value: Any) -> Any:
    return await value


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return (
        asdict(value)
        if hasattr(value, "__dataclass_fields__")
        else {"text": str(value)}
    )


def _replace_text(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, dict):
        return {key: _replace_text(child, old, new) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_text(child, old, new) for child in value]
    if isinstance(value, tuple):
        return tuple(_replace_text(child, old, new) for child in value)
    if isinstance(value, set):
        return {_replace_text(child, old, new) for child in value}
    return value


def _paired_action_observation(
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    actions = {
        str(event.get("id")): event
        for event in events
        if event.get("kind") == "ActionEvent"
    }
    for event in events:
        if event.get("kind") == "ObservationEvent":
            return actions.get(str(event.get("action_id"))), event
    action = next(iter(actions.values()), None)
    return action, None


def _assistant_text(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        message = event.get("llm_message") or {}
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _action_reasoning(action: dict[str, Any] | None) -> str:
    if not action:
        return ""
    return str(action.get("thought") or action.get("reasoning") or "")


def _tool_input(action: dict[str, Any] | None) -> dict[str, Any]:
    if not action:
        return {}
    payload = action.get("action") or action.get("tool_input") or {}
    return payload if isinstance(payload, dict) else {"value": payload}


def _action_text(action: dict[str, Any] | None) -> str:
    return str(_tool_input(action)) if action else ""


def _observation_text(observation: dict[str, Any] | None) -> str:
    if not observation:
        return ""
    payload = observation.get("observation") or observation
    pieces: list[str] = []
    if isinstance(payload, dict):
        for key in ("content", "text", "output", "error"):
            value = payload.get(key)
            if isinstance(value, str):
                pieces.append(value)
            elif isinstance(value, list):
                pieces.extend(
                    str(item.get("text", item)) if isinstance(item, dict) else str(item)
                    for item in value
                )
    return "\n".join(pieces)[:16000]


def _repository_paths(text: str, root: str) -> set[str]:
    prefix = re.escape(root.rstrip("/"))
    matches = re.findall(rf"{prefix}/[A-Za-z0-9_./+@-]+", text)
    return {match[len(root.rstrip("/")) + 1 :].rstrip(".,:;)") for match in matches}
