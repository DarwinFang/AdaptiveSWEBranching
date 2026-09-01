from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from adaptive_swe_branching.agents.registry import (
    CONTAINERS,
    FILE_EDITOR_TOOL,
    TERMINAL_TOOL,
)
from adaptive_swe_branching.environments.container import PathMap

_REGISTERED = False
_LOCK = threading.Lock()


def register_tools() -> None:
    global _REGISTERED
    with _LOCK:
        if _REGISTERED:
            return
        from openhands.sdk.tool import register_tool

        register_tool(FILE_EDITOR_TOOL, ContainerFileEditorTool)
        register_tool(TERMINAL_TOOL, ContainerTerminalTool)
        _REGISTERED = True


def tool_names(configured: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    aliases = {
        "terminal": TERMINAL_TOOL,
        "file_editor": FILE_EDITOR_TOOL,
        TERMINAL_TOOL: TERMINAL_TOOL,
        FILE_EDITOR_TOOL: FILE_EDITOR_TOOL,
    }
    try:
        return tuple(dict.fromkeys(aliases[name] for name in configured))
    except KeyError as error:
        raise ValueError(
            f"tool has no container-routed implementation: {error.args[0]}"
        ) from None


# These imports remain at module scope because OpenHands state is pickled by
# qualified class name. The package itself imports this module lazily.
from openhands.sdk.tool import ToolAnnotations, ToolDefinition  # noqa: E402
from openhands.tools.file_editor.definition import (  # noqa: E402
    TOOL_DESCRIPTION,
    FileEditorAction,
    FileEditorObservation,
)
from openhands.tools.file_editor.impl import FileEditorExecutor  # noqa: E402
from openhands.tools.terminal.definition import TerminalTool  # noqa: E402


class ContainerFileEditorExecutor(FileEditorExecutor):
    def __init__(self, paths: PathMap):
        super().__init__(workspace_root=str(paths.host_root))
        self.paths = paths

    def __call__(self, action: FileEditorAction, conversation=None):
        try:
            host_path = self.paths.to_host(action.path)
        except ValueError as error:
            return FileEditorObservation.from_text(
                text=f"Invalid path {action.path}: {error}",
                command=action.command,
                is_error=True,
            )
        observation = super().__call__(
            action.model_copy(update={"path": str(host_path)}), conversation
        )
        update: dict[str, Any] = {}
        if observation.path:
            update["path"] = observation.path.replace(
                str(self.paths.host_root), self.paths.container_root
            )
        if getattr(observation, "content", None):
            update["content"] = [
                item.model_copy(
                    update={
                        "text": item.text.replace(
                            str(self.paths.host_root), self.paths.container_root
                        )
                    }
                )
                if getattr(item, "text", None)
                else item
                for item in observation.content
            ]
        return observation.model_copy(update=update) if update else observation


class ContainerFileEditorTool(ToolDefinition):
    name = FILE_EDITOR_TOOL

    @classmethod
    def create(cls, conv_state) -> Sequence[ContainerFileEditorTool]:
        container = CONTAINERS.resolve(conv_state.workspace.working_dir)
        paths = PathMap(container.workspace, container.container_root)
        return [
            cls(
                name=FILE_EDITOR_TOOL,
                action_type=FileEditorAction,
                observation_type=FileEditorObservation,
                description=(
                    f"{TOOL_DESCRIPTION}\n\nUse absolute paths under "
                    f"{paths.container_root}; the terminal uses the same namespace."
                ),
                annotations=ToolAnnotations(
                    title="file_editor",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=ContainerFileEditorExecutor(paths),
            )
        ]


class ContainerTerminalTool(TerminalTool):
    name = TERMINAL_TOOL

    @classmethod
    def create(cls, conv_state) -> Sequence[TerminalTool]:
        container = CONTAINERS.resolve(conv_state.workspace.working_dir)
        return super().create(
            conv_state,
            terminal_type="subprocess",
            shell_path=str(container.shell_shim()),
        )
