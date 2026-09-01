from __future__ import annotations

import threading
from pathlib import Path

from adaptive_swe_branching.environments.container import DockerContainer

TERMINAL_TOOL = "asb_terminal"
FILE_EDITOR_TOOL = "asb_file_editor"


class ContainerRegistry:
    def __init__(self) -> None:
        self._items: dict[str, DockerContainer] = {}
        self._lock = threading.Lock()

    def register(self, container: DockerContainer) -> None:
        with self._lock:
            self._items[str(container.workspace.resolve())] = container

    def unregister(self, container: DockerContainer) -> None:
        with self._lock:
            self._items.pop(str(container.workspace.resolve()), None)

    def resolve(self, workspace: str | Path) -> DockerContainer:
        key = str(Path(workspace).resolve())
        with self._lock:
            if key not in self._items:
                raise KeyError(f"no container registered for {key}")
            return self._items[key]


CONTAINERS = ContainerRegistry()
