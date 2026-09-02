from __future__ import annotations

import shlex
from pathlib import Path

from adaptive_swe_branching.environments.container import DockerContainer


def test_shell_shim_keeps_openhands_stdin_attached(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    container = DockerContainer(image="test-image", workspace=workspace)
    monkeypatch.setattr(
        "adaptive_swe_branching.environments.container.shutil.which",
        lambda command: "/usr/bin/docker" if command == "docker" else None,
    )

    shim = container.shell_shim()
    command = shlex.split(shim.read_text(encoding="utf-8").splitlines()[1])

    assert command == [
        "exec",
        "/usr/bin/docker",
        "exec",
        "-i",
        "-w",
        "/testbed",
        "-u",
        "root",
        container.name,
        "/bin/bash",
        "$@",
    ]
    assert shim.stat().st_mode & 0o700 == 0o700
    shim.unlink()


def test_shell_shim_requires_docker(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    container = DockerContainer(image="test-image", workspace=workspace)
    monkeypatch.setattr(
        "adaptive_swe_branching.environments.container.shutil.which",
        lambda _command: None,
    )

    try:
        container.shell_shim()
    except RuntimeError as error:
        assert str(error) == "docker executable not found"
    else:
        raise AssertionError("missing docker executable should be rejected")
