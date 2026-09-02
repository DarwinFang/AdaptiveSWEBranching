from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathMap:
    host_root: Path
    container_root: str = "/testbed"

    def to_host(self, path: str) -> Path:
        root = self.host_root.resolve()
        if path == self.container_root:
            return root
        prefix = self.container_root.rstrip("/") + "/"
        if not path.startswith(prefix):
            raise ValueError(f"path is outside container workspace: {path}")
        mapped = (root / path.removeprefix(prefix)).resolve(strict=False)
        if not mapped.is_relative_to(root):
            raise ValueError(f"path escapes workspace: {path}")
        return mapped

    def to_container(self, path: Path) -> str:
        relative = path.resolve(strict=False).relative_to(self.host_root.resolve())
        return f"{self.container_root.rstrip('/')}/{relative.as_posix()}"


def image_digest(image: str) -> str:
    raw = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    digests = json.loads(raw)
    if digests:
        return digests[0].split("@", 1)[1]
    image_id = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return image_id


class DockerContainer:
    """One disposable container backed by one host workspace."""

    def __init__(
        self,
        *,
        image: str,
        workspace: Path,
        container_root: str = "/testbed",
        platform: str = "linux/amd64",
    ) -> None:
        self.image = image
        self.workspace = workspace.resolve()
        self.container_root = container_root
        self.platform = platform
        self.name = f"asb-{uuid.uuid4().hex[:12]}"
        self._shim_path = self.workspace.parent / f".{self.name}-shell"
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                self.name,
                "--platform",
                self.platform,
                "--user",
                "root",
                "-v",
                f"{self.workspace}:{self.container_root}",
                "-w",
                self.container_root,
                self.image,
                "sleep",
                "infinity",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.started = True
        self.exec(
            "git config --global --add safe.directory "
            + shlex.quote(self.container_root)
        )

    def exec(
        self, command: str, *, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        if not self.started:
            raise RuntimeError("container is not running")
        return subprocess.run(
            ["docker", "exec", self.name, "bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def shell_command(self, command: str) -> str:
        return " ".join(
            shlex.quote(value)
            for value in ["docker", "exec", self.name, "bash", "-lc", command]
        )

    def shell_shim(self) -> Path:
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("docker executable not found")
        argv = (
            docker,
            "exec",
            "-i",
            "-w",
            self.container_root,
            "-u",
            "root",
            self.name,
            "/bin/bash",
        )
        self._shim_path.write_text(
            "#!/bin/sh\nexec "
            + " ".join(shlex.quote(value) for value in argv)
            + ' "$@"\n',
            encoding="utf-8",
        )
        self._shim_path.chmod(0o700)
        return self._shim_path

    def stop(self) -> None:
        if not self.started:
            return
        subprocess.run(
            [
                "docker",
                "exec",
                self.name,
                "chown",
                "-R",
                f"{os.getuid()}:{os.getgid()}",
                self.container_root,
            ],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["docker", "stop", "--time", "5", self.name],
            capture_output=True,
            text=True,
        )
        self.started = False
        self._shim_path.unlink(missing_ok=True)

    def __enter__(self) -> DockerContainer:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
