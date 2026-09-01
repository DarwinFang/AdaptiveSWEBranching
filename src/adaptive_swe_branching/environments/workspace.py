from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path


def run_git(workspace: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *args],
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def normalise_index(workspace: Path) -> None:
    # Intent-to-add makes untracked files visible in both status and diff.
    run_git(workspace, "add", "-A", "--intent-to-add")


def git_state(workspace: Path) -> tuple[str, str, str, tuple[str, ...]]:
    normalise_index(workspace)
    head = run_git(workspace, "rev-parse", "HEAD").strip()
    diff = run_git(workspace, "diff", "--binary", "--no-ext-diff")
    status = run_git(workspace, "status", "--short", "--untracked-files=all")
    modified = tuple(
        sorted(
            line[3:].strip()
            for line in status.splitlines()
            if len(line) >= 4 and line[3:].strip()
        )
    )
    return head, diff, status, modified


def workspace_hash(workspace: Path) -> str:
    digest = hashlib.sha256()
    ignored = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
    for path in sorted(workspace.rglob("*")):
        relative = path.relative_to(workspace)
        if any(part in ignored for part in relative.parts):
            continue
        if path.is_symlink():
            digest.update(b"L\0" + str(relative).encode() + b"\0")
            digest.update(os.readlink(path).encode() + b"\0")
        elif path.is_file():
            digest.update(b"F\0" + str(relative).encode() + b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def assert_safe_symlinks(workspace: Path) -> None:
    root = workspace.resolve()
    for path in workspace.rglob("*"):
        if not path.is_symlink():
            continue
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise ValueError(f"workspace symlink escapes root: {path} -> {resolved}")


def copy_workspace(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    assert_safe_symlinks(source)
    shutil.copytree(source, destination, symlinks=True)


def apply_patch(workspace: Path, patch: str) -> None:
    if not patch.strip():
        return
    completed = subprocess.run(
        ["git", "-C", str(workspace), "apply", "--binary", "--whitespace=nowarn", "-"],
        input=patch,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(f"git apply failed: {completed.stderr.strip()}")
