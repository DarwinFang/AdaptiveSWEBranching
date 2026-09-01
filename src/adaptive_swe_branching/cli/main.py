from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from adaptive_swe_branching.checkpoints.store import CheckpointStore
from adaptive_swe_branching.config import load_config
from adaptive_swe_branching.data.records import Cost
from adaptive_swe_branching.data.store import (
    ExperimentManifest,
    RawStore,
    stable_sha256,
)
from adaptive_swe_branching.environments.container import DockerContainer
from adaptive_swe_branching.environments.swesmith import SWESmithDataset, WorkspaceCache
from adaptive_swe_branching.seeds import derive_seed


def main() -> None:
    parser = argparse.ArgumentParser(prog="asb")
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor = subcommands.add_parser("doctor", help="verify frozen external resources")
    doctor.add_argument("--config", required=True)
    smoke = subcommands.add_parser(
        "smoke-checkpoint", help="one bounded live OpenHands checkpoint/restore test"
    )
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--steps", type=int, default=1)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.command == "doctor":
        report = doctor_report(config)
    else:
        report = checkpoint_smoke(config, steps=arguments.steps)
    print(json.dumps(report, indent=2, sort_keys=True))


def doctor_report(config: dict[str, Any]) -> dict[str, Any]:
    benchmark = config["benchmark"]
    agent = config["agent"]
    paths = {
        name: Path(benchmark[name]).expanduser().resolve()
        for name in (
            "dataset_path",
            "dataset_manifest",
            "split_manifest",
            "harness_path",
            "workspace_cache",
        )
    }
    path_checks = {name: path.exists() for name, path in paths.items()}
    if not all(path_checks.values()):
        raise FileNotFoundError(
            {name: str(paths[name]) for name, ok in path_checks.items() if not ok}
        )
    harness_commit = subprocess.run(
        ["git", "-C", str(paths["harness_path"]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    harness_dirty = bool(
        subprocess.run(
            ["git", "-C", str(paths["harness_path"]), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if harness_commit != benchmark["harness_commit"] or harness_dirty:
        message = (
            f"harness is not frozen as configured: commit={harness_commit}, "
            f"dirty={harness_dirty}"
        )
        raise RuntimeError(message)
    tags_url = agent["base_url"].rstrip("/") + "/api/tags"
    with urllib.request.urlopen(tags_url, timeout=10) as response:
        tags = json.load(response)
    model_name = agent["model"].removeprefix("ollama/")
    model = next(
        (item for item in tags.get("models", []) if item.get("name") == model_name),
        None,
    )
    if model is None:
        raise RuntimeError(f"Ollama model is absent: {model_name}")
    if model.get("digest") != agent["model_digest"]:
        message = (
            f"Ollama model digest drifted: {model.get('digest')} != "
            f"{agent['model_digest']}"
        )
        raise RuntimeError(message)
    return {
        "ok": True,
        "paths": {name: str(path) for name, path in paths.items()},
        "harness_commit": harness_commit,
        "harness_clean": True,
        "openhands_sdk": importlib.metadata.version("openhands-sdk"),
        "swesmith": _swesmith_version(paths["harness_path"]),
        "ollama_model": model_name,
        "ollama_digest": model["digest"],
        "docker": subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }


def checkpoint_smoke(config: dict[str, Any], *, steps: int) -> dict[str, Any]:
    doctor = doctor_report(config)
    benchmark = config["benchmark"]
    agent_config = config["agent"]
    experiment = config["experiment"]
    task_id = config["smoke"]["task_id"]
    root_seed = int(experiment["root_seed"])
    seed = derive_seed(root_seed, task_id, "source")
    dataset = SWESmithDataset(
        snapshot=Path(benchmark["dataset_path"]),
        revision=benchmark["version"],
        split_manifest=Path(benchmark["split_manifest"]),
        container_workdir=benchmark["container_workdir"],
        platform=benchmark["platform"],
    )
    task = dataset.load(task_id)
    output = Path(experiment["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"smoke output is immutable and already exists: {output}")
    raw = RawStore(output / "raw")
    manifest = ExperimentManifest.create(
        experiment_name=experiment["name"],
        git_commit=_git_commit(Path.cwd()),
        config=config,
        root_seed=root_seed,
        external_resources=doctor,
    )
    raw.initialise(manifest)
    source_workspace = output / "workspaces" / "source"
    task = WorkspaceCache(Path(benchmark["workspace_cache"])).prepare(
        task, source_workspace
    )
    raw.put("task", task.record.task_id, task.record)
    container = DockerContainer(
        image=task.record.image_name,
        workspace=source_workspace,
        container_root=task.record.container_workdir,
        platform=task.record.platform,
    )
    source_agent = _agent(agent_config, seed)
    container.start()
    source_agent.start(task.record, container)
    total = Cost()
    try:
        for _ in range(steps):
            result = source_agent.step()
            total = total + result.cost
            if result.finished:
                break
        checkpoint_id = stable_sha256(
            {"task": task_id, "trajectory": "smoke-source", "step": total.steps}
        )
        checkpoints = CheckpointStore(output / "checkpoints", raw)
        record = checkpoints.create(
            checkpoint_id=checkpoint_id,
            task=task.record,
            parent_trajectory_id="smoke-source",
            absolute_step=total.steps,
            cost_to_checkpoint=total,
            workspace=source_workspace,
            agent=source_agent,
        )
    finally:
        source_agent.close()
        container.stop()
    restored_agent = _agent(agent_config, derive_seed(root_seed, task_id, "restored"))
    restored_workspace = output / "workspaces" / "restored"
    restored_record, restored_container = checkpoints.restore(
        checkpoint_id=checkpoint_id,
        task=task.record,
        destination=restored_workspace,
        agent=restored_agent,
    )
    try:
        continued = None
        if not restored_agent.finished:
            continued = restored_agent.step()
    finally:
        restored_agent.close()
        restored_container.stop()
    return {
        "ok": True,
        "task_id": task_id,
        "checkpoint_id": record.checkpoint_id,
        "workspace_hash": restored_record.workspace_hash,
        "restored": True,
        "continued_one_step": continued is not None,
        "output_dir": str(output),
    }


def _agent(config: dict[str, Any], seed: int):
    from adaptive_swe_branching.agents.openhands import OpenHandsSession

    return OpenHandsSession(
        model=config["model"],
        base_url=config["base_url"],
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        max_output_tokens=int(config["max_output_tokens"]),
        timeout_seconds=float(config["timeout_seconds"]),
        retries=int(config["retries"]),
        native_tool_calling=bool(config["native_tool_calling"]),
        tools=tuple(config["tools"]),
        max_iterations_per_step=int(config["max_iterations_per_step"]),
        seed=seed,
    )


def _git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _swesmith_version(harness_path: Path) -> str:
    source = (harness_path / "swesmith" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)', source, re.MULTILINE)
    if not match:
        raise RuntimeError("cannot resolve SWE-smith harness package version")
    return match.group(1)


if __name__ == "__main__":
    main()
