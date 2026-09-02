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
    replay = subcommands.add_parser(
        "smoke-swe-replay", help="two-trial bounded faithful SWE-Replay smoke"
    )
    replay.add_argument("--config", required=True)
    screening = subcommands.add_parser(
        "screen-difficulty", help="run or resume independent root task screening"
    )
    screening.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.command == "doctor":
        report = doctor_report(config)
    elif arguments.command == "smoke-checkpoint":
        report = checkpoint_smoke(config, steps=arguments.steps)
    elif arguments.command == "smoke-swe-replay":
        report = swe_replay_smoke(config)
    else:
        report = difficulty_screen(config)
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


def swe_replay_smoke(config: dict[str, Any]) -> dict[str, Any]:
    from dataclasses import asdict

    from adaptive_swe_branching.baselines.swe_replay.openhands_backend import (
        OpenHandsReplayBackend,
    )
    from adaptive_swe_branching.baselines.swe_replay.runner import SWEReplayRunner
    from adaptive_swe_branching.environments.verifier import SWESmithVerifier

    doctor = doctor_report(config)
    benchmark = config["benchmark"]
    experiment = config["experiment"]
    smoke = config["smoke"]
    output = Path(experiment["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"smoke output is immutable and already exists: {output}")
    root_seed = int(experiment["root_seed"])
    dataset = SWESmithDataset(
        snapshot=Path(benchmark["dataset_path"]),
        revision=benchmark["version"],
        split_manifest=Path(benchmark["split_manifest"]),
        container_workdir=benchmark["container_workdir"],
        platform=benchmark["platform"],
    )
    cache = WorkspaceCache(Path(benchmark["workspace_cache"]))
    task = cache.resolve(dataset.load(smoke["task_id"]))
    raw = RawStore(output / "raw")
    raw.initialise(
        ExperimentManifest.create(
            experiment_name=experiment["name"],
            git_commit=_git_commit(Path.cwd()),
            config=config,
            root_seed=root_seed,
            external_resources=doctor,
        )
    )
    raw.put("task", task.record.task_id, task.record)
    checkpoints = CheckpointStore(output / "checkpoints", raw)
    verifier = SWESmithVerifier(
        harness_path=Path(benchmark["harness_path"]),
        timeout_seconds=float(benchmark["verifier_timeout_seconds"]),
        store=raw,
    )
    backend = OpenHandsReplayBackend(
        task=task,
        workspace_cache=cache,
        checkpoint_store=checkpoints,
        raw_store=raw,
        verifier=verifier,
        agent_factory=lambda seed: _agent(config["agent"], seed),
        run_root=output,
        max_steps=int(smoke["max_steps"]),
    )
    result = SWEReplayRunner(
        trials=int(smoke["trials"]),
        explore_probability=float(config["swe_replay"]["explore_probability"]),
        max_steps=int(smoke["max_steps"]),
        root_seed=root_seed,
        backend=backend,
    ).run()
    run_id = stable_sha256(
        {"task": task.record.task_id, "experiment": experiment["name"]}
    )
    raw.put(
        "swe_replay_run",
        run_id,
        {
            "run_id": run_id,
            "task_id": task.record.task_id,
            "trials": [asdict(trial) for trial in result.trials],
            "archive_trajectory_ids": [
                item.trajectory.trajectory_id for item in result.archive
            ],
            "selected_trajectory_id": result.selected_trajectory_id,
            "selected_patch_sha256": stable_sha256(result.selected_patch),
            "majority_tied": result.majority_tied,
            "generation_costs": [
                item.generation_cost.to_dict() for item in result.archive
            ],
        },
    )
    return {
        "ok": True,
        "task_id": task.record.task_id,
        "trials": [trial.mode for trial in result.trials],
        "selected_trajectory_id": result.selected_trajectory_id,
        "majority_tied": result.majority_tied,
        "output_dir": str(output),
    }


def difficulty_screen(config: dict[str, Any]) -> dict[str, Any]:
    from adaptive_swe_branching.screening.runner import DifficultyScreeningRunner

    doctor = doctor_report(config)
    return DifficultyScreeningRunner(
        config=config,
        git_commit=_git_commit(Path.cwd()),
        external_resources=doctor,
    ).run()


def _agent(config: dict[str, Any], seed: int):
    from adaptive_swe_branching.agents.openhands import OpenHandsSession

    return OpenHandsSession.from_config(config, seed=seed)


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
