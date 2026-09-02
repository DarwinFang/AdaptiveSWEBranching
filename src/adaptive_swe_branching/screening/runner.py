from __future__ import annotations

import csv
import fcntl
import json
import random
import shutil
import subprocess
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from adaptive_swe_branching.agents.openhands import OpenHandsSession
from adaptive_swe_branching.data.records import (
    Cost,
    Outcome,
    ScreeningPlanRecord,
    ScreeningRunRecord,
    ScreeningTaskRecord,
    TrajectoryRecord,
)
from adaptive_swe_branching.data.store import (
    ExperimentManifest,
    RawStore,
    stable_sha256,
)
from adaptive_swe_branching.environments.container import DockerContainer
from adaptive_swe_branching.environments.swesmith import (
    SWESmithDataset,
    SWESmithTask,
    WorkspaceCache,
)
from adaptive_swe_branching.environments.verifier import SWESmithVerifier
from adaptive_swe_branching.environments.workspace import git_state
from adaptive_swe_branching.seeds import derive_seed

PURPOSE = "difficulty_screen"
SCREEN_HARD = "screen_hard"
SCREEN_MEDIUM = "screen_medium"
SCREEN_EASY = "screen_easy"
SCREEN_INVALID = "screening_invalid"


def difficulty_class(successes: int, *, valid_runs: int = 5) -> str:
    if valid_runs not in {5, 8}:
        raise ValueError("difficulty class supports the frozen 5- or 8-run rules")
    if not 0 <= successes <= valid_runs:
        raise ValueError("successes must be between zero and valid_runs")
    hard_maximum = 1 if valid_runs == 5 else 2
    medium_maximum = 3 if valid_runs == 5 else 5
    if successes <= hard_maximum:
        return SCREEN_HARD
    if successes <= medium_maximum:
        return SCREEN_MEDIUM
    return SCREEN_EASY


def resolved_difficulty_class(
    successes: int,
    *,
    observed_valid_runs: int,
    target_valid_runs: int = 5,
) -> tuple[str | None, tuple[int, int]]:
    """Resolve a class once every remaining outcome gives the same class."""
    if not 0 <= successes <= observed_valid_runs <= target_valid_runs:
        raise ValueError("success/run counts must satisfy 0 <= k <= n <= target")
    remaining = target_valid_runs - observed_valid_runs
    possible = (successes, successes + remaining)
    classes = {
        difficulty_class(candidate, valid_runs=target_valid_runs)
        for candidate in range(possible[0], possible[1] + 1)
    }
    return (classes.pop() if len(classes) == 1 else None), possible


def quotas_satisfied(counts: Counter[str], quotas: dict[str, int]) -> bool:
    return all(counts[name] >= target for name, target in quotas.items())


def selected_cohort(
    summaries: list[dict[str, Any]], quotas: dict[str, int]
) -> dict[str, Any]:
    selected: dict[str, list[str]] = {}
    for name, target in quotas.items():
        matching = sorted(
            (item for item in summaries if item["difficulty_class"] == name),
            key=lambda item: (item["sample_index"], item["task_id"]),
        )
        if len(matching) < target:
            raise ValueError(f"quota is not complete for {name}")
        selected[name] = [item["task_id"] for item in matching[:target]]
    return {
        "purpose": PURPOSE,
        "selection": "first_by_frozen_sampling_order_within_class",
        "quotas": quotas,
        "task_ids_by_class": selected,
        "all_task_ids": [
            task_id for name in quotas for task_id in selected[name]
        ],
    }


def order_screening_pool(
    pool: list[tuple[str, str]],
    *,
    root_seed: int,
    imported: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    """Put compatible legacy tasks first, retaining random order otherwise."""
    random.Random(root_seed).shuffle(pool)
    rank = {task_id: index for index, task_id in enumerate(imported)}
    return sorted(pool, key=lambda item: (rank.get(item[0], len(rank)),))


def safe_parallel_batch_size(
    *,
    successes: int,
    observed_valid_runs: int,
    target_valid_runs: int,
    maximum_workers: int,
) -> int:
    """Parallelize only runs that cannot yet make later launches redundant."""
    if not 0 <= observed_valid_runs < target_valid_runs:
        return 0
    until_first_possible_resolution = target_valid_runs - observed_valid_runs
    for additional in range(1, target_valid_runs - observed_valid_runs + 1):
        if any(
            resolved_difficulty_class(
                successes + added_successes,
                observed_valid_runs=observed_valid_runs + additional,
                target_valid_runs=target_valid_runs,
            )[0]
            is not None
            for added_successes in range(additional + 1)
        ):
            until_first_possible_resolution = additional
            break
    return min(
        maximum_workers,
        target_valid_runs - observed_valid_runs,
        until_first_possible_resolution,
    )


class DifficultyScreeningRunner:
    """Independent task stratification under the configured frozen rule."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        git_commit: str,
        external_resources: dict[str, Any],
    ) -> None:
        self.config = config
        self.git_commit = git_commit
        self.external_resources = external_resources
        self.benchmark = config["benchmark"]
        self.agent_config = config["agent"]
        self.experiment = config["experiment"]
        self.screen = config["screening"]
        self.output = Path(self.experiment["output_dir"]).expanduser().resolve()
        self.raw = RawStore(self.output / "raw")
        self.screen_version = str(self.screen["screen_version"])
        self.root_seed = int(self.experiment["root_seed"])
        self.target_valid = int(self.screen["valid_runs_per_task"])
        self.maximum_attempts = int(self.screen["maximum_total_attempts_per_task"])
        if self.target_valid not in {5, 8}:
            raise ValueError("difficulty screening supports 5 or 8 valid root runs")
        if self.maximum_attempts < self.target_valid:
            raise ValueError("maximum attempts cannot be below valid-run target")
        self.quotas = {
            SCREEN_MEDIUM: int(self.screen["quotas"][SCREEN_MEDIUM]),
            SCREEN_EASY: int(self.screen["quotas"][SCREEN_EASY]),
            SCREEN_HARD: int(self.screen["quotas"][SCREEN_HARD]),
        }
        self.imported = self._load_imported_counts()
        endpoints = self.screen.get("agent_base_urls") or [
            self.agent_config["base_url"]
        ]
        self.agent_base_urls = tuple(str(value) for value in endpoints)
        self.parallel_workers = int(
            self.screen.get("parallel_root_runs", len(self.agent_base_urls))
        )
        if not 1 <= self.parallel_workers <= len(self.agent_base_urls):
            raise ValueError("parallel_root_runs must fit configured agent_base_urls")
        self.dataset = SWESmithDataset(
            snapshot=Path(self.benchmark["dataset_path"]),
            revision=self.benchmark["version"],
            split_manifest=Path(self.benchmark["split_manifest"]),
            container_workdir=self.benchmark["container_workdir"],
            platform=self.benchmark["platform"],
        )
        self.cache = WorkspaceCache(Path(self.benchmark["workspace_cache"]))
        self.verifier = SWESmithVerifier(
            harness_path=Path(self.benchmark["harness_path"]),
            timeout_seconds=float(self.benchmark["verifier_timeout_seconds"]),
            store=self.raw,
        )
        self.current_task: str | None = None
        self.started = time.monotonic()

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "runner.lock").open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("difficulty screening is already running") from error
            self._initialise_manifest()
            plan = self._load_or_create_plan()
            summaries = self._load_records("screening_tasks")
            runs = self._load_records("screening_runs")
            self._write_progress(summaries, runs)
            for item in plan["ordered_tasks"]:
                counts = Counter(
                    summary["difficulty_class"]
                    for summary in summaries
                    if summary["difficulty_class"] != SCREEN_INVALID
                )
                if quotas_satisfied(counts, self.quotas):
                    break
                task_id = item["task_id"]
                if any(summary["task_id"] == task_id for summary in summaries):
                    continue
                self.current_task = task_id
                self._write_progress(summaries, runs)
                summary, new_runs = self._screen_task(
                    task_id=task_id,
                    repository=item["repository"],
                    sample_index=int(item["sample_index"]),
                    prior_runs=[run for run in runs if run["task_id"] == task_id],
                    imported_successes=int(item.get("imported_successes", 0)),
                    imported_valid_runs=int(item.get("imported_valid_runs", 0)),
                    evidence_source=str(item.get("evidence_source", "fresh")),
                )
                summaries.append(summary.to_dict())
                runs.extend(run.to_dict() for run in new_runs)
                self.raw.put("screening_task", task_id, summary)
                self.current_task = None
                self._write_progress(summaries, runs)
                self._write_screening_results(summaries)
                counts = Counter(
                    record["difficulty_class"]
                    for record in summaries
                    if record["difficulty_class"] != SCREEN_INVALID
                )
                print(
                    json.dumps(
                        {
                            "completed_task": task_id,
                            "result": summary.difficulty_class,
                            "successes": summary.n_success,
                            "valid": summary.n_valid,
                            "counts": dict(counts),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            counts = Counter(
                item["difficulty_class"]
                for item in summaries
                if item["difficulty_class"] != SCREEN_INVALID
            )
            if not quotas_satisfied(counts, self.quotas):
                raise RuntimeError(
                    "eligible SWE-smith task pool exhausted before quotas"
                )
            cohort = selected_cohort(summaries, self.quotas)
            cohort["screen_version"] = self.screen_version
            cohort["raw_screened_task_count"] = len(summaries)
            self.raw.put("screening_cohort", self.screen_version, cohort)
            self._write_progress(summaries, runs, complete=True)
            return self._progress_payload(summaries, runs, complete=True)

    def _load_imported_counts(self) -> dict[str, dict[str, Any]]:
        path_value = self.screen.get("initial_counts_path")
        if not path_value:
            return {}
        payload = json.loads(Path(path_value).expanduser().read_text(encoding="utf-8"))
        expected_definition = "ordinary root Agent outcome within at most 20 steps"
        if payload.get("definition") != expected_definition:
            raise ValueError("imported screening counts use a different definition")
        imported = {row["task_id"]: row for row in payload["tasks"]}
        if len(imported) != len(payload["tasks"]):
            raise ValueError("imported screening task IDs are not unique")
        return imported

    def _initialise_manifest(self) -> None:
        manifest_path = self.raw.root / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "experiment_name": self.experiment["name"],
                "git_commit": self.git_commit,
                "root_seed": self.root_seed,
            }
            actual = {key: existing.get(key) for key in expected}
            if actual != expected:
                raise RuntimeError(
                    f"cannot resume screening with changed provenance: {actual}"
                )
            return
        manifest = ExperimentManifest.create(
            experiment_name=self.experiment["name"],
            git_commit=self.git_commit,
            config=self.config,
            root_seed=self.root_seed,
            external_resources=self.external_resources,
        )
        self.raw.initialise(manifest)

    def _load_or_create_plan(self) -> dict[str, Any]:
        try:
            return self.raw.get("screening_plan", self.screen_version)
        except FileNotFoundError:
            pool = list(
                self.dataset.screening_pool(split=str(self.benchmark["split"]))
            )
            pool = order_screening_pool(
                pool, root_seed=self.root_seed, imported=self.imported
            )
            ordered = tuple(
                {
                    "task_id": task_id,
                    "repository": repository,
                    "sample_index": str(index),
                    "imported_successes": str(
                        self.imported.get(task_id, {}).get("successes", 0)
                    ),
                    "imported_valid_runs": str(
                        self.imported.get(task_id, {}).get("valid_runs", 0)
                    ),
                    "evidence_source": (
                        "legacy_source20+fresh" if task_id in self.imported else "fresh"
                    ),
                }
                for index, (task_id, repository) in enumerate(pool)
            )
            plan = ScreeningPlanRecord(
                screen_version=self.screen_version,
                purpose=PURPOSE,
                dataset_revision=str(self.benchmark["version"]),
                split=str(self.benchmark["split"]),
                sampling_seed=self.root_seed,
                sampling_algorithm=(
                    "compatible_import_order_then_python_random_shuffle_of_remaining"
                ),
                eligible_pool_size=len(ordered),
                eligible_pool_sha256=stable_sha256(ordered),
                ordered_tasks=ordered,
            )
            self.raw.put("screening_plan", self.screen_version, plan)
            return plan.to_dict()

    def _screen_task(
        self,
        *,
        task_id: str,
        repository: str,
        sample_index: int,
        prior_runs: list[dict[str, Any]],
        imported_successes: int,
        imported_valid_runs: int,
        evidence_source: str,
    ) -> tuple[ScreeningTaskRecord, list[ScreeningRunRecord]]:
        try:
            task = self.cache.resolve(self.dataset.load(task_id))
            self.raw.put("task", task.record.task_id, task.record)
        except Exception as error:
            reason = _error_text(error)
            return (
                ScreeningTaskRecord(
                    screen_version=self.screen_version,
                    purpose=PURPOSE,
                    task_id=task_id,
                    repository=repository,
                    sample_index=sample_index,
                    n_valid=0,
                    n_success=0,
                    n_failure=0,
                    difficulty_class=SCREEN_INVALID,
                    valid_run_ids=(),
                    all_attempt_run_ids=(),
                    screening_invalid_reason=f"task_initialization_failed: {reason}",
                ),
                [],
            )

        accumulated = list(prior_runs)
        new_records: list[ScreeningRunRecord] = []
        next_attempt = 0
        while next_attempt < self.maximum_attempts:
            valid = [item for item in accumulated if not item["infrastructure_invalid"]]
            successes = imported_successes + sum(
                item["outcome"] == Outcome.SOLVED.value for item in valid
            )
            resolved_class, _ = resolved_difficulty_class(
                successes,
                observed_valid_runs=imported_valid_runs + len(valid),
                target_valid_runs=self.target_valid,
            )
            if resolved_class is not None:
                break
            used_attempts = {int(item["attempt_index"]) for item in accumulated}
            while next_attempt in used_attempts:
                next_attempt += 1
            if next_attempt >= self.maximum_attempts:
                break
            batch_size = safe_parallel_batch_size(
                successes=successes,
                observed_valid_runs=imported_valid_runs + len(valid),
                target_valid_runs=self.target_valid,
                maximum_workers=self.parallel_workers,
            )
            attempts = []
            while len(attempts) < batch_size and next_attempt < self.maximum_attempts:
                if next_attempt not in used_attempts:
                    attempts.append(next_attempt)
                next_attempt += 1
            with ThreadPoolExecutor(max_workers=len(attempts)) as executor:
                futures = {
                    executor.submit(
                        self._run_once,
                        task=task,
                        sample_index=sample_index,
                        attempt_index=attempt_index,
                        agent_base_url=self.agent_base_urls[
                            attempt_index % len(self.agent_base_urls)
                        ],
                    ): attempt_index
                    for attempt_index in attempts
                }
                for future in as_completed(futures):
                    record = future.result()
                    accumulated.append(record.to_dict())
                    new_records.append(record)
                    self._write_progress(
                        self._load_records("screening_tasks"),
                        self._load_records("screening_runs"),
                    )

        valid = [item for item in accumulated if not item["infrastructure_invalid"]]
        all_ids = tuple(item["screen_run_id"] for item in accumulated)
        valid_ids = tuple(item["screen_run_id"] for item in valid)
        successes = imported_successes + sum(
            item["outcome"] == Outcome.SOLVED.value for item in valid
        )
        total_valid = imported_valid_runs + len(valid)
        resolved_class, possible = resolved_difficulty_class(
            successes,
            observed_valid_runs=total_valid,
            target_valid_runs=self.target_valid,
        )
        if resolved_class is None:
            summary = ScreeningTaskRecord(
                screen_version=self.screen_version,
                purpose=PURPOSE,
                task_id=task_id,
                repository=repository,
                sample_index=sample_index,
                n_valid=total_valid,
                n_success=sum(
                    item["outcome"] == Outcome.SOLVED.value for item in valid
                ),
                n_failure=total_valid - successes,
                difficulty_class=SCREEN_INVALID,
                valid_run_ids=valid_ids,
                all_attempt_run_ids=all_ids,
                possible_final_success_min=possible[0],
                possible_final_success_max=possible[1],
                imported_valid_runs=imported_valid_runs,
                imported_successes=imported_successes,
                evidence_source=evidence_source,
                screening_invalid_reason=(
                    f"class unresolved after {len(valid)} valid runs and "
                    f"{len(accumulated)} attempts"
                ),
            )
            return summary, new_records
        summary = ScreeningTaskRecord(
            screen_version=self.screen_version,
            purpose=PURPOSE,
            task_id=task_id,
            repository=repository,
            sample_index=sample_index,
            n_valid=total_valid,
            n_success=successes,
            n_failure=total_valid - successes,
            difficulty_class=resolved_class,
            valid_run_ids=valid_ids,
            all_attempt_run_ids=all_ids,
            early_stopped=total_valid < self.target_valid,
            possible_final_success_min=possible[0],
            possible_final_success_max=possible[1],
            imported_valid_runs=imported_valid_runs,
            imported_successes=imported_successes,
            evidence_source=evidence_source,
        )
        return summary, new_records

    def _run_once(
        self,
        *,
        task: SWESmithTask,
        sample_index: int,
        attempt_index: int,
        agent_base_url: str,
    ) -> ScreeningRunRecord:
        seed = derive_seed(
            self.root_seed,
            self.screen_version,
            PURPOSE,
            task.record.task_id,
            attempt_index,
        )
        run_id = stable_sha256(
            {
                "screen_version": self.screen_version,
                "task_id": task.record.task_id,
                "attempt_index": attempt_index,
                "seed": seed,
            }
        )
        trajectory_id = stable_sha256({"screen_run_id": run_id, "trajectory": True})
        try:
            existing = self.raw.get("trajectory", trajectory_id)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            recovered = self._screen_run_from_trajectory(
                task=task,
                sample_index=sample_index,
                attempt_index=attempt_index,
                seed=seed,
                run_id=run_id,
                trajectory=existing,
            )
            self.raw.put("screening_run", run_id, recovered)
            return recovered

        workspace = self.output / "scratch" / task.record.task_id / run_id
        container: DockerContainer | None = None
        agent: OpenHandsSession | None = None
        started = time.monotonic()
        verification = None
        invalid_reason: str | None = None
        final_patch = ""
        termination_reason = "infrastructure_error"
        try:
            prepared = self.cache.prepare(task, workspace)
            if prepared.record.base_commit != task.record.base_commit:
                raise RuntimeError("workspace base commit drifted during screen run")
            container = DockerContainer(
                image=task.record.image_name,
                workspace=workspace,
                container_root=task.record.container_workdir,
                platform=task.record.platform,
            )
            agent_config = dict(self.agent_config)
            agent_config["base_url"] = agent_base_url
            agent = OpenHandsSession.from_config(agent_config, seed=seed)
            container.start()
            agent.start(task.record, container)
            for _ in range(int(self.screen["safety_cap_steps"])):
                result = agent.step()
                if result.finished:
                    break
            _, final_patch, _, _ = git_state(workspace)
            verification = self.verifier.verify(
                task, final_patch, record_scope=run_id
            )
            invalid_reason = verification.record.invalid_reason
            termination_reason = (
                agent.termination_reason if agent.finished else "absolute_step_cap"
            )
        except Exception as error:
            invalid_reason = _error_text(error)
            if workspace.exists():
                try:
                    _, final_patch, _, _ = git_state(workspace)
                except Exception:
                    final_patch = ""
        finally:
            steps = agent.steps if agent is not None else ()
            final_answer = agent.final_answer if agent is not None else None
            if agent is not None:
                agent.close()
            if container is not None:
                container.stop()

        elapsed = time.monotonic() - started
        step_cost = sum((step.cost for step in steps), Cost())
        verifier_cost = verification.record.cost if verification is not None else Cost()
        measured = step_cost + verifier_cost
        total_cost = replace(measured, wall_clock_seconds=elapsed)
        outcome = (
            verification.record.outcome if verification is not None else Outcome.INVALID
        )
        if outcome == Outcome.INVALID and invalid_reason is None:
            invalid_reason = "verifier returned invalid without a reason"
        trajectory = TrajectoryRecord(
            trajectory_id=trajectory_id,
            task_id=task.record.task_id,
            seed=seed,
            steps=tuple(steps),
            outcome=outcome,
            final_patch=final_patch,
            total_cost=total_cost,
            termination_reason=termination_reason,
            final_answer=final_answer,
            invalid_reason=invalid_reason,
            verifier_record_id=(
                verification.record.verifier_record_id
                if verification is not None
                else None
            ),
            purpose=PURPOSE,
        )
        self.raw.put("trajectory", trajectory_id, trajectory)
        record = ScreeningRunRecord(
            screen_run_id=run_id,
            screen_version=self.screen_version,
            purpose=PURPOSE,
            task_id=task.record.task_id,
            repository=task.record.repository,
            sample_index=sample_index,
            attempt_index=attempt_index,
            seed=seed,
            trajectory_id=trajectory_id,
            outcome=outcome,
            infrastructure_invalid=outcome == Outcome.INVALID,
            invalid_reason=invalid_reason,
            steps=total_cost.steps,
            input_tokens=total_cost.input_tokens,
            output_tokens=total_cost.output_tokens,
            total_tokens=total_cost.total_tokens,
            wall_clock_seconds=elapsed,
        )
        self.raw.put("screening_run", run_id, record)
        shutil.rmtree(workspace, ignore_errors=True)
        return record

    def _screen_run_from_trajectory(
        self,
        *,
        task: SWESmithTask,
        sample_index: int,
        attempt_index: int,
        seed: int,
        run_id: str,
        trajectory: dict[str, Any],
    ) -> ScreeningRunRecord:
        if trajectory.get("purpose") != PURPOSE:
            raise RuntimeError("existing trajectory has the wrong purpose")
        cost = trajectory["total_cost"]
        outcome = Outcome(trajectory["outcome"])
        return ScreeningRunRecord(
            screen_run_id=run_id,
            screen_version=self.screen_version,
            purpose=PURPOSE,
            task_id=task.record.task_id,
            repository=task.record.repository,
            sample_index=sample_index,
            attempt_index=attempt_index,
            seed=seed,
            trajectory_id=trajectory["trajectory_id"],
            outcome=outcome,
            infrastructure_invalid=outcome == Outcome.INVALID,
            invalid_reason=trajectory.get("invalid_reason"),
            steps=int(cost["steps"]),
            input_tokens=int(cost["input_tokens"]),
            output_tokens=int(cost["output_tokens"]),
            total_tokens=int(cost["input_tokens"]) + int(cost["output_tokens"]),
            wall_clock_seconds=float(cost["wall_clock_seconds"]),
        )

    def _load_records(self, folder: str) -> list[dict[str, Any]]:
        root = self.raw.root / folder
        if not root.exists():
            return []
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in root.glob("*.json")
        ]

    def _write_progress(
        self,
        summaries: list[dict[str, Any]],
        runs: list[dict[str, Any]],
        *,
        complete: bool = False,
    ) -> None:
        # Include records completed since the caller's last in-memory refresh.
        on_disk_runs = self._load_records("screening_runs")
        if len(on_disk_runs) > len(runs):
            runs = on_disk_runs
        payload = self._progress_payload(summaries, runs, complete=complete)
        _write_json(self.output / "progress.json", payload)
        _write_json(
            self.output / "repository_distribution.json",
            payload["repository_distribution"],
        )

    def _write_screening_results(self, summaries: list[dict[str, Any]]) -> None:
        path = self.output / "screening_results.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["task_id", "k", "m", "class", "source"],
            )
            writer.writeheader()
            for item in sorted(summaries, key=lambda row: row["sample_index"]):
                writer.writerow(
                    {
                        "task_id": item["task_id"],
                        "k": item["n_success"],
                        "m": item["n_valid"],
                        "class": item["difficulty_class"],
                        "source": item.get("evidence_source", "fresh"),
                    }
                )
        temporary.replace(path)

    def _progress_payload(
        self,
        summaries: list[dict[str, Any]],
        runs: list[dict[str, Any]],
        *,
        complete: bool,
    ) -> dict[str, Any]:
        counts = Counter(
            item["difficulty_class"]
            for item in summaries
            if item["difficulty_class"] != SCREEN_INVALID
        )
        distribution: dict[str, Counter[str]] = defaultdict(Counter)
        for item in summaries:
            distribution[item["repository"]][item["difficulty_class"]] += 1
        valid_runs = [item for item in runs if not item["infrastructure_invalid"]]
        return {
            "screen_version": self.screen_version,
            "purpose": PURPOSE,
            "complete": complete,
            "total_tasks_screened": len(summaries),
            "currently_running_tasks": (
                [self.current_task] if self.current_task is not None else []
            ),
            "counts": {
                name: {"current": counts[name], "target": target}
                for name, target in self.quotas.items()
            },
            "infrastructure_invalid_tasks": sum(
                item["difficulty_class"] == SCREEN_INVALID for item in summaries
            ),
            "infrastructure_invalid_runs": sum(
                item["infrastructure_invalid"] for item in runs
            ),
            "distinct_repositories": len(distribution),
            "total_valid_root_trajectories": len(valid_runs),
            "approximate_token_usage": sum(item["total_tokens"] for item in runs),
            "accumulated_run_wall_clock_seconds": sum(
                item["wall_clock_seconds"] for item in runs
            ),
            "current_process_wall_clock_seconds": time.monotonic() - self.started,
            "gpu_status": _gpu_status(),
            "repository_distribution": {
                repository: dict(class_counts)
                for repository, class_counts in sorted(distribution.items())
            },
        }


def _error_text(error: Exception) -> str:
    trace = "".join(traceback.format_exception_only(type(error), error)).strip()
    return trace[-4000:]


def _gpu_status() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in output.splitlines():
        index, used, total, utilization = (part.strip() for part in line.split(","))
        rows.append(
            {
                "index": int(index),
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
                "utilization_percent": int(utilization),
            }
        )
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
