#!/usr/bin/env python3
"""Reuse frozen rollout data for a zero-GPU Oracle viability pilot.

This analysis is deliberately separate from the Phase-4 data generator.  It
joins a point-in-time screening snapshot to old, already-paid-for step-2/6
checkpoint continuations.  The result is exploratory because the root screen
partly reuses the source trajectories that created those checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


AXES = ("steps", "tokens", "wall_clock_seconds")
METRICS = (
    "mean_checkpoint_success",
    "max_checkpoint_success",
    "true_value_spread",
    "apparent_oracle_selection_gain_n1",
    "crossfit_oracle_selection_gain_n1",
    "mixed_checkpoint_fraction",
    "random_best_of_4",
    "checkpoint_oracle_best_of_4",
    "oracle_headroom_best_of_4",
)


@dataclass(frozen=True)
class Cost:
    steps: float
    tokens: float
    wall_clock_seconds: float

    def value(self, axis: str) -> float:
        return float(getattr(self, axis))

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            self.steps + other.steps,
            self.tokens + other.tokens,
            self.wall_clock_seconds + other.wall_clock_seconds,
        )


@dataclass(frozen=True)
class Attempt:
    solved: bool
    cost: Cost


@dataclass(frozen=True)
class Child:
    checkpoint_id: str
    prefix_cost: Cost
    attempts: tuple[Attempt, ...]

    @property
    def q(self) -> float:
        return sum(item.solved for item in self.attempts) / len(self.attempts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening-csv", type=Path, required=True)
    parser.add_argument("--competence-json", type=Path, required=True)
    parser.add_argument("--run-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crossfit-repetitions", type=int, default=200)
    parser.add_argument("--restart-simulations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(root_seed: int, *parts: object) -> int:
    text = "|".join((str(root_seed), *(str(part) for part in parts)))
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def read_screen(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            k, m = int(row["k"]), int(row["m"])
            rows[row["task_id"]] = {
                "k": k,
                "m": m,
                "bucket": f"{k}/{m}",
                "q": k / m,
                "uncertainty": 4.0 * (k / m) * (1.0 - k / m),
                "screen_class": row["class"],
                "screen_source": row["source"],
            }
    return rows


def joined_group_rows(
    competence_path: Path, screen: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(competence_path.read_text(encoding="utf-8"))
    joined: list[dict[str, Any]] = []
    for original in payload["group_rows"]:
        screened = screen.get(original["task_id"])
        if screened is None or int(original["step"]) not in {2, 6}:
            continue
        row = dict(original)
        row.update({f"screen_{key}": value for key, value in screened.items()})
        joined.append(row)
    return joined, payload


def load_children(run_roots: list[Path]) -> dict[tuple[str, int], list[Child]]:
    observations: dict[str, list[Attempt]] = defaultdict(list)
    identity_seen: set[tuple[str, str]] = set()
    root_by_checkpoint: dict[str, Path] = {}
    for root in run_roots:
        with (root / "observations.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if int(row["step"]) not in {2, 6}:
                    continue
                identity = (row["checkpoint_id"], row["continuation_id"])
                if identity in identity_seen:
                    raise ValueError(f"duplicate observation: {identity}")
                identity_seen.add(identity)
                if row["outcome"] == "invalid":
                    continue
                total = row["total_cost"]
                observations[row["checkpoint_id"]].append(
                    Attempt(
                        solved=row["outcome"] == "solved",
                        cost=Cost(
                            steps=float(row["steps_used"]),
                            tokens=float(
                                total["input_tokens"] + total["output_tokens"]
                            ),
                            wall_clock_seconds=float(total["wall_clock_seconds"]),
                        ),
                    )
                )
                root_by_checkpoint[row["checkpoint_id"]] = root

    grouped: dict[tuple[str, int], list[Child]] = defaultdict(list)
    for checkpoint_id, attempts in observations.items():
        root = root_by_checkpoint[checkpoint_id]
        manifest_path = root / "checkpoints" / checkpoint_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prefix = manifest["metadata"]["cost_to_checkpoint"]
        child = Child(
            checkpoint_id=checkpoint_id,
            prefix_cost=Cost(
                steps=float(manifest["step"]),
                tokens=float(prefix["input_tokens"] + prefix["output_tokens"]),
                wall_clock_seconds=float(prefix["wall_clock_seconds"]),
            ),
            attempts=tuple(attempts),
        )
        grouped[(manifest["task_id"], int(manifest["step"]))].append(child)
    return grouped


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.fmean(items) if items else math.nan


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        tied_rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = tied_rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True)
    )
    left_scale = sum((item - left_mean) ** 2 for item in left)
    right_scale = sum((item - right_mean) ** 2 for item in right)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else None


def bucket_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["step"]), row["screen_bucket"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (step, bucket), members in sorted(
        groups.items(), key=lambda item: (item[0][0], _bucket_key(item[0][1]))
    ):
        summary: dict[str, Any] = {
            "step": step,
            "screen_bucket": bucket,
            "tasks": len(members),
            "parent_q": mean(float(row["screen_q"]) for row in members),
            "parent_uncertainty": mean(
                float(row["screen_uncertainty"]) for row in members
            ),
            "mean_siblings": mean(float(row["siblings"]) for row in members),
        }
        for metric in METRICS:
            values = [
                float(row[metric])
                for row in members
                if row.get(metric) is not None
            ]
            summary[metric] = mean(values)
        summaries.append(summary)
    return summaries


def uncertainty_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for step in (2, 6):
        for metric in (
            "true_value_spread",
            "apparent_oracle_selection_gain_n1",
            "crossfit_oracle_selection_gain_n1",
        ):
            members = [
                row
                for row in rows
                if int(row["step"]) == step and row.get(metric) is not None
            ]
            uncertainty = [float(row["screen_uncertainty"]) for row in members]
            values = [float(row[metric]) for row in members]
            results.append(
                {
                    "step": step,
                    "metric": metric,
                    "tasks": len(members),
                    "pearson": pearson(uncertainty, values),
                    "spearman": pearson(rank(uncertainty), rank(values)),
                }
            )
    return results


def matched_compute(
    rows: list[dict[str, Any]],
    children_by_group: dict[tuple[str, int], list[Child]],
    *,
    crossfit_repetitions: int,
    restart_simulations: int,
    root_seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        key = (row["task_id"], int(row["step"]))
        children = sorted(
            children_by_group.get(key, []), key=lambda child: child.checkpoint_id
        )
        if len(children) < 2 or any(len(child.attempts) < 4 for child in children):
            continue
        for axis in AXES:
            attempt_costs = [
                child.prefix_cost.value(axis) + attempt.cost.value(axis)
                for child in children
                for attempt in child.attempts
            ]
            one_attempt = mean(attempt_costs)
            for multiplier in (1.0, 2.0, 3.0, 4.0):
                budget = multiplier * one_attempt
                probabilities = {
                    "single_chain": _single_chain(children, axis, budget),
                    "equal_cost_restarts": _equal_cost_restarts(
                        children,
                        axis,
                        budget,
                        simulations=restart_simulations,
                        seed=stable_seed(root_seed, *key, axis, multiplier, "restart"),
                    ),
                    "random_branch": _random_branch(children, axis, budget),
                    "plugin_q_oracle_branch": _plugin_oracle_branch(
                        children, axis, budget
                    ),
                    "crossfit_q_oracle_branch": _crossfit_oracle_branch(
                        children,
                        axis,
                        budget,
                        repetitions=crossfit_repetitions,
                        seed=stable_seed(root_seed, *key, axis, multiplier, "crossfit"),
                    ),
                }
                for strategy, probability in probabilities.items():
                    results.append(
                        {
                            "task_id": row["task_id"],
                            "step": int(row["step"]),
                            "screen_bucket": row["screen_bucket"],
                            "axis": axis,
                            "budget_in_mean_attempts": multiplier,
                            "absolute_budget": budget,
                            "siblings": len(children),
                            "strategy": strategy,
                            "solve_probability": probability,
                        }
                    )
    return results


def _single_chain(children: list[Child], axis: str, budget: float) -> float:
    values = [
        float(
            attempt.solved
            and child.prefix_cost.value(axis) + attempt.cost.value(axis) <= budget
        )
        for child in children
        for attempt in child.attempts
    ]
    return mean(values)


def _random_branch(children: list[Child], axis: str, budget: float) -> float:
    prefix = sum(child.prefix_cost.value(axis) for child in children)
    values = [
        float(attempt.solved and prefix + attempt.cost.value(axis) <= budget)
        for child in children
        for attempt in child.attempts
    ]
    return mean(values)


def _plugin_oracle_branch(children: list[Child], axis: str, budget: float) -> float:
    best_q = max(child.q for child in children)
    chosen = [child for child in children if child.q == best_q]
    prefix = sum(child.prefix_cost.value(axis) for child in children)
    return mean(
        float(attempt.solved and prefix + attempt.cost.value(axis) <= budget)
        for child in chosen
        for attempt in child.attempts
    )


def _crossfit_oracle_branch(
    children: list[Child],
    axis: str,
    budget: float,
    *,
    repetitions: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    prefix = sum(child.prefix_cost.value(axis) for child in children)
    scores: list[float] = []
    for _ in range(repetitions):
        folds: dict[str, tuple[list[Attempt], list[Attempt]]] = {}
        for child in children:
            shuffled = list(child.attempts)
            rng.shuffle(shuffled)
            cut = len(shuffled) // 2
            folds[child.checkpoint_id] = (shuffled[:cut], shuffled[cut:])
        for select_index, evaluate_index in ((0, 1), (1, 0)):
            q_by_child = {
                child.checkpoint_id: mean(
                    float(item.solved)
                    for item in folds[child.checkpoint_id][select_index]
                )
                for child in children
            }
            best_q = max(q_by_child.values())
            chosen = rng.choice(
                [
                    child
                    for child in children
                    if q_by_child[child.checkpoint_id] == best_q
                ]
            )
            evaluation = folds[chosen.checkpoint_id][evaluate_index]
            scores.append(
                mean(
                    float(item.solved and prefix + item.cost.value(axis) <= budget)
                    for item in evaluation
                )
            )
    return mean(scores)


def _equal_cost_restarts(
    children: list[Child],
    axis: str,
    budget: float,
    *,
    simulations: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    solved = 0
    for _ in range(simulations):
        remaining = budget
        while remaining > 0:
            child = rng.choice(children)
            attempt = rng.choice(child.attempts)
            cost = child.prefix_cost.value(axis) + attempt.cost.value(axis)
            if cost > remaining:
                break
            remaining -= cost
            if attempt.solved:
                solved += 1
                break
    return solved / simulations


def aggregate_matched(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["step"],
            row["screen_bucket"],
            row["axis"],
            row["budget_in_mean_attempts"],
            row["strategy"],
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(
        groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        step, bucket, axis, multiplier, strategy = key
        output.append(
            {
                "step": step,
                "screen_bucket": bucket,
                "axis": axis,
                "budget_in_mean_attempts": multiplier,
                "strategy": strategy,
                "tasks": len(members),
                "solve_probability": mean(
                    float(row["solve_probability"]) for row in members
                ),
            }
        )
    return output


def _bucket_key(bucket: str) -> tuple[float, int, int]:
    k, m = (int(part) for part in bucket.split("/"))
    return k / m, m, k


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(
    bucket_rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    *,
    joined_tasks: int,
) -> str:
    lines = [
        "# Reused-data Oracle viability pilot",
        "",
        f"This zero-GPU pilot joins {joined_tasks} screened tasks to old step-2/6 ",
        "checkpoint continuations. It does not stop or modify the live screen.",
        "",
        "## Interpretation limits",
        "",
        "- Parent is the task initial state; children are source trajectories at "
        "absolute step 2 or 6. This is not an arbitrary intermediate-parent audit.",
        "- The current screen partly reuses the source trajectories that created the "
        "children. Exact k/m buckets are therefore not fully independent.",
        "- Parent q is measured at a 20-step screen; child q uses the old 60-step "
        "continuation protocol. The relation is exploratory, not a calibrated "
        "identity.",
        "- Plug-in Oracle chooses the largest empirical child q using the same finite "
        "pool. Cross-fit Oracle selects and evaluates on disjoint halves and is safer.",
        "",
        "## Per-bucket value results",
        "",
        "|step|k/m|tasks|mean child q|max child q|q spread|plug-in gain|"
        "cross-fit gain|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bucket_rows:
        lines.append(
            f"|{row['step']}|{row['screen_bucket']}|{row['tasks']}|"
            f"{row['mean_checkpoint_success']:.3f}|"
            f"{row['max_checkpoint_success']:.3f}|"
            f"{row['true_value_spread']:.3f}|"
            f"{row['apparent_oracle_selection_gain_n1']:.3f}|"
            f"{row['crossfit_oracle_selection_gain_n1']:.3f}|"
        )
    lines.extend(
        [
            "",
            "## Parent uncertainty relation",
            "",
            "|step|target|tasks|Pearson|Spearman|",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in correlations:
        p = "NA" if row["pearson"] is None else f"{row['pearson']:.3f}"
        s = "NA" if row["spearman"] is None else f"{row['spearman']:.3f}"
        lines.append(f"|{row['step']}|{row['metric']}|{row['tasks']}|{p}|{s}|")
    lines.extend(
        [
            "",
            "## Matched-compute headline (agent steps, 1x mean attempt budget)",
            "",
            "The branch strategies pay for every sibling prefix. Equal-cost restart "
            "uses the same budget for independent whole attempts.",
            "",
            "|step|k/m|tasks|single|restart|random branch|plug-in Oracle|"
            "cross-fit Oracle|",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    indexed = {
        (row["step"], row["screen_bucket"], row["strategy"]): row
        for row in matched
        if row["axis"] == "steps" and row["budget_in_mean_attempts"] == 1.0
    }
    bucket_keys = sorted(
        {(row["step"], row["screen_bucket"]) for row in matched},
        key=lambda item: (item[0], _bucket_key(item[1])),
    )
    strategies = (
        "single_chain",
        "equal_cost_restarts",
        "random_branch",
        "plugin_q_oracle_branch",
        "crossfit_q_oracle_branch",
    )
    for step, bucket in bucket_keys:
        values = [indexed.get((step, bucket, strategy)) for strategy in strategies]
        if any(value is None for value in values):
            continue
        lines.append(
            f"|{step}|{bucket}|{values[0]['tasks']}|"
            + "|".join(f"{value['solve_probability']:.3f}" for value in values)
            + "|"
        )
    lines.extend(
        [
            "",
            "Full matched-compute curves for 1x-4x budgets and steps/tokens/wall time "
            "are in `matched_compute_summary.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    snapshot = args.output_dir / "screening_results.snapshot.csv"
    shutil.copyfile(args.screening_csv, snapshot)
    screen = read_screen(snapshot)
    joined, competence = joined_group_rows(args.competence_json, screen)
    children = load_children(args.run_root)
    buckets = bucket_summary(joined)
    correlations = uncertainty_correlations(joined)
    matched_rows = matched_compute(
        joined,
        children,
        crossfit_repetitions=args.crossfit_repetitions,
        restart_simulations=args.restart_simulations,
        root_seed=args.seed,
    )
    matched_summary = aggregate_matched(matched_rows)

    write_csv(args.output_dir / "joined_group_rows.csv", joined)
    write_csv(args.output_dir / "bucket_summary.csv", buckets)
    write_csv(args.output_dir / "uncertainty_correlations.csv", correlations)
    write_csv(args.output_dir / "matched_compute_task_rows.csv", matched_rows)
    write_csv(args.output_dir / "matched_compute_summary.csv", matched_summary)
    report = report_text(
        buckets,
        correlations,
        matched_summary,
        joined_tasks=len({row["task_id"] for row in joined}),
    )
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        "analysis": "reused_root_parent_oracle_viability_pilot",
        "analysis_code_commit": subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[1]),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "created_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "crossfit_repetitions": args.crossfit_repetitions,
        "restart_simulations": args.restart_simulations,
        "screening_snapshot": str(snapshot),
        "inputs": {
            "screening_csv": {
                "path": str(args.screening_csv),
                "snapshot_sha256": file_sha256(snapshot),
            },
            "competence_json": {
                "path": str(args.competence_json),
                "sha256": file_sha256(args.competence_json),
            },
            "run_roots": [
                {
                    "path": str(path),
                    "observations_sha256": file_sha256(path / "observations.jsonl"),
                    "manifest_sha256": file_sha256(path / "manifest.json"),
                }
                for path in args.run_root
            ],
            "old_analysis_design": competence["design"],
        },
        "counts": {
            "screened_tasks_in_snapshot": len(screen),
            "joined_tasks": len({row["task_id"] for row in joined}),
            "joined_task_step_groups": len(joined),
            "matched_compute_task_step_groups": len(
                {(row["task_id"], row["step"]) for row in matched_rows}
            ),
        },
        "limitations": [
            "root-parent reuse pilot, not arbitrary intermediate-parent branching",
            "screen k/m partly reuses the source trajectories that form siblings",
            "parent q uses a 20-step protocol while child q uses 60-step continuations",
            "plugin oracle reuses the finite pool; cross-fit result is less biased",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
