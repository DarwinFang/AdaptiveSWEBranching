#!/usr/bin/env python3
"""Export the old 20-step root outcomes as a compact screening seed file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def resolved_k5(successes: int, valid_runs: int) -> bool:
    remaining = 5 - valid_runs
    classes = set()
    for final_successes in range(successes, successes + remaining + 1):
        classes.add(
            "hard"
            if final_successes <= 1
            else "medium"
            if final_successes <= 3
            else "easy"
        )
    return len(classes) == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--screening-run-root")
    parser.add_argument("--target-valid-runs", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    roots = [Path(value).expanduser().resolve() for value in args.run_root]
    rows = []
    for root in roots:
        for path in sorted((root / "tasks").glob("*.json")):
            task = json.loads(path.read_text(encoding="utf-8"))
            sources = task.get("sources", [])
            successes = sum(bool(source.get("solved")) for source in sources)
            valid_runs = len(sources)
            if valid_runs == 3:
                old_class = (
                    "screen_hard"
                    if successes == 0
                    else "screen_easy"
                    if successes == 3
                    else "screen_medium"
                )
            else:
                old_class = "incomplete"
            rows.append(
                {
                    "task_id": task["task_id"],
                    "successes": successes,
                    "valid_runs": valid_runs,
                    "old_class": old_class,
                    "source": root.name,
                }
            )

    if len({row["task_id"] for row in rows}) != len(rows):
        raise RuntimeError("legacy run roots contain duplicate task IDs")
    if args.screening_run_root:
        screening_root = Path(args.screening_run_root).expanduser().resolve()
        fresh_by_task: dict[str, list[dict]] = {}
        for path in (screening_root / "raw" / "screening_runs").glob("*.json"):
            run = json.loads(path.read_text(encoding="utf-8"))
            if not run["infrastructure_invalid"]:
                fresh_by_task.setdefault(run["task_id"], []).append(run)
        for row in rows:
            fresh = sorted(
                fresh_by_task.get(row["task_id"], []),
                key=lambda run: run["attempt_index"],
            )
            for run in fresh:
                if row["valid_runs"] >= args.target_valid_runs or (
                    args.target_valid_runs == 5
                    and resolved_k5(row["successes"], row["valid_runs"])
                ):
                    break
                row["valid_runs"] += 1
                row["successes"] += run["outcome"] == "solved"
            if fresh:
                row["source"] += f"+{screening_root.name}"

    class_order = {
        "screen_medium": 0,
        "screen_easy": 1,
        "screen_hard": 2,
        "incomplete": 3,
    }
    rows.sort(key=lambda row: (class_order[row["old_class"]], row["task_id"]))
    payload = {
        "schema_version": 1,
        "definition": "ordinary root Agent outcome within at most 20 steps",
        "purpose": "screening_seed_only_not_training_or_evaluation",
        "source_run_roots": [str(root) for root in roots],
        "source_manifest_sha256": {
            root.name: hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
            for root in roots
        },
        "supplemental_screening_run_root": args.screening_run_root,
        "target_valid_runs": args.target_valid_runs,
        "tasks": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
