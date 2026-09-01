#!/usr/bin/env python3
"""Export the old 20-step root outcomes as a compact screening seed file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", required=True)
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
        "tasks": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
