from __future__ import annotations

from adaptive_swe_branching.data.records import (
    Cost,
    Outcome,
    StepRecord,
    TrajectoryRecord,
)


def make_step(
    index: int, *, files: tuple[str, ...] = (), reasoning: str = "think"
) -> StepRecord:
    return StepRecord(
        absolute_step=index + 1,
        reasoning=reasoning,
        tool_name="terminal",
        tool_input={"command": "true"},
        action_text="true",
        observation_text="",
        is_error=False,
        explored_files_before=files,
        cost=Cost(steps=1, model_calls=1, tool_calls=1),
    )


def make_trajectory(
    identity: str,
    *,
    solved: bool = False,
    steps: tuple[StepRecord, ...] | None = None,
    patch: str | None = None,
) -> TrajectoryRecord:
    steps = steps or (make_step(0),)
    return TrajectoryRecord(
        trajectory_id=identity,
        task_id="task",
        seed=1,
        steps=steps,
        outcome=Outcome.SOLVED if solved else Outcome.UNSOLVED,
        final_patch=patch if patch is not None else f"patch-{identity}\n",
        total_cost=Cost(steps=len(steps), model_calls=len(steps)),
        termination_reason="finish",
    )
