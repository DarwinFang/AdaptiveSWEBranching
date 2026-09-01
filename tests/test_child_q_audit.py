from __future__ import annotations

from adaptive_swe_branching.data.records import Cost, Outcome
from adaptive_swe_branching.oracle.child_q_audit import analyse_child_q_group
from adaptive_swe_branching.oracle.records import FutureSample


def samples(prefix: str, successes: int) -> tuple[FutureSample, ...]:
    return tuple(
        FutureSample(
            trajectory_id=f"{prefix}-{index}",
            seed=index,
            outcome=Outcome.SOLVED if index < successes else Outcome.UNSOLVED,
            cost_from_state=Cost(steps=1),
            final_patch="",
            termination_reason="finish",
        )
        for index in range(8)
    )


def test_child_q_audit_exposes_outcome_oracle_regret() -> None:
    result = analyse_child_q_group(
        parent_checkpoint_id="parent",
        parent_branchability=1.0,
        samples_by_child={
            "low": samples("low", 2),
            "high": samples("high", 6),
            "middle-a": samples("middle-a", 4),
            "middle-b": samples("middle-b", 4),
        },
        original_outcome_by_child={
            "low": Outcome.SOLVED,
            "high": Outcome.UNSOLVED,
            "middle-a": Outcome.UNSOLVED,
            "middle-b": Outcome.UNSOLVED,
        },
    )
    assert result.max_minus_mean_q == 0.25
    assert result.max_minus_min_q == 0.5
    assert result.child_q_oracle_checkpoint_ids == ("high",)
    assert result.trajectory_outcome_oracle_candidate_ids == ("low",)
    assert result.candidate_sets_overlap is False
    assert result.trajectory_outcome_best_case_q_regret == 0.5
    assert result.trajectory_outcome_worst_case_q_regret == 0.5
    assert result.trajectory_outcome_uniform_tie_expected_q_regret == 0.5
