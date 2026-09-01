from __future__ import annotations

import math
from dataclasses import dataclass

from adaptive_swe_branching.data.records import Outcome
from adaptive_swe_branching.oracle.records import FutureSample


@dataclass(frozen=True)
class ChildQEstimate:
    child_checkpoint_id: str
    valid_k: int
    successes: int
    empirical_q: float
    invalid_trajectory_ids: tuple[str, ...]
    excluded_cap_hit_trajectory_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChildQAuditResult:
    parent_checkpoint_id: str
    parent_branchability: float
    children: tuple[ChildQEstimate, ...]
    max_minus_mean_q: float
    max_minus_min_q: float
    population_standard_deviation: float
    child_q_oracle_checkpoint_ids: tuple[str, ...]
    trajectory_outcome_oracle_candidate_ids: tuple[str, ...]
    candidate_sets_overlap: bool
    trajectory_outcome_best_case_q_regret: float
    trajectory_outcome_worst_case_q_regret: float
    trajectory_outcome_uniform_tie_expected_q_regret: float


def estimate_child_q(
    child_checkpoint_id: str,
    samples: tuple[FutureSample, ...],
    *,
    minimum_valid_k: int = 6,
    exclude_cap_hits: bool = False,
) -> ChildQEstimate:
    valid = tuple(
        sample
        for sample in samples
        if sample.outcome != Outcome.INVALID
        and not (exclude_cap_hits and sample.cap_hit)
    )
    if len(valid) < minimum_valid_k:
        raise ValueError(
            f"child {child_checkpoint_id} needs at least {minimum_valid_k} "
            f"valid continuations; found {len(valid)}"
        )
    successes = sum(sample.outcome == Outcome.SOLVED for sample in valid)
    return ChildQEstimate(
        child_checkpoint_id=child_checkpoint_id,
        valid_k=len(valid),
        successes=successes,
        empirical_q=successes / len(valid),
        invalid_trajectory_ids=tuple(
            sample.trajectory_id
            for sample in samples
            if sample.outcome == Outcome.INVALID
        ),
        excluded_cap_hit_trajectory_ids=tuple(
            sample.trajectory_id
            for sample in samples
            if exclude_cap_hits
            and sample.cap_hit
            and sample.outcome != Outcome.INVALID
        ),
    )


def analyse_child_q_group(
    *,
    parent_checkpoint_id: str,
    parent_branchability: float,
    samples_by_child: dict[str, tuple[FutureSample, ...]],
    original_outcome_by_child: dict[str, Outcome],
    minimum_valid_k: int = 6,
    exclude_cap_hits: bool = False,
) -> ChildQAuditResult:
    if set(samples_by_child) != set(original_outcome_by_child):
        raise ValueError("nested samples and original outcomes need identical children")
    if len(samples_by_child) < 2:
        raise ValueError("child-q spread needs at least two children")
    children = tuple(
        estimate_child_q(
            child_id,
            samples_by_child[child_id],
            minimum_valid_k=minimum_valid_k,
            exclude_cap_hits=exclude_cap_hits,
        )
        for child_id in sorted(samples_by_child)
    )
    q_by_child = {
        child.child_checkpoint_id: child.empirical_q for child in children
    }
    values = tuple(q_by_child.values())
    maximum = max(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    child_q_choices = tuple(
        sorted(child_id for child_id, q in q_by_child.items() if q == maximum)
    )
    realized_successes = sorted(
        child_id
        for child_id, outcome in original_outcome_by_child.items()
        if outcome == Outcome.SOLVED
    )
    trajectory_candidates = tuple(
        realized_successes or sorted(original_outcome_by_child)
    )
    trajectory_candidate_q = tuple(
        q_by_child[child_id] for child_id in trajectory_candidates
    )
    expected_trajectory_q = sum(trajectory_candidate_q) / len(
        trajectory_candidate_q
    )
    return ChildQAuditResult(
        parent_checkpoint_id=parent_checkpoint_id,
        parent_branchability=parent_branchability,
        children=children,
        max_minus_mean_q=maximum - mean,
        max_minus_min_q=maximum - min(values),
        population_standard_deviation=math.sqrt(variance),
        child_q_oracle_checkpoint_ids=child_q_choices,
        trajectory_outcome_oracle_candidate_ids=trajectory_candidates,
        candidate_sets_overlap=bool(
            set(child_q_choices).intersection(trajectory_candidates)
        ),
        trajectory_outcome_best_case_q_regret=(
            maximum - max(trajectory_candidate_q)
        ),
        trajectory_outcome_worst_case_q_regret=(
            maximum - min(trajectory_candidate_q)
        ),
        trajectory_outcome_uniform_tie_expected_q_regret=(
            maximum - expected_trajectory_q
        ),
    )
