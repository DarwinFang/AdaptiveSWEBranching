from __future__ import annotations

from collections import Counter

import pytest

from adaptive_swe_branching.screening.runner import (
    SCREEN_EASY,
    SCREEN_HARD,
    SCREEN_MEDIUM,
    difficulty_class,
    quotas_satisfied,
    resolved_difficulty_class,
    selected_cohort,
)


@pytest.mark.parametrize(
    ("successes", "expected"),
    [
        (0, SCREEN_HARD),
        (2, SCREEN_HARD),
        (3, SCREEN_MEDIUM),
        (5, SCREEN_MEDIUM),
        (6, SCREEN_EASY),
        (8, SCREEN_EASY),
    ],
)
def test_frozen_eight_run_screening_boundaries(successes, expected) -> None:
    assert difficulty_class(successes) == expected


@pytest.mark.parametrize(
    ("successes", "observed", "expected", "possible"),
    [
        (0, 6, SCREEN_HARD, (0, 2)),
        (3, 6, SCREEN_MEDIUM, (3, 5)),
        (6, 6, SCREEN_EASY, (6, 8)),
        (2, 7, None, (2, 3)),
        (5, 7, None, (5, 6)),
        (2, 8, SCREEN_HARD, (2, 2)),
        (5, 8, SCREEN_MEDIUM, (5, 5)),
        (8, 8, SCREEN_EASY, (8, 8)),
    ],
)
def test_screening_stops_only_when_final_class_is_mathematically_fixed(
    successes, observed, expected, possible
) -> None:
    assert resolved_difficulty_class(
        successes, observed_valid_runs=observed
    ) == (expected, possible)


def test_all_three_quotas_must_be_reached() -> None:
    quotas = {SCREEN_MEDIUM: 3, SCREEN_EASY: 1, SCREEN_HARD: 1}
    assert not quotas_satisfied(Counter({SCREEN_MEDIUM: 4, SCREEN_EASY: 1}), quotas)
    assert quotas_satisfied(
        Counter({SCREEN_MEDIUM: 4, SCREEN_EASY: 1, SCREEN_HARD: 1}), quotas
    )


def test_selected_cohort_is_first_by_frozen_sampling_order() -> None:
    quotas = {SCREEN_MEDIUM: 2, SCREEN_EASY: 1, SCREEN_HARD: 1}
    summaries = [
        {"task_id": "m-late", "sample_index": 8, "difficulty_class": SCREEN_MEDIUM},
        {"task_id": "easy", "sample_index": 3, "difficulty_class": SCREEN_EASY},
        {"task_id": "m-first", "sample_index": 1, "difficulty_class": SCREEN_MEDIUM},
        {"task_id": "hard", "sample_index": 2, "difficulty_class": SCREEN_HARD},
        {"task_id": "m-second", "sample_index": 4, "difficulty_class": SCREEN_MEDIUM},
    ]
    cohort = selected_cohort(summaries, quotas)
    assert cohort["task_ids_by_class"][SCREEN_MEDIUM] == ["m-first", "m-second"]
