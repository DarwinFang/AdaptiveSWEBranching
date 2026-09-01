from __future__ import annotations

import random
from dataclasses import dataclass, replace

from adaptive_swe_branching.data.records import CheckpointRecord
from adaptive_swe_branching.seeds import derive_seed

STRATA = ("early", "middle", "late")


@dataclass(frozen=True)
class ParentCandidate:
    checkpoint: CheckpointRecord
    candidate_sources: tuple[str, ...]


def stratified_random_parents(
    eligible_nonterminal: tuple[CheckpointRecord, ...],
    *,
    root_seed: int,
    task_id: str,
) -> tuple[ParentCandidate, ...]:
    """Sample one parent from each relative-progress third.

    The caller removes initial and terminal states. Checkpoints are ordered by
    their trajectory position (absolute step), then split into three contiguous
    chunks whose sizes differ by at most one.
    """
    ordered = _ordered_unique(eligible_nonterminal)
    thirds = _split_thirds(ordered)
    return tuple(
        ParentCandidate(
            checkpoint=random.Random(
                derive_seed(root_seed, "phase4_parent", task_id, stratum)
            ).choice(chunk),
            candidate_sources=(f"random_{stratum}",),
        )
        for stratum, chunk in zip(STRATA, thirds, strict=True)
    )


def merge_swe_replay_candidate(
    eligible_nonterminal: tuple[CheckpointRecord, ...],
    random_candidates: tuple[ParentCandidate, ...],
    *,
    swe_replay_checkpoint_id: str,
    root_seed: int,
    task_id: str,
) -> tuple[ParentCandidate, ...]:
    """Keep four unique parents and preserve overlap as dual provenance."""
    ordered = _ordered_unique(eligible_nonterminal)
    if len(ordered) < 4:
        raise ValueError("four unique parents require at least four checkpoints")
    if len(random_candidates) != 3:
        raise ValueError("expected exactly three stratified random candidates")
    by_id = {item.checkpoint.checkpoint_id: item for item in random_candidates}
    eligible_by_id = {item.checkpoint_id: item for item in ordered}
    if swe_replay_checkpoint_id not in eligible_by_id:
        raise ValueError("SWE-Replay candidate is not an eligible nonterminal state")

    if swe_replay_checkpoint_id not in by_id:
        return random_candidates + (
            ParentCandidate(
                eligible_by_id[swe_replay_checkpoint_id], ("swe_replay",)
            ),
        )

    overlap = by_id[swe_replay_checkpoint_id]
    by_id[swe_replay_checkpoint_id] = replace(
        overlap,
        candidate_sources=overlap.candidate_sources + ("swe_replay",),
    )
    used = set(by_id)
    stratum = overlap.candidate_sources[0].removeprefix("random_")
    thirds = dict(zip(STRATA, _split_thirds(ordered), strict=True))
    same_stratum = [
        checkpoint
        for checkpoint in thirds[stratum]
        if checkpoint.checkpoint_id not in used
    ]
    pool = same_stratum or [
        checkpoint for checkpoint in ordered if checkpoint.checkpoint_id not in used
    ]
    if not pool:
        raise ValueError("no unused checkpoint is available to replace overlap")
    replacement = random.Random(
        derive_seed(root_seed, "phase4_parent_replacement", task_id, stratum)
    ).choice(pool)
    replacement_source = (
        f"random_{stratum}" if same_stratum else "random_replacement"
    )
    return tuple(by_id.values()) + (
        ParentCandidate(replacement, (replacement_source,)),
    )


def select_phase4_parents(
    eligible_nonterminal: tuple[CheckpointRecord, ...],
    *,
    swe_replay_checkpoint_id: str,
    root_seed: int,
    task_id: str,
) -> tuple[ParentCandidate, ...]:
    random_candidates = stratified_random_parents(
        eligible_nonterminal, root_seed=root_seed, task_id=task_id
    )
    return merge_swe_replay_candidate(
        eligible_nonterminal,
        random_candidates,
        swe_replay_checkpoint_id=swe_replay_checkpoint_id,
        root_seed=root_seed,
        task_id=task_id,
    )


def _ordered_unique(
    checkpoints: tuple[CheckpointRecord, ...],
) -> tuple[CheckpointRecord, ...]:
    ordered = tuple(sorted(checkpoints, key=lambda item: item.absolute_step))
    if any(item.absolute_step <= 0 for item in ordered):
        raise ValueError("initial state must be removed before parent selection")
    ids = [item.checkpoint_id for item in ordered]
    steps = [item.absolute_step for item in ordered]
    if len(ids) != len(set(ids)) or len(steps) != len(set(steps)):
        raise ValueError("eligible checkpoints must have unique IDs and steps")
    return ordered


def _split_thirds(
    checkpoints: tuple[CheckpointRecord, ...],
) -> tuple[tuple[CheckpointRecord, ...], ...]:
    if len(checkpoints) < 3:
        raise ValueError("relative-progress thirds need at least three checkpoints")
    quotient, remainder = divmod(len(checkpoints), 3)
    sizes = tuple(
        quotient + (1 if index < remainder else 0) for index in range(3)
    )
    chunks: list[tuple[CheckpointRecord, ...]] = []
    start = 0
    for size in sizes:
        chunks.append(checkpoints[start : start + size])
        start += size
    return tuple(chunks)
