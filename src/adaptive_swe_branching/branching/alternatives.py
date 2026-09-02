from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

LOW_Q_ACTIONS = ("cold_continue", "ranked_rollback")


@dataclass(frozen=True)
class RankedRetryPolicy:
    """Frozen online policy after one temporary branch group is created.

    ``cold_continue`` is deliberately the default: a low-q active child keeps
    running as the single chain. ``ranked_rollback`` enables the more active
    alternative that restores the branch point and tries the next saved child.
    """

    low_q_threshold: float
    max_candidate_attempts_p: int
    q_reassessment_interval_steps: int
    low_q_action: str = "cold_continue"

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_q_threshold <= 1.0:
            raise ValueError("low_q_threshold must be in [0, 1]")
        if self.max_candidate_attempts_p < 1:
            raise ValueError("max_candidate_attempts_p must be positive")
        if self.q_reassessment_interval_steps < 1:
            raise ValueError("q_reassessment_interval_steps must be positive")
        if self.low_q_action not in LOW_Q_ACTIONS:
            raise ValueError(
                f"low_q_action must be one of {LOW_Q_ACTIONS}, "
                f"got {self.low_q_action!r}"
            )


@dataclass(frozen=True)
class RankedAlternative:
    candidate_id: str
    checkpoint_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class BranchPointState:
    """Persistent state for one generated, ranked set of alternatives."""

    branch_point_id: str
    parent_checkpoint_id: str
    alternatives: tuple[RankedAlternative, ...]
    attempted_candidate_ids: tuple[str, ...]
    current_candidate_id: str | None
    num_attempted: int
    max_attempts_p: int
    exhausted: bool
    creation_seed: int
    creation_config: dict[str, Any]
    revision: int = 0

    @classmethod
    def create(
        cls,
        *,
        branch_point_id: str,
        parent_checkpoint_id: str,
        candidates: tuple[tuple[str, str, float], ...],
        max_attempts_p: int,
        creation_seed: int,
        creation_config: dict[str, Any],
    ) -> BranchPointState:
        if not candidates:
            raise ValueError("a branch point needs at least one candidate")
        if not 1 <= max_attempts_p <= len(candidates):
            raise ValueError("max_attempts_p must satisfy 1 <= P <= N")
        candidate_ids = [candidate_id for candidate_id, _, _ in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")
        ordered = sorted(candidates, key=lambda item: (-item[2], item[0]))
        alternatives = tuple(
            RankedAlternative(
                candidate_id=candidate_id,
                checkpoint_id=checkpoint_id,
                score=float(score),
                rank=rank,
            )
            for rank, (candidate_id, checkpoint_id, score) in enumerate(
                ordered, start=1
            )
        )
        return cls(
            branch_point_id=branch_point_id,
            parent_checkpoint_id=parent_checkpoint_id,
            alternatives=alternatives,
            attempted_candidate_ids=(),
            current_candidate_id=None,
            num_attempted=0,
            max_attempts_p=max_attempts_p,
            exhausted=False,
            creation_seed=creation_seed,
            creation_config=creation_config,
        )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.alternatives)

    @property
    def candidate_scores(self) -> tuple[float, ...]:
        return tuple(item.score for item in self.alternatives)

    @property
    def deterministic_ranking(self) -> tuple[str, ...]:
        return self.candidate_ids

    def alternative(self, candidate_id: str) -> RankedAlternative:
        try:
            return next(
                item for item in self.alternatives if item.candidate_id == candidate_id
            )
        except StopIteration as error:
            raise KeyError(candidate_id) from error

    def next_untried(self) -> RankedAlternative | None:
        if self.num_attempted >= self.max_attempts_p:
            return None
        attempted = set(self.attempted_candidate_ids)
        return next(
            (item for item in self.alternatives if item.candidate_id not in attempted),
            None,
        )

    def mark_attempted(self, candidate: RankedAlternative) -> BranchPointState:
        if candidate.candidate_id in self.attempted_candidate_ids:
            raise ValueError("an attempted candidate cannot be reused")
        if self.num_attempted >= self.max_attempts_p:
            raise RuntimeError("branch candidates are exhausted")
        attempted = self.attempted_candidate_ids + (candidate.candidate_id,)
        return replace(
            self,
            attempted_candidate_ids=attempted,
            current_candidate_id=candidate.candidate_id,
            num_attempted=len(attempted),
            exhausted=len(attempted) >= self.max_attempts_p,
            revision=self.revision + 1,
        )

    def advance_revision(self, **changes: Any) -> BranchPointState:
        return replace(self, revision=self.revision + 1, **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BranchPointState:
        return cls(
            **{
                **payload,
                "alternatives": tuple(
                    RankedAlternative(**item) for item in payload["alternatives"]
                ),
                "attempted_candidate_ids": tuple(
                    payload["attempted_candidate_ids"]
                ),
            }
        )


@dataclass(frozen=True)
class BranchControllerEvent:
    branch_point_id: str
    revision: int
    action: str
    parent_checkpoint_id: str
    from_candidate_id: str | None
    to_candidate_id: str | None
    active_q: float | None
    low_q_threshold: float | None
    termination_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RollbackDecision:
    action: str
    state: BranchPointState
    selected: RankedAlternative | None
    termination_reason: str | None


class AlternativeRestorer(Protocol):
    """Restore the branch point first, then one previously generated child."""

    def restore_parent(self, checkpoint_id: str) -> None: ...

    def restore_candidate(self, checkpoint_id: str) -> None: ...


class RankedAlternativeStore:
    """Append-only state snapshots and controller events for restartability."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()

    def save(
        self, state: BranchPointState, event: BranchControllerEvent
    ) -> None:
        if state.branch_point_id != event.branch_point_id:
            raise ValueError("state and event branch point IDs differ")
        if state.revision != event.revision:
            raise ValueError("state and event revisions differ")
        branch_root = self.root / state.branch_point_id
        _write_immutable_json(
            branch_root / "states" / f"{state.revision:06d}.json",
            state.to_dict(),
        )
        _write_immutable_json(
            branch_root / "events" / f"{event.revision:06d}.json",
            event.to_dict(),
        )

    def load(self, branch_point_id: str) -> BranchPointState:
        states = sorted((self.root / branch_point_id / "states").glob("*.json"))
        if not states:
            raise FileNotFoundError(branch_point_id)
        return BranchPointState.from_dict(
            json.loads(states[-1].read_text(encoding="utf-8"))
        )

    def events(self, branch_point_id: str) -> tuple[dict[str, Any], ...]:
        paths = sorted((self.root / branch_point_id / "events").glob("*.json"))
        return tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)


class RankedAlternativeController:
    def __init__(self, *, store: RankedAlternativeStore | None = None):
        self.store = store

    def start(self, state: BranchPointState) -> RollbackDecision:
        if state.num_attempted:
            raise ValueError("start requires a new branch point")
        selected = state.next_untried()
        if selected is None:
            raise RuntimeError("new branch point has no selectable candidate")
        updated = state.mark_attempted(selected)
        event = BranchControllerEvent(
            branch_point_id=updated.branch_point_id,
            revision=updated.revision,
            action="select_initial_candidate",
            parent_checkpoint_id=updated.parent_checkpoint_id,
            from_candidate_id=None,
            to_candidate_id=selected.candidate_id,
            active_q=None,
            low_q_threshold=None,
            termination_reason=None,
        )
        self._save(updated, event)
        return RollbackDecision("select_initial_candidate", updated, selected, None)

    def evaluate_rollback(
        self,
        state: BranchPointState,
        *,
        active_q: float,
        low_q_threshold: float,
        restorer: AlternativeRestorer,
    ) -> RollbackDecision:
        if not 0.0 <= active_q <= 1.0:
            raise ValueError("active_q must be in [0, 1]")
        if not 0.0 <= low_q_threshold <= 1.0:
            raise ValueError("low_q_threshold must be in [0, 1]")
        previous = state.current_candidate_id
        if active_q >= low_q_threshold:
            updated = state.advance_revision()
            event = BranchControllerEvent(
                branch_point_id=updated.branch_point_id,
                revision=updated.revision,
                action="continue_active_candidate",
                parent_checkpoint_id=updated.parent_checkpoint_id,
                from_candidate_id=previous,
                to_candidate_id=previous,
                active_q=active_q,
                low_q_threshold=low_q_threshold,
                termination_reason=None,
            )
            self._save(updated, event)
            return RollbackDecision("continue_candidate", updated, None, None)

        selected = state.next_untried()
        if selected is None:
            updated = state.advance_revision(
                current_candidate_id=None,
                exhausted=True,
            )
            reason = "branch_candidates_exhausted"
            event = BranchControllerEvent(
                branch_point_id=updated.branch_point_id,
                revision=updated.revision,
                action="terminate_candidates_exhausted",
                parent_checkpoint_id=updated.parent_checkpoint_id,
                from_candidate_id=previous,
                to_candidate_id=None,
                active_q=active_q,
                low_q_threshold=low_q_threshold,
                termination_reason=reason,
            )
            self._save(updated, event)
            return RollbackDecision("terminate", updated, None, reason)

        # The explicit parent restore makes rollback provenance auditable. The
        # candidate restore then selects an already-generated executable child;
        # no new branch is sampled here.
        restorer.restore_parent(state.parent_checkpoint_id)
        restorer.restore_candidate(selected.checkpoint_id)
        updated = state.mark_attempted(selected)
        event = BranchControllerEvent(
            branch_point_id=updated.branch_point_id,
            revision=updated.revision,
            action="rollback_to_ranked_alternative",
            parent_checkpoint_id=updated.parent_checkpoint_id,
            from_candidate_id=previous,
            to_candidate_id=selected.candidate_id,
            active_q=active_q,
            low_q_threshold=low_q_threshold,
            termination_reason=None,
        )
        self._save(updated, event)
        return RollbackDecision(
            "rollback_to_ranked_alternative", updated, selected, None
        )

    def record_cold_continue(
        self,
        state: BranchPointState,
        *,
        active_q: float,
        low_q_threshold: float,
    ) -> RollbackDecision:
        """Record that low q was observed but rollback is disabled by policy."""

        if not 0.0 <= active_q <= 1.0:
            raise ValueError("active_q must be in [0, 1]")
        if not 0.0 <= low_q_threshold <= 1.0:
            raise ValueError("low_q_threshold must be in [0, 1]")
        current = state.current_candidate_id
        updated = state.advance_revision()
        event = BranchControllerEvent(
            branch_point_id=updated.branch_point_id,
            revision=updated.revision,
            action="cold_continue_active_candidate",
            parent_checkpoint_id=updated.parent_checkpoint_id,
            from_candidate_id=current,
            to_candidate_id=current,
            active_q=active_q,
            low_q_threshold=low_q_threshold,
            termination_reason=None,
        )
        self._save(updated, event)
        return RollbackDecision("cold_continue_candidate", updated, None, None)

    def _save(
        self, state: BranchPointState, event: BranchControllerEvent
    ) -> None:
        if self.store is not None:
            self.store.save(state, event)


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"immutable controller record differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)
