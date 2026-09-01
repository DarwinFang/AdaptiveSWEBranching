from __future__ import annotations

import random
from dataclasses import dataclass

from adaptive_swe_branching.data.records import Cost, Outcome
from adaptive_swe_branching.oracle.records import (
    ChildFutures,
    CounterfactualExperiment,
)
from adaptive_swe_branching.oracle.utility import OutcomeUtility, cost_value


@dataclass(frozen=True)
class OracleBRank:
    child_checkpoint_id: str
    success_rate: float
    mean_success_cost: float | None
    evidence_samples: int


@dataclass(frozen=True)
class OracleAMeasurement:
    utility_name: str
    no_branch_value: float
    selected_branch_value: float
    branching_headroom: float
    selected_child_checkpoint_id: str
    local_branch_cost: Cost
    evidence_samples: int
    evaluation_samples: int


class OracleB:
    """Rank realized children using measured downstream futures.

    V1 preference is success > failure, then lower successful continuation
    cost. Failure/failure cost is deliberately absent from the key.
    """

    def __init__(self, *, success_cost_axis: str = "total_tokens") -> None:
        self.success_cost_axis = success_cost_axis

    def rank(self, children: tuple[ChildFutures, ...]) -> tuple[OracleBRank, ...]:
        ranks = []
        for child in children:
            successes = [
                sample for sample in child.samples if sample.outcome == Outcome.SOLVED
            ]
            mean_cost = (
                sum(
                    cost_value(sample.cost_from_state, self.success_cost_axis)
                    for sample in successes
                )
                / len(successes)
                if successes
                else None
            )
            ranks.append(
                OracleBRank(
                    child_checkpoint_id=child.child_checkpoint_id,
                    success_rate=len(successes) / len(child.samples),
                    mean_success_cost=mean_cost,
                    evidence_samples=len(child.samples),
                )
            )
        return tuple(
            sorted(
                ranks,
                key=lambda item: (
                    -item.success_rate,
                    item.mean_success_cost
                    if item.mean_success_cost is not None
                    else float("inf"),
                    item.child_checkpoint_id,
                ),
            )
        )

    def select(self, children: tuple[ChildFutures, ...]) -> ChildFutures:
        best = self.rank(children)[0].child_checkpoint_id
        return next(child for child in children if child.child_checkpoint_id == best)


class OracleA:
    """Measure headroom under a caller-supplied, named utility definition."""

    def __init__(self, ranker: OracleB | None = None):
        self.ranker = ranker or OracleB()

    def measure(
        self,
        experiment: CounterfactualExperiment,
        *,
        utility: OutcomeUtility,
    ) -> OracleAMeasurement:
        selected = self.ranker.select(experiment.children)
        no_branch = _mean(
            utility.value(sample) for sample in experiment.no_branch_samples
        )
        local_cost = sum(
            (child.local_branch_cost for child in experiment.children), Cost()
        )
        selected_value = _mean(
            utility.value(sample, local_cost) for sample in selected.samples
        )
        return OracleAMeasurement(
            utility_name=utility.name,
            no_branch_value=no_branch,
            selected_branch_value=selected_value,
            branching_headroom=selected_value - no_branch,
            selected_child_checkpoint_id=selected.child_checkpoint_id,
            local_branch_cost=local_cost,
            evidence_samples=sum(len(child.samples) for child in experiment.children),
            evaluation_samples=len(experiment.no_branch_samples)
            + len(selected.samples),
        )

    def split_half_measure(
        self,
        experiment: CounterfactualExperiment,
        *,
        utility: OutcomeUtility,
        seed: int,
    ) -> OracleAMeasurement:
        """Select on one half and report value on disjoint futures."""
        rng = random.Random(seed)
        evidence_children: list[ChildFutures] = []
        evaluation_by_id: dict[str, ChildFutures] = {}
        for child in experiment.children:
            shuffled = list(child.samples)
            rng.shuffle(shuffled)
            split = max(1, len(shuffled) // 2)
            if split == len(shuffled):
                raise ValueError(
                    "split-half Oracle needs at least two samples per child"
                )
            evidence_children.append(
                ChildFutures(
                    child.child_checkpoint_id,
                    child.local_branch_cost,
                    tuple(shuffled[:split]),
                )
            )
            evaluation_by_id[child.child_checkpoint_id] = ChildFutures(
                child.child_checkpoint_id,
                child.local_branch_cost,
                tuple(shuffled[split:]),
            )
        selected_evidence = self.ranker.select(tuple(evidence_children))
        selected = evaluation_by_id[selected_evidence.child_checkpoint_id]
        no_branch_samples = list(experiment.no_branch_samples)
        rng.shuffle(no_branch_samples)
        no_branch_eval = no_branch_samples[len(no_branch_samples) // 2 :]
        if not no_branch_eval:
            raise ValueError("split-half Oracle needs at least two no-branch samples")
        local_cost = sum(
            (child.local_branch_cost for child in experiment.children), Cost()
        )
        no_branch = _mean(utility.value(sample) for sample in no_branch_eval)
        selected_value = _mean(
            utility.value(sample, local_cost) for sample in selected.samples
        )
        return OracleAMeasurement(
            utility_name=utility.name,
            no_branch_value=no_branch,
            selected_branch_value=selected_value,
            branching_headroom=selected_value - no_branch,
            selected_child_checkpoint_id=selected.child_checkpoint_id,
            local_branch_cost=local_cost,
            evidence_samples=sum(len(child.samples) for child in evidence_children),
            evaluation_samples=len(no_branch_eval) + len(selected.samples),
        )


def _mean(values) -> float:
    collected = list(values)
    if not collected:
        raise ValueError("mean of empty evidence")
    return sum(collected) / len(collected)
