# Ranked-alternative rollback

At one accepted branch point, the controller generates `N` executable children
once, scores every child with the shared success model, and freezes a
deterministic order by descending `q` with candidate ID as the tie-breaker.
Candidate IDs and resulting checkpoint IDs are stable.

The first attempt uses rank 1. When a later controller decision observes an
active-state `q` below its configured rollback threshold, it restores the
original parent checkpoint and then restores the highest-ranked untried child.
No new alternative is sampled and no natural-language failure memory is added.

`P <= N` is supplied by experiment configuration. After `P` candidates have
been attempted, another rollback terminates with
`branch_candidates_exhausted`. The state and event stream are append-only, so a
restart recovers exactly which candidates have already been tried.

The online policy has four independent, manifest-frozen hyperparameters:
generated children `N`, maximum ranked attempts `P`, the low-`q` rollback
threshold, and the number of Agent steps between active-state `q`
reassessments. They are selected on validation data. They are not parameters of
the shared `q(state)` model and must not be selected from test outcomes.

`SelectiveBranchingScheduler.reassess_active_branch` connects the shared model
to this controller. Above the threshold it keeps the active branch. Below the
threshold it performs the parent-then-child restore and returns the explicit
`rollback_to_ranked_alternative` action. Once `P` attempted candidate IDs are
persisted, the next low-`q` decision returns `terminate` with
`branch_candidates_exhausted`.

This module is infrastructure only. The project has not frozen `N`, `P`, local
branch span or rollback threshold, and it does not launch adaptive search as
part of difficulty screening.
