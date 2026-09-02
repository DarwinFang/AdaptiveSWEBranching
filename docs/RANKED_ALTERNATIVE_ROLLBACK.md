# Low-q policy after temporary branching

At one accepted branch point, the controller generates `N` executable children
once, scores every child with the shared success model, and freezes a
deterministic order by descending `q` with candidate ID as the tie-breaker.
Candidate IDs and resulting checkpoint IDs are stable.

The first attempt uses rank 1. When a later controller decision observes an
active-state `q` below its configured threshold, the manifest-frozen
`low_q_action` chooses one of two policies:

- `cold_continue` (default): keep following the active child as the single
  chain. Do not restore, consume another candidate, or spend more branching
  compute.
- `ranked_rollback`: restore the original parent checkpoint and then the
  highest-ranked untried child. No new alternative is sampled and no
  natural-language failure memory is added.

Under `ranked_rollback`, `P <= N` is supplied by experiment configuration.
After `P` candidates have been attempted, another rollback terminates with
`branch_candidates_exhausted`. The state and event stream are append-only, so a
restart recovers exactly which candidates have already been tried.

The online policy has five independent, manifest-frozen hyperparameters: the
low-`q` action, generated children `N`, maximum ranked attempts `P`, the low-`q`
threshold, and the number of Agent steps between active-state `q` reassessments.
They are selected on validation data. They are not parameters of the shared
`q(state)` model and must not be selected from test outcomes.

`SelectiveBranchingScheduler.reassess_active_branch` connects the shared model
to this controller. Above the threshold it keeps the active branch. Below the
threshold, `cold_continue` returns `cold_continue_candidate` without calling
the restorer, while `ranked_rollback` performs the parent-then-child restore and
returns `rollback_to_ranked_alternative`. Once `P` attempted candidate IDs are
persisted under the latter policy, the next low-`q` decision returns `terminate`
with `branch_candidates_exhausted`.

This module is infrastructure only. The project has not frozen `N`, `P`, local
branch span, low-`q` threshold or the eventual validated low-`q` policy, and it
does not launch adaptive search as part of difficulty screening.
