# Low-q policy after temporary branching

At one accepted branch point, the controller generates `N` executable children
once, scores every child with the shared success model, and freezes a
deterministic order by descending `q` with candidate ID as the tie-breaker.
Candidate IDs and resulting checkpoint IDs are stable.

The first attempt uses rank 1. The manifest-frozen `low_q_action` chooses one of
three policies:

- `cold_continue`: keep following the active child as the single
  chain. Do not restore, consume another candidate, or spend more branching
  compute.
- `ranked_rollback`: restore the original parent checkpoint and then the
  highest-ranked untried child. No new alternative is sampled and no
  natural-language failure memory is added.
- `branch_current` (current default): if the evaluated state is below the
  high-q no-branch cutoff, generate a fresh temporary branch group at that
  exact executable checkpoint, rank its children and collapse to rank 1. The
  previous branch-point event links to the newly spawned branch-group ID.

Under `ranked_rollback`, `P <= N` is supplied by experiment configuration.
After `P` candidates have been attempted, another rollback terminates with
`branch_candidates_exhausted`. The state and event stream are append-only, so a
restart recovers exactly which candidates have already been tried.

The online policy has independent, manifest-frozen hyperparameters: the low-q
action, generated children `N`, maximum ranked attempts `P`, the rollback
threshold, the high-q no-branch cutoff, and the number of Agent steps between
q reassessments. They are selected on validation data. They are not parameters
of the shared `q(state)` model and must not be selected from test outcomes.

`SelectiveBranchingScheduler.reassess_active_branch` connects the shared model
to this controller. Above the threshold it keeps the active branch. Below the
threshold, `cold_continue` returns `cold_continue_candidate` without calling
the restorer, while `ranked_rollback` performs the parent-then-child restore and
returns `rollback_to_ranked_alternative`. `branch_current` instead returns
`branch_current_and_collapse` with the newly selected executable child; at or
above its high-q cutoff it returns `continue_high_q_without_branching`. Once
`P` attempted candidate IDs are persisted under `ranked_rollback`, the next
low-q decision returns `terminate` with `branch_candidates_exhausted`.

This module is infrastructure only. The project has not frozen `N`, `P`, local
branch span, thresholds or the eventual validated policy, and it does not
launch adaptive search as part of difficulty screening.
