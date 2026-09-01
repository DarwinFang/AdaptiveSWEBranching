# Raw data schema

Schema version `4` is intentionally event-oriented and label-agnostic. Version
3 introduced continuation retry provenance, explicit cap hits, multiple candidate
sources for one deduplicated parent, and the separately scoped Child-q audit.
Version 4 adds independent difficulty-screen records and append-only
ranked-alternative rollback state.

## Experiment manifest

One immutable `manifest.json` records the repository commit, complete resolved
configuration and its SHA-256, root seed, step semantics, dataset revision,
harness commit, model digest, scaffold version, creation time, and host runtime.

## TaskRecord

Identity and provenance: task ID, repository, base commit, issue, benchmark,
dataset revision, split, image tag and digest, container workdir, and raw dataset
row hash.

## TrajectoryRecord

Trajectory ID, task ID, seed, optional parent checkpoint, every full
action/observation step, final outcome, final patch, termination reason, and
measured total cost. Each step includes the structured tool input as well as
model-visible action/observation text, so action replay does not depend on lossy
summaries.

## CheckpointRecord

Checkpoint ID, task/trajectory identity, absolute step, pinned image, workspace
hash, Git state and diff, modified files, history/model-input hashes, restore
fingerprint, cost to checkpoint, and paths to the workspace and scaffold state.

## ContinuationRecord and ParentContinuationGroupRecord

This is the Phase-4 source of truth. For one parent checkpoint, a parent group
links `K` independent full continuation records and records all
`candidate_sources`. Each continuation stores its exact
source checkpoint, seed, full trajectory link, terminal outcome, post-parent
vector cost, final patch and termination reason. The linked trajectory preserves
every post-parent action/observation step.

`post_parent_step_start` and `post_parent_step_count` locate the post-parent
slice explicitly even if a storage backend later embeds it in a longer
trajectory. Variable `K` is valid; invalid infrastructure continuations remain
recorded but are excluded from `K_s` when labels are derived.

`attempt_index` and `replacement_for_invalid_continuation_id` preserve every
invalid attempt and its deterministic replacement. `cap_hit` is explicit: the
main protocol counts it as unsolved, while sensitivity analysis can exclude it
without guessing from free-form termination text.

At each configured prefix depth, `prefix_checkpoint_ids_by_depth` links the
actual saved executable child checkpoint. This is necessary because B must score
the real state after `d` steps, including workspace and Git changes, not only a
text transcript.

The same group yields:

- one grouped parent success-probability target `(successes, valid_K)`;
- one Bernoulli success-probability target for every saved prefix state at
  `d in {1,2,4,6}`, using that continuation's terminal outcome.

Both target types train the same `q(state)` model. A transforms the parent's
prediction with `4 q(1-q)`; B selects the largest child prediction.

No short child is rolled out another `K` times in the Phase-4 protocol.

`ParentContinuationGroupRecord.candidate_sources` is a tuple because one unique
parent can simultaneously have provenance such as `random_middle` and
`swe_replay`. The group freezes target valid `K=8`, minimum formal `K=6`, and
links every initial and replacement attempt.

## ChildQAuditGroupRecord

This record exists only under the small, separately manifested nested audit. It
links one high-branchability parent, four depth-6 child checkpoints, and repeated
continuations from each child. It must not be confused with or silently merged
into the primary single-layer Phase-4 dataset.

## Generic online branch records

The following records remain useful for executing a later online temporary
branch decision, but they are not the Phase-4 A/B label-generation tree.

### BranchGroupRecord

Parent checkpoint, configured `N`, local span, seeds, child trajectory IDs and
resulting child checkpoint IDs. It records what was generated, not whether an
analysis later calls the group useful.

### ChildBranchRecord

Branch index/seed and links to the actual local trajectory and resulting child
checkpoint. A bounded online run may attach one selected downstream result.

### Ranked alternative controller records

Every accepted online branch group receives a deterministic ranking by
`(-q, candidate_id)`. Append-only state snapshots preserve the original parent,
stable child and checkpoint IDs, scores, full ranking, attempted IDs, current
candidate, `N`, configurable maximum attempts `P`, exhaustion state, creation
seed/config and revision. Matching events record initial selection, low-q
rollback, candidate transition, non-rollback continuation and explicit
`branch_candidates_exhausted` termination. A rollback restores the original
parent before selecting an already-generated untried child; it never resamples.

## Independent difficulty screening

`ScreeningPlanRecord` freezes the priority-then-shuffled SWE-smith-py task
order, seed, eligible-pool hash, repository mapping and any imported 20-step
`k/m` counts. `ScreeningRunRecord` links one fresh root trajectory and records
its seed, terminal outcome, invalid status and measured costs.
`ScreeningTaskRecord` is written when the frozen five-run class is
mathematically fixed, after five valid outcomes, or after explicit
infrastructure invalidation. It stores the operational `screen_hard`,
`screen_medium` or `screen_easy` class plus imported/fresh evidence counts. A deterministic cohort is
frozen only after all three quotas are reached. Every linked trajectory has
`purpose=difficulty_screen`; training label code rejects that purpose.

For day-to-day inspection, `screening_results.csv` intentionally exposes only
`task_id,k,m,class,source`. The richer raw records exist for provenance, not as
extra screening targets.

### CounterfactualGroupRecord (generic, not Phase 4)

This older generic schema can represent nested counterfactual experiments for a
future method that explicitly needs them. Phase 4 neither produces nor consumes
it. Keeping the dataclass avoids deleting a potentially useful general record;
its docstring and storage kind mark it outside the finalized Phase-4 protocol.

## VerifierRecord

Task and patch hashes, solved/unsolved/invalid, regression status, upstream
report reference, measured verifier cost, and infrastructure error when invalid.

## Derived data

Success-probability targets, branchability and Oracle ranks belong under a
separate `derived/` analysis output. They are never added to or used to rewrite
raw records.
