# Raw data schema

Schema version `1` is intentionally event-oriented and utility-agnostic.

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

## BranchGroupRecord

Parent checkpoint, configured `N`, local span, seeds, child trajectory IDs and
resulting child checkpoint IDs. It records what was generated, not whether an
analysis later calls the group useful.

## ChildBranchRecord

Branch index/seed and links to the actual local trajectory and resulting child
checkpoint. A bounded run may attach one immediate downstream result here; the
general repeated-continuation evidence is represented below.

## ContinuationRecord and CounterfactualGroupRecord

Every no-branch or child-downstream draw is a separate continuation record with
its source checkpoint, role, seed, full trajectory link, outcome, vector cost,
patch and termination. A counterfactual group links the parent checkpoint, its
no-branch draws, one concrete branch group, and all downstream draws for every
child. This preserves the complete empirical distributions needed by Oracle A
and B without storing success probability, headroom, or a preferred utility in
the raw layer.

## VerifierRecord

Task and patch hashes, solved/unsolved/invalid, regression status, upstream
report reference, measured verifier cost, and infrastructure error when invalid.

## Derived data

Success probabilities, conditional costs, branching headroom, pairwise
preference and Oracle ranks belong under a separate `derived/` analysis output.
They are never added to or used to rewrite raw records.
