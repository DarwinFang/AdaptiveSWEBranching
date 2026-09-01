# Phase 4 Oracle pilot protocol

Status: **planned, not launched**. This document and
`configs/experiments/phase4_oracle_pilot.yaml` are the review boundary before
any expensive rollout.

## One counterfactual pool, one success-probability model

For every executable parent checkpoint `s`, restore exactly that state `K`
times and run `K` complete independent continuations. Phase 4 normally requests
`K=8`, while the schema and label code accept a variable number of valid runs.
No short child checkpoint receives another set of downstream rollouts.

Eight valid continuations are the target. Infrastructure-invalid attempts are
kept as raw records and retried with new deterministic seeds, up to 12 total
attempts per parent. A parent with 8 valid runs is complete; 6 or 7 is accepted
with its actual `K_valid`; fewer than 6 is excluded from formal parent/Oracle-A
analysis while all raw attempts remain available.

Every continuation preserves the complete post-parent action/observation
history, terminal verifier outcome, post-parent steps and vector cost, final
patch and termination reason. Raw continuations are immutable source data.

The only learned quantity is:

```text
q(state) = P(the ordinary Agent policy eventually succeeds | current state)
```

The model, serialization and binomial training loss are identical for A and B.
Only the inference-time input and use of `q` differ.

For a parent, if `k` of `K_valid` continuations succeed, its grouped training
target is `(k, K_valid)` and empirical success rate is:

```text
q = k / K_valid
```

At A inference time, the gate derives `4 q(parent) (1-q(parent))`. Both an almost
certain success and an almost certain failure therefore lead to little reason
to branch, even though the model itself remains an ordinary success predictor.

For B training, save the real executable child checkpoint after
`d = 1, 2, 4, 6` steps along each full continuation. Its one observed terminal
outcome contributes a Bernoulli target `(1,1)` or `(0,1)` to the same model. At
B inference time, score all generated children with `q(child)` and select the
largest. There is no pairwise head, cost-ranking head, or separate B model.

The parent estimate is statistically better measured because it has repeated
same-state futures. Each child prefix normally has only one observed future;
across many child states those Bernoulli samples can train `q(state)`, but they
do not reveal an exact empirical `q` for any individual child.

## Independent task screening

The active screen is `difficulty_screen_v2_20step`. It measures success within
20 ordinary Agent steps, budgets up to eight valid root outcomes per task and
classifies 0--2 successes as
`screen_hard`, 3--5 as `screen_medium`, and 6--8 as `screen_easy`. Fixed-seed
sampling prioritizes the earlier 60+60 tasks, then proceeds without replacement
until all quotas are met: 300 medium,
100 easy and 100 hard. Infrastructure-invalid attempts are retained and
replaced; they never become ordinary failures. A task stops as soon as all
remaining outcomes imply the same class. The earlier task files contribute
their compatible source-run `k/m` counts, while fresh runs supply the missing
outcomes. These trajectories are selection data only and cannot become shared-q
labels or evaluation samples. Later continuation value remains a separate
60-step quantity; a `q20` screening class is never reported as `q60`.

Independent root runs may execute concurrently across the two frozen Qwen
endpoints. Batches stop before the earliest point at which a class could become
fixed: a fresh task launches four then two runs, while a task with three
imported outcomes launches three. After six outcomes, any still-needed run is
launched one at a time so concurrency never creates a known-redundant seventh
or eighth run.

The older six-run screen below remains historical pilot evidence only.

The existing screen at
`/home/fangzhaohao/recov-runs/oracle_branchability_root_screen_v1` used fresh
root trajectories, six requested runs per task, and a medium criterion of two
to four successes. Its trajectories will never be reused for A/B labels or
evaluation.

The 2026-09-01 audit found:

- 40 task directories and 196 root trajectories;
- 28 tasks with all six runs;
- only 5 complete tasks meeting the medium criterion;
- 12 incomplete task screens.

Five is below the pilot minimum of 12. The label rollout therefore remains
blocked. The independent screening pool must be extended with fresh complete
root runs; tasks are selected mechanically from the recorded success counts,
never by subjective inspection.

## Fresh source trajectories and parents

After at least 12 medium tasks exist, run one fresh source trajectory for each
selected task. Remove the initial and terminal states, order the remaining
checkpoints by trajectory position, and split the sequence into three contiguous
chunks whose sizes differ by at most one. This is relative progress, not fixed
absolute step ranges. Then choose four nonterminal parents per task:

1. one uniformly sampled from the early third of eligible checkpoint ranks;
2. one from the middle third;
3. one from the late third;
4. one proposed by the faithful SWE-Replay critical-step selector.

Every parent records all `candidate_sources`. If SWE-Replay selects a random
parent, that state is rolled out only once and keeps both provenance values,
such as `random_middle` and `swe_replay`. A fixed-seed replacement is first drawn
from the overlapped stratum, then from all remaining eligible checkpoints if the
stratum is empty. Each task therefore retains four unique parents.

## Termination and cap hits

The deployment protocol remains Qwen3-Coder + OpenHands + verifier with a
60-agent-step operational safety cap. A cap hit is explicitly recorded and is
counted as unsolved in the main analysis. A pre-specified sensitivity analysis
recomputes results after excluding cap-hit continuations. If the cap-hit rate
exceeds 5%, the adequacy of the 60-step protocol is reviewed before later phases.

## Counts before rollout

| Medium tasks | Fresh sources | Parents (4/task) | Target-valid continuations (8/parent) | Grouped parent q targets | Maximum prefix Bernoulli q targets |
|---:|---:|---:|---:|---:|---:|
| 12 | 12 | 48 | 384 | 48 | 1,536 |
| 20 | 20 | 80 | 640 | 80 | 2,560 |

The prefix maximum is `parents × 8 continuations × 4 depths`. Invalid or
too-short continuations reduce the actual number.

The target-valid totals are 384 and 640. With at most 12 total attempts per
parent, the corresponding hard attempt caps are 576 and 960. Compute proxies
below describe target-valid runs and do not silently assume that retries are
free.

The completed root screen provides only a conservative compute proxy because a
post-parent continuation is normally shorter than a full root run. Across its
196 trajectories, total-token use was 47,909,817: mean 244,438 and median
194,145 per root trajectory. Applying those root-run figures to label
continuations gives:

- 12-task pilot: about 74.6M tokens at the median proxy or 93.9M at the mean;
- 20-task pilot: about 124.3M tokens at the median proxy or 156.4M at the mean.

Fresh source trajectories add roughly 2.3M–4.9M tokens across 12–20 tasks,
depending on whether the median or mean proxy is used. Screening extension is
separate: the observed medium yield was only `5/28`, so its cost cannot be
promised from the target count. A crude extrapolation needs about 39 additional
fully screened tasks (roughly 234 root trajectories) to find seven more medium
tasks, but that is a planning heuristic, not a sampling guarantee.

## Oracle and matched compute

Oracle A sweeps thresholds over observed parent `4 q(1-q)` values. `0.75` is
only a diagnostic operating point. For the Phase-4 upper bound, the
**Trajectory-Outcome Oracle** selects a successful sampled sibling when one is
visible in the frozen full continuations. Primary `N=4`; secondary `N=2`.

This is deliberately not called a true-value or Child-q Oracle: it knows a
realized terminal outcome, not the child's latent success probability. The later
learned B uses predicted `q(child)` and is the actual deployable rule.

The simulator charges the selected full continuation plus the observed prefix
cost of every discarded sibling. It reports solve rate against total input plus
output tokens as the primary frontier, with model calls, steps and additive
worker wall-clock as secondary axes. Normalized budgets are `1x`, `1.5x`, `2x`
and `4x`, where `1x` is one ordinary single-chain protocol.

## Separate Child-q audit

The primary dataset remains single-layer. After it finishes, a separately
manifested audit samples 12 (allowed range 10–20) formal parents with empirical
branchability at least 0.75. For each parent it samples four already-saved
depth-6 child checkpoints, then requests eight valid independent continuations
from every child, again requiring at least six.

The primary audit statistic is:

```text
max_j q(child_j) - mean_j q(child_j)
```

It also reports range, standard deviation, parent-uncertainty association, and
the agreement/regret between the Trajectory-Outcome Oracle and an empirical
Child-q Oracle. At 12 parents this adds 48 child states and 384 target-valid
nested continuations (at most 576 attempts under the retry cap). These records
are not used for the primary shared-q training set.

No learned value-model training begins in Phase 4. If the medium-only Oracle
gate is positive, the later shared model's training set must add easy and hard
states as well; medium states may be oversampled but are not the only domain.
