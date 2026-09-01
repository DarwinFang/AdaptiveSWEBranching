# Phase 4 Oracle pilot protocol

Status: **planned, not launched**. This document and
`configs/experiments/phase4_oracle_pilot.yaml` are the review boundary before
any expensive rollout.

## One counterfactual pool, one success-probability model

For every executable parent checkpoint `s`, restore exactly that state `K`
times and run `K` complete independent continuations. Phase 4 normally requests
`K=8`, while the schema and label code accept a variable number of valid runs.
No short child checkpoint receives another set of downstream rollouts.

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
selected task. Choose four nonterminal parents per task:

1. one uniformly sampled from the early third of eligible checkpoint ranks;
2. one from the middle third;
3. one from the late third;
4. one proposed by the faithful SWE-Replay critical-step selector.

Every parent records `candidate_source` as `random` or `swe_replay`. Duplicate
absolute steps are removed. The planned replacement rule is in the manifest and
is explicitly a reviewable interpretation, not a fact supplied by the paper.

## Counts before rollout

| Medium tasks | Fresh sources | Parents (4/task) | Full continuations (8/parent) | Grouped parent q targets | Maximum prefix Bernoulli q targets |
|---:|---:|---:|---:|---:|---:|
| 12 | 12 | 48 | 384 | 48 | 1,536 |
| 20 | 20 | 80 | 640 | 80 | 2,560 |

The prefix maximum is `parents × 8 continuations × 4 depths`. Invalid or
too-short continuations reduce the actual number.

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
only a diagnostic operating point. For the Phase-4 upper bound, realized-outcome
Oracle B selects a successful sampled sibling when one is visible in the frozen
full continuations. Primary `N=4`; secondary `N=2`.

This Oracle B is deliberately named precisely: without nested repeats from each
child state, it knows a realized terminal outcome, not the child's true latent
success probability. The later learned B uses predicted `q(child)` and is the
actual deployable rule.

The simulator charges the selected full continuation plus the observed prefix
cost of every discarded sibling. It reports solve rate against total input plus
output tokens as the primary frontier, with model calls, steps and additive
worker wall-clock as secondary axes. Normalized budgets are `1x`, `1.5x`, `2x`
and `4x`, where `1x` is one ordinary single-chain protocol.

## Items that must be frozen at review

1. Minimum acceptable valid `K` after infrastructure-invalid runs.
2. Whether rank thirds over all nonterminal recorded checkpoints is the desired
   exact interpretation of broad early/middle/late regions.
3. Whether the duplicate replacement and parent-shortfall rules in the
   manifest are acceptable.
4. Whether the 60-step operational safety cap remains the normal full-run
   protocol for this pilot.

No learned value-model training begins in Phase 4. If the medium-only Oracle
gate is positive, the later shared model's training set must add easy and hard
states as well; medium states may be oversampled but are not the only domain.
