# AdaptiveSWEBranching

AdaptiveSWEBranching studies **when temporary branching is worth its cost** in
software-engineering agents, and **which generated child should receive the
remaining compute**.

The default policy is one ordinary SWE-agent trajectory. It does not maintain a
permanent particle population. At a candidate executable checkpoint:

1. a candidate proposer identifies a place where a branch decision may be useful;
2. **Judger A / Branch Gate** predicts whether the parent's possible outcomes
   straddle the success/failure boundary;
3. only when the gate accepts, the system restores the checkpoint into `N`
   isolated workspaces and runs each child for a configurable local span;
4. the same model scores the resulting executable child states for **Judger B**;
5. the search collapses to the chosen child and resumes as a single trajectory.

If the selected suffix later falls below a configured threshold, the default
`cold_continue` policy simply keeps following that child as the single chain:
it spends no additional branching compute and performs no restore. An optional
`ranked_rollback` policy instead restores the original branch point and then the
highest-ranked previously generated child that has not been attempted. It does
not sample a new child. After configurable `P <= N` attempts, another rollback
terminates with `branch_candidates_exhausted`. The selected policy, ranked list,
attempts and every transition are persisted in the experiment records.

This switch applies **after branching**. It is separate from Judger A's
parent-state rule: both low and high parent `q` have low `4q(1-q)` and therefore
do not trigger the ordinary uncertainty gate. The low-`q` action, generated-child
count `N`, attempted-child cap `P`, threshold and `q` reassessment interval are
independent, manifest-frozen online-policy hyperparameters. The current default
is `cold_continue`; `ranked_rollback` remains available for a controlled
comparison selected on validation data.

```text
single trajectory
      |
candidate checkpoint  <- proposer only; this is not the branch decision
      |
shared model q(parent)
      |
Judger A: 4 q(parent) (1-q(parent))
      | yes
N short, executable child branches
      |
same shared model q(child): choose max
      |
collapse to one child and continue
```

## Research questions

The project uses one conventional success-probability value model in two
different inference decisions:

- **A — Branch Gate:** evaluate `q(parent)`, then compute
  `4 q(parent) (1 - q(parent))`. A high score means the parent lies near an
  outcome boundary where temporary branching may be useful.
- **B — Branch Ranker:** after short executable child states exist, evaluate the
  same `q(child)` for each state and keep the largest.

Easy states have low branchability because almost every continuation succeeds.
Hopeless states also have low branchability because almost every continuation
fails. The expected opportunity lies near the policy's competence boundary,
once the trajectory contains enough closed-loop diagnostic evidence.

There are not two learned models, heads, or losses. The shared model estimates
`q(state) = P(success | state)` with binomial supervision. Parent states usually
have grouped evidence `(k, K)`; each saved continuation-prefix state has one
observed future and therefore contributes a Bernoulli target `(0,1)` or `(1,1)`.

All training evidence is derived from one raw unit: `K` complete, independent
continuations restored from exactly the same parent checkpoint. There is no
second child-to-downstream rollout level. The full trajectories, outcomes and
vector costs remain the source of truth, so labels can be recomputed offline.

## Oracle-first development

Before training the shared model, the same-parent continuation pool must support:

- **Oracle A:** compute `4 q(s) (1 - q(s))` and sweep gate thresholds;
- **Trajectory-Outcome Oracle:** truncate full continuations at configurable
  prefix depth `d`, then use their known terminal outcomes as a clairvoyant
  upper bound. Without nested rollout it is not an exact measurement of each
  child's latent `q`;
- matched-compute comparison of Single Chain, Best-of-N, Random Branching,
  faithful SWE-Replay, Oracle A+B, and later shared-model A/B decisions.

Phase 4 uses `K=8` only as a sampling budget, not as part of the method
definition. The raw schema and label construction support variable valid `K`.
Online simulation uses `N=4` as the primary setting, `N=2` as a lower-cost
setting, and derives `d in {1, 2, 4, 6}` from the same full continuations.

The primary one-layer experiment is accompanied by a separate, small Child-q
audit. It repeats futures from a limited number of depth-6 child checkpoints to
test whether high parent uncertainty actually produces useful true child-value
spread. This audit is not part of the primary rollout tree or training set.
Its frozen protocol is in
[`docs/CHILD_Q_AUDIT.md`](docs/CHILD_Q_AUDIT.md).

Before Phase 4, the screening stage independently budgets up to five valid
complete root trajectories per sampled SWE-smith-py task. It stops early only
when every possible outcome of the remaining runs gives the same frozen class,
and records the observed count and possible final-success interval. It labels only operational
screening classes: `screen_hard` for 0--1 successes, `screen_medium` for 2--3,
and `screen_easy` for 4--5. Screening continues without replacement until the
retained pool has at least 300 medium, 100 easy and 100 hard tasks. These runs
have `purpose=difficulty_screen` and are rejected by VF label construction;
later labels and evaluation must use fresh trajectories.

The current screening definition is success within 20 Agent steps (`q20`). A
compact `k/m` index from the earlier 60+60 task runs seeds the first three
outcomes when available; new clean runs complete the classification. This
screening-only `q20` must not be confused with the 60-step state value used by
later continuation experiments.

Screening uses a rolling four-task worker pool. The four model-service slots
map to two concurrent requests on each GPU. When one task finishes, its slot is
immediately assigned the next task instead of waiting for the slowest member of
a fixed batch.

Compute accounting for all strategy simulations is defined in
[`docs/MATCHED_COMPUTE.md`](docs/MATCHED_COMPUTE.md).

## SWE-Replay baseline

`adaptive_swe_branching.baselines.swe_replay` reproduces the complete algorithm
from Ding and Zhang, *SWE-Replay: Efficient Test-Time Scaling for Software
Engineering Agents*, arXiv:2601.22129v2:

- archive initialization and update;
- Bernoulli explore/exploit scheduling;
- regression-failure filtering;
- exact file-set state abstraction;
- rarity-weighted state selection;
- paragraph-weighted concrete-step selection;
- restore-by-diff or action replay;
- replacement of the selected step and suffix generation;
- regression filtering and majority-vote final candidate processing.

The critical-step selector is also exposed as an optional proposer for our
method. The faithful baseline runner and proposer wrapper are separate modules;
using the proposer does not imply that Judger A accepts the candidate.

Paper underspecification and our frozen interpretations are recorded in
[`docs/SWE_REPLAY_REPRODUCTION.md`](docs/SWE_REPLAY_REPRODUCTION.md).

## Repository map

```text
configs/                         immutable experiment inputs
src/adaptive_swe_branching/
  agents/                        OpenHands adapter and container-routed tools
  environments/                  SWE-smith task loading and Docker workspaces
  checkpoints/                   executable checkpoint save/verified restore
  data/                          raw records and immutable experiment manifests
  branching/                     proposer, gate, ranker, temporary branching
                                 and ranked-alternative rollback state
  screening/                     independent 8-root task stratification
  oracle/                        same-parent Oracle A/B analysis
  training/                      offline shared-q target construction
  baselines/swe_replay/          faithful SWE-Replay reproduction
  evaluation/                    matched-compute traces and curves
  cli/                           doctor and bounded smoke commands
tests/                           pure unit tests and opt-in live smoke
docs/                            glossary, schema, resources, reproduction notes
```

## Reproducibility rules

- The Hugging Face snapshot, harness commit, container digest, model digest,
  OpenHands version, full config, Git commit, and root seed are recorded in each
  experiment manifest.
- Every stochastic draw is derived from `(root_seed, semantic identifiers)`;
  concurrency order cannot change seeds.
- Raw records are append-only source data. Derived success probabilities,
  branchability scores and rankings are never written back into raw records.
- Workspaces for siblings are isolated and restores are validated before a
  continuation becomes usable evidence.
- Invalid infrastructure runs are distinct from ordinary unsolved outcomes.

## Quick checks on kb3

```bash
conda activate openhands
pip install -e .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
asb doctor --config configs/experiments/phase1_smoke.yaml
```

The live one-task smoke is opt-in because it starts Docker and makes a bounded
Ollama call:

```bash
asb smoke-checkpoint --config configs/experiments/phase1_smoke.yaml --steps 1
```

The bounded Phase-2 baseline smoke uses two one-step trials (one explore and
one paper-defined replay attempt):

```bash
asb smoke-swe-replay --config configs/experiments/swe_replay_smoke.yaml
```

The long-running independent task screen is resumable and protected by a
single-process lock:

```bash
asb screen-difficulty --config configs/experiments/difficulty_screen_v1.yaml
```

Its lightweight status is written to
`/home/fangzhaohao/asb-runs/difficulty_screen_v1/progress.json`.
