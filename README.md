# AdaptiveSWEBranching

AdaptiveSWEBranching studies **when temporary branching is worth its cost** in
software-engineering agents, and **which generated child should receive the
remaining compute**.

The default policy is one ordinary SWE-agent trajectory. It does not maintain a
permanent particle population. At a candidate executable checkpoint:

1. a candidate proposer identifies a place where a branch decision may be useful;
2. **Judger A / Branch Gate** estimates the expected benefit of buying local branches;
3. only when the gate accepts, the system restores the checkpoint into `N`
   isolated workspaces and runs each child for a configurable local span;
4. **Judger B / Branch Ranker** scores the resulting executable child states;
5. the search collapses to the chosen child and resumes as a single trajectory.

```text
single trajectory
      |
candidate checkpoint  <- proposer only; this is not the branch decision
      |
Judger A: is branching worth its measured compute?
      | yes
N short, executable child branches
      |
Judger B: which realized child is best?
      |
collapse to one child and continue
```

## Research questions

The project separates two decisions that a conventional state-value model
conflates:

- **A — Branching utility:** compared with continuing once from the parent,
  how much outcome/cost improvement is available after generating and selecting
  among local children?
- **B — Child preference:** after the children exist, which child has the best
  downstream outcome distribution?

Easy states may have no branching utility because every continuation succeeds.
Hopeless states may have no branching utility because every continuation fails.
The expected opportunity lies near the policy's competence boundary, once the
trajectory contains enough closed-loop diagnostic evidence.

Neither A nor B is permanently defined by one scalar formula. Rollouts preserve
raw outcomes and vector costs; analysis code can recompute alternative utilities.

## Oracle-first development

Before training either judge, counterfactual rollouts must support:

- **Oracle A:** compute branching headroom under an explicitly chosen utility;
- **Oracle B:** select children using their measured downstream futures;
- matched-compute comparison of Single Chain, Best-of-N, Random Branching,
  faithful SWE-Replay, Oracle A+B, and later learned A/B.

Phase 4 is blocked until Oracle selective branching improves the solve-rate /
compute frontier on a small gate. This repository currently implements Phases
0–3 only and does not start a large rollout.

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
  oracle/                        formula-explicit Oracle A/B analysis
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
  utilities, preferences, and rankings are never written back into raw records.
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
