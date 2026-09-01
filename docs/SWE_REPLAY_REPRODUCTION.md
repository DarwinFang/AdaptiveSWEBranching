# SWE-Replay reproduction contract

Paper: Yifeng Ding and Lingming Zhang, *SWE-Replay: Efficient Test-Time Scaling
for Software Engineering Agents*, arXiv:2601.22129v2 (2026-02-05).

No official implementation was discoverable as of 2026-09-01. This reproduction
therefore distinguishes paper-defined behavior from underspecified engineering
choices. Section and equation references below point to the v2 source.

## Paper-defined behavior

| Paper location | Required implementation |
|---|---|
| Algorithm 1, lines 1–3 | Start with an empty archive; the first trial is always a fresh run. |
| §2.1.4 / Algorithm 1 | Each later trial independently chooses explore with probability 0.5, otherwise exploit. |
| §2.1.1 | Exclude archived trajectories whose final patch introduces regression failures from replay selection. |
| §2.1.2, Eq. 1 | Represent every concrete step by the exact set of repository files explored before it; group equal sets; sample a group with `softmax(1 / group_size)`. |
| §2.1.3, Eq. 2 | Within the sampled group, sample a concrete step with `softmax(reasoning_paragraph_count)`. |
| §2.2 | Restore the state immediately before the selected step. Use accumulated repository diff when no non-repository mutation occurred; otherwise replay prefix actions. |
| §2.3 | Generate a replacement for the selected step, continue the suffix, concatenate prefix and suffix, and add the new trajectory to the archive. |
| Appendix Algorithm 1 | After all trials, regression-filter candidate patches and choose the final patch by majority vote. |
| Appendix A.1 | Verified/Multilingual experiments use 10 trials; SWE-Bench Pro uses 5. |

The code comments `Paper Algorithm 1 line …` map these operations to functions.

## Underspecified details and frozen interpretations

### 1. What counts as an explored repository file?

**Paper pointer:** §2.1.2 says the state is the set of files explored before a
step, but gives no extraction algorithm.

**Interpretation:** collect canonical repository-relative paths from structured
tool input and tool observations. Directories and paths outside the mounted
repository are excluded; symlinks are resolved only when the target stays inside
the repository. The set is cumulative and excludes the selected step itself.

**Reason:** this matches the paper's file-level abstraction while avoiding model
prose and host-path leakage.

### 2. What is a reasoning paragraph?

**Paper pointer:** §2.1.3 defines intensity as paragraph count but does not define
blank-line parsing or empty reasoning.

**Interpretation:** normalize newlines, strip Markdown fences only as ordinary
text, and count non-empty blocks separated by one or more blank lines. Empty
reasoning has count zero.

**Reason:** deterministic, scaffold-independent structural segmentation.

### 3. No regression-clean trajectory is available

**Paper pointer:** §2.1.1 requires filtering but does not define an empty-pool
fallback.

**Interpretation:** the exploit attempt falls back to a fresh explore run and
records `exploit_fallback_no_eligible_trajectory`.

**Reason:** a replay cannot satisfy the paper's reliability condition without an
eligible prefix; silently selecting a filtered trajectory would contradict it.

### 4. Detecting non-repository mutation

**Minimal paper excerpt:** “checking with pattern matching whether agents'
actions have mutated non-repo state” (§2.2).

**Interpretation:** conservatively classify package-manager commands, writes or
deletes outside the repository, process/service management, environment mutation,
and permission/ownership changes as action-replay-required. The matched rule is
stored with the restore plan.

**Reason:** the paper does not publish its patterns. False positives cost replay
time but preserve state; false negatives silently corrupt restoration.

### 5. Action replay and nondeterminism

**Paper pointer:** §2.2 does not state how failed/nondeterministic prefix actions
are audited.

**Interpretation:** replay structured actions in order, then require the resulting
workspace hash and Git state to match the archived pre-step checkpoint. A mismatch
invalidates that replay attempt rather than becoming an ordinary task failure.

### 6. Suffix step budget

**Paper pointer:** Appendix A.1 reports per-agent maximum step limits but does not
say whether a replay receives a fresh full limit.

**Interpretation:** the concatenated trajectory has the same absolute maximum as
a fresh trajectory; the suffix receives `max_steps - prefix_steps`.

**Reason:** this preserves the paper's cost-saving claim and compares trajectories
under one common total step cap.

### 7. Majority vote identity and ties

**Paper pointer:** Appendix A.1 names majority voting but does not specify patch
normalization or ties.

**Interpretation:** vote by SHA-256 of normalized unified-diff text (LF endings,
trailing whitespace removed, final newline). Ties choose the earliest generated
candidate, and the tie is recorded.

**Reason:** exact and auditable; no unreported semantic patch judge is introduced.

### 8. Regression tests on SWE-smith

**Paper pointer:** the experiments use Agentless-derived existing regression tests;
SWE-smith was not evaluated in the paper.

**Interpretation:** use the frozen SWE-smith `PASS_TO_PASS` tests. The ordinary
official harness report provides both resolution and regression status, so a patch
is replay-eligible exactly when no required `PASS_TO_PASS` test failed.

### 9. Explicit prompt-cache cost

**Paper pointer:** Appendix A.3 reports explicit commercial prompt-cache cost.

**Interpretation:** raw input/output tokens and wall time are authoritative on the
local Ollama setting. Cache-adjusted monetary cost is unavailable and is not
fabricated. Prefix sharing is recorded so a future backend-specific cost model can
be applied offline.

## Baseline versus proposer

`SWEReplayRunner` owns archive filtering, explore/exploit scheduling, restoration,
branch suffix generation, archive update, and final candidate processing.
`SWEReplayCriticalStepProposer` only exposes candidate steps. The adaptive method
must still ask Judger A whether to spend branching compute.

