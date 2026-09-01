# Glossary

**Candidate checkpoint** — an executable state proposed for a branch decision.
Proposal does not mean branching is approved.

**Judger A / Branch Gate** — predicts the incremental utility of paying for
temporary branching relative to continuing once from the parent state.

**Branching headroom** — a derived quantity computed under an explicit utility
and compute accounting rule. It is not stored as raw rollout truth.

**Temporary branch group** — `N` isolated child executions produced from one
parent checkpoint for a configured local span.

**Judger B / Branch Ranker** — ranks already-realized executable children using
downstream outcome supervision.

**Collapse** — retain the selected child and return to a single main trajectory.

**Oracle A** — computes a gate decision from measured no-branch and branching
counterfactuals under a named utility.

**Oracle B** — chooses among children using their measured downstream outcome
distributions. Its success-cost tie-break axis is an explicit analysis setting
(total tokens by default), never an unrecorded property of raw rollout data.

**Proposer** — cheaply identifies candidate steps. A proposer never decides that
branching is worthwhile.

**SWE-Replay baseline** — the complete archive/explore/exploit/replay/candidate
algorithm from Ding and Zhang. It is distinct from its critical-step selector.

**Raw record** — task, trajectory, checkpoint, branch-group, child, and verifier
facts that were directly observed. Probabilities and rankings are derived data.

**Usable outcome** — solved or unsolved under a successfully executed verifier.
Infrastructure failures are invalid, not negative labels.

**Agent step** — one model response and its resulting tool action/observation.
This definition is recorded in the experiment manifest.
