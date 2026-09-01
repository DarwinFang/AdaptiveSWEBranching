# Glossary

**Candidate checkpoint** — an executable state proposed for a branch decision.
Proposal does not mean branching is approved.

**Shared success-probability model** — the only learned value model. It predicts
`q(state) = P(success | state)` for either a parent or an executable child
checkpoint, using the same parameters and binomial training objective.

**Judger A / Branch Gate** — applies the shared model to the parent, then uses
`4 q(1-q)` as its branchability score. A is an inference rule, not a separate
learned head.

**Branchability** — the inference-time score `4 q(1-q)`, where `q` is predicted
by the shared model. Oracle diagnostics substitute the empirical success
fraction among valid same-parent continuations. It is not a separate training
target.

**Temporary branch group** — `N` isolated child executions produced from one
parent checkpoint for a configured local span.

**Judger B / Branch Ranker** — applies the same shared model to each already
realized executable child checkpoint and selects the largest `q`. B is also an
inference rule, not a separate learned head.

**Collapse** — retain the selected child and return to a single main trajectory.

**Oracle A** — computes exact branchability from one parent's valid full
continuations. Its threshold is swept for analysis rather than frozen as a
single method constant.

**Trajectory-Outcome Oracle** — chooses a successful sibling when the frozen
sampled futures reveal one. It is a clairvoyant upper bound for that sample, not
an exact estimate of a child's latent `q`; exact child `q` would require nested
repeated continuations, which Phase 4 deliberately avoids.

**Child-q Oracle** — available only in the small nested audit. It estimates each
selected child's success probability from repeated continuations and chooses the
largest empirical child `q`.

**Full continuation** — one complete post-parent Agent run restored from the
same executable checkpoint, including every step, final verifier result, patch
and vector cost. The same pool supplies both A and B labels.

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
