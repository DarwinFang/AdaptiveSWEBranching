# Child-q audit

The main Phase-4 dataset deliberately avoids nested rollouts. Consequently, the
terminal outcome attached to one child prefix is one Bernoulli draw, not the
child state's true success probability.

The separately configured Child-q audit tests the missing mechanism directly:

> When a parent has high `4 q(parent)(1-q(parent))`, do short sibling branches
> actually reach states whose true future success probabilities spread apart?

The frozen initial setting is:

- 12 high-branchability parents, with an allowed range of 10–20;
- four distinct depth-6 child checkpoints per parent;
- eight valid independent continuations per child, with six required;
- the same 60-step continuation cap and invalid-replacement rule as Phase 4.

Depth 6 is used because it is the current primary local decision point and keeps
this mechanism audit bounded. Changing the span requires a new manifest rather
than silently pooling records.

For child `j`, derive `q_j = successes_j / valid_K_j`. The primary statistic is:

```text
max(q_j) - mean(q_j)
```

Also report child range and standard deviation, the association between parent
branchability and child spread, and two comparisons between selectors:

- Trajectory-Outcome Oracle: picks using the original one realized outcome;
- Child-q Oracle: picks the largest empirical repeated-rollout `q_j`.

Their agreement and the Child-q regret of the Trajectory-Outcome choice quantify
how misleading “peeking at this one sampled future” can be.

When several children share the same realized success/failure label, the audit
reports tie-aware best, worst, and uniform-tie expected regret rather than
pretending a lexicographic child ID is a scientific preference.

At the 12-parent target, the audit needs 384 valid nested continuations and has a
hard cap of 576 attempts after invalid replacements. Reusing the conservative
root-run proxy gives roughly 74.6M–93.9M tokens for the target-valid runs; actual
post-depth-6 continuations should usually be shorter.

The audit output lives at
`/home/fangzhaohao/asb-runs/phase4_child_q_audit_v1`. It is not merged into the
primary single-layer training dataset.
