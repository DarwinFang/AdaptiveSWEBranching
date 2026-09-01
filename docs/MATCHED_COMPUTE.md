# Matched-compute contract

Every strategy produces one `StrategyTrace` per task. Curves are evaluated only
after confirming that all strategies contain the exact same task IDs.
Infrastructure-invalid traces are never silently counted as failed tasks.

The evaluator reports separate axes rather than inventing one universal scalar:

- agent steps (one model response plus its action/observation);
- input, output and total tokens;
- model calls and tool calls;
- verifier calls;
- aggregate worker wall-clock seconds.

`wall_clock_seconds` is additive compute occupancy, not optimistic parallel
latency. If four branches each run for ten seconds, the total is forty worker
seconds. A separate latency analysis can later use timestamps in raw records.

Accounting rules are explicit:

- Single Chain pays for its selected continuation.
- Best-of-N pays for all N complete attempts; it yields a selected answer only
  after this compute is spent.
- Random or Trajectory-Outcome-Oracle selective branching samples `N`
  same-parent continuations,
  runs every one for the configured prefix span `d`, discards `N-1`, and runs
  the selected sibling to its recorded terminal result. The selected full
  continuation already includes its own prefix, so the simulator charges its
  full cost plus only the `N-1` discarded prefix costs.
- SWE-Replay pays the incremental cost of every explore/exploit trial, including
  restore/replay actions and regression verification.
- Checkpoint copying has no model/token cost; its measured time remains in the
  relevant restore or branching wall-clock record.

The simulation helpers consume frozen same-parent full-continuation pools. They
choose siblings using recorded seeds but never edit those pools. Oracle A
sweeps visible thresholds over `4 q(1-q)`; every sweep point is derived analysis,
not a modification of raw data.

Total input plus output tokens is the primary scientific compute axis. Model
calls, agent steps and additive worker wall-clock are also reported. Curves are
evaluated around normalized `1x`, `1.5x`, `2x`, and `4x` regimes, where `1x` is
one ordinary single-chain protocol. The central comparison is whether adaptive
branching approaches Best-of-4 solve rate with substantially less than `4x`
compute.
