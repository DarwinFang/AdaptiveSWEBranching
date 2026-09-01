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
- Random or Oracle branching pays for every local child span plus the chosen
  child's downstream continuation.
- SWE-Replay pays the incremental cost of every explore/exploit trial, including
  restore/replay actions and regression verification.
- Checkpoint copying has no model/token cost; its measured time remains in the
  relevant restore or branching wall-clock record.

The simulation helpers consume frozen raw counterfactual pools. They choose a
sample using a recorded seed but never edit those pools. Oracle A receives a
named utility object and a visible threshold; changing either creates derived
analysis, not a new interpretation of raw data.
