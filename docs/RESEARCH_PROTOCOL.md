# Research protocol

The objective is to find out whether a rule set survives realistic tests—not to
find the prettiest equity curve.

## Data contract

- Fetch with explicit requested start and end dates.
- Keep the automatic warm-up outside the scored interval.
- Use corporate-action adjustment `all`.
- Prefer consolidated SIP data; record IEX whenever subscription limits require it.
- Never overwrite a snapshot. Create a new directory and fingerprint instead.
- Backtests read snapshots only; they never make network calls.
- Do not publish provider-licensed raw bars from a public repository. Hosted
  runs may publish the hash manifest and derived reports while keeping the raw
  snapshot inside the ephemeral runner.
- Use a fixed ETF universe first. Historical stock tests require licensed or otherwise reliable point-in-time constituents before making claims.

## Execution contract

- A close at *t* may create an order for *t+1* only.
- No field from *t+1* can alter the signal from *t*.
- DAY stop-limit entries expire after one session.
- A buy entry cannot fill until price reaches its stop trigger, and fills are
  capped at the limit.
- Stops include configured adverse slippage; limits receive no favorable fantasy fill.
- A stop wins every ambiguous stop/target daily bar.
- Commission and slippage assumptions must be reported, even when set to zero.

## Evaluation sequence

1. Preserve the `burns-book-v1` result as the record of the strict-divergence
   interpretation; do not rewrite or discard it.
2. Run `burns-book-v2` on the identical snapshot and costs. Attribute the change
   with its reconstructed v1 strict count and Cycle funnel before considering
   returns.
3. Use the early development interval only to find coding errors and understand
   trade frequency.
4. Do not optimize configuration or frozen strategy constants against return.
5. Evaluate later calendar periods separately and keep the last interval
   untouched until the rules are frozen.
6. Compare against SPY buy-and-hold and a simple trend baseline.
7. Break results down by symbol, year, volatility regime, and entry gap.
8. Inspect every extreme winner/loss and a random trade sample against the raw
   bars.
9. Repeat with higher slippage and, when available, a second data vendor.
10. Paper trade long enough to compare planned, submitted, filled, rejected,
    and canceled orders.

## Evidence expected before considering further automation

No single threshold proves an edge. At minimum, look for:

- positive expectancy after costs across multiple non-overlapping periods;
- no dependence on one symbol, one year, or a handful of outliers;
- acceptable drawdown relative to the benchmark and intended account size;
- stable results under moderately worse slippage and small rule perturbations;
- paper behavior that matches the event assumptions;
- enough trades for uncertainty ranges to be meaningful.

If the result fails, preserve it. A clearly falsified simple hypothesis is more
valuable than another layer of filters tuned to the same sample.
