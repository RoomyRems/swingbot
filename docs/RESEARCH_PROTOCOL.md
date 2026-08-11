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
- Use a fixed ETF universe first. Historical stock tests require licensed or otherwise reliable point-in-time constituents before making claims.

## Execution contract

- A close at *t* may create an order for *t+1* only.
- No field from *t+1* can alter the signal from *t*.
- DAY limits expire after one session.
- Buy-limit fills are capped at the limit.
- Stops include configured adverse slippage; limits receive no favorable fantasy fill.
- A stop wins every ambiguous stop/target daily bar.
- Commission and slippage assumptions must be reported, even when set to zero.

## Evaluation sequence

1. Freeze `research-v1` and the data snapshot.
2. Use an early development interval only to find coding errors and understand trade frequency.
3. Do not optimize the eleven configuration values against return.
4. Evaluate later calendar periods separately and keep the last interval untouched until the rules are frozen.
5. Compare against SPY buy-and-hold and a simple trend baseline.
6. Break results down by symbol, year, volatility regime, and entry gap.
7. Inspect every extreme winner/loss and a random trade sample against the raw bars.
8. Repeat with higher slippage and, when available, a second data vendor.
9. Paper trade long enough to compare planned, submitted, filled, rejected, and canceled orders.

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
