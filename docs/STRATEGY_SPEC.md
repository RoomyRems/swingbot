# Strategy specification: Burns book baseline

Version: `burns-book-v1`

This document is the machine authority for the current strategy. The primary
human source is Barry Burns, *Trend Trading For Dummies* (Wiley, 2014). Page
references below are PDF page numbers in the supplied 377-page edition. See
[the implementation map](BOOK_IMPLEMENTATION_MAP.md) for the broader book audit.

A rule must not be changed merely because it improves a backtest. A rule change
requires a new strategy version, a rationale, and an untouched out-of-sample
evaluation. `strategy_fingerprint()` records the frozen rule values.

## Scope

- Long-only US ETFs
- Daily setup chart
- Last completed weekly chart for Scale
- Signal only after the daily bar closes
- One next-session DAY stop-limit bracket
- Backtest and paper planning use the same signal and sizing functions

The book presents the method symmetrically for long and short trades and across
many timeframes. Shorts remain outside this version because stock-borrow,
locate, gap, and paper/live parity rules require separate validation.

## Machine rules

At daily close *t*, the implementation evaluates five independent energies.
The setup must score at least four of five. Scale is always required, as Burns
states. Trend and Cycle are also operationally required in this long-only
translation because they establish direction and the entry trigger. Therefore,
Momentum or Support may be the one missing energy, but not both.

| Energy | `burns-book-v1` rule | Book basis | Translation |
|---|---|---|---|
| Trend | Close is above the 50-SMA; its five-bar fractional slope is positive; the active stochastic retrace is the first or second since that rising-SMA epoch began | Ch. 6, PDF 94-96; Ch. 14, PDF 196-197 | Five bars quantify an otherwise visual angle |
| Momentum | Daily MACD(12,26,9) line is above zero at the active cycle low | Ch. 12, PDF 170-176; Ch. 14, PDF 197-198 | Direct implementation |
| Cycle | Burns 5-2-3 stochastic: smoothed %K hooks upward while %D is below 50, and price makes a lower low while %K makes a higher low | Ch. 5, PDF 81-83; Ch. 10, PDF 141-144; Ch. 14, PDF 198-200 | Direct, causal mini-divergence implementation |
| Support | The active cycle low is within 0.25 ATR of the 15-EMA, 50-SMA, previous confirmed cycle low, or previous confirmed cycle high, and the signal close is above that level | Ch. 6, PDF 94-95; Ch. 11, PDF 146-163; Ch. 15, PDF 205-225 | ATR converts Burns's price “zone” into a frozen rule |
| Scale | The MACD line on the last completed weekly bar is higher than on the preceding completed week | Ch. 13, PDF 177-185; Ch. 14, PDF 201-202 | Daily/weekly is the book's common swing pairing; positive delta quantifies angle |

Stochastic uses a 5-bar range, a two-period simple smoothing for %K, and a
three-period exponential average for %D. A cycle-low interval begins when %D
moves below 50 and ends when it returns above 50. The active cycle low is the
lowest price seen in that interval through *t*.

The first retrace begins the first time %D falls below 55 after the 50-SMA
starts its current rising epoch. Each later move from at/above 55 to below 55
increments the retrace count. Only retraces one and two qualify.

## Entry, initial stop, and size

After a qualifying bar closes:

1. The buy-stop trigger is one tick above that closed hook bar's high.
2. The stop-limit ceiling is the trigger plus at most `max_entry_gap_r` times
   trigger-to-protective-stop risk.
3. The hard protective stop is one tick below the active cycle low.
4. The order is valid for the next regular session only.
5. Position size uses the conservative limit price and is the smallest quantity
   allowed by per-trade risk, total open risk, notional, buying power, slot, and
   liquidity caps.
6. Quantity may not exceed 0.1% of trailing 90-session average daily volume,
   matching the liquidity rule in Chapter 18 (PDF 255).

Burns describes buying one tick above the hook bar (PDF 284), entering with a
stop-limit order, using hard stop-market protection (PDF 317-320), and placing
the initial stop one tick below the cycle low (PDF 317).

## Causal daily-bar execution

- Information through close *t* creates an order for *t+1*.
- The next bar must trade at or above the buy-stop trigger.
- A gap between trigger and limit may fill at the open plus configured adverse
  slippage, capped at the limit.
- A gap above the limit does not fill unless price later trades down to the
  limit after the stop has triggered.
- No next-bar close is used to approve a fill.
- If a daily bar cannot reveal whether entry, stop, or target occurred first,
  the backtest takes the adverse path.

## Exit translation

The current full-position 2R target is deliberately **not** attributed to Burns.
It remains a simple, broker-supported paper bracket so every filled position has
both a hard protective stop and a deterministic target without a continuously
running order manager.

Burns's book exit is richer: take a partial exit at the next cycle high, then
trail the runner below successive cycle lows; near the fifth wave, tighten to a
one-bar stop (Ch. 23, PDF 321-327). That workflow is recorded for the next
version but must not replace the bracket until backtest event ordering, partial
fills, broker reconciliation, restart recovery, and stop replacement are all
implemented fail-closed.

## Intentionally excluded from the signal

- Fibonacci support: the book does not provide a machine-unique anchor pair.
- Visually “major” highs/lows: explicitly described as subjective.
- Floor-trader pivots: objective, but chiefly defined from the prior session for
  intraday use; their daily swing interpretation is not specified.
- Optional higher-scale stochastic confirmation: useful context, but Scale's
  required signal is the higher-scale MACD-line direction.
- Relative-strength ranking: the book uses a 90-day comparison as a market
  selection tool, not as one of the five setup energies.
- Correlation, spread, and gap filters: valuable universe/execution controls,
  but daily OHLCV snapshots cannot validate live bid/ask spread.
- Dynamic loss stops by day, week, month, quarter, and year: Burns labels the
  percentages as examples and tells traders to choose their own values.

These exclusions prevent discretionary ideas from becoming unacknowledged
optimization knobs. They can be introduced only as separately versioned,
testable research hypotheses.

This project is independent research software. It is not affiliated with or
endorsed by Barry Burns, Wiley, or Top Dog Trading, and it is not investment
advice.
