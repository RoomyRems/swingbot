# Barry Burns book implementation map

This is the implementation audit for Barry Burns, *Trend Trading For Dummies*
(Wiley, 2014), using the supplied 377-page PDF edition. It separates what the
book says from the exact rules the software can execute. Page numbers are PDF
page numbers, not the printed-page numbers.

The book is a trading framework, not a complete unambiguous program. Whenever a
visual or discretionary instruction needed a numeric definition, that choice is
labeled as a translation. Unlabeled optimization would make a backtest look more
authoritative than it is.

## Implemented in `burns-book-v2`

| Book concept | Operational rule | Location | Status |
|---|---|---|---|
| Five energies | Evaluate Trend, Momentum, Cycle, Support/Resistance, and Scale independently | Throughout; summary checklist on PDF 205-207 | Implemented with per-energy evidence in every signal record |
| Four of five | A setup may proceed when four energies agree | PDF 205-207 | Implemented; score must be at least four |
| Scale veto | Higher-timeframe direction must agree even when four other energies pass | PDF 202, 205-207 | Implemented as a mandatory veto |
| Trend direction | For a long, price is above an upward-sloping 50-period SMA | PDF 94-96, 196-197 | Implemented; slope is measured over five daily bars |
| Early trend entry | Prefer the first two pullbacks after the 50-SMA turns up; the first pullback is recognized when stochastic %D first moves below 55 | PDF 196-197 | Implemented with a causal retrace counter |
| Cycle settings | Use stochastic 5-2-3; the second line is a three-period EMA of smoothed %K | PDF 81-83, 141-144 | Implemented exactly for indicator construction |
| Cycle region | A long cycle-low interval occurs while %D is below the 50 midpoint | PDF 81-83 | Implemented as the interval boundary; a prior %K move below 20 is reported separately because PDF 284 presents it as a scan example |
| Cycle hook | On a completed bar, %K changes from falling to rising while %D remains in the below-50 cycle-low interval | PDF 81-83, 205-226, 284 | Implemented as the Cycle energy and next-session entry reference |
| Mini-divergence | On the early retrace, price makes a lower low while %K makes a higher low, then %K hooks upward | PDF 141-144, 198-200 | Recorded as higher-probability evidence and used to rank otherwise eligible candidates; not a Cycle veto |
| Objective cycle low | The cycle low is the lowest price while %D remains below 50 | PDF 81-83 | Implemented and recorded with date and price |
| Momentum | At the cycle low, the daily MACD line should remain above zero for a long | PDF 170-176, 197-198 | Implemented at the bar containing the active cycle's price low |
| Support as a zone | Treat support as an area rather than a single exact print | PDF 146-163, 205-225 | Implemented as a fixed ATR-normalized zone |
| Moving-average support | Check the 15-EMA and 50-SMA as support | PDF 94-95, 205-225 | Implemented using values known on the cycle-low bar |
| Prior cycle levels | Prior wave/cycle highs and lows can become support | PDF 146-163, 205-225 | Implemented using only previously completed stochastic intervals |
| Scale direction | Use the direction of the MACD **line** on the next-higher chart, not histogram acceleration | PDF 177-185, 201-202 | Implemented from completed weekly bars |
| Swing scale pair | Daily setup with weekly confirmation is a common swing-trading pairing | PDF 177-185 | Implemented as the frozen timeframe pair |
| Hook-bar entry | Enter a long one tick above the completed hook bar | PDF 284 | Implemented as the stop trigger on the next session |
| Stop-limit entry | Bound the highest acceptable fill after the stop triggers | PDF 284 and order discussion | Implemented as a DAY buy stop-limit |
| Initial protective stop | Place the initial hard stop one tick below the full active cycle low | PDF 317-320 | Implemented as a stop-market bracket leg |
| Risk per trade | Examples use conservative fractional-equity risk and warn against exceeding 2% | Risk-management chapters | Default is 0.5%; the strict config permits a maximum of 2% |
| Liquidity lookback | Judge liquidity over roughly 90 sessions | PDF 255 | Implemented as trailing 90-session average daily volume |
| Market-impact cap | Position size should be no more than 0.1% of average daily volume | PDF 255 | Implemented as a hard sizing cap |
| First profit exit | Exit part of a long at the next cycle high before trailing | PDF 321-323 | `burns-cycle-v1` sells a frozen 50% on the session after a closed cycle-high hook; the exact peak is never used as a fill |
| Cycle-low trailing stop | After the first exit, move the runner stop below each new higher cycle low | PDF 324-325 | Implemented from completed 5-2-3 intervals; a stop may rise but never fall |
| Fifth-wave tightening | Near the fifth impulse, trail below the previous closed bar | PDF 324-327 | V1 retains the original shortcut; research V2 tracks confirmed wave highs continuously and can activate one-bar mode later in the runner (see REPAIR_STUDY.md) |
| Audit discipline | Record the reason for entry/exit and evaluate average wins, average losses, fees, and mistakes | Journaling and evaluation chapters | Signals include all energy values/rules/context; reports include win/loss, payoff, fees, and expectancy fields |
| Avoid curve fitting | Freeze rules before out-of-sample evaluation | Testing and system-development discussion | Strategy version and rule fingerprint are written into every summary |

## Explicit machine translations

These rules are necessary for deterministic execution but are not numeric claims
made by the book.

| Translation | Frozen choice | Why |
|---|---|---|
| “50-SMA angled up” | Current 50-SMA is above its value five sessions ago | Converts a visual angle into a causal comparison |
| Support “zone” | Cycle low within 0.25 ATR of the closest eligible level, with signal close at or above it | Keeps price-scale differences comparable without using future pivots |
| Acceptable stop-limit slippage | Limit ceiling is trigger plus `max_entry_gap_r` times trigger-to-stop risk | Prevents an unbounded gap fill; default is 0.25 risk units |
| Direction/timeframe scope | Long-only US ETFs on daily bars with weekly Scale | Gives one testable baseline and avoids silently mixing structurally different markets |
| Entry lifetime | Next regular session only | Forces a stale setup to be re-evaluated after another close |
| Same-bar ambiguity | Take the adverse feasible path | Daily OHLC cannot reveal intrabar order |
| Candidate priority | Five-energy setups rank above four-energy setups; first retraces rank above second retraces; mini-divergence ranks above a plain hook within those groups | Deterministic portfolio selection when capital or slots are scarce, while preserving the book's probability distinction |
| Partial fraction | Sell 50% of original quantity | Burns says “part”; 50% is frozen from the 400/200-share example on PDF 322-323 |
| Cycle-high execution | Confirm the %K turn at close and sell next session with adverse slippage | Daily OHLC cannot provide Burns's visual cycle-high decision and a simultaneous historical peak fill without look-ahead |
| Fifth impulse | A second-retrace entry followed by a cycle high whose interval has both open and close at/above the prior cycle high | Combines Burns's objective wave definition with the already frozen retrace count |

Trend and Cycle are operationally mandatory in addition to Scale. A long entry
without an uptrend or without the hook that defines its trigger is not a coherent
instance of this translation. Consequently, Momentum or Support may be the one
missing energy, but not both.

## Corrected v1 interpretation

The first translation made mini-divergence mandatory for Cycle. The supplied
book does not support that equivalence:

- PDF 143-144 says cycle lows and highs do not always contain divergence and
  treats the pattern as higher-probability evidence.
- The worked five-energy examples on PDF 206, 209, 212-213, 218, and 222-223
  award Cycle when the stochastic turns; they do not require two troughs.
- PDF 220 explicitly mentions divergence as an additional property when it is
  present, confirming the distinction.

`burns-book-v2` therefore uses the closed stochastic hook for Cycle and retains
mini-divergence in the signal record and candidate ordering. Aggregate reports
also reconstruct how many setups would have survived the old strict veto and
separately count hooks that followed a %K move below 20.

## Useful material deliberately deferred

Deferred does not mean ignored. These ideas are preserved here so later versions
can add them without confusing them with the current evidence.

| Book material | Why it is not silently added now | Safe next implementation |
|---|---|---|
| “First retrace after the cross” before the 50-SMA has turned | The book labels this an aggressive four-of-five setup (PDF 217), while the present baseline deliberately requires a confirmed rising 50-SMA | Add a separately versioned precursor-trend state that requires a causal price/50-SMA cross and exactly one subsequent retrace |
| Elliott-style five-wave context | The book treats wave interpretation as useful but less objective than the core energy rules | Add only after a deterministic, separately versioned wave-state definition is written |
| Fibonacci support | Anchor selection is not machine-unique and would create many tuning choices | Require a frozen causal anchor algorithm and evaluate it as a separate hypothesis |
| Visually major highs and lows | “Major” is intentionally judgment-based | Define an objective pivot algorithm before testing |
| Floor-trader pivots | The book's precise prior-session formula is most directly applicable to intraday charts, not this daily swing baseline | Add to an intraday version with session-aware data |
| Higher-scale stochastic | Presented as additional confirmation; the required Scale test is MACD-line direction | Preserve as report context before considering it as a filter |
| 90-day relative strength versus a benchmark | Primarily a market-selection/ranking tool, not one of the five setup energies | Add a point-in-time universe-ranking layer and keep its results separate from setup validity |
| Low correlation across positions | The principle is sound, but no numeric threshold or lookback is mandated | Add a frozen return-correlation policy after measuring how it changes capacity and turnover |
| Bid/ask spread and gap quality | Daily OHLCV snapshots do not contain historical quotes or reliable order-book depth | Enforce with quote data in paper/live planning and licensed quote history in research |
| Visually rhythmic, orderly markets | The book's sine-wave/rhythm test is useful but visual and has no unique numeric definition | Research a separate, frozen market-quality classifier without changing the five-energy labels |
| Daily/weekly/monthly loss stops | The example percentages are explicitly illustrative and account-dependent | Add only after the owner selects limits and defines reset, liquidation, and restart semantics |
| Overnight option hedges | Options introduce contract selection, expiry, liquidity, assignment, and multi-leg recovery risks outside this ETF baseline | Treat gap stress first; design hedging as a separately approved portfolio system |
| Short trades | Borrow availability, locate costs, upward-gap risk, and order parity differ from longs | Create and validate a distinct short strategy version |
| Other chart pairs and instruments | The book applies the framework broadly, but mixing timeframes/markets creates different hypotheses | Validate each pair and market as its own frozen experiment |
| Volume-spike support/resistance zones | The phrase “volume spikes” occurs only in a scanner-feature list (PDF 279); the book does not define a lookback, spike threshold, price zone, or expiry rule | Record continuous relative-volume diagnostics first; any zone algorithm must be a separately sourced and preregistered hypothesis |

## Exit boundary in the current release

The full-position 2R take-profit is **not** Burns's exit. It is retained as the
paper-trading engineering control: every paper entry has a broker-held protective
stop and deterministic target even if SwingBot is offline. The separately
fingerprinted `burns-cycle-v1` policy is now implemented for research with
partial-fill accounting and conservative daily-bar ordering. It cannot be used
for paper submission until broker reconciliation and restart recovery exist.

## Validation consequences

- This implementation is a falsifiable interpretation, not proof that the
  strategy has an edge.
- No parameter should be changed merely to improve the same backtest interval.
- Entry and fill logic must be validated before interpreting returns; a plain
  buy-limit backtest would incorrectly fill below the hook without a breakout.
- The first credible performance result requires a frozen snapshot, development
  interval, untouched out-of-sample interval, cost stress, and paper-fill audit.
- Any future feature in the deferred table receives a new strategy version and
  an attribution showing whether it came from the book or from engineering.

This project is independent research software. It is not affiliated with or
endorsed by Barry Burns, Wiley, or Top Dog Trading, and it is not investment
advice.
