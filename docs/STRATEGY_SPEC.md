# Strategy specification: public facts and v2 hypotheses

Version: `research-v1`

This document is the authority for the initial implementation. A rule should not
be added merely because it improves a backtest. Rule changes require a new
strategy version, a written rationale, and an untouched out-of-sample evaluation.

## What the public sources support

| Energy | Publicly described idea | Source |
|---|---|---|
| Trend | Direction/slope of the 50-period SMA; trend-retrace examples hold the 15 EMA | [Top Dog public course PDF](https://www.topdogtrading.com/Foundations_Trend_Trading/Top_Dog_Course1_S.pdf) |
| Momentum | Distinguishes a strong from a weak trend; public examples discuss momentum relative to zero | [Top Dog five-energy webinar outline](https://www.topdogtrading.com/WebinarRecordings/How%20To%20Trade%20With%20Complete%20Confidence%20Completed.pdf) |
| Cycle | Slow stochastic times a hook back with the trend; the public course gives 5, 2, and 3 settings when momentum is also present | [Top Dog public course PDF](https://www.topdogtrading.com/Foundations_Trend_Trading/Top_Dog_Course1_S.pdf) |
| Support/resistance | Provides entry/target price context; examples include moving averages and prior cycle levels | [Top Dog public course PDF](https://www.topdogtrading.com/Foundations_Trend_Trading/Top_Dog_Course1_S.pdf) |
| Scale | A larger timeframe should confirm with momentum; public examples prefer roughly a 3:1 timeframe ratio | [Top Dog public article/transcript](https://www.topdogtrading.com/learn-day-trading-strategies-that-work/) |

The public material also describes first and second trend retraces, a price trigger
beyond the stochastic-hook bar, partial exits, and a trailing runner. Those ideas
are not in `research-v1` because their daily-ETF translation and paper execution
need the user's notes and a more precise specification.

## Machine rules in `research-v1`

The baseline is long-only. All five energies must pass at daily close *t*:

| Energy | Exact v2 rule | Classification |
|---|---|---|
| Trend | `close > SMA50` and `SMA50[t] > SMA50[t-5]` | Public idea, quantified for v2 |
| Momentum | daily MACD(12,26,9) line is above zero | Hypothesis |
| Cycle | prior slow K <= 20, prior slow D <= 50, and current K and D both rise; stochastic is 5-2-3 | Public settings plus explicit hook definition |
| Support | low is no more than 0.25 ATR above EMA15 and close is at/above EMA15 | Hypothesis defining “EMA area holds” |
| Scale | last completed weekly MACD histogram increased from its prior completed week | Hypothesis implementing higher-timeframe momentum angle |

Weekly values are labeled on Friday. A Monday-through-Thursday daily signal can
therefore see only the prior Friday's completed weekly value. Tests require the
indicator value at any cutoff to remain identical when future bars are appended.

## Order and exit model

After a passing close:

1. Initial stop is one tick below the lower low of the signal bar and preceding bar.
2. Entry is a next-session DAY buy limit no higher than `signal close + 0.25R`.
3. Quantity is the minimum allowed by per-trade risk, total open risk, single-position notional, available cash/buying power, and position-count caps.
4. Target is a single fixed 2R limit from the planned entry limit.
5. If the next bar never touches the entry limit, the order expires.
6. The next bar's close never revalidates the prior signal.
7. If a daily bar touches stop and target, the backtest assigns the stop.
8. A target is not awarded on the entry bar when an intraday limit fill makes event order unknowable.

This fixed 2R exit is intentionally plain. It makes the first experiment easy to
falsify and maps to a broker-supported bracket. It is not claimed as a Burns exit.

## Open decisions for the user's notes

| Decision | Why it matters |
|---|---|
| Momentum definition | MACD is currently an explicit proxy, not a verified course rule |
| Retrace/wave count | The public source says first and second retraces; robust machine labeling is unresolved |
| Price trigger | The public material describes a stop trigger beyond the hook bar; v2 uses a next-session limit bracket for safe paper protection |
| S/R hierarchy | EMA support is implemented; swing, Fibonacci, and pivot priority are intentionally absent |
| Exit structure | Public material describes partial profit plus runner; v2 uses one 2R bracket |
| Shorts | Excluded until long-only behavior is validated and borrow/gap assumptions are specified |

The project is inspired by public descriptions and is not affiliated with or
endorsed by Barry Burns or Top Dog Trading.
