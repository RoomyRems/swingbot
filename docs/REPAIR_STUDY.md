# Audit repair study: frozen protocol

Declared before fetching or inspecting repair results. This is a development
sample (2018–2025), not a new holdout. Universe, entries in the initial comparisons,
position limits, provider, adjustment and dates stay at the pilot settings.
The success hurdles remain 15% and 20% CAGR, considered with drawdown and costs.
No optimization, leverage increase or volume filter is part of this study.

## Comparisons

Run every case on one verified snapshot, in this fixed order:

| Case | Source/entries | Exit | Slippage per side |
|---|---|---|---|
| original-static | original source, v2 entries | original limit-based target | 5 bps |
| original-burns | original source, v2 entries | original cycle manager | 5 bps |
| repaired-static | repaired execution, v2 entries | same limit-based target | 5 bps |
| repaired-burns | repaired execution, v2 entries | same original cycle manager | 5 bps |
| fill-risk-static | repaired execution, v2 entries | 2R from actual fill | 5 bps |
| wave-manager | repaired execution, v2 entries | continuous wave manager | 5 bps |
| wave-entry-static | objective-wave retraces, v3 entries | 2R from actual fill | 5 bps |
| wave-entry-burns | objective-wave retraces, v3 entries | continuous wave manager | 5 bps |
| stress-static | v3 entries | 2R from actual fill | 10 bps |
| stress-burns | v3 entries | continuous wave manager | 10 bps |

Original source is pinned to commit
`5cd61af3f1b44e236bd153fcc7c306df4e98f5b0`, tree
`654423c6aae2686e8b95c0871edfa513f596494d`. It executes in a separate
process against the same snapshot and dependency environment. Never simulate the
old source by selectively reintroducing defects in the new engine.
Compare the snapshot to prior content fingerprint
`d1f0b306459ca2837a0490152c3becf0ddc80dc06c228844b7d6c07de91016b1`.
If provider history has changed, disclose it and compare all cases on the fresh
common snapshot; do not describe it as identical to the original pilot.

## Causal rules and attribution

Under the daily model's continuous-crossing assumption, an intraday breakout
from below the trigger can reach its target afterward.
A gap above the entry limit followed by a retrace cannot use an earlier high:
only a close at/above target establishes that a subsequent target was reached.
Stop precedence remains conservative whenever the daily bar touches the stop.
Report entry-session stop ambiguity; do not delete those losses.

Sizing and portfolio reservations retain worst-fill risk. Trade R is net P&L
divided by initial quantity times (actual entry fill minus initial stop). Also
retain reserved-risk R explicitly, so old reports remain comparable. The legacy
target stays unchanged in the execution-only comparison. True fill-based 2R is
a separate research policy, rounded up to the tradable tick.

Objective waves use Burns's distinction between cycles and waves (book PDF
pages 81–83) and later-wave stop management (pages 324–327). A positive
five-session change in the 50-SMA defines a trend episode, as in the existing
translation. The first completed %D>50 interval confirms wave 1. A later such
interval advances to wave 3, 5, etc. only if a completed bar's open and close are
at/above the last confirmed wave high. The interval's maximum price becomes the
new wave high only when %D returns to/below 50. Lesser oscillator excursions do
not advance the wave count. Developing impulses can be identified from a closed
body breakout before the interval is complete. A non-rising SMA resets state.
These choices make the book operational; they are not a claim that every
discretionary chart annotation has a unique automated interpretation.

After the first partial, a runner enters one-bar trailing when the developing
or confirmed impulse reaches wave 5, including when it entered on the first
retrace. Stops only rise and changes apply from the next session. V3 replaces
the count of %D<55 excursions with the corrective retrace following confirmed
wave 1 or 3, while retaining the active %D<55 and all other entry conditions.
V2 remains available and is the default/paper entry strategy.

Report equity/cash reconciliation, exposure, per-year and per-symbol results,
actual and reserved R, all-five-plus-divergence counts, matched-entry changes,
and stop-order ambiguity. Include a fixed SPY 50-SMA next-open trend control
with the same costs, explicitly fully invested when long and not risk matched.
No positive result here is independent validation. A broader stock universe,
intraday resolution of ambiguous fills and independent data verification are
separate experiments, not implied successes or unannounced extensions.

## Publication

The hosted runner fetches once. Publish only derived reports, fingerprints and
source provenance. Never upload raw provider bars, credentials, the attached
book or extracted book text. All new policies are research-only.
