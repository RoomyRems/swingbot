# swingbot

Quant swing trading framework implementing a 5-core energy methodology (inspired by Dr. Barry Burns) plus optional higher timeframe and pattern filters.

## Core Energies (scored)
1. Trend (EMA20 vs EMA50 + price vs EMA50)
2. Momentum (MACD vs Signal + optional RSI filter + zero line + optional expansion)
3. Cycle (Stochastic %K/%D with optional hysteresis requirements)
4. Support/Resistance (pivot proximity; optional yesterday touch; EMA50 fallback if allowed)
5. Scale (weekly timeframe momentum alignment via MACD)

Volume (RVOL / Chaikin A/D slope / OBV) is now a confirmation layer and does NOT add to the core score; it’s logged for diagnostics.

Weekly (MTF) alignment can optionally add a bonus +1 to the raw core score if enabled and aligned.

## New / Updated Config Parameters
See `config.yaml` for defaults. Highlights:

### Cycle
```
cycle:
  rise_bars: 0          # %K must rise/fall N prior bars for validity (0 = off)
  require_d_slope: false  # Require SlowD slope confirmation
```

### Scale (5th core energy)
Weekly momentum alignment using MACD on W-FRI bars. In pure Burns mode, Scale is counted inside the core evaluator.

### Support / Resistance
```
signals:
  sr_pivots:
    ema50_fallback_pct: 0.025
    allow_fallback_as_core: false
```
`allow_fallback_as_core=false` means EMA50 proximity alone can't satisfy S/R; a pivot structure is required unless you flip this to true.

### Volume Confirmation
```
volume:
  min_components: 2   # of {RVOL, AD, OBV}
```
Stricter default now requires 2 agreeing components.

### Scoring & Thresholds
```
trading:
  min_core_energies: 4  # strict Burns rule: must have 4 of 5 BEFORE bonus
  min_score: 4          # effective threshold after adding MTF bonus (if any)
  energy_weights:
    trend: 1.0
    momentum: 1.0
    cycle: 1.0
    sr: 1.0
  scale: 1.0
```
`score` = raw count of core passes. `score_weighted` = sum of weights for passed energies (informational).

### Pattern Trigger Layer
```
trading:
  patterns:
    enabled: false
    modes: ["engulf", "inside_break", "nr4_break"]
```
If enabled, at least one selected pattern must be present on the signal bar. Patterns are evaluated AFTER energies pass.

### Weekly (MTF) Confirmation
Unchanged conceptually; if `counts_as_energy: true` and alignment passes, +1 bonus is added to `eff_score`. Mismatch can veto if `reject_on_mismatch: true`.

## Trade Notes Field
`TradeSignal.notes` now includes: Trend, Mom, Cycle, S/R, Scale, VolC, core count, weighted score, and MTF status.

## Upgrading From Previous Version
1. Add new sections (`cycle`, `trading.patterns`, `trading.energy_weights`, and `min_core_energies`).
2. Decide whether EMA fallback should count toward S/R (`allow_fallback_as_core`).
3. Adjust `volume.min_components` if you prefer more lenient volume confirmation.
4. (Optional) Begin tuning `energy_weights`—weights do NOT affect pass/fail thresholds unless you incorporate them externally; they are informational for ranking.

## Backtest Impact
Expect fewer signals due to stricter S/R and core minimum enforcement (volume no longer inflates score). Re-run historical backtests to recalibrate position/risk parameters.

## Quick Start
1. Populate `watchlist.txt` with tickers.
2. Configure `config.yaml` (ensure API keys via env vars for Alpaca).
3. Run daily scan:
```
python main.py
```
4. Backtest (example period configured in `config.yaml`):
```
python scripts/run_backtest.py
```

## Future Ideas
-- Add alternative scale/HTF momentum filters.
- Incorporate volatility contraction pattern detection.
- Persist volume component breakdown into trade logs.

---
Feel free to open issues or extend the framework for additional edges.
