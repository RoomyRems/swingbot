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

### Generating / Updating the Watchlist

The legacy `generate_watchlist_sp1500.py` script has been replaced by a unified generator supporting multiple index families (S&P and Russell).

Config section (excerpt):
```yaml
watchlist:
  universes: ["sp1500"]   # options: sp500, sp1000, sp1500, russell3000, all
  alpaca_filter: false      # apply active & tradable filter (requires credentials)
  include_iwv: false        # optionally merge IWV ETF holdings (Russell 3000 proxy)
  iwv_ttl_days: 2
  iwv_force_refresh: false
```

Universes:
- `sp500` – S&P 500
- `sp1000` – S&P 400 Mid + S&P 600 Small
- `sp1500` – S&P 500 + 400 + 600 (default)
- `russell3000` – Full Russell 3000 constituents (direct scrape)
- `all` – Expands to every supported list above

Command line overrides config (always writes to canonical `watchlist.txt` in project root; previous file is overwritten):
```
python scripts/generate_watchlist.py                    # uses config.yaml universes
python scripts/generate_watchlist.py --universes sp500  # only S&P 500
python scripts/generate_watchlist.py --universes sp1000 russell3000 --alpaca-filter
python scripts/generate_watchlist.py --universes all --include-iwv
```

The script now intentionally ignores any custom output path and always overwrites `watchlist.txt` to avoid proliferation of stale watchlist files. Duplicates across indices are removed and (optionally) filtered via Alpaca for active + tradable symbols.

If you enable `include_iwv`, current IWV ETF holdings are merged (with cache TTL) to approximate broader Russell exposure; note survivorship bias & API rate considerations.

### Specifying an Explicit Backtest Date Range

You can now supply a fixed start/end date instead of relying on `start_days_ago`.

Precedence (highest first):
1. CLI `--start-date` / `--end-date`
2. `backtest.start_date` / `backtest.end_date` in `config.yaml`
3. `backtest.start_days_ago` fallback (end = today)

Accepted formats: `YYYY-MM-DD` or `M/D/YYYY` (e.g. `2024-01-01` or `1/1/2024`).

Rules:
* If only start provided → end = today
* If only end provided → start = end - `start_days_ago`
* If start > end → error
* A warning prints if the inferred window < 30 days (unless both dates explicitly set)

Examples:
```
python scripts/run_backtest.py --start-date 2024-01-01 --end-date 2024-12-21
python scripts/run_backtest.py --start-date 1/1/2024 --end-date 12/21/2024
python scripts/run_backtest.py --end-date 2024-06-30   # start inferred from start_days_ago
```

In `config.yaml`:
```yaml
backtest:
  start_days_ago: 600
  start_date: 2024-01-01   # overrides start_days_ago when set
  end_date: 2024-12-21
```

Runtime output will echo the resolved range:
```
[Backtest] Date range: 2024-01-01 -> 2024-12-21 (355 days)
```

## Environment & Credentials
The project reads API credentials from environment variables (loaded automatically via `python-dotenv` if a `.env` file is present in the project root).

Required / Optional keys:

| Purpose | Variable | Required | Notes |
|---------|----------|----------|-------|
| Alpaca API Key | `ALPACA_API_KEY` | Yes (for live/data) | Needed for fresh daily bars & any live/account operations |
| Alpaca API Secret | `ALPACA_API_SECRET` | Yes (for live/data) | Without these, broker features are disabled (tests use a stub) |
| Alpaca Paper Endpoint | `ALPACA_PAPER_ENDPOINT` | No | Defaults to `https://paper-api.alpaca.markets` |
| Financial Modeling Prep | `FMP_API_KEY` | Optional | Enables earnings blackout + fundamentals filters |
| Marketaux News | `MARKETAUX_API_KEY` | Optional | Enables news/sentiment veto if turned on in config |

### Creating a `.env` file
```
ALPACA_API_KEY=YOUR_KEY
ALPACA_API_SECRET=YOUR_SECRET
ALPACA_PAPER_ENDPOINT=https://paper-api.alpaca.markets
FMP_API_KEY=YOUR_FMP_KEY
MARKETAUX_API_KEY=YOUR_NEWS_KEY
```
`.env` is **git-ignored** (see `.gitignore`). Do not commit secrets.

### Loading env vars in PowerShell
Either restart VS Code (auto-load via `dotenv`) or dot-source the helper script:
```
. ./scripts/load_env.ps1
```
You should see:
```
[load_env] Loaded N variable(s) from <path>
```

### Verifying
```
python -c "import os; print(os.getenv('ALPACA_API_KEY') is not None, os.getenv('FMP_API_KEY') is not None)"
```

If running tests without keys, the Alpaca module now provides a dummy stub so the suite passes. For live usage, ensure real credentials are present.

## Future Ideas
-- Add alternative scale/HTF momentum filters.
- Incorporate volatility contraction pattern detection.
- Persist volume component breakdown into trade logs.

---
Feel free to open issues or extend the framework for additional edges.
