from pathlib import Path
from datetime import date
import pandas as pd
import time

from utils.config import load_config
from utils.logger import log_dataframe, today_filename
from strategies.swing_strategy import add_indicators, evaluate_five_energies, build_trade_signal
from broker.alpaca import get_daily_bars, filter_active_tradable
from execution.engine import execute_today


cfg = load_config("config.yaml")

# ---- read watch-list -------------------------------------------------
tickers = [t.strip().upper() for t in Path("watchlist.txt").read_text().splitlines() if t.strip()]

# ---- filter by Alpaca 'active & tradable' ----------------------------
filtered = filter_active_tradable(tickers)
if not tickers:
    print("No symbols to scan (watchlist empty or all filtered out).")
    raise SystemExit(0)

# Optional: show what got dropped (first few only, to keep output tidy)
dropped = sorted(set(tickers) - set(filtered))
if dropped:
    preview = ", ".join(dropped[:15]) + ("…" if len(dropped) > 15 else "")
    print(f"Filtered out {len(dropped)} inactive/untradable symbols: {preview}")

# Use the filtered list from here on
tickers = filtered

all_evals: list[dict] = []   # each symbol’s 5-energy result
trade_rows: list[dict] = []  # only those that become TradeSignal objects

for i, sym in enumerate(tickers, start=1):
    try:
        df = get_daily_bars(sym, lookback_days=120)
        if df.empty:
            print(f"[{sym}] no data (API returned empty)")
            # throttle even on empty to respect rate limits
            if i < len(tickers):
                time.sleep(0.30)
            continue

        # Ensure enough history for indicators (EMAs, ATR, etc.)
        if len(df) < 60:
            print(f"[{sym}] not enough data (have {len(df)} rows, need ~60+)")
            if i < len(tickers):
                time.sleep(0.30)
            continue

        df = add_indicators(df)
        energies = evaluate_five_energies(df)

        # store full evaluation row
        all_evals.append({
            "date": date.today(),
            "symbol": sym,
            "score": energies["score"],
            "direction": energies["direction"],
            "trend": energies["trend"],
            "momentum": energies["momentum"],
            "cycle": energies["cycle"],
            "sr": energies["sr"],
            "volume": energies["volume"],
        })

        # build risk-aware trade signal
        signal = build_trade_signal(sym, df, cfg)
        if signal:
            trade_rows.append(signal.__dict__)   # dataclass → dict

    except Exception as e:
        print(f"[{sym}] ERROR: {e}")

    # --- THROTTLE API CALLS (Alpaca paper ≈ 200 req/min) ---
    # Sleep ~0.30s between symbols; skip after the last one.
    if i < len(tickers):
        time.sleep(0.30)

# ---- write CSV logs --------------------------------------------------
if all_evals:
    log_dataframe(pd.DataFrame(all_evals), today_filename("evaluations"))

if trade_rows:
    log_dataframe(pd.DataFrame(trade_rows), today_filename("trade_signals"))
    print(f"\n✓ {len(trade_rows)} trade signal(s) logged to CSV.")
else:
    print("\nNo trade signals today.")

if not cfg.get("dry_run", False):
    execute_today()   # place any signals we just created
else:
    print("DRY RUN: Skipping order execution.")
