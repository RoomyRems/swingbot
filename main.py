# main.py
from pathlib import Path
from datetime import date
import time
import pandas as pd
from tqdm import tqdm

from utils.config import load_config
from utils.logger import log_dataframe, today_filename
from strategies.swing_strategy import add_indicators, evaluate_five_energies, build_trade_signal
from broker.alpaca import get_daily_bars, filter_active_tradable
from execution.engine import execute_today
from fundamentals.screener import screen_universe   # fundamentals step

cfg = load_config("config.yaml")

# ---- read watch-list -------------------------------------------------
watch_path = Path("watchlist.txt")
if not watch_path.exists():
    print("watchlist.txt not found.")
    raise SystemExit(1)

tickers = [t.strip().upper() for t in watch_path.read_text().splitlines() if t.strip()]
if not tickers:
    print("No symbols in watchlist.txt.")
    raise SystemExit(0)

# ---- filter by Alpaca 'active & tradable' ----------------------------
filtered = filter_active_tradable(tickers)
if not filtered:
    print("No symbols to scan (watchlist all filtered out by Alpaca active/tradable).")
    raise SystemExit(0)

# Optional: show what got dropped (first few only)
dropped = sorted(set(tickers) - set(filtered))
if dropped:
    preview = ", ".join(dropped[:15]) + ("…" if len(dropped) > 15 else "")
    print(f"Filtered out {len(dropped)} inactive/untradable symbols: {preview}")

# ---- [1/3] fundamentals screen --------------------------------------
print(f"\n[1/3] Fundamentals screen on {len(filtered)} symbols…")
if cfg.get("fundamentals", {}).get("enabled", True):
    kept, report = screen_universe(filtered, cfg)
    # Log a daily fundamentals report
    if not report.empty:
        log_dataframe(report, today_filename("fundamentals"))
    if not kept:
        print("After fundamentals, no symbols left to scan.")
        raise SystemExit(0)
    tickers = kept
else:
    tickers = filtered

# ---- [2/3] technical scan + signal build ----------------------------
print(f"\n[2/3] Technical scan + signal building on {len(tickers)} symbols…")
all_evals: list[dict] = []
trade_rows: list[dict] = []

for i, sym in enumerate(tqdm(tickers, desc="TA + signals"), start=1):
    try:
        df = get_daily_bars(sym, lookback_days=120)
        if df.empty:
            # keep output clean under tqdm; print only if you want:
            # tqdm.write(f"[{sym}] no data (API returned empty)")
            if i < len(tickers):
                time.sleep(0.30)
            continue

        if len(df) < 60:
            # tqdm.write(f"[{sym}] not enough data (have {len(df)} rows, need ~60+)")
            if i < len(tickers):
                time.sleep(0.30)
            continue

        df = add_indicators(df)
        energies = evaluate_five_energies(df, cfg)

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

        signal = build_trade_signal(sym, df, cfg)
        if signal:
            trade_rows.append(signal.__dict__)

    except Exception as e:
        tqdm.write(f"[{sym}] ERROR: {e}")

    if i < len(tickers):
        time.sleep(0.30)

# ---- diagnostics: show top candidates by score -----------------------
if all_evals:
    ev = pd.DataFrame(all_evals).sort_values("score", ascending=False)
    cols = ["symbol", "score", "direction", "trend", "momentum", "cycle", "sr", "volume"]
    print("\nTop candidates by score:")
    print(ev[cols].head(10).to_string(index=False))

# ---- write CSV logs --------------------------------------------------
if all_evals:
    log_dataframe(pd.DataFrame(all_evals), today_filename("evaluations"))

if trade_rows:
    # (Optional) cap signals per run if you want an upper bound:
    max_n = int(cfg.get("trading", {}).get("max_signals_per_run", 0) or 0)
    if max_n > 0 and len(trade_rows) > max_n:
        trade_rows = trade_rows[:max_n]

    log_dataframe(pd.DataFrame(trade_rows), today_filename("trade_signals"))
    print(f"\n✓ {len(trade_rows)} trade signal(s) logged to CSV.")
else:
    print("\nNo trade signals today.")

# ---- [3/3] execution (prompt only if this run produced signals) -----
print("\n[3/3] Execution")
has_signals = bool(trade_rows)

if not has_signals:
    print("No trade signals to execute — skipping execution step.")
else:
    if cfg.get("dry_run", False):
        print("DRY RUN: Skipping order execution.")
    else:
        if cfg.get("trading", {}).get("confirm_before_place", True):
            resp = input("Place orders now? [y/N]: ").strip().lower()
            if resp != "y":
                print("Canceled by user.")
            else:
                execute_today()
        else:
            execute_today()
