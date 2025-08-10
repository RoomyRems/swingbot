# scripts/run_backtest.py
from __future__ import annotations

import sys
from pathlib import Path

# --- repo root on sys.path (keep) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.config import load_config
from backtest.engine import run_backtest


def _load_universe_from_watchlist(path: str = "watchlist.txt") -> list[str]:
    p = Path(path)
    if not p.exists():
        raise SystemExit("watchlist.txt not found.")
    syms = [s.strip().upper() for s in p.read_text().splitlines() if s.strip()]
    if not syms:
        raise SystemExit("watchlist.txt is empty.")
    return syms


if __name__ == "__main__":
    cfg_path = "config.yaml"
    cfg = load_config(cfg_path)

    # Dates from config
    bt = cfg.get("backtest", {}) or {}
    days = int(bt.get("start_days_ago", 300))
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=days)

    # Universe
    universe = _load_universe_from_watchlist()

    # Run the backtest (engine prints summary and writes CSVs)
    summary = run_backtest(universe, start, end, cfg_path=cfg_path)

    # Short recap after engine output
    print("\n— Backtest Summary (recap) —")
    for k in ("start_equity","end_equity","total_return","CAGR","max_drawdown",
              "daily_sharpe","days","trades","win_rate","avg_win","avg_loss",
              "profit_factor","avg_R"):
        v = summary.get(k)
        if isinstance(v, float) and k in {"total_return","CAGR","max_drawdown","win_rate"}:
            print(f"{k:<15}: {v*100:.2f}%")
        else:
            print(f"{k:<15}: {v}")

    files = summary.get("files", [])
    if files:
        print("\nFiles written:")
        for f in files:
            print(f"  - {f}")
