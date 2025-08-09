# scripts/run_backtest.py
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.config import load_config
from utils.logger import log_dataframe, today_filename
from broker.alpaca import get_daily_bars
from backtest.engine import run_backtest

def _load_universe_from_watchlist(path: str = "watchlist.txt") -> list[str]:
    p = Path(path)
    if not p.exists():
        raise SystemExit("watchlist.txt not found.")
    syms = [s.strip().upper() for s in p.read_text().splitlines() if s.strip()]
    if not syms:
        raise SystemExit("watchlist.txt is empty.")
    return syms

def _loader(symbol: str, lookback_days: int) -> pd.DataFrame:
    # Reuse your live data path
    return get_daily_bars(symbol, lookback_days=lookback_days)

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    universe = _load_universe_from_watchlist()

    res = run_backtest(universe, cfg, _loader)

    trades = res["trades"]
    equity = res["equity"]
    metrics = res["metrics"]

    # Save artifacts
    out_trades = today_filename("backtest_trades")
    out_equity = today_filename("backtest_equity")
    if not trades.empty:
        log_dataframe(trades, out_trades)
    if not equity.empty:
        log_dataframe(equity, out_equity)

    # Print summary
    print("\n— Backtest Summary —")
    for k, v in metrics.items():
        if isinstance(v, float):
            if "rate" in k or "return" in k or k in {"CAGR","max_drawdown"}:
                print(f"{k:15s}: {v:.2%}")
            else:
                print(f"{k:15s}: {v:.4f}")
        else:
            print(f"{k:15s}: {v}")

    # Quick peek at top/bottom trades by R
    if not trades.empty:
        print("\nTop 5 trades by R:")
        print(trades.sort_values("r_multiple", ascending=False).head(5).to_string(index=False))
        print("\nBottom 5 trades by R:")
        print(trades.sort_values("r_multiple", ascending=True).head(5).to_string(index=False))

    print("\nFiles written:")
    if not trades.empty:
        print(f"  - {out_trades.name}")
    if not equity.empty:
        print(f"  - {out_equity.name}")
