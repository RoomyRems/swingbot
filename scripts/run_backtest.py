# scripts/run_backtest.py
from __future__ import annotations

import sys
from pathlib import Path

# --- repo root on sys.path (keep) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import argparse
from datetime import datetime, date

from utils.config import load_config
from utils.logger import log_dataframe, today_filename
from fundamentals.screener import screen_universe
from backtest.engine import run_backtest


def _load_universe_from_watchlist(path: str = "watchlist.txt") -> list[str]:
    p = Path(path)
    if not p.exists():
        raise SystemExit("watchlist.txt not found.")
    syms = [s.strip().upper() for s in p.read_text().splitlines() if s.strip()]
    if not syms:
        raise SystemExit("watchlist.txt is empty.")
    return syms


def _parse_date(s: str) -> pd.Timestamp:
    """Parse a user-provided date string into a normalized Timestamp.

    Accepts formats:
      - YYYY-MM-DD (ISO)
      - M/D/YYYY or M/D/YY
    Raises ValueError if unparseable.
    """
    s = s.strip()
    fmts = ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return pd.Timestamp(dt.date())
        except Exception:
            continue
    raise ValueError(f"Unrecognized date format: '{s}'. Use YYYY-MM-DD or M/D/YYYY.")


def _resolve_dates(cfg: dict, args) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Determine start/end dates with precedence: CLI > config explicit > start_days_ago fallback."""
    bt = cfg.get("backtest", {}) or {}
    today = pd.Timestamp.today().normalize()
    # CLI precedence
    cli_start = _parse_date(args.start_date) if getattr(args, "start_date", None) else None
    cli_end = _parse_date(args.end_date) if getattr(args, "end_date", None) else None
    if cli_start or cli_end:
        end = cli_end or today
        start = cli_start or (end - pd.Timedelta(days=int(bt.get("start_days_ago", 300))))
    else:
        # Config explicit
        cfg_start_raw = bt.get("start_date")
        cfg_end_raw = bt.get("end_date")
        cfg_start = _parse_date(str(cfg_start_raw)) if cfg_start_raw not in (None, "", "null") else None
        cfg_end = _parse_date(str(cfg_end_raw)) if cfg_end_raw not in (None, "", "null") else None
        if cfg_start or cfg_end:
            end = cfg_end or today
            start = cfg_start or (end - pd.Timedelta(days=int(bt.get("start_days_ago", 300))))
        else:
            # Fallback to start_days_ago
            days = int(bt.get("start_days_ago", 300))
            end = today
            start = end - pd.Timedelta(days=days)
    if start > end:
        raise SystemExit(f"start date {start.date()} is after end date {end.date()}.")
    # Basic sanity: require >= 30 days unless user explicitly set both (skip enforcement if both explicit)
    if (not (cli_start and cli_end)) and (not (cfg_start and cfg_end)):
        if (end - start).days < 30:
            print(f"[Warn] Short backtest window ({(end-start).days} days). Consider >= 60 days for stability.")
    return start, end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run swingbot backtest over a specified date range.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file (default: config.yaml)")
    parser.add_argument("--start-date", dest="start_date", help="Explicit start date (YYYY-MM-DD or M/D/YYYY)", required=False)
    parser.add_argument("--end-date", dest="end_date", help="Explicit end date (YYYY-MM-DD or M/D/YYYY)", required=False)
    args = parser.parse_args()

    cfg_path = args.config
    cfg = load_config(cfg_path)

    start, end = _resolve_dates(cfg, args)
    print(f"[Backtest] Date range: {start.date()} -> {end.date()} ({(end-start).days} days)")

    # Universe
    universe = _load_universe_from_watchlist()

    # Optional fundamentals screen (reduces universe and logs a report)
    fcfg = cfg.get("fundamentals", {}) or {}
    if fcfg.get("enabled", False):
        print(f"[Fundamentals] Screening {len(universe)} symbols…")
        kept, report = screen_universe(universe, cfg)
        if not report.empty:
            log_dataframe(report, today_filename("fundamentals", unique=True))
        print(f"[Fundamentals] Kept {len(kept)} / {len(universe)} symbols.")
        universe = kept
        if not universe:
            print("No symbols left after fundamentals screen — exiting.")
            raise SystemExit(0)

    # Run the backtest (engine prints summary and writes CSVs)
    summary = run_backtest(universe, start, end, cfg_path=cfg_path)

    # Optional recap: engine already prints a summary; gate to avoid duplication
    bt_log = (cfg.get("backtest", {}).get("logging", {}) or {})
    if bool(bt_log.get("print_recap", False)):
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
