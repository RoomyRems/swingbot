"""
execution.engine
----------------
Reads today's trade_signals CSV and sends bracket orders to Alpaca (SDK).
"""

from pathlib import Path
import time
import pandas as pd

from broker.alpaca import place_bracket, api  # uses SDK under the hood
from utils.logger import today_filename


# Map a few alternate column names -> our standard names
_COLUMN_ALIASES = {
    "qty": "quantity",
    "take_profit": "target",
    "stop_loss": "stop",
    "side": "direction",  # sometimes we may already store 'buy'/'sell'
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known alias columns to our expected names."""
    col_map = {}
    for k, v in _COLUMN_ALIASES.items():
        if k in df.columns and v not in df.columns:
            col_map[k] = v
    if col_map:
        df = df.rename(columns=col_map)
    return df


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"trade_signals CSV missing required columns: {missing}")


def execute_today(csv_path: Path | None = None, sleep_between: float = 0.20) -> None:
    """
    Load today's signals and place MARKET bracket orders:
      - entry: market
      - exits: take_profit (limit) + stop_loss (stop)
    """
    # Resolve default path like 'logs/2025-08-02_trade_signals.csv'
    if csv_path is None:
        csv_path = today_filename("trade_signals")

    # Make sure the file exists & has data
    if not csv_path.exists():
        print("No trade_signals CSV for today → nothing to execute.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("trade_signals CSV is empty → nothing to execute.")
        return

    # Standardize column names and validate required fields
    df = _normalize_columns(df)
    _require_columns(df, ["symbol", "direction", "quantity", "target", "stop"])

    # --------- FIXES / PROTECTIONS --------------------------------------

    # 1) Drop duplicates for this run (symbol+direction)
    df = df.drop_duplicates(subset=["symbol", "direction"], keep="first")

    # 2) Skip symbols that already have an open position OR open order at Alpaca
    try:
        open_pos_syms = {p.symbol.upper() for p in api.list_positions()}
    except Exception:
        open_pos_syms = set()

    try:
        # nested=True ensures we consider child legs; status="open" filters to active orders
        open_order_syms = {o.symbol.upper() for o in api.list_orders(status="open", nested=True)}
    except Exception:
        open_order_syms = set()

    df = df[~df["symbol"].str.upper().isin(open_pos_syms | open_order_syms)]

    if df.empty:
        print("All signals already have open orders/positions → nothing to execute.")
        return

    # 3) Optional: prioritize by score (desc), then smaller entry first
    if {"score", "entry"}.issubset(df.columns):
        df = df.sort_values(by=["score", "entry"], ascending=[False, True])

    # 4) Basic sanity: drop rows with non-positive qty, NaNs in critical fields
    df = df.dropna(subset=["symbol", "direction", "quantity", "target", "stop"])
    df = df[df["quantity"].astype(float) > 0]

    if df.empty:
        print("No valid rows to execute after sanity checks.")
        return

    # --------------------------------------------------------------------

    # Place orders
    for idx, row in df.reset_index(drop=True).iterrows():
        symbol = str(row["symbol"]).upper().strip()

        # direction may be 'long'/'short' OR already 'buy'/'sell'
        direction = str(row["direction"]).lower().strip()
        side = direction
        if direction in {"long", "bull", "buy"}:
            side = "buy"
        elif direction in {"short", "bear", "sell"}:
            side = "sell"
        else:
            print(f"[{symbol}] SKIP: unknown direction '{direction}'")
            continue

        # Estimate notional cost at current price and skip if not enough BP
        try:
            latest = api.get_latest_trade(symbol)
            # SDKs may expose .price or .p depending on feed; fallback to CSV entry
            last_price = float(getattr(latest, "price", getattr(latest, "p", row.get("entry", 0.0))))
        except Exception:
            last_price = float(row.get("entry", 0.0)) or 0.0

        qty = int(row["quantity"])
        notional = last_price * qty

        try:
            bp_now = float(api.get_account().buying_power)
        except Exception:
            bp_now = float("inf")  # if API hiccups, don't block (you can change this to 'continue')

        # Keep a small buffer to avoid edge rejections when price ticks up
        if notional > bp_now * 0.98:
            print(f"[{symbol}] Skipped: notional ${notional:,.2f} exceeds current BP ${bp_now:,.2f}")
            continue

        try:
            tp = float(row["target"])
            sl = float(row["stop"])

            oid = place_bracket(
                symbol=symbol,
                qty=qty,
                side=side,
                take_profit=tp,
                stop_loss=sl,
                time_in_force="day",  # change to "gtc" if you prefer
            )
            print(f"[{symbol}] {side.upper()} {qty} → order id {oid}")

        except Exception as e:
            print(f"[{symbol}] FAILED: {e}")

        # Throttle requests a bit to be nice to the API
        if idx < len(df) - 1 and sleep_between > 0:
            time.sleep(sleep_between)


if __name__ == "__main__":
    # Respect the same dry-run semantics when running this module directly.
    import os
    from utils.config import load_config

    cfg = load_config("config.yaml")
    dry = cfg.get("dry_run", False) or os.getenv("DRY_RUN", "").lower() in {"1", "true", "yes"}

    if dry:
        print("DRY RUN: Skipping order execution.")
    else:
        execute_today()
