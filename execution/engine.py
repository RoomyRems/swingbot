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
from utils.config import load_config

# ---- progress bar helper (tqdm optional) -----------------------------
try:
    from tqdm import tqdm
    def pbar(iterable, **kw):
        return tqdm(iterable, **kw)
except Exception:
    def pbar(iterable, **kw):
        return iterable

# Map a few alternate column names -> our standard names
_COLUMN_ALIASES = {
    "qty": "quantity",
    "take_profit": "target",
    "stop_loss": "stop",
    "side": "direction",  # sometimes we may already store 'buy'/'sell'
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    cfg = load_config("config.yaml")
    tif = cfg.get("trading", {}).get("time_in_force", "day").lower()
    if tif not in {"day", "gtc"}:
        tif = "day"

    if csv_path is None:
        csv_path = today_filename("trade_signals")

    if not csv_path.exists():
        print("No trade_signals CSV for today → nothing to execute.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("trade_signals CSV is empty → nothing to execute.")
        return

    df = _normalize_columns(df)
    _require_columns(df, ["symbol", "direction", "quantity", "target", "stop"])

    # ---------- Protections / pre-filters --------------------------------
    orig_rows = len(df)

    # 1) De-duplicate by (symbol, direction)
    df = df.drop_duplicates(subset=["symbol", "direction"], keep="first")
    dedup_rows = len(df)
    dropped_dupes = orig_rows - dedup_rows

    # 2) Skip symbols with open positions or open orders
    try:
        open_pos_syms   = {p.symbol.upper() for p in api.list_positions()}
    except Exception:
        open_pos_syms = set()
    try:
        open_order_syms = {o.symbol.upper() for o in api.list_orders(status="open", nested=True)}
    except Exception:
        open_order_syms = set()

    df = df[~df["symbol"].str.upper().isin(open_pos_syms | open_order_syms)]
    filtered_rows = len(df)
    dropped_open  = dedup_rows - filtered_rows

    if df.empty:
        print("All signals already have open orders/positions → nothing to execute.")
        # status banner
        print("\n— Execution summary —")
        print(f"Signals in file: {orig_rows}")
        print(f"After de-dup:   {dedup_rows} (dropped {dropped_dupes})")
        print(f"After open flt: {filtered_rows} (dropped {dropped_open})")
        print(f"Placed: 0 | Skipped BP: 0 | Failed: 0")
        return

    # 3) Prioritize by score (desc), then smaller entry first (cheaper)
    if {"score", "entry"}.issubset(df.columns):
        df = df.sort_values(by=["score", "entry"], ascending=[False, True])

    # 4) Basic sanity: positive qty, no NaNs in key fields
    df = df.dropna(subset=["symbol", "direction", "quantity", "target", "stop"])
    df = df[df["quantity"].astype(float) > 0]
    if df.empty:
        print("No valid rows to execute after sanity checks.")
        print("\n— Execution summary —")
        print(f"Signals in file: {orig_rows}")
        print(f"After de-dup:   {dedup_rows} (dropped {dropped_dupes})")
        print(f"After open flt: 0 (dropped {dedup_rows})")
        print(f"Placed: 0 | Skipped BP: 0 | Failed: 0")
        return

    # ---------- Execution loop ------------------------------------------
    placed = 0
    skipped_bp = 0
    failed = 0

    rows_iter = df.reset_index(drop=True).iterrows()
    for idx, row in pbar(rows_iter, total=len(df), desc="Placing orders", unit="order"):
        symbol = str(row["symbol"]).upper().strip()

        direction = str(row["direction"]).lower().strip()
        if direction in {"long", "bull", "buy"}:
            side = "buy"
        elif direction in {"short", "bear", "sell"}:
            side = "sell"
        else:
            print(f"[{symbol}] SKIP: unknown direction '{direction}'")
            failed += 1
            continue

        # Estimate notional and check current buying power (with buffer)
        try:
            latest = api.get_latest_trade(symbol)
            last_price = float(getattr(latest, "price", getattr(latest, "p", row.get("entry", 0.0))))
        except Exception:
            last_price = float(row.get("entry", 0.0)) or 0.0

        qty = int(row["quantity"])
        notional = last_price * qty

        try:
            acct = api.get_account()
            bp_now = float(acct.buying_power)
            # optional cap from config.risk.bp_utilization
            bp_cap = cfg.get("risk", {}).get("bp_utilization", 0.95)
            bp_allowed = bp_now * float(bp_cap)
        except Exception:
            bp_now = float("inf")
            bp_allowed = bp_now

        if notional > bp_allowed:
            print(f"[{symbol}] Skipped: notional ${notional:,.2f} exceeds BP cap (${bp_allowed:,.2f} of ${bp_now:,.2f})")
            skipped_bp += 1
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
                time_in_force=tif,
            )
            print(f"[{symbol}] {side.upper()} {qty} → order id {oid}")
            placed += 1

        except Exception as e:
            print(f"[{symbol}] FAILED: {e}")
            failed += 1

        if idx < len(df) - 1 and sleep_between > 0:
            time.sleep(sleep_between)

    # ---------- Status banner -------------------------------------------
    print("\n— Execution summary —")
    print(f"Signals in file: {orig_rows}")
    print(f"After de-dup:   {dedup_rows} (dropped {dropped_dupes})")
    print(f"After open flt: {filtered_rows} (dropped {dropped_open})")
    print(f"Placed: {placed} | Skipped BP: {skipped_bp} | Failed: {failed}")

if __name__ == "__main__":
    import os
    cfg = load_config("config.yaml")
    dry = cfg.get("dry_run", False) or os.getenv("DRY_RUN", "").lower() in {"1", "true", "yes"}
    if dry:
        print("DRY RUN: Skipping order execution.")
    else:
        execute_today()
