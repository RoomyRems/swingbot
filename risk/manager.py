"""
risk.manager
------------
Position sizing and level calculation helpers for equities.
- compute_levels: build stop/target using ATR multiples
- size_position : shares based on % risk of equity, with optional caps
"""

from math import floor
from typing import Tuple

# Optional: read equity / buying power from Alpaca if configured
try:
    from broker.alpaca import api  # only used when use_broker_equity is True
except Exception:
    api = None


def _round_cent(x: float) -> float:
    # Round to a penny to satisfy Alpaca (no sub-penny prices)
    return round(float(x) + 1e-8, 2)


def compute_levels(
    direction: str,
    entry: float,
    atr: float,
    atr_mult: float,
    reward_mult: float,
) -> Tuple[float, float] | Tuple[None, None]:
    """
    Compute stop & target from entry using ATR multiples.
      long : stop = entry - atr_mult*ATR, target = entry + reward_mult*(entry-stop)
      short: stop = entry + atr_mult*ATR, target = entry - reward_mult*(stop-entry)
    Returns (stop, target) pennies-rounded; (None, None) when inputs invalid.
    """
    if entry is None or atr is None:
        return None, None
    if entry <= 0 or atr <= 0 or atr_mult <= 0 or reward_mult <= 0:
        return None, None

    d = (direction or "").lower()
    if d in {"long", "buy", "bull"}:
        stop = entry - atr_mult * atr
        if stop <= 0:
            return None, None
        r   = entry - stop
        tgt = entry + reward_mult * r
    elif d in {"short", "sell", "bear"}:
        stop = entry + atr_mult * atr
        r   = stop - entry
        tgt = entry - reward_mult * r
        if tgt <= 0:
            return None, None
    else:
        return None, None

    stop = _round_cent(stop)
    tgt  = _round_cent(tgt)

    # --- Enforce minimum stop distance (percent of entry) if configured in config.yaml ---
    try:
        from utils.config import load_config  # local import to avoid cycles
        cfg = load_config("config.yaml")
        min_stop_pct = float(cfg.get("risk", {}).get("min_stop_pct", 0.0))
    except Exception:
        min_stop_pct = 0.0
    if min_stop_pct > 0 and entry > 0:
        floor_dist = entry * min_stop_pct
        dist = abs(entry - stop)
        if dist < floor_dist:
            if d in {"long", "buy", "bull"}:
                stop = _round_cent(entry - floor_dist)
                r = entry - stop
                tgt = _round_cent(entry + reward_mult * r)
            else:
                stop = _round_cent(entry + floor_dist)
                r = stop - entry
                tgt = _round_cent(entry - reward_mult * r)
                if tgt <= 0:
                    return None, None

    # Guard against rounding collapsing R to zero
    if abs(entry - stop) < 0.01:
        if d in {"long", "buy", "bull"}:
            stop = _round_cent(entry - 0.01)
        else:
            stop = _round_cent(entry + 0.01)

    return stop, tgt


def _get_equity(cfg: dict) -> float:
    """
    Read equity from Alpaca if cfg['risk']['use_broker_equity'] else from config.
    """
    risk = cfg.get("risk", {})
    use_broker = bool(risk.get("use_broker_equity", False))
    if use_broker and api is not None:
        try:
            return float(api.get_account().equity)
        except Exception:
            pass  # fall back to config if API hiccups
    return float(risk.get("account_equity", 0.0))


def _get_buying_power(cfg: dict) -> float | None:
    """
    Only consult broker buying power when use_broker_equity is True.
    This prevents live BP from constraining backtests.
    """
    risk = cfg.get("risk", {})
    if not bool(risk.get("use_broker_equity", False)):
        return None
    if api is None:
        return None
    try:
        return float(api.get_account().buying_power)
    except Exception:
        return None


def size_position(cfg: dict, entry: float, stop: float) -> int:
    """
    Shares = floor( (risk_per_trade_pct * equity) / per_share_risk ).
    Caps by:
      - optional max_notional_pct of equity (if 0<pct<1)
      - optional max_shares (if >0)
      - broker buying power * bp_utilization (only if use_broker_equity=True)
    Returns 0 if not feasible.
    """
    if entry is None or stop is None or entry <= 0:
        return 0
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0:
        return 0

    risk = cfg.get("risk", {})
    equity = _get_equity(cfg)
    risk_dollars = float(risk.get("risk_per_trade_pct", 0.0)) * equity
    if risk_dollars <= 0:
        return 0

    # Apply floor to per-share risk if configured (avoid micro-stop oversizing)
    min_stop_pct = float(risk.get("min_stop_pct", 0.0))
    if min_stop_pct > 0 and entry > 0:
        psr_floor = entry * min_stop_pct
        if per_share_risk < psr_floor:
            per_share_risk = psr_floor

    # Base size from risk (recomputed after floor)
    qty = floor(risk_dollars / per_share_risk)

    # Enforce min_shares
    min_shares = int(risk.get("min_shares", 1))
    if qty < min_shares:
        return 0

    # Optional: cap by % of equity as notional (backward compat) OR new key max_position_notional_pct
    max_notional_pct_legacy = float(risk.get("max_notional_pct", 1.0))
    max_pos_notional_pct = float(risk.get("max_position_notional_pct", max_notional_pct_legacy))
    if 0.0 < max_pos_notional_pct < 1.0:
        max_qty_by_notional = floor((equity * max_pos_notional_pct) / entry)
        qty = min(qty, max_qty_by_notional)

    # Optional: absolute share cap
    max_shares = int(risk.get("max_shares", 0))
    if max_shares > 0:
        qty = min(qty, max_shares)

    # Optional: broker buying power cap (live only)
    bp_util = float(risk.get("bp_utilization", 1.0))
    bp_now = _get_buying_power(cfg)
    if (bp_now is not None) and (0 < bp_util <= 1.0) and entry > 0:
        notional_cap = bp_now * bp_util
        max_by_bp = floor(notional_cap / entry)
        qty = min(qty, max_by_bp)

    return max(int(qty), 0)
