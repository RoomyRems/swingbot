"""
risk.manager
------------
Position sizing and level calculation helpers for equities.
- compute_levels: build stop/target using ATR multiples
- size_position : shares based on % risk of equity, with optional BP cap
"""

from math import floor
from typing import Tuple

# Optional: we’ll read equity / buying power from Alpaca if configured
try:
    from broker.alpaca import api  # only used when use_broker_equity is True
except Exception:  # keep module import-safe in non-broker contexts
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

    d = direction.lower()
    if d in {"long", "buy", "bull"}:
        stop = entry - atr_mult * atr
        if stop <= 0:
            return None, None
        r    = entry - stop  # R
        tgt  = entry + reward_mult * r
    elif d in {"short", "sell", "bear"}:
        stop = entry + atr_mult * atr
        r    = stop - entry  # R
        tgt  = entry - reward_mult * r
        if tgt <= 0:
            return None, None
    else:
        return None, None

    return _round_cent(stop), _round_cent(tgt)


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


def _get_buying_power() -> float | None:
    if api is None:
        return None
    try:
        return float(api.get_account().buying_power)
    except Exception:
        return None


def size_position(cfg: dict, entry: float, stop: float) -> int:
    """
    Shares = floor( (risk_per_trade_pct * equity) / per_share_risk ).
    Also caps by buying power * bp_utilization (if broker is available).
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

    # Base size from risk
    qty = floor(risk_dollars / per_share_risk)

    # Enforce min_shares
    min_shares = int(risk.get("min_shares", 1))
    if qty < min_shares:
        return 0

    # Optional BP cap
    bp_util = float(risk.get("bp_utilization", 1.0))
    bp_now = _get_buying_power()
    if bp_now is not None and 0 < bp_util <= 1.0:
        notional_cap = bp_now * bp_util
        max_by_bp = floor(notional_cap / entry) if entry > 0 else 0
        qty = min(qty, max_by_bp)

    return max(int(qty), 0)
