import math
from typing import Tuple, Optional

def calc_stop_and_target(
    direction: str,
    close: float,
    atr: float,
    atr_mult: float,
    reward_mult: float
) -> Tuple[Optional[float], Optional[float]]:
    """Return (stop, target) prices given direction and ATR rules."""
    if any(v is None for v in (close, atr, atr_mult, reward_mult)):
        return None, None
    if atr <= 0:
        return None, None

    if direction == "long":
        stop = close - atr_mult * atr
        r = close - stop
        target = close + reward_mult * r
    elif direction == "short":
        stop = close + atr_mult * atr
        r = stop - close
        target = close - reward_mult * r
    else:
        return None, None

    return float(stop), float(target)

def calc_position_size(
    account_equity: float,
    risk_pct: float,
    entry: float,
    stop: float
) -> int:
    """Floor to whole shares using risk dollars / per-share risk."""
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0 or entry <= 0 or account_equity <= 0 or risk_pct <= 0:
        return 0
    risk_dollars = account_equity * risk_pct
    qty = math.floor(risk_dollars / per_share_risk)
    return max(qty, 0)
