from dataclasses import dataclass
from typing import Optional

@dataclass
class TradeSignal:
    symbol: str
    direction: str             # "long" or "short"
    score: int                 # 0..6 (MTF may add +1)
    entry: float
    stop: float
    target: float
    quantity: int
    per_share_risk: float
    total_risk: float
    notes: Optional[str] = None
