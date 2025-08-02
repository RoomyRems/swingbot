# data/fetcher.py
"""
Unified OHLCV fetcher that delegates to Alpaca (daily bars).
"""

from __future__ import annotations
import pandas as pd
from broker.alpaca import get_daily_bars

def fetch_ohlcv(symbol: str, lookback_days: int = 120) -> pd.DataFrame:
    """
    Fetch daily OHLCV via Alpaca (IEX feed).
    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    and a Date index (UTC dropped).
    """
    return get_daily_bars(symbol, lookback_days=lookback_days)
