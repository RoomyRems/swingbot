"""Fixed next-open SPY trend control; not a risk-matched strategy competitor."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from .config import AppConfig


def simple_trend_control(
    frame: pd.DataFrame,
    config: AppConfig,
    start: date,
    end: date,
) -> tuple[dict[str, object], pd.DataFrame]:
    sma = frame["close"].rolling(50, min_periods=50).mean()
    # State for today's open comes exclusively from yesterday's close.
    desired = frame["close"].gt(sma).shift(1, fill_value=False)
    bars = frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    cash, quantity, entries = config.risk.initial_equity, 0, 0
    slip = config.execution.slippage_bps / 10_000
    fee = config.execution.commission_per_share
    rows = []
    for timestamp, bar in bars.iterrows():
        if desired.loc[timestamp] and quantity == 0:
            fill = float(bar["open"]) * (1 + slip)
            quantity = math.floor(cash / (fill + fee))
            cash -= quantity * (fill + fee)
            entries += int(quantity > 0)
        elif not desired.loc[timestamp] and quantity:
            cash += quantity * (float(bar["open"]) * (1 - slip) - fee)
            quantity = 0
        if timestamp == bars.index[-1] and quantity:
            cash += quantity * (float(bar["close"]) * (1 - slip) - fee)
            quantity = 0
        rows.append(
            {
                "date": timestamp.date().isoformat(),
                "cash": cash,
                "equity": cash + quantity * float(bar["close"]),
                "quantity": quantity,
            }
        )
    equity = pd.DataFrame(rows)
    marks = pd.Series([config.risk.initial_equity, *equity["equity"]])
    total = cash / config.risk.initial_equity - 1
    years = max((bars.index[-1] - bars.index[0]).days / 365.25, 1 / 365.25)
    summary = {
        "control": "SPY close above SMA50, next-session open, fully invested when long",
        "risk_matched": False,
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1,
        "max_drawdown": float((marks / marks.cummax() - 1).min()),
        "entries": entries,
        "end_equity": cash,
        "slippage_bps": config.execution.slippage_bps,
    }
    return summary, equity
