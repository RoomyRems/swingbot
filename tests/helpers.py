from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from swingbot.config import AppConfig, DataConfig, ExecutionConfig, RiskConfig
from swingbot.models import EnergyEvidence, Signal


def make_bars(
    rows: int = 320,
    start: str = "2020-01-01",
    slope: float = 0.08,
) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=rows)
    x = np.arange(rows, dtype=float)
    close = 100.0 + slope * x + 1.8 * np.sin(x / 8.0)
    open_ = close - 0.15 * np.cos(x / 5.0)
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000_000.0 + x * 100.0,
        },
        index=index,
    )


def app_config(*symbols: str, slippage_bps: float = 0.0) -> AppConfig:
    return AppConfig(
        symbols=tuple(symbols),
        data=DataConfig(feed="sip", adjustment="all"),
        risk=RiskConfig(
            initial_equity=100_000.0,
            risk_per_trade=0.005,
            max_total_risk=0.03,
            max_positions=5,
            max_position_fraction=0.20,
            reward_r=2.0,
        ),
        execution=ExecutionConfig(
            max_entry_gap_r=0.25,
            slippage_bps=slippage_bps,
            commission_per_share=0.0,
        ),
    )


def complete_signal(symbol: str, signal_date: date) -> Signal:
    energies = {
        name: EnergyEvidence(True, 1.0, "test")
        for name in ("trend", "momentum", "cycle", "support", "scale")
    }
    return Signal(
        symbol=symbol,
        signal_date=signal_date,
        reference_price=100.0,
        stop_price=90.0,
        entry_limit=101.0,
        target_price=120.0,
        score=5,
        quality=1.0,
        energies=energies,
    )
