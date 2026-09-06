"""Causal cycle-to-wave translation; no retrospectively relabelled bars."""

from __future__ import annotations

import math

import pandas as pd


def add_wave_context(prepared: pd.DataFrame, slope_bars: int = 5) -> pd.DataFrame:
    result = prepared.copy()
    rising = result["sma50"].diff(slope_bars).gt(0).to_numpy()
    values = result[["open", "high", "close", "stoch_d"]].to_numpy()
    confirmed = 0
    reference = float("nan")
    interval_high: float | None = None
    breakout = False
    rows: list[tuple[int, int, int, float]] = []
    for is_rising, (open_price, high, close, d) in zip(rising, values, strict=True):
        if not is_rising or not math.isfinite(d):
            confirmed, reference, interval_high, breakout = 0, float("nan"), None, False
        elif d > 50:
            interval_high = high if interval_high is None else max(interval_high, high)
            breakout = breakout or bool(
                confirmed > 0 and open_price >= reference and close >= reference
            )
        elif interval_high is not None:
            if confirmed == 0 or breakout:
                confirmed = 1 if confirmed == 0 else confirmed + 2
                reference = interval_high
            interval_high, breakout = None, False
        active = confirmed
        if interval_high is not None:
            active = 1 if confirmed == 0 else confirmed + 2 if breakout else confirmed
        rows.append((confirmed, active, (confirmed + 1) // 2, reference))
    result[
        [
            "wave_confirmed_impulse",
            "wave_active_impulse",
            "wave_retrace_number",
            "wave_reference_high",
        ]
    ] = rows
    return result
