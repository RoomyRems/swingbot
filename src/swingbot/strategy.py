from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .indicators import add_indicators
from .models import EnergyEvidence, Signal


@dataclass(frozen=True)
class StrategyRules:
    """Versioned research rules; intentionally not exposed as dozens of config knobs."""

    minimum_bars: int = 220
    trend_slope_bars: int = 5
    cycle_oversold: float = 20.0
    cycle_midline: float = 50.0
    pullback_atr_tolerance: float = 0.25


DEFAULT_RULES = StrategyRules()


def prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    return add_indicators(frame)


def _evidence(passed: bool, value: float | None, rule: str) -> EnergyEvidence:
    clean_value = None if value is None or not np.isfinite(value) else float(value)
    return EnergyEvidence(passed=bool(passed), value=clean_value, rule=rule)


def evaluate_energies(
    prepared: pd.DataFrame,
    as_of: date | str | pd.Timestamp,
    rules: StrategyRules = DEFAULT_RULES,
) -> dict[str, EnergyEvidence]:
    timestamp = pd.Timestamp(as_of).normalize()
    if timestamp not in prepared.index:
        raise ValueError(f"no bar exists on {timestamp.date()}")
    location = int(prepared.index.get_loc(timestamp))
    if location < rules.minimum_bars or location < rules.trend_slope_bars + 1:
        return {
            name: _evidence(False, None, "insufficient warm-up history")
            for name in ("trend", "momentum", "cycle", "support", "scale")
        }

    row = prepared.iloc[location]
    previous = prepared.iloc[location - 1]
    slope_base = prepared.iloc[location - rules.trend_slope_bars]["sma50"]

    trend_slope = (row["sma50"] - slope_base) / slope_base
    trend_ok = row["close"] > row["sma50"] and trend_slope > 0

    momentum_ok = row["macd"] > 0

    cycle_ok = (
        previous["stoch_k"] <= rules.cycle_oversold
        and previous["stoch_d"] <= rules.cycle_midline
        and row["stoch_k"] > previous["stoch_k"]
        and row["stoch_d"] > previous["stoch_d"]
    )

    pullback_distance_atr = (row["low"] - row["ema15"]) / row["atr14"]
    support_ok = (
        row["low"] <= row["ema15"] + rules.pullback_atr_tolerance * row["atr14"]
        and row["close"] >= row["ema15"]
    )

    scale_ok = row["weekly_macd_hist_delta"] > 0

    return {
        "trend": _evidence(
            trend_ok,
            trend_slope,
            "close above a 50-SMA whose five-bar slope is positive",
        ),
        "momentum": _evidence(momentum_ok, row["macd"], "daily MACD line above zero"),
        "cycle": _evidence(
            cycle_ok,
            row["stoch_d"],
            "5-2-3 slow stochastic hooks up after K<=20 and D<=50",
        ),
        "support": _evidence(
            support_ok,
            pullback_distance_atr,
            "bar tests the 15-EMA area and closes at or above it",
        ),
        "scale": _evidence(
            scale_ok,
            row["weekly_macd_hist_delta"],
            "last completed weekly MACD histogram is rising",
        ),
    }


def _tick_size(price: float) -> float:
    return 0.01 if price >= 1.0 else 0.0001


def _round_down(price: float, tick: float) -> float:
    return round(math.floor((price + 1e-12) / tick) * tick, 4)


def _round_up(price: float, tick: float) -> float:
    return round(math.ceil((price - 1e-12) / tick) * tick, 4)


def generate_signal(
    symbol: str,
    prepared: pd.DataFrame,
    as_of: date | str | pd.Timestamp,
    *,
    max_entry_gap_r: float,
    reward_r: float,
    rules: StrategyRules = DEFAULT_RULES,
) -> Signal | None:
    """Generate the one signal used by both backtesting and paper trading."""
    timestamp = pd.Timestamp(as_of).normalize()
    energies = evaluate_energies(prepared, timestamp, rules)
    score = sum(evidence.passed for evidence in energies.values())
    if score != 5:
        return None

    location = int(prepared.index.get_loc(timestamp))
    row = prepared.iloc[location]
    previous = prepared.iloc[location - 1]
    reference = float(row["close"])
    tick = _tick_size(reference)
    stop = _round_down(min(float(row["low"]), float(previous["low"])) - tick, tick)
    if stop <= 0:
        return None
    reference_risk = reference - stop
    if not np.isfinite(reference_risk) or reference_risk <= tick:
        return None

    entry_limit = _round_up(reference + max_entry_gap_r * reference_risk, tick)
    risk_per_share = entry_limit - stop
    target = _round_up(entry_limit + reward_r * risk_per_share, tick)
    trend_quality = max(0.0, energies["trend"].value or 0.0) * 10_000
    scale_quality = max(0.0, energies["scale"].value or 0.0) / reference * 10_000
    quality = trend_quality + scale_quality

    return Signal(
        symbol=symbol.upper(),
        signal_date=timestamp.date(),
        reference_price=reference,
        stop_price=stop,
        entry_limit=entry_limit,
        target_price=target,
        score=score,
        quality=float(quality),
        energies=energies,
    )
