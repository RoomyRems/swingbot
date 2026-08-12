from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd


class ExitPolicy(StrEnum):
    """Backtest-only exit policies; paper submission remains a static bracket."""

    STATIC_2R = "static-2r"
    BURNS_CYCLE_V1 = "burns-cycle-v1"


@dataclass(frozen=True)
class BurnsExitRules:
    """Frozen, causal translations of Burns's Chapter 23 trade management."""

    cycle_midline: float = 50.0
    partial_fraction: float = 0.50


DEFAULT_BURNS_EXIT_RULES = BurnsExitRules()


@dataclass(frozen=True)
class CycleTurn:
    signal_date: date
    interval_start_date: date
    extreme_date: date
    extreme_price: float
    hook_bar_low: float
    wave_breakout: bool


def exit_policy_fingerprint(
    policy: ExitPolicy,
    rules: BurnsExitRules = DEFAULT_BURNS_EXIT_RULES,
) -> str:
    payload: dict[str, object] = {"policy": policy.value}
    if policy is ExitPolicy.BURNS_CYCLE_V1:
        payload["rules"] = asdict(rules)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tick_size(price: float) -> float:
    return 0.01 if price >= 1.0 else 0.0001


def stop_below(price: float) -> float:
    tick = tick_size(price)
    return round(math.floor((price - tick + 1e-12) / tick) * tick, 4)


def _finite_turn_values(prepared: pd.DataFrame, location: int) -> bool:
    if location < 2:
        return False
    values = (
        prepared["stoch_k"].iloc[location - 2],
        prepared["stoch_k"].iloc[location - 1],
        prepared["stoch_k"].iloc[location],
        prepared["stoch_d"].iloc[location],
    )
    return all(np.isfinite(value) for value in values)


def long_cycle_high_turn(
    prepared: pd.DataFrame,
    timestamp: pd.Timestamp,
    *,
    previous_wave_high: float | None = None,
    rules: BurnsExitRules = DEFAULT_BURNS_EXIT_RULES,
) -> CycleTurn | None:
    """Return a closed-bar long exit hook without assigning the unknowable peak fill."""

    location = int(prepared.index.get_loc(timestamp))
    if not _finite_turn_values(prepared, location):
        return None
    k = prepared["stoch_k"]
    d = prepared["stoch_d"]
    hook = bool(
        d.iloc[location] > rules.cycle_midline
        and k.iloc[location - 1] >= k.iloc[location - 2]
        and k.iloc[location] < k.iloc[location - 1]
    )
    if not hook:
        return None

    start = location
    while start > 0 and np.isfinite(d.iloc[start - 1]) and d.iloc[start - 1] > rules.cycle_midline:
        start -= 1
    interval = prepared.iloc[start : location + 1]
    extreme_date = interval["high"].idxmax()
    wave_breakout = False
    if previous_wave_high is not None and np.isfinite(previous_wave_high):
        wave_breakout = bool(
            (
                (interval["open"] >= previous_wave_high) & (interval["close"] >= previous_wave_high)
            ).any()
        )
    return CycleTurn(
        signal_date=timestamp.date(),
        interval_start_date=prepared.index[start].date(),
        extreme_date=extreme_date.date(),
        extreme_price=float(interval["high"].max()),
        hook_bar_low=float(prepared["low"].iloc[location]),
        wave_breakout=wave_breakout,
    )


def long_cycle_low_turn(
    prepared: pd.DataFrame,
    timestamp: pd.Timestamp,
    *,
    rules: BurnsExitRules = DEFAULT_BURNS_EXIT_RULES,
) -> CycleTurn | None:
    """Return the closed stochastic hook and active price low for a long runner."""

    location = int(prepared.index.get_loc(timestamp))
    if not _finite_turn_values(prepared, location):
        return None
    k = prepared["stoch_k"]
    d = prepared["stoch_d"]
    hook = bool(
        d.iloc[location] < rules.cycle_midline
        and k.iloc[location - 1] <= k.iloc[location - 2]
        and k.iloc[location] > k.iloc[location - 1]
    )
    if not hook:
        return None

    start = location
    while start > 0 and np.isfinite(d.iloc[start - 1]) and d.iloc[start - 1] < rules.cycle_midline:
        start -= 1
    interval = prepared.iloc[start : location + 1]
    extreme_date = interval["low"].idxmin()
    return CycleTurn(
        signal_date=timestamp.date(),
        interval_start_date=prepared.index[start].date(),
        extreme_date=extreme_date.date(),
        extreme_price=float(interval["low"].min()),
        hook_bar_low=float(prepared["low"].iloc[location]),
        wave_breakout=False,
    )
