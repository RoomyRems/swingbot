from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd

from .indicators import add_indicators
from .models import EnergyEvidence, Signal

STRATEGY_VERSION = "burns-book-v2"


@dataclass(frozen=True)
class StrategyRules:
    """Frozen translations of the objective rules in Burns's 2014 book."""

    minimum_bars: int = 220
    trend_slope_bars: int = 5
    cycle_midline: float = 50.0
    cycle_extreme_level: float = 20.0
    early_retrace_level: float = 55.0
    maximum_retrace_number: int = 2
    support_atr_tolerance: float = 0.25
    objective_wave_retraces: bool = False


DEFAULT_RULES = StrategyRules()


@dataclass(frozen=True)
class _Evaluation:
    energies: dict[str, EnergyEvidence]
    context: dict[str, object]


@dataclass(frozen=True)
class SetupAssessment:
    """One strategy evaluation shared by explain, research, and signal creation."""

    signal: Signal | None
    energies: dict[str, EnergyEvidence]
    context: dict[str, object]


def prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    return add_indicators(frame)


def strategy_version(rules: StrategyRules = DEFAULT_RULES) -> str:
    return "burns-book-v3" if rules.objective_wave_retraces else STRATEGY_VERSION


def strategy_fingerprint(rules: StrategyRules = DEFAULT_RULES) -> str:
    settings = asdict(rules)
    if not rules.objective_wave_retraces:
        settings.pop("objective_wave_retraces")  # preserve the frozen v2 fingerprint
    payload = json.dumps(
        {"version": strategy_version(rules), "rules": settings},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evidence(passed: bool, value: float | None, rule: str) -> EnergyEvidence:
    clean_value = None if value is None or not np.isfinite(value) else float(value)
    return EnergyEvidence(passed=bool(passed), value=clean_value, rule=rule)


def _insufficient_evaluation() -> _Evaluation:
    return _Evaluation(
        energies={
            name: _evidence(False, None, "insufficient warm-up history")
            for name in ("trend", "momentum", "cycle", "support", "scale")
        },
        context={},
    )


def _trend_evidence(
    prepared: pd.DataFrame,
    location: int,
    rules: StrategyRules,
) -> tuple[EnergyEvidence, dict[str, object]]:
    slope = prepared["sma50"] / prepared["sma50"].shift(rules.trend_slope_bars) - 1.0
    slope_value = float(slope.iloc[location])
    row = prepared.iloc[location]
    if not np.isfinite(slope_value) or not np.isfinite(row["sma50"]):
        return _evidence(False, None, "50-SMA trend is unavailable"), {}

    rising = slope > 0
    trend_start = location
    while trend_start > 0 and bool(rising.iloc[trend_start - 1]):
        trend_start -= 1

    cycle_d = prepared["stoch_d"].iloc[trend_start : location + 1]
    below_retrace_level = cycle_d.lt(rules.early_retrace_level) & cycle_d.notna()
    retrace_starts = below_retrace_level & ~below_retrace_level.shift(
        1,
        fill_value=False,
    )
    retrace_number = int(retrace_starts.sum())
    legacy_retrace_number = retrace_number
    if rules.objective_wave_retraces:
        retrace_number = int(row["wave_retrace_number"])
    early_retrace = (
        bool(below_retrace_level.iloc[-1]) and 1 <= retrace_number <= rules.maximum_retrace_number
    )
    passed = bool(row["close"] > row["sma50"] and slope_value > 0 and early_retrace)
    context: dict[str, object] = {
        "trend_start_date": prepared.index[trend_start].date().isoformat(),
        "retrace_number": retrace_number,
    }
    if rules.objective_wave_retraces:
        context["legacy_retrace_number"] = legacy_retrace_number
        context["wave_confirmed_impulse"] = int(row["wave_confirmed_impulse"])
    return (
        _evidence(
            passed,
            slope_value,
            "close above a rising 50-SMA on the first or second retrace",
        ),
        context,
    )


def _cycle_evidence(
    prepared: pd.DataFrame,
    location: int,
    rules: StrategyRules,
) -> tuple[EnergyEvidence, dict[str, object]]:
    k = prepared["stoch_k"]
    d = prepared["stoch_d"]
    if location < 2 or not all(
        np.isfinite(value)
        for value in (
            k.iloc[location - 2],
            k.iloc[location - 1],
            k.iloc[location],
            d.iloc[location],
        )
    ):
        return _evidence(False, None, "5-2-3 stochastic cycle is unavailable"), {}

    cycle_active = bool(d.iloc[location] < rules.cycle_midline)
    k_turn_up = bool(
        k.iloc[location - 1] <= k.iloc[location - 2] and k.iloc[location] > k.iloc[location - 1]
    )
    hook = cycle_active and k_turn_up
    cycle_start = location
    while (
        cycle_start > 0
        and np.isfinite(d.iloc[cycle_start - 1])
        and d.iloc[cycle_start - 1] < rules.cycle_midline
    ):
        cycle_start -= 1

    active_k = k.iloc[cycle_start : location + 1].dropna()
    active_k_min = float(active_k.min()) if not active_k.empty else float("nan")
    reached_extreme = bool(active_k_min < rules.cycle_extreme_level)
    active_lows = prepared["low"].iloc[cycle_start : location + 1]
    active_low_date = active_lows.idxmin()
    active_low_location = int(prepared.index.get_loc(active_low_date))
    active_low = float(active_lows.min())

    second_trough = location - 1
    earlier_troughs: list[int] = []
    for candidate in range(cycle_start + 1, second_trough):
        values = (k.iloc[candidate - 1], k.iloc[candidate], k.iloc[candidate + 1])
        if not all(np.isfinite(value) for value in values):
            continue
        if values[1] <= values[0] and values[1] < values[2]:
            earlier_troughs.append(candidate)

    first_trough = earlier_troughs[-1] if earlier_troughs else None
    divergence = False
    divergence_size: float | None = None
    if hook and first_trough is not None:
        first_price_low = float(prepared["low"].iloc[first_trough])
        second_price_low = min(
            float(prepared["low"].iloc[second_trough]),
            float(prepared["low"].iloc[location]),
        )
        first_k_low = float(k.iloc[first_trough])
        second_k_low = float(k.iloc[second_trough])
        divergence = second_price_low < first_price_low and second_k_low > first_k_low
        divergence_size = second_k_low - first_k_low

    if not hook:
        divergence_reason = "no closed %K hook in the cycle-low interval"
    elif first_trough is None:
        divergence_reason = "no earlier %K trough in the active cycle-low interval"
    elif divergence:
        divergence_reason = "price made a lower low while %K made a higher low"
    else:
        divergence_reason = "the paired price and %K troughs did not diverge"

    k_delta = float(k.iloc[location] - k.iloc[location - 1])
    context: dict[str, object] = {
        "cycle_start_date": prepared.index[cycle_start].date().isoformat(),
        "cycle_low_date": active_low_date.date().isoformat(),
        "cycle_low_price": active_low,
        "cycle_active": cycle_active,
        "cycle_reached_extreme": reached_extreme,
        "cycle_extreme_level": rules.cycle_extreme_level,
        "cycle_active_k_min": active_k_min,
        "cycle_k_turn_up": k_turn_up,
        "cycle_hook": hook,
        "cycle_k_delta": k_delta,
        "mini_divergence": divergence,
        "mini_divergence_reason": divergence_reason,
        "_active_cycle_low_location": active_low_location,
        "_cycle_start_location": cycle_start,
    }
    if first_trough is not None:
        context["first_stochastic_trough_date"] = prepared.index[first_trough].date().isoformat()
    if divergence_size is not None:
        context["mini_divergence_k_delta"] = divergence_size
    return (
        _evidence(
            hook,
            k_delta,
            "5-2-3 %K hooks up during a %D-defined cycle-low interval",
        ),
        context,
    )


def _prior_cycle_levels(
    prepared: pd.DataFrame,
    cycle_start: int,
    midline: float,
) -> list[tuple[str, float]]:
    d = prepared["stoch_d"]
    levels: list[tuple[str, float]] = []

    low_end = cycle_start - 1
    while low_end >= 0 and (not np.isfinite(d.iloc[low_end]) or d.iloc[low_end] >= midline):
        low_end -= 1
    if low_end >= 0:
        low_start = low_end
        while (
            low_start > 0 and np.isfinite(d.iloc[low_start - 1]) and d.iloc[low_start - 1] < midline
        ):
            low_start -= 1
        levels.append(
            (
                "previous cycle low",
                float(prepared["low"].iloc[low_start : low_end + 1].min()),
            )
        )

    high_end = cycle_start - 1
    while high_end >= 0 and (not np.isfinite(d.iloc[high_end]) or d.iloc[high_end] <= midline):
        high_end -= 1
    if high_end >= 0:
        high_start = high_end
        while (
            high_start > 0
            and np.isfinite(d.iloc[high_start - 1])
            and d.iloc[high_start - 1] > midline
        ):
            high_start -= 1
        levels.append(
            (
                "previous cycle high",
                float(prepared["high"].iloc[high_start : high_end + 1].max()),
            )
        )
    return levels


def _support_evidence(
    prepared: pd.DataFrame,
    location: int,
    cycle_context: dict[str, object],
    rules: StrategyRules,
) -> tuple[EnergyEvidence, dict[str, object]]:
    active_low_location = cycle_context.get("_active_cycle_low_location")
    cycle_start = cycle_context.get("_cycle_start_location")
    if not isinstance(active_low_location, int) or not isinstance(cycle_start, int):
        return _evidence(False, None, "cycle-low support cannot be evaluated"), {}

    active_row = prepared.iloc[active_low_location]
    active_low = float(cycle_context["cycle_low_price"])
    prior_cycle_levels = _prior_cycle_levels(prepared, cycle_start, rules.cycle_midline)
    context: dict[str, object] = {
        f"{prior_label.replace(' ', '_')}_price": prior_level
        for prior_label, prior_level in prior_cycle_levels
    }
    atr_value = float(active_row["atr14"])
    if not np.isfinite(atr_value) or atr_value <= 0:
        return _evidence(False, None, "ATR support zone is unavailable"), context

    candidates = [
        ("15-EMA", float(active_row["ema15"])),
        ("50-SMA", float(active_row["sma50"])),
        *prior_cycle_levels,
    ]
    current_close = float(prepared["close"].iloc[location])
    distances = [
        (label, level, abs(active_low - level) / atr_value)
        for label, level in candidates
        if np.isfinite(level) and current_close >= level
    ]
    if not distances:
        return _evidence(False, None, "no causal support level is available"), context

    label, level, distance = min(distances, key=lambda item: item[2])
    passed = distance <= rules.support_atr_tolerance
    context.update(
        {
            "support_source": label,
            "support_level": level,
            "support_distance_atr": distance,
        }
    )
    return (
        _evidence(
            passed,
            distance,
            f"active cycle low tests {label} support within 0.25 ATR",
        ),
        context,
    )


def _evaluate_setup(
    prepared: pd.DataFrame,
    timestamp: pd.Timestamp,
    rules: StrategyRules,
) -> _Evaluation:
    location = int(prepared.index.get_loc(timestamp))
    if location < rules.minimum_bars or location < rules.trend_slope_bars + 1:
        return _insufficient_evaluation()

    trend, trend_context = _trend_evidence(prepared, location, rules)
    cycle, cycle_context = _cycle_evidence(prepared, location, rules)
    active_low_location = cycle_context.get("_active_cycle_low_location", location)
    if not isinstance(active_low_location, int):
        active_low_location = location

    momentum_value = float(prepared["macd"].iloc[active_low_location])
    momentum = _evidence(
        momentum_value > 0,
        momentum_value,
        "daily MACD line remains above zero at the cycle low",
    )
    support, support_context = _support_evidence(
        prepared,
        location,
        cycle_context,
        rules,
    )
    scale_value = float(prepared["weekly_macd_delta"].iloc[location])
    scale = _evidence(
        scale_value > 0,
        scale_value,
        "last completed weekly MACD line is angling up",
    )
    volume_context: dict[str, object] = {}
    for label, volume_location in (
        ("cycle_low", active_low_location),
        ("signal", location),
    ):
        history = prepared["volume"].iloc[max(0, volume_location - 90) : volume_location]
        average = float(history.mean()) if not history.empty else float("nan")
        current_volume = float(prepared["volume"].iloc[volume_location])
        if np.isfinite(average) and average > 0:
            volume_context[f"{label}_relative_volume_90"] = current_volume / average
    context = {**trend_context, **cycle_context, **support_context, **volume_context}
    return _Evaluation(
        energies={
            "trend": trend,
            "momentum": momentum,
            "cycle": cycle,
            "support": support,
            "scale": scale,
        },
        context=context,
    )


def evaluate_energies(
    prepared: pd.DataFrame,
    as_of: date | str | pd.Timestamp,
    rules: StrategyRules = DEFAULT_RULES,
) -> dict[str, EnergyEvidence]:
    timestamp = pd.Timestamp(as_of).normalize()
    if timestamp not in prepared.index:
        raise ValueError(f"no bar exists on {timestamp.date()}")
    return _evaluate_setup(prepared, timestamp, rules).energies


def _tick_size(price: float) -> float:
    return 0.01 if price >= 1.0 else 0.0001


def _round_down(price: float, tick: float) -> float:
    return round(math.floor((price + 1e-12) / tick) * tick, 4)


def _round_up(price: float, tick: float) -> float:
    return round(math.ceil((price - 1e-12) / tick) * tick, 4)


def _signal_from_evaluation(
    symbol: str,
    prepared: pd.DataFrame,
    timestamp: pd.Timestamp,
    evaluation: _Evaluation,
    *,
    max_entry_gap_r: float,
    reward_r: float,
    rules: StrategyRules = DEFAULT_RULES,
) -> Signal | None:
    energies = evaluation.energies
    score = sum(evidence.passed for evidence in energies.values())

    # Scale is Burns's explicit veto. Trend and Cycle are operationally required
    # here because this long-only implementation needs both direction and a trigger.
    required = ("trend", "cycle", "scale")
    if score < 4 or not all(energies[name].passed for name in required):
        return None

    location = int(prepared.index.get_loc(timestamp))
    row = prepared.iloc[location]
    reference = float(row["close"])
    tick = _tick_size(reference)
    cycle_low = evaluation.context.get("cycle_low_price")
    if not isinstance(cycle_low, (int, float)) or not np.isfinite(cycle_low):
        return None
    stop = _round_down(float(cycle_low) - tick, tick)
    entry_stop = _round_up(float(row["high"]) + tick, tick)
    if stop <= 0 or entry_stop <= stop:
        return None

    trigger_risk = entry_stop - stop
    entry_limit = _round_up(entry_stop + max_entry_gap_r * trigger_risk, tick)
    risk_per_share = entry_limit - stop
    target = _round_up(entry_limit + reward_r * risk_per_share, tick)
    average_daily_volume = float(row["adv90"])
    if not np.isfinite(average_daily_volume) or average_daily_volume <= 0:
        return None

    retrace_number = evaluation.context.get(
        "retrace_number",
        rules.maximum_retrace_number,
    )
    retrace_quality = (rules.maximum_retrace_number + 1 - int(retrace_number)) * 10_000
    divergence_quality = int(bool(evaluation.context.get("mini_divergence"))) * 1_000
    # Keep the ordering lexicographic: score, early retrace, mini-divergence,
    # then bounded trend/scale tie-breakers. Burns describes divergence as a
    # higher-probability Cycle pattern, not as a prerequisite for Cycle itself.
    trend_quality = min(max(0.0, energies["trend"].value or 0.0) * 10_000, 499.0)
    scale_quality = min(
        max(0.0, energies["scale"].value or 0.0) / reference * 10_000,
        499.0,
    )
    quality = (
        score * 1_000_000 + retrace_quality + divergence_quality + trend_quality + scale_quality
    )
    public_context = {
        key: value for key, value in evaluation.context.items() if not key.startswith("_")
    }

    return Signal(
        symbol=symbol.upper(),
        signal_date=timestamp.date(),
        reference_price=reference,
        stop_price=stop,
        entry_stop=entry_stop,
        entry_limit=entry_limit,
        target_price=target,
        average_daily_volume=average_daily_volume,
        score=score,
        quality=float(quality),
        energies=energies,
        context=public_context,
    )


def assess_setup(
    symbol: str,
    prepared: pd.DataFrame,
    as_of: date | str | pd.Timestamp,
    *,
    max_entry_gap_r: float,
    reward_r: float,
    rules: StrategyRules = DEFAULT_RULES,
) -> SetupAssessment:
    """Evaluate once and expose causal evidence even when no signal qualifies."""
    timestamp = pd.Timestamp(as_of).normalize()
    if timestamp not in prepared.index:
        raise ValueError(f"no bar exists on {timestamp.date()}")
    evaluation = _evaluate_setup(prepared, timestamp, rules)
    signal = _signal_from_evaluation(
        symbol,
        prepared,
        timestamp,
        evaluation,
        max_entry_gap_r=max_entry_gap_r,
        reward_r=reward_r,
        rules=rules,
    )
    public_context = {
        key: value for key, value in evaluation.context.items() if not key.startswith("_")
    }
    return SetupAssessment(
        signal=signal,
        energies=evaluation.energies,
        context=public_context,
    )


def assess_signal(
    symbol: str,
    prepared: pd.DataFrame,
    as_of: date | str | pd.Timestamp,
    *,
    max_entry_gap_r: float,
    reward_r: float,
    rules: StrategyRules = DEFAULT_RULES,
) -> tuple[Signal | None, dict[str, EnergyEvidence]]:
    """Backward-compatible pair of signal and evidence from one assessment."""
    assessment = assess_setup(
        symbol,
        prepared,
        as_of,
        max_entry_gap_r=max_entry_gap_r,
        reward_r=reward_r,
        rules=rules,
    )
    return assessment.signal, assessment.energies


def generate_signal(
    symbol: str,
    prepared: pd.DataFrame,
    as_of: date | str | pd.Timestamp,
    *,
    max_entry_gap_r: float,
    reward_r: float,
    rules: StrategyRules = DEFAULT_RULES,
) -> Signal | None:
    """Generate the one causal signal used by backtesting and paper trading."""
    assessment = assess_setup(
        symbol,
        prepared,
        as_of,
        max_entry_gap_r=max_entry_gap_r,
        reward_r=reward_r,
        rules=rules,
    )
    return assessment.signal
