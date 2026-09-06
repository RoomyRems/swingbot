from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AppConfig, config_fingerprint
from .exits import (
    DEFAULT_BURNS_EXIT_RULES,
    ExitPolicy,
    exit_policy_fingerprint,
    fill_risk_target,
    long_cycle_high_turn,
    long_cycle_low_turn,
    stop_below,
)
from .indicators import normalize_bars
from .models import (
    BacktestResult,
    EnergyEvidence,
    ExitFill,
    ExitReason,
    PlannedOrder,
    Position,
    Signal,
    Trade,
)
from .portfolio import Capacity, size_order
from .strategy import (
    DEFAULT_RULES,
    StrategyRules,
    assess_setup,
    prepare_indicators,
    strategy_fingerprint,
    strategy_version,
)
from .waves import add_wave_context

ENGINE_VERSION = "daily-execution-v3"

_ENERGY_NAMES = ("trend", "momentum", "cycle", "support", "scale")
_REQUIRED_ENERGIES = ("trend", "cycle", "scale")
_CYCLE_FUNNEL_NAMES = (
    "active_cycle_low_interval",
    "reached_cycle_extreme",
    "k_turn_up_anywhere",
    "closed_cycle_hook",
    "hook_after_cycle_extreme",
    "hook_with_mini_divergence",
    "hook_without_mini_divergence",
)


@dataclass
class _StrategyDiagnostics:
    symbols: tuple[str, ...]
    evaluated_symbol_sessions: int = 0
    energy_pass_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in _ENERGY_NAMES}
    )
    score_counts: dict[int, int] = field(
        default_factory=lambda: {score: 0 for score in range(len(_ENERGY_NAMES) + 1)}
    )
    pattern_counts: dict[str, int] = field(default_factory=dict)
    score_at_least_four: int = 0
    required_energies_pass: int = 0
    eligible_setups: int = 0
    eligible_with_mini_divergence: int = 0
    eligible_without_mini_divergence: int = 0
    burns_book_v1_strict_eligible_setups: int = 0
    cycle_funnel_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in _CYCLE_FUNNEL_NAMES}
    )
    eligible_by_symbol: dict[str, int] = field(init=False)
    eligible_by_year: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.eligible_by_symbol = {symbol: 0 for symbol in self.symbols}

    def observe(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        energies: dict[str, EnergyEvidence],
        context: Mapping[str, object] | None = None,
    ) -> None:
        passed = {name: bool(energies[name].passed) for name in _ENERGY_NAMES}
        context = context or {}
        cycle_active = bool(context.get("cycle_active"))
        reached_cycle_extreme = bool(context.get("cycle_reached_extreme"))
        k_turn_up = bool(context.get("cycle_k_turn_up"))
        cycle_hook = bool(context.get("cycle_hook"))
        mini_divergence = bool(context.get("mini_divergence"))

        score = sum(passed.values())
        self.evaluated_symbol_sessions += 1
        self.score_counts[score] += 1
        for name, value in passed.items():
            self.energy_pass_counts[name] += int(value)

        pattern = ",".join(name for name in _ENERGY_NAMES if passed[name]) or "none"
        self.pattern_counts[pattern] = self.pattern_counts.get(pattern, 0) + 1
        self.cycle_funnel_counts["active_cycle_low_interval"] += int(cycle_active)
        self.cycle_funnel_counts["reached_cycle_extreme"] += int(reached_cycle_extreme)
        self.cycle_funnel_counts["k_turn_up_anywhere"] += int(k_turn_up)
        self.cycle_funnel_counts["closed_cycle_hook"] += int(cycle_hook)
        self.cycle_funnel_counts["hook_after_cycle_extreme"] += int(
            cycle_hook and reached_cycle_extreme
        )
        self.cycle_funnel_counts["hook_with_mini_divergence"] += int(cycle_hook and mini_divergence)
        self.cycle_funnel_counts["hook_without_mini_divergence"] += int(
            cycle_hook and not mini_divergence
        )

        has_score = score >= 4
        has_required = all(passed[name] for name in _REQUIRED_ENERGIES)
        self.score_at_least_four += int(has_score)
        self.required_energies_pass += int(has_required)
        if has_score and has_required:
            self.eligible_setups += 1
            self.eligible_with_mini_divergence += int(mini_divergence)
            self.eligible_without_mini_divergence += int(not mini_divergence)
            self.eligible_by_symbol[symbol] += 1
            year = int(timestamp.year)
            self.eligible_by_year[year] = self.eligible_by_year.get(year, 0) + 1

        # Reconstruct the v1 Cycle veto from the same evaluation. This isolates
        # the effect of the corrected classification without rerunning indicators.
        v1_score = score - int(passed["cycle"]) + int(mini_divergence)
        v1_has_required = passed["trend"] and mini_divergence and passed["scale"]
        self.burns_book_v1_strict_eligible_setups += int(v1_score >= 4 and v1_has_required)

    def as_dict(self) -> dict[str, object]:
        evaluated = self.evaluated_symbol_sessions
        pass_rates = {
            name: count / evaluated if evaluated else 0.0
            for name, count in self.energy_pass_counts.items()
        }
        cycle_funnel_rates = {
            name: count / evaluated if evaluated else 0.0
            for name, count in self.cycle_funnel_counts.items()
        }
        return {
            "evaluated_symbol_sessions": evaluated,
            "energy_pass_counts": self.energy_pass_counts,
            "energy_pass_rates": pass_rates,
            "score_counts": {str(score): count for score, count in self.score_counts.items()},
            "score_at_least_four": self.score_at_least_four,
            "required_trend_cycle_scale": self.required_energies_pass,
            "eligible_setups": self.eligible_setups,
            "eligible_with_mini_divergence": self.eligible_with_mini_divergence,
            "eligible_without_mini_divergence": self.eligible_without_mini_divergence,
            "burns_book_v1_strict_eligible_setups": (self.burns_book_v1_strict_eligible_setups),
            "cycle_funnel_counts": self.cycle_funnel_counts,
            "cycle_funnel_rates": cycle_funnel_rates,
            "eligible_by_symbol": self.eligible_by_symbol,
            "eligible_by_year": {
                str(year): count for year, count in sorted(self.eligible_by_year.items())
            },
            "pass_patterns": dict(
                sorted(self.pattern_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
        }


def _slippage_fraction(config: AppConfig) -> float:
    return config.execution.slippage_bps / 10_000.0


def _mark_price(frame: pd.DataFrame, timestamp: pd.Timestamp) -> float:
    history = frame.loc[:timestamp, "close"]
    if history.empty:
        raise ValueError(f"no mark available on or before {timestamp.date()}")
    return float(history.iloc[-1])


def _equity(
    cash: float,
    positions: dict[str, Position],
    frames: dict[str, pd.DataFrame],
    timestamp: pd.Timestamp,
) -> float:
    return cash + sum(
        position.quantity * _mark_price(frames[symbol], timestamp)
        for symbol, position in positions.items()
    )


def _committed_risk(
    positions: Iterable[Position],
    pending: Iterable[PlannedOrder],
) -> float:
    open_risk = sum(
        position.quantity * (position.reserved_risk_per_share or position.initial_risk_per_share)
        + 2.0 * position.entry_commission
        for position in positions
    )
    return open_risk + sum(order.reserved_risk for order in pending)


def _existing_exit(
    position: Position,
    bar: pd.Series,
    slip: float,
    exit_policy: ExitPolicy,
) -> tuple[float, ExitReason] | None:
    if float(bar["open"]) <= position.stop_price:
        return (
            max(float(bar["low"]), float(bar["open"]) * (1.0 - slip)),
            position.stop_reason,
        )
    if exit_policy.is_static and float(bar["open"]) >= position.target_price:
        return position.target_price, ExitReason.TARGET

    stop_touched = float(bar["low"]) <= position.stop_price
    target_touched = exit_policy.is_static and float(bar["high"]) >= position.target_price
    if stop_touched:
        # Daily OHLC cannot reveal whether a same-day stop or target came first.
        # The conservative convention always assigns the stop.
        return (
            max(float(bar["low"]), position.stop_price * (1.0 - slip)),
            position.stop_reason,
        )
    if target_touched:
        return position.target_price, ExitReason.TARGET
    return None


def _entry_fill(
    order: PlannedOrder,
    bar: pd.Series,
    slip: float,
) -> tuple[float, bool] | None:
    entry_stop = order.signal.entry_stop
    entry_limit = order.signal.entry_limit
    open_price = float(bar["open"])
    high_price = float(bar["high"])
    low_price = float(bar["low"])
    if high_price < entry_stop:
        return None
    if open_price >= entry_stop:
        if open_price <= entry_limit:
            return min(entry_limit, open_price * (1.0 + slip)), True
        if low_price <= entry_limit:
            # The stop triggers at the gap open and its limit can fill later.
            return entry_limit, False
        return None
    # The stop triggers intraday and becomes a limit order capped at entry_limit.
    return min(entry_limit, entry_stop * (1.0 + slip)), False


def _new_position_exit(
    position: Position,
    bar: pd.Series,
    filled_at_open: bool,
    slip: float,
    exit_policy: ExitPolicy,
    *,
    entry_path: str = "gap_retrace",
) -> tuple[float, ExitReason] | None:
    if filled_at_open and float(bar["open"]) <= position.stop_price:
        return max(float(bar["low"]), float(bar["open"]) * (1.0 - slip)), position.stop_reason
    if float(bar["low"]) <= position.stop_price:
        return (
            max(float(bar["low"]), position.stop_price * (1.0 - slip)),
            position.stop_reason,
        )
    if (
        exit_policy.is_static
        and (
            filled_at_open
            or entry_path == "breakout"
            or float(bar["close"]) >= position.target_price
        )
        and float(bar["high"]) >= position.target_price
    ):
        return position.target_price, ExitReason.TARGET
    # A gap-retrace high can precede the fill. A close at/above target,
    # unlike the high alone, establishes a post-fill target touch.
    return None


def _record_exit(
    position: Position,
    exit_date: date,
    exit_price: float,
    reason: ExitReason,
    commission_per_share: float,
) -> float:
    quantity = position.quantity
    return _record_partial_exit(
        position,
        exit_date,
        exit_price,
        quantity,
        reason,
        commission_per_share,
    )


def _record_partial_exit(
    position: Position,
    exit_date: date,
    exit_price: float,
    quantity: int,
    reason: ExitReason,
    commission_per_share: float,
) -> float:
    if quantity < 1 or quantity > position.quantity:
        raise ValueError("exit quantity must be within the remaining position")
    exit_fee = quantity * commission_per_share
    proceeds = quantity * exit_price - exit_fee
    position.exit_fills.append(
        ExitFill(
            position_id=position.position_id,
            symbol=position.symbol,
            exit_date=exit_date,
            quantity=quantity,
            price=exit_price,
            reason=reason,
            fees=exit_fee,
        )
    )
    position.quantity -= quantity
    return proceeds


def _finalize_trade(position: Position) -> Trade:
    if position.quantity != 0:
        raise ValueError("cannot finalize a position with shares remaining")
    exited_quantity = sum(fill.quantity for fill in position.exit_fills)
    if exited_quantity != position.initial_quantity:
        raise ValueError("exit fills do not reconcile to the initial position")
    exit_value = sum(fill.quantity * fill.price for fill in position.exit_fills)
    exit_fees = sum(fill.fees for fill in position.exit_fills)
    weighted_exit = exit_value / position.initial_quantity
    fees = position.entry_commission + exit_fees
    pnl = exit_value - position.initial_quantity * position.entry_price - fees
    initial_risk = position.initial_quantity * position.initial_risk_per_share
    r_multiple = pnl / initial_risk if initial_risk > 0 else float("nan")
    partial_dates = [
        fill.exit_date
        for fill in position.exit_fills
        if fill.reason is ExitReason.CYCLE_HIGH_PARTIAL
    ]
    return Trade(
        position_id=position.position_id,
        symbol=position.symbol,
        signal_date=position.signal_date,
        entry_date=position.entry_date,
        exit_date=position.exit_fills[-1].exit_date,
        quantity=position.initial_quantity,
        entry_price=position.entry_price,
        exit_price=weighted_exit,
        stop_price=position.initial_stop_price,
        target_price=position.target_price,
        reason=position.exit_fills[-1].reason,
        pnl=pnl,
        r_multiple=r_multiple,
        fees=fees,
        exit_legs=len(position.exit_fills),
        first_exit_date=min(partial_dates) if partial_dates else None,
        initial_stop_price=position.initial_stop_price,
        final_stop_price=position.stop_price,
        reserved_r_multiple=pnl
        / (
            position.initial_quantity
            * (position.reserved_risk_per_share or position.initial_risk_per_share)
        ),
        actual_risk_per_share=position.initial_risk_per_share,
        reserved_risk_per_share=(
            position.reserved_risk_per_share or position.initial_risk_per_share
        ),
    )


def _exit_row(fill: ExitFill) -> dict[str, object]:
    return {
        "position_id": fill.position_id,
        "symbol": fill.symbol,
        "exit_date": fill.exit_date.isoformat(),
        "quantity": fill.quantity,
        "price": fill.price,
        "reason": fill.reason.value,
        "fees": fill.fees,
    }


def _signal_row(signal: Signal, status: str, reason: str = "") -> dict[str, object]:
    row: dict[str, object] = {
        "symbol": signal.symbol,
        "signal_date": signal.signal_date.isoformat(),
        "status": status,
        "reason": reason,
        "score": signal.score,
        "quality": signal.quality,
        "reference_price": signal.reference_price,
        "entry_stop": signal.entry_stop,
        "entry_limit": signal.entry_limit,
        "stop_price": signal.stop_price,
        "target_price": signal.target_price,
        "average_daily_volume": signal.average_daily_volume,
    }
    for name, evidence in signal.energies.items():
        row[f"{name}_passed"] = evidence.passed
        row[f"{name}_value"] = evidence.value
        row[f"{name}_rule"] = evidence.rule
    for name, value in signal.context.items():
        row[f"context_{name}"] = value
    return row


def _execution_diagnostics(
    signal_rows: list[dict[str, object]],
    order_rows: list[dict[str, object]],
) -> dict[str, object]:
    signal_status_counts: dict[str, int] = {}
    for row in signal_rows:
        status = str(row.get("status", "unknown"))
        signal_status_counts[status] = signal_status_counts.get(status, 0) + 1
    order_status_counts: dict[str, int] = {}
    for row in order_rows:
        status = str(row.get("status", "unknown"))
        order_status_counts[status] = order_status_counts.get(status, 0) + 1
    planned_orders = sum(row.get("status") == "planned" for row in signal_rows)
    filled_orders = sum(row.get("status") == "filled" for row in order_rows)
    unfilled_orders = sum(row.get("status") == "not_filled" for row in order_rows)
    return {
        "generated_signals": len(signal_rows),
        "signal_status_counts": signal_status_counts,
        "order_status_counts": order_status_counts,
        "planned_orders": planned_orders,
        "filled_orders": filled_orders,
        "unfilled_orders": unfilled_orders,
        "fill_rate": filled_orders / planned_orders if planned_orders else None,
        "entry_stop_sequence_ambiguous": sum(
            bool(row.get("entry_stop_sequence_ambiguous")) for row in order_rows
        ),
    }


def _exit_diagnostics(
    trades: list[Trade],
    exit_rows: list[dict[str, object]],
    management_rows: list[dict[str, object]],
) -> dict[str, object]:
    fill_reasons: dict[str, int] = {}
    for row in exit_rows:
        reason = str(row["reason"])
        fill_reasons[reason] = fill_reasons.get(reason, 0) + 1
    final_reasons: dict[str, int] = {}
    for trade in trades:
        reason = trade.reason.value
        final_reasons[reason] = final_reasons.get(reason, 0) + 1
    management_events: dict[str, int] = {}
    for row in management_rows:
        event = str(row["event"])
        management_events[event] = management_events.get(event, 0) + 1
    partial_positions = sum(trade.exit_legs > 1 for trade in trades)
    return {
        "exit_fill_reason_counts": fill_reasons,
        "final_exit_reason_counts": final_reasons,
        "management_event_counts": management_events,
        "partial_exit_positions": partial_positions,
        "partial_exit_rate": partial_positions / len(trades) if trades else None,
        "average_exit_legs": (
            float(np.mean([trade.exit_legs for trade in trades])) if trades else None
        ),
    }


def _partial_fill_price(bar: pd.Series, slip: float) -> float:
    return max(float(bar["low"]), float(bar["open"]) * (1.0 - slip))


def _process_scheduled_partial(
    position: Position,
    bar: pd.Series,
    current_date: date,
    slip: float,
    commission: float,
) -> float | None:
    if position.partial_exit_on != current_date or position.first_exit_taken:
        return None
    quantity = max(
        1,
        math.floor(position.initial_quantity * DEFAULT_BURNS_EXIT_RULES.partial_fraction),
    )
    quantity = min(quantity, position.quantity)
    proceeds = _record_partial_exit(
        position,
        current_date,
        _partial_fill_price(bar, slip),
        quantity,
        ExitReason.CYCLE_HIGH_PARTIAL,
        commission,
    )
    position.first_exit_taken = True
    position.partial_exit_on = None
    position.one_bar_mode = position.partial_is_fifth_wave
    if position.one_bar_mode and position.partial_hook_bar_low is not None:
        candidate = stop_below(position.partial_hook_bar_low)
        if candidate > position.stop_price:
            position.stop_price = candidate
            position.stop_reason = ExitReason.ONE_BAR_TRAIL
    return proceeds


def _observe_burns_management(
    position: Position,
    prepared: pd.DataFrame,
    timestamp: pd.Timestamp,
    next_timestamp: pd.Timestamp | None,
    exit_policy: ExitPolicy = ExitPolicy.BURNS_CYCLE_V1,
) -> dict[str, object] | None:
    if not position.first_exit_taken and position.partial_exit_on is None:
        previous_wave_high = position.entry_context.get("previous_cycle_high_price")
        high_turn = long_cycle_high_turn(
            prepared,
            timestamp,
            previous_wave_high=(
                float(previous_wave_high) if isinstance(previous_wave_high, (int, float)) else None
            ),
        )
        if high_turn is not None and next_timestamp is not None:
            position.partial_exit_on = next_timestamp.date()
            position.partial_signal_date = high_turn.signal_date
            position.partial_hook_bar_low = high_turn.hook_bar_low
            position.partial_cycle_high = high_turn.extreme_price
            position.partial_is_fifth_wave = bool(
                int(position.entry_context.get("retrace_number", 1)) >= 2
                and high_turn.wave_breakout
            )
            if exit_policy is ExitPolicy.BURNS_CYCLE_V2:
                position.partial_is_fifth_wave = prepared.loc[timestamp, "wave_active_impulse"] >= 5
            return {
                "position_id": position.position_id,
                "symbol": position.symbol,
                "date": timestamp.date().isoformat(),
                "event": "cycle_high_partial_scheduled",
                "effective_on": next_timestamp.date().isoformat(),
                "cycle_high": high_turn.extreme_price,
                "fifth_wave": position.partial_is_fifth_wave,
            }
        return None

    if (
        exit_policy is ExitPolicy.BURNS_CYCLE_V2
        and not position.one_bar_mode
        and prepared.loc[timestamp, "wave_active_impulse"] >= 5
    ):
        position.one_bar_mode = True
        old_stop = position.stop_price
        position.stop_price = max(old_stop, stop_below(float(prepared.loc[timestamp, "low"])))
        if position.stop_price > old_stop:
            position.stop_reason = ExitReason.ONE_BAR_TRAIL
        return {
            "position_id": position.position_id,
            "symbol": position.symbol,
            "date": timestamp.date().isoformat(),
            "event": "later_fifth_wave_activated",
            "old_stop": old_stop,
            "new_stop": position.stop_price,
        }

    if position.one_bar_mode:
        candidate = stop_below(float(prepared.loc[timestamp, "low"]))
        if candidate > position.stop_price:
            old_stop = position.stop_price
            position.stop_price = candidate
            position.stop_reason = ExitReason.ONE_BAR_TRAIL
            return {
                "position_id": position.position_id,
                "symbol": position.symbol,
                "date": timestamp.date().isoformat(),
                "event": "one_bar_stop_raised",
                "old_stop": old_stop,
                "new_stop": candidate,
            }
        return None

    low_turn = long_cycle_low_turn(prepared, timestamp)
    if low_turn is None:
        return None
    position.runner_cycle_low_seen = True
    candidate = stop_below(low_turn.extreme_price)
    if candidate <= position.stop_price:
        return {
            "position_id": position.position_id,
            "symbol": position.symbol,
            "date": timestamp.date().isoformat(),
            "event": "cycle_low_stop_not_raised",
            "cycle_low": low_turn.extreme_price,
            "current_stop": position.stop_price,
        }
    old_stop = position.stop_price
    position.stop_price = candidate
    position.stop_reason = ExitReason.CYCLE_LOW_TRAIL
    return {
        "position_id": position.position_id,
        "symbol": position.symbol,
        "date": timestamp.date().isoformat(),
        "event": "cycle_low_stop_raised",
        "cycle_low": low_turn.extreme_price,
        "old_stop": old_stop,
        "new_stop": candidate,
    }


def run_backtest(
    frames: dict[str, pd.DataFrame],
    config: AppConfig,
    start: date,
    end: date,
    *,
    benchmark_symbol: str = "SPY",
    exit_policy: ExitPolicy = ExitPolicy.STATIC_2R,
    strategy_rules: StrategyRules = DEFAULT_RULES,
) -> BacktestResult:
    exit_policy = ExitPolicy(exit_policy)
    if start > end:
        raise ValueError("start date must not be after end date")
    missing = sorted(set(config.symbols) - set(frames))
    if missing:
        raise ValueError(f"snapshot is missing configured symbol(s): {', '.join(missing)}")

    normalized = {symbol: normalize_bars(frames[symbol]) for symbol in config.symbols}
    prepared = {symbol: prepare_indicators(frame) for symbol, frame in normalized.items()}
    if exit_policy is ExitPolicy.BURNS_CYCLE_V2 or strategy_rules.objective_wave_retraces:
        prepared = {
            symbol: add_wave_context(frame, strategy_rules.trend_slope_bars)
            for symbol, frame in prepared.items()
        }
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    dates = sorted(
        {
            timestamp
            for frame in normalized.values()
            for timestamp in frame.loc[start_ts:end_ts].index
        }
    )
    if not dates:
        raise ValueError("no bars exist in the requested backtest range")
    strategy_diagnostics = _StrategyDiagnostics(tuple(config.symbols))

    next_date: dict[str, dict[pd.Timestamp, pd.Timestamp]] = {}
    for symbol, frame in normalized.items():
        index = frame.index
        next_date[symbol] = {index[i]: index[i + 1] for i in range(len(index) - 1)}

    cash = config.risk.initial_equity
    positions: dict[str, Position] = {}
    pending: dict[str, PlannedOrder] = {}
    trades: list[Trade] = []
    equity_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    exit_rows: list[dict[str, object]] = []
    management_rows: list[dict[str, object]] = []
    slip = _slippage_fraction(config)
    commission = config.execution.commission_per_share

    for timestamp in dates:
        current_date = timestamp.date()

        # 1) Exit positions that were open before this session.
        for symbol in sorted(tuple(positions)):
            if timestamp not in normalized[symbol].index:
                continue
            position = positions[symbol]
            bar = normalized[symbol].loc[timestamp]
            partial_due = bool(
                exit_policy.is_burns
                and position.partial_exit_on == current_date
                and not position.first_exit_taken
            )
            if partial_due and float(bar["open"]) > position.stop_price:
                # A market-at-open partial is knowably earlier than an intraday
                # stop when the session opens above the existing stop. Recheck
                # the runner afterward because today's low can still stop it.
                partial_proceeds = _process_scheduled_partial(
                    position,
                    bar,
                    current_date,
                    slip,
                    commission,
                )
                if partial_proceeds is None:  # pragma: no cover - guarded above
                    raise RuntimeError("scheduled partial was not processed")
                cash += partial_proceeds
                exit_rows.append(_exit_row(position.exit_fills[-1]))
                if position.quantity == 0:
                    positions.pop(symbol)
                    trades.append(_finalize_trade(position))
                    continue
                runner_outcome = _existing_exit(position, bar, slip, exit_policy)
                if runner_outcome is not None:
                    exit_price, reason = runner_outcome
                    positions.pop(symbol)
                    proceeds = _record_exit(
                        position,
                        current_date,
                        exit_price,
                        reason,
                        commission,
                    )
                    cash += proceeds
                    exit_rows.append(_exit_row(position.exit_fills[-1]))
                    trades.append(_finalize_trade(position))
                continue

            outcome = _existing_exit(position, bar, slip, exit_policy)
            if outcome is None:
                if not exit_policy.is_burns:
                    continue
                partial_proceeds = _process_scheduled_partial(
                    position,
                    bar,
                    current_date,
                    slip,
                    commission,
                )
                if partial_proceeds is None:
                    continue
                cash += partial_proceeds
                exit_rows.append(_exit_row(position.exit_fills[-1]))
                if position.quantity == 0:
                    positions.pop(symbol)
                    trades.append(_finalize_trade(position))
                continue
            exit_price, reason = outcome
            position = positions.pop(symbol)
            proceeds = _record_exit(position, current_date, exit_price, reason, commission)
            cash += proceeds
            exit_rows.append(_exit_row(position.exit_fills[-1]))
            trades.append(_finalize_trade(position))

        # 2) Process only orders created from a prior close and valid this session.
        for symbol in sorted(tuple(pending)):
            order = pending[symbol]
            if current_date < order.valid_on:
                continue
            pending.pop(symbol)
            if current_date > order.valid_on or timestamp not in normalized[symbol].index:
                order_rows.append(
                    {
                        "symbol": symbol,
                        "signal_date": order.signal.signal_date.isoformat(),
                        "valid_on": order.valid_on.isoformat(),
                        "status": "expired_missing_bar",
                    }
                )
                continue

            fill = _entry_fill(order, normalized[symbol].loc[timestamp], slip)
            if fill is None:
                order_rows.append(
                    {
                        "symbol": symbol,
                        "signal_date": order.signal.signal_date.isoformat(),
                        "valid_on": order.valid_on.isoformat(),
                        "status": "not_filled",
                    }
                )
                continue

            fill_price, filled_at_open = fill
            affordable = math.floor(cash / (fill_price + commission))
            quantity = min(order.quantity, affordable)
            if quantity < 1:
                order_rows.append(
                    {
                        "symbol": symbol,
                        "signal_date": order.signal.signal_date.isoformat(),
                        "valid_on": order.valid_on.isoformat(),
                        "status": "rejected_cash",
                    }
                )
                continue

            entry_fee = quantity * commission
            cash -= quantity * fill_price + entry_fee
            position = Position(
                position_id=(
                    f"{symbol}-{order.signal.signal_date.strftime('%Y%m%d')}-{current_date:%Y%m%d}"
                ),
                symbol=symbol,
                initial_quantity=quantity,
                quantity=quantity,
                signal_date=order.signal.signal_date,
                entry_date=current_date,
                entry_price=fill_price,
                initial_stop_price=order.signal.stop_price,
                stop_price=order.signal.stop_price,
                target_price=(
                    fill_risk_target(fill_price, order.signal.stop_price, config.risk.reward_r)
                    if exit_policy is ExitPolicy.STATIC_FILL_2R
                    else order.signal.target_price
                ),
                initial_risk_per_share=fill_price - order.signal.stop_price,
                reserved_risk_per_share=order.signal.risk_per_share,
                entry_commission=entry_fee,
                entry_context=dict(order.signal.context),
            )
            positions[symbol] = position
            entry_bar = normalized[symbol].loc[timestamp]
            entry_path = (
                "open"
                if filled_at_open
                else "breakout"
                if float(entry_bar["open"]) < order.signal.entry_stop
                else "gap_retrace"
            )
            order_rows.append(
                {
                    "symbol": symbol,
                    "signal_date": order.signal.signal_date.isoformat(),
                    "valid_on": order.valid_on.isoformat(),
                    "status": "filled",
                    "quantity": quantity,
                    "fill_price": fill_price,
                    "entry_stop": order.signal.entry_stop,
                    "entry_limit": order.signal.entry_limit,
                    "entry_path": entry_path,
                    "actual_risk_per_share": position.initial_risk_per_share,
                    "reserved_risk_per_share": position.reserved_risk_per_share,
                    "effective_target": position.target_price,
                    "entry_stop_sequence_ambiguous": bool(
                        not filled_at_open
                        and entry_path == "breakout"
                        and float(entry_bar["low"]) <= position.stop_price
                        and float(entry_bar["close"]) > position.stop_price
                    ),
                }
            )

            same_bar_exit = _new_position_exit(
                position,
                normalized[symbol].loc[timestamp],
                filled_at_open,
                slip,
                exit_policy,
                entry_path=entry_path,
            )
            if same_bar_exit is not None:
                exit_price, reason = same_bar_exit
                position = positions.pop(symbol)
                proceeds = _record_exit(position, current_date, exit_price, reason, commission)
                cash += proceeds
                exit_rows.append(_exit_row(position.exit_fills[-1]))
                trades.append(_finalize_trade(position))

        # 3) Update Burns's manager from this completed close. New stops and
        # partial exits become active next session, never retroactively today.
        if exit_policy.is_burns:
            for symbol in sorted(positions):
                if timestamp not in prepared[symbol].index:
                    continue
                event = _observe_burns_management(
                    positions[symbol],
                    prepared[symbol],
                    timestamp,
                    next_date[symbol].get(timestamp),
                    exit_policy,
                )
                if event is not None:
                    management_rows.append(event)

        # 4) Evaluate this completed close. No fill-time or next-bar data enters here.
        candidates: list[Signal] = []
        for symbol in config.symbols:
            if timestamp not in prepared[symbol].index:
                continue
            assessment = assess_setup(
                symbol,
                prepared[symbol],
                timestamp,
                max_entry_gap_r=config.execution.max_entry_gap_r,
                reward_r=config.risk.reward_r,
                rules=strategy_rules,
            )
            strategy_diagnostics.observe(
                symbol,
                timestamp,
                assessment.energies,
                assessment.context,
            )
            if symbol in positions or symbol in pending:
                continue
            if assessment.signal is not None:
                candidates.append(assessment.signal)

        current_equity = _equity(cash, positions, normalized, timestamp)
        capacity = Capacity(
            equity=current_equity,
            buying_power=cash,
            committed_risk=_committed_risk(positions.values(), pending.values()),
            used_slots=len(positions) + len(pending),
        )
        for signal in sorted(candidates, key=lambda item: (-item.quality, item.symbol)):
            signal_timestamp = pd.Timestamp(signal.signal_date)
            valid_timestamp = next_date[signal.symbol].get(signal_timestamp)
            if valid_timestamp is None or valid_timestamp > end_ts:
                signal_rows.append(_signal_row(signal, "not_planned", "no in-range next session"))
                continue
            order = size_order(signal, valid_timestamp.date(), config, capacity)
            if order is None:
                signal_rows.append(_signal_row(signal, "not_planned", "portfolio capacity"))
                continue
            pending[signal.symbol] = order
            signal_rows.append(_signal_row(signal, "planned"))
            capacity = Capacity(
                equity=capacity.equity,
                buying_power=capacity.buying_power - order.reserved_notional,
                committed_risk=capacity.committed_risk + order.reserved_risk,
                used_slots=capacity.used_slots + 1,
            )

        equity_rows.append(
            {
                "date": current_date.isoformat(),
                "equity": _equity(cash, positions, normalized, timestamp),
                "cash": cash,
                "positions": len(positions),
                "pending_orders": len(pending),
            }
        )

    # Close remaining positions at the final known close so every reported run is finite.
    final_timestamp = dates[-1]
    for symbol in sorted(tuple(positions)):
        mark = _mark_price(normalized[symbol], final_timestamp)
        final_bar = normalized[symbol].loc[:final_timestamp].iloc[-1]
        exit_price = max(float(final_bar["low"]), mark * (1.0 - slip))
        position = positions.pop(symbol)
        proceeds = _record_exit(
            position,
            final_timestamp.date(),
            exit_price,
            ExitReason.END_OF_DATA,
            commission,
        )
        cash += proceeds
        exit_rows.append(_exit_row(position.exit_fills[-1]))
        trades.append(_finalize_trade(position))
    pending.clear()
    equity_rows[-1].update({"equity": cash, "cash": cash, "positions": 0, "pending_orders": 0})

    summary = _summarize(
        equity_rows,
        trades,
        normalized,
        start,
        end,
        benchmark_symbol,
        config,
        exit_policy,
    )
    summary["strategy_version"] = strategy_version(strategy_rules)
    summary["strategy_fingerprint"] = strategy_fingerprint(strategy_rules)
    summary["strategy_diagnostics"] = strategy_diagnostics.as_dict()
    summary["execution_diagnostics"] = _execution_diagnostics(signal_rows, order_rows)
    summary["exit_diagnostics"] = _exit_diagnostics(trades, exit_rows, management_rows)
    return BacktestResult(
        summary=summary,
        trades=trades,
        equity_rows=equity_rows,
        signal_rows=signal_rows,
        order_rows=order_rows,
        exit_rows=exit_rows,
        management_rows=management_rows,
    )


def _summarize(
    equity_rows: list[dict[str, object]],
    trades: list[Trade],
    frames: dict[str, pd.DataFrame],
    start: date,
    end: date,
    benchmark_symbol: str,
    config: AppConfig,
    exit_policy: ExitPolicy,
) -> dict[str, object]:
    equity = pd.Series(
        [float(row["equity"]) for row in equity_rows],
        index=pd.to_datetime([row["date"] for row in equity_rows]),
        dtype=float,
    )
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    elapsed_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1.0 / 365.25)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / elapsed_years) - 1.0
    drawdown = equity / equity.cummax() - 1.0
    sharpe = None
    if len(returns) > 1 and returns.std(ddof=1) > 0:
        sharpe = float(np.sqrt(252.0) * returns.mean() / returns.std(ddof=1))

    profits = [trade.pnl for trade in trades if trade.pnl > 0]
    losses = [trade.pnl for trade in trades if trade.pnl < 0]
    breakeven_trades = sum(trade.pnl == 0 for trade in trades)
    gross_profit = sum(profits)
    gross_loss = sum(losses)
    profit_factor = None
    if losses:
        profit_factor = gross_profit / abs(gross_loss)
    win_rate = sum(trade.pnl > 0 for trade in trades) / len(trades) if trades else None
    average_winner = float(np.mean(profits)) if profits else None
    average_loser = float(np.mean(losses)) if losses else None
    payoff_ratio = None
    if average_winner is not None and average_loser is not None:
        payoff_ratio = average_winner / abs(average_loser)
    expectancy_r = float(np.mean([trade.r_multiple for trade in trades])) if trades else None

    benchmark_return = None
    benchmark_cagr = None
    if benchmark_symbol in frames:
        benchmark = frames[benchmark_symbol].loc[pd.Timestamp(start) : pd.Timestamp(end), "close"]
        if len(benchmark) >= 2:
            benchmark_return = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0)
            benchmark_cagr = (1.0 + benchmark_return) ** (1.0 / elapsed_years) - 1.0

    invested_sessions = sum(int(row.get("positions", 0)) > 0 for row in equity_rows)
    average_positions = float(np.mean([int(row.get("positions", 0)) for row in equity_rows]))

    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "start_equity": config.risk.initial_equity,
        "end_equity": float(equity.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "daily_sharpe": sharpe,
        "trades": len(trades),
        "winning_trades": len(profits),
        "losing_trades": len(losses),
        "breakeven_trades": breakeven_trades,
        "win_rate": win_rate,
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "average_winner": average_winner,
        "average_loser": average_loser,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "expectancy_r": expectancy_r,
        "fees": float(sum(trade.fees for trade in trades)),
        "benchmark_symbol": benchmark_symbol,
        "benchmark_return": benchmark_return,
        "benchmark_cagr": benchmark_cagr,
        "invested_sessions": invested_sessions,
        "invested_session_fraction": invested_sessions / len(equity_rows),
        "average_positions": average_positions,
        "max_concurrent_positions": max(int(row.get("positions", 0)) for row in equity_rows),
        "meets_15_percent_cagr_hurdle": bool(cagr >= 0.15),
        "meets_20_percent_cagr_hurdle": bool(cagr >= 0.20),
        "engine_version": ENGINE_VERSION,
        "r_basis": "actual entry fill minus initial stop; net of fees",
        "expectancy_reserved_r": (
            float(np.mean([t.reserved_r_multiple for t in trades])) if trades else None
        ),
        "strategy_version": strategy_version(),
        "strategy_fingerprint": strategy_fingerprint(),
        "exit_policy": exit_policy.value,
        "exit_policy_fingerprint": exit_policy_fingerprint(exit_policy),
        "config_fingerprint": config_fingerprint(config),
    }


def write_report(
    result: BacktestResult,
    output_directory: str | Path,
    *,
    provenance: dict[str, object] | None = None,
) -> Path:
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"report directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    summary = dict(result.summary)
    if provenance:
        summary["provenance"] = provenance
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([asdict(trade) for trade in result.trades]).to_csv(
        output / "trades.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(result.equity_rows).to_csv(output / "equity.csv", index=False, lineterminator="\n")
    pd.DataFrame(result.signal_rows).to_csv(
        output / "signals.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(result.order_rows).to_csv(output / "orders.csv", index=False, lineterminator="\n")
    pd.DataFrame(result.exit_rows).to_csv(output / "exits.csv", index=False, lineterminator="\n")
    pd.DataFrame(result.management_rows).to_csv(
        output / "management.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(_yearly_rows(result)).to_csv(
        output / "yearly.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(_symbol_rows(result)).to_csv(
        output / "by_symbol.csv", index=False, lineterminator="\n"
    )
    return output


def _yearly_rows(result: BacktestResult) -> list[dict[str, object]]:
    equity = pd.Series(
        [float(row["equity"]) for row in result.equity_rows],
        index=pd.to_datetime([row["date"] for row in result.equity_rows]),
        dtype=float,
    )
    rows: list[dict[str, object]] = []
    prior_close: float | None = None
    for year, group in equity.groupby(equity.index.year):
        start_value = prior_close if prior_close is not None else float(group.iloc[0])
        end_value = float(group.iloc[-1])
        curve = pd.concat(
            [pd.Series([start_value], index=[group.index[0] - pd.Timedelta(microseconds=1)]), group]
        )
        drawdown = curve / curve.cummax() - 1.0
        year_trades = [trade for trade in result.trades if trade.exit_date.year == int(year)]
        rows.append(
            {
                "year": int(year),
                "start_equity": start_value,
                "end_equity": end_value,
                "return": end_value / start_value - 1.0,
                "max_drawdown": float(drawdown.min()),
                "trades": len(year_trades),
                "pnl": sum(trade.pnl for trade in year_trades),
            }
        )
        prior_close = end_value
    return rows


def _symbol_rows(result: BacktestResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in sorted({trade.symbol for trade in result.trades}):
        trades = [trade for trade in result.trades if trade.symbol == symbol]
        rows.append(
            {
                "symbol": symbol,
                "trades": len(trades),
                "win_rate": sum(trade.pnl > 0 for trade in trades) / len(trades),
                "pnl": sum(trade.pnl for trade in trades),
                "expectancy_r": float(np.mean([trade.r_multiple for trade in trades])),
            }
        )
    return rows
