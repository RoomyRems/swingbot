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
from .indicators import normalize_bars
from .models import (
    BacktestResult,
    EnergyEvidence,
    ExitReason,
    PlannedOrder,
    Position,
    Signal,
    Trade,
)
from .portfolio import Capacity, size_order
from .strategy import (
    STRATEGY_VERSION,
    assess_setup,
    prepare_indicators,
    strategy_fingerprint,
)

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
        position.quantity * position.initial_risk_per_share + 2.0 * position.entry_commission
        for position in positions
    )
    return open_risk + sum(order.reserved_risk for order in pending)


def _existing_exit(
    position: Position,
    bar: pd.Series,
    slip: float,
) -> tuple[float, ExitReason] | None:
    if float(bar["open"]) <= position.stop_price:
        return max(float(bar["low"]), float(bar["open"]) * (1.0 - slip)), ExitReason.STOP
    if float(bar["open"]) >= position.target_price:
        return position.target_price, ExitReason.TARGET

    stop_touched = float(bar["low"]) <= position.stop_price
    target_touched = float(bar["high"]) >= position.target_price
    if stop_touched:
        # Daily OHLC cannot reveal whether a same-day stop or target came first.
        # The conservative convention always assigns the stop.
        return max(float(bar["low"]), position.stop_price * (1.0 - slip)), ExitReason.STOP
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
) -> tuple[float, ExitReason] | None:
    if filled_at_open and float(bar["open"]) <= position.stop_price:
        return max(float(bar["low"]), float(bar["open"]) * (1.0 - slip)), ExitReason.STOP
    if float(bar["low"]) <= position.stop_price:
        return max(float(bar["low"]), position.stop_price * (1.0 - slip)), ExitReason.STOP
    if filled_at_open and float(bar["high"]) >= position.target_price:
        return position.target_price, ExitReason.TARGET
    # If a limit filled intraday, the day's high may have occurred before entry.
    # We therefore never award a same-bar target to an intraday fill.
    return None


def _close_position(
    position: Position,
    exit_date: date,
    exit_price: float,
    reason: ExitReason,
    commission_per_share: float,
) -> tuple[Trade, float]:
    exit_fee = position.quantity * commission_per_share
    proceeds = position.quantity * exit_price - exit_fee
    fees = position.entry_commission + exit_fee
    pnl = position.quantity * (exit_price - position.entry_price) - fees
    initial_risk = position.quantity * position.initial_risk_per_share
    r_multiple = pnl / initial_risk if initial_risk > 0 else float("nan")
    trade = Trade(
        symbol=position.symbol,
        signal_date=position.signal_date,
        entry_date=position.entry_date,
        exit_date=exit_date,
        quantity=position.quantity,
        entry_price=position.entry_price,
        exit_price=exit_price,
        stop_price=position.stop_price,
        target_price=position.target_price,
        reason=reason,
        pnl=pnl,
        r_multiple=r_multiple,
        fees=fees,
    )
    return trade, proceeds


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
    }


def run_backtest(
    frames: dict[str, pd.DataFrame],
    config: AppConfig,
    start: date,
    end: date,
    *,
    benchmark_symbol: str = "SPY",
) -> BacktestResult:
    if start > end:
        raise ValueError("start date must not be after end date")
    missing = sorted(set(config.symbols) - set(frames))
    if missing:
        raise ValueError(f"snapshot is missing configured symbol(s): {', '.join(missing)}")

    normalized = {symbol: normalize_bars(frames[symbol]) for symbol in config.symbols}
    prepared = {symbol: prepare_indicators(frame) for symbol, frame in normalized.items()}
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
    slip = _slippage_fraction(config)
    commission = config.execution.commission_per_share

    for timestamp in dates:
        current_date = timestamp.date()

        # 1) Exit positions that were open before this session.
        for symbol in sorted(tuple(positions)):
            if timestamp not in normalized[symbol].index:
                continue
            outcome = _existing_exit(positions[symbol], normalized[symbol].loc[timestamp], slip)
            if outcome is None:
                continue
            exit_price, reason = outcome
            trade, proceeds = _close_position(
                positions.pop(symbol), current_date, exit_price, reason, commission
            )
            cash += proceeds
            trades.append(trade)

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
                symbol=symbol,
                quantity=quantity,
                signal_date=order.signal.signal_date,
                entry_date=current_date,
                entry_price=fill_price,
                stop_price=order.signal.stop_price,
                target_price=order.signal.target_price,
                initial_risk_per_share=order.signal.risk_per_share,
                entry_commission=entry_fee,
            )
            positions[symbol] = position
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
                }
            )

            same_bar_exit = _new_position_exit(
                position,
                normalized[symbol].loc[timestamp],
                filled_at_open,
                slip,
            )
            if same_bar_exit is not None:
                exit_price, reason = same_bar_exit
                trade, proceeds = _close_position(
                    positions.pop(symbol), current_date, exit_price, reason, commission
                )
                cash += proceeds
                trades.append(trade)

        # 3) Evaluate this completed close. No fill-time or next-bar data enters here.
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
        trade, proceeds = _close_position(
            positions.pop(symbol),
            final_timestamp.date(),
            exit_price,
            ExitReason.END_OF_DATA,
            commission,
        )
        cash += proceeds
        trades.append(trade)
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
    )
    summary["strategy_diagnostics"] = strategy_diagnostics.as_dict()
    summary["execution_diagnostics"] = _execution_diagnostics(signal_rows, order_rows)
    return BacktestResult(
        summary=summary,
        trades=trades,
        equity_rows=equity_rows,
        signal_rows=signal_rows,
        order_rows=order_rows,
    )


def _summarize(
    equity_rows: list[dict[str, object]],
    trades: list[Trade],
    frames: dict[str, pd.DataFrame],
    start: date,
    end: date,
    benchmark_symbol: str,
    config: AppConfig,
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
    if benchmark_symbol in frames:
        benchmark = frames[benchmark_symbol].loc[pd.Timestamp(start) : pd.Timestamp(end), "close"]
        if len(benchmark) >= 2:
            benchmark_return = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0)

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
        "strategy_version": STRATEGY_VERSION,
        "strategy_fingerprint": strategy_fingerprint(),
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
