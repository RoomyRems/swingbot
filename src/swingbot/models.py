from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class ExitReason(StrEnum):
    STOP = "stop"
    TARGET = "target"
    CYCLE_HIGH_PARTIAL = "cycle_high_partial"
    CYCLE_LOW_TRAIL = "cycle_low_trail"
    ONE_BAR_TRAIL = "one_bar_trail"
    END_OF_DATA = "end_of_data"


@dataclass(frozen=True)
class EnergyEvidence:
    passed: bool
    value: float | None
    rule: str


@dataclass(frozen=True)
class Signal:
    symbol: str
    signal_date: date
    reference_price: float
    stop_price: float
    entry_stop: float
    entry_limit: float
    target_price: float
    average_daily_volume: float
    score: int
    quality: float
    energies: Mapping[str, EnergyEvidence]
    context: Mapping[str, object]

    @property
    def risk_per_share(self) -> float:
        return self.entry_limit - self.stop_price


@dataclass(frozen=True)
class PlannedOrder:
    signal: Signal
    quantity: int
    reserved_risk: float
    reserved_notional: float
    valid_on: date


@dataclass
class Position:
    position_id: str
    symbol: str
    initial_quantity: int
    quantity: int
    signal_date: date
    entry_date: date
    entry_price: float
    initial_stop_price: float
    stop_price: float
    target_price: float
    initial_risk_per_share: float
    reserved_risk_per_share: float | None = None
    entry_commission: float = 0.0
    entry_context: Mapping[str, object] = field(default_factory=dict)
    exit_fills: list[ExitFill] = field(default_factory=list)
    partial_exit_on: date | None = None
    partial_signal_date: date | None = None
    partial_hook_bar_low: float | None = None
    partial_cycle_high: float | None = None
    partial_is_fifth_wave: bool = False
    first_exit_taken: bool = False
    runner_cycle_low_seen: bool = False
    one_bar_mode: bool = False
    stop_reason: ExitReason = ExitReason.STOP


@dataclass(frozen=True)
class ExitFill:
    position_id: str
    symbol: str
    exit_date: date
    quantity: int
    price: float
    reason: ExitReason
    fees: float


@dataclass(frozen=True)
class Trade:
    position_id: str
    symbol: str
    signal_date: date
    entry_date: date
    exit_date: date
    quantity: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    reason: ExitReason
    pnl: float
    r_multiple: float
    fees: float
    exit_legs: int = 1
    first_exit_date: date | None = None
    initial_stop_price: float | None = None
    final_stop_price: float | None = None
    reserved_r_multiple: float | None = None
    actual_risk_per_share: float | None = None
    reserved_risk_per_share: float | None = None


@dataclass
class BacktestResult:
    summary: dict[str, object]
    trades: list[Trade] = field(default_factory=list)
    equity_rows: list[dict[str, object]] = field(default_factory=list)
    signal_rows: list[dict[str, object]] = field(default_factory=list)
    order_rows: list[dict[str, object]] = field(default_factory=list)
    exit_rows: list[dict[str, object]] = field(default_factory=list)
    management_rows: list[dict[str, object]] = field(default_factory=list)
