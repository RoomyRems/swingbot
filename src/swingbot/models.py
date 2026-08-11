from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class ExitReason(StrEnum):
    STOP = "stop"
    TARGET = "target"
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
    entry_limit: float
    target_price: float
    score: int
    quality: float
    energies: Mapping[str, EnergyEvidence]

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
    symbol: str
    quantity: int
    signal_date: date
    entry_date: date
    entry_price: float
    stop_price: float
    target_price: float
    initial_risk_per_share: float
    entry_commission: float = 0.0


@dataclass(frozen=True)
class Trade:
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


@dataclass
class BacktestResult:
    summary: dict[str, float | int | str | None]
    trades: list[Trade] = field(default_factory=list)
    equity_rows: list[dict[str, object]] = field(default_factory=list)
    signal_rows: list[dict[str, object]] = field(default_factory=list)
    order_rows: list[dict[str, object]] = field(default_factory=list)
