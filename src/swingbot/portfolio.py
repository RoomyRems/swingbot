from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from .config import AppConfig
from .models import PlannedOrder, Signal


@dataclass(frozen=True)
class Capacity:
    equity: float
    buying_power: float
    committed_risk: float
    used_slots: int


def size_order(
    signal: Signal,
    valid_on: date,
    config: AppConfig,
    capacity: Capacity,
) -> PlannedOrder | None:
    """Risk-size one order while enforcing portfolio, position, and cash caps."""
    if capacity.equity <= 0 or capacity.buying_power <= 0:
        return None
    if capacity.used_slots >= config.risk.max_positions:
        return None

    fee_per_round_trip_share = 2.0 * config.execution.commission_per_share
    effective_risk_per_share = signal.risk_per_share + fee_per_round_trip_share
    if effective_risk_per_share <= 0:
        return None

    trade_risk_budget = capacity.equity * config.risk.risk_per_trade
    portfolio_risk_left = max(
        0.0,
        capacity.equity * config.risk.max_total_risk - capacity.committed_risk,
    )
    notional_cap = capacity.equity * config.risk.max_position_fraction
    cash_per_share = signal.entry_limit + config.execution.commission_per_share

    quantity = math.floor(
        min(
            trade_risk_budget / effective_risk_per_share,
            portfolio_risk_left / effective_risk_per_share,
            notional_cap / cash_per_share,
            capacity.buying_power / cash_per_share,
        )
    )
    if quantity < 1:
        return None

    return PlannedOrder(
        signal=signal,
        quantity=quantity,
        reserved_risk=quantity * effective_risk_per_share,
        reserved_notional=quantity * cash_per_share,
        valid_on=valid_on,
    )
