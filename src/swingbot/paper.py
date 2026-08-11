from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig, alpaca_credentials
from .models import PlannedOrder, Signal
from .portfolio import Capacity, size_order
from .strategy import generate_signal, prepare_indicators


@dataclass(frozen=True)
class BrokerState:
    equity: float
    buying_power: float
    committed_risk: float
    used_slots: int
    blocked_symbols: frozenset[str]


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _flatten_orders(orders: Iterable[Any]) -> list[Any]:
    flattened: list[Any] = []
    for order in orders:
        flattened.append(order)
        legs = getattr(order, "legs", None) or []
        flattened.extend(_flatten_orders(legs))
    return flattened


def _next_weekday(value: date) -> date:
    candidate = value + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


class AlpacaPaperBroker:
    """Fail-closed Alpaca adapter with paper mode permanently enabled."""

    def __init__(self) -> None:
        key, secret = alpaca_credentials()
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise RuntimeError(
                "install paper dependencies with: pip install -e '.[paper]'"
            ) from exc
        self._client = TradingClient(key, secret, paper=True)

    def state(self) -> BrokerState:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        try:
            account = self._client.get_account()
            positions = list(self._client.get_all_positions())
            parents = list(
                self._client.get_orders(
                    filter=GetOrdersRequest(
                        status=QueryOrderStatus.OPEN,
                        limit=500,
                        nested=True,
                    )
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "paper account reconciliation failed; no orders were submitted"
            ) from exc

        if bool(getattr(account, "trading_blocked", False)):
            raise RuntimeError("Alpaca reports that paper trading is blocked")
        if len(parents) >= 500:
            raise RuntimeError(
                "paper account has at least 500 open orders; reconciliation is incomplete"
            )
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        if equity <= 0 or buying_power < 0:
            raise RuntimeError("Alpaca returned invalid account equity or buying power")

        all_orders = _flatten_orders(parents)
        blocked_symbols = {
            str(getattr(position, "symbol", "")).upper() for position in positions
        }
        blocked_symbols.update(
            str(getattr(order, "symbol", "")).upper()
            for order in parents
            if getattr(order, "symbol", None)
        )
        blocked_symbols.discard("")

        stops_by_symbol: dict[str, list[float]] = {}
        for order in all_orders:
            symbol = str(getattr(order, "symbol", "")).upper()
            side = _enum_text(getattr(order, "side", ""))
            order_type = _enum_text(getattr(order, "type", ""))
            stop_price = getattr(order, "stop_price", None)
            if symbol and side == "sell" and order_type in {"stop", "stop_limit"} and stop_price:
                stops_by_symbol.setdefault(symbol, []).append(float(stop_price))

        committed_risk = 0.0
        position_symbols: set[str] = set()
        for position in positions:
            symbol = str(position.symbol).upper()
            position_symbols.add(symbol)
            if _enum_text(getattr(position, "side", "long")) not in {"long", "buy"}:
                raise RuntimeError(f"unsupported non-long paper position exists: {symbol}")
            stops = stops_by_symbol.get(symbol, [])
            if not stops:
                raise RuntimeError(
                    f"paper position {symbol} has no visible protective stop; refusing new orders"
                )
            current_price = float(position.current_price)
            valid_stops = [price for price in stops if price < current_price]
            if not valid_stops:
                raise RuntimeError(
                    f"paper position {symbol} has no protective stop below market; "
                    "refusing new orders"
                )
            protective_stop = max(valid_stops)
            committed_risk += float(position.qty) * max(0.0, current_price - protective_stop)

        # Reserve risk for unfilled entry parents as well as filled positions.
        for parent in parents:
            symbol = str(getattr(parent, "symbol", "")).upper()
            if not symbol or symbol in position_symbols:
                continue
            side = _enum_text(getattr(parent, "side", ""))
            if side != "buy":
                continue
            entry_price = getattr(parent, "limit_price", None)
            legs = _flatten_orders(getattr(parent, "legs", None) or [])
            stop_prices = [
                float(leg.stop_price)
                for leg in legs
                if getattr(leg, "stop_price", None)
                and _enum_text(getattr(leg, "side", "")) == "sell"
            ]
            if entry_price is None or not stop_prices:
                raise RuntimeError(
                    f"open paper entry for {symbol} cannot be risk-reconciled; refusing new orders"
                )
            committed_risk += float(parent.qty) * max(0.0, float(entry_price) - max(stop_prices))

        return BrokerState(
            equity=equity,
            buying_power=buying_power,
            committed_risk=committed_risk,
            used_slots=len(blocked_symbols),
            blocked_symbols=frozenset(blocked_symbols),
        )

    def submit(
        self,
        plans: Iterable[PlannedOrder],
        config: AppConfig,
        expected_state: BrokerState,
        *,
        confirmation: str,
    ) -> list[dict[str, str]]:
        if confirmation != "PAPER":
            raise ValueError("submission requires --confirm PAPER")

        try:
            clock = self._client.get_clock()
        except Exception as exc:
            raise RuntimeError("could not verify that the market is closed") from exc
        if bool(getattr(clock, "is_open", True)):
            raise RuntimeError("paper submission is disabled while the market is open")

        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        plans_list = list(plans)
        symbols = [plan.signal.symbol for plan in plans_list]
        if len(symbols) != len(set(symbols)):
            raise RuntimeError("paper plan contains duplicate symbols")
        before = self.state()
        if before.blocked_symbols != expected_state.blocked_symbols:
            raise RuntimeError("paper positions or open orders changed after planning")
        conflicts = sorted(
            plan.signal.symbol
            for plan in plans_list
            if plan.signal.symbol in before.blocked_symbols
        )
        if conflicts:
            raise RuntimeError(
                f"paper state changed; existing position/order for: {', '.join(conflicts)}"
            )

        total_risk = sum(plan.reserved_risk for plan in plans_list)
        total_notional = sum(plan.reserved_notional for plan in plans_list)
        if before.used_slots + len(plans_list) > config.risk.max_positions:
            raise RuntimeError("paper position-count capacity changed after planning")
        if before.committed_risk + total_risk > before.equity * config.risk.max_total_risk:
            raise RuntimeError("paper total-risk capacity changed after planning")
        if total_notional > before.buying_power:
            raise RuntimeError("paper buying power changed after planning")
        for plan in plans_list:
            if plan.reserved_risk > before.equity * config.risk.risk_per_trade + 0.01:
                raise RuntimeError(f"paper risk budget changed for {plan.signal.symbol}")
            if plan.reserved_notional > before.equity * config.risk.max_position_fraction + 0.01:
                raise RuntimeError(f"paper notional cap changed for {plan.signal.symbol}")

        submitted: list[dict[str, str]] = []
        for plan in plans_list:
            signal = plan.signal
            request = LimitOrderRequest(
                symbol=signal.symbol,
                qty=plan.quantity,
                side=OrderSide.BUY,
                limit_price=signal.entry_limit,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=signal.target_price),
                stop_loss=StopLossRequest(stop_price=signal.stop_price),
                client_order_id=(
                    f"swingbot-{signal.signal_date.strftime('%Y%m%d')}-{signal.symbol}"
                ),
            )
            try:
                order = self._client.submit_order(order_data=request)
            except Exception as exc:
                completed = ", ".join(item["symbol"] for item in submitted) or "none"
                raise RuntimeError(
                    f"Alpaca rejected {signal.symbol}; already submitted: {completed}; "
                    "remaining plans were not submitted"
                ) from exc
            submitted.append({"symbol": signal.symbol, "order_id": str(order.id)})
        return submitted


def build_paper_plan(
    frames: dict[str, pd.DataFrame],
    as_of: date,
    config: AppConfig,
    broker_state: BrokerState,
) -> list[PlannedOrder]:
    timestamp = pd.Timestamp(as_of)
    candidates: list[Signal] = []
    symbols_with_bar = 0
    for symbol in config.symbols:
        if symbol not in frames:
            raise ValueError(f"no paper-scan data for {symbol}")
        prepared = prepare_indicators(frames[symbol])
        if timestamp not in prepared.index:
            continue
        symbols_with_bar += 1
        if symbol in broker_state.blocked_symbols:
            continue
        signal = generate_signal(
            symbol,
            prepared,
            timestamp,
            max_entry_gap_r=config.execution.max_entry_gap_r,
            reward_r=config.risk.reward_r,
        )
        if signal is not None:
            candidates.append(signal)

    if symbols_with_bar == 0:
        raise ValueError(f"none of the configured symbols has a completed bar on {as_of}")

    capacity = Capacity(
        equity=broker_state.equity,
        buying_power=broker_state.buying_power,
        committed_risk=broker_state.committed_risk,
        used_slots=broker_state.used_slots,
    )
    plans: list[PlannedOrder] = []
    for signal in sorted(candidates, key=lambda item: (-item.quality, item.symbol)):
        plan = size_order(signal, _next_weekday(as_of), config, capacity)
        if plan is None:
            continue
        plans.append(plan)
        capacity = Capacity(
            equity=capacity.equity,
            buying_power=capacity.buying_power - plan.reserved_notional,
            committed_risk=capacity.committed_risk + plan.reserved_risk,
            used_slots=capacity.used_slots + 1,
        )
    return plans


def write_paper_plan(plans: Iterable[PlannedOrder], path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"paper plan already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for plan in plans:
        item = asdict(plan)
        item["valid_on"] = plan.valid_on.isoformat()
        item["signal"]["signal_date"] = plan.signal.signal_date.isoformat()
        payload.append(item)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
