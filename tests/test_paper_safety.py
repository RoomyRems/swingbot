from __future__ import annotations

import sys
import types
import unittest
from datetime import date
from unittest.mock import patch

import swingbot.paper as paper_module
from swingbot.paper import AlpacaPaperBroker, BrokerState, build_paper_plan
from swingbot.portfolio import Capacity, size_order
from tests.helpers import app_config, complete_signal, make_bars


class PaperSafetyTests(unittest.TestCase):
    def test_alpaca_client_is_permanently_created_in_paper_mode(self):
        captured = {}

        class FakeTradingClient:
            def __init__(self, key, secret, *, paper):
                captured.update(key=key, secret=secret, paper=paper)

        modules = {
            "alpaca": types.ModuleType("alpaca"),
            "alpaca.trading": types.ModuleType("alpaca.trading"),
            "alpaca.trading.client": types.ModuleType("alpaca.trading.client"),
        }
        modules["alpaca.trading.client"].TradingClient = FakeTradingClient
        with (
            patch.dict(sys.modules, modules),
            patch.object(paper_module, "alpaca_credentials", return_value=("key", "secret")),
        ):
            AlpacaPaperBroker()
        self.assertEqual(captured, {"key": "key", "secret": "secret", "paper": True})

    def test_paper_plan_uses_shared_signal_and_portfolio_caps(self):
        frame = make_bars(rows=320)
        as_of = frame.index[-1].date()
        signal = complete_signal("SPY", as_of)
        with patch.object(paper_module, "generate_signal", return_value=signal):
            plans = build_paper_plan(
                {"SPY": frame},
                as_of,
                app_config("SPY"),
                BrokerState(
                    equity=100_000.0,
                    buying_power=100_000.0,
                    committed_risk=0.0,
                    used_slots=0,
                    blocked_symbols=frozenset(),
                ),
            )
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].quantity, 45)

    def test_existing_symbol_is_never_planned(self):
        frame = make_bars(rows=320)
        as_of = frame.index[-1].date()
        signal = complete_signal("SPY", as_of)
        with patch.object(paper_module, "generate_signal", return_value=signal):
            plans = build_paper_plan(
                {"SPY": frame},
                as_of,
                app_config("SPY"),
                BrokerState(
                    equity=100_000.0,
                    buying_power=100_000.0,
                    committed_risk=500.0,
                    used_slots=1,
                    blocked_symbols=frozenset({"SPY"}),
                ),
            )
        self.assertEqual(plans, [])

    def test_submission_requires_explicit_paper_confirmation(self):
        broker = object.__new__(AlpacaPaperBroker)
        with self.assertRaisesRegex(ValueError, "confirm PAPER"):
            broker.submit(
                [],
                app_config("SPY"),
                BrokerState(100_000.0, 100_000.0, 0.0, 0, frozenset()),
                confirmation="yes",
            )

    def test_reconciliation_refuses_unprotected_position(self):
        class FakeClient:
            def get_account(self):
                return types.SimpleNamespace(
                    equity="100000",
                    buying_power="100000",
                    trading_blocked=False,
                )

            def get_all_positions(self):
                return [
                    types.SimpleNamespace(
                        symbol="SPY",
                        side="long",
                        current_price="500",
                        qty="10",
                    )
                ]

            def get_orders(self, filter):
                return []

        modules = {
            "alpaca.trading.enums": types.ModuleType("alpaca.trading.enums"),
            "alpaca.trading.requests": types.ModuleType("alpaca.trading.requests"),
        }
        modules["alpaca.trading.enums"].QueryOrderStatus = types.SimpleNamespace(OPEN="open")

        class FakeGetOrdersRequest:
            def __init__(self, **kwargs):
                self.values = kwargs

        modules["alpaca.trading.requests"].GetOrdersRequest = FakeGetOrdersRequest
        broker = object.__new__(AlpacaPaperBroker)
        broker._client = FakeClient()
        with (
            patch.dict(sys.modules, modules),
            self.assertRaisesRegex(RuntimeError, "no visible protective stop"),
        ):
            broker.state()

    def test_submission_builds_a_day_paper_bracket(self):
        config = app_config("SPY")
        signal = complete_signal("SPY", date(2026, 8, 7))
        state = BrokerState(100_000.0, 100_000.0, 0.0, 0, frozenset())
        plan = size_order(
            signal,
            date(2026, 8, 10),
            config,
            Capacity(100_000.0, 100_000.0, 0.0, 0),
        )
        assert plan is not None
        captured = {}

        class Request:
            def __init__(self, **kwargs):
                self.values = kwargs

        class FakeClient:
            def get_clock(self):
                return types.SimpleNamespace(is_open=False)

            def submit_order(self, order_data):
                captured.update(order_data.values)
                return types.SimpleNamespace(id="paper-order-id")

        enums = types.ModuleType("alpaca.trading.enums")
        enums.OrderClass = types.SimpleNamespace(BRACKET="bracket")
        enums.OrderSide = types.SimpleNamespace(BUY="buy")
        enums.TimeInForce = types.SimpleNamespace(DAY="day")
        requests = types.ModuleType("alpaca.trading.requests")
        requests.StopLossRequest = Request
        requests.StopLimitOrderRequest = Request
        requests.TakeProfitRequest = Request

        broker = object.__new__(AlpacaPaperBroker)
        broker._client = FakeClient()
        with (
            patch.dict(
                sys.modules,
                {
                    "alpaca.trading.enums": enums,
                    "alpaca.trading.requests": requests,
                },
            ),
            patch.object(broker, "state", return_value=state),
        ):
            submitted = broker.submit(
                [plan],
                config,
                state,
                confirmation="PAPER",
            )

        self.assertEqual(submitted[0]["order_id"], "paper-order-id")
        self.assertEqual(captured["order_class"], "bracket")
        self.assertEqual(captured["time_in_force"], "day")
        self.assertEqual(captured["stop_price"], signal.entry_stop)
        self.assertEqual(captured["limit_price"], signal.entry_limit)
        self.assertEqual(captured["stop_loss"].values["stop_price"], signal.stop_price)

    def test_submission_refuses_an_open_market(self):
        class FakeClient:
            def get_clock(self):
                return types.SimpleNamespace(is_open=True)

        broker = object.__new__(AlpacaPaperBroker)
        broker._client = FakeClient()
        with self.assertRaisesRegex(RuntimeError, "while the market is open"):
            broker.submit(
                [],
                app_config("SPY"),
                BrokerState(100_000.0, 100_000.0, 0.0, 0, frozenset()),
                confirmation="PAPER",
            )


if __name__ == "__main__":
    unittest.main()
