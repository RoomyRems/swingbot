from __future__ import annotations

import sys
import types
import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

import swingbot.data as data_module
from swingbot.config import DataConfig
from swingbot.data import AlpacaDataSource


class DataSourceTests(unittest.TestCase):
    def test_alpaca_request_uses_explicit_dates_adjustments_feed_and_asof(self):
        captured = {}
        index = pd.MultiIndex.from_tuples(
            [
                ("SPY", pd.Timestamp("2024-01-02", tz="UTC")),
                ("SPY", pd.Timestamp("2024-01-03", tz="UTC")),
            ],
            names=["symbol", "timestamp"],
        )
        response_frame = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1_000_000.0, 1_100_000.0],
            },
            index=index,
        )

        class FakeClient:
            def __init__(self, key, secret):
                captured["credentials"] = (key, secret)

            def get_stock_bars(self, request):
                captured["request"] = request.values
                return types.SimpleNamespace(df=response_frame)

        class FakeRequest:
            def __init__(self, **kwargs):
                self.values = kwargs

        modules = {
            "alpaca": types.ModuleType("alpaca"),
            "alpaca.data": types.ModuleType("alpaca.data"),
            "alpaca.data.historical": types.ModuleType("alpaca.data.historical"),
            "alpaca.data.enums": types.ModuleType("alpaca.data.enums"),
            "alpaca.data.requests": types.ModuleType("alpaca.data.requests"),
            "alpaca.data.timeframe": types.ModuleType("alpaca.data.timeframe"),
        }
        modules["alpaca.data.historical"].StockHistoricalDataClient = FakeClient
        modules["alpaca.data.enums"].Adjustment = types.SimpleNamespace(ALL="all")
        modules["alpaca.data.enums"].DataFeed = types.SimpleNamespace(SIP="sip", IEX="iex")
        modules["alpaca.data.requests"].StockBarsRequest = FakeRequest
        modules["alpaca.data.timeframe"].TimeFrame = types.SimpleNamespace(Day="1Day")

        with (
            patch.dict(sys.modules, modules),
            patch.object(data_module, "alpaca_credentials", return_value=("key", "secret")),
        ):
            source = AlpacaDataSource()
            frames = source.fetch_daily(
                ["SPY"],
                date(2024, 1, 2),
                date(2024, 1, 3),
                DataConfig(feed="sip", adjustment="all"),
            )

        request = captured["request"]
        self.assertEqual(request["asof"], "2024-01-03")
        self.assertEqual(request["adjustment"], "all")
        self.assertEqual(request["feed"], "sip")
        self.assertEqual(request["start"].date(), date(2024, 1, 2))
        self.assertEqual(request["end"].date(), date(2024, 1, 4))
        self.assertEqual(len(frames["SPY"]), 2)


if __name__ == "__main__":
    unittest.main()
