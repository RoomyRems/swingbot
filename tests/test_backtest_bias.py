from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

import swingbot.backtest as backtest_module
from swingbot.backtest import run_backtest
from swingbot.models import ExitReason
from tests.helpers import app_config, complete_signal


def _three_day_frame(
    second_close: float,
    *,
    second_open: float = 100.0,
    second_high: float = 105.0,
    second_low: float = 95.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [99.0, second_open, 95.0],
            "high": [101.0, second_high, 96.0],
            "low": [98.0, second_low, 85.0],
            "close": [100.0, second_close, 90.0],
            "volume": [1_000_000.0] * 3,
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )


class BacktestBiasTests(unittest.TestCase):
    def _fake_generate(self, symbol, prepared, as_of, **kwargs):
        signal = complete_signal("TEST", date(2024, 1, 2))
        return signal if pd.Timestamp(as_of).date() == signal.signal_date else None

    def test_next_bar_close_cannot_revalidate_or_change_entry(self):
        config = app_config("TEST")
        with patch.object(backtest_module, "generate_signal", side_effect=self._fake_generate):
            low_close = run_backtest(
                {"TEST": _three_day_frame(96.0)},
                config,
                date(2024, 1, 2),
                date(2024, 1, 4),
                benchmark_symbol="TEST",
            )
            high_close = run_backtest(
                {"TEST": _three_day_frame(104.0)},
                config,
                date(2024, 1, 2),
                date(2024, 1, 4),
                benchmark_symbol="TEST",
            )
        self.assertEqual(
            low_close.order_rows[0]["fill_price"],
            high_close.order_rows[0]["fill_price"],
        )
        self.assertEqual(low_close.trades[0].entry_price, 100.0)
        self.assertEqual(high_close.trades[0].entry_price, 100.0)
        self.assertEqual(low_close.trades[0].reason, ExitReason.STOP)
        self.assertEqual(high_close.trades[0].reason, ExitReason.STOP)
        self.assertEqual(low_close.summary["strategy_version"], "burns-book-v1")
        self.assertEqual(len(low_close.summary["strategy_fingerprint"]), 64)

    def test_ambiguous_same_day_stop_and_target_uses_stop(self):
        frame = _three_day_frame(100.0, second_high=130.0, second_low=89.0)
        with patch.object(backtest_module, "generate_signal", side_effect=self._fake_generate):
            result = run_backtest(
                {"TEST": frame},
                app_config("TEST"),
                date(2024, 1, 2),
                date(2024, 1, 4),
                benchmark_symbol="TEST",
            )
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].entry_date, result.trades[0].exit_date)
        self.assertEqual(result.trades[0].reason, ExitReason.STOP)

    def test_next_session_limit_can_expire_unfilled(self):
        frame = _three_day_frame(
            110.0,
            second_open=110.0,
            second_high=112.0,
            second_low=102.0,
        )
        with patch.object(backtest_module, "generate_signal", side_effect=self._fake_generate):
            result = run_backtest(
                {"TEST": frame},
                app_config("TEST"),
                date(2024, 1, 2),
                date(2024, 1, 4),
                benchmark_symbol="TEST",
            )
        self.assertEqual(result.order_rows[0]["status"], "not_filled")
        self.assertEqual(result.trades, [])

    def test_price_below_limit_does_not_fill_before_buy_stop_triggers(self):
        frame = _three_day_frame(
            98.0,
            second_open=98.0,
            second_high=99.0,
            second_low=95.0,
        )
        with patch.object(backtest_module, "generate_signal", side_effect=self._fake_generate):
            result = run_backtest(
                {"TEST": frame},
                app_config("TEST"),
                date(2024, 1, 2),
                date(2024, 1, 4),
                benchmark_symbol="TEST",
            )
        self.assertEqual(result.order_rows[0]["status"], "not_filled")
        self.assertEqual(result.trades, [])

    def test_gap_above_limit_can_fill_only_after_retracing_to_limit(self):
        frame = _three_day_frame(
            101.0,
            second_open=110.0,
            second_high=112.0,
            second_low=100.5,
        )
        with patch.object(backtest_module, "generate_signal", side_effect=self._fake_generate):
            result = run_backtest(
                {"TEST": frame},
                app_config("TEST"),
                date(2024, 1, 2),
                date(2024, 1, 4),
                benchmark_symbol="TEST",
            )
        self.assertEqual(result.order_rows[0]["status"], "filled")
        self.assertEqual(result.order_rows[0]["fill_price"], 101.0)


if __name__ == "__main__":
    unittest.main()
