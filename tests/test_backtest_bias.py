from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date
from unittest.mock import patch

import pandas as pd

import swingbot.backtest as backtest_module
from swingbot.backtest import run_backtest
from swingbot.exits import ExitPolicy
from swingbot.models import ExitReason
from swingbot.strategy import SetupAssessment
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
    def _fake_assess(self, symbol, prepared, as_of, **kwargs):
        signal = complete_signal("TEST", date(2024, 1, 2))
        candidate = signal if pd.Timestamp(as_of).date() == signal.signal_date else None
        return SetupAssessment(candidate, dict(signal.energies), dict(signal.context))

    def test_next_bar_close_cannot_revalidate_or_change_entry(self):
        config = app_config("TEST")
        with patch.object(backtest_module, "assess_setup", side_effect=self._fake_assess):
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
        self.assertEqual(low_close.summary["strategy_version"], "burns-book-v2")
        self.assertEqual(len(low_close.summary["strategy_fingerprint"]), 64)

    def test_ambiguous_same_day_stop_and_target_uses_stop(self):
        frame = _three_day_frame(100.0, second_high=130.0, second_low=89.0)
        with patch.object(backtest_module, "assess_setup", side_effect=self._fake_assess):
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
        with patch.object(backtest_module, "assess_setup", side_effect=self._fake_assess):
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
        with patch.object(backtest_module, "assess_setup", side_effect=self._fake_assess):
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
        with patch.object(backtest_module, "assess_setup", side_effect=self._fake_assess):
            result = run_backtest(
                {"TEST": frame},
                app_config("TEST"),
                date(2024, 1, 2),
                date(2024, 1, 4),
                benchmark_symbol="TEST",
            )
        self.assertEqual(result.order_rows[0]["status"], "filled")
        self.assertEqual(result.order_rows[0]["fill_price"], 101.0)

    def test_burns_partial_uses_next_open_and_then_trails_on_closed_cycle_low(self):
        dates = pd.bdate_range("2024-01-02", periods=10)
        frame = pd.DataFrame(
            {
                "open": [99, 100, 104, 106, 105, 104, 103, 105, 107, 101],
                "high": [101, 103, 106, 108, 107, 106, 105, 107, 109, 102],
                "low": [98, 95, 102, 104, 103, 102, 100, 103, 105, 98],
                "close": [100, 102, 105, 107, 104, 103, 104, 106, 108, 99],
                "volume": [1_000_000.0] * 10,
            },
            index=dates,
        )

        def fake_prepare(source):
            prepared = source.copy()
            prepared["stoch_k"] = [30, 40, 60, 80, 70, 40, 30, 35, 65, 55]
            prepared["stoch_d"] = [30, 40, 60, 70, 65, 45, 35, 40, 60, 55]
            return prepared

        signal_date = dates[0].date()

        def fake_assess(symbol, prepared, as_of, **kwargs):
            signal = complete_signal("TEST", signal_date)
            candidate = signal if pd.Timestamp(as_of).date() == signal_date else None
            return SetupAssessment(candidate, dict(signal.energies), dict(signal.context))

        with (
            patch.object(backtest_module, "prepare_indicators", side_effect=fake_prepare),
            patch.object(backtest_module, "assess_setup", side_effect=fake_assess),
        ):
            result = run_backtest(
                {"TEST": frame},
                app_config("TEST"),
                dates[0].date(),
                dates[-1].date(),
                benchmark_symbol="TEST",
                exit_policy=ExitPolicy.BURNS_CYCLE_V1,
            )

        self.assertEqual(
            [row["reason"] for row in result.exit_rows],
            [
                ExitReason.CYCLE_HIGH_PARTIAL.value,
                ExitReason.CYCLE_LOW_TRAIL.value,
            ],
        )
        self.assertEqual(result.exit_rows[0]["exit_date"], dates[5].date().isoformat())
        self.assertEqual(result.exit_rows[0]["price"], 104.0)
        self.assertEqual(result.exit_rows[1]["exit_date"], dates[9].date().isoformat())
        self.assertEqual(result.trades[0].exit_legs, 2)
        self.assertEqual(result.summary["exit_policy"], ExitPolicy.BURNS_CYCLE_V1.value)

    def test_burns_manager_never_awards_the_historical_cycle_peak(self):
        dates = pd.bdate_range("2024-01-02", periods=6)
        frame = pd.DataFrame(
            {
                "open": [99, 100, 104, 107, 101, 100],
                "high": [101, 103, 106, 110, 103, 102],
                "low": [98, 95, 102, 105, 99, 98],
                "close": [100, 102, 105, 109, 101, 99],
                "volume": [1_000_000.0] * 6,
            },
            index=dates,
        )

        def fake_prepare(source):
            prepared = source.copy()
            prepared["stoch_k"] = [30, 40, 60, 85, 70, 40]
            prepared["stoch_d"] = [30, 40, 60, 75, 70, 45]
            return prepared

        def fake_assess(symbol, prepared, as_of, **kwargs):
            signal = complete_signal("TEST", dates[0].date())
            candidate = signal if pd.Timestamp(as_of).date() == signal.signal_date else None
            return SetupAssessment(candidate, dict(signal.energies), dict(signal.context))

        with (
            patch.object(backtest_module, "prepare_indicators", side_effect=fake_prepare),
            patch.object(backtest_module, "assess_setup", side_effect=fake_assess),
        ):
            result = run_backtest(
                {"TEST": frame},
                app_config("TEST"),
                dates[0].date(),
                dates[-1].date(),
                benchmark_symbol="TEST",
                exit_policy=ExitPolicy.BURNS_CYCLE_V1,
            )

        partial = next(row for row in result.exit_rows if row["reason"] == "cycle_high_partial")
        self.assertEqual(partial["price"], 100.0)
        self.assertNotEqual(partial["price"], 110.0)

    def test_scheduled_partial_precedes_a_later_same_day_initial_stop(self):
        dates = pd.bdate_range("2024-01-02", periods=6)
        frame = pd.DataFrame(
            {
                "open": [99, 100, 104, 107, 100, 100],
                "high": [101, 103, 106, 110, 103, 102],
                "low": [98, 95, 102, 105, 99, 89],
                "close": [100, 102, 105, 109, 101, 91],
                "volume": [1_000_000.0] * 6,
            },
            index=dates,
        )

        def fake_prepare(source):
            prepared = source.copy()
            prepared["stoch_k"] = [30, 40, 60, 85, 70, 40]
            prepared["stoch_d"] = [30, 40, 60, 75, 70, 45]
            return prepared

        def fake_assess(symbol, prepared, as_of, **kwargs):
            signal = complete_signal("TEST", dates[0].date())
            candidate = signal if pd.Timestamp(as_of).date() == signal.signal_date else None
            return SetupAssessment(candidate, dict(signal.energies), dict(signal.context))

        with (
            patch.object(backtest_module, "prepare_indicators", side_effect=fake_prepare),
            patch.object(backtest_module, "assess_setup", side_effect=fake_assess),
        ):
            result = run_backtest(
                {"TEST": frame},
                app_config("TEST"),
                dates[0].date(),
                dates[-1].date(),
                benchmark_symbol="TEST",
                exit_policy=ExitPolicy.BURNS_CYCLE_V1,
            )

        self.assertEqual(
            [row["reason"] for row in result.exit_rows],
            [ExitReason.CYCLE_HIGH_PARTIAL.value, ExitReason.STOP.value],
        )
        self.assertEqual(result.exit_rows[0]["exit_date"], dates[5].date().isoformat())
        self.assertEqual(result.exit_rows[1]["exit_date"], dates[5].date().isoformat())
        self.assertEqual(result.trades[0].exit_legs, 2)

    def test_fifth_wave_stop_can_protect_the_runner_on_partial_day(self):
        dates = pd.bdate_range("2024-01-02", periods=6)
        frame = pd.DataFrame(
            {
                "open": [99, 100, 104, 107, 101, 100],
                "high": [101, 103, 106, 110, 103, 102],
                "low": [98, 95, 102, 105, 99, 98],
                "close": [100, 102, 105, 109, 101, 99],
                "volume": [1_000_000.0] * 6,
            },
            index=dates,
        )

        def fake_prepare(source):
            prepared = source.copy()
            prepared["stoch_k"] = [30, 40, 60, 85, 70, 40]
            prepared["stoch_d"] = [30, 40, 60, 75, 70, 45]
            return prepared

        def fake_assess(symbol, prepared, as_of, **kwargs):
            base = complete_signal("TEST", dates[0].date())
            signal = replace(
                base,
                context={"retrace_number": 2, "previous_cycle_high_price": 105.0},
            )
            candidate = signal if pd.Timestamp(as_of).date() == signal.signal_date else None
            return SetupAssessment(candidate, dict(signal.energies), dict(signal.context))

        with (
            patch.object(backtest_module, "prepare_indicators", side_effect=fake_prepare),
            patch.object(backtest_module, "assess_setup", side_effect=fake_assess),
        ):
            result = run_backtest(
                {"TEST": frame},
                app_config("TEST"),
                dates[0].date(),
                dates[-1].date(),
                benchmark_symbol="TEST",
                exit_policy=ExitPolicy.BURNS_CYCLE_V1,
            )

        self.assertEqual(
            [row["reason"] for row in result.exit_rows],
            [ExitReason.CYCLE_HIGH_PARTIAL.value, ExitReason.ONE_BAR_TRAIL.value],
        )
        self.assertEqual(result.exit_rows[0]["exit_date"], result.exit_rows[1]["exit_date"])
        self.assertEqual(result.trades[0].exit_legs, 2)


if __name__ == "__main__":
    unittest.main()
