from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from swingbot import backtest
from swingbot.controls import simple_trend_control
from swingbot.exits import ExitPolicy
from swingbot.models import ExitReason, Position
from swingbot.paper import _covered_position_risk
from swingbot.strategy import DEFAULT_RULES, SetupAssessment, _trend_evidence
from swingbot.waves import add_wave_context
from tests.helpers import app_config, complete_signal


class ExecutionRepairTests(unittest.TestCase):
    def run_case(self, entry_bar, policy=ExitPolicy.STATIC_2R):
        frame = pd.DataFrame(
            [[99, 101, 98, 100], entry_bar, [95, 96, 85, 90]],
            columns=["open", "high", "low", "close"],
            index=pd.bdate_range("2024-01-02", periods=3),
        )
        frame["volume"] = 1_000_000.0
        signal = complete_signal("TEST", frame.index[0].date())

        def assess(symbol, prepared, timestamp, **kwargs):
            return SetupAssessment(
                signal if timestamp == frame.index[0] else None,
                dict(signal.energies),
                dict(signal.context),
            )

        with patch.object(backtest, "assess_setup", side_effect=assess):
            return backtest.run_backtest(
                {"TEST": frame},
                app_config("TEST"),
                frame.index[0].date(),
                frame.index[-1].date(),
                exit_policy=policy,
            )

    def test_intraday_breakout_target_replaces_false_next_day_loss(self):
        result = self.run_case([99, 122, 98, 121])
        trade = result.trades[0]
        self.assertEqual(trade.reason, ExitReason.TARGET)
        self.assertEqual(trade.entry_date, trade.exit_date)
        self.assertEqual(trade.pnl, 900.0)
        self.assertEqual(trade.r_multiple, 2.0)
        self.assertAlmostEqual(trade.reserved_r_multiple, 20 / 11)
        self.assertAlmostEqual(result.summary["end_equity"], 100_000 + trade.pnl)

    def test_gap_retrace_does_not_claim_earlier_high(self):
        result = self.run_case([125, 130, 99, 105])
        self.assertEqual(result.trades[0].reason, ExitReason.STOP)
        self.assertNotEqual(result.trades[0].entry_date, result.trades[0].exit_date)

    def test_gap_retrace_close_establishes_post_fill_target(self):
        result = self.run_case([125, 130, 99, 121])
        self.assertEqual(result.trades[0].reason, ExitReason.TARGET)
        self.assertEqual(result.trades[0].entry_date, result.trades[0].exit_date)

    def test_ambiguous_stop_stays_conservative_and_is_reported(self):
        result = self.run_case([99, 122, 89, 121])
        self.assertEqual(result.trades[0].reason, ExitReason.STOP)
        self.assertEqual(
            result.summary["execution_diagnostics"]["entry_stop_sequence_ambiguous"], 1
        )

    def test_fill_target_uses_filled_price_and_initial_stop(self):
        result = self.run_case([100.5, 122, 98, 121], ExitPolicy.STATIC_FILL_2R)
        self.assertEqual(result.trades[0].target_price, 121.5)
        self.assertEqual(result.trades[0].r_multiple, 2.0)


def wave_fixture():
    # Wave 1, a lower oscillator cycle, wave 3, then developing wave 5.
    close = [99, 100, 100, 99, 98, 99, 103, 104, 102, 107, 108, 106]
    frame = pd.DataFrame(
        {
            "open": close,
            "close": close,
            "high": [x + 1 for x in close],
            "low": [x - 1 for x in close],
            "stoch_d": [40, 60, 60, 40, 60, 40, 60, 60, 40, 60, 60, 40],
            "stoch_k": [30, 60, 70, 30, 60, 30, 60, 70, 30, 60, 70, 30],
            "sma50": [80 + i for i in range(12)],
        },
        index=pd.bdate_range("2024-01-02", periods=12),
    )
    return frame


class WaveRepairTests(unittest.TestCase):
    def test_lesser_cycles_do_not_count_as_new_waves_and_prefixes_are_invariant(self):
        frame = wave_fixture()
        full = add_wave_context(frame, slope_bars=1)
        self.assertEqual(
            full["wave_confirmed_impulse"].tolist(), [0, 0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 5]
        )
        self.assertEqual(full.iloc[9]["wave_active_impulse"], 5)
        for i in range(1, len(frame) + 1):
            pd.testing.assert_frame_equal(add_wave_context(frame.iloc[:i], 1), full.iloc[:i])

    def test_lower_cycle_does_not_prematurely_veto_second_retrace(self):
        frame = add_wave_context(wave_fixture(), 1)
        rules = replace(DEFAULT_RULES, trend_slope_bars=1, objective_wave_retraces=True)
        _, legacy = _trend_evidence(frame, 8, replace(rules, objective_wave_retraces=False))
        evidence, context = _trend_evidence(frame, 8, rules)
        self.assertEqual(legacy["retrace_number"], 3)
        self.assertEqual(context["retrace_number"], 2)
        self.assertTrue(evidence.passed)

    def test_first_retrace_runner_can_later_activate_fifth_wave(self):
        frame = add_wave_context(wave_fixture(), 1)
        position = Position(
            "test",
            "TEST",
            100,
            50,
            date(2024, 1, 2),
            date(2024, 1, 3),
            100,
            90,
            90,
            120,
            10,
            first_exit_taken=True,
            entry_context={"retrace_number": 1},
        )
        old = replace(position)
        backtest._observe_burns_management(old, frame, frame.index[9], frame.index[10])
        self.assertFalse(old.one_bar_mode)
        event = backtest._observe_burns_management(
            position,
            frame,
            frame.index[9],
            frame.index[10],
            ExitPolicy.BURNS_CYCLE_V2,
        )
        self.assertEqual(event["event"], "later_fifth_wave_activated")
        self.assertTrue(position.one_bar_mode)
        self.assertEqual(position.stop_price, 105.99)

    def test_trend_loss_resets_wave_state(self):
        frame = wave_fixture()
        frame.loc[frame.index[10], "sma50"] = 79
        self.assertEqual(add_wave_context(frame, 1).iloc[10]["wave_confirmed_impulse"], 0)


class ProtectiveCoverageTests(unittest.TestCase):
    def stop(self, identifier="one", quantity=100, filled=0, status="new"):
        return SimpleNamespace(
            id=identifier,
            symbol="TEST",
            side="sell",
            type="stop",
            status=status,
            qty=quantity,
            filled_qty=filled,
            stop_price=90,
        )

    def test_partial_duplicate_and_inactive_stops_cannot_fake_full_coverage(self):
        for orders in (
            [self.stop(quantity=1)],
            [self.stop(quantity=50)] * 2,
            [self.stop(filled=1)],
            [self.stop(status="canceled")],
        ):
            with self.subTest(orders=orders), self.assertRaisesRegex(RuntimeError, "full quantity"):
                _covered_position_risk("TEST", 100, 100, orders)

    def test_distinct_remaining_stop_quantities_cover_the_holding(self):
        orders = [self.stop("one", 60, 10), self.stop("two", 50)]
        self.assertEqual(_covered_position_risk("TEST", 100, 100, orders), 1000)


class TrendControlTests(unittest.TestCase):
    def test_control_acts_on_prior_close_and_pays_entry_and_liquidation_costs(self):
        index = pd.bdate_range("2024-01-02", periods=55)
        frame = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
        frame.loc[index[50] :, "close"] = 110.0
        summary, equity = simple_trend_control(
            frame, app_config("SPY", slippage_bps=10), index[49].date(), index[-1].date()
        )
        self.assertEqual(equity.iloc[1]["quantity"], 0)
        self.assertGreater(equity.iloc[2]["quantity"], 0)
        self.assertEqual(equity.iloc[-1]["quantity"], 0)
        self.assertEqual(summary["entries"], 1)
        quantity = 999
        expected = 100_000 - quantity * 100.1 + quantity * 110 * 0.999
        self.assertAlmostEqual(summary["end_equity"], expected)
