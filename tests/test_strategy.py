from __future__ import annotations

import math
import unittest
from dataclasses import replace

import pandas as pd

from swingbot.strategy import (
    DEFAULT_RULES,
    assess_signal,
    evaluate_energies,
    generate_signal,
    prepare_indicators,
    strategy_fingerprint,
)
from tests.helpers import make_bars


def _force_complete_setup(
    prepared: pd.DataFrame,
    timestamp: pd.Timestamp | None = None,
) -> pd.Timestamp:
    timestamp = prepared.index[-1] if timestamp is None else timestamp
    location = int(prepared.index.get_loc(timestamp))
    reference = float(prepared.loc[timestamp, "close"])

    # Create one recent, uninterrupted rising 50-SMA epoch.
    prepared["sma50"] = reference - 12.0
    trend_start = location - 20
    for offset, position in enumerate(range(trend_start, location + 1)):
        prepared.iloc[position, prepared.columns.get_loc("sma50")] = reference - 12.0 + 0.1 * offset

    # Create one below-50 stochastic excursion with two %K troughs. Price
    # makes a lower low at the second trough while %K makes a higher low.
    prepared["stoch_k"] = 70.0
    prepared["stoch_d"] = 70.0
    d_values = [45.0, 42.0, 39.0, 36.0, 33.0, 30.0, 28.0, 32.0]
    k_values = [35.0, 15.0, 30.0, 28.0, 26.0, 24.0, 20.0, 25.0]
    cycle_positions = list(range(location - 7, location + 1))
    prepared.iloc[cycle_positions, prepared.columns.get_loc("stoch_d")] = d_values
    prepared.iloc[cycle_positions, prepared.columns.get_loc("stoch_k")] = k_values

    prepared.iloc[cycle_positions, prepared.columns.get_loc("low")] = reference - 0.2
    first_trough = location - 6
    second_trough = location - 1
    prepared.iloc[first_trough, prepared.columns.get_loc("low")] = reference - 1.0
    prepared.iloc[second_trough, prepared.columns.get_loc("low")] = reference - 1.5
    prepared.iloc[location, prepared.columns.get_loc("low")] = reference - 1.0
    prepared.iloc[location, prepared.columns.get_loc("high")] = reference + 1.0

    prepared.iloc[second_trough, prepared.columns.get_loc("ema15")] = reference - 1.5
    prepared.iloc[second_trough, prepared.columns.get_loc("atr14")] = 2.0
    prepared.iloc[second_trough, prepared.columns.get_loc("macd")] = 1.0
    prepared.iloc[location, prepared.columns.get_loc("weekly_macd_delta")] = 0.1
    prepared.iloc[location, prepared.columns.get_loc("adv90")] = 1_000_000.0
    return timestamp


class StrategyTests(unittest.TestCase):
    def test_assessment_reuses_the_signal_energy_evidence(self):
        prepared = prepare_indicators(make_bars())
        timestamp = _force_complete_setup(prepared)
        assessed, energies = assess_signal(
            "SPY",
            prepared,
            timestamp,
            max_entry_gap_r=0.25,
            reward_r=2.0,
        )
        generated = generate_signal(
            "SPY",
            prepared,
            timestamp,
            max_entry_gap_r=0.25,
            reward_r=2.0,
        )
        self.assertEqual(assessed, generated)
        assert assessed is not None
        self.assertIs(assessed.energies, energies)

    def test_all_five_energies_create_one_transparent_signal(self):
        prepared = prepare_indicators(make_bars())
        timestamp = _force_complete_setup(prepared)
        signal = generate_signal("SPY", prepared, timestamp, max_entry_gap_r=0.25, reward_r=2.0)
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.score, 5)
        self.assertEqual(set(signal.energies), {"trend", "momentum", "cycle", "support", "scale"})
        self.assertEqual(signal.context["retrace_number"], 1)
        self.assertTrue(signal.context["mini_divergence"])
        self.assertEqual(signal.context["support_source"], "15-EMA")
        self.assertLess(signal.stop_price, signal.reference_price)
        self.assertLess(signal.reference_price, signal.entry_stop)
        self.assertLess(signal.entry_stop, signal.entry_limit)
        self.assertLess(signal.entry_limit, signal.target_price)

    def test_setup_momentum_may_be_the_one_missing_energy(self):
        prepared = prepare_indicators(make_bars())
        timestamp = _force_complete_setup(prepared)
        location = int(prepared.index.get_loc(timestamp))
        prepared.iloc[location - 1, prepared.columns.get_loc("macd")] = -0.1
        signal = generate_signal("SPY", prepared, timestamp, max_entry_gap_r=0.25, reward_r=2.0)
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.score, 4)
        self.assertFalse(signal.energies["momentum"].passed)

    def test_setup_support_may_be_the_one_missing_energy(self):
        prepared = prepare_indicators(make_bars())
        timestamp = _force_complete_setup(prepared)
        location = int(prepared.index.get_loc(timestamp))
        active_low = location - 1
        prepared.iloc[active_low, prepared.columns.get_loc("ema15")] -= 20.0
        prepared.iloc[active_low, prepared.columns.get_loc("sma50")] -= 20.0
        prepared.iloc[active_low, prepared.columns.get_loc("atr14")] = 0.1
        signal = generate_signal("SPY", prepared, timestamp, max_entry_gap_r=0.25, reward_r=2.0)
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.score, 4)
        self.assertFalse(signal.energies["support"].passed)

    def test_momentum_and_support_cannot_both_be_missing(self):
        prepared = prepare_indicators(make_bars())
        timestamp = _force_complete_setup(prepared)
        location = int(prepared.index.get_loc(timestamp))
        active_low = location - 1
        prepared.iloc[active_low, prepared.columns.get_loc("macd")] = -0.1
        prepared.iloc[active_low, prepared.columns.get_loc("ema15")] -= 20.0
        prepared.iloc[active_low, prepared.columns.get_loc("sma50")] -= 20.0
        prepared.iloc[active_low, prepared.columns.get_loc("atr14")] = 0.1
        self.assertIsNone(
            generate_signal("SPY", prepared, timestamp, max_entry_gap_r=0.25, reward_r=2.0)
        )

    def test_scale_is_a_mandatory_veto(self):
        prepared = prepare_indicators(make_bars())
        timestamp = _force_complete_setup(prepared)
        prepared.loc[timestamp, "weekly_macd_delta"] = -0.1
        evidence = evaluate_energies(prepared, timestamp)
        self.assertFalse(evidence["scale"].passed)
        self.assertIsNone(
            generate_signal("SPY", prepared, timestamp, max_entry_gap_r=0.25, reward_r=2.0)
        )

    def test_scale_uses_macd_line_slope_not_histogram_acceleration(self):
        prepared = prepare_indicators(make_bars())
        timestamp = _force_complete_setup(prepared)
        prepared.loc[timestamp, "weekly_macd_delta"] = -0.1
        prepared.loc[timestamp, "weekly_macd_hist_delta"] = 10.0
        evidence = evaluate_energies(prepared, timestamp)
        self.assertFalse(evidence["scale"].passed)

    def test_cycle_requires_price_k_mini_divergence(self):
        prepared = prepare_indicators(make_bars())
        timestamp = _force_complete_setup(prepared)
        location = int(prepared.index.get_loc(timestamp))
        prepared.iloc[location - 1, prepared.columns.get_loc("stoch_k")] = 10.0
        evidence = evaluate_energies(prepared, timestamp)
        self.assertFalse(evidence["cycle"].passed)
        self.assertIsNone(
            generate_signal("SPY", prepared, timestamp, max_entry_gap_r=0.25, reward_r=2.0)
        )

    def test_third_retrace_is_not_an_early_trend_entry(self):
        prepared = prepare_indicators(make_bars())
        timestamp = _force_complete_setup(prepared)
        location = int(prepared.index.get_loc(timestamp))
        d_column = prepared.columns.get_loc("stoch_d")
        prepared.iloc[location - 18 : location - 16, d_column] = 40.0
        prepared.iloc[location - 16 : location - 14, d_column] = 70.0
        prepared.iloc[location - 14 : location - 12, d_column] = 40.0
        prepared.iloc[location - 12 : location - 7, d_column] = 70.0
        evidence = evaluate_energies(prepared, timestamp)
        self.assertFalse(evidence["trend"].passed)

    def test_second_retrace_is_still_an_early_trend_entry(self):
        prepared = prepare_indicators(make_bars())
        timestamp = _force_complete_setup(prepared)
        location = int(prepared.index.get_loc(timestamp))
        d_column = prepared.columns.get_loc("stoch_d")
        prepared.iloc[location - 16 : location - 14, d_column] = 40.0
        prepared.iloc[location - 14 : location - 7, d_column] = 70.0
        signal = generate_signal("SPY", prepared, timestamp, max_entry_gap_r=0.25, reward_r=2.0)
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.context["retrace_number"], 2)

    def test_stop_and_entry_use_cycle_low_and_hook_bar_high(self):
        prepared = prepare_indicators(make_bars())
        timestamp = _force_complete_setup(prepared)
        signal = generate_signal("SPY", prepared, timestamp, max_entry_gap_r=0.25, reward_r=2.0)
        assert signal is not None
        cycle_low = float(signal.context["cycle_low_price"])
        hook_high = float(prepared.loc[timestamp, "high"])
        expected_stop = math.floor((cycle_low - 0.01 + 1e-12) / 0.01) * 0.01
        self.assertEqual(signal.stop_price, round(expected_stop, 2))
        expected_entry = math.ceil((hook_high + 0.01 - 1e-12) / 0.01) * 0.01
        self.assertEqual(signal.entry_stop, round(expected_entry, 2))

    def test_missing_as_of_bar_fails_loudly(self):
        prepared = prepare_indicators(make_bars())
        with self.assertRaisesRegex(ValueError, "no bar exists"):
            evaluate_energies(prepared, "1999-01-01")

    def test_signal_at_cutoff_is_unchanged_when_future_bars_are_appended(self):
        bars = make_bars()
        cutoff = bars.index[260]
        prefix = prepare_indicators(bars.loc[:cutoff])
        full = prepare_indicators(bars)
        _force_complete_setup(prefix, cutoff)
        _force_complete_setup(full, cutoff)
        prefix_signal = generate_signal("SPY", prefix, cutoff, max_entry_gap_r=0.25, reward_r=2.0)
        full_signal = generate_signal("SPY", full, cutoff, max_entry_gap_r=0.25, reward_r=2.0)
        self.assertEqual(prefix_signal, full_signal)

    def test_strategy_fingerprint_changes_with_a_frozen_rule(self):
        changed = replace(DEFAULT_RULES, support_atr_tolerance=0.30)
        self.assertEqual(strategy_fingerprint(), strategy_fingerprint())
        self.assertNotEqual(strategy_fingerprint(), strategy_fingerprint(changed))


if __name__ == "__main__":
    unittest.main()
