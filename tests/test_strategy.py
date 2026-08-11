from __future__ import annotations

import unittest

import pandas as pd

from swingbot.strategy import evaluate_energies, generate_signal, prepare_indicators
from tests.helpers import make_bars


def _force_complete_setup(
    prepared: pd.DataFrame,
    timestamp: pd.Timestamp | None = None,
) -> pd.Timestamp:
    timestamp = prepared.index[-1] if timestamp is None else timestamp
    location = int(prepared.index.get_loc(timestamp))
    prior = prepared.index[location - 1]
    slope_base = prepared.index[location - 5]
    close = float(prepared.loc[timestamp, "close"])
    prepared.loc[slope_base, "sma50"] = close - 6.0
    prepared.loc[timestamp, "sma50"] = close - 5.0
    prepared.loc[timestamp, "macd"] = 1.0
    prepared.loc[prior, ["stoch_k", "stoch_d"]] = [10.0, 30.0]
    prepared.loc[timestamp, ["stoch_k", "stoch_d"]] = [15.0, 35.0]
    prepared.loc[timestamp, "ema15"] = close - 0.5
    prepared.loc[timestamp, "atr14"] = 2.0
    prepared.loc[timestamp, "low"] = close - 1.0
    prepared.loc[timestamp, "weekly_macd_hist_delta"] = 0.1
    return timestamp


class StrategyTests(unittest.TestCase):
    def test_all_five_energies_create_one_transparent_signal(self):
        prepared = prepare_indicators(make_bars())
        timestamp = _force_complete_setup(prepared)
        signal = generate_signal(
            "SPY", prepared, timestamp, max_entry_gap_r=0.25, reward_r=2.0
        )
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.score, 5)
        self.assertEqual(
            set(signal.energies), {"trend", "momentum", "cycle", "support", "scale"}
        )
        self.assertLess(signal.stop_price, signal.reference_price)
        self.assertLess(signal.reference_price, signal.entry_limit)
        self.assertLess(signal.entry_limit, signal.target_price)

    def test_one_failed_energy_rejects_signal(self):
        prepared = prepare_indicators(make_bars())
        timestamp = _force_complete_setup(prepared)
        prepared.loc[timestamp, "weekly_macd_hist_delta"] = -0.1
        evidence = evaluate_energies(prepared, timestamp)
        self.assertFalse(evidence["scale"].passed)
        self.assertIsNone(
            generate_signal(
                "SPY", prepared, timestamp, max_entry_gap_r=0.25, reward_r=2.0
            )
        )

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
        prefix_signal = generate_signal(
            "SPY", prefix, cutoff, max_entry_gap_r=0.25, reward_r=2.0
        )
        full_signal = generate_signal(
            "SPY", full, cutoff, max_entry_gap_r=0.25, reward_r=2.0
        )
        self.assertEqual(prefix_signal, full_signal)


if __name__ == "__main__":
    unittest.main()
