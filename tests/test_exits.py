from __future__ import annotations

import unittest

import pandas as pd

from swingbot.exits import (
    ExitPolicy,
    exit_policy_fingerprint,
    long_cycle_high_turn,
    long_cycle_low_turn,
    stop_below,
)


def _prepared() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=9)
    return pd.DataFrame(
        {
            "open": [99, 100, 102, 106, 105, 103, 101, 102, 104],
            "high": [101, 102, 104, 108, 107, 105, 103, 104, 106],
            "low": [98, 99, 101, 105, 103, 101, 99, 100, 102],
            "close": [100, 101, 103, 107, 104, 102, 100, 103, 105],
            "stoch_k": [30, 40, 60, 80, 70, 45, 25, 35, 60],
            "stoch_d": [30, 40, 60, 70, 65, 45, 35, 40, 60],
        },
        index=index,
    )


class ExitSignalTests(unittest.TestCase):
    def test_cycle_high_is_confirmed_only_by_a_closed_k_turn(self):
        prepared = _prepared()
        before_turn = long_cycle_high_turn(prepared, prepared.index[3])
        turn = long_cycle_high_turn(prepared, prepared.index[4], previous_wave_high=105.0)
        self.assertIsNone(before_turn)
        assert turn is not None
        self.assertEqual(turn.extreme_price, 108.0)
        self.assertEqual(turn.extreme_date, prepared.index[3].date())
        self.assertEqual(turn.signal_date, prepared.index[4].date())
        self.assertTrue(turn.wave_breakout)

    def test_cycle_low_uses_only_the_active_closed_interval(self):
        prepared = _prepared()
        turn = long_cycle_low_turn(prepared, prepared.index[7])
        assert turn is not None
        self.assertEqual(turn.interval_start_date, prepared.index[5].date())
        self.assertEqual(turn.extreme_price, 99.0)
        self.assertEqual(turn.extreme_date, prepared.index[6].date())

    def test_exit_turn_at_cutoff_is_prefix_invariant(self):
        prepared = _prepared()
        cutoff = prepared.index[4]
        prefix = long_cycle_high_turn(prepared.loc[:cutoff], cutoff)
        full = long_cycle_high_turn(prepared, cutoff)
        self.assertEqual(prefix, full)

    def test_stop_is_one_tradable_tick_below_and_fingerprinted(self):
        self.assertEqual(stop_below(100.0), 99.99)
        self.assertEqual(stop_below(0.5001), 0.5)
        self.assertNotEqual(
            exit_policy_fingerprint(ExitPolicy.STATIC_2R),
            exit_policy_fingerprint(ExitPolicy.BURNS_CYCLE_V1),
        )


if __name__ == "__main__":
    unittest.main()
