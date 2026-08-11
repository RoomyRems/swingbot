from __future__ import annotations

import unittest

import pandas as pd

from swingbot.indicators import normalize_bars
from swingbot.strategy import prepare_indicators
from tests.helpers import make_bars


class IndicatorTests(unittest.TestCase):
    def test_indicators_are_prefix_invariant(self):
        bars = make_bars()
        cutoff = bars.index[251]
        full = prepare_indicators(bars)
        prefix = prepare_indicators(bars.loc[:cutoff])
        columns = [
            "ema15",
            "sma50",
            "atr14",
            "stoch_k",
            "stoch_d",
            "macd",
            "macd_signal",
            "weekly_macd_hist",
            "weekly_macd_hist_delta",
        ]
        pd.testing.assert_series_equal(
            full.loc[cutoff, columns],
            prefix.loc[cutoff, columns],
            check_names=False,
        )

    def test_normalization_rejects_duplicate_dates(self):
        bars = make_bars(rows=5)
        duplicated = pd.concat([bars, bars.iloc[[-1]]])
        with self.assertRaisesRegex(ValueError, "duplicate dates"):
            normalize_bars(duplicated)

    def test_normalization_rejects_impossible_ohlc(self):
        bars = make_bars(rows=5)
        bars.iloc[-1, bars.columns.get_loc("high")] = 1.0
        with self.assertRaisesRegex(ValueError, "bar high"):
            normalize_bars(bars)


if __name__ == "__main__":
    unittest.main()
