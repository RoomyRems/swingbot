from __future__ import annotations

import unittest

import pandas as pd

from swingbot.indicators import normalize_bars, slow_stochastic
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
            "adv90",
            "stoch_k",
            "stoch_d",
            "macd",
            "macd_signal",
            "weekly_macd_hist",
            "weekly_macd_delta",
            "weekly_macd_hist_delta",
        ]
        pd.testing.assert_series_equal(
            full.loc[cutoff, columns],
            prefix.loc[cutoff, columns],
            check_names=False,
        )

    def test_burns_stochastic_d_is_an_exponential_average(self):
        close = pd.Series([10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.0, 14.0, 13.0])
        high = close + 1.0
        low = close - 1.0
        k, d = slow_stochastic(high, low, close)
        expected = k.ewm(span=3, adjust=False, min_periods=3).mean()
        pd.testing.assert_series_equal(d, expected)

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
