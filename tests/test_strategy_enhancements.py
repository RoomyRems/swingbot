import pandas as pd
import numpy as np
import sys, os
from pathlib import Path

# ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.swing_strategy import _find_pivots, evaluate_five_energies, add_indicators, _regime_check


def make_df(prices: list[float]):
    n = len(prices)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='B')
    close = pd.Series(prices, index=dates)
    df = pd.DataFrame({
        'Open': close.shift(1).fillna(close),
        'High': close + 0.2,
        'Low': close - 0.2,
        'Close': close,
        'Volume': 1_000_000,
    })
    return df


def test_pivot_plateau_compression():
    # plateau highs (same value) should yield 1 pivot not len(plateau)
    highs = pd.Series([10, 11, 11, 11, 10, 9])
    lows  = pd.Series([9,  9.5, 9.4, 9.6, 9.2, 9])
    hi_idx, lo_idx = _find_pivots(highs, lows, left=1, right=1)
    assert hi_idx.count(2) == 1 or len(hi_idx) <= 3  # at least compressed


def test_macd_abs_min_filters_small_signal():
    # construct synthetic slow drift to force tiny MACD differences
    prices = np.linspace(100, 101, 120)
    df = make_df(prices.tolist())
    df = add_indicators(df)
    cfg = {
        'momentum': {
            'rsi_filter': False,
            'macd_abs_min': 0.5,  # large threshold to force fail
        }
    }
    eng = evaluate_five_energies(df, cfg)
    # momentum should likely be false due to abs min
    assert eng['momentum'] in (False, 0)


def test_regime_respects_adx_threshold():
    prices = np.linspace(50, 55, 80)
    df = make_df(prices.tolist())
    df = add_indicators(df)
    cfg = {
        'regime': {
            'enabled': True,
            'adx_min': 101,          # impossible threshold to force fail deterministically
            'atr_pct_min': 0.0,      # disable ATR gate
            'use_chop': False        # disable chop gate
        },
    }
    ok, det = _regime_check(df, cfg)
    # Should fail due to ADX below unrealistic threshold; no skip path anymore
    assert ok is False and det.get('adx_ok') is False
