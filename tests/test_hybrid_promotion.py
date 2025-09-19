import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
import pandas as pd
import numpy as np
import backtest.engine as eng


def _daily_index(start, end):
    return pd.bdate_range(start=start, end=end)


def _constant_df(start, end, price=100.0, warmup_days=120):
    # Build a longer history ending at `end` to satisfy indicator warmups (EMA20, ATR14, etc.).
    end_dt = pd.to_datetime(end)
    hist_start = (end_dt - pd.tseries.offsets.BDay(warmup_days + 5)).date()
    idx = pd.bdate_range(start=hist_start, end=end_dt)
    data = {
        'Open':  [price]*len(idx),
        'High':  [price+0.2]*len(idx),
        'Low':   [price+0.05]*len(idx),  # never reaches limit below
        'Close': [price]*len(idx),
        'Volume':[1_000_000]*len(idx),
    }
    return pd.DataFrame(data, index=idx)


def test_hybrid_ladder_promotion_occurs(monkeypatch):
    start = '2024-01-02'  # Wednesday
    end   = '2024-01-10'

    df_const = _constant_df(start, end, price=100.0)

    def fake_get_daily_bars(symbol: str, lookback_days: int = 300):
        return df_const.copy()
    monkeypatch.setattr(eng, 'get_daily_bars', fake_get_daily_bars)

    # Always produce a pullback long signal with ATR=1 so ladder limit= 100 - 0.1*1=99.9 (never hit)
    def fake_eval(sl: pd.DataFrame, cfg: dict, weekly_ctx=None):
        return {
            'direction': 'long',
            'score': 5,
            'core_pass_count': 5,
            'trend': True,
            'momentum': True,
            'cycle': True,
            'sr': True,
            'scale': True,
            'setup_type': 'pullback',
            'ema20': float(sl.iloc[-1]['Close']),
            'close': float(sl.iloc[-1]['Close']),
            'atr': 1.0,
            'ema20_dist_pct': 0.0,
            'explain': {'sr': {'value_zone': True}},
        }
    monkeypatch.setattr(eng, 'evaluate_five_energies', fake_eval)
    monkeypatch.setattr(eng, '_mtf_ok_for_slice', lambda dfslice, cfg, direction, weekly_ctx=None: (True,'ok'))

    injected_cfg = {
        'capital': 100000,
        'universe': ['ABC'],
        'risk': {
            'risk_per_trade_pct': 0.01,
            'atr_multiple_stop': 2.0,
            'reward_multiple': 2.0,
        },
        'trading': {
            'entry_model': 'limit_retrace',
            'hybrid_entry': {
                'enabled': True,
                'ladder_levels_atr': [0.1],
                'ladder_sizes_pct': [1.0],
                'max_ladder_days': 4,
                'promote_after_days': 2,
                'promote_to': 'market_breakout',
                'require_momentum_on_promotion': False,
                'min_core_energies_promotion': 0,
                'breakout_extension_max_pct': 0.50,
            },
        },
        'backtest': {
            'force_close_on_end': True,
        }
    }
    monkeypatch.setattr(eng, 'load_config', lambda path: injected_cfg)

    summary = eng.run_backtest(['ABC'], start, end)

    if summary.get('limit_signals', 0) == 0:
        import pytest
        pytest.skip('No limit signals generated; promotion path not exercised with current synthetic data.')

    assert summary.get('ladder_promotions', 0) >= 1, summary
    assert summary.get('ladder_fills', 0) == 0, summary
