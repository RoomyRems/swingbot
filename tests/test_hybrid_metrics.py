import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
import pandas as pd
import numpy as np
import backtest.engine as eng


def _df_simple(periods=40):
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    base = np.linspace(50, 52, len(idx))
    close = base + 0.1*np.sin(np.linspace(0, 6, len(idx)))
    open_ = close
    high = close + 0.3
    low = close - 0.3
    vol = np.full(len(idx), 1_000_000)
    df = pd.DataFrame({
        'Open': open_.round(2),
        'High': high.round(2),
        'Low': low.round(2),
        'Close': close.round(2),
        'Volume': vol,
    }, index=idx)
    return df


def test_hybrid_summary_keys_presence(monkeypatch):
    def fake_get_daily_bars(symbol: str, lookback_days: int = 300):
        return _df_simple()
    monkeypatch.setattr(eng, "get_daily_bars", fake_get_daily_bars)

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

    monkeypatch.setattr(eng, '_mtf_ok_for_slice', lambda dfslice, cfg, direction, weekly_ctx=None: (True, 'ok'))

    injected_cfg = {
        'capital': 100000,
        'universe': ['TEST'],
        'risk': {
            'risk_per_trade_pct': 0.01,
            'atr_multiple_stop': 2.0,
            'reward_multiple': 2.0,
        },
        'trading': {
            'entry_model': 'limit_retrace',
            'hybrid_entry': {
                'enabled': True,
                'ladder_levels_atr': [0.5],
                'ladder_sizes_pct': [1.0],
                'max_ladder_days': 3,
                'promote_after_days': 2,
                'promote_to': 'market_breakout',
                'require_momentum_on_promotion': False,
                'min_core_energies_promotion': 0,
                'breakout_extension_max_pct': 1.0,
            },
            'weekend_gap_protect': {
                'enabled': True,
                'friday_scout_pct': 0.5,
                'monday_gap_R_tolerance': 0.5,
            },
        },
        'backtest': {
            'force_close_on_end': True,
        }
    }
    monkeypatch.setattr(eng, 'load_config', lambda path: injected_cfg)

    start = str(_df_simple().index[0].date())
    end = str(_df_simple().index[-1].date())

    summary = eng.run_backtest(['TEST'], start, end)

    for key in ['ladder_fills', 'ladder_promotions', 'friday_scout_partials', 'monday_gap_deferrals']:
        assert key in summary, f"Missing hybrid metric '{key}' in summary"
        val = summary.get(key)
        assert isinstance(val, int) and val >= 0
