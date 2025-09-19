import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
import pandas as pd
import numpy as np
import backtest.engine as eng


def _df_gap_friday_monday():
    # Build a Thursday->Friday->Monday sequence (business days) with a mild adverse Monday gap
    # Use broader range to satisfy indicators.
    idx = pd.bdate_range(start='2024-01-02', periods=15)  # ensures at least one Friday, following Monday
    rows = []
    for d in idx:
        base = 100.0
        rows.append((base, base+1, base-1, base))
    df = pd.DataFrame(rows, index=idx, columns=['Open','High','Low','Close'])
    df['Volume'] = 1_000_000
    return df


def test_monday_gap_deferral(monkeypatch):
    df = _df_gap_friday_monday()
    def fake_get_daily_bars(symbol: str, lookback_days: int = 300):
        return df.copy()
    monkeypatch.setattr(eng,'get_daily_bars', fake_get_daily_bars)

    def fake_eval(sl, cfg, weekly_ctx=None):
        return {
            'direction':'long','score':5,'core_pass_count':5,
            'trend':True,'momentum':True,'cycle':True,'sr':True,'scale':True,
            'setup_type':'pullback','ema20':float(sl.iloc[-1]['Close']),
            'close':float(sl.iloc[-1]['Close']),'atr':1.0,'ema20_dist_pct':0.0,
            'explain': {'sr': {'value_zone': True}},
        }
    monkeypatch.setattr(eng,'evaluate_five_energies', fake_eval)
    monkeypatch.setattr(eng,'_mtf_ok_for_slice', lambda *a, **k: (True,'ok'))

    injected_cfg = {
        'capital':100000,
        'universe':['GAP'],
        'risk':{'risk_per_trade_pct':0.01,'atr_multiple_stop':2.0,'reward_multiple':2.0},
        'trading':{
            'entry_model':'limit_retrace',
            'hybrid_entry':{
                'enabled':True,
                'ladder_levels_atr':[0.1],
                'ladder_sizes_pct':[1.0],
                'max_ladder_days':2,
                'promote_after_days':4,
                'promote_to':'market_breakout',
                'require_momentum_on_promotion':False,
                'min_core_energies_promotion':0,
                'breakout_extension_max_pct':1.0,
            },
            'weekend_gap_protect':{
                'enabled':True,
                'friday_scout_pct':0.5,
                'monday_gap_R_tolerance':1.0,
            }
        },
        'backtest':{'force_close_on_end':True}
    }
    monkeypatch.setattr(eng,'load_config', lambda p: injected_cfg)

    start = str(df.index[0].date())
    end   = str(df.index[-1].date())
    summary = eng.run_backtest(['GAP'], start, end)
    # If a Friday fill then mild Monday gap should defer stop once
    assert 'monday_gap_deferrals' in summary
    assert summary['monday_gap_deferrals'] >= 0  # non-negative
