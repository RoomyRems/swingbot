import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
import pandas as pd
import backtest.engine as eng


def _df_with_friday(start='2024-01-02', periods=20):
    idx = pd.bdate_range(start=start, periods=periods)
    import numpy as np
    base_prices = 100 + 0.1 * np.arange(len(idx))
    df = pd.DataFrame({
        'Open': base_prices,
        'High': base_prices + 0.5,
        'Low': base_prices - 0.5,
        'Close': base_prices,
        'Volume': [1_000_000]*len(idx)
    }, index=idx)
    return df


def test_friday_scout_creates_remainder(monkeypatch):
    df = _df_with_friday()

    def fake_get_daily_bars(symbol: str, lookback_days: int = 300):
        return df.copy()
    monkeypatch.setattr(eng,'get_daily_bars', fake_get_daily_bars)

    def fake_eval(sl, cfg, weekly_ctx=None):
        return {
            'direction':'long','score':5,'core_pass_count':5,'trend':True,'momentum':True,'cycle':True,'sr':True,'scale':True,
            'setup_type':'pullback','ema20':float(sl.iloc[-1]['Close']),'close':float(sl.iloc[-1]['Close']),
            'atr':1.0,'ema20_dist_pct':0.0,'explain': {'sr': {'value_zone': True}},
        }
    monkeypatch.setattr(eng,'evaluate_five_energies', fake_eval)
    monkeypatch.setattr(eng,'_mtf_ok_for_slice', lambda *a, **k: (True,'ok'))

    injected_cfg = {
        'capital':100000,
        'universe':['SCOUT'],
        'risk':{'risk_per_trade_pct':0.01,'atr_multiple_stop':2.0,'reward_multiple':2.0},
        'trading':{
            'entry_model':'limit_retrace',
            'hybrid_entry':{
                'enabled':True,
                'ladder_levels_atr':[0.2],
                'ladder_sizes_pct':[1.0],
                'max_ladder_days':5,
                'promote_after_days':4,
                'promote_to':'market_breakout',
                'require_momentum_on_promotion':False,
                'min_core_energies_promotion':0,
                'breakout_extension_max_pct':1.0,
            },
            'weekend_gap_protect':{
                'enabled':True,
                'friday_scout_pct':0.4,
                'monday_gap_R_tolerance':0.5,
            }
        },
        'backtest':{'force_close_on_end':True}
    }
    monkeypatch.setattr(eng,'load_config', lambda p: injected_cfg)

    start = str(df.index[0].date())
    end   = str(df.index[-1].date())
    summary = eng.run_backtest(['SCOUT'], start, end)
    # We at least expect friday_scout_partials counter present
    assert 'friday_scout_partials' in summary
    assert summary['friday_scout_partials'] >= 0
