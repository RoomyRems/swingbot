import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
import pandas as pd
import backtest.engine as eng


def _df_extending(start, end, base=100.0, step=2.0):
    idx = pd.bdate_range(start=start, end=end)
    rows = []
    price = base
    for i, d in enumerate(idx):
        price = base + i*step
        rows.append((price, price+0.5, price-0.5, price))
    df = pd.DataFrame(rows, index=idx, columns=['Open','High','Low','Close'])
    df['Volume'] = 1_000_000
    return df


def test_promotion_blocked_by_extension(monkeypatch):
    start='2024-01-02'
    end='2024-01-12'
    df = _df_extending(start, end)

    def fake_get_daily_bars(symbol: str, lookback_days: int = 300):
        return df.copy()
    monkeypatch.setattr(eng, 'get_daily_bars', fake_get_daily_bars)

    # Limit below price so never hit; extension grows fast above original signal close
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
        'universe':['XYZ'],
        'risk':{'risk_per_trade_pct':0.01,'atr_multiple_stop':2.0,'reward_multiple':2.0},
        'trading':{
            'entry_model':'limit_retrace',
            'hybrid_entry':{
                'enabled':True,
                'ladder_levels_atr':[0.1],
                'ladder_sizes_pct':[1.0],
                'max_ladder_days':5,
                'promote_after_days':2,
                'promote_to':'market_breakout',
                'require_momentum_on_promotion':False,
                'min_core_energies_promotion':0,
                'breakout_extension_max_pct':0.02, # 2% cap forces block
            }
        },
        'backtest':{'force_close_on_end':True}
    }
    monkeypatch.setattr(eng,'load_config', lambda p: injected_cfg)

    summary = eng.run_backtest(['XYZ'], start, end)
    # Expect no promotion because price extended beyond 2%
    assert summary.get('ladder_promotions',0) == 0, summary
