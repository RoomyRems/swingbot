import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def _synthetic_daily_df(days: int) -> pd.DataFrame:
    # Build a simple trending series with enough bars for indicators
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=max(days, 260))
    base = np.linspace(50, 60, len(idx))
    close = base + np.sin(np.linspace(0, 10, len(idx)))
    open_ = close * (1 + np.random.default_rng(0).normal(0, 0.001, len(idx)))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    vol = np.full(len(idx), 1_000_000)
    df = pd.DataFrame({
        "Open": open_.round(2),
        "High": high.round(2),
        "Low": low.round(2),
        "Close": close.round(2),
        "Volume": vol,
    }, index=idx)
    return df


def test_backtest_smoke_executes_quickly(monkeypatch):
    # Import here to ensure monkeypatching module-level symbols works
    import backtest.engine as eng

    # 1) Monkeypatch data fetch to avoid network and be fast
    def fake_get_daily_bars(symbol: str, lookback_days: int = 300):
        return _synthetic_daily_df(lookback_days)

    monkeypatch.setattr(eng, "get_daily_bars", fake_get_daily_bars)

    # 2) Monkeypatch evaluator to always accept simple long signals
    def fake_eval(sl: pd.DataFrame, cfg: dict, weekly_ctx=None):
        return {
            "direction": "long",
            "score": 5,
            "core_pass_count": 5,
            "trend": True,
            "momentum": True,
            "cycle": True,
            "sr": True,
            "scale": True,
            "setup_type": "pullback",
            "explain": {"sr": {"value_zone": True}},
        }

    monkeypatch.setattr(eng, "evaluate_five_energies", fake_eval)

    # 3) Monkeypatch MTF check to always OK
    def fake_mtf_ok(dfslice, cfg, direction, weekly_ctx=None):
        return True, "mtf-ok"

    monkeypatch.setattr(eng, "_mtf_ok_for_slice", fake_mtf_ok)

    # 4) Provide a minimal config via monkeypatch (avoid reading file)
    def fake_load_config(_):
        return {
            "trading": {
                "min_score": 1,
                "min_core_energies": 1,
                "mtf": {"enabled": False},
            },
            "risk": {
                "atr_multiple_stop": 1.4,
                "reward_multiple": 1.8,
                "max_total_risk_pct": 0.5,
                "account_equity": 20000,
            },
            "backtest": {
                "initial_equity": 20000,
                "max_open_positions": 5,
                "commission_per_share": 0.0,
                "slippage_bps": 0.0,
                "force_close_on_end": True,
                "entry_model": {"type": "market"},
                "evidence_exit": {"enabled": True},
            },
        }

    monkeypatch.setattr(eng, "load_config", fake_load_config)

    # 5) Run a tiny backtest window with 2 symbols
    start = (pd.Timestamp.today() - pd.Timedelta(days=40)).normalize()
    end = pd.Timestamp.today().normalize()
    summary = eng.run_backtest(["AAA", "BBB"], start, end, cfg_path="ignored.yaml")

    # Smoke assertions: engine returned a summary with basic fields and did not crash
    assert isinstance(summary, dict)
    for k in ("start_equity", "end_equity", "trades", "profit_factor", "files"):
        assert k in summary
    # Ensure CSVs were written paths are present in summary["files"]
    assert isinstance(summary["files"], list)
    # Trades can be zero or more; most importantly, the engine executed.
