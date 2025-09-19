import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from risk.manager import compute_levels, size_position
import risk.manager as rm


def test_compute_levels_basic_long(monkeypatch):
    monkeypatch.setattr('utils.config.load_config', lambda path: {'risk': {'min_stop_pct': 0.0}})
    stop, tgt = compute_levels('long', 100.0, 2.0, 2.0, 3.0)
    assert stop == 96.0 and tgt == 112.0


def test_compute_levels_min_stop_pct(monkeypatch):
    monkeypatch.setattr('utils.config.load_config', lambda path: {'risk': {'min_stop_pct': 0.05}})
    stop, tgt = compute_levels('long', 100.0, 0.5, 0.2, 2.0)
    # min_stop_pct=5% -> floor distance=5 -> stop=95, R=5, tgt=110
    assert stop == 95.0
    assert tgt == 110.0


def test_size_position_basic():
    cfg = {'risk': {'risk_per_trade_pct': 0.01, 'account_equity': 100000}}
    qty = size_position(cfg, 100.0, 98.0)
    assert qty == 500


def test_size_position_caps():
    cfg = {'risk': {
        'risk_per_trade_pct': 0.01,
        'account_equity': 100000,
        'max_position_notional_pct': 0.20
    }}
    qty = size_position(cfg, 100.0, 98.0)
    assert qty == 200
