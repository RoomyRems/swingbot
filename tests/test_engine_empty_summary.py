import pandas as pd


class DummyAPI:
    def list_positions(self):
        return []

    def list_orders(self, *args, **kwargs):
        return []


def test_summary_counts_after_sanity_check(monkeypatch, tmp_path, capsys):
    # Provide dummy API credentials before importing the module
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_API_SECRET", "secret")

    # Ensure repository root is on sys.path then import the function
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.engine import execute_today

    # Create CSV with a row that will be dropped by sanity checks (qty <= 0)
    csv_path = tmp_path / "trade_signals.csv"
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "direction": ["buy"],
        "quantity": [0],  # invalid quantity -> triggers empty after sanity checks
        "target": [150],
        "stop": [140],
    })
    df.to_csv(csv_path, index=False)

    # Patch API to avoid external calls
    monkeypatch.setattr("execution.engine.api", DummyAPI())

    execute_today(csv_path=csv_path)

    out = capsys.readouterr().out
    assert "After open flt: 1 (dropped 0)" in out
