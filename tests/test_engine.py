import os
import importlib
import sys
from pathlib import Path
import pandas as pd


def test_normalize_columns_case_and_spaces(monkeypatch):
    # Ensure broker.alpaca can import without real credentials
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_API_SECRET", "y")

    # Ensure repository root on sys.path then import
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    engine = importlib.reload(importlib.import_module("execution.engine"))

    df = pd.DataFrame({
        "Qty": [1],
        "Take Profit": [10],
        "Stop Loss": [9],
        "Side": ["buy"],
        "Symbol": ["AAPL"],
    })

    norm = engine._normalize_columns(df)
    engine._require_columns(norm, ["symbol", "direction", "quantity", "target", "stop"])

    assert set(norm.columns) >= {"symbol", "direction", "quantity", "target", "stop"}
