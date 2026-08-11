from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from swingbot.backtest import write_report
from swingbot.models import BacktestResult


class ReportingTests(unittest.TestCase):
    def test_report_has_research_breakdowns(self):
        result = BacktestResult(
            summary={"total_return": 0.01},
            equity_rows=[
                {"date": "2024-01-02", "equity": 100_000.0, "cash": 100_000.0},
                {"date": "2024-12-31", "equity": 101_000.0, "cash": 101_000.0},
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = write_report(result, Path(directory) / "report")
            names = {path.name for path in output.iterdir()}
        self.assertEqual(
            names,
            {
                "summary.json",
                "trades.csv",
                "equity.csv",
                "signals.csv",
                "orders.csv",
                "yearly.csv",
                "by_symbol.csv",
            },
        )


if __name__ == "__main__":
    unittest.main()
