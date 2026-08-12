from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from swingbot.backtest import _StrategyDiagnostics, write_report
from swingbot.models import BacktestResult, EnergyEvidence


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

    def test_strategy_diagnostics_count_independent_energy_gates(self):
        accumulator = _StrategyDiagnostics(("SPY", "QQQ"))
        for symbol in ("SPY", "QQQ"):
            for day, all_pass in (("2024-01-02", False), ("2024-01-03", True)):
                energies = {
                    name: EnergyEvidence(all_pass or name == "trend", 1.0, "test")
                    for name in ("trend", "momentum", "cycle", "support", "scale")
                }
                accumulator.observe(symbol, pd.Timestamp(day), energies)
        diagnostics = accumulator.as_dict()

        self.assertEqual(diagnostics["evaluated_symbol_sessions"], 4)
        self.assertEqual(diagnostics["energy_pass_counts"]["trend"], 4)
        self.assertEqual(diagnostics["energy_pass_counts"]["cycle"], 2)
        self.assertEqual(
            diagnostics["score_counts"], {"0": 0, "1": 2, "2": 0, "3": 0, "4": 0, "5": 2}
        )
        self.assertEqual(diagnostics["eligible_setups"], 2)
        self.assertEqual(diagnostics["eligible_by_symbol"], {"SPY": 1, "QQQ": 1})


if __name__ == "__main__":
    unittest.main()
