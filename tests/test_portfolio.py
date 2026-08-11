from __future__ import annotations

import unittest
from datetime import date

from swingbot.portfolio import Capacity, size_order
from tests.helpers import app_config, complete_signal


class PortfolioTests(unittest.TestCase):
    def test_position_size_respects_risk_notional_and_cash_caps(self):
        config = app_config("SPY")
        signal = complete_signal("SPY", date(2024, 1, 2))
        order = size_order(
            signal,
            date(2024, 1, 3),
            config,
            Capacity(
                equity=100_000.0,
                buying_power=100_000.0,
                committed_risk=0.0,
                used_slots=0,
            ),
        )
        self.assertIsNotNone(order)
        assert order is not None
        self.assertEqual(order.quantity, 45)
        self.assertLessEqual(order.reserved_risk, 500.0)
        self.assertLessEqual(order.reserved_notional, 20_000.0)

    def test_total_risk_and_slot_caps_fail_closed(self):
        config = app_config("SPY")
        signal = complete_signal("SPY", date(2024, 1, 2))
        no_risk = size_order(
            signal,
            date(2024, 1, 3),
            config,
            Capacity(100_000.0, 100_000.0, 3_000.0, 0),
        )
        no_slot = size_order(
            signal,
            date(2024, 1, 3),
            config,
            Capacity(100_000.0, 100_000.0, 0.0, 5),
        )
        self.assertIsNone(no_risk)
        self.assertIsNone(no_slot)


if __name__ == "__main__":
    unittest.main()
