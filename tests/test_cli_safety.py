from __future__ import annotations

import unittest
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from swingbot.cli import _latest_completed_bar_date, _validate_submission_date


class CliSafetyTests(unittest.TestCase):
    def test_future_and_stale_paper_dates_are_rejected(self):
        today = datetime.now(ZoneInfo("America/New_York")).date()
        with self.assertRaisesRegex(ValueError, "future"):
            _validate_submission_date(today + timedelta(days=1))
        with self.assertRaisesRegex(ValueError, "four days old"):
            _validate_submission_date(today - timedelta(days=5))

    def test_latest_completed_date_uses_a_common_prior_bar(self):
        today = datetime.now(ZoneInfo("America/New_York")).date()
        earlier = today - timedelta(days=2)
        latest = today - timedelta(days=1)
        index = pd.to_datetime([earlier, latest])
        frames = {
            "SPY": pd.DataFrame(index=index),
            "QQQ": pd.DataFrame(index=index),
        }
        self.assertEqual(_latest_completed_bar_date(frames), latest)

    def test_regular_session_submission_is_rejected(self):
        during_session = datetime.combine(
            date(2026, 8, 10),
            time(11, 0),
            tzinfo=ZoneInfo("America/New_York"),
        )
        with self.assertRaisesRegex(ValueError, "9:30 AM through 4:15 PM"):
            _validate_submission_date(date(2026, 8, 7), during_session)


if __name__ == "__main__":
    unittest.main()
