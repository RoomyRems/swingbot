from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from swingbot.data import SnapshotMetadata, SnapshotStore
from swingbot.research import execute_research_request, load_research_request
from tests.helpers import make_bars


def _write_config(root: Path) -> None:
    (root / "swingbot.toml").write_text(
        """symbols = [\"SPY\"]

[data]
feed = \"sip\"
adjustment = \"all\"

[risk]
initial_equity = 100000.0
risk_per_trade = 0.005
max_total_risk = 0.03
max_positions = 5
max_position_fraction = 0.20
reward_r = 2.0

[execution]
max_entry_gap_r = 0.25
slippage_bps = 5.0
commission_per_share = 0.0
""",
        encoding="utf-8",
    )


def _write_request(root: Path, **replacements: str) -> Path:
    values = {
        "schema_version": "1",
        "name": '"pilot-test"',
        "config": '"swingbot.toml"',
        "start": '"2021-01-04"',
        "end": '"2022-12-30"',
        "benchmark": '"SPY"',
        **replacements,
    }
    path = root / "research" / "requests" / "pilot.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{key} = {value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )
    return path


class ResearchRequestTests(unittest.TestCase):
    def test_request_is_strict_date_bounded_and_repository_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_config(root)
            path = _write_request(root)
            request = load_research_request(
                path,
                repository_root=root,
                today=date(2026, 8, 11),
            )
            self.assertEqual(request.name, "pilot-test")
            self.assertEqual(request.start, date(2021, 1, 4))
            self.assertEqual(request.end, date(2022, 12, 30))
            self.assertEqual(request.config_path, root / "swingbot.toml")

            path.write_text(path.read_text(encoding="utf-8") + "surprise = true\n")
            with self.assertRaisesRegex(ValueError, "unknown research request"):
                load_research_request(path, repository_root=root, today=date(2026, 8, 11))

    def test_request_rejects_path_escape_and_incomplete_end_date(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_config(root)
            path = _write_request(root, config='"../outside.toml"')
            with self.assertRaisesRegex(ValueError, "escapes the repository"):
                load_research_request(path, repository_root=root, today=date(2026, 8, 11))

            path = _write_request(root, end='"2026-08-11"')
            with self.assertRaisesRegex(ValueError, "before today"):
                load_research_request(path, repository_root=root, today=date(2026, 8, 11))

            with self.assertRaisesRegex(ValueError, "under research/requests"):
                load_research_request(
                    root / "swingbot.toml",
                    repository_root=root,
                    today=date(2026, 8, 11),
                )

    def test_execution_verifies_snapshot_and_writes_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_config(root)
            request_path = _write_request(root)
            output = root / "output"

            def fake_fetch(destination, symbols, start, end, data_config):
                bars = make_bars(rows=900, start="2019-01-01")
                metadata = SnapshotMetadata(
                    requested_start=start.isoformat(),
                    requested_end=end.isoformat(),
                    warmup_start="2019-08-23",
                    provider="test",
                    feed=data_config.feed,
                    adjustment=data_config.adjustment,
                    symbols=tuple(symbols),
                    created_at_utc="2026-08-11T00:00:00+00:00",
                )
                return SnapshotStore.create(destination, {"SPY": bars}, metadata)

            with patch("swingbot.research.fetch_snapshot", side_effect=fake_fetch):
                result = execute_research_request(
                    request_path,
                    output,
                    repository_root=root,
                    today=date(2026, 8, 11),
                )

            run = json.loads((result / "run.json").read_text(encoding="utf-8"))
            summary = json.loads((result / "report" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(run["name"], "pilot-test")
            self.assertEqual(len(run["research_request_fingerprint"]), 64)
            self.assertEqual(
                run["research_request_fingerprint"],
                summary["provenance"]["research_request_fingerprint"],
            )
            self.assertEqual(
                run["snapshot_fingerprint"],
                summary["provenance"]["snapshot_fingerprint"],
            )
            self.assertEqual(
                run["data_fingerprint"],
                summary["provenance"]["data_fingerprint"],
            )


if __name__ == "__main__":
    unittest.main()
