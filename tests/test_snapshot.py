from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from swingbot.data import SnapshotMetadata, SnapshotStore
from tests.helpers import make_bars


def _metadata() -> SnapshotMetadata:
    return SnapshotMetadata(
        requested_start="2024-01-01",
        requested_end="2024-12-31",
        warmup_start="2022-08-19",
        provider="test",
        feed="sip",
        adjustment="all",
        symbols=("SPY",),
        created_at_utc="2025-01-01T00:00:00+00:00",
    )


class SnapshotTests(unittest.TestCase):
    def test_snapshot_round_trip_and_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            SnapshotStore.create(snapshot, {"SPY": make_bars(rows=20)}, _metadata())
            frames, manifest = SnapshotStore.load(snapshot)
            self.assertEqual(tuple(manifest["symbols"]), ("SPY",))
            self.assertEqual(len(frames["SPY"]), 20)

            with (snapshot / "SPY.csv").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                SnapshotStore.load(snapshot)

    def test_snapshot_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            SnapshotStore.create(snapshot, {"SPY": make_bars(rows=5)}, _metadata())
            with self.assertRaises(FileExistsError):
                SnapshotStore.create(snapshot, {"SPY": make_bars(rows=5)}, _metadata())


if __name__ == "__main__":
    unittest.main()
