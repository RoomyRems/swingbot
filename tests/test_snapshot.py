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

    def test_data_fingerprint_is_stable_across_creation_times(self):
        first_metadata = _metadata()
        second_metadata = SnapshotMetadata(
            requested_start=first_metadata.requested_start,
            requested_end=first_metadata.requested_end,
            warmup_start=first_metadata.warmup_start,
            provider=first_metadata.provider,
            feed=first_metadata.feed,
            adjustment=first_metadata.adjustment,
            symbols=first_metadata.symbols,
            created_at_utc="2026-01-01T00:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bars = make_bars(rows=20)
            SnapshotStore.create(root / "first", {"SPY": bars}, first_metadata)
            SnapshotStore.create(root / "second", {"SPY": bars}, second_metadata)
            _, first = SnapshotStore.load(root / "first")
            _, second = SnapshotStore.load(root / "second")

        self.assertEqual(first["data_fingerprint"], second["data_fingerprint"])
        self.assertNotEqual(first["snapshot_fingerprint"], second["snapshot_fingerprint"])


if __name__ == "__main__":
    unittest.main()
