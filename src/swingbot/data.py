from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from .config import DataConfig, alpaca_credentials
from .indicators import normalize_bars


@dataclass(frozen=True)
class SnapshotMetadata:
    requested_start: str
    requested_end: str
    warmup_start: str
    provider: str
    feed: str
    adjustment: str
    symbols: tuple[str, ...]
    created_at_utc: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_fingerprint(manifest: dict[str, object]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SnapshotStore:
    """Immutable CSV snapshots with per-file hashes and explicit provenance."""

    MANIFEST = "manifest.json"

    @classmethod
    def create(
        cls,
        directory: str | Path,
        frames: dict[str, pd.DataFrame],
        metadata: SnapshotMetadata,
    ) -> Path:
        destination = Path(directory)
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"snapshot directory is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)

        files: dict[str, dict[str, object]] = {}
        for symbol in metadata.symbols:
            if symbol not in frames:
                raise ValueError(f"missing frame for {symbol}")
            bars = normalize_bars(frames[symbol])
            filename = f"{symbol}.csv"
            path = destination / filename
            bars.to_csv(path, date_format="%Y-%m-%d", lineterminator="\n")
            files[symbol] = {
                "path": filename,
                "rows": len(bars),
                "first_date": bars.index[0].date().isoformat(),
                "last_date": bars.index[-1].date().isoformat(),
                "sha256": _sha256(path),
            }

        manifest: dict[str, object] = {"schema_version": 1, **asdict(metadata), "files": files}
        manifest["snapshot_fingerprint"] = snapshot_fingerprint(manifest)
        (destination / cls.MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, directory: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
        source = Path(directory)
        manifest_path = source / cls.MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported snapshot schema")

        expected_fingerprint = manifest.get("snapshot_fingerprint")
        without_fingerprint = dict(manifest)
        without_fingerprint.pop("snapshot_fingerprint", None)
        if expected_fingerprint != snapshot_fingerprint(without_fingerprint):
            raise ValueError("snapshot manifest fingerprint does not match")

        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("snapshot manifest has no files")

        frames: dict[str, pd.DataFrame] = {}
        for symbol, details in files.items():
            if not isinstance(details, dict):
                raise ValueError(f"invalid manifest entry for {symbol}")
            relative_path = Path(str(details["path"]))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"unsafe snapshot path for {symbol}")
            path = source / relative_path
            if _sha256(path) != details.get("sha256"):
                raise ValueError(f"snapshot hash mismatch for {symbol}")
            frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
            frames[str(symbol)] = normalize_bars(frame)
        return frames, manifest


class AlpacaDataSource:
    def __init__(self) -> None:
        key, secret = alpaca_credentials()
        try:
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as exc:
            raise RuntimeError(
                "install paper dependencies with: pip install -e '.[paper]'"
            ) from exc
        self._client = StockHistoricalDataClient(key, secret)

    def fetch_daily(
        self,
        symbols: Iterable[str],
        start: date,
        end: date,
        config: DataConfig,
    ) -> dict[str, pd.DataFrame]:
        if start > end:
            raise ValueError("start date must not be after end date")
        requested_symbols = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
        if not requested_symbols:
            raise ValueError("at least one symbol is required")

        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        feed = DataFeed.SIP if config.feed == "sip" else DataFeed.IEX
        adjustment = Adjustment.ALL
        request = StockBarsRequest(
            symbol_or_symbols=list(requested_symbols),
            timeframe=TimeFrame.Day,
            start=datetime.combine(start, time.min, tzinfo=UTC),
            end=datetime.combine(end + timedelta(days=1), time.min, tzinfo=UTC),
            adjustment=adjustment,
            feed=feed,
            asof=end.isoformat(),
        )
        try:
            response = self._client.get_stock_bars(request).df
        except Exception as exc:
            raise RuntimeError(
                f"Alpaca daily-bar request failed for feed={config.feed}; "
                "check credentials and data entitlements"
            ) from exc
        if response.empty:
            raise RuntimeError("Alpaca returned no bars for the requested range")

        frames: dict[str, pd.DataFrame] = {}
        for symbol in requested_symbols:
            try:
                symbol_frame = response.xs(symbol, level="symbol")
            except (KeyError, ValueError) as exc:
                raise RuntimeError(f"Alpaca returned no bars for {symbol}") from exc
            normalized = normalize_bars(symbol_frame)
            mask = (normalized.index.date >= start) & (normalized.index.date <= end)
            normalized = normalized.loc[mask]
            if normalized.empty:
                raise RuntimeError(f"Alpaca returned no in-range bars for {symbol}")
            frames[symbol] = normalized
        return frames


def fetch_snapshot(
    directory: str | Path,
    symbols: Iterable[str],
    requested_start: date,
    requested_end: date,
    data_config: DataConfig,
    warmup_days: int = 500,
) -> Path:
    warmup_start = requested_start - timedelta(days=warmup_days)
    symbols_tuple = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
    frames = AlpacaDataSource().fetch_daily(
        symbols_tuple,
        warmup_start,
        requested_end,
        data_config,
    )
    metadata = SnapshotMetadata(
        requested_start=requested_start.isoformat(),
        requested_end=requested_end.isoformat(),
        warmup_start=warmup_start.isoformat(),
        provider="alpaca",
        feed=data_config.feed,
        adjustment=data_config.adjustment,
        symbols=symbols_tuple,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    return SnapshotStore.create(directory, frames, metadata)
