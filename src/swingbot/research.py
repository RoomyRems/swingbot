from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .backtest import run_backtest, write_report
from .config import AppConfig, config_fingerprint, load_config
from .data import SnapshotStore, fetch_snapshot
from .exits import ExitPolicy
from .strategy import STRATEGY_VERSION, strategy_fingerprint

_REQUEST_KEYS = {
    "schema_version",
    "name",
    "config",
    "start",
    "end",
    "benchmark",
    "exit_policy",
}
_RUN_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_MAX_CALENDAR_DAYS = 10 * 366
_MAX_SYMBOLS = 25


@dataclass(frozen=True)
class ResearchRequest:
    schema_version: int
    name: str
    config_path: Path
    start: date
    end: date
    benchmark: str
    exit_policy: ExitPolicy


def _request_date(value: Any, name: str) -> date:
    if isinstance(value, datetime):
        raise ValueError(f"research request {name} must be a date without a time")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"research request {name} must be an ISO date") from exc
    raise ValueError(f"research request {name} must be an ISO date")


def _repository_path(repository_root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"research request {name} must be a repository-relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"research request {name} must be a repository-relative path")
    resolved = (repository_root / relative).resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"research request {name} escapes the repository") from exc
    return resolved


def load_research_request(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
    today: date | None = None,
) -> ResearchRequest:
    root = Path(repository_root or Path.cwd()).resolve()
    request_path = Path(path)
    if not request_path.is_absolute():
        request_path = root / request_path
    request_path = request_path.resolve()
    try:
        request_path.relative_to((root / "research" / "requests").resolve())
    except ValueError as exc:
        raise ValueError("research request path must be under research/requests") from exc

    try:
        raw = tomllib.loads(request_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in {request_path}: {exc}") from exc
    unknown = sorted(set(raw) - _REQUEST_KEYS)
    if unknown:
        raise ValueError(f"unknown research request setting(s): {', '.join(unknown)}")
    missing = sorted(_REQUEST_KEYS - set(raw))
    if missing:
        raise ValueError(f"missing research request setting(s): {', '.join(missing)}")

    schema_version = raw["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError("research request schema_version must be 1")
    name = raw["name"]
    if not isinstance(name, str) or not _RUN_NAME.fullmatch(name):
        raise ValueError("research request name must be 1-64 lowercase letters, digits, or hyphens")
    start = _request_date(raw["start"], "start")
    end = _request_date(raw["end"], "end")
    if start >= end:
        raise ValueError("research request start must be before end")
    if (end - start).days > _MAX_CALENDAR_DAYS:
        raise ValueError("research request cannot span more than ten years")
    if end >= (today or date.today()):
        raise ValueError("research request end must be before today so every daily bar is complete")

    benchmark = raw["benchmark"]
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ValueError("research request benchmark must be a symbol")
    return ResearchRequest(
        schema_version=1,
        name=name,
        config_path=_repository_path(root, raw["config"], "config"),
        start=start,
        end=end,
        benchmark=benchmark.strip().upper(),
        exit_policy=ExitPolicy(str(raw["exit_policy"])),
    )


def research_request_fingerprint(request: ResearchRequest, config: AppConfig) -> str:
    payload = {
        "schema_version": request.schema_version,
        "name": request.name,
        "start": request.start.isoformat(),
        "end": request.end.isoformat(),
        "benchmark": request.benchmark,
        "exit_policy": request.exit_policy.value,
        "config_fingerprint": config_fingerprint(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def execute_research_request(
    request_path: str | Path,
    output_directory: str | Path,
    *,
    repository_root: str | Path | None = None,
    today: date | None = None,
) -> Path:
    root = Path(repository_root or Path.cwd()).resolve()
    request = load_research_request(request_path, repository_root=root, today=today)
    config = load_config(request.config_path)
    if len(config.symbols) > _MAX_SYMBOLS:
        raise ValueError(f"authenticated research runs are capped at {_MAX_SYMBOLS} symbols")
    if request.benchmark not in config.symbols:
        raise ValueError("research request benchmark must be present in the configured symbols")

    output = Path(output_directory)
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"research output path is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"research output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "snapshot"
    report_path = output / "report"

    fetch_snapshot(
        snapshot_path,
        config.symbols,
        request.start,
        request.end,
        config.data,
    )
    frames, manifest = SnapshotStore.load(snapshot_path)
    if tuple(manifest.get("symbols", ())) != tuple(config.symbols):
        raise ValueError("snapshot symbols do not exactly match the research configuration")
    if manifest.get("requested_start") != request.start.isoformat():
        raise ValueError("snapshot requested start does not match the research request")
    if manifest.get("requested_end") != request.end.isoformat():
        raise ValueError("snapshot requested end does not match the research request")
    if manifest.get("feed") != config.data.feed:
        raise ValueError("snapshot feed does not match the research configuration")
    if manifest.get("adjustment") != config.data.adjustment:
        raise ValueError("snapshot adjustment does not match the research configuration")

    request_fingerprint = research_request_fingerprint(request, config)
    result = run_backtest(
        frames,
        config,
        request.start,
        request.end,
        benchmark_symbol=request.benchmark,
        exit_policy=request.exit_policy,
    )
    provenance = {
        "research_request": request.name,
        "research_request_fingerprint": request_fingerprint,
        "snapshot_fingerprint": manifest.get("snapshot_fingerprint"),
        "data_fingerprint": manifest.get("data_fingerprint"),
        "provider": manifest.get("provider"),
        "feed": manifest.get("feed"),
        "adjustment": manifest.get("adjustment"),
    }
    write_report(result, report_path, provenance=provenance)

    baseline_report: str | None = None
    if request.exit_policy is not ExitPolicy.STATIC_2R:
        baseline_result = run_backtest(
            frames,
            config,
            request.start,
            request.end,
            benchmark_symbol=request.benchmark,
            exit_policy=ExitPolicy.STATIC_2R,
        )
        baseline_path = output / "baseline-report"
        baseline_provenance = {**provenance, "comparison_role": "static-2r baseline"}
        write_report(baseline_result, baseline_path, provenance=baseline_provenance)
        baseline_report = "baseline-report"

        comparison_keys = (
            "total_return",
            "cagr",
            "benchmark_cagr",
            "max_drawdown",
            "daily_sharpe",
            "trades",
            "win_rate",
            "profit_factor",
            "expectancy_r",
            "invested_session_fraction",
            "meets_15_percent_cagr_hurdle",
            "meets_20_percent_cagr_hurdle",
        )
        comparison = {
            "candidate_exit_policy": request.exit_policy.value,
            "baseline_exit_policy": ExitPolicy.STATIC_2R.value,
            "data_fingerprint": manifest.get("data_fingerprint"),
            "candidate": {key: result.summary[key] for key in comparison_keys},
            "baseline": {key: baseline_result.summary[key] for key in comparison_keys},
            "same_strategy_fingerprint": (
                result.summary["strategy_fingerprint"]
                == baseline_result.summary["strategy_fingerprint"]
            ),
        }
        (output / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    run_manifest = {
        "schema_version": 1,
        "name": request.name,
        "start": request.start.isoformat(),
        "end": request.end.isoformat(),
        "benchmark": request.benchmark,
        "exit_policy": request.exit_policy.value,
        "symbols": list(config.symbols),
        "strategy_version": STRATEGY_VERSION,
        "strategy_fingerprint": strategy_fingerprint(),
        "config_fingerprint": config_fingerprint(config),
        "research_request_fingerprint": request_fingerprint,
        "snapshot_fingerprint": manifest.get("snapshot_fingerprint"),
        "data_fingerprint": manifest.get("data_fingerprint"),
        "snapshot_manifest": "snapshot/manifest.json",
        "report": "report",
        "baseline_report": baseline_report,
    }
    (output / "run.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
