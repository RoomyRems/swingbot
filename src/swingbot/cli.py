from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .backtest import run_backtest, write_report
from .config import load_config
from .data import AlpacaDataSource, SnapshotStore, fetch_snapshot
from .paper import AlpacaPaperBroker, build_paper_plan, write_paper_plan
from .research import execute_research_request
from .strategy import (
    STRATEGY_VERSION,
    assess_setup,
    prepare_indicators,
    strategy_fingerprint,
)


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use an ISO date such as 2024-01-31") from exc


def _fetch(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    path = fetch_snapshot(
        args.snapshot,
        config.symbols,
        args.start,
        args.end,
        config.data,
    )
    print(f"created immutable snapshot: {path}")
    return 0


def _backtest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    frames, manifest = SnapshotStore.load(args.snapshot)
    snapshot_symbols = tuple(manifest.get("symbols", ()))
    if tuple(config.symbols) != snapshot_symbols:
        raise ValueError("configured symbols do not exactly match the snapshot manifest")
    if manifest.get("adjustment") != config.data.adjustment:
        raise ValueError("configured adjustment does not match the snapshot manifest")
    if manifest.get("feed") != config.data.feed:
        raise ValueError("configured feed does not match the snapshot manifest")

    result = run_backtest(frames, config, args.start, args.end, benchmark_symbol=args.benchmark)
    provenance = {
        "snapshot_fingerprint": manifest.get("snapshot_fingerprint"),
        "data_fingerprint": manifest.get("data_fingerprint"),
        "provider": manifest.get("provider"),
        "feed": manifest.get("feed"),
        "adjustment": manifest.get("adjustment"),
    }
    output = write_report(result, args.output, provenance=provenance)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(f"wrote report: {output}")
    return 0


def _research(args: argparse.Namespace) -> int:
    output = execute_research_request(args.request, args.output)
    summary = json.loads((output / "report" / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote authenticated research run: {output}")
    return 0


def _validate_submission_date(as_of: date, now: datetime | None = None) -> None:
    now = now or datetime.now(ZoneInfo("America/New_York"))
    if as_of > now.date():
        raise ValueError("paper submission date cannot be in the future")
    if (now.date() - as_of).days > 4:
        raise ValueError("paper submission refuses an as-of date more than four days old")
    if now.weekday() < 5 and time(9, 30) <= now.time() < time(16, 15):
        raise ValueError("paper submission is disabled from 9:30 AM through 4:15 PM New York time")
    if as_of == now.date() and now.time() < time(16, 15):
        raise ValueError("today's daily bar is not considered complete until 4:15 PM New York time")


def _latest_completed_bar_date(frames: dict[str, pd.DataFrame]) -> date:
    if not frames:
        raise ValueError("no market data was returned for paper-date verification")
    common_dates = set.intersection(
        *({timestamp.date() for timestamp in frame.index} for frame in frames.values())
    )
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.time() < time(16, 15):
        common_dates = {value for value in common_dates if value < now.date()}
    else:
        common_dates = {value for value in common_dates if value <= now.date()}
    if not common_dates:
        raise ValueError("could not identify a completed common daily bar")
    return max(common_dates)


def _paper(args: argparse.Namespace) -> int:
    if args.submit and args.confirm != "PAPER":
        raise ValueError("--submit requires --confirm PAPER")
    if args.submit:
        _validate_submission_date(args.as_of)
    config = load_config(args.config)
    data_source = AlpacaDataSource()
    fetch_end = datetime.now(ZoneInfo("America/New_York")).date() if args.submit else args.as_of
    frames = data_source.fetch_daily(
        config.symbols,
        args.as_of - timedelta(days=500),
        fetch_end,
        config.data,
    )
    if args.submit:
        latest_completed = _latest_completed_bar_date(frames)
        if args.as_of != latest_completed:
            raise ValueError(
                f"paper submission requires the latest completed common bar: {latest_completed}"
            )
    broker = AlpacaPaperBroker()
    state = broker.state()
    plans = build_paper_plan(frames, args.as_of, config, state)

    printable = [
        {
            "symbol": plan.signal.symbol,
            "signal_date": plan.signal.signal_date.isoformat(),
            "quantity": plan.quantity,
            "entry_stop": plan.signal.entry_stop,
            "entry_limit": plan.signal.entry_limit,
            "stop_price": plan.signal.stop_price,
            "target_price": plan.signal.target_price,
            "reserved_risk": round(plan.reserved_risk, 2),
        }
        for plan in plans
    ]
    print(json.dumps(printable, indent=2))
    if args.plan_out:
        print(f"wrote paper plan: {write_paper_plan(plans, args.plan_out)}")
    if args.submit:
        submitted = broker.submit(
            plans,
            config,
            state,
            confirmation=args.confirm,
        )
        print(json.dumps({"submitted": submitted}, indent=2))
    else:
        print("dry run only; no paper orders submitted")
    return 0


def _explain(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    frames, manifest = SnapshotStore.load(args.snapshot)
    symbol = args.symbol.upper()
    if symbol not in frames:
        raise ValueError(f"snapshot has no data for {symbol}")
    if manifest.get("adjustment") != config.data.adjustment:
        raise ValueError("configured adjustment does not match the snapshot manifest")
    prepared = prepare_indicators(frames[symbol])
    assessment = assess_setup(
        symbol,
        prepared,
        args.as_of,
        max_entry_gap_r=config.execution.max_entry_gap_r,
        reward_r=config.risk.reward_r,
    )
    energies = assessment.energies
    signal = assessment.signal
    payload = {
        "symbol": symbol,
        "as_of": args.as_of.isoformat(),
        "strategy_version": STRATEGY_VERSION,
        "strategy_fingerprint": strategy_fingerprint(),
        "score": sum(item.passed for item in energies.values()),
        "energies": {name: asdict(item) for name, item in energies.items()},
        "context": assessment.context,
        "signal": None
        if signal is None
        else {
            "entry_stop": signal.entry_stop,
            "entry_limit": signal.entry_limit,
            "stop_price": signal.stop_price,
            "target_price": signal.target_price,
            "average_daily_volume": signal.average_daily_volume,
            "quality": signal.quality,
            "context": signal.context,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swingbot",
        description="Bias-aware 5-energy swing research and Alpaca paper trading",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="download an immutable data snapshot")
    fetch_parser.add_argument("--config", default="swingbot.toml", help="strict TOML config")
    fetch_parser.add_argument("--start", type=_date, required=True)
    fetch_parser.add_argument("--end", type=_date, required=True)
    fetch_parser.add_argument("--snapshot", type=Path, required=True)
    fetch_parser.set_defaults(handler=_fetch)

    backtest_parser = subparsers.add_parser("backtest", help="run from a verified snapshot")
    backtest_parser.add_argument("--config", default="swingbot.toml", help="strict TOML config")
    backtest_parser.add_argument("--snapshot", type=Path, required=True)
    backtest_parser.add_argument("--start", type=_date, required=True)
    backtest_parser.add_argument("--end", type=_date, required=True)
    backtest_parser.add_argument("--output", type=Path, required=True)
    backtest_parser.add_argument("--benchmark", default="SPY")
    backtest_parser.set_defaults(handler=_backtest)

    research_parser = subparsers.add_parser(
        "research", help="fetch a frozen snapshot and run one strict research request"
    )
    research_parser.add_argument("--request", type=Path, required=True)
    research_parser.add_argument("--output", type=Path, required=True)
    research_parser.set_defaults(handler=_research)

    paper_parser = subparsers.add_parser(
        "paper", help="plan or explicitly submit Alpaca paper orders"
    )
    paper_parser.add_argument("--config", default="swingbot.toml", help="strict TOML config")
    paper_parser.add_argument(
        "--as-of",
        type=_date,
        required=True,
        help="completed daily bar to evaluate; run only after that session closes",
    )
    paper_parser.add_argument("--plan-out", type=Path)
    paper_parser.add_argument("--submit", action="store_true")
    paper_parser.add_argument("--confirm", default="")
    paper_parser.set_defaults(handler=_paper)

    explain_parser = subparsers.add_parser(
        "explain", help="show every energy decision for one snapshot bar"
    )
    explain_parser.add_argument("--config", default="swingbot.toml", help="strict TOML config")
    explain_parser.add_argument("--snapshot", type=Path, required=True)
    explain_parser.add_argument("--symbol", required=True)
    explain_parser.add_argument("--as-of", type=_date, required=True)
    explain_parser.set_defaults(handler=_explain)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
