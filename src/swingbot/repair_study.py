"""Execute the finite, predeclared audit repair comparison on one snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .backtest import run_backtest, write_report
from .config import config_fingerprint, load_config
from .controls import simple_trend_control
from .data import SnapshotStore, fetch_snapshot
from .exits import ExitPolicy
from .research import load_research_request
from .strategy import DEFAULT_RULES

BASELINE_COMMIT = "5cd61af3f1b44e236bd153fcc7c306df4e98f5b0"
BASELINE_TREE = "654423c6aae2686e8b95c0871edfa513f596494d"
PRIOR_DATA = "d1f0b306459ca2837a0490152c3becf0ddc80dc06c228844b7d6c07de91016b1"
CASES = (
    ("original-static", "static-2r", False, 5, True),
    ("original-burns", "burns-cycle-v1", False, 5, True),
    ("repaired-static", "static-2r", False, 5, False),
    ("repaired-burns", "burns-cycle-v1", False, 5, False),
    ("fill-risk-static", "static-fill-2r", False, 5, False),
    ("wave-manager", "burns-cycle-v2", False, 5, False),
    ("wave-entry-static", "static-fill-2r", True, 5, False),
    ("wave-entry-burns", "burns-cycle-v2", True, 5, False),
    ("stress-static", "static-fill-2r", True, 10, False),
    ("stress-burns", "burns-cycle-v2", True, 10, False),
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify_baseline(root: Path) -> None:
    if (
        _git(root, "rev-parse", "HEAD") != BASELINE_COMMIT
        or _git(root, "rev-parse", "HEAD^{tree}") != BASELINE_TREE
        or _git(root, "status", "--porcelain")
    ):
        raise ValueError("baseline checkout must be clean and match the pinned commit and tree")


def _save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def report_checks(path: Path) -> dict[str, object]:
    summary = json.loads((path / "summary.json").read_text())
    trades, equity = _read_csv(path / "trades.csv"), _read_csv(path / "equity.csv")
    pnl = float(trades["pnl"].sum()) if len(trades) else 0.0
    error = float(summary["end_equity"]) - float(summary["start_equity"]) - pnl
    if abs(error) > 0.00001:
        raise ValueError(f"cash/trade P&L reconciliation failed: {path.name}")
    actual_r = (
        (
            trades["pnl"] / (trades["quantity"] * (trades["entry_price"] - trades["stop_price"]))
        ).mean()
        if len(trades)
        else None
    )
    signals = _read_csv(path / "signals.csv")
    all_five_divergence = 0
    if len(trades) and len(signals):
        joined = trades.merge(signals, on=["symbol", "signal_date"], suffixes=("", "_signal"))
        all_five_divergence = int(
            ((joined["score"] == 5) & joined["context_mini_divergence"].eq(True)).sum()
        )
    return {
        "cash_pnl_reconciliation_error": error,
        "expectancy_actual_r_recomputed": float(actual_r) if actual_r is not None else None,
        "mean_end_of_day_exposure": float(
            ((equity["equity"] - equity["cash"]) / equity["equity"]).mean()
        ),
        "all_five_plus_divergence_filled": all_five_divergence,
    }


def execute_study(request_path: Path, baseline_source: Path, output: Path) -> Path:
    root, baseline_source, output = (
        Path.cwd().resolve(),
        baseline_source.resolve(),
        output.resolve(),
    )
    verify_baseline(baseline_source)
    request = load_research_request(request_path, repository_root=root)
    config = load_config(request.config_path)
    baseline_config = load_config(baseline_source / "swingbot.toml")
    if (
        config_fingerprint(config) != config_fingerprint(baseline_config)
        or str(request.start) != "2018-01-01"
        or str(request.end) != "2025-12-31"
        or request.benchmark != "SPY"
    ):
        raise ValueError("repair study requires the unchanged pilot inputs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("repair study output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "snapshot"
    fetch_snapshot(snapshot, config.symbols, request.start, request.end, config.data)
    frames, manifest = SnapshotStore.load(snapshot)
    expected = {
        "symbols": list(config.symbols),
        "requested_start": str(request.start),
        "requested_end": str(request.end),
        "feed": config.data.feed,
        "adjustment": config.data.adjustment,
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("snapshot does not match the frozen study request")
    provenance = {
        "data_fingerprint": manifest["data_fingerprint"],
        "snapshot_fingerprint": manifest["snapshot_fingerprint"],
        "baseline_commit": BASELINE_COMMIT,
        "study": "audit-repair-v1",
    }
    outcomes = {}
    for name, policy, wave_entries, bps, original in CASES:
        print(f"Starting {name}", flush=True)
        destination = output / "reports" / name
        if original:
            # The original module is selected before editable-site imports. No
            # broker credentials or other job secrets enter this child process.
            environment = {k: os.environ[k] for k in ("PATH", "LANG") if k in os.environ}
            environment["PYTHONPATH"] = str(baseline_source / "src")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "swingbot",
                    "backtest",
                    "--config",
                    str(request.config_path),
                    "--snapshot",
                    str(snapshot),
                    "--start",
                    str(request.start),
                    "--end",
                    str(request.end),
                    "--output",
                    str(destination),
                    "--benchmark",
                    request.benchmark,
                    "--exit-policy",
                    policy,
                ],
                cwd=baseline_source,
                env=environment,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        else:
            case_config = replace(config, execution=replace(config.execution, slippage_bps=bps))
            result = run_backtest(
                frames,
                case_config,
                request.start,
                request.end,
                benchmark_symbol=request.benchmark,
                exit_policy=ExitPolicy(policy),
                strategy_rules=replace(DEFAULT_RULES, objective_wave_retraces=wave_entries),
            )
            write_report(result, destination, provenance={**provenance, "case": name})
        summary = json.loads((destination / "summary.json").read_text())
        outcomes[name] = {"summary": summary, "checks": report_checks(destination)}
        print(
            f"Completed {name}: {summary['trades']} trades; CAGR {summary['cagr']:.6%}", flush=True
        )
    control, control_equity = simple_trend_control(
        frames["SPY"], config, request.start, request.end
    )
    control_equity.to_csv(output / "control-equity.csv", index=False)
    _save(output / "control.json", control)
    # Matched entry differences are descriptive; trades are not independent samples.
    matched = {}
    for left, right in (
        ("original-static", "repaired-static"),
        ("original-burns", "repaired-burns"),
        ("repaired-burns", "wave-manager"),
        ("wave-entry-static", "wave-entry-burns"),
    ):
        a = _read_csv(output / "reports" / left / "trades.csv")
        b = _read_csv(output / "reports" / right / "trades.csv")
        if a.empty or b.empty:
            matched[f"{left} -> {right}"] = {"matched_entries": 0}
            continue
        merged = a.merge(b, on=["symbol", "signal_date", "entry_date"], suffixes=("_a", "_b"))
        for suffix in ("a", "b"):
            merged[f"actual_r_{suffix}"] = merged[f"pnl_{suffix}"] / (
                merged[f"quantity_{suffix}"]
                * (merged[f"entry_price_{suffix}"] - merged[f"stop_price_{suffix}"])
            )
        matched[f"{left} -> {right}"] = {
            "matched_entries": len(merged),
            "mean_change_actual_r": (
                float((merged["actual_r_b"] - merged["actual_r_a"]).mean()) if len(merged) else None
            ),
        }
    _save(
        output / "comparison.json",
        {"cases": outcomes, "matched_entries": matched, "control": control, **provenance},
    )
    _save(
        output / "run.json",
        {
            **provenance,
            "cases": CASES,
            "source_commit": _git(root, "rev-parse", "HEAD"),
            "source_tree": _git(root, "rev-parse", "HEAD^{tree}"),
            "config_fingerprint": config_fingerprint(config),
            "protocol_sha256": hashlib.sha256(
                (root / "docs/REPAIR_STUDY.md").read_bytes()
            ).hexdigest(),
            "prior_data_fingerprint": PRIOR_DATA,
            "matches_original_pilot_data": manifest["data_fingerprint"] == PRIOR_DATA,
            "sample_role": "development; no untouched holdout",
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--baseline-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    execute_study(args.request, args.baseline_source, args.output)


if __name__ == "__main__":
    main()
