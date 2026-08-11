from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DataConfig:
    feed: str = "sip"
    adjustment: str = "all"


@dataclass(frozen=True)
class RiskConfig:
    initial_equity: float = 100_000.0
    risk_per_trade: float = 0.005
    max_total_risk: float = 0.03
    max_positions: int = 5
    max_position_fraction: float = 0.20
    reward_r: float = 2.0


@dataclass(frozen=True)
class ExecutionConfig:
    max_entry_gap_r: float = 0.25
    slippage_bps: float = 5.0
    commission_per_share: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    symbols: tuple[str, ...]
    data: DataConfig
    risk: RiskConfig
    execution: ExecutionConfig


_ROOT_KEYS = {"symbols", "data", "risk", "execution"}
_DATA_KEYS = {"feed", "adjustment"}
_RISK_KEYS = {
    "initial_equity",
    "risk_per_trade",
    "max_total_risk",
    "max_positions",
    "max_position_fraction",
    "reward_r",
}
_EXECUTION_KEYS = {"max_entry_gap_r", "slippage_bps", "commission_per_share"}
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")


def _expect_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _reject_unknown(table: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} setting(s): {', '.join(unknown)}")


def _number(table: Mapping[str, Any], key: str, default: float) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _integer(table: Mapping[str, Any], key: str, default: int) -> int:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def load_config(path: str | Path = "swingbot.toml") -> AppConfig:
    config_path = Path(path)
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in {config_path}: {exc}") from exc

    _reject_unknown(raw, _ROOT_KEYS, "top-level")
    symbols_value = raw.get("symbols")
    if not isinstance(symbols_value, list) or not symbols_value:
        raise ValueError("symbols must be a non-empty TOML array")
    symbols = tuple(str(symbol).strip().upper() for symbol in symbols_value)
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must not contain duplicates")
    invalid_symbols = [symbol for symbol in symbols if not _SYMBOL.fullmatch(symbol)]
    if invalid_symbols:
        raise ValueError(f"invalid symbol(s): {', '.join(invalid_symbols)}")

    data_raw = _expect_mapping(raw.get("data", {}), "data")
    _reject_unknown(data_raw, _DATA_KEYS, "data")
    feed = str(data_raw.get("feed", "sip")).lower()
    if feed not in {"sip", "iex"}:
        raise ValueError("data.feed must be 'sip' or 'iex'")
    adjustment = str(data_raw.get("adjustment", "all")).lower()
    if adjustment != "all":
        raise ValueError("data.adjustment must be 'all' for comparable research")
    data = DataConfig(feed=feed, adjustment=adjustment)

    risk_raw = _expect_mapping(raw.get("risk", {}), "risk")
    _reject_unknown(risk_raw, _RISK_KEYS, "risk")
    risk = RiskConfig(
        initial_equity=_number(risk_raw, "initial_equity", 100_000.0),
        risk_per_trade=_number(risk_raw, "risk_per_trade", 0.005),
        max_total_risk=_number(risk_raw, "max_total_risk", 0.03),
        max_positions=_integer(risk_raw, "max_positions", 5),
        max_position_fraction=_number(risk_raw, "max_position_fraction", 0.20),
        reward_r=_number(risk_raw, "reward_r", 2.0),
    )
    if risk.initial_equity <= 0:
        raise ValueError("risk.initial_equity must be positive")
    if not 0 < risk.risk_per_trade <= 0.02:
        raise ValueError("risk.risk_per_trade must be in (0, 0.02]")
    if not risk.risk_per_trade <= risk.max_total_risk <= 0.10:
        raise ValueError("risk.max_total_risk must be between risk_per_trade and 0.10")
    if not 1 <= risk.max_positions <= 20:
        raise ValueError("risk.max_positions must be between 1 and 20")
    if not 0 < risk.max_position_fraction <= 1:
        raise ValueError("risk.max_position_fraction must be in (0, 1]")
    if not 1 <= risk.reward_r <= 10:
        raise ValueError("risk.reward_r must be between 1 and 10")

    execution_raw = _expect_mapping(raw.get("execution", {}), "execution")
    _reject_unknown(execution_raw, _EXECUTION_KEYS, "execution")
    execution = ExecutionConfig(
        max_entry_gap_r=_number(execution_raw, "max_entry_gap_r", 0.25),
        slippage_bps=_number(execution_raw, "slippage_bps", 5.0),
        commission_per_share=_number(execution_raw, "commission_per_share", 0.0),
    )
    if not 0 <= execution.max_entry_gap_r <= 1:
        raise ValueError("execution.max_entry_gap_r must be between 0 and 1")
    if not 0 <= execution.slippage_bps <= 100:
        raise ValueError("execution.slippage_bps must be between 0 and 100")
    if execution.commission_per_share < 0:
        raise ValueError("execution.commission_per_share must not be negative")

    return AppConfig(symbols=symbols, data=data, risk=risk, execution=execution)


def config_fingerprint(config: AppConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_dotenv_if_available() -> None:
    """Load a local .env when python-dotenv is installed; never require it for research."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def alpaca_credentials() -> tuple[str, str]:
    load_dotenv_if_available()
    key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_API_SECRET", "").strip()
    if not key or not secret:
        raise RuntimeError(
            "missing ALPACA_API_KEY or ALPACA_API_SECRET; copy .env.example to .env "
            "and use paper-account credentials"
        )
    return key, secret
