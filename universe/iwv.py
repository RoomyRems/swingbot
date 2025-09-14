"""IWV (iShares Russell 3000 ETF) holdings helper.

Provides a *current* Russell 3000 proxy via the IWV ETF holdings CSV.

CAUTION: This is a point-in-time snapshot (survivorship bias) and should
not be used for strict historical (point-in-time) backtests requiring
exact historical membership. For that you need a licensed dataset.
"""
from __future__ import annotations

import hashlib
import io
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

__all__ = [
    "get_iwv_constituents",
    "merge_watchlists",
]

# ------------------- Normalization -------------------

def _normalize_us_ticker(t: str) -> Optional[str]:
    """Normalize iShares holdings tickers to standard US forms.

    Examples:
      BRK/B -> BRK.B, BF/B -> BF.B
    Remove unit/warrant/when-issued suffixes (.U .UN .UW .WI .WS)
    Strip spaces; keep only [A-Z0-9.-]
    """
    if not isinstance(t, str):
        return None
    t = t.strip().upper()
    if not t or t in {"—", "-", "N/A", "NA"}:
        return None
    t = t.replace("/", ".")
    t = re.sub(r"\.(U|UN|UW|WI|WS)$", "", t)
    t = t.rstrip(".")
    t = re.sub(r"[^A-Z0-9.\-]", "", t)
    return t or None

# ------------------- CSV Parsing -------------------

def _read_iwv_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    asset_col = next((c for c in df.columns if c.strip().lower() == "asset class"), None)
    if asset_col:
        df = df[df[asset_col].astype(str).str.contains("Equity", case=False, na=False)]
    ticker_col = None
    for cand in ["Ticker", "Ticker Symbol", "Ticker symbol", "Symbol"]:
        if cand in df.columns:
            ticker_col = cand
            break
    if not ticker_col:
        raise ValueError("Could not find a 'Ticker' column in IWV holdings CSV")
    df = df[[ticker_col]].rename(columns={ticker_col: "Ticker"})
    df["Ticker"] = df["Ticker"].map(_normalize_us_ticker)
    df = df.dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return df

# ------------------- HTTP w/ retries -------------------

def _http_get_with_retries(url: str, headers: dict, retries: int = 3, backoff: float = 1.5, timeout: int = 20) -> str:
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.content.decode("utf-8-sig", errors="ignore")
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise last_exc  # type: ignore[misc]

# ------------------- Public API -------------------

def get_iwv_constituents(
    cache_dir: str | Path = "cache/iwv",
    ttl_days: int = 2,
    force_refresh: bool = False,
) -> List[str]:
    """Download IWV holdings, cache to disk, and return clean tickers.

    Parameters
    ----------
    cache_dir : path-like
        Directory to store raw downloaded CSV.
    ttl_days : int
        Days to keep cached file before refreshing.
    force_refresh : bool
        Ignore cache and force a fresh download.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = (
        "https://www.ishares.com/us/products/239707/ishares-russell-3000-etf/"
        "1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
    )
    key = hashlib.md5(url.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"iwv_holdings_{key}.csv"

    if cache_file.exists() and not force_refresh:
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime < timedelta(days=ttl_days):
            text = cache_file.read_text(encoding="utf-8")
            df = _read_iwv_csv(text)
            return sorted(df["Ticker"].tolist())

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/csv, */*;q=0.1",
        "Referer": "https://www.ishares.com/",
    }
    text = _http_get_with_retries(url, headers=headers)
    cache_file.write_text(text, encoding="utf-8")
    df = _read_iwv_csv(text)
    return sorted(df["Ticker"].tolist())


def merge_watchlists(*lists: Iterable[str]) -> List[str]:
    s: set[str] = set()
    for lst in lists:
        for t in lst:
            if isinstance(t, str) and t.strip():
                s.add(t.strip().upper())
    return sorted(s)
