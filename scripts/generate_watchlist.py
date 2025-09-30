"""Unified watchlist generator.

Reads desired universes from config.yaml (watchlist.universes) and produces a de-duplicated
`watchlist.txt` file at project root. Supports:
  - sp500 (S&P 500)
  - sp1000 (S&P 1000 = 400 mid + 600 small)
  - sp1500 (default = 500 + 400 + 600)
  - russell3000 (direct scrape) + optional IWV ETF holdings merge
  - all (expands to every supported set)

Config knobs (config.yaml):
watchlist:
  universes: ["sp1500", "russell3000"]
  alpaca_filter: false
  include_iwv: false
  iwv_ttl_days: 2
  iwv_force_refresh: false

CLI override options are also provided.

Usage:
  python scripts/generate_watchlist.py                # use config
  python scripts/generate_watchlist.py --universes sp500 russell3000 --alpaca-filter
  python scripts/generate_watchlist.py --out custom_watchlist.txt

"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Iterable, List, Set

import requests
from bs4 import BeautifulSoup

# Local imports (ensure project root on path)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_config  # type: ignore

WIKI_PAGES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    "russell3000": "https://en.wikipedia.org/wiki/Russell_3000_Index",
}

_SYMBOL_CLEAN_RE = re.compile(r"[^A-Z0-9.\-]", re.IGNORECASE)

SUPPORTED_UNIVERSES = {"sp500", "sp1000", "sp1500", "russell3000", "all"}


def _http_get(url: str, timeout: int = 25) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _pick_table_with_ticker(soup: BeautifulSoup):
    for table in soup.find_all("table"):
        headers = []
        thead = table.find("thead")
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        else:
            first_tr = table.find("tr")
            if first_tr:
                headers = [th.get_text(strip=True) for th in first_tr.find_all(["th", "td"])]
        if not headers:
            continue
        joined = " ".join(h.lower() for h in headers)
        if "symbol" in joined or "ticker" in joined:
            return table
    return None


def _clean_symbol(raw: str) -> str | None:
    if not raw:
        return None
    s = raw.strip().upper()
    s = _SYMBOL_CLEAN_RE.sub("", s)
    return s or None


def parse_symbols_from_wikipedia(url: str) -> List[str]:
    html = _http_get(url)
    soup = BeautifulSoup(html, "html.parser")
    table = _pick_table_with_ticker(soup)
    if not table:
        raise RuntimeError(f"No ticker table found at {url}")
    header_row = table.find("tr")
    header_cells = header_row.find_all(["th", "td"]) if header_row else []
    col_idx = None
    for idx, cell in enumerate(header_cells):
        h = cell.get_text(strip=True).lower()
        if "symbol" in h or "ticker" in h:
            col_idx = idx
            break
    if col_idx is None:
        col_idx = 0
    symbols: List[str] = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all(["td", "th"])
        if not tds or len(tds) <= col_idx:
            continue
        raw = tds[col_idx].get_text(strip=True)
        sym = _clean_symbol(raw)
        if sym:
            symbols.append(sym)
    return symbols


def unique_sorted(symbols: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return sorted(out)


def try_alpaca_filter(symbols: List[str]) -> List[str]:
    try:
        from broker.alpaca import filter_active_tradable  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Alpaca not available ({e}); skipping filter.")
        return symbols
    try:
        filtered = filter_active_tradable(symbols)
        if not filtered:
            print("[warn] Alpaca filter returned 0 symbols; using unfiltered list.")
            return symbols
        return filtered
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Alpaca filter failed: {e}; using unfiltered list.")
        return symbols


def _expand_universes(requested: List[str]) -> List[str]:
    req_set = set(u.lower() for u in requested)
    if "all" in req_set:
        return ["sp500", "sp400", "sp600", "russell3000"]
    expanded: List[str] = []
    for u in req_set:
        if u == "sp1500":
            expanded.extend(["sp500", "sp400", "sp600"])
        elif u == "sp1000":
            expanded.extend(["sp400", "sp600"])
        elif u in ("sp500", "sp400", "sp600", "russell3000"):
            expanded.append(u)
        else:
            print(f"[warn] Unsupported universe '{u}' ignored.")
    # de-dupe while preserving order
    seen = set()
    out: List[str] = []
    for x in expanded:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def default_output_path() -> str:
    return os.path.join(PROJECT_ROOT, "watchlist.txt")


def generate(universes: List[str], alpaca_filter: bool, include_iwv: bool, iwv_ttl_days: int, iwv_force_refresh: bool) -> List[str]:
    expanded = _expand_universes(universes)
    print(f"Expanded universes: {expanded}")
    all_syms: List[str] = []
    for key in expanded:
        url = WIKI_PAGES.get(key)
        if not url:
            print(f"[warn] No URL for {key}; skipping")
            continue
        print(f"Fetching {key}: {url}")
        try:
            syms = parse_symbols_from_wikipedia(url)
            print(f"  -> {len(syms)} symbols")
            all_syms.extend(syms)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] Failed to fetch {key}: {e}")
    syms = unique_sorted(all_syms)
    print(f"Total unique before filter: {len(syms)}")

    if alpaca_filter:
        syms = try_alpaca_filter(syms)
        print(f"After Alpaca filter: {len(syms)}")

    if include_iwv:
        try:
            from universe.iwv import get_iwv_constituents, merge_watchlists  # type: ignore
            iwv_list = get_iwv_constituents(ttl_days=iwv_ttl_days, force_refresh=iwv_force_refresh)
            if not iwv_list:
                print("[warn] IWV holdings empty; skipping merge")
            else:
                pre = len(syms)
                merged = merge_watchlists(syms, iwv_list)
                print(f"IWV holdings: {len(iwv_list)}; merged delta: {len(merged) - pre}")
                syms = merged
        except Exception as e:  # noqa: BLE001
            print(f"[warn] IWV merge failed: {e}")

    return syms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate configurable multi-index watchlist")
    parser.add_argument("--universes", nargs="*", help="Universes to include (override config)")
    parser.add_argument("--out", default=default_output_path(), help="Output file path")
    parser.add_argument("--alpaca-filter", action="store_true", help="Apply Alpaca active/tradable filter")
    parser.add_argument("--include-iwv", action="store_true", help="Merge IWV holdings (Russell proxy)")
    parser.add_argument("--iwv-ttl-days", type=int, default=None, help="IWV cache TTL days (override config)")
    parser.add_argument("--iwv-force-refresh", action="store_true", help="Force IWV cache refresh")
    args = parser.parse_args(argv)

    cfg = load_config()
    wcfg = (cfg.get("watchlist") or {})
    universes = args.universes if args.universes else list(wcfg.get("universes", ["sp1500"]))
    alpaca_filter = args.alpaca_filter or bool(wcfg.get("alpaca_filter", False))
    include_iwv = args.include_iwv or bool(wcfg.get("include_iwv", False))
    iwv_ttl_days = args.iwv_ttl_days if args.iwv_ttl_days is not None else int(wcfg.get("iwv_ttl_days", 2))
    iwv_force_refresh = args.iwv_force_refresh or bool(wcfg.get("iwv_force_refresh", False))

    syms = generate(universes, alpaca_filter, include_iwv, iwv_ttl_days, iwv_force_refresh)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in syms:
            f.write(s + "\n")
    print(f"Wrote {len(syms)} symbols to {out_path}")
    if "russell3000" in [u.lower() for u in universes] or include_iwv:
        print("Note: Expanded universe may amplify data/API usage. Monitor rate limits.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
