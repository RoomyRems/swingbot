"""Generate a unified S&P 1500 + optional Russell 3000 watchlist by scraping Wikipedia.

Features:
  - Scrapes official component lists from Wikipedia for each selected index.
  - Cleans tickers (keeps alnum, '.', '-') and de‑duplicates across indices.
  - Optional Alpaca filter (active + tradable) if your broker credentials exist.

Usage (PowerShell):
    python scripts/generate_watchlist_sp1500.py               # full S&P 1500
    python scripts/generate_watchlist_sp1500.py --include 500 400 600 russell3000
  python scripts/generate_watchlist_sp1500.py --include 500 # only S&P 500
  python scripts/generate_watchlist_sp1500.py --alpaca-filter

Arguments:
  --out PATH              Output file path (default: project_root/watchlist.txt)
    --include 500 400 600 russell3000   Indices to include (default S&P 1500 only)
  --alpaca-filter         Apply Alpaca active & tradable symbol filter

Notes:
  - Falls back gracefully if Alpaca filter not available or fails.
  - Wikipedia HTML structure can change; basic heuristics used to locate the symbol column.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Iterable, List, Set

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:  # ensure local package imports (broker.alpaca) work
    sys.path.insert(0, PROJECT_ROOT)

WIKI_PAGES = {
    "500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    # Russell 3000 is indirectly the union of Russell 1000 + 2000; Wikipedia page lists all constituents
    "russell3000": "https://en.wikipedia.org/wiki/Russell_3000_Index",
}

_SYMBOL_CLEAN_RE = re.compile(r"[^A-Z0-9.\-]", re.IGNORECASE)


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
    """Return the first table whose header contains 'Symbol' or 'Ticker'."""
    for table in soup.find_all("table"):
        headers: list[str] = []
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
    symbols: List[str] = []
    header_row = table.find("tr")
    header_cells = header_row.find_all(["th", "td"]) if header_row else []
    col_idx = None
    for idx, cell in enumerate(header_cells):
        h = cell.get_text(strip=True).lower()
        if "symbol" in h or "ticker" in h:
            col_idx = idx
            break
    if col_idx is None:
        col_idx = 0  # fallback
    for tr in table.find_all("tr")[1:]:  # skip header
        tds = tr.find_all(["td", "th"])  # sometimes mis-tagged
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


def default_output_path() -> str:
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, os.pardir))
    return os.path.join(root, "watchlist.txt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate S&P 1500 (+ Russell 3000 optional) watchlist from Wikipedia")
    parser.add_argument("--out", dest="out", default=default_output_path(), help="Output file path")
    parser.add_argument(
        "--include",
        nargs="*",
    default=["500", "400", "600"],
    choices=["500", "400", "600", "russell3000"],
    help="Indices to include (default: 500 400 600). Add russell3000 to expand universe.",
    )
    parser.add_argument(
        "--alpaca-filter",
        action="store_true",
        help="Filter to active & tradable symbols using Alpaca",
    )
    args = parser.parse_args(argv)

    urls = [WIKI_PAGES[i] for i in args.include]
    all_syms: List[str] = []
    for url in urls:
        print(f"Fetching: {url}")
        try:
            syms = parse_symbols_from_wikipedia(url)
            print(f"  -> {len(syms)} symbols")
            all_syms.extend(syms)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] Failed to parse {url}: {e}")

    # Even though unique_sorted de-dupes, we perform an explicit final duplicate scan for transparency.
    syms = unique_sorted(all_syms)
    print(f"Total unique before filter: {len(syms)}")
    # Final duplicate safety (in case of future modifications bypassing unique_sorted)
    # Scan original combined list for duplicates
    dup_counts = {}
    for s in all_syms:
        dup_counts[s] = dup_counts.get(s, 0) + 1
    dups = [s for s,cnt in dup_counts.items() if cnt > 1]
    if dups:
        print(f"Found {len(dups)} symbols appearing in multiple source lists (e.g. first 10): {dups[:10]}")
    if args.alpaca_filter:
        syms = try_alpaca_filter(syms)
        print(f"After Alpaca filter: {len(syms)}")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in syms:
            f.write(s + "\n")
    print(f"Wrote {len(syms)} symbols to {out_path}")
    if "russell3000" in args.include:
        print("Note: Russell 3000 adds substantial breadth; validate fundamentals/rate limits accordingly.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
