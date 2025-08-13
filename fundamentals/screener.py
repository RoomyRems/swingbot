# fundamentals/screener.py
from __future__ import annotations

import os
import time
import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd
from dotenv import load_dotenv

# Prefer Alpaca for price/avgvol (cheap + reliable)
from broker.alpaca import get_daily_bars

load_dotenv()

FMP_KEY = os.getenv("FMP_API_KEY", "")
MX_KEY  = os.getenv("MARKETAUX_API_KEY", "")

FMP_BASE = "https://financialmodelingprep.com/api/v3"
MX_BASE  = "https://api.marketaux.com/v1/news/all"

# ---- small util -------------------------------------------------------

def _today_utc_date() -> dt.date:
    return dt.datetime.utcnow().date()

def _iso_date(d: dt.date) -> str:
    return d.isoformat()

def _throttle(seconds: float = 0.10) -> None:
    """Sleep a bit to be nice with free tiers (used for FMP/MX; Alpaca is fine)."""
    if seconds > 0:
        time.sleep(seconds)

def _canon_for_fmp(sym: str) -> str:
    # FMP expects dashes for class shares, e.g. BRK.B -> BRK-B, BF.B -> BF-B
    return sym.upper().replace(".", "-").strip()

def _canon_for_mx(sym: str) -> str:
    # Marketaux typically accepts dot form; keep original unless you’ve seen issues.
    return sym.upper().strip()

def _get_json(url: str, params: Optional[dict] = None, timeout: int = 15, retries: int = 2) -> Optional[dict | list]:
    """
    GET with small retry/backoff for 429/5xx.

    Retries on request exceptions and returns ``None`` for any non-OK response.
    """
    backoff = 0.6
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                if i < retries:
                    time.sleep(backoff)
                    backoff *= 1.6
                    continue
            if r.ok:
                return r.json()
            return None
        except requests.RequestException:
            if i < retries:
                time.sleep(backoff)
                backoff *= 1.6
                continue
            return None
    return None

# ---- Alpaca (preferred) ----------------------------------------------

def _alpaca_price_avgvol(symbol: str) -> Optional[dict]:
    """
    Use Alpaca daily bars for last close + 50d average volume.
    Much more reliable than scraping quote endpoints elsewhere.
    """
    try:
        df = get_daily_bars(symbol, lookback_days=60)
        if df.empty or len(df) < 2:
            return None
        price = float(df["Close"].iloc[-1])
        avgv  = float(df["Volume"].tail(50).mean())
        if price <= 0 or avgv <= 0:
            return None
        return {"price": price, "avgVolume": avgv}
    except Exception:
        return None

# ---- FMP calls (used for earnings/sector/profitability) --------------

def fmp_quote(symbol: str) -> Optional[dict]:
    if not FMP_KEY:
        return None
    sym = _canon_for_fmp(symbol)
    url = f"{FMP_BASE}/quote/{sym}"
    js = _get_json(url, params={"apikey": FMP_KEY})
    if isinstance(js, list) and js:
        return js[0]
    return None

def fmp_profile(symbol: str) -> Optional[dict]:
    if not FMP_KEY:
        return None
    sym = _canon_for_fmp(symbol)
    url = f"{FMP_BASE}/profile/{sym}"
    js = _get_json(url, params={"apikey": FMP_KEY})
    if isinstance(js, list) and js:
        return js[0]
    return None

def fmp_earnings_window(symbol: str, days: int) -> bool:
    if not FMP_KEY or days <= 0:
        return False
    today = _today_utc_date()
    start = _iso_date(today - dt.timedelta(days=days))
    end   = _iso_date(today + dt.timedelta(days=days))
    sym = _canon_for_fmp(symbol)
    url = f"{FMP_BASE}/earning_calendar"
    js = _get_json(url, params={"symbol": sym, "from": start, "to": end, "apikey": FMP_KEY})
    return bool(js) if isinstance(js, list) else False

def fmp_eps_ttm(symbol: str) -> Optional[float]:
    # Try quote.eps first
    q = fmp_quote(symbol)
    _throttle()
    if q and "eps" in q and q["eps"] is not None:
        try:
            return float(q["eps"])
        except Exception:
            pass
    # Fallback: key-metrics-ttm → treat positive margin as "profitable"
    sym = _canon_for_fmp(symbol)
    url = f"{FMP_BASE}/key-metrics-ttm/{sym}"
    js = _get_json(url, params={"apikey": FMP_KEY})
    if isinstance(js, list) and js:
        margin = js[0].get("netProfitMarginTTM")
        if isinstance(margin, (int, float)) and margin > 0:
            return 0.01
    return None

def fmp_revenue_yoy_pct(symbol: str) -> Optional[float]:
    if not FMP_KEY:
        return None
    sym = _canon_for_fmp(symbol)
    url = f"{FMP_BASE}/income-statement/{sym}"
    js = _get_json(url, params={"limit": 2, "apikey": FMP_KEY})
    if not (isinstance(js, list) and len(js) >= 2):
        return None
    rev_cur = js[0].get("revenue")
    rev_pri = js[1].get("revenue")
    if not rev_cur or not rev_pri:
        return None
    try:
        return (float(rev_cur) / float(rev_pri) - 1.0) * 100.0
    except ZeroDivisionError:
        return None

# ---- Marketaux calls (optional) --------------------------------------

RED_FLAG_KEYWORDS = ("bankruptcy", "fraud", "going concern")

def mx_news_scan(symbol: str, lookback_days: int, min_sentiment: float, forbid_terms: List[str]) -> Tuple[bool, dict]:
    if not MX_KEY or lookback_days <= 0:
        return True, {"reason": "news-skip (no key or disabled)"}

    published_after = dt.datetime.utcnow() - dt.timedelta(days=lookback_days)
    params = {
        "symbols": _canon_for_mx(symbol),
        "filter_entities": "true",
        "published_after": published_after.strftime("%Y-%m-%dT%H:%M"),
        "limit": 50,
        "api_token": MX_KEY,
    }
    js = _get_json(MX_BASE, params=params)
    if not isinstance(js, dict):
        return True, {"reason": "news-fetch-failed"}

    articles = js.get("data", []) or []
    if not articles:
        return True, {"reason": "news-none"}

    forbid = [t.lower() for t in (forbid_terms or [])]
    for a in articles:
        text = f"{a.get('title','')} {a.get('description','')}".lower()
        if any(term in text for term in forbid + list(RED_FLAG_KEYWORDS)):
            return False, {"reason": "news-red-flag"}

    scores = []
    for a in articles:
        for key in ("overall_sentiment_score", "sentiment", "sentiment_score"):
            val = a.get(key)
            if isinstance(val, (int, float)):
                scores.append(float(val))
                break
    if scores:
        avg = sum(scores) / max(1, len(scores))
        if avg < float(min_sentiment):
            return False, {"reason": f"news-sentiment({avg:.2f})<min({min_sentiment})"}
        return True, {"reason": f"news-sentiment-ok({avg:.2f})"}
    return True, {"reason": "news-no-sentiment-fields"}

# ---- Public screen function ------------------------------------------

@dataclass
class ScreenResult:
    keep: bool
    reasons: Dict[str, str]  # details for logging/inspection

def screen_symbol(symbol: str, cfg: dict, throttle: float = 0.10) -> ScreenResult:
    """Apply fundamentals filters against one symbol."""
    fcfg = cfg.get("fundamentals", {}) or {}
    reasons: Dict[str, str] = {}

    # 1) price range + avg volume
    min_p   = float(fcfg.get("min_price", 0))
    max_p   = float(fcfg.get("max_price", math.inf))
    min_vol = float(fcfg.get("min_avg_vol", 0))

    # Prefer Alpaca for price/avgVol; fallback to FMP quote if needed
    q = _alpaca_price_avgvol(symbol)
    if not q and FMP_KEY:
        q = fmp_quote(symbol); _throttle(throttle)

    if not q:
        return ScreenResult(False, {"error": "quote-unavailable"})

    price = float(q.get("price") or q.get("previousClose") or q.get("close") or 0.0)
    avgv  = float(q.get("avgVolume") or q.get("avg_volume") or 0.0)

    if price <= 0:
        return ScreenResult(False, {"price": "invalid"})
    if not avgv or avgv <= 0:
        return ScreenResult(False, {"avg_vol": "missing"})

    if price < min_p or price > max_p:
        return ScreenResult(False, {"price": f"outside [{min_p},{max_p}]"})
    if avgv < min_vol:
        return ScreenResult(False, {"avg_vol": f"{avgv:.0f} < {min_vol:.0f}"})

    # 2) earnings blackout
    blackout_days = int(fcfg.get("earnings_blackout_days", 0))
    if blackout_days > 0:
        if fmp_earnings_window(symbol, blackout_days):
            _throttle(throttle)
            return ScreenResult(False, {"earnings": f"within +/-{blackout_days}d"})
        _throttle(throttle)

    # 3) sector allow/deny (if available)
    allow = set(s.strip().lower() for s in fcfg.get("sectors_allow", []))
    deny  = set(s.strip().lower() for s in fcfg.get("sectors_deny", []))
    if allow or deny:
        prof = fmp_profile(symbol); _throttle(throttle)
        sector = (prof or {}).get("sector", "")
        s_low  = str(sector).strip().lower()
        if deny and s_low in deny:
            return ScreenResult(False, {"sector": f"denied({sector})"})
        if allow and s_low not in allow:
            return ScreenResult(False, {"sector": f"not-allowed({sector})"})
        reasons["sector"] = sector or "n/a"

    # 4) basic profitability (optional)
    pcfg = fcfg.get("profitability", {}) or {}
    if pcfg.get("enabled", False):
        eps = fmp_eps_ttm(symbol); _throttle(throttle)
        if pcfg.get("require_positive_eps_ttm", False):
            if eps is None or eps <= 0:
                return ScreenResult(False, {"profit": "eps_ttm<=0 or missing"})
        min_yoy = float(pcfg.get("min_revenue_yoy_growth_pct", 0))
        yoy = fmp_revenue_yoy_pct(symbol); _throttle(throttle)
        if yoy is not None and yoy < min_yoy:
            return ScreenResult(False, {"revenue_yoy_pct": f"{yoy:.2f} < {min_yoy:.2f}"})
        if yoy is not None:
            reasons["revenue_yoy_pct"] = f"{yoy:.2f}"

    # 5) news / red-flag scan (optional)
    ncfg = fcfg.get("news", {}) or {}
    if ncfg.get("enabled", False):
        ok, info = mx_news_scan(
            symbol=symbol,
            lookback_days=int(ncfg.get("lookback_days", 3)),
            min_sentiment=float(ncfg.get("min_sentiment", -0.1)),
            forbid_terms=list(ncfg.get("forbid_red_flags", [])),
        ); _throttle(throttle)
        if not ok:
            return ScreenResult(False, {"news": info.get("reason", "blocked")})
        reasons["news"] = info.get("reason", "ok")

    # passed all
    reasons["price"]   = f"{price}"
    reasons["avg_vol"] = f"{avgv:.0f}"
    return ScreenResult(True, reasons)

def screen_universe(symbols: List[str], cfg: dict) -> Tuple[List[str], pd.DataFrame]:
    """
    Apply screen_symbol to a list; return (kept, report_df).
    """
    rows: List[dict] = []
    kept: List[str]  = []
    for s in symbols:
        res = screen_symbol(s, cfg)
        row = {"symbol": s, "keep": res.keep, **res.reasons}
        rows.append(row)
        if res.keep:
            kept.append(s)
    report = pd.DataFrame(rows)
    return kept, report
