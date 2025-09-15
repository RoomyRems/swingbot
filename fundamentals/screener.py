# fundamentals/screener.py
from __future__ import annotations

import os
import time
import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import requests
import pandas as pd
from dotenv import load_dotenv

# Prefer Alpaca for price/avgvol (cheap + reliable)
from broker.alpaca import get_daily_bars
from utils.file_cache import FileCache

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

def _get_backtest_network_mode(cfg: dict) -> str:
    try:
        return (cfg.get("fundamentals", {})
                  .get("backtest_mode", {})
                  .get("network", "off"))
    except Exception:
        return "off"

def _is_live_ok(cfg: dict) -> bool:
    return _get_backtest_network_mode(cfg) == "live_ok"

def _is_backtest_prefetch_only(cfg: dict) -> bool:
    return _get_backtest_network_mode(cfg) == "prefetch_only"

def _min_avg_vol(cfg: dict) -> int:
    """Return the technical min avg volume (single source of truth).

    Fundamentals-specific min_avg_vol has been removed; rely solely on
    trading.filters.min_avg_vol50. Fallback to 300k if unset.
    """
    tfv = (cfg.get("trading", {}).get("filters", {}) or {}).get("min_avg_vol50")
    return int(tfv or 300000)

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

# ---- request counters (observability) --------------------------------
REQUEST_COUNTER: Dict[str, int] = {
    "fmp": 0,
    "mx": 0,
    "alpaca": 0,
    "cache_hit": 0,
}

def _inc(key: str, n: int = 1) -> None:
    try:
        REQUEST_COUNTER[key] = REQUEST_COUNTER.get(key, 0) + int(n)
    except Exception:
        pass

# ---- cached JSON fetcher ---------------------------------------------

def _cached_get_json(url: str, params: Optional[dict], cache: Optional[FileCache], vendor: str, network: str = "live_ok") -> Optional[dict | list]:
    """
    vendor: 'fmp' | 'mx' (used for counters)
    network:
      - 'off'            -> never hit network; return cache if present else None
      - 'prefetch_only'  -> allowed during prefetch (callers should only use in prefetch stage)
      - 'live_ok'        -> may call network freely
    """
    key = None
    if cache is not None:
        try:
            key = f"{url}?{sorted(params.items()) if params else ''}"
            v = cache.get(key)
            if v is not None:
                _inc("cache_hit")
                return v
        except Exception:
            pass
    if network == "off":
        return None
    # network allowed
    js = _get_json(url, params=params)
    if js is not None and cache is not None and key is not None:
        try:
            cache.set(key, js)
        except Exception:
            pass
    if vendor in ("fmp","mx"):
        _inc(vendor)
    return js

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

def fmp_earnings_window(symbol: str, days: int, ref_date: Optional[dt.date] = None) -> bool:
    if not FMP_KEY or days <= 0:
        return False
    base = ref_date or _today_utc_date()
    start = _iso_date(base - dt.timedelta(days=days))
    end   = _iso_date(base + dt.timedelta(days=days))
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

def mx_news_scan(symbol: str, lookback_days: int, min_sentiment: float, forbid_terms: List[str], ref_datetime: Optional[dt.datetime] = None) -> Tuple[bool, dict]:
    if not MX_KEY or lookback_days <= 0:
        return True, {"reason": "news-skip (no key or disabled)"}

    now_ref = ref_datetime or dt.datetime.utcnow()
    published_after = now_ref - dt.timedelta(days=lookback_days)
    params = {
        "symbols": _canon_for_mx(symbol),
        "filter_entities": "true",
        "published_after": published_after.strftime("%Y-%m-%dT%H:%M"),
        # Try to bound by simulated time to reduce lookahead; API may ignore if unsupported
        "published_before": now_ref.strftime("%Y-%m-%dT%H:%M"),
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

# ---- Backtest prefetch helpers --------------------------------------

def init_fundamentals_cache(cfg: dict) -> FileCache:
    fcfg = (cfg.get("fundamentals", {}) or {})
    btcfg = (fcfg.get("backtest_mode", {}) or {})
    cache_dir = str(btcfg.get("cache_dir", os.path.join("cache", "fundamentals")))
    ttl_days = int(btcfg.get("ttl_days", 14))
    return FileCache(cache_dir, ttl_days=ttl_days)

def prefetch_earnings_calendar(symbols: List[str], start_date: dt.date, end_date: dt.date, cache: FileCache, cfg: dict) -> Dict[str, List[dt.date]]:
    if not FMP_KEY:
        return {s: [] for s in symbols}
    fcfg = (cfg.get("fundamentals", {}) or {})
    btcfg = (fcfg.get("backtest_mode", {}) or {})
    network = str(btcfg.get("network", "prefetch_only"))
    out: Dict[str, List[dt.date]] = {}
    for s in symbols:
        sym = _canon_for_fmp(s)
        url = f"{FMP_BASE}/earning_calendar"
        params = {
            "symbol": sym,
            "from": _iso_date(start_date),
            "to": _iso_date(end_date),
            "apikey": FMP_KEY,
        }
        js = _cached_get_json(url, params, cache, vendor="fmp", network=network)
        dates: List[dt.date] = []
        if isinstance(js, list):
            for row in js:
                for k in ("date","reportedDate","epsDate","fiscalDateEnding"):
                    v = row.get(k)
                    if v:
                        try:
                            d = dt.datetime.fromisoformat(str(v).split("T")[0]).date()
                            dates.append(d)
                            break
                        except Exception:
                            continue
        out[s] = sorted(set(dates))
    return out

def build_fund_ctx(cfg: dict, symbols: List[str], start_date: dt.date, end_date: dt.date) -> Dict[str, Any]:
    fcfg = (cfg.get("fundamentals", {}) or {})
    if not fcfg.get("earnings_blackout", False):
        # Earnings disabled: return minimal ctx (still include request counter for diagnostics)
        return {"earnings_calendar": {}, "blackout_days": 0, "request_counter": dict(REQUEST_COUNTER)}
    blackout = int(fcfg.get("earnings_blackout_days", 0))
    cache = init_fundamentals_cache(cfg)
    # pad range by blackout window to cover near-boundary fills
    pad = max(blackout, 0)
    s = start_date - dt.timedelta(days=pad)
    e = end_date + dt.timedelta(days=pad)
    cal = prefetch_earnings_calendar(symbols, s, e, cache, cfg)
    return {
        "earnings_calendar": cal,
        "blackout_days": blackout,
        "request_counter": dict(REQUEST_COUNTER),
    }

def earnings_in_window(symbol: str, day: dt.date, blackout_days: int, ctx: Dict[str, Any]) -> bool:
    cal = (ctx or {}).get("earnings_calendar", {}) or {}
    dates = cal.get(symbol, []) or []
    if blackout_days <= 0 or not dates:
        return False
    for d in dates:
        if abs((day - d).days) <= blackout_days:
            return True
    return False

def fundamentals_pass_at_fill(symbol: str, fill_date: dt.date, cfg: dict, ctx: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    fcfg = (cfg.get("fundamentals", {}) or {})
    if not fcfg.get("earnings_blackout", False):
        return True, "earnings_disabled"
    blackout = int(fcfg.get("earnings_blackout_days", 0))
    if blackout <= 0:
        return True, "no-blackout"
    if ctx is None:
        # offline path: fail open by default in backtests
        return True, "no-ctx"
    blocked = earnings_in_window(symbol, fill_date, blackout, ctx)
    if blocked:
        return False, f"earnings_blackout(+/-{blackout}d)"
    return True, "ok"

# ---- Public screen function ------------------------------------------

@dataclass
class ScreenResult:
    keep: bool
    reasons: Dict[str, str]  # details for logging/inspection

def screen_symbol(
    symbol: str,
    cfg: dict,
    throttle: float = 0.10,
    asof_date: Optional[dt.date] = None,
    asof_datetime: Optional[dt.datetime] = None,
    price_override: Optional[float] = None,
    avgvol_override: Optional[float] = None,
) -> ScreenResult:
    """Apply fundamentals filters against one symbol."""
    fcfg = cfg.get("fundamentals", {}) or {}
    reasons: Dict[str, str] = {}

    # 1) price range (volume gating removed; enforced by trading.filters.min_avg_vol50 in engine)
    min_p   = float(fcfg.get("min_price", 0))
    max_p   = float(fcfg.get("max_price", math.inf))

    # Prefer overrides from caller (e.g., backtest) to avoid lookahead and network
    q: Optional[dict]
    if price_override is not None and avgvol_override is not None:
        q = {"price": float(price_override), "avgVolume": float(avgvol_override)}
    else:
        q = _alpaca_price_avgvol(symbol)
        if not q and FMP_KEY:
            q = fmp_quote(symbol); _throttle(throttle)

    if not q:
        return ScreenResult(False, {"error": "quote-unavailable"})

    price = float(q.get("price") or q.get("previousClose") or q.get("close") or 0.0)
    avgv  = float(q.get("avgVolume") or q.get("avg_volume") or 0.0)

    if price <= 0:
        return ScreenResult(False, {"price": "invalid"})
    if price < min_p or price > max_p:
        return ScreenResult(False, {"price": f"outside [{min_p},{max_p}]"})

    # 2) earnings blackout
    blackout_days = int(fcfg.get("earnings_blackout_days", 0))
    if blackout_days > 0:
        if fmp_earnings_window(symbol, blackout_days, ref_date=asof_date):
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
            ref_datetime=asof_datetime,
        ); _throttle(throttle)
        if not ok:
            return ScreenResult(False, {"news": info.get("reason", "blocked")})
        reasons["news"] = info.get("reason", "ok")

    # passed all
    reasons["price"]   = f"{price}"
    if avgv > 0:
        reasons["avg_vol"] = f"{avgv:.0f}"
    else:
        reasons["avg_vol"] = "n/a"
    return ScreenResult(True, reasons)

def screen_universe(symbols: List[str], cfg: dict) -> Tuple[List[str], pd.DataFrame]:
    """Backtest-safe universe pre-screen.

        - Fail-open on any API/network error when not in live_ok mode.
        - Avoid per-symbol network calls unless live_ok.
        - Volume gating has been fully removed from fundamentals; liquidity is
            enforced ONLY inside the trading engine via trading.filters.min_avg_vol50.
        - Apply an optional price floor (if prescreen_price enabled).
        - Skip earnings blackout here (engine enforces at fill time via prefetched ctx).
        - Skip news in backtests (unless explicitly enabled + live_ok).
        Returns (kept_symbols, report_df) and writes an audit CSV to logs/.
    """
    import random
    def _tiny_jitter():
        time.sleep(random.uniform(0.075, 0.150))

    live_ok = _is_live_ok(cfg)
    prefetch_only = _is_backtest_prefetch_only(cfg)
    fail_open = not live_ok
    use_news = bool(((cfg.get("fundamentals", {}).get("news", {}) or {}).get("enabled", False)) and live_ok)
    min_price = float((cfg.get("fundamentals", {}) or {}).get("min_price", 0))
    # Renamed from prescreen_price_volume -> prescreen_price (backwards compat: fall back to old key if new absent)
    fcfg_local = (cfg.get("fundamentals", {}) or {})
    prescreen_pv = bool(fcfg_local.get("prescreen_price", fcfg_local.get("prescreen_price_volume", True)))

    # Deprecation warning if legacy fundamentals.min_avg_vol still present
    if "min_avg_vol" in fcfg_local:
        try:
            print("[Fundamentals] WARNING: 'fundamentals.min_avg_vol' is deprecated & ignored; use trading.filters.min_avg_vol50.")
        except Exception:
            pass

    drop_log: List[dict] = []
    kept: List[str] = []

    # Optional progress bar when fundamentals screening is explicitly enabled
    from tqdm import tqdm  # lightweight; already used elsewhere (backtest)
    it = symbols
    f_enabled = bool((cfg.get("fundamentals", {}) or {}).get("enabled", False))
    if f_enabled:
        try:
            it = tqdm(symbols, desc="Fundamentals", unit="sym")
        except Exception:
            it = symbols

    for sym in it:
        try:
            price = None
            avgvol = None

            # Price/avgvol: avoid network in backtests
            if live_ok and prescreen_pv:
                q = _alpaca_price_avgvol(sym)
                _tiny_jitter()
                if q:
                    price = float(q.get("price")) if q.get("price") is not None else None
                    avgvol = float(q.get("avgVolume")) if q.get("avgVolume") is not None else None
            # Apply min price check (volume deliberately ignored here)
            price_ok = True if (not live_ok or not prescreen_pv or price is None) else (price >= min_price)
            if not price_ok:
                drop_log.append({"symbol": sym, "kept": False, "reason": f"price_below_min({price}<{min_price})"})
                continue

            # News only in live_ok and enabled
            if use_news:
                ncfg = (cfg.get("fundamentals", {}).get("news", {}) or {})
                try:
                    ok_news, info = mx_news_scan(
                        symbol=sym,
                        lookback_days=int(ncfg.get("lookback_days", 3)),
                        min_sentiment=float(ncfg.get("min_sentiment", -0.1)),
                        forbid_terms=list(ncfg.get("forbid_red_flags", [])),
                        ref_datetime=None,
                    )
                    _tiny_jitter()
                except Exception as e:
                    if fail_open:
                        ok_news = True
                        info = {"reason": f"news_api_error_fail_open:{type(e).__name__}"}
                    else:
                        raise
                if not ok_news:
                    drop_log.append({"symbol": sym, "kept": False, "reason": info.get("reason", "news_block")})
                    continue

            # Earnings blackout skipped here (enforced at fill via engine prefetch)
            kept.append(sym)
            drop_log.append({"symbol": sym, "kept": True, "reason": "kept"})
        except Exception as e:
            if fail_open:
                kept.append(sym)
                drop_log.append({"symbol": sym, "kept": True, "reason": f"api_error_fail_open:{type(e).__name__}"})
            else:
                drop_log.append({"symbol": sym, "kept": False, "reason": f"api_error_fail_closed:{type(e).__name__}"})

    # Write screen log CSV
    try:
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out = os.path.join("logs", f"fundamentals_screen_{ts}.csv")
        os.makedirs("logs", exist_ok=True)
        pd.DataFrame(drop_log, columns=["symbol","kept","reason"]).to_csv(out, index=False)
        print(f"[Fundamentals] Wrote screen log → {out}")
    except Exception:
        pass
    print(f"[Fundamentals] Kept {len(kept)} / {len(symbols)} symbols (mode={_get_backtest_network_mode(cfg)}, liquidity=engine_only, min_price={min_price}, news={'ON' if use_news else 'OFF'})")
    report = pd.DataFrame(drop_log, columns=["symbol","kept","reason"])  # compatibility
    return kept, report
