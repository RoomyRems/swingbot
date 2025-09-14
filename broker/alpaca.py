"""
broker.alpaca
-------------
Thin wrapper around Alpaca Trading API for data + bracket orders.
"""

from __future__ import annotations
import os
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from functools import lru_cache
import threading, time
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
import pandas as pd
from alpaca_trade_api import REST
from alpaca_trade_api.rest import TimeFrame

# ---- env & client ----------------------------------------------------
load_dotenv()  # read .env

API_KEY    = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
ENDPOINT   = os.getenv("ALPACA_PAPER_ENDPOINT", "https://paper-api.alpaca.markets")

_ALPACA_DISABLED = False
if not API_KEY or not API_SECRET:
    if "PYTEST_CURRENT_TEST" in os.environ:
        _ALPACA_DISABLED = True
        class _DummyAPI:
            def get_account(self):
                class _Acct: equity = 0; buying_power = 0
                return _Acct()
            def get_latest_trade(self, symbol):
                class _Trade: price = 0; p = 0
                return _Trade()
            def submit_order(self, **kwargs):
                class _Order: id = "DUMMY"
                return _Order()
            def get_bars(self, *a, **k):
                import pandas as _pd
                class _Bars:
                    df = _pd.DataFrame(columns=["open","high","low","close","volume"])
                return _Bars()
            def list_assets(self, *a, **k):
                return []
            def list_positions(self):
                return []
            def list_orders(self, *a, **k):
                return []
        api = _DummyAPI()  # type: ignore
    else:
        raise EnvironmentError("Missing ALPACA_API_KEY or ALPACA_API_SECRET in .env")
else:
    api = REST(key_id=API_KEY, secret_key=API_SECRET, base_url=ENDPOINT, api_version="v2")

    # ---- simple rate limiting & retry -----------------------------------
    _RL_MAX_PER_MIN = 180  # soft cap (leave headroom below 200/min)
    _RL_RATE_PER_SEC = _RL_MAX_PER_MIN / 60.0
    _RL_TOKENS = float(_RL_MAX_PER_MIN)
    _RL_LAST_REFILL = time.time()
    _RL_LOCK = threading.Lock()
    _RL_429_COUNT = 0
    _RL_TOTAL_CALLS = 0
    _RL_RETRIES = 0

    def _rate_acquire():
        """Leaky-bucket style limiter.

        Allows smoother distribution instead of a hard minute cliff that caused
        ~60s stalls after the first ~180 calls. Refill occurs continuously based
        on elapsed time. If <1 token available, sleeps just enough to accrue it.
        """
        global _RL_TOKENS, _RL_LAST_REFILL
        while True:
            with _RL_LOCK:
                now = time.time()
                elapsed = now - _RL_LAST_REFILL
                if elapsed > 0:
                    _RL_TOKENS = min(float(_RL_MAX_PER_MIN), _RL_TOKENS + elapsed * _RL_RATE_PER_SEC)
                    _RL_LAST_REFILL = now
                if _RL_TOKENS >= 1.0:
                    _RL_TOKENS -= 1.0
                    return
                # need to wait for deficit
                deficit = 1.0 - _RL_TOKENS
                wait = deficit / _RL_RATE_PER_SEC
            # outside lock
            if wait > 0.0:
                if wait > 0.25 and os.getenv("SB_RATE_LIMIT_VERBOSE"):
                    print(f"[rate-limit] sleeping {wait:.2f}s (tokens={_RL_TOKENS:.2f})")
                time.sleep(wait)

    def _alpaca_call(fn, *a, **kw):
        """Wrap Alpaca SDK call with client-side rate limiting + retry for 429/timeouts.
        Tracks counters for diagnostics consumed by backtest summary.
        """
        global _RL_429_COUNT, _RL_TOTAL_CALLS, _RL_RETRIES
        attempts = 0
        backoff = 0.6
        while True:
            attempts += 1
            _rate_acquire()
            try:
                res = fn(*a, **kw)
                with _RL_LOCK:
                    _RL_TOTAL_CALLS += 1
                return res
            except Exception as e:  # broad catch; filter on message
                msg = str(e).lower()
                transient = ("429" in msg) or ("too many" in msg) or ("timeout" in msg) or ("rate" in msg)
                if transient:
                    with _RL_LOCK: _RL_429_COUNT += 1
                if transient and attempts < 4:
                    _RL_RETRIES += 1
                    time.sleep(backoff)
                    backoff *= 1.6
                    continue
                # propagate final failure
                with _RL_LOCK:
                    _RL_TOTAL_CALLS += 1
                raise

    def get_rate_limit_counts() -> dict:
        with _RL_LOCK:
            return {
                "total_calls": _RL_TOTAL_CALLS,
                "retries": _RL_RETRIES,
                "status_429": _RL_429_COUNT,
                "max_per_min": _RL_MAX_PER_MIN,
            }


# ---- account helpers -------------------------------------------------
def get_equity() -> float:
    return float(api.get_account().equity)


def get_latest_price(symbol: str) -> float:
    """Best-effort last price (trade) with retry & throttling."""
    try:
        lt = _alpaca_call(api.get_latest_trade, symbol)
        return float(getattr(lt, "price", getattr(lt, "p", 0.0)) or 0.0)
    except Exception:
        return 0.0


# ---- tick / price helpers -------------------------------------------
def _safe_decimal(x: float | int | str) -> Decimal:
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError):
        return Decimal(0)

def _round_to_tick(price: float | Decimal, tick: Decimal) -> Decimal:
    """Round price to a valid tick using half-up rounding."""
    p = _safe_decimal(price)
    # quantize to tick step (e.g., Decimal("0.01") or Decimal("0.0001"))
    return (p / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick

@lru_cache(maxsize=2048)
def _tick_size_for_symbol(symbol: str) -> Decimal:
    """
    Decide a tick size by looking at the latest trade price:
    - If last trade >= 1.00 → $0.01
    - Else (sub-dollar)     → $0.0001
    (Replace with asset.min_order_price_increment if you enable fractional ticks.)
    """
    try:
        last = get_latest_price(symbol)
    except Exception:
        last = 10.0  # sensible default if API hiccups

    if last >= 1.0:
        return Decimal("0.01")
    else:
        return Decimal("0.0001")


# ---- order submission ------------------------------------------------
def place_bracket(
    symbol: str,
    qty: int,
    side: str,                       # "buy" or "sell"
    take_profit: float,
    stop_loss: float,
    time_in_force: str = "day",
    order_type: str = "market",      # "market" or "limit"
    entry_limit: Optional[float] = None,  # required if order_type == "limit" (if None -> fallback to latest)
) -> str:
    """
    Submit a bracket order:
      - entry: market OR limit (when order_type == 'limit')
      - exits: take_profit (limit) + stop_loss (stop)
    Returns Alpaca order_id (str).

    NOTE: If you use 'limit', pass a computed limit price via `entry_limit`
          (e.g., breakout/pullback logic from your strategy).
          If omitted, we'll fallback to latest trade price.
    """
    side = side.lower().strip()
    if side not in {"buy", "sell"}:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    order_type = order_type.lower().strip()
    if order_type not in {"market", "limit"}:
        raise ValueError("order_type must be 'market' or 'limit'")

    # Normalize prices to valid tick
    tick = _tick_size_for_symbol(symbol)
    tp = _round_to_tick(take_profit, tick)
    sl = _round_to_tick(stop_loss, tick)

    # Basic sanity for LONG/SHORT brackets
    if side == "buy":
        if not (tp > sl):
            raise ValueError(f"[{symbol}] invalid exits for LONG: TP({tp}) must be > SL({sl})")
    else:  # short
        if not (tp < sl):
            raise ValueError(f"[{symbol}] invalid exits for SHORT: TP({tp}) must be < SL({sl})")

    submit_kwargs = dict(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=time_in_force,
        order_class="bracket",
        take_profit={"limit_price": float(tp)},
        stop_loss={"stop_price": float(sl)},
    )

    if order_type == "market":
        submit_kwargs["type"] = "market"
    else:
        # limit entry
        lim = entry_limit if entry_limit is not None else get_latest_price(symbol)
        lim_rounded = _round_to_tick(lim, tick)
        submit_kwargs["type"] = "limit"
        submit_kwargs["limit_price"] = float(lim_rounded)

        # Optional safety: for longs, don't let limit be wildly above last; for shorts, not wildly below.
        # You can move this guard into your strategy if you want more control.

    order = _alpaca_call(api.submit_order, **submit_kwargs)
    return order.id


# ---- market data -----------------------------------------------------
@lru_cache(maxsize=4096)
def get_daily_bars(symbol: str, lookback_days: int = 120) -> pd.DataFrame:
    """Fetch daily OHLCV from Alpaca (cached)."""
    start_date = (datetime.utcnow() - timedelta(days=lookback_days * 2)).date().isoformat()
    raw_limit  = lookback_days + 50
    try:
        bars_obj = _alpaca_call(
            api.get_bars,
            symbol,
            TimeFrame.Day,
            start=start_date,
            limit=raw_limit,
            feed="iex",
            adjustment="raw"
        )
        bars = bars_obj.df
    except Exception:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    if bars.empty:
        return bars
    bars = bars.tz_convert(None)
    bars = bars[['open','high','low','close','volume']]
    bars.columns = ['Open','High','Low','Close','Volume']
    return bars.tail(lookback_days)


# ---- asset filtering -------------------------------------------------
@lru_cache(maxsize=1)
def _active_assets_map() -> dict[str, object]:
    """
    Cache SYMBOL -> Asset for all *active* assets.
    We DO NOT filter by asset_class here to avoid over-filtering.
    """
    assets = {}
    for a in _alpaca_call(api.list_assets, status="active"):
        try:
            sym = a.symbol.upper()
            assets[sym] = a
        except Exception:
            continue
    return assets

def filter_active_tradable(symbols: list[str]) -> list[str]:
    """
    Return only symbols that are 'active' and 'tradable' per Alpaca.
    """
    assets = _active_assets_map()
    out: list[str] = []
    for s in symbols:
        sym = s.upper().strip()
        a = assets.get(sym)
        if a and getattr(a, "tradable", False):
            out.append(sym)
    return out
