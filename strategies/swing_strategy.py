# strategies/swing_strategy.py
import numpy as np
import pandas as pd
import talib

from models.signal import TradeSignal
from risk.manager import compute_levels, size_position


# ---------- 0) lightweight logging helper ----------
def _explain_enabled(cfg: dict | None) -> bool:
    logcfg = (cfg or {}).get("logging", {}) or {}
    v = logcfg.get("explain_rejects", True)  # default ON
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)

def _logline(msg: str) -> None:
    try:
        from tqdm import tqdm
        tqdm.write(msg)
    except Exception:
        print(msg)

def _explain_log(symbol: str, reason: str, details: dict | None, cfg: dict | None) -> None:
    if not _explain_enabled(cfg):
        return
    parts = []
    if details:
        for k, v in list(details.items())[:6]:
            parts.append(f"{k}={v}")
    suffix = f" — {', '.join(parts)}" if parts else ""
    _logline(f"[{symbol}] reject: {reason}{suffix}")


# ---------- 1) compute all indicators we need ----------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty DataFrame")

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    close = np.asarray(df["Close"],  dtype="float64").ravel()
    high  = np.asarray(df["High"],   dtype="float64").ravel()
    low   = np.asarray(df["Low"],    dtype="float64").ravel()
    vol   = np.asarray(df["Volume"], dtype="float64").ravel()

    # Trend
    ema20 = talib.EMA(close, timeperiod=20)
    ema50 = talib.EMA(close, timeperiod=50)
    sma50 = talib.SMA(close, timeperiod=50)
    # simple slope proxy: delta vs prior
    sma50_delta = pd.Series(sma50, index=df.index).diff()

    # Momentum
    macd, macds, macdh = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    rsi14 = talib.RSI(close, timeperiod=14)

    # Cycle (Stoch)
    # Legacy default: Slow Stoch 14-3-3
    slowk_14, slowd_14 = talib.STOCH(
        high, low, close,
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )
    # Burns timing alt: 5-3-2 (Full Stoch emulation): %K over 5, smoothed by 2; %D = 3-SMA of smoothed K
    ll5 = pd.Series(low, index=df.index).rolling(5, min_periods=5).min()
    hh5 = pd.Series(high, index=df.index).rolling(5, min_periods=5).max()
    denom5 = (hh5 - ll5).replace(0, np.nan)
    k_raw5 = ((pd.Series(close, index=df.index) - ll5) / denom5) * 100.0
    # Clip to [0, 100] to stabilize near-flat ranges and avoid propagation glitches
    k_raw5 = k_raw5.clip(lower=0.0, upper=100.0)
    k_sm2 = k_raw5.rolling(2, min_periods=2).mean()
    d_sm3 = k_sm2.rolling(3, min_periods=3).mean()

    # Volatility / stops
    atr14 = talib.ATR(high, low, close, timeperiod=14)

    # Trend-strength (for regime filter)
    adx14 = talib.ADX(high, low, close, timeperiod=14)

    # Volume confirmations
    obv = talib.OBV(close, vol)
    avgvol50 = pd.Series(vol, index=df.index).rolling(50).mean().to_numpy()

    # RVOL vs 20-day rolling median
    vol_series = pd.Series(vol, index=df.index)
    rvol20 = vol_series / vol_series.rolling(20).median()

    # Chaikin A/D line
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
    mfm = mfm.fillna(0.0)
    mfv = mfm * df["Volume"]
    ad_line = mfv.cumsum()

    out = df.copy()
    out["EMA20"]    = pd.Series(ema20, index=df.index)
    out["EMA50"]    = pd.Series(ema50, index=df.index)
    out["SMA50"]    = pd.Series(sma50, index=df.index)
    out["SMA50_SLOPE"] = sma50_delta
    out["MACD"]     = pd.Series(macd,  index=df.index)
    out["MACDs"]    = pd.Series(macds, index=df.index)
    out["MACDh"]    = pd.Series(macdh, index=df.index)
    out["RSI14"]    = pd.Series(rsi14, index=df.index)
    out["SlowK"]    = pd.Series(slowk_14, index=df.index)
    out["SlowD"]    = pd.Series(slowd_14, index=df.index)
    out["SlowK_5_3_2"] = k_sm2
    out["SlowD_5_3_2"] = d_sm3
    out["ATR14"]    = pd.Series(atr14, index=df.index)
    out["ADX14"]    = pd.Series(adx14, index=df.index)
    out["OBV"]      = pd.Series(obv,   index=df.index)
    out["AvgVol50"] = pd.Series(avgvol50, index=df.index)
    out["RVOL20"]   = rvol20
    out["ADLine"]   = ad_line
    # ATR percent of price for adaptive logic (avoid division by zero)
    out["ATR_PCT"]  = out["ATR14"] / out["Close"].where(out["Close"] != 0, np.nan)
    return out


# ---------- small helpers: config defaults ----------
def _cfg_path(cfg: dict | None, *keys, default=None):
    d = cfg or {}
    for k in keys:
        d = d.get(k, {})
    return d if d != {} else (default if default is not None else {})

def _get_float(d: dict, key: str, default: float) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return default

def _get_int(d: dict, key: str, default: int) -> int:
    try:
        return int(d.get(key, default))
    except Exception:
        return default

def _get_bool(d: dict, key: str, default: bool) -> bool:
    v = d.get(key, default)
    if isinstance(v, str):
        return v.strip().lower() in {"1","true","yes","y"}
    return bool(v)


# ---------- S/R pivots ----------
def _find_pivots(high: pd.Series, low: pd.Series, left: int, right: int, tol: float = 1e-6) -> tuple[list[int], list[int]]:
    """Return indices of local pivot highs/lows using a tolerance and plateau compression.
    Only first bar of a plateau high (or low) is kept to reduce multi-bar duplication.
    """
    highs_idx, lows_idx = [], []
    H = high.to_numpy()
    L = low.to_numpy()
    n = len(H)
    last_high_plateau = last_low_plateau = -2
    for i in range(n):
        l = max(0, i - left)
        r = min(n, i + right + 1)
        if r - l < 2:  # need at least 2 bars context
            continue
        winH = H[l:r]; winL = L[l:r]
        mx = np.nanmax(winH); mn = np.nanmin(winL)
        if np.isfinite(H[i]) and np.isfinite(mx) and (H[i] >= mx - tol):
            if i - 1 != last_high_plateau:  # compress plateau
                highs_idx.append(i)
            last_high_plateau = i
        if np.isfinite(L[i]) and np.isfinite(mn) and (L[i] <= mn + tol):
            if i - 1 != last_low_plateau:
                lows_idx.append(i)
            last_low_plateau = i
    return highs_idx, lows_idx


# ---------- Trend-strength helpers (LLR slope, basic structure) ----------
def _llr_slope_bps_per_day(series: pd.Series, lookback: int) -> float:
    """Return linear regression slope in basis points/day relative to series mean.
    Uses simple y ~ a + b*t with t = [0..L-1], slope b converted to pct/day via b/mean(y), then to bps.
    """
    s = pd.to_numeric(series.tail(lookback), errors="coerce").astype(float).dropna()
    if len(s) < max(5, min(lookback, 8)):
        return float("nan")
    y = s.to_numpy()
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    denom = float((x**2).sum())
    if denom <= 0:
        return float("nan")
    y_mean = float(y.mean())
    if not np.isfinite(y_mean) or y_mean == 0:
        return float("nan")
    b = float(((x * (y - y_mean)).sum()) / denom)  # units per day
    pct_per_day = b / y_mean
    return float(pct_per_day * 10000.0)  # bps/day

def _structure_hh_hl_ok(df: pd.DataFrame, direction: str, lookback: int, left: int = 3, right: int = 3) -> bool:
    """Check recent pivot structure: long requires HH and HL; short requires LH and LL.
    Fail-open if insufficient pivots in window.
    """
    if direction not in {"long", "short"}:
        return False
    tail = df.tail(lookback)
    hi_idx, lo_idx = _find_pivots(tail["High"], tail["Low"], left=left, right=right)
    if len(hi_idx) < 2 or len(lo_idx) < 2:
        return True  # fail-open on sparse pivots
    # absolute indices
    base = len(df) - len(tail)
    hi_abs = [base + i for i in hi_idx]
    lo_abs = [base + i for i in lo_idx]
    # take last two of each by position
    hi_last2 = sorted(hi_abs)[-2:]
    lo_last2 = sorted(lo_abs)[-2:]
    h1, h2 = float(df["High"].iloc[hi_last2[0]]), float(df["High"].iloc[hi_last2[1]])
    l1, l2 = float(df["Low"].iloc[lo_last2[0]]),  float(df["Low"].iloc[lo_last2[1]])
    if direction == "long":
        return (h2 > h1) and (l2 > l1)
    else:
        return (h2 < h1) and (l2 < l1)


# ---------- Medium Trend points system (3 of 5) with light confirmation ----------
def _trend_points_medium(df: pd.DataFrame, cfg: dict) -> tuple[bool, dict]:
    """Compute medium-loose Trend using a points system.
    Returns (trend_ok_raw, details) for the LAST bar only.
    details keys: score, align_ok, slope_ok, sep_ok, price_ok, adx_ok
    """
    scfg = (((cfg or {}).get("signals", {}) or {}).get("trend", {}) or {})
    min_points = _get_int(scfg, "min_points", 3)
    require_align = _get_bool(scfg, "require_ema20_gt_ema50", True)
    slope_lb = _get_int(scfg, "slope_lookback", 3)
    min_slope_bp = float(scfg.get("min_slope_ema50_bp", 0.0))
    allow_ema20_sub = _get_bool(scfg, "allow_ema20_slope_substitute", True)
    sep_min_atr_mult = float(scfg.get("sep_min_atr_mult", 0.25))
    req_close_above_ema50 = _get_bool(scfg, "require_close_above_ema50", True)
    adx_soft_min = float(scfg.get("adx_soft_min", 16.0))
    adx_need_mult = float(scfg.get("adx_required_if_sep_below_mult", 0.50))

    row = df.iloc[-1]
    close = float(row.get("Close", np.nan))
    ema20 = float(row.get("EMA20", np.nan))
    ema50 = float(row.get("EMA50", np.nan))
    atr = float(row.get("ATR14", np.nan))
    adx = float(row.get("ADX14", np.nan)) if "ADX14" in row.index else np.nan

    # 1) Alignment
    align_ok = True
    if require_align:
        align_ok = np.isfinite(ema20) and np.isfinite(ema50) and (ema20 > ema50)

    # 2) Slope over short lookback (EMA50 preferred; EMA20 as substitute)
    ema50_prev = float(df["EMA50"].iloc[-slope_lb]) if (len(df) > slope_lb and np.isfinite(df["EMA50"].iloc[-slope_lb])) else ema50
    ema20_prev = float(df["EMA20"].iloc[-slope_lb]) if (len(df) > slope_lb and np.isfinite(df["EMA20"].iloc[-slope_lb])) else ema20
    ema50_slope = (ema50 - ema50_prev)
    ema20_slope = (ema20 - ema20_prev)
    slope_ok = (np.isfinite(ema50_slope) and (ema50_slope > (min_slope_bp / 10000.0) * max(close, 1e-9)))
    if not slope_ok and allow_ema20_sub:
        slope_ok = np.isfinite(ema20_slope) and (ema20_slope > 0)

    # 3) Separation vs volatility normalized by ATR%
    atr_pct = (atr / close) if (np.isfinite(atr) and np.isfinite(close) and close > 0) else np.nan
    sep_pct = ((ema20 - ema50) / close) if (np.isfinite(ema20) and np.isfinite(ema50) and np.isfinite(close) and close > 0) else np.nan
    sep_ok = (np.isfinite(sep_pct) and np.isfinite(atr_pct) and (sep_pct >= sep_min_atr_mult * atr_pct))

    # 4) Price location
    price_ok = True if not req_close_above_ema50 else (np.isfinite(close) and np.isfinite(ema50) and (close >= ema50))

    # 5) ADX assist only when separation weak
    adx_ok = True
    if np.isfinite(adx) and np.isfinite(sep_pct) and np.isfinite(atr_pct) and atr_pct > 0:
        need_adx = sep_pct < (adx_need_mult * atr_pct)
        if need_adx:
            adx_ok = (adx >= adx_soft_min)

    score = int(align_ok) + int(slope_ok) + int(sep_ok) + int(price_ok) + int(adx_ok)
    return (score >= min_points), {
        "score": score,
        "align_ok": bool(align_ok),
        "slope_ok": bool(slope_ok),
        "sep_ok": bool(sep_ok),
        "price_ok": bool(price_ok),
        "adx_ok": bool(adx_ok),
        "atr_pct": float(atr_pct) if np.isfinite(atr_pct) else np.nan,
        "sep_pct": float(sep_pct) if np.isfinite(sep_pct) else np.nan,
    }

def _nearest_pivot_levels(df: pd.DataFrame, lookback: int, left: int, right: int, price: float | None = None):
    tail = df.tail(lookback)
    highs_idx, lows_idx = _find_pivots(tail["High"], tail["Low"], left, right)
    base = len(df) - len(tail)

    highs_abs = [base + i for i in highs_idx]
    lows_abs  = [base + i for i in lows_idx]

    if price is None:
        price = float(df["Close"].iloc[-1])

    low_levels  = [float(df["Low"].iloc[i])  for i in lows_abs]
    high_levels = [float(df["High"].iloc[i]) for i in highs_abs]

    # support: nearest pivot low BELOW or EQUAL to price
    sup_below = [lvl for lvl in low_levels  if lvl <= price]
    support = max(sup_below) if sup_below else None

    # resistance: nearest pivot high ABOVE or EQUAL to price
    res_above = [lvl for lvl in high_levels if lvl >= price]
    resistance = min(res_above) if res_above else None

    return support, resistance


def _pivot_cluster_ok(
    df: pd.DataFrame,
    direction: str,
    lookback: int,
    left: int,
    right: int,
    price: float,
    min_pivots: int,
    max_span_pct: float,
    max_price_dist_pct: float,
) -> bool:
    """Check for a cluster of pivots near price.
    For longs, use pivot highs (resistance cluster); for shorts, pivot lows (support cluster).
    Conditions:
      - at least `min_pivots` pivots in lookback
      - vertical span of those pivot levels within `max_span_pct` of price
      - current price is within `max_price_dist_pct` of the nearest pivot level
    """
    tail = df.tail(lookback)
    hi_idx, lo_idx = _find_pivots(tail["High"], tail["Low"], left, right)
    base = len(df) - len(tail)
    if direction == "long":
        idxs = [base + i for i in hi_idx]
        levels = [float(df["High"].iloc[i]) for i in idxs]
    elif direction == "short":
        idxs = [base + i for i in lo_idx]
        levels = [float(df["Low"].iloc[i]) for i in idxs]
    else:
        return False
    levels = [lvl for lvl in levels if np.isfinite(lvl)]
    if len(levels) < max(1, min_pivots):
        return False
    span = (max(levels) - min(levels)) if levels else np.inf
    if not (np.isfinite(span) and price > 0):
        return False
    if (span / price) > max_span_pct:
        return False
    nearest = min(levels, key=lambda x: abs(x - price))
    return (abs(price - nearest) / price) <= max_price_dist_pct


def _last_swing_levels(df: pd.DataFrame, lookback: int, left: int, right: int) -> tuple[float | None, float | None]:
    """Return (swing_low, swing_high) for the most recent completed swing using pivots within lookback."""
    tail = df.tail(lookback)
    highs_idx, lows_idx = _find_pivots(tail["High"], tail["Low"], left, right)
    if not highs_idx and not lows_idx:
        return None, None
    base = len(df) - len(tail)
    highs_abs = [base + i for i in highs_idx]
    lows_abs  = [base + i for i in lows_idx]
    last_hi = max(highs_abs) if highs_abs else -1
    last_lo = max(lows_abs) if lows_abs else -1
    if last_hi < 0 and last_lo < 0:
        return None, None
    if last_hi > last_lo and last_lo >= 0:
        # up-leg low -> high
        lo = float(df["Low"].iloc[last_lo])
        hi = float(df["High"].iloc[last_hi])
        return (lo, hi)
    if last_lo > last_hi and last_hi >= 0:
        # down-leg high -> low
        hi = float(df["High"].iloc[last_hi])
        lo = float(df["Low"].iloc[last_lo])
        return (lo, hi)
    return None, None


def _fib_confluence_ok(df: pd.DataFrame, direction: str, lookback: int, left: int, right: int, price: float, tol_abs: float, tol_pct: float) -> bool:
    """Check if current price is near a key Fibonacci retrace of the last swing leg."""
    lo, hi = _last_swing_levels(df, lookback, left, right)
    if lo is None or hi is None or lo <= 0 or hi <= 0 or lo == hi:
        return False
    low = min(lo, hi); high = max(lo, hi)
    # classic retracement set
    ratios = [0.382, 0.5, 0.618]
    levels = [high - r * (high - low) for r in ratios] if direction == "long" else [low + r * (high - low) for r in ratios]
    tol = min(tol_abs, tol_pct)
    return any(abs(price - lvl) <= tol for lvl in levels if np.isfinite(lvl))


# ---------- momentum quality ----------
def _macd_expansion_ok(df: pd.DataFrame, direction: str, mcfg: dict) -> bool:
    if direction not in {"long", "short"}:
        return False
    lookback = _get_int(mcfg, "expansion_lookback", 3)
    min_steps = _get_int(mcfg, "expansion_min_steps", 1)
    if lookback < 1:
        return True
    if len(df) < lookback + 1:
        # stricter behavior now optional
        permissive = _get_bool(mcfg, "expansion_permissive_on_short_history", False)
        return True if permissive else False

    macdh = df["MACDh"].to_numpy()
    if not np.isfinite(macdh[-1]):
        return False

    diffs = np.diff(macdh[-(lookback+1):])  # len = lookback
    good = np.sum(diffs > 0) if direction == "long" else np.sum(diffs < 0)
    return good >= min_steps

def _macd_zero_line_ok(df: pd.DataFrame, direction: str, mcfg: dict) -> bool:
    mode = str(mcfg.get("zero_line_mode", "confirm")).lower()
    if mode == "none":
        return True
    macd = df["MACD"].to_numpy()
    if not np.isfinite(macd[-1]):
        return False
    if mode == "confirm":
        return (macd[-1] > 0) if direction == "long" else (macd[-1] < 0)
    if mode == "cross":
        k = _get_int(mcfg, "zero_cross_lookback", 3)
        if k < 1 or len(macd) < k + 1:
            return True
        window = macd[-(k+1):]
        if direction == "long":
            return (window[-1] > 0) and np.any(window[:-1] <= 0)
        else:
            return (window[-1] < 0) and np.any(window[:-1] >= 0)
    return True


# ---------- volume context (ANY-OF) ----------
# (Removed) _volume_energy_ok

# ---------- 2) five energies on the last bar ----------
"""Note: removed extended evaluator and fractal/volume helpers to standardize on pure Burns."""



# -------- Pure Dr. Barry Burns 5 Energies evaluation (simplified canonical form) --------
def evaluate_five_energies(df: pd.DataFrame, cfg: dict | None = None, weekly_ctx: dict | None = None) -> dict:
    """Pure Burns style energies:
    1. Trend: EMA20 above EMA50 & price within value zone (pullback) or valid breakout (close above recent pivot cluster with momentum).
    2. Momentum: MACD histogram expanding in direction OR MACD > signal (bull) / < signal (bear) with positive slope.
    3. Cycle: Stochastic SlowK turns up (crosses above SlowD) out of oversold (< oversold_thr) within lookback OR turns down out of overbought (> overbought_thr).
    4. Support/Resistance (Value Zone / Pivot Confluence): Pullback touches EMA20-EMA50 band with rejection tail OR breakout above (below) resistance (support) pivot cluster.
    5. Scale (MTF momentum): weekly MACD alignment in the direction.
    Volume confirmation is advisory (not counted in 5 but can veto marginal case if configured).
    """
    if len(df) < 60:
        raise ValueError("Need ~60 bars for stable signals")
    cfg = cfg or {}
    filt = (cfg.get("trading", {}).get("filters", {}) or {})
    oversold = float(filt.get("cycle_oversold", 25))
    overbought = float(filt.get("cycle_overbought", 75))
    cycle_cross_lb = int(filt.get("cycle_cross_lookback", 3))
    hist_expand_lb = int(filt.get("momentum_hist_expand_lookback", 2))
    macd_min = float(filt.get("momentum_macd_min", 0.0))
    # removed legacy fractal-specific thresholds; Scale (weekly) is the 5th energy
    value_ext_max = float(filt.get("value_zone_max_ext_pct", 0.025))
    # Relaxed tail minima (allow more passes); original 0.55
    tail_min_long = float(filt.get("value_zone_reject_tail_min", 0.40))
    tail_min_short = float(filt.get("value_zone_reject_tail_min_short", 0.40))
    touch_atr_max = float(filt.get("value_zone_touch_atr_max", 0.60))

    row = df.iloc[-1]; prev = df.iloc[-2]
    ema20 = float(row["EMA20"]); ema50 = float(row["EMA50"])
    sma50 = float(row.get("SMA50", np.nan)); sma50_slope = float(row.get("SMA50_SLOPE", np.nan))
    close = float(row["Close"]); high = float(row["High"]); low = float(row["Low"])
    atr = float(row.get("ATR14", np.nan))
    macd = float(row.get("MACD", np.nan)); macds = float(row.get("MACDs", np.nan)); macdh = float(row.get("MACDh", np.nan))
    k = float(row.get("SlowK", np.nan)); d = float(row.get("SlowD", np.nan))
    adx = float(row.get("ADX14", np.nan)) if "ADX14" in row.index else np.nan

    # 1 Trend & Direction (medium points system with light confirmation)
    # Compute raw medium trend for long side only (Burns long bias); shorts keep legacy behavior for now.
    trend_bits = {}
    trend_up_raw, bits = _trend_points_medium(df, cfg)
    trend_bits = bits
    # Debounce: 2-of-3 confirmation (synthetic using last 3 raw computations on rolling window)
    n_conf = 3
    try:
        pair = ((cfg or {}).get("signals", {}) or {}).get("trend", {}).get("confirm_n_of_m", [2,3])
        n_conf = int(pair[1]) if isinstance(pair, (list, tuple)) and len(pair) == 2 else 3
    except Exception:
        n_conf = 3
    raw_hist = []
    for i in range(min(n_conf, len(df))):
        slice_df = df.iloc[: len(df) - i]
        ok, _ = _trend_points_medium(slice_df, cfg)
        raw_hist.append(bool(ok))
    trend_up = (sum(1 for v in raw_hist if v) >= max(2, min(3, len(raw_hist))))

    # Direction: long if trend_up; otherwise try legacy short detect (unchanged)
    direction = "long" if trend_up else "none"
    if direction == "none":
        cand_short = (ema20 < ema50)
        bearish = cand_short and (close <= ema50) and (np.isfinite(sma50_slope) and sma50_slope < 0)
        if bearish:
            direction = "short"

    # Value zone proximity (pullback) ext pct
    ext_pct = (close - ema20)/ema20 if ema20>0 else np.nan
    in_value_zone = np.isfinite(ext_pct) and abs(ext_pct) <= value_ext_max

    # 2 Momentum (MACD)
    momentum_ok = False
    mcfg = (cfg.get("momentum", {}) or {})
    mom_lb = int(mcfg.get("recent_confirm_lookback", 1))
    dist_min = float(mcfg.get("min_cross_distance", 0.0))
    if direction == "long":
        if np.isfinite(macd) and np.isfinite(macds):
            slope_ok = len(df) >= 3 and (macd - float(df.iloc[-2]["MACD"])) > 0
            hist_expand = True
            if hist_expand_lb > 0 and len(df) >= hist_expand_lb + 1:
                prior_h = df["MACDh"].iloc[-(hist_expand_lb+1):-1]
                hist_expand = bool(np.isfinite(macdh) and np.isfinite(prior_h.max()) and (macdh > float(prior_h.max())))
            # Burns emphasis: zero-line confirm on pullback
            zl_ok = macd > 0
            mom_cross = (macd > macds)
            if (not mom_cross) and mom_lb > 1 and len(df) >= mom_lb:
                prev_macd = df["MACD"].iloc[-mom_lb:-1].to_numpy()
                prev_sig  = df["MACDs"].iloc[-mom_lb:-1].to_numpy()
                mom_cross = np.any(prev_macd > prev_sig)
            dist_ok = (abs(macd - macds) >= dist_min) if dist_min > 0 else True
            momentum_ok = mom_cross and dist_ok and slope_ok and hist_expand and zl_ok and (abs(macd) >= macd_min)
    elif direction == "short":
        if np.isfinite(macd) and np.isfinite(macds):
            slope_ok = len(df) >= 3 and (macd - float(df.iloc[-2]["MACD"])) < 0
            hist_expand = True
            if hist_expand_lb > 0 and len(df) >= hist_expand_lb + 1:
                prior_h = df["MACDh"].iloc[-(hist_expand_lb+1):-1]
                hist_expand = bool(np.isfinite(macdh) and np.isfinite(prior_h.min()) and (macdh < float(prior_h.min())))
            zl_ok = macd < 0
            mom_cross = (macd < macds)
            if (not mom_cross) and mom_lb > 1 and len(df) >= mom_lb:
                prev_macd = df["MACD"].iloc[-mom_lb:-1].to_numpy()
                prev_sig  = df["MACDs"].iloc[-mom_lb:-1].to_numpy()
                mom_cross = np.any(prev_macd < prev_sig)
            dist_ok = (abs(macd - macds) >= dist_min) if dist_min > 0 else True
            momentum_ok = mom_cross and dist_ok and slope_ok and hist_expand and zl_ok and (abs(macd) >= macd_min)

    # 3 Cycle (Burns "wave" dip-and-turn): enhanced K/D cross with configurable lookback and midline relax
    cycle_ok = False
    try:
        ccfg = (cfg.get("cycle", {}) or {})
        use_burns_stoch = str(ccfg.get("stoch_mode", "legacy")).lower() in {"burns","5_3_2","burns_5_3_2"}
        k_col = "SlowK_5_3_2" if (use_burns_stoch and "SlowK_5_3_2" in df.columns) else "SlowK"
        d_col = "SlowD_5_3_2" if (use_burns_stoch and "SlowD_5_3_2" in df.columns) else "SlowD"
        kd = df[[k_col, d_col]].dropna()
        mid_long   = float(ccfg.get("midline_long", 55.0))
        mid_short  = float(ccfg.get("midline_short", 45.0))
        lookback   = int(ccfg.get("lookback", max(3, cycle_cross_lb)))
        confirm_k  = int(ccfg.get("confirm_bars", 0))  # require %K to continue in direction for N bars AFTER the cross
        require_d  = bool(ccfg.get("require_d_slope", False))
        relax_mid  = float(ccfg.get("relax_midline_in_value_zone", 0.0))
        from_extreme = bool(ccfg.get("from_extreme", False))
        extreme_lb = int(ccfg.get("extreme_lookback", max(lookback, 10)))
        os_thr = float(ccfg.get("oversold", 20.0)) if "oversold" in ccfg else float(filt.get("cycle_oversold", 25.0))
        ob_thr = float(ccfg.get("overbought", 80.0)) if "overbought" in ccfg else float(filt.get("cycle_overbought", 75.0))
        min_gap = float(ccfg.get("min_cross_gap", 0.0))
        need_vz = bool(ccfg.get("require_value_zone_for_trigger", False))
        if len(kd) >= 2 and direction in {"long","short"}:
            start = max(1, len(kd) - lookback + 1)
            for i in range(start, len(kd)):
                k_prev, d_prev = float(kd[k_col].iloc[i-1]), float(kd[d_col].iloc[i-1])
                k_curr, d_curr = float(kd[k_col].iloc[i]),   float(kd[d_col].iloc[i])
                # Stateful cross detection
                bull_cross = (k_prev <= d_prev) and (k_curr > d_curr)
                bear_cross = (k_prev >= d_prev) and (k_curr < d_curr)
                if direction == "long" and bull_cross:
                    # Origin from extreme within window prior to the cross
                    from_ext_ok = True
                    if from_extreme:
                        j0 = max(0, i - extreme_lb)
                        k_min = float(kd[k_col].iloc[j0:i].min()) if i > j0 else np.inf
                        from_ext_ok = (np.isfinite(k_min) and k_min <= os_thr)
                    # Post-cross confirmation: K rising for confirm_k bars after i
                    conf_ok = True
                    if confirm_k > 0:
                        if (i + confirm_k) < len(kd):
                            segK = kd[k_col].iloc[i:(i + confirm_k + 1)].to_numpy()
                            conf_ok = (len(segK) >= 2) and np.all(np.diff(segK) > 0)
                        else:
                            conf_ok = False
                    # Midline with value-zone relaxation applied to post-cross K
                    mid_ok = (k_curr > (50.0 - (relax_mid if in_value_zone else 0.0)))
                    d_slope_ok = True if not require_d else (d_curr >= d_prev)
                    gap_ok = True if min_gap <= 0 else (abs(k_curr - d_curr) >= min_gap)
                    vz_ok = True if not need_vz else bool(in_value_zone)
                    if from_ext_ok and conf_ok and mid_ok and d_slope_ok and gap_ok and vz_ok:
                        cycle_ok = True; break
                elif direction == "short" and bear_cross:
                    from_ext_ok = True
                    if from_extreme:
                        j0 = max(0, i - extreme_lb)
                        k_max = float(kd[k_col].iloc[j0:i].max()) if i > j0 else -np.inf
                        from_ext_ok = (np.isfinite(k_max) and k_max >= ob_thr)
                    conf_ok = True
                    if confirm_k > 0:
                        if (i + confirm_k) < len(kd):
                            segK = kd[k_col].iloc[i:(i + confirm_k + 1)].to_numpy()
                            conf_ok = (len(segK) >= 2) and np.all(np.diff(segK) < 0)
                        else:
                            conf_ok = False
                    mid_ok = (k_curr < (50.0 + (relax_mid if in_value_zone else 0.0)))
                    d_slope_ok = True if not require_d else (d_curr <= d_prev)
                    gap_ok = True if min_gap <= 0 else (abs(k_curr - d_curr) >= min_gap)
                    vz_ok = True if not need_vz else bool(in_value_zone)
                    if from_ext_ok and conf_ok and mid_ok and d_slope_ok and gap_ok and vz_ok:
                        cycle_ok = True; break
    except Exception:
        cycle_ok = False

    # 4 Support/Resistance & Value Zone (Burns: pullback = value zone + tail/depth, breakout = close above resistance + momentum)
    sr_ok = False
    reject_ok = False
    setup_type = ""
    # Pullback: in value zone, require tail or depth
    if direction == "long" and in_value_zone and np.isfinite(atr) and atr>0:
        body_low = min(row.Open, row.Close)
        tail_len = body_low - low
        full_range = high - low
        tail_ratio = (tail_len/full_range) if full_range>0 else 0.0
        depth_atr = (ema20 - low)/atr if atr>0 else np.nan
        reject_ok = (tail_ratio >= tail_min_long and np.isfinite(depth_atr) and depth_atr <= touch_atr_max)
        sr_ok = bool(reject_ok or (abs(ext_pct) <= value_ext_max and (np.isfinite(depth_atr) and depth_atr <= touch_atr_max)))
        setup_type = "pullback"
    elif direction == "short" and in_value_zone and np.isfinite(atr) and atr>0:
        body_high = max(row.Open, row.Close)
        tail_len = high - body_high
        full_range = high - low
        tail_ratio = (tail_len/full_range) if full_range>0 else 0.0
        depth_atr = (high - ema20)/atr if atr>0 else np.nan
        reject_ok = (tail_ratio >= tail_min_short and np.isfinite(depth_atr) and depth_atr <= touch_atr_max)
        sr_ok = bool(reject_ok or (abs(ext_pct) <= value_ext_max and (np.isfinite(depth_atr) and depth_atr <= touch_atr_max)))
        setup_type = "pullback"
    # Breakout: require momentum AND proximity to pivot level (with tolerance_mode) and optional cluster gating
    elif direction == "long" and not in_value_zone and momentum_ok:
        scfg2 = _cfg_path(cfg, "signals", "sr_pivots", default={})
        lookback2 = _get_int(scfg2, "lookback", 60)
        left2     = _get_int(scfg2, "left", 3)
        right2    = _get_int(scfg2, "right", 3)
        tol_abs2  = float(scfg2.get("near_atr_mult", 0.75)) * (atr if np.isfinite(atr) else 0.0)
        tol_pct2  = float(scfg2.get("near_pct", 0.02)) * close
        tol_mode  = str(scfg2.get("tolerance_mode", "min")).lower()
        tol2 = min(tol_abs2, tol_pct2) if tol_mode == "min" else max(tol_abs2, tol_pct2)
        sup2, res2 = _nearest_pivot_levels(df, lookback2, left2, right2, price=close)
        near_res = (res2 is not None) and (abs(close - res2) <= tol2)
        # optional cluster gating
        cl = (scfg2.get("cluster", {}) or {})
        if bool(cl.get("enabled", False)) and near_res:
            c_lb   = int(cl.get("lookback", 90))
            c_min  = int(cl.get("min_pivots", 2))
            c_span = float(cl.get("max_span_pct", 0.02))
            c_pr   = float(cl.get("max_price_dist_pct", 0.035))
            near_res = near_res and _pivot_cluster_ok(df, "long", c_lb, left2, right2, close, c_min, c_span, c_pr)
        sr_ok = bool(near_res)
        setup_type = "breakout"
    elif direction == "short" and not in_value_zone and momentum_ok:
        scfg2 = _cfg_path(cfg, "signals", "sr_pivots", default={})
        lookback2 = _get_int(scfg2, "lookback", 60)
        left2     = _get_int(scfg2, "left", 3)
        right2    = _get_int(scfg2, "right", 3)
        tol_abs2  = float(scfg2.get("near_atr_mult", 0.75)) * (atr if np.isfinite(atr) else 0.0)
        tol_pct2  = float(scfg2.get("near_pct", 0.02)) * close
        tol_mode  = str(scfg2.get("tolerance_mode", "min")).lower()
        tol2 = min(tol_abs2, tol_pct2) if tol_mode == "min" else max(tol_abs2, tol_pct2)
        sup2, res2 = _nearest_pivot_levels(df, lookback2, left2, right2, price=close)
        near_sup = (sup2 is not None) and (abs(close - sup2) <= tol2)
        cl = (scfg2.get("cluster", {}) or {})
        if bool(cl.get("enabled", False)) and near_sup:
            c_lb   = int(cl.get("lookback", 90))
            c_min  = int(cl.get("min_pivots", 2))
            c_span = float(cl.get("max_span_pct", 0.02))
            c_pr   = float(cl.get("max_price_dist_pct", 0.035))
            near_sup = near_sup and _pivot_cluster_ok(df, "short", c_lb, left2, right2, close, c_min, c_span, c_pr)
        sr_ok = bool(near_sup)
        setup_type = "breakout"

    # Optional: Fibonacci confluence near current price
    scfg = _cfg_path(cfg, "signals", "sr_pivots", default={})
    fib_cfg = (scfg.get("fib", {}) or {})
    fib_enabled = bool(fib_cfg.get("enabled", False))
    fib_used = False
    if fib_enabled and direction in {"long","short"} and np.isfinite(atr) and atr>0:
        lookback = int(scfg.get("lookback", 60))
        left     = int(scfg.get("left", 3))
        right    = int(scfg.get("right", 3))
        tol_abs  = float(scfg.get("near_atr_mult", 0.75)) * atr
        tol_pct  = float(scfg.get("near_pct", 0.02)) * close
        if _fib_confluence_ok(df, direction, lookback, left, right, close, tol_abs, tol_pct):
            fib_used = True
            # Fib only reinforces SR; do not flip SR from False → True alone

    # 5 SCALE / MTF MOMENTUM (weekly MACD alignment per Burns "use momentum on higher timeframe")
    scale_ok = True
    mtf_cfg = (cfg.get("trading", {}).get("mtf") or {})
    if bool(mtf_cfg.get("enabled", True)):
        try:
            if weekly_ctx is not None:
                wmacd = float(weekly_ctx.get("wmacd", np.nan)); wmacds = float(weekly_ctx.get("wmacds", np.nan)); wmacdh = float(weekly_ctx.get("wmacdh", np.nan))
            else:
                wk = _resample_weekly_ohlcv(df)
                wk_ind = add_indicators(wk)
                wrow = wk_ind.iloc[-1]
                # Use last completed weekly bar (avoid partial-week bias)
                if isinstance(df.index, pd.DatetimeIndex):
                    is_fri = (df.index[-1].weekday() == 4)
                    if (not is_fri) and (len(wk_ind) >= 2):
                        wrow = wk_ind.iloc[-2]
                wmacd = float(wrow.get("MACD", np.nan)); wmacds = float(wrow.get("MACDs", np.nan)); wmacdh = float(wrow.get("MACDh", np.nan))
            if direction == "long":
                scale_ok = np.isfinite(wmacd) and np.isfinite(wmacds) and (wmacd > 0) and (wmacd > wmacds) and (wmacdh > 0)
            elif direction == "short":
                scale_ok = np.isfinite(wmacd) and np.isfinite(wmacds) and (wmacd < 0) and (wmacd < wmacds) and (wmacdh < 0)
            else:
                scale_ok = False
        except Exception:
            scale_ok = False  # fail-closed if weekly unavailable

    # Volume confirmation (optional, advisory) – require N of {RVOL, AD slope, OBV slope}
    vol_cfg = (cfg.get("volume", {}) or {})
    need = int(vol_cfg.get("min_components", 1))
    win  = int(vol_cfg.get("window", 20))
    rthr = float(vol_cfg.get("rvol_threshold", 1.0))
    volume_ok = True
    if direction in {"long","short"}:
        comps = 0
        # RVOL
        if "RVOL20" in df.columns:
            rv = float(row.get("RVOL20", np.nan))
            if np.isfinite(rv):
                comps += int(rv >= rthr)
        # Chaikin A/D slope
        if "ADLine" in df.columns and len(df) >= win + 1:
            ad_slope = float(df["ADLine"].iloc[-win:].diff().mean())
            comps += int((ad_slope > 0) if direction=="long" else (ad_slope < 0))
        # OBV slope
        if "OBV" in df.columns and len(df) >= win + 1:
            obv_slope = float(df["OBV"].iloc[-win:].diff().mean())
            comps += int((obv_slope > 0) if direction=="long" else (obv_slope < 0))
        volume_ok = (comps >= need)

    trend_ok = (direction == "long") or (direction == "short")
    # Early-in-trend gating (waves): robust anchor + strict dip count; fail-closed on errors
    waves_ok = False
    try:
        if direction in {"long", "short"}:
            s = df["SMA50_SLOPE"].dropna()
            ccfg = (cfg.get("cycle", {}) or {})
            waves_max = int(ccfg.get("waves_max", 2))
            anchor_lb = int(ccfg.get("waves_anchor_lookback", 252))
            fail_if_no_turn = bool(ccfg.get("waves_fail_if_no_turn", True))
            max_bars_since_turn = int(ccfg.get("waves_max_bars_since_turn", 0))  # 0 = off
            # Use stoch mode consistent with Cycle
            use_burns = str(ccfg.get("stoch_mode", "burns_5_3_2")).lower() in {"burns","5_3_2","burns_5_3_2"}
            k_col = "SlowK_5_3_2" if (use_burns and "SlowK_5_3_2" in df.columns) else "SlowK"
            d_col = "SlowD_5_3_2" if (use_burns and "SlowD_5_3_2" in df.columns) else "SlowD"
            os_thr = float(ccfg.get("oversold", (cfg.get("trading", {}).get("filters", {}) or {}).get("cycle_oversold", 25.0)))
            ob_thr = float(ccfg.get("overbought", (cfg.get("trading", {}).get("filters", {}) or {}).get("cycle_overbought", 75.0)))

            if len(s) >= 3:
                s_window = s.tail(anchor_lb) if anchor_lb > 0 else s
                fav = (s_window > 0) if direction == "long" else (s_window < 0)
                flips = np.where((fav.values[1:] & ~fav.values[:-1]))[0]
                if len(flips) > 0:
                    last_flip_pos = int(flips[-1] + 1)
                    anchor_idx = s_window.index[last_flip_pos]
                    # bars since turn
                    try:
                        bars_since_turn = int(len(df) - 1 - df.index.get_loc(anchor_idx))
                    except Exception:
                        bars_since_turn = int(len(df) - 1)
                    seg = df.loc[anchor_idx:]
                    kd = seg[[k_col, d_col]].dropna()
                    dips = 0
                    for i in range(1, len(kd)):
                        k_prev, d_prev = float(kd[k_col].iloc[i-1]), float(kd[d_col].iloc[i-1])
                        k_curr, d_curr = float(kd[k_col].iloc[i]),   float(kd[d_col].iloc[i])
                        if direction == "long":
                            if (k_prev <= d_prev) and (k_prev <= os_thr) and (k_curr > d_curr):
                                dips += 1
                        else:
                            if (k_prev >= d_prev) and (k_prev >= ob_thr) and (k_curr < d_curr):
                                dips += 1
                    waves_ok = (dips <= waves_max)
                    if max_bars_since_turn > 0 and bars_since_turn > max_bars_since_turn:
                        waves_ok = False
                else:
                    waves_ok = (not fail_if_no_turn)
            else:
                waves_ok = False
        else:
            waves_ok = False
    except Exception:
        waves_ok = False

    core_pass = {
        "trend": trend_ok and waves_ok,
        "momentum": momentum_ok,
        "cycle": cycle_ok,
        "sr": sr_ok,
        "scale": scale_ok,
    }
    score = sum(core_pass.values())
    return {
        "direction": direction,
        # Report core-trend (what is actually scored), plus raw components for diagnostics
        "trend": (trend_ok and waves_ok),
        "trend_raw": trend_ok,
        "waves_ok": waves_ok,
        "momentum": momentum_ok,
        "cycle": cycle_ok,
        "sr": sr_ok,
        "scale": scale_ok,
        "volume": volume_ok,
        "score": score,
        "core_pass_count": score,
        "ext_pct": ext_pct,
        "setup_type": setup_type,
        "explain": {
            "trend": {
                "ema20": ema20, "ema50": ema50,
                "up_raw": bool(trend_up_raw),
                "score": int(trend_bits.get("score", 0)),
                "align_ok": bool(trend_bits.get("align_ok", False)),
                "slope_ok": bool(trend_bits.get("slope_ok", False)),
                "sep_ok": bool(trend_bits.get("sep_ok", False)),
                "price_ok": bool(trend_bits.get("price_ok", False)),
                "adx_ok": bool(trend_bits.get("adx_ok", True)),
                "atr_pct": trend_bits.get("atr_pct"),
                "sep_pct": trend_bits.get("sep_pct"),
                "waves_ok": bool(waves_ok),
            },
            "momentum": {"macd": macd, "macds": macds, "macdh": macdh, "expand_lb": hist_expand_lb},
            "cycle": {"oversold": oversold, "overbought": overbought, "found": cycle_ok, "stoch_mode": ("5_3_2" if ("SlowK_5_3_2" in df.columns and "SlowD_5_3_2" in df.columns) else "14_3_3")},
            "sr": {"value_zone": in_value_zone, "reject": reject_ok, "ext_pct": ext_pct, "fib_used": fib_used},
            "scale": {"ok": scale_ok, "mode": "weekly_macd"},
            "volume": {"ok": volume_ok},
        }
    }

# ---------- 2.5) Higher-timeframe (weekly) confirmation helpers ----------
def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index(pd.to_datetime(df["Date"]))
            df = df.drop(columns=["Date"])
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'Date' column for MTF resampling.")
    return df

def _resample_weekly_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_dt_index(df)
    wk = df[["Open", "High", "Low", "Close", "Volume"]].resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna(how="all")
    return wk

def _weekly_trend_view(wk_ind: pd.DataFrame, mtf_cfg: dict) -> str:
    if wk_ind.shape[0] < 10:
        return "none"
    mode = str(mtf_cfg.get("trend_filter", "ema")).lower()
    r = wk_ind.iloc[-1]
    if mode == "macd":
        if pd.isna(r["MACD"]) or pd.isna(r["MACDs"]) or pd.isna(r["MACDh"]):
            return "none"
        if (r["MACD"] > r["MACDs"]) and (r["MACDh"] > 0):
            return "up"
        if (r["MACD"] < r["MACDs"]) and (r["MACDh"] < 0):
            return "down"
        return "none"
    else:
        if pd.isna(r["EMA20"]) or pd.isna(r["EMA50"]) or pd.isna(r["Close"]):
            return "none"
        if (r["EMA20"] > r["EMA50"]) and (r["Close"] > r["EMA50"]):
            return "up"
        if (r["EMA20"] < r["EMA50"]) and (r["Close"] < r["EMA50"]):
            return "down"
        return "none"

def _mtf_alignment_ok(daily_direction: str, weekly_trend: str, mtf_cfg: dict) -> tuple[bool, str]:
    if daily_direction not in {"long", "short"}:
        return True, "daily-none"
    if weekly_trend == "none":
        return True, "weekly-none"
    if daily_direction == "long" and weekly_trend == "up":
        return True, "aligned-up"
    if daily_direction == "short" and weekly_trend == "down":
        return True, "aligned-down"
    if bool(mtf_cfg.get("reject_on_mismatch", True)):
        return False, f"mtf-mismatch({weekly_trend})"
    else:
        return True, f"mtf-mismatch-allowed({weekly_trend})"


# ---------- small perf helper for engine ----------
def weekly_context(df_slice: pd.DataFrame) -> dict:
    """Return a tiny dict with current weekly MACD values from the provided daily slice.
    Engine can cache per-symbol per-day to avoid repeated resamples.
    Keys: wmacd, wmacds, wmacdh. Empty when not available.
    """
    try:
        wk = _resample_weekly_ohlcv(df_slice)
        wk_ind = add_indicators(wk)
        if wk_ind.empty:
            return {}
        r = wk_ind.iloc[-1]
        return {
            "wmacd": float(r.get("MACD", np.nan)),
            "wmacds": float(r.get("MACDs", np.nan)),
            "wmacdh": float(r.get("MACDh", np.nan)),
        }
    except Exception:
        return {}


# ---------- market regime filters (with details for logging) ----------
def _chop_index_latest(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> float:
    if window <= 1 or len(close) < window:
        return np.nan
    h = high[-window:]; l = low[-window:]; c = close[-window:]
    tr = talib.TRANGE(h, l, c)
    tr_sum = float(np.nansum(tr))
    hi = float(np.nanmax(h))
    lo = float(np.nanmin(l))
    denom = hi - lo
    if denom <= 0 or tr_sum <= 0:
        return np.nan
    return float(100.0 * (np.log10(tr_sum / denom) / np.log10(window)))

def _regime_check(df: pd.DataFrame, cfg: dict) -> tuple[bool, dict]:
    rcfg = (cfg or {}).get("regime", {}) or {}
    if not rcfg.get("enabled", True):
        return True, {"enabled": False}

    adx_min      = float(rcfg.get("adx_min", 16))
    atr_pct_min  = float(rcfg.get("atr_pct_min", 0.010))
    use_chop     = bool(rcfg.get("use_chop", True))
    chop_window  = int(rcfg.get("chop_window", 14))
    chop_max     = float(rcfg.get("chop_max", 60))

    last = df.iloc[-1]
    det = {"adx_min": adx_min, "atr_pct_min": atr_pct_min, "use_chop": use_chop,
           "chop_window": chop_window, "chop_max": chop_max}

    adx_val = float(last.get("ADX14", np.nan))
    adx_ok = True if np.isnan(adx_val) else (adx_val >= adx_min)
    det["adx"] = f"{adx_val:.2f}" if np.isfinite(adx_val) else "nan"
    det["adx_ok"] = adx_ok

    atr_val = float(last.get("ATR14", np.nan))
    close_val = float(last.get("Close", np.nan))
    atr_ok = True
    atr_pct = np.nan
    if np.isfinite(atr_val) and np.isfinite(close_val) and close_val > 0:
        atr_pct = atr_val / close_val
        atr_ok = atr_pct >= atr_pct_min
    det["atr_pct"] = f"{atr_pct:.3f}" if np.isfinite(atr_pct) else "nan"
    det["atr_ok"] = atr_ok
    if not np.isfinite(atr_val) or not np.isfinite(close_val):
        det["atr_note"] = "atr_or_close_nan"

    chop_ok = True
    if use_chop:
        high = np.asarray(df["High"], dtype="float64").ravel()
        low  = np.asarray(df["Low"],  dtype="float64").ravel()
        close= np.asarray(df["Close"],dtype="float64").ravel()
        chop = _chop_index_latest(high, low, close, chop_window)
        det["chop"] = f"{chop:.1f}" if np.isfinite(chop) else "nan"
        chop_ok = (not np.isfinite(chop)) or (chop <= chop_max)
        det["chop_ok"] = chop_ok
    else:
        det["chop"] = "off"
        det["chop_ok"] = True

    ok = adx_ok and atr_ok and chop_ok
    det["ok"] = ok
    return ok, det


# ---------- build trade ----------
def build_trade_signal(symbol: str, df: pd.DataFrame, cfg: dict) -> TradeSignal | None:
    """
    If 4/5 energies align (plus optional MTF bonus), create a TradeSignal with ATR-based stop/target and position size.
    Weekly alignment can add +1 to score when enabled; no hard reject when reject_on_mismatch=false.
    """
    # 0) Market regime filter
    regime_ok, regime_det = _regime_check(df, cfg)
    if not regime_ok:
        _explain_log(symbol, "regime", regime_det, cfg)
        return None

    # 1) Energies (5)
    energies = evaluate_five_energies(df, cfg)
    if energies["direction"] not in ("long", "short"):
        _explain_log(symbol, "no-trend", {"ema20>ema50 & price>ema50 (long) OR opposite (short)": False}, cfg)
        return None

    # Volume tie-break when exactly at core minimum (H)
    core_min_tb = int(cfg.get("trading", {}).get("min_core_energies", cfg.get("trading", {}).get("min_score", 4)))
    vt_enabled = bool(cfg.get("trading", {}).get("volume_tiebreak_core4", True))
    if vt_enabled and int(energies.get("score", 0)) == core_min_tb and not energies.get("volume", False):
        _explain_log(symbol, "vol_tiebreak_fail", {"core": energies.get("score"), "volume": energies.get("volume")}, cfg)
        return None

    # ---- MTF as BONUS energy ----
    mtf_cfg = (cfg.get("trading", {}).get("mtf") or {})
    burns_mode = bool((cfg.get("trading", {}) or {}).get("burns_mode", False))
    # In pure Burns mode, Scale is already counted as a core energy; do not add MTF as a separate bonus
    mtf_counts = False if burns_mode else bool(mtf_cfg.get("counts_as_energy", True))
    mtf_reject = bool(mtf_cfg.get("reject_on_mismatch", False))     # hard veto or not

    mtf_ok = True
    mtf_reason = "mtf-disabled"
    if bool(mtf_cfg.get("enabled", False)):
        try:
            wk = _resample_weekly_ohlcv(df)
            wk_ind = add_indicators(wk)
            w_trend = _weekly_trend_view(wk_ind, mtf_cfg)
            mtf_ok, mtf_reason = _mtf_alignment_ok(energies["direction"], w_trend, mtf_cfg)
        except Exception as e:
            mtf_ok, mtf_reason = True, f"mtf-skip-error:{e}"

    mtf_bonus = 1 if (mtf_counts and mtf_ok) else 0
    core_min = int(cfg.get("trading", {}).get("min_core_energies", cfg.get("trading", {}).get("min_score", 4)))
    eff_score = int(energies["score"]) + mtf_bonus
    # enforce core_min BEFORE bonus (strict Burns 4 of 5 rule)
    if int(energies["score"]) < core_min:
        passes = {k: bool(energies.get(k)) for k in ("trend","momentum","cycle","sr","scale")}
        passes["core_pass_count"] = int(energies["score"])
        _explain_log(symbol, f"core<{core_min}", passes, cfg)
        return None
    # Minimum score gate (possibly same as core_min or higher including bonus)
    min_score = int(cfg.get("trading", {}).get("min_score", core_min))

    if (mtf_reject and not mtf_ok):
        _explain_log(symbol, "mtf", {"reason": mtf_reason}, cfg)
        return None

    if eff_score < min_score:
        passes = {k: bool(energies.get(k)) for k in ("trend","momentum","cycle","sr","scale")}
        passes["mtf_bonus"] = bool(mtf_bonus)
        passes["volume_confirm"] = bool(energies.get("volume"))
        _explain_log(symbol, f"eff_score<{min_score}", passes, cfg)
        return None

    # 2.5) Optional pattern trigger layer (price action) AFTER energies pass
    pcfg = (cfg.get("trading", {}) or {}).get("patterns", {}) or {}
    if bool(pcfg.get("enabled", False)):
        ok_pat, pdet = _pattern_trigger(df, energies["direction"], pcfg)
        if not ok_pat:
            _explain_log(symbol, "pattern", pdet, cfg)
            return None

    # 2.6) Pullback & extension gating (avoid extended entries, require proximity to EMA20)
    f2cfg = (cfg.get("trading", {}) or {}).get("filters", {}) or {}
    max_extension_pct = float(f2cfg.get("max_extension_pct", 0.0))  # e.g. 0.06 means <=6% above/below EMA20 for longs/shorts
    pullback_tolerance_pct = float(f2cfg.get("pullback_tolerance_pct", 0.0))  # require price within X% of EMA20
    if max_extension_pct > 0 or pullback_tolerance_pct > 0:
        if len(df) >= 1:
            row_last = df.iloc[-1]
            ema20 = float(row_last.get("EMA20", np.nan))
            close_px = float(row_last.get("Close", np.nan))
            if np.isfinite(ema20) and np.isfinite(close_px) and ema20 > 0:
                ext_pct = (close_px - ema20) / ema20
                # Extension rejection: price too far in direction already
                if max_extension_pct > 0:
                    if energies["direction"] == "long" and ext_pct > max_extension_pct:
                        _explain_log(symbol, "over_extended", {"ext_pct": f"{ext_pct:.3f}", "max": max_extension_pct}, cfg)
                        return None
                    if energies["direction"] == "short" and (-ext_pct) > max_extension_pct:  # price far below EMA20
                        _explain_log(symbol, "over_extended", {"ext_pct": f"{ext_pct:.3f}", "max": max_extension_pct}, cfg)
                        return None
                # Pullback proximity: demand we are not too far from EMA20
                if pullback_tolerance_pct > 0:
                    if abs(ext_pct) > pullback_tolerance_pct:
                        _explain_log(symbol, "not_pulled_back", {"ext_pct": f"{ext_pct:.3f}", "tol": pullback_tolerance_pct}, cfg)
                        return None

    # 2.9) Additional volatility/liquidity filters (ATR % floor)
    fcfg = (cfg.get("trading", {}) or {}).get("filters", {}) or {}
    min_atr_pct = float(fcfg.get("min_atr_pct", 0.0))

    # 3) Risk levels and sizing (at close)
    row   = df.iloc[-1]
    close = float(row["Close"])
    atr   = float(row["ATR14"])

    if min_atr_pct > 0 and np.isfinite(atr) and close > 0:
        atr_pct_val = atr / close
        if atr_pct_val < min_atr_pct:
            _explain_log(symbol, "atr_pct_floor", {"atr_pct": f"{atr_pct_val:.3f}", "min": min_atr_pct}, cfg)
            return None

    stop, target = compute_levels(
        direction   = energies["direction"],
        entry       = close,
        atr         = atr,
        atr_mult    = float(cfg["risk"]["atr_multiple_stop"]),
        reward_mult = float(cfg["risk"]["reward_multiple"]),
    )
    if stop is None or target is None:
        _explain_log(symbol, "levels-none", {"atr": atr}, cfg)
        return None

    # In Burns mode, prefer a pivot-based stop just beyond recent S/R
    if burns_mode:
        try:
            # reuse S/R nearest levels for pivot stop
            scfg = _cfg_path(cfg, "signals", "sr_pivots", default={})
            lookback = _get_int(scfg, "lookback", 60)
            left     = _get_int(scfg, "left", 3)
            right    = _get_int(scfg, "right", 3)
            support, resistance = _nearest_pivot_levels(df, lookback, left, right, price=close)
            pad_atr_mult = float(cfg.get("risk", {}).get("pad_atr_for_pivot_stop", 0.25))
            min_stop_pct = float(cfg.get("risk", {}).get("min_stop_pct", 0.0))
            pad = 0.0
            if np.isfinite(atr) and atr > 0:
                pad = max(pad, pad_atr_mult * atr)
            if close > 0 and min_stop_pct > 0:
                pad = max(pad, min_stop_pct * close)
            if energies["direction"] == "long" and support is not None:
                stop_pivot = max(0.01, support - pad)
                stop = round(stop_pivot, 2)
                per_r = close - stop
                if per_r <= 0:
                    _explain_log(symbol, "pivot_stop_invalid", {"support": support, "pad": pad}, cfg)
                    return None
                target = round(close + float(cfg["risk"]["reward_multiple"]) * per_r, 2)
            elif energies["direction"] == "short" and resistance is not None:
                stop_pivot = resistance + pad
                stop = round(stop_pivot, 2)
                per_r = stop - close
                if per_r <= 0:
                    _explain_log(symbol, "pivot_stop_invalid", {"resistance": resistance, "pad": pad}, cfg)
                    return None
                target = round(close - float(cfg["risk"]["reward_multiple"]) * per_r, 2)
        except Exception as e:
            _explain_log(symbol, "pivot_stop_error", {"err": str(e)}, cfg)

    qty = size_position(cfg=cfg, entry=close, stop=stop)
    if qty <= 0:
        _explain_log(symbol, "qty<=0", {"entry": close, "stop": stop}, cfg)
        return None

    per_share_risk = abs(close - stop)
    total_risk     = per_share_risk * qty

    return TradeSignal(
        symbol   = symbol,
        direction= energies["direction"],
        score    = int(eff_score),  # include MTF bonus if any
        entry    = close,
        stop     = stop,
        target   = target,
        quantity = qty,
        per_share_risk = per_share_risk,
        total_risk = total_risk,
        notes = (
            f"Trend:{energies['trend']} Mom:{energies['momentum']} Cycle:{energies['cycle']} "
            f"S/R:{energies['sr']} Scale:{energies.get('scale')} VolC:{energies['volume']} | "
            f"core:{energies['score']} MTF:{mtf_reason} (+{mtf_bonus})"
        ),
    )


# ---------- Pattern triggers (post-energy) ----------
def _pattern_trigger(df: pd.DataFrame, direction: str, pcfg: dict) -> tuple[bool, dict]:
    """Return (ok, details). Supports simple patterns:
    modes: engulf, inside_break, nr4_break
    All optional; pass if ANY selected pattern satisfied.
    """
    if direction not in {"long","short"}:
        return False, {"reason": "no-direction"}
    modes = pcfg.get("modes", ["engulf"]) or []
    if not modes:
        return True, {"reason": "no-modes"}
    if len(df) < 3:
        return False, {"reason": "insufficient-bars"}
    row = df.iloc[-1]; prev = df.iloc[-2]
    o, h, l, c = float(row.Open), float(row.High), float(row.Low), float(row.Close)
    po, ph, pl, pc = float(prev.Open), float(prev.High), float(prev.Low), float(prev.Close)
    passed = []
    # Engulfing (real body engulf)
    if "engulf" in modes:
        prev_body_high = max(po, pc)
        prev_body_low  = min(po, pc)
        curr_body_high = max(o, c)
        curr_body_low  = min(o, c)
        bull = (direction == "long" and c > o and pc < po and curr_body_high >= prev_body_high and curr_body_low <= prev_body_low)
        bear = (direction == "short" and c < o and pc > po and curr_body_high >= prev_body_high and curr_body_low <= prev_body_low)
        if bull or bear:
            passed.append("engulf")
    # Inside bar breakout (current range breaks prior high/low after an inside bar prev compared to its prior)
    if "inside_break" in modes and len(df) >= 3:
        prev2 = df.iloc[-3]
        inside_prev = (ph <= float(prev2.High) and pl >= float(prev2.Low))
        if inside_prev:
            if direction == "long" and h > ph:
                passed.append("inside_break")
            if direction == "short" and l < pl:
                passed.append("inside_break")
    # NR4 breakout (narrowest range of last 4, then breakout in direction)
    if "nr4_break" in modes and len(df) >= 4:
        last4 = df.iloc[-4:]
        ranges = (last4.High - last4.Low).to_numpy()
        if np.argmin(ranges) == 2:  # the bar BEFORE current is narrowest
            if direction == "long" and h > ph:
                passed.append("nr4_break")
            if direction == "short" and l < pl:
                passed.append("nr4_break")
    if passed:
        return True, {"patterns": passed}
    return False, {"reason": "no-pattern", "tested": modes}


# ---------- (optional) simple demo signals ----------
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    close = np.asarray(df["Close"], dtype="float64").ravel()
    rsi   = talib.RSI(close, 14)
    ema20 = talib.EMA(close, 20)
    df = df.copy()
    df["RSI"] = pd.Series(rsi, index=df.index)
    df["EMA20"] = pd.Series(ema20, index=df.index)
    buy_mask  = (rsi < 30) & (close > ema20)
    sell_mask = (rsi > 70) & (close < ema20)
    df["Signal"] = 0
    df.loc[buy_mask, "Signal"] = 1
    df.loc[sell_mask, "Signal"] = -1
    return df
