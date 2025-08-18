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

    # Momentum
    macd, macds, macdh = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    rsi14 = talib.RSI(close, timeperiod=14)

    # Cycle (Stoch)
    slowk, slowd = talib.STOCH(
        high, low, close,
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )

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
    out["MACD"]     = pd.Series(macd,  index=df.index)
    out["MACDs"]    = pd.Series(macds, index=df.index)
    out["MACDh"]    = pd.Series(macdh, index=df.index)
    out["RSI14"]    = pd.Series(rsi14, index=df.index)
    out["SlowK"]    = pd.Series(slowk, index=df.index)
    out["SlowD"]    = pd.Series(slowd, index=df.index)
    out["ATR14"]    = pd.Series(atr14, index=df.index)
    out["ADX14"]    = pd.Series(adx14, index=df.index)
    out["OBV"]      = pd.Series(obv,   index=df.index)
    out["AvgVol50"] = pd.Series(avgvol50, index=df.index)
    out["RVOL20"]   = rvol20
    out["ADLine"]   = ad_line
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
def _volume_energy_ok(df: pd.DataFrame, direction: str, vcfg: dict) -> tuple[bool, dict]:
    """
    Pass if AT LEAST `min_components` of {RVOL, AD slope, OBV tick/slope} agree with direction.
    Config:
      rvol_threshold (float)
      ad_trend_lookback (int)
      obv_mode: "tick" (last>prev) or "slope" (lookback)
      min_components: int (default 1)
    """
    if direction not in {"long", "short"}:
        return False, {"reason": "no-direction"}

    rvol_thr = _get_float(vcfg, "rvol_threshold", 1.20)
    ad_look  = _get_int(vcfg, "ad_trend_lookback", 3)
    obv_mode = str(vcfg.get("obv_mode", "tick")).lower()   # "tick" or "slope"
    # default stricter now (>=2 components) unless user overrides
    min_components = _get_int(vcfg, "min_components", 2)

    row = df.iloc[-1]
    rvol = float(row.get("RVOL20", np.nan))
    rvol_ok = bool(np.isfinite(rvol) and (rvol >= rvol_thr))

    # A/D slope over exactly `ad_look` bars (compare -1 vs -ad_look)
    if ad_look < 1:
        ad_ok = False
        ad_now = ad_prev = np.nan
        ad_available = False
    elif len(df) <= ad_look:
        ad_ok = False
        ad_now = ad_prev = np.nan
        ad_available = False
    else:
        ad_now  = float(df["ADLine"].iloc[-1])
        ad_prev = float(df["ADLine"].iloc[-ad_look])
        ad_ok   = (ad_now > ad_prev) if direction == "long" else (ad_now < ad_prev)
        ad_available = True

    # OBV confirmation: tick or slope mode
    if obv_mode == "slope" and ad_look >= 1 and len(df) > ad_look:
        obv_now  = float(df["OBV"].iloc[-1])
        obv_prev = float(df["OBV"].iloc[-ad_look])
        obv_ok   = (obv_now > obv_prev) if direction == "long" else (obv_now < obv_prev)
        obv_available = True
    else:
        if len(df) >= 2:
            obv_now  = float(df["OBV"].iloc[-1])
            obv_prev = float(df["OBV"].iloc[-2])
            obv_ok   = (obv_now > obv_prev) if direction == "long" else (obv_now < obv_prev)
            obv_available = True
        else:
            obv_now = obv_prev = np.nan
            obv_ok = False
            obv_available = False

    # Only count components that are available
    components = []
    if np.isfinite(rvol):
        components.append(rvol_ok)
    if ad_available:
        components.append(ad_ok)
    if obv_available:
        components.append(obv_ok)
    passed = int(sum(1 for c in components if c))
    avail = max(1, len(components))
    ok = passed >= max(1, min_components)

    details = {
        "rvol": f"{rvol:.2f}" if np.isfinite(rvol) else "nan",
        "rvol_ok": rvol_ok, "rvol_thr": rvol_thr,
        "ad_look": ad_look, "ad_ok": ad_ok,
    "obv_mode": obv_mode, "obv_ok": obv_ok,
    "passed": passed, "available_components": avail, "min_components": min_components,
    }
    return ok, details

# ---------- 2) five energies on the last bar ----------
def evaluate_five_energies(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    if len(df) < 60:
        raise ValueError("Need ~60 bars for stable signals")

    row  = df.iloc[-1]
    prev = df.iloc[-2]

    # 1) TREND
    bullish_trend = (row["EMA20"] > row["EMA50"]) and (row["Close"] > row["EMA50"])
    bearish_trend = (row["EMA20"] < row["EMA50"]) and (row["Close"] < row["EMA50"])
    direction = "long" if bullish_trend else "short" if bearish_trend else "none"
    trend_ok = bullish_trend or bearish_trend

    # 2) MOMENTUM with quality gates (expansion optional)
    mcfg = _cfg_path(cfg, "momentum", default={})
    rsi_filter     = _get_bool(mcfg, "rsi_filter", True)
    rsi_threshold  = _get_float(mcfg, "rsi_threshold", 50.0)
    use_expansion  = _get_bool(mcfg, "use_macd_expansion", False)
    macd_abs_min   = _get_float(mcfg, "macd_abs_min", 0.0)  # require |MACD - signal| >= threshold or |MACD| >= threshold if signal NA

    base_bull = (row["MACD"] > row["MACDs"])
    base_bear = (row["MACD"] < row["MACDs"])
    if rsi_filter:
        base_bull = base_bull and (row["RSI14"] >= rsi_threshold)
        base_bear = base_bear and (row["RSI14"] <= (100 - rsi_threshold))

    zl_ok   = _macd_zero_line_ok(df, direction, mcfg) if direction in {"long","short"} else False
    exp_ok  = _macd_expansion_ok(df, direction, mcfg) if (use_expansion and direction in {"long","short"}) else True

    if direction == "long":
        momentum_ok = bool(base_bull and zl_ok and exp_ok)
    elif direction == "short":
        momentum_ok = bool(base_bear and zl_ok and exp_ok)
    else:
        momentum_ok = False

    if momentum_ok and macd_abs_min > 0:
        macd_val = float(row["MACD"])
        macds_val = float(row["MACDs"]) if np.isfinite(row["MACDs"]) else macd_val
        dist = abs(macd_val - macds_val)
        if dist < macd_abs_min and abs(macd_val) < macd_abs_min:
            momentum_ok = False

    # 3) CYCLE with optional hysteresis / debounce
    cycfg = _cfg_path(cfg, "cycle", default={})
    rise_bars = _get_int(cycfg, "rise_bars", 0)  # number of prior bars %K must be strictly rising (0 = disabled)
    require_d_slope = _get_bool(cycfg, "require_d_slope", False)
    # base triggers
    bull_cycle_raw = (prev["SlowK"] < 20) and (row["SlowK"] > row["SlowD"])
    bear_cycle_raw = (prev["SlowK"] > 80) and (row["SlowK"] < row["SlowD"])
    def _k_rising(n: int) -> bool:
        if n <= 1:
            return True
        if len(df) < n:
            return False
        k_vals = df["SlowK"].iloc[-n:].to_numpy()
        return np.all(np.diff(k_vals) > 0)
    def _k_falling(n: int) -> bool:
        if n <= 1:
            return True
        if len(df) < n:
            return False
        k_vals = df["SlowK"].iloc[-n:].to_numpy()
        return np.all(np.diff(k_vals) < 0)
    bull_cycle = bull_cycle_raw and _k_rising(rise_bars) and (not require_d_slope or row["SlowD"] > prev["SlowD"])
    bear_cycle = bear_cycle_raw and _k_falling(rise_bars) and (not require_d_slope or row["SlowD"] < prev["SlowD"])
    cycle_ok = bull_cycle if direction == "long" else bear_cycle if direction == "short" else False

    # 4) SUPPORT/RESISTANCE — pivot structure; EMA fallback optional separate flag (not alone unless enabled)
    scfg = _cfg_path(cfg, "signals", "sr_pivots", default={})
    lookback = _get_int(scfg, "lookback", 60)
    left     = _get_int(scfg, "left", 3)
    right    = _get_int(scfg, "right", 3)
    near_pct = _get_float(scfg, "near_pct", 0.02)
    near_atr = _get_float(scfg, "near_atr_mult", 0.75)
    yday_ok  = _get_bool(scfg, "yesterday_touch_ok", True)

    ema_fall_raw = scfg.get("ema50_fallback_pct", 0.025)
    allow_fallback_as_sr = _get_bool(scfg, "allow_fallback_as_core", False)  # new: fallback alone can't satisfy S/R unless True
    ema_fall = None
    try:
        if ema_fall_raw is not None:
            ema_fall = float(ema_fall_raw)
    except Exception:
        ema_fall = None

    price = float(row["Close"])
    support, resistance = _nearest_pivot_levels(df, lookback, left, right, price=price)

    tol_abs = float(row["ATR14"]) * near_atr if np.isfinite(row["ATR14"]) else np.inf
    tol_pct = price * near_pct
    # tolerance mode configurable: "max" (original) or "min"
    tol_mode = str(scfg.get("tolerance_mode", "max")).lower()
    tol = max(tol_abs, tol_pct) if tol_mode == "max" else min(tol_abs, tol_pct)

    sr_used = None
    sr_ok = False

    if direction == "long" and (support is not None):
        sr_ok = (support <= price) and (abs(price - support) <= tol)
        sr_used = "pivot-support" if sr_ok else None
        # yesterday touch
        if (not sr_ok) and yday_ok and len(df) >= 2:
            prev_price = float(prev["Close"])
            prev_atr = float(prev.get("ATR14", np.nan))
            tol_prev = max((prev_atr * near_atr) if np.isfinite(prev_atr) else 0.0, prev_price * near_pct)
            if (support <= prev_price) and (abs(prev_price - support) <= tol_prev):
                sr_ok = True
                sr_used = "pivot-support-yday"
    elif direction == "short" and (resistance is not None):
        sr_ok = (resistance >= price) and (abs(resistance - price) <= tol)
        sr_used = "pivot-resistance" if sr_ok else None
        if (not sr_ok) and yday_ok and len(df) >= 2:
            prev_price = float(prev["Close"])
            prev_atr = float(prev.get("ATR14", np.nan))
            tol_prev = max((prev_atr * near_atr) if np.isfinite(prev_atr) else 0.0, prev_price * near_pct)
            if (resistance >= prev_price) and (abs(resistance - prev_price) <= tol_prev):
                sr_ok = True
                sr_used = "pivot-resistance-yday"

    sr_fallback_used = False
    if (not sr_ok) and (ema_fall is not None):
        sr_fallback_hit = (abs(price - float(row["EMA50"])) / price) <= ema_fall
        if sr_fallback_hit:
            sr_fallback_used = True
            if allow_fallback_as_sr:
                sr_ok = True
                sr_used = "ema50-fallback"

    # 5) FRACTAL / VOLATILITY ENERGY (new core component) using ADX + (optional) choppiness
    fcfg = _cfg_path(cfg, "fractal", default={})
    f_enabled = _get_bool(fcfg, "enabled", True)
    f_use_chop = _get_bool(fcfg, "use_chop", True)
    f_adx_min = _get_float(fcfg, "adx_min", 18.0)
    f_chop_max = _get_float(fcfg, "chop_max", 58.0)
    # reuse existing values if present
    adx_val = float(row.get("ADX14", np.nan))
    # compute choppiness via regime helper if not already explained
    # we'll lazily compute with last 14 window by calling _chop_index_latest
    if f_use_chop:
        high_np = df["High"].to_numpy(dtype="float64")
        low_np = df["Low"].to_numpy(dtype="float64")
        close_np = df["Close"].to_numpy(dtype="float64")
        chop_val = _chop_index_latest(high_np, low_np, close_np, int(fcfg.get("chop_window", 14)))
    else:
        chop_val = np.nan
    adx_ok_f = (not np.isfinite(adx_val)) or (adx_val >= f_adx_min)
    chop_ok_f = (not f_use_chop) or (not np.isfinite(chop_val)) or (chop_val <= f_chop_max)
    fractal_ok = (not f_enabled) or (adx_ok_f and chop_ok_f)

    # VOLUME now treated as confirmation (not core energy)
    vcfg = _cfg_path(cfg, "volume", default={})
    volume_ok, vol_details = _volume_energy_ok(df, direction, vcfg)
    vol_details["counts_in_score"] = False

    # Energy weighting (core five only) - default weight=1 each unless provided
    weights_cfg = (cfg.get("trading", {}) or {}).get("energy_weights", {}) or {}
    def w(name: str) -> float:
        try:
            return float(weights_cfg.get(name, 1.0))
        except Exception:
            return 1.0
    core_pass = {
        "trend": trend_ok,
        "momentum": momentum_ok,
        "cycle": cycle_ok,
        "sr": sr_ok,
        "fractal": fractal_ok,
    }
    score_count = int(sum(core_pass.values()))
    score_weighted = float(sum(w(k) for k, v in core_pass.items() if v))
    # maintain backward-compatible 'score' as count
    score = score_count

    return {
        "direction": direction,
        "trend": trend_ok,
        "momentum": momentum_ok,
        "cycle": cycle_ok,
        "sr": sr_ok,
        "fractal": fractal_ok,
        "volume": volume_ok,  # confirmation only
        "score": score,  # count of core passes
        "score_weighted": score_weighted,
        "core_pass_count": score_count,
        "explain": {
            "trend": {"bull": bool(bullish_trend), "bear": bool(bearish_trend)},
            "momentum": {
                "base_bull": bool(base_bull), "base_bear": bool(base_bear),
                "rsi_filter": rsi_filter, "rsi_threshold": rsi_threshold,
                "zero_line_mode": mcfg.get("zero_line_mode", "confirm"),
                "zl_ok": bool(zl_ok),
                "use_expansion": use_expansion,
                "exp_lookback": _get_int(mcfg, "expansion_lookback", 3),
                "exp_min_steps": _get_int(mcfg, "expansion_min_steps", 1),
                "exp_ok": bool(exp_ok),
                "macd_abs_min": macd_abs_min,
            },
            "cycle": {"bull": bool(bull_cycle), "bear": bool(bear_cycle), "rise_bars": rise_bars, "require_d_slope": require_d_slope},
            "sr": {
                "support": support, "resistance": resistance,
                "tol": tol, "near_pct": near_pct, "near_atr_mult": near_atr,
                "ema50_fallback_pct": ema_fall, "used": sr_used,
                "fallback_used": sr_fallback_used,
                "allow_fallback_as_core": allow_fallback_as_sr,
                "yesterday_touch_ok": yday_ok
            },
            "fractal": {
                "adx": adx_val if np.isfinite(adx_val) else None,
                "adx_min": f_adx_min,
                "chop": chop_val if np.isfinite(chop_val) else None,
                "chop_max": f_chop_max,
                "use_chop": f_use_chop,
                "enabled": f_enabled,
                "ok": fractal_ok,
            },
            "volume": vol_details,
            "weights": {k: w(k) for k in core_pass.keys()},
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

    # Optionally skip ADX in regime if fractal energy already handles it
    skip_adx_if_fractal = bool(rcfg.get("skip_adx_if_fractal", True))
    fractal_enabled = bool((cfg.get("fractal", {}) or {}).get("enabled", True))
    adx_val = float(last.get("ADX14", np.nan))
    if skip_adx_if_fractal and fractal_enabled:
        adx_ok = True
        det["adx_skipped"] = True
    else:
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

    # ---- MTF as BONUS energy ----
    mtf_cfg = (cfg.get("trading", {}).get("mtf") or {})
    mtf_counts = bool(mtf_cfg.get("counts_as_energy", True))        # whether weekly adds +1
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
        passes = {k: bool(energies.get(k)) for k in ("trend","momentum","cycle","sr","fractal")}
        passes["core_pass_count"] = int(energies["score"])
        _explain_log(symbol, f"core<{core_min}", passes, cfg)
        return None
    # Minimum score gate (possibly same as core_min or higher including bonus)
    min_score = int(cfg.get("trading", {}).get("min_score", core_min))

    if (mtf_reject and not mtf_ok):
        _explain_log(symbol, "mtf", {"reason": mtf_reason}, cfg)
        return None

    if eff_score < min_score:
        passes = {k: bool(energies.get(k)) for k in ("trend","momentum","cycle","sr","fractal")}
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
            f"S/R:{energies['sr']} Frac:{energies['fractal']} VolC:{energies['volume']} | "
            f"core:{energies['score']} w:{energies['score_weighted']:.2f} MTF:{mtf_reason} (+{mtf_bonus})"
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
