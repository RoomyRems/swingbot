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
        # stay nice with tqdm progress bars if present
        from tqdm import tqdm
        tqdm.write(msg)
    except Exception:
        print(msg)

def _explain_log(symbol: str, reason: str, details: dict | None, cfg: dict | None) -> None:
    if not _explain_enabled(cfg):
        return
    # keep it short: show up to ~6 key=val pairs
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

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Ensure we have a 'Close' column (rename Adj Close if needed)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    # Ensure basic columns exist
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
def _find_pivots(high: pd.Series, low: pd.Series, left: int, right: int) -> tuple[list[int], list[int]]:
    highs_idx, lows_idx = [], []
    H = high.to_numpy()
    L = low.to_numpy()
    n = len(H)
    for i in range(n):
        l = max(0, i - left)
        r = min(n, i + right + 1)
        if r - l < 1:
            continue
        if np.isfinite(H[i]) and H[i] == np.nanmax(H[l:r]):
            highs_idx.append(i)
        if np.isfinite(L[i]) and L[i] == np.nanmin(L[l:r]):
            lows_idx.append(i)
    return highs_idx, lows_idx

def _nearest_pivot_levels(df: pd.DataFrame, lookback: int, left: int, right: int, price: float | None = None):
    tail = df.tail(lookback)
    highs_idx, lows_idx = _find_pivots(tail["High"], tail["Low"], left, right)
    base = len(df) - len(tail)

    highs_abs = [base + i for i in highs_idx]
    lows_abs  = [base + i for i in lows_idx]

    if price is None:
        price = float(df["Close"].iloc[-1])

    # candidates near current price with direction
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
    min_steps = _get_int(mcfg, "expansion_min_steps", 2)
    if lookback < 1 or len(df) < lookback + 1:
        return True  # permissive when insufficient data

    macdh = df["MACDh"].to_numpy()
    if not np.isfinite(macdh[-1]):
        return False

    # sign-aware improvement over last `lookback` diffs
    diffs = np.diff(macdh[-(lookback+1):])  # len = lookback
    if direction == "long":
        # allow “improving toward/above zero” (doesn’t force >0 here;
        # zero-line policy is handled in _macd_zero_line_ok)
        good = np.sum(diffs > 0)
    else:
        good = np.sum(diffs < 0)

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


# ---------- volume context ----------
def _volume_energy_ok(df: pd.DataFrame, direction: str, vcfg: dict) -> tuple[bool, dict]:
    if direction not in {"long", "short"}:
        return False, {"reason": "no-direction"}

    rvol_thr = _get_float(vcfg, "rvol_threshold", 1.20)
    ad_look  = _get_int(vcfg, "ad_trend_lookback", 3)
    obv_conf = _get_bool(vcfg, "obv_confirm", True)

    row  = df.iloc[-1]
    rvol = float(row.get("RVOL20", np.nan))
    det = {"rvol": f"{rvol:.2f}" if np.isfinite(rvol) else "nan", "rvol_thr": rvol_thr,
           "ad_look": ad_look, "obv_conf": obv_conf}

    if not np.isfinite(rvol):
        return False, {"reason": "rvol-nan", **det}
    rvol_ok = rvol >= rvol_thr

    if ad_look < 1 or len(df) <= ad_look:
        ad_ok = True
    else:
        ad_now = float(df["ADLine"].iloc[-1])
        ad_prev= float(df["ADLine"].iloc[-ad_look-1])
        ad_ok  = (ad_now > ad_prev) if direction == "long" else (ad_now < ad_prev)

    if obv_conf:
        obv_now  = float(df["OBV"].iloc[-1])
        obv_prev = float(df["OBV"].iloc[-2])
        obv_ok   = (obv_now > obv_prev) if direction == "long" else (obv_now < obv_prev)
    else:
        obv_ok = True

    ok = rvol_ok and ad_ok and obv_ok
    return ok, {**det, "ad_ok": ad_ok, "obv_ok": obv_ok}


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

    # 2) MOMENTUM with quality gates
    mcfg = _cfg_path(cfg, "momentum", default={})
    rsi_filter     = _get_bool(mcfg, "rsi_filter", True)
    rsi_threshold  = _get_float(mcfg, "rsi_threshold", 50.0)
    use_expansion  = _get_bool(mcfg, "use_macd_expansion", True)

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

    # 3) CYCLE
    bull_cycle = (prev["SlowK"] < 20) and (row["SlowK"] > row["SlowD"])
    bear_cycle = (prev["SlowK"] > 80) and (row["SlowK"] < row["SlowD"])
    cycle_ok = bull_cycle if direction == "long" else bear_cycle if direction == "short" else False

    # 4) SUPPORT/RESISTANCE — pivot-based + EMA50 fallback
    # --- S/R config ---
    scfg = _cfg_path(cfg, "signals", "sr_pivots", default={})
    lookback = _get_int(scfg, "lookback", 60)
    left     = _get_int(scfg, "left", 3)
    right    = _get_int(scfg, "right", 3)
    near_pct = _get_float(scfg, "near_pct", 0.02)
    near_atr = _get_float(scfg, "near_atr_mult", 0.5)

    # IMPORTANT: allow disabling EMA50 fallback when None/null
    ema_fall_raw = scfg.get("ema50_fallback_pct", 0.02)
    ema_fall = None
    try:
        if ema_fall_raw is not None:
            ema_fall = float(ema_fall_raw)
    except Exception:
        ema_fall = None  # if malformed, treat as disabled

    price = float(row["Close"])
    support, resistance = _nearest_pivot_levels(df, lookback, left, right, price=price)

    tol_abs = float(row["ATR14"]) * near_atr if np.isfinite(row["ATR14"]) else np.inf
    tol_pct = price * near_pct
    tol = max(tol_abs, tol_pct)

    sr_used = None
    if direction == "long" and (support is not None):
        sr_ok = (support <= price) and (abs(price - support) <= tol)
        sr_used = "pivot-support" if sr_ok else None
    elif direction == "short" and (resistance is not None):
        sr_ok = (resistance >= price) and (abs(resistance - price) <= tol)
        sr_used = "pivot-resistance" if sr_ok else None
    else:
        sr_ok = False

    # only if pivot check failed AND fallback is enabled
    if (not sr_ok) and (ema_fall is not None):
        sr_ok = (abs(price - float(row["EMA50"])) / price) <= ema_fall
        if sr_ok:
            sr_used = "ema50-fallback"

    # 5) VOLUME — RVOL + Chaikin A/D + optional OBV confirm
    vcfg = _cfg_path(cfg, "volume", default={})
    volume_ok, vol_details = _volume_energy_ok(df, direction, vcfg)

    score = int(sum([trend_ok, momentum_ok, cycle_ok, sr_ok, volume_ok]))

    return {
        "direction": direction,
        "trend": trend_ok,
        "momentum": momentum_ok,
        "cycle": cycle_ok,
        "sr": sr_ok,
        "volume": volume_ok,
        "score": score,
        "explain": {
            "trend": {"bull": bool(bullish_trend), "bear": bool(bearish_trend)},
            "momentum": {
                "base_bull": bool(base_bull), "base_bear": bool(base_bear),
                "rsi_filter": rsi_filter, "rsi_threshold": rsi_threshold,
                "zero_line_mode": mcfg.get("zero_line_mode", "confirm"),
                "zl_ok": bool(zl_ok),
                "use_expansion": use_expansion,
                "exp_lookback": _get_int(mcfg, "expansion_lookback", 3),
                "exp_min_steps": _get_int(mcfg, "expansion_min_steps", 2),
                "exp_ok": bool(exp_ok),
            },
            "cycle": {"bull": bool(bull_cycle), "bear": bool(bear_cycle)},
            "sr": {
                "support": support, "resistance": resistance,
                "tol": tol, "near_pct": near_pct, "near_atr_mult": near_atr,
                "ema50_fallback_pct": ema_fall, "used": sr_used
            },
            "volume": vol_details,
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
    """
    Resample daily → weekly (Friday close). Keep partial weeks.
    """
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

    adx_min      = float(rcfg.get("adx_min", 18))
    atr_pct_min  = float(rcfg.get("atr_pct_min", 0.015))
    use_chop     = bool(rcfg.get("use_chop", True))
    chop_window  = int(rcfg.get("chop_window", 14))
    chop_max     = float(rcfg.get("chop_max", 55))

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
    If 4/5 energies align, create a TradeSignal with ATR-based stop/target and position size.
    Adds higher-timeframe (weekly) confirmation if enabled in config.
    Also logs concise rejection reasons when a trade is skipped.
    """
    min_score = cfg["trading"]["min_score"]

    # 0) Market regime filter: bail early if conditions are poor
    regime_ok, regime_det = _regime_check(df, cfg)
    if not regime_ok:
        _explain_log(symbol, "regime", regime_det, cfg)
        return None

    # 1) Energies
    energies = evaluate_five_energies(df, cfg)
    if energies["direction"] not in ("long", "short"):
        _explain_log(symbol, "no-trend", {"ema20>ema50 & price>ema50 (long) OR opposite (short)": False}, cfg)
        return None

    if energies["score"] < min_score:
        # show which energies failed
        fails = {k: bool(energies[k]) for k in ("trend","momentum","cycle","sr","volume")}
        _explain_log(symbol, f"score<{min_score}", fails, cfg)
        return None

    # 2) Higher timeframe confirmation (weekly)
    mtf_cfg = (cfg.get("trading", {}).get("mtf") or {})
    if bool(mtf_cfg.get("enabled", False)):
        try:
            wk = _resample_weekly_ohlcv(df)
            wk_ind = add_indicators(wk)
            w_trend = _weekly_trend_view(wk_ind, mtf_cfg)
            mtf_ok, mtf_reason = _mtf_alignment_ok(energies["direction"], w_trend, mtf_cfg)
            if not mtf_ok:
                _explain_log(symbol, "mtf", {"reason": mtf_reason, "weekly": w_trend}, cfg)
                return None
        except Exception as e:
            # Don't block, but log the issue
            _explain_log(symbol, "mtf-error", {"err": str(e)}, cfg)
            mtf_reason = "mtf-skip-error"
    else:
        mtf_reason = "mtf-disabled"

    # 3) Risk levels and sizing
    row   = df.iloc[-1]
    close = float(row["Close"])
    atr   = float(row["ATR14"])

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
        score    = int(energies["score"]),
        entry    = close,
        stop     = stop,
        target   = target,
        quantity = qty,
        per_share_risk = per_share_risk,
        total_risk = total_risk,
        notes = (
            f"Trend:{energies['trend']} Mom:{energies['momentum']} "
            f"Cycle:{energies['cycle']} S/R:{energies['sr']} Vol:{energies['volume']} | "
            f"MTF:{mtf_reason}"
        ),
    )


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
