# strategies/swing_strategy.py
from __future__ import annotations

import numpy as np
import pandas as pd
import talib
from typing import Tuple, Optional

from models.signal import TradeSignal
from risk.manager import compute_levels, size_position


# ----------------------------- Indicators ------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty DataFrame")

    # Flatten yfinance MultiIndex if present: ('Adj Close','AAPL') → 'Adj Close'
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Ensure we have 'Close'
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    for col in ("Close", "High", "Low", "Volume"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    close = np.asarray(df["Close"], dtype="float64").ravel()
    high  = np.asarray(df["High"],  dtype="float64").ravel()
    low   = np.asarray(df["Low"],   dtype="float64").ravel()
    vol   = np.asarray(df["Volume"], dtype="float64").ravel()

    # Trend
    ema20 = talib.EMA(close, timeperiod=20)
    ema50 = talib.EMA(close, timeperiod=50)

    # Momentum (MACD)
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

    # Volume & confirmation
    obv = talib.OBV(close, vol)
    avgvol50 = pd.Series(vol, index=df.index).rolling(50).mean().to_numpy()

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
    out["OBV"]      = pd.Series(obv,   index=df.index)
    out["AvgVol50"] = pd.Series(avgvol50, index=df.index)
    return out


# --------------------------- Pivot utilities ---------------------------
def _find_pivots(high: pd.Series, low: pd.Series, left: int, right: int) -> Tuple[pd.Series, pd.Series]:
    """
    Return boolean Series for pivot highs and pivot lows using left/right window.
    A pivot high at i means: High[i] is the max within [i-left, i+right].
    A pivot low  at i means: Low[i]  is the min within [i-left, i+right].
    """
    # Rolling windows
    rh = high.rolling(window=left + right + 1, center=True).max()
    rl = low.rolling(window=left + right + 1, center=True).min()

    # To ensure strict pivots, compare to shifted windows for edges
    ph = (high == rh)
    pl = (low == rl)

    # Avoid NaNs at edges
    ph = ph.fillna(False)
    pl = pl.fillna(False)
    return ph, pl


def _nearest_pivot_dist(row_close: float,
                        pivots: pd.Series,
                        prices: pd.Series) -> Optional[float]:
    """
    Distance (in % of close) to the nearest pivot level from the recent window.
    """
    if pivots is None or pivots.empty:
        return None
    lvls = prices[pivots].dropna()
    if lvls.empty:
        return None
    dists = (lvls - row_close).abs() / max(1e-6, row_close)
    return float(dists.min()) if not dists.empty else None


def _sr_ok_with_pivots(df: pd.DataFrame, direction: str, cfg: dict) -> Tuple[bool, dict]:
    """
    Decide S/R energy using swing pivots with EMA50 fallback.
    Returns (ok, explain_dict).
    """
    scfg = (cfg.get("signals", {}) or {}).get("sr_pivots", {}) or {}
    lookback = int(scfg.get("lookback", 60))
    left = int(scfg.get("left", 3))
    right = int(scfg.get("right", 3))
    near_pct = float(scfg.get("near_pct", 0.02))
    near_atr_mult = float(scfg.get("near_atr_mult", 0.5))
    ema50_fallback_pct = float(scfg.get("ema50_fallback_pct", 0.02))

    sub = df.tail(lookback).copy()
    row = sub.iloc[-1]

    # Find pivots over the lookback
    ph, pl = _find_pivots(sub["High"], sub["Low"], left, right)

    # Nearest distances
    dist_to_low  = _nearest_pivot_dist(float(row["Close"]), pl, sub["Low"])
    dist_to_high = _nearest_pivot_dist(float(row["Close"]), ph, sub["High"])

    # Dynamic ATR buffer
    atr = float(row["ATR14"]) if pd.notna(row["ATR14"]) else 0.0
    atr_buffer = (atr / max(1e-6, float(row["Close"]))) * near_atr_mult

    ok_long = False
    ok_short = False

    # Longs: prefer proximity to pivot LOW (support)
    if dist_to_low is not None:
        if (dist_to_low <= near_pct) or (atr_buffer > 0 and dist_to_low <= atr_buffer):
            ok_long = True

    # Shorts: prefer proximity to pivot HIGH (resistance)
    if dist_to_high is not None:
        if (dist_to_high <= near_pct) or (atr_buffer > 0 and dist_to_high <= atr_buffer):
            ok_short = True

    # Fallback: EMA50 “value zone”
    ema50_dist = abs(float(row["Close"]) - float(row["EMA50"])) / max(1e-6, float(row["Close"]))
    if not ok_long and direction == "long":
        ok_long = ema50_dist <= ema50_fallback_pct
    if not ok_short and direction == "short":
        ok_short = ema50_dist <= ema50_fallback_pct

    explain = {
        "dist_to_pivot_low_pct":  None if dist_to_low is None else f"{dist_to_low:.4f}",
        "dist_to_pivot_high_pct": None if dist_to_high is None else f"{dist_to_high:.4f}",
        "atr_buffer_pct": f"{atr_buffer:.4f}",
        "ema50_dist_pct": f"{ema50_dist:.4f}",
    }

    return (ok_long if direction == "long" else ok_short), explain


# ------------------------- Momentum utilities --------------------------
def _macd_zero_context(macd_line: pd.Series, direction: str, mode: str, cross_lookback: int) -> bool:
    """
    Zero-line context for MACD line.
      - 'none'    : always True
      - 'confirm' : long requires MACD > 0, short requires MACD < 0
      - 'cross'   : a zero-cross occurred within the last `cross_lookback` bars
    """
    if macd_line.isna().iloc[-1]:
        return False

    mode = (mode or "none").lower()
    if mode == "none":
        return True

    last = float(macd_line.iloc[-1])

    if mode == "confirm":
        return (last > 0) if direction == "long" else (last < 0)

    if mode == "cross":
        # Require sign change inside the last K bars
        k = max(1, int(cross_lookback))
        seg = macd_line.tail(k + 1).dropna()
        if seg.shape[0] < 2:
            return False
        signs = np.sign(seg.values)
        return np.any(signs[:-1] * signs[1:] <= 0)  # any adjacent sign change

    # Fallback
    return True


def _macd_expansion(macdh: pd.Series, direction: str, lookback: int, min_steps: int) -> bool:
    """
    Require histogram expansion: for longs, MACDh increasing (more positive);
    for shorts, MACDh decreasing (more negative). Not strictly monotonic:
    we need at least `min_steps` of the last `lookback-1` diffs in the desired direction.
    Also enforces current MACDh sign ( >0 for long, <0 for short ).
    """
    k = max(2, int(lookback))
    seg = macdh.tail(k).dropna()
    if seg.shape[0] < k:
        return False

    diffs = seg.diff().dropna()  # length k-1
    if direction == "long":
        in_dir = int((diffs > 0).sum())
        return (seg.iloc[-1] > 0) and (in_dir >= max(1, int(min_steps)))
    else:
        in_dir = int((diffs < 0).sum())
        return (seg.iloc[-1] < 0) and (in_dir >= max(1, int(min_steps)))


def _momentum_ok(df: pd.DataFrame, direction: str, cfg: dict) -> Tuple[bool, dict]:
    """
    Apply base momentum rules + optional expansion and zero-line context.
    Returns (ok, explain_dict)
    """
    row = df.iloc[-1]
    prev = df.iloc[-2]

    mcfg = (cfg.get("signals", {}) or {}).get("momentum", {}) or {}

    # Base momentum (as before)
    bull_base = (row["MACDh"] > 0) and (row["MACD"] > row["MACDs"])
    bear_base = (row["MACDh"] < 0) and (row["MACD"] < row["MACDs"])

    rsi_filter = bool(mcfg.get("rsi_filter", True))
    rsi_thr = float(mcfg.get("rsi_threshold", 50))

    if rsi_filter:
        bull_base = bull_base and (row["RSI14"] >= rsi_thr)
        bear_base = bear_base and (row["RSI14"] <= (100 - rsi_thr))  # symmetric gate

    base_ok = bull_base if direction == "long" else bear_base if direction == "short" else (bull_base or bear_base)

    # Zero-line context for MACD line
    zero_mode = str(mcfg.get("zero_line_mode", "none")).lower()
    zero_lb = int(mcfg.get("zero_cross_lookback", 3))
    zero_ok = _macd_zero_context(df["MACD"], direction, zero_mode, zero_lb)

    # Histogram expansion
    use_exp = bool(mcfg.get("use_macd_expansion", True))
    exp_lb = int(mcfg.get("expansion_lookback", 3))
    exp_steps = int(mcfg.get("expansion_min_steps", 2))
    exp_ok = _macd_expansion(df["MACDh"], direction, exp_lb, exp_steps) if use_exp else True

    final_ok = bool(base_ok and zero_ok and exp_ok)
    explain = {
        "base_ok": bool(base_ok),
        "zero_ok": bool(zero_ok),
        "exp_ok": bool(exp_ok),
        "rsi_filter": bool(rsi_filter),
        "zero_mode": zero_mode,
        "expansion_lb": exp_lb,
        "expansion_steps_req": exp_steps,
    }
    return final_ok, explain


# ------------------------ Five Energies score --------------------------
def evaluate_five_energies(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    """
    Evaluate the five energies on the last bar.
    `cfg` is optional; when provided, momentum & S/R logic can use its signal settings.
    """
    if len(df) < 60:
        raise ValueError("Need ~60 bars for stable signals")

    cfg = cfg or {}

    row  = df.iloc[-1]
    prev = df.iloc[-2]

    # 1) TREND (20/50 EMA + price relative to EMA50)
    bullish_trend = (row["EMA20"] > row["EMA50"]) and (row["Close"] > row["EMA50"])
    bearish_trend = (row["EMA20"] < row["EMA50"]) and (row["Close"] < row["EMA50"])
    direction = "long" if bullish_trend else "short" if bearish_trend else "none"
    trend_ok = bullish_trend or bearish_trend

    # 2) MOMENTUM (with optional MACD expansion & zero-line context)
    momentum_ok, mom_explain = _momentum_ok(df, direction, cfg)

    # 3) CYCLE (Stoch turning from extreme)
    bull_cycle = (prev["SlowK"] < 20) and (row["SlowK"] > row["SlowD"])
    bear_cycle = (prev["SlowK"] > 80) and (row["SlowK"] < row["SlowD"])
    cycle_ok = bull_cycle if direction == "long" else bear_cycle if direction == "short" else (bull_cycle or bear_cycle)

    # 4) SUPPORT/RESISTANCE (pivot-based + EMA50 fallback)
    sr_ok, sr_explain = _sr_ok_with_pivots(df, direction, cfg)

    # 5) VOLUME (above avg & OBV slope direction)
    bull_vol = pd.notna(row["AvgVol50"]) and (row["Volume"] > 1.2 * row["AvgVol50"]) and (row["OBV"] > prev["OBV"])
    bear_vol = pd.notna(row["AvgVol50"]) and (row["Volume"] > 1.2 * row["AvgVol50"]) and (row["OBV"] < prev["OBV"])
    volume_ok = bull_vol if direction == "long" else bear_vol if direction == "short" else False

    score = int(sum([trend_ok, momentum_ok, cycle_ok, sr_ok, volume_ok]))

    explain = {
        "bullish_trend": bullish_trend, "bearish_trend": bearish_trend,
        "bull_cycle": bull_cycle, "bear_cycle": bear_cycle,
        "bull_vol": bull_vol, "bear_vol": bear_vol,
        # Momentum + SR details:
        "momentum": mom_explain,
        "sr": sr_explain,
    }

    return {
        "direction": direction,
        "trend": trend_ok,
        "momentum": momentum_ok,
        "cycle": cycle_ok,
        "sr": sr_ok,
        "volume": volume_ok,
        "score": score,
        "explain": explain,
    }


# ---------------------------- Demo signals -----------------------------
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


# ------------------------- Trade signal builder ------------------------
def build_trade_signal(symbol: str, df: pd.DataFrame, cfg: dict) -> TradeSignal | None:
    """
    If min_score energies align, create a TradeSignal with ATR-based stop/target and position size.
    """
    min_score = int(cfg.get("trading", {}).get("min_score", 4))

    energies = evaluate_five_energies(df, cfg=cfg)
    if energies["score"] < min_score or energies["direction"] not in ("long", "short"):
        return None

    row   = df.iloc[-1]
    close = float(row["Close"])
    atr   = float(row["ATR14"])

    # 1) Compute exits
    stop, target = compute_levels(
        direction   = energies["direction"],
        entry       = close,
        atr         = atr,
        atr_mult    = float(cfg["risk"]["atr_multiple_stop"]),
        reward_mult = float(cfg["risk"]["reward_multiple"]),
    )
    if stop is None or target is None:
        return None

    # 2) Position size with centralized manager
    qty = size_position(cfg=cfg, entry=close, stop=stop)
    if qty <= 0:
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
            f"Cycle:{energies['cycle']} S/R:{energies['sr']} Vol:{energies['volume']}"
        ),
    )
