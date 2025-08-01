import numpy as np
import pandas as pd
import talib

from models.signal import TradeSignal
from risk.position import calc_stop_and_target, calc_position_size


# ---------- 1) compute all indicators we need ----------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty DataFrame")

    # ----- flatten yfinance MultiIndex: ('Adj Close','AAPL') → 'Adj Close'
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # ----- ensure we have a 'Close' column (rename Adj Close if needed)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    # Ensure basic columns exist
    for col in ("Close", "High", "Low", "Volume"):
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

    # Volume & confirmation
    obv = talib.OBV(close, vol)
    avgvol50 = pd.Series(vol, index=df.index).rolling(50).mean().to_numpy()

    out = df.copy()
    out["EMA20"]  = pd.Series(ema20, index=df.index)
    out["EMA50"]  = pd.Series(ema50, index=df.index)
    out["MACD"]   = pd.Series(macd,  index=df.index)
    out["MACDs"]  = pd.Series(macds, index=df.index)
    out["MACDh"]  = pd.Series(macdh, index=df.index)
    out["RSI14"]  = pd.Series(rsi14, index=df.index)
    out["SlowK"]  = pd.Series(slowk, index=df.index)
    out["SlowD"]  = pd.Series(slowd, index=df.index)
    out["ATR14"]  = pd.Series(atr14, index=df.index)
    out["OBV"]    = pd.Series(obv,   index=df.index)
    out["AvgVol50"] = pd.Series(avgvol50, index=df.index)
    return out


# ---------- 2) evaluate the five energies on the last bar ----------
def evaluate_five_energies(df: pd.DataFrame) -> dict:
    if len(df) < 60:
        raise ValueError("Need ~60 bars for stable signals")

    row  = df.iloc[-1]
    prev = df.iloc[-2]

    # 1) TREND (20/50 EMA + price relative to EMA50)
    bullish_trend = (row["EMA20"] > row["EMA50"]) and (row["Close"] > row["EMA50"])
    bearish_trend = (row["EMA20"] < row["EMA50"]) and (row["Close"] < row["EMA50"])
    direction = "long" if bullish_trend else "short" if bearish_trend else "none"
    trend_ok = bullish_trend or bearish_trend

    # 2) MOMENTUM (MACD hist sign + line >/< signal + RSI over/under 50)
    bull_mom = (row["MACDh"] > 0) and (row["MACD"] > row["MACDs"]) and (row["RSI14"] > 50)
    bear_mom = (row["MACDh"] < 0) and (row["MACD"] < row["MACDs"]) and (row["RSI14"] < 50)
    momentum_ok = bull_mom if direction == "long" else bear_mom if direction == "short" else (bull_mom or bear_mom)

    # 3) CYCLE (Stoch turning from extreme)
    bull_cycle = (prev["SlowK"] < 20) and (row["SlowK"] > row["SlowD"])
    bear_cycle = (prev["SlowK"] > 80) and (row["SlowK"] < row["SlowD"])
    cycle_ok = bull_cycle if direction == "long" else bear_cycle if direction == "short" else (bull_cycle or bear_cycle)

    # 4) SUPPORT/RESISTANCE (near recent swing or EMA50 “value zone”)
    look = df.tail(20)
    support   = look["Low"].min()
    resistance= look["High"].max()
    near_support    = (abs(row["Close"] - support)  / row["Close"] < 0.03) or (abs(row["Close"] - row["EMA50"]) / row["Close"] < 0.02)
    near_resistance = (abs(row["Close"] - resistance)/ row["Close"] < 0.03) or (abs(row["Close"] - row["EMA50"]) / row["Close"] < 0.02)
    sr_ok = near_support if direction == "long" else near_resistance if direction == "short" else False

    # 5) VOLUME (above avg & OBV slope direction)
    bull_vol = (row["Volume"] > 1.2 * row["AvgVol50"]) and (row["OBV"] > prev["OBV"])
    bear_vol = (row["Volume"] > 1.2 * row["AvgVol50"]) and (row["OBV"] < prev["OBV"])
    volume_ok = bull_vol if direction == "long" else bear_vol if direction == "short" else False

    score = int(sum([trend_ok, momentum_ok, cycle_ok, sr_ok, volume_ok]))

    return {
        "direction": direction,
        "trend": trend_ok,
        "momentum": momentum_ok,
        "cycle": cycle_ok,
        "sr": sr_ok,
        "volume": volume_ok,
        "score": score,
        # for transparency/debug
        "explain": {
            "bullish_trend": bullish_trend, "bearish_trend": bearish_trend,
            "bull_mom": bull_mom, "bear_mom": bear_mom,
            "bull_cycle": bull_cycle, "bear_cycle": bear_cycle,
            "near_support": near_support, "near_resistance": near_resistance,
            "bull_vol": bull_vol, "bear_vol": bear_vol,
        }
    }


# ---------- 3) (optional) keep your simple demo signals ----------
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    # light demo rule you already tested
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


def build_trade_signal(symbol: str, df: pd.DataFrame, cfg: dict) -> TradeSignal | None:
    """
    If 4/5 energies align, create a TradeSignal with ATR-based stop/target and position size.
    Returns None if no valid trade.
    """
    min_score = cfg["trading"]["min_score"]
    risk      = cfg["risk"]

    energies = evaluate_five_energies(df)
    if energies["score"] < min_score or energies["direction"] not in ("long", "short"):
        return None

    row = df.iloc[-1]
    close = float(row["Close"])
    atr   = float(row["ATR14"])

    stop, target = calc_stop_and_target(
        direction   = energies["direction"],
        close       = close,
        atr         = atr,
        atr_mult    = float(risk["atr_multiple_stop"]),
        reward_mult = float(risk["reward_multiple"]),
    )
    if stop is None or target is None:
        return None

    qty = calc_position_size(
        account_equity = float(risk["account_equity"]),
        risk_pct       = float(risk["risk_per_trade_pct"]),
        entry          = close,
        stop           = stop,
    )
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
        notes = f"Trend:{energies['trend']} Mom:{energies['momentum']} Cycle:{energies['cycle']} S/R:{energies['sr']} Vol:{energies['volume']}"
    )
