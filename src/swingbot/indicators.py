from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Return validated, date-indexed OHLCV bars with canonical lowercase columns."""
    if frame.empty:
        raise ValueError("bar data is empty")

    result = frame.copy()
    result.columns = [str(column).strip().lower().replace(" ", "_") for column in result.columns]
    if not isinstance(result.index, pd.DatetimeIndex):
        timestamp_column = next(
            (column for column in ("timestamp", "date", "datetime") if column in result.columns),
            None,
        )
        if timestamp_column is None:
            raise ValueError("bars require a DatetimeIndex or timestamp/date column")
        result.index = pd.to_datetime(result.pop(timestamp_column), utc=True)
    else:
        result.index = pd.to_datetime(result.index, utc=True)

    result.index = result.index.tz_convert(None).normalize()
    missing = [column for column in REQUIRED_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"bar data is missing: {', '.join(missing)}")
    result = result.loc[:, REQUIRED_COLUMNS].sort_index()
    if result.index.has_duplicates:
        duplicates = result.index[result.index.duplicated()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"bar data contains duplicate dates: {', '.join(duplicates[:3])}")

    result = result.apply(pd.to_numeric, errors="raise").astype(float)
    if result.isna().any().any():
        raise ValueError("bar data contains missing OHLCV values")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (result["volume"] < 0).any():
        raise ValueError("volume must not be negative")
    if (result["high"] < result[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("a bar high is below another OHLC value")
    if (result["low"] > result[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("a bar low is above another OHLC value")
    result.index.name = "date"
    return result


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_line = ema(close, fast)
    slow_line = ema(close, slow)
    line = fast_line - slow_line
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = line - signal_line
    return line, signal_line, histogram


def slow_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 5,
    k_smoothing: int = 2,
    d_smoothing: int = 3,
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(lookback, min_periods=lookback).min()
    highest = high.rolling(lookback, min_periods=lookback).max()
    width = (highest - lowest).where((highest - lowest) != 0)
    raw_k = 100.0 * (close - lowest) / width
    slow_k = raw_k.rolling(k_smoothing, min_periods=k_smoothing).mean()
    slow_d = slow_k.rolling(d_smoothing, min_periods=d_smoothing).mean()
    return slow_k, slow_d


def atr(frame: pd.DataFrame, lookback: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(lookback, min_periods=lookback).mean()


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add only causal indicators; values at date t never use bars after t."""
    result = normalize_bars(frame)
    result["ema15"] = ema(result["close"], 15)
    result["sma50"] = result["close"].rolling(50, min_periods=50).mean()
    result["atr14"] = atr(result, 14)

    result["stoch_k"], result["stoch_d"] = slow_stochastic(
        result["high"], result["low"], result["close"]
    )
    result["macd"], result["macd_signal"], result["macd_hist"] = macd(result["close"])

    # A resampled week is labeled on Friday. Monday-Thursday therefore see only
    # the previous completed Friday, even though the full daily frame is present.
    weekly_close = result["close"].resample("W-FRI").last()
    weekly_macd, weekly_signal, weekly_hist = macd(weekly_close)
    weekly = pd.DataFrame(
        {
            "weekly_macd": weekly_macd,
            "weekly_macd_signal": weekly_signal,
            "weekly_macd_hist": weekly_hist,
            "weekly_macd_hist_delta": weekly_hist.diff(),
        }
    )
    aligned_weekly = weekly.reindex(result.index, method="ffill")
    for column in aligned_weekly:
        result[column] = aligned_weekly[column]
    return result
