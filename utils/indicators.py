import numpy as np
import pandas as pd

def linreg_slope(series: pd.Series) -> float:
    """Return raw OLS slope per bar for the provided series.

    Parameters
    ----------
    series : pd.Series
        Numeric series ordered in time. Index is ignored.
    Returns
    -------
    float
        Slope coefficient (units of series units per bar). 0.0 if insufficient data.
    """
    if series is None:
        return 0.0
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0
    y = s.values.astype(float)
    x = np.arange(len(y), dtype=float)
    try:
        m, _ = np.polyfit(x, y, 1)
        return float(m)
    except Exception:
        return 0.0

def pct_norm_slope(series: pd.Series) -> float:
    """Return slope normalized as pct per bar using last value as scale.

    Returns 0.0 when series too short or last value non-finite / zero.
    """
    if series is None:
        return 0.0
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0
    last = float(s.iloc[-1])
    if not np.isfinite(last) or last == 0:
        return 0.0
    raw = linreg_slope(s)
    return raw / last if last != 0 else 0.0
