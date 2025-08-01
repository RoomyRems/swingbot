import yfinance as yf
import pandas as pd

def fetch_ohlcv(symbol: str, interval: str = "1d", lookback_days: int = 100) -> pd.DataFrame:
    """
    Fetch historical OHLCV data using yfinance.

    Args:
        symbol (str): Ticker symbol (e.g. 'AAPL', 'MSFT').
        interval (str): Data interval (e.g. '1d', '1h').
        lookback_days (int): Number of past days to retrieve.

    Returns:
        pd.DataFrame: OHLCV dataframe with datetime index.
    """
    df = yf.download(
        tickers=symbol,
        period=f"{lookback_days}d",
        interval=interval,
        progress=False,
        auto_adjust=False
    )
    return df
