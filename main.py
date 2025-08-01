from data.fetcher import fetch_ohlcv
from strategies.swing_strategy import add_indicators, evaluate_five_energies

# Pull ~120 daily bars so 50EMA & 50-day volume average are solid
df = fetch_ohlcv("AAPL", interval="1d", lookback_days=120)

df = add_indicators(df)
energies = evaluate_five_energies(df)

print(df[["Close","EMA20","EMA50","RSI14","MACD","MACDs","MACDh","SlowK","SlowD","OBV","AvgVol50"]].tail(5))
print("\n5-Energy verdict:", energies)
