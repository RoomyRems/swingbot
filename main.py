from data.fetcher import fetch_ohlcv
from strategies.swing_strategy import add_indicators, evaluate_five_energies, build_trade_signal
from utils.config import load_config

SYMBOL = "AAPL"
cfg = load_config("config.yaml")

# Fetch ~120 bars for stability
df = fetch_ohlcv(SYMBOL, interval="1d", lookback_days=120)
df = add_indicators(df)

energies = evaluate_five_energies(df)
print(df[["Close","EMA20","EMA50","RSI14","MACD","MACDs","MACDh","SlowK","SlowD","OBV","AvgVol50"]].tail(5))
print("\n5-Energy verdict:", energies)

signal = build_trade_signal(SYMBOL, df, cfg)
if signal is None:
    print("\nNo trade (score below threshold or invalid risk setup).")
else:
    print("\nTRADE SIGNAL")
    print(f"Symbol:   {signal.symbol}")
    print(f"Side:     {signal.direction.upper()}   (score={signal.score})")
    print(f"Entry:    {signal.entry:.2f}")
    print(f"Stop:     {signal.stop:.2f}    (per-share risk {signal.per_share_risk:.2f})")
    print(f"Target:   {signal.target:.2f}  (~{cfg['risk']['reward_multiple']}R)")
    print(f"Quantity: {signal.quantity}    (total risk ${signal.total_risk:,.2f})")
    print(f"Notes:    {signal.notes}")
