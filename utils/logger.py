from pathlib import Path
import pandas as pd
from datetime import date

def log_dataframe(df: pd.DataFrame, out_path: Path):
    """
    Append or create a CSV. Keeps header only on first write.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=header, index=False)

def today_filename(prefix: str) -> Path:
    """logs/signals_2025-08-01.csv  (uses today’s date)"""
    return Path("logs") / f"{prefix}_{date.today()}.csv"
