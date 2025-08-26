from pathlib import Path
import pandas as pd
from datetime import date, datetime

def log_dataframe(df: pd.DataFrame, out_path: Path, overwrite: bool = True):
    """
    Write DataFrame to CSV.
    Default now overwrites each run to avoid skewing diagnostics with appended prior runs.
    Set overwrite=False to append (retains previous behavior).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        df.to_csv(out_path, mode="w", header=True, index=False)
    else:
        header = not out_path.exists()
        df.to_csv(out_path, mode="a", header=header, index=False)

def today_filename(prefix: str, unique: bool = False) -> Path:
    """
    Returns path like logs/prefix_YYYY-MM-DD.csv.
    If unique=True, include timestamp to second for multiple runs per day.
    """
    if unique:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        return Path("logs") / f"{prefix}_{stamp}.csv"
    return Path("logs") / f"{prefix}_{date.today()}.csv"
