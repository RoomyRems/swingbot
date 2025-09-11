# scripts/summarize_evals.py
from __future__ import annotations
import sys
from pathlib import Path

# --- make repo root importable when run as a script ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # parent of "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from utils.config import load_config

def main(path: str | None = None):
    # pick today's by default
    if path is None:
        from utils.logger import today_filename
        p = today_filename("backtest_evaluations")
    else:
        p = Path(path)
    if not p.exists():
        print(f">> \n\nFile not found: {p}")
        sys.exit(1)

    df = pd.read_csv(p, parse_dates=["date"])
    cfg = load_config("config.yaml")
    min_score = int(cfg.get("trading", {}).get("min_score", 4))

    # Prefer effective score if present (includes MTF bonus)
    score_col = "score_eff" if "score_eff" in df.columns else "score"
    max_score = int(df[score_col].max()) if not df.empty else 0

    syms = df["symbol"].nunique()
    d0, d1 = df["date"].min(), df["date"].max()
    print(">>\n")
    print(f"File: {p.name}")
    print(f"Rows: {len(df):,}  Symbols: {syms}  Dates: {d0} → {d1}\n")

    # Pass rates
    regime_ok = df["regime_ok"].mean() if "regime_ok" in df.columns else float("nan")
    has_dir   = (df["direction"].isin(["long","short"])).mean()
    score_ok  = (df[score_col] >= min_score).mean()
    mtf_ok    = df["mtf_ok"].mean() if "mtf_ok" in df.columns else float("nan")

    print("Pass rates (overall):")
    print(f"  regime_ok      : {regime_ok:0.1%}")
    print(f"  has_direction  : {has_dir:0.1%}")
    print(f"  score>={min_score}: {score_ok:0.1%}")
    if "mtf_ok" in df.columns:
        print(f"  mtf_ok         : {mtf_ok:0.1%}")
    print()

    # Energy component “true rates” (eligible rows only)
    elig = df.copy()
    if "regime_ok" in elig.columns:
        elig = elig[elig["regime_ok"] == True]
    if "direction" in elig.columns:
        elig = elig[elig["direction"].isin(["long","short"])]
    def rate(col):
        if col not in elig.columns or len(elig) == 0:
            return float("nan")
        return float(elig[col].mean())
    print("Energy components true-rate (eligible rows):")
    for col in ["trend","momentum","cycle","sr","scale","volume_confirm"]:
        print(f"  {col:<13}: {rate(col):0.1%}")
    # If trend_raw is present, show it (represents 'has direction' trend prior to waves gating)
    if "trend_raw" in elig.columns:
        print(f"  {'trend_raw':<13}: {rate('trend_raw'):0.1%}  (direction present)")
    print()

    # Score distribution (effective if available)
    print("Score distribution (using " + score_col + "):")
    # Expect 0..6 when score_eff exists
    top = max(max_score, 6 if score_col=="score_eff" else 5)
    for s in range(0, top+1):
        cnt = int((df[score_col] == s).sum())
        pct = cnt / len(df) if len(df) else 0.0
        print(f"  score {s}: {cnt:7d}  ({pct:0.1%})")
    print()

    # Sequential survivors (use effective score in the gate)
    seq1 = df[df.get("regime_ok", True) == True]
    seq2 = seq1[seq1["direction"].isin(["long","short"])]
    seq3 = seq2[seq2[score_col] >= min_score]
    seq4 = seq3[(seq3.get("mtf_ok", True) == True)]  # only meaningful if you later set reject_on_mismatch True

    def line(name, x):
        print(f"  after {name:<14}: {len(x):7d}  ({(len(x)/len(df) if len(df) else 0):0.1%})")
    print("Sequential survivors:")
    line("regime_ok", seq1)
    line("direction", seq2)
    line(f"score>={min_score}", seq3)
    line("MTF", seq4)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
