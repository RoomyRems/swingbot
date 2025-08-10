# scripts/summarize_evals.py
from __future__ import annotations

import sys, glob
from pathlib import Path
import pandas as pd

# --- ensure project root on sys.path (like run_backtest.py) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config  # now import works


def _latest_eval_path() -> Path | None:
    # prefer logs/, fall back to CWD
    candidates = sorted(
        [*Path("logs").glob("backtest_evaluations_*.csv"),
         *Path(".").glob("backtest_evaluations_*.csv")]
    )
    return candidates[-1] if candidates else None


def main():
    cfg = load_config("config.yaml")
    min_score = int(cfg["trading"]["min_score"])

    p = _latest_eval_path()
    if not p:
        print("No backtest_evaluations_*.csv found in logs/ or current dir.")
        sys.exit(1)

    df = pd.read_csv(p)
    if df.empty:
        print(f"{p.name} is empty.")
        sys.exit(0)

    # coerce bool-ish columns
    for col in ("regime_ok","trend","momentum","cycle","sr","volume","mtf_ok"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["1","true","t","yes","y"])

    total = len(df)
    uniq_syms = df["symbol"].nunique() if "symbol" in df.columns else 0
    date_min = df["date"].min() if "date" in df.columns else "?"
    date_max = df["date"].max() if "date" in df.columns else "?"

    print(f"\nFile: {p.name}")
    print(f"Rows: {total:,}  Symbols: {uniq_syms:,}  Dates: {date_min} → {date_max}")

    # Overall pass rates
    def rate(s): return float(s.mean()) if len(s) else 0.0

    has_dir = df.get("direction", pd.Series(index=df.index, dtype=object)).isin(["long","short"])

    print("\nPass rates (overall):")
    print(f"  {'regime_ok':15s}: {rate(df.get('regime_ok', pd.Series(False, index=df.index))):.1%}")
    print(f"  {'has_direction':15s}: {rate(has_dir):.1%}")
    print(f"  {'score>=min':15s}: {rate(df.get('score', pd.Series(0)).ge(min_score)):.1%}")
    if "mtf_ok" in df.columns:
        print(f"  {'mtf_ok':15s}: {rate(df['mtf_ok']):.1%}")

    # Energy components
    print("\nEnergy components true-rate:")
    for k in ["trend","momentum","cycle","sr","volume"]:
        if k in df.columns:
            print(f"  {k:15s}: {rate(df[k]):.1%}")

    # Score distribution
    if "score" in df.columns:
        print("\nScore distribution:")
        sc = df["score"].value_counts().sort_index()
        tot = sc.sum() or 1
        for s, c in sc.items():
            print(f"  score {int(s):>1}: {c:7d}  ({c/tot:.1%})")

    # Sequential survivors
    g0 = df
    g1 = g0[g0.get("regime_ok", False)]
    g2 = g1[g1.get("direction", "").isin(["long","short"])]
    g3 = g2[g2.get("score", 0) >= min_score]
    g4 = g3[g3.get("mtf_ok", True)]  # if mtf not present, treat as pass

    print("\nSequential survivors:")
    print(f"  after regime_ok      : {len(g1):7d}  ({len(g1)/total:.1%})")
    print(f"  after direction      : {len(g2):7d}  ({len(g2)/total:.1%})")
    print(f"  after score>= {min_score}: {len(g3):7d}  ({len(g3)/total:.1%})")
    if "mtf_ok" in df.columns:
        print(f"  after MTF            : {len(g4):7d}  ({len(g4)/total:.1%})")

    # Quick bottleneck hint
    if len(g1)==0:
        print("\nBottleneck: regime. Try easing adx_min/atr_pct_min/chop_max.")
    elif len(g2)==0:
        print("\nBottleneck: direction (trend model).")
    elif len(g3)==0:
        print("\nBottleneck: score. See which energy has lowest true-rate.")
    elif "mtf_ok" in df.columns and len(g4)==0:
        print("\nBottleneck: MTF alignment. Consider reject_on_mismatch: false for a test.")


if __name__ == "__main__":
    main()
