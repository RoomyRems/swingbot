# backtest/engine.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict
import math
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm

from risk.manager import compute_levels as rm_compute_levels, size_position as rm_size_position
from utils.config import load_config
from utils.logger import today_filename, log_dataframe
from broker.alpaca import get_daily_bars
from strategies.swing_strategy import (
    add_indicators,
    evaluate_five_energies,
    _resample_weekly_ohlcv,
    _weekly_trend_view,
    _mtf_alignment_ok,
    _regime_check,
    weekly_context,
    _nearest_pivot_levels,
)

# ---------- Helpers ----------

def _trading_calendar(dfs: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
    days = set()
    for df in dfs.values():
        days.update(df.index)
    return sorted(d for d in days)

def _gap_exit_price(side: str, o: float, level: float) -> float:
    # simple gap rule: exit at worse of open vs level
    if side == "long":
        return min(o, level)
    else:
        return max(o, level)

def _drawdown_stats(equity: pd.Series) -> tuple[float, pd.Timestamp]:
    if equity.empty:
        return 0.0, pd.NaT
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    dd_date = dd.idxmin() if len(dd) else pd.NaT
    return max_dd, dd_date

def _annualized_sharpe(daily_pct: pd.Series) -> float:
    if daily_pct.std(ddof=1) == 0 or len(daily_pct) < 2:
        return 0.0
    return float((daily_pct.mean() / daily_pct.std(ddof=1)) * math.sqrt(252))

def _week_key(ts: pd.Timestamp) -> pd.Period:
    return pd.to_datetime(ts).to_period("W-FRI")

def _qty_bp_cap(entry_px: float, cash: float, bp_multiple: float) -> int:
    if entry_px <= 0 or cash <= 0 or bp_multiple <= 0:
        return 0
    return int(math.floor((cash * bp_multiple) / entry_px))

def _detect_split(prev_close: float, today_open: float) -> int:
    """Return forward split factor (>=2) if a likely corporate split occurred, else 1.
    Heuristic: large downward gap where ratio prev_close / today_open is near an integer >= 2
    and absolute gap exceeds 60%. Handles cases like 50:1 (CMG 2024) producing huge artificial losses otherwise.
    """
    if not (np.isfinite(prev_close) and np.isfinite(today_open)) or prev_close <= 0 or today_open <= 0:
        return 1
    ratio = prev_close / today_open
    if ratio < 1.8:  # require substantial factor
        return 1
    cand = int(round(ratio))
    if cand < 2:
        return 1
    # relative diff threshold 2%
    if abs(ratio - cand) / ratio <= 0.02:
        # also ensure big absolute gap to filter ordinary moves
        gap_pct = abs(today_open - prev_close) / prev_close
        if gap_pct > 0.60:
            return cand
    return 1

def _apply_split_adjustment(pos, factor: int):
    """Adjust an open Position in-place for a forward split factor (>=2)."""
    if factor <= 1:
        return
    pos.entry_price /= factor
    pos.stop /= factor
    pos.target /= factor
    pos.per_share_risk = abs(pos.entry_price - pos.stop)
    pos.qty *= factor

@dataclass
class PendingLimit:
    symbol: str
    direction: str   # "long"/"short"
    signal_date: pd.Timestamp
    expires: pd.Timestamp
    limit_px: float
    atr: float       # <-- keep ATR so we can compute levels at fill
    score: int
    setup_type: str = ""
    ema20_dist_pct: float = float("nan")

@dataclass
class PendingMarket:
    symbol: str
    direction: str    # "long"/"short"
    fill_date: pd.Timestamp
    atr: float
    score: int
    setup_type: str = ""
    ema20_dist_pct: float = float("nan")

@dataclass
class Position:
    symbol: str
    side: str           # "long"/"short"
    entry_date: pd.Timestamp
    entry_price: float
    qty: int
    stop: float
    target: float
    per_share_risk: float
    score: int
    be_armed: bool = False  # BREAKEVEN: armed after price reaches +1R
    bars_held: int = 0
    max_favorable_R: float = 0.0
    scaled: bool = False
    # Track whether runner target ultimately achieved (set on exit for diagnostics)
    runner_win: bool = False
    # Phase0 instrumentation
    mae_R: float = 0.0   # most adverse excursion in R (positive number)
    setup_type: str = ""  # pullback or breakout
    signal_ext_pct: float = float("nan")
    signal_rvol: float = float("nan")
    pivot_dist_pct: float = float("nan")
    ema20_dist_pct: float = float("nan")
    # Cash flow tracking
    entry_cash_flow: float = 0.0   # net cash change at entry (negative for long buy)
    entry_commission: float = 0.0
    entry_slippage_cost: float = 0.0
    position_id: int = field(default=-1)

    def market_value(self, close_px: float) -> float:
        sign = 1 if self.side == "long" else -1
        return sign * self.qty * close_px

    def unrealized_pnl(self, close_px: float) -> float:
        sign = 1 if self.side == "long" else -1
        return sign * self.qty * (close_px - self.entry_price)

# ---------- Core engine ----------

def _load_symbol_df(sym: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    lookback_days = int((end_date - start_date).days) + 220
    df = get_daily_bars(sym, lookback_days=lookback_days)
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index(pd.to_datetime(df["Date"])).drop(columns=["Date"])
        else:
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    cutoff = start_date - pd.Timedelta(days=60)
    df = df[(df.index <= end_date) & (df.index >= cutoff)]
    return add_indicators(df)

def _mtf_ok_for_slice(dfslice: pd.DataFrame, cfg: dict, direction: str, weekly_ctx: dict | None = None) -> Tuple[bool, str]:
    mtf_cfg = (cfg.get("trading", {}).get("mtf") or {})
    if not bool(mtf_cfg.get("enabled", False)):
        return True, "mtf-disabled"
    try:
        if weekly_ctx is not None and all(k in weekly_ctx for k in ("wmacd","wmacds","wmacdh")):
            # Derive up/down/none from cached MACD context
            wmacd = weekly_ctx.get("wmacd"); wmacds = weekly_ctx.get("wmacds"); wmacdh = weekly_ctx.get("wmacdh")
            if all([np.isfinite(wmacd), np.isfinite(wmacds), np.isfinite(wmacdh)]):
                if (wmacd > wmacds) and (wmacdh > 0):
                    w_trend = "up"
                elif (wmacd < wmacds) and (wmacdh < 0):
                    w_trend = "down"
                else:
                    w_trend = "none"
            else:
                w_trend = "none"
        else:
            wk = _resample_weekly_ohlcv(dfslice)
            wk_ind = add_indicators(wk)
            w_trend = _weekly_trend_view(wk_ind, mtf_cfg)
        return _mtf_alignment_ok(direction, w_trend, mtf_cfg)
    except Exception:
        return True, "mtf-skip-error"

def run_backtest(
    tickers: List[str],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    cfg_path: str = "config.yaml",
) -> dict:
    cfg = load_config(cfg_path)

    bt_cfg = cfg.get("backtest", {}) or {}
    # caps
    max_open = int(bt_cfg.get("max_open_positions", bt_cfg.get("max_positions", 6)))
    max_risk_pct = float(bt_cfg.get("max_total_risk_pct", cfg.get("risk", {}).get("max_total_risk_pct", 0.08)))
    max_new_week = int(bt_cfg.get("max_new_trades_per_week", 0))  # 0 = unlimited
    bp_multiple = float(bt_cfg.get("bp_multiple", 1.0))  # simple notional cap
    equity_halt_floor = float(bt_cfg.get("halt_on_equity_at_or_below", 0.0))
    force_close_on_end = bool(bt_cfg.get("force_close_on_end", True))
    # Time-stop config
    ts_cfg = bt_cfg.get("time_stop", {}) or {}
    ts_enabled = bool(ts_cfg.get("enabled", False))
    ts_bars = int(ts_cfg.get("bars", 0))
    ts_min_R = float(ts_cfg.get("min_R", 0.0))
    # Early soft-stop config
    es_cfg = bt_cfg.get("early_stop", {}) or {}
    es_enabled = bool(es_cfg.get("enabled", False))
    es_bars = int(es_cfg.get("bars", 0))
    es_min_mfe = float(es_cfg.get("min_mfe_R", 0.0))
    es_adverse_atr = float(es_cfg.get("adverse_atr_mult", 0.0))
    # Gap fail-safe
    gap_cfg = bt_cfg.get("gap_fail_safe", {}) or {}
    gap_enabled = bool(gap_cfg.get("enabled", False))
    gap_R_multiple = float(gap_cfg.get("gap_R_multiple", 0.0))
    # Loss guard early MAE cap
    lg_cfg = bt_cfg.get("loss_guard", {}) or {}
    lg_enabled = bool(lg_cfg.get("enabled", False))
    lg_mae_R = float(lg_cfg.get("mae_R_threshold", 0.0))
    lg_bars = int(lg_cfg.get("bars", 0))

    # entry model
    if "entry_model" in bt_cfg:
        entry_model = bt_cfg["entry_model"] or {"type": "market"}
        entry_type = str(entry_model.get("type", "market")).lower()
        retrace_ref = str(entry_model.get("retrace_to", "EMA20")).upper()
        atr_frac = float(entry_model.get("atr_fraction", 0.25))
        horizon = int(entry_model.get("horizon_days", 3))
    else:
        entry_type = "market"
        retrace_ref = "EMA20"
        atr_frac = 0.25
        horizon = 3
        if str(bt_cfg.get("entry_price", "")).lower() not in {"", "next_open"}:
            entry_type = "market"

    burns_mode = bool((cfg.get("trading", {}) or {}).get("burns_mode", False))

    allow_short = bool(bt_cfg.get("allow_short", True))
    slip_bps = float(bt_cfg.get("slippage_bps", 0.0))
    commission_ps = float(bt_cfg.get("commission_per_share", 0.0))

    # BREAKEVEN
    be_R = float(bt_cfg.get("breakeven_at_R", 0.0))
    be_intrabar = str(bt_cfg.get("breakeven_intrabar", "favor_be")).lower()
    # Adaptive target scaling config (optional)
    adapt_cfg = bt_cfg.get("adaptive", {}) or {}
    adapt_enabled = bool(adapt_cfg.get("enabled", False))
    adapt_low_atr_pct = float(adapt_cfg.get("low_atr_pct", 0.012))
    adapt_primary_mult = float(adapt_cfg.get("primary_mult", 0.85))  # shrink target distance when volatility low

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # ---------- load data ----------
    data: Dict[str, pd.DataFrame] = {}
    for sym in tqdm(tickers, desc="Load+indicators", leave=False):
        try:
            df = _load_symbol_df(sym, start_date, end_date)
            if df.empty:
                continue
            data[sym] = df
        except Exception:
            continue
    if not data:
        raise RuntimeError("No data loaded for backtest.")

    calendar = [d for d in _trading_calendar(data) if (start_date <= d <= end_date)]
    calendar = sorted(calendar)
    if not calendar:
        raise RuntimeError("No trading days in range.")

    # Precompute daily candidates by date
    candidates_by_day: Dict[pd.Timestamp, List[dict]] = {d: [] for d in calendar}
    evaluations: List[dict] = []
    # tiny cache: (sym, i_idx) -> weekly_ctx
    _wcache: Dict[Tuple[str, int], dict] = {}
    for sym, df in tqdm(data.items(), desc="Scan signals", leave=False):
        mask = (df.index >= start_date) & (df.index <= end_date)
        # Fast path references
        idx_all = df.index
        for d in df.index[mask]:
            i = idx_all.get_loc(d)
            if i < 60:
                continue
            sl = df.iloc[: i + 1]
            row = sl.iloc[-1]
            # Precompute weekly context once for this (sym, day) using cache
            key = (sym, i)
            wctx = _wcache.get(key)
            if wctx is None:
                wctx = weekly_context(sl)
                _wcache[key] = wctx

            # Regime
            regime_ok, _ = _regime_check(sl, cfg)

            # Energies/components (pure Burns)
            eng = evaluate_five_energies(sl, cfg, weekly_ctx=wctx) or {}
            direction = eng.get("direction")
            score = int(eng.get("score", 0))
            trend_ok = bool(eng.get("trend", False))
            mom_ok   = bool(eng.get("momentum", False))
            cycle_ok = bool(eng.get("cycle", False))
            sr_ok    = bool(eng.get("sr", False))
            scale_ok = bool(eng.get("scale", False))
            vol_ok   = bool(eng.get("volume", False))  # confirmation only

            # MTF bonus
            if direction in ("long", "short"):
                mtf_ok, _mtf_reason = _mtf_ok_for_slice(sl, cfg, direction, weekly_ctx=wctx)
            else:
                mtf_ok, _mtf_reason = False, "no-dir"
            mtf_cfg = (cfg.get("trading", {}).get("mtf") or {})
            mtf_counts = bool(mtf_cfg.get("counts_as_energy", True))
            core_min = int(cfg.get("trading", {}).get("min_core_energies", cfg.get("trading", {}).get("min_score", 4)))
            # enforce core gate before bonus
            score_eff = score + (1 if (mtf_counts and mtf_ok) else 0)
            # Burns mode: temporarily lower min_score to core_min for fill generation if configured
            if bool((cfg.get("trading", {}) or {}).get("burns_mode", False)):
                min_score = max(core_min, int(cfg.get("trading", {}).get("min_score", core_min)))
            else:
                min_score = int(cfg["trading"]["min_score"])  # effective threshold (post bonus)
            allowed_dir = (direction != "short") or allow_short
            mtf_cfg = (cfg.get("trading", {}).get("mtf") or {})
            mtf_veto = bool(mtf_cfg.get("reject_on_mismatch", False))
            burns_mode_active = bool((cfg.get("trading", {}) or {}).get("burns_mode", False))
            # Extract value_zone flag from Burns evaluation explain (if available)
            value_zone = False
            try:
                value_zone = bool(eng.get("explain", {}).get("sr", {}).get("value_zone", False))
            except Exception:
                value_zone = False
            if burns_mode_active:
                # Burns canonical: 5 energies are Trend, Momentum, Cycle, S/R, Scale (weekly momentum).
                # Accept when at least core_min of these are true (default 4). Breakouts must also have momentum.
                setup_type = eng.get("setup_type", "")
                core_count = int(eng.get("core_pass_count", score))
                # Basic marketability filters
                min_atr_pct = float(cfg.get("trading", {}).get("min_atr_pct", 0.008))
                min_price = float(cfg.get("trading", {}).get("min_price", 5.0))
                atr_pct = float(row.get("ATR_PCT", np.nan))
                price_ok = float(row.get("Close", 0.0)) >= min_price
                atr_ok = np.isfinite(atr_pct) and atr_pct >= min_atr_pct
                # Min R expectancy guard (use ATR-based as a quick proxy; final stop may use pivots at fill)
                reward_mult = float(cfg.get("risk", {}).get("reward_multiple", 2.0))
                stop_tmp, target_tmp = rm_compute_levels(direction, float(row.get("Close", np.nan)), float(row.get("ATR14", np.nan)), float(cfg["risk"]["atr_multiple_stop"]), reward_mult)
                if stop_tmp is not None and target_tmp is not None and abs(float(row.get("Close", np.nan)) - stop_tmp) > 0:
                    R = abs(target_tmp - float(row.get("Close", np.nan))) / abs(float(row.get("Close", np.nan)) - stop_tmp)
                    min_R_ok = R >= max(1.2, reward_mult * 0.8)
                else:
                    min_R_ok = False
                if not regime_ok:
                    reason = "regime"
                    accept = False
                elif direction not in ("long", "short"):
                    reason = "no_dir"
                    accept = False
                elif not allowed_dir:
                    reason = "short_blocked"
                    accept = False
                elif core_count < core_min:
                    reason = "core_min"
                    accept = False
                elif setup_type == "breakout" and not mom_ok:
                    reason = "breakout_no_mom"
                    accept = False
                elif not price_ok:
                    reason = "price"
                    accept = False
                elif not atr_ok:
                    reason = "atr"
                    accept = False
                elif not min_R_ok:
                    reason = "minR"
                    accept = False
                else:
                    reason = "pass"
                    accept = True
            else:
                # Non-Burns mode still uses Burns evaluator and same acceptance criteria
                setup_type = eng.get("setup_type", "")
                core_count = int(eng.get("core_pass_count", score))
                min_atr_pct = float(cfg.get("trading", {}).get("min_atr_pct", 0.008))
                min_price = float(cfg.get("trading", {}).get("min_price", 5.0))
                atr_pct = float(row.get("ATR_PCT", np.nan))
                price_ok = float(row.get("Close", 0.0)) >= min_price
                atr_ok = np.isfinite(atr_pct) and atr_pct >= min_atr_pct
                reward_mult = float(cfg.get("risk", {}).get("reward_multiple", 2.0))
                stop_tmp, target_tmp = rm_compute_levels(direction, float(row.get("Close", np.nan)), float(row.get("ATR14", np.nan)), float(cfg["risk"]["atr_multiple_stop"]), reward_mult)
                if stop_tmp is not None and target_tmp is not None and abs(float(row.get("Close", np.nan)) - stop_tmp) > 0:
                    R = abs(target_tmp - float(row.get("Close", np.nan))) / abs(float(row.get("Close", np.nan)) - stop_tmp)
                    min_R_ok = R >= max(1.2, reward_mult * 0.8)
                else:
                    min_R_ok = False
                if not regime_ok:
                    reason = "regime"; accept = False
                elif direction not in ("long","short"):
                    reason = "no_dir"; accept = False
                elif not allowed_dir:
                    reason = "short_blocked"; accept = False
                elif core_count < core_min:
                    reason = "core_min"; accept = False
                elif setup_type == "breakout" and not mom_ok:
                    reason = "breakout_no_mom"; accept = False
                elif not price_ok:
                    reason = "price"; accept = False
                elif not atr_ok:
                    reason = "atr"; accept = False
                elif not min_R_ok:
                    reason = "minR"; accept = False
                else:
                    reason = "pass"; accept = True

            # Evaluation row
            # Compute setup classification heuristic: pullback if Close within 1 ATR of EMA20 and not extended; else breakout
            try:
                ema20_val = float(row.get("EMA20", np.nan))
                close_val = float(row.get("Close", np.nan))
                atr_val = float(row.get("ATR14", np.nan))
                ext_pct = float(eng.get("ext_pct", np.nan)) if eng else np.nan
                if np.isfinite(ema20_val) and np.isfinite(close_val) and np.isfinite(atr_val) and atr_val > 0:
                    pullback_cond = abs(close_val - ema20_val) <= atr_val and (abs(ext_pct) <= 0.035 if np.isfinite(ext_pct) else True)
                else:
                    pullback_cond = False
                setup_type = "pullback" if pullback_cond else "breakout"
                ema20_dist_pct = ((close_val - ema20_val)/ema20_val) if np.isfinite(ema20_val) and ema20_val>0 and np.isfinite(close_val) else np.nan
            except Exception:
                setup_type = ""
                ema20_dist_pct = np.nan
            evaluations.append({
                "date": d,
                "symbol": sym,
                "direction": direction or "",
                "score": score,
                "score_eff": int(score_eff),
                "mtf_bonus": int(1 if (mtf_counts and mtf_ok) else 0),
                "trend": trend_ok,
                "momentum": mom_ok,
                "cycle": cycle_ok,
                "sr": sr_ok,
                "scale": scale_ok,
                "volume_confirm": vol_ok,
                "regime_ok": bool(regime_ok),
                "mtf_ok": bool(mtf_ok),
                "reject_reason": reason,
                "close": float(row.get("Close", np.nan)),
                "ema20": float(row.get("EMA20", np.nan)),
                    "atr14": float(row.get("ATR14", np.nan)),
                    "adx14": float(row.get("ADX14", np.nan)) if "ADX14" in row.index else np.nan,
                    "atr_percent": float(row.get("ATR_PCT", np.nan)) if "ATR_PCT" in row.index else np.nan,
                    "setup_type": setup_type,
                    "ema20_dist_pct": ema20_dist_pct,
            })
            if accept:
                # Time-stop configuration (placeholder for future use)
                candidates_by_day[d].append({
                    "symbol": sym,
                    "date": d,
                    "direction": direction,
                    "score": int(score),
                    "score_eff": int(score_eff),
                    "close": float(row["Close"]),
                    "ema20": float(row.get("EMA20", np.nan)),
                    "atr": float(row.get("ATR14", np.nan)),
                    "adx": float(row.get("ADX14", np.nan)),
                    "atr_percent": float(row.get("ATR_PCT", np.nan)),
                    "setup_type": setup_type,
                    "ema20_dist_pct": ema20_dist_pct,
                })

    # --- DIAGNOSTIC ---
    total_sigs = sum(len(v) for v in candidates_by_day.values())
    uniq_syms = len({s["symbol"] for v in candidates_by_day.values() for s in v})
    from tqdm import tqdm as _tqdm
    _tqdm.write(f"Found {total_sigs} daily signal candidates across the test window from {uniq_syms} symbols.")

    # ---------- portfolio state ----------
    start_equity_cfg = float(bt_cfg.get("initial_equity", cfg.get("risk", {}).get("account_equity", 50000)))
    equity = float(start_equity_cfg)
    cash = equity  # free cash; equity = cash + MV(open positions)
    open_positions: Dict[str, Position] = {}
    pending_limits: List[PendingLimit] = []
    pending_markets: List[PendingMarket] = []
    position_id_counter = 0
    entries_audit: List[dict] = []
    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    trades: List[dict] = []
    week_new_count: defaultdict[pd.Period, int] = defaultdict(int)

    def _current_total_risk() -> float:
        if not open_positions:
            return 0.0
        # Use a tiny floor in denominator to avoid divide-by-zero; treat <=0 equity as maxed risk.
        eq = max(equity, 1e-9)
        risk_amt = sum(p.per_share_risk * p.qty for p in open_positions.values())
        return float(risk_amt / eq)

    # ---------- backtest loop ----------
    min_stop_pct_cfg = float(cfg.get("risk", {}).get("min_stop_pct", 0.0))

    def _enforce_floor(direction: str, entry_px: float, stop: float, target: float) -> tuple[float, float]:
        if min_stop_pct_cfg <= 0 or entry_px <= 0:
            return stop, target
        dist = abs(entry_px - stop)
        floor_dist = entry_px * min_stop_pct_cfg
        if dist >= floor_dist:
            return stop, target
        rm = float(cfg["risk"]["reward_multiple"])
        if direction == "long":
            stop = entry_px - floor_dist
            target = entry_px + rm * floor_dist
        else:
            stop = entry_px + floor_dist
            target = entry_px - rm * floor_dist
        if target <= 0:
            return stop, target  # will be rejected later
        return round(stop, 2), round(target, 2)

    for day in tqdm(calendar, desc="Backtest", leave=True):
        # 1) exits first
        for sym, pos in list(open_positions.items()):
            df = data.get(sym)
            if df is None or day not in df.index:
                continue
            o = float(df.at[day, "Open"]); h = float(df.at[day, "High"]); l = float(df.at[day, "Low"]); c = float(df.at[day, "Close"])

            # Forward split detection
            try:
                idx = df.index.get_loc(day)
                if idx > 0:
                    prev_close = float(df.iloc[idx-1]["Close"])
                    split_factor = _detect_split(prev_close, o)
                    if split_factor > 1:
                        _apply_split_adjustment(pos, split_factor)
            except Exception:
                pass

            # Breakeven arming
            if be_R > 0 and not pos.be_armed:
                if pos.side == "long":
                    if h >= pos.entry_price + be_R * pos.per_share_risk and not (be_intrabar == "favor_stop" and l <= pos.stop):
                        pos.stop = max(pos.stop, pos.entry_price); pos.be_armed = True
                else:
                    if l <= pos.entry_price - be_R * pos.per_share_risk and not (be_intrabar == "favor_stop" and h >= pos.stop):
                        pos.stop = min(pos.stop, pos.entry_price); pos.be_armed = True
            # Adaptive breakeven ratchet: if armed but pullback deep and insufficient follow-through, keep stop at BE
            if pos.be_armed and pos.per_share_risk > 0:
                # measure max favorable R so far vs current price
                if pos.side == "long":
                    curr_R = (c - pos.entry_price) / pos.per_share_risk
                else:
                    curr_R = (pos.entry_price - c) / pos.per_share_risk
                if pos.max_favorable_R < (be_R * 1.2) and curr_R < 0.2:
                    pos.stop = pos.entry_price

            # Gap fail-safe (evaluate at open before other exits if large adverse gap vs entry in R multiples)
            if gap_enabled and gap_R_multiple > 0 and pos.per_share_risk > 0:
                adverse_gap_R = 0.0
                if pos.side == "long":
                    if o < pos.entry_price:
                        adverse_gap_R = (pos.entry_price - o) / pos.per_share_risk
                else:
                    if o > pos.entry_price:
                        adverse_gap_R = (o - pos.entry_price) / pos.per_share_risk
                if adverse_gap_R >= gap_R_multiple:
                    fill_px = o; exit_reason = "gap_fail_safe"; slip=0.0
                    commission = commission_ps * pos.qty
                    cash += (pos.qty * fill_px) if pos.side == "long" else (-pos.qty * fill_px)
                    cash -= commission
                    pnl = (fill_px - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - fill_px) * pos.qty
                    trades.append({
                        "symbol": sym, "side": pos.side, "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                        "exit_date": day, "exit_price": fill_px, "qty": pos.qty, "pnl": pnl,
                        "r_multiple": pnl / (pos.per_share_risk * pos.qty) if pos.per_share_risk > 0 else np.nan,
                        "commission": commission, "slippage_cost": 0.0, "reason": "gap_fail_safe",
                        "part": ("runner" if pos.scaled else "full"), "bars_held": pos.bars_held, "mfe_R": pos.max_favorable_R,
                    })
                    del open_positions[sym]
                    continue

            exit_reason = None; fill_px = None; target_hit = False
            if pos.side == "long":
                if l <= pos.stop:
                    fill_px = _gap_exit_price("long", o, pos.stop); exit_reason = "stop"
                elif h >= pos.target:
                    target_hit = True
            else:
                if h >= pos.stop:
                    fill_px = _gap_exit_price("short", o, pos.stop); exit_reason = "stop"
                elif l <= pos.target:
                    target_hit = True

            # Scaling logic
            if target_hit and exit_reason is None:
                tgt_cfg = (cfg.get("trading", {}) or {}).get("targets", {}) or {}
                if bool(tgt_cfg.get("enabled", False)):
                    primary_R = float(tgt_cfg.get("primary_R", float(cfg.get("risk", {}).get("reward_multiple", 1.5))))
                    runner_R = float(tgt_cfg.get("runner_R", 0.0))
                    scale_pct = float(tgt_cfg.get("scale_out_pct", 0.0))
                    adx_min_runner = float(tgt_cfg.get("adx_min_for_runner", 0.0))
                    if (not pos.scaled) and (0 < scale_pct < 1.0) and runner_R > primary_R and pos.per_share_risk > 0:
                        qty_scale = int(round(pos.qty * scale_pct))
                        if 0 < qty_scale < pos.qty:
                            fill_px_primary = pos.target if ((pos.side == "long" and o <= pos.target) or (pos.side == "short" and o >= pos.target)) else o
                            slip_p = (slip_bps/10000.0)*fill_px_primary
                            eff_primary = fill_px_primary - slip_p if pos.side == "long" else fill_px_primary + slip_p
                            commission_primary = commission_ps * qty_scale
                            cash += (qty_scale * eff_primary) if pos.side == "long" else (-qty_scale * eff_primary)
                            cash -= commission_primary
                            pnl_primary = (eff_primary - pos.entry_price) * qty_scale if pos.side == "long" else (pos.entry_price - eff_primary) * qty_scale
                            trades.append({
                                "symbol": sym, "side": pos.side, "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                                "exit_date": day, "exit_price": eff_primary, "qty": qty_scale, "pnl": pnl_primary,
                                "r_multiple": pnl_primary / (pos.per_share_risk * qty_scale) if pos.per_share_risk > 0 else np.nan,
                                "commission": commission_primary, "slippage_cost": slip_p * qty_scale, "reason": "scale_out",
                                "part": "scale_out", "bars_held": pos.bars_held, "mfe_R": pos.max_favorable_R,
                            })
                            pos.qty -= qty_scale; pos.scaled = True
                            try:
                                adx_val_now = float(df.at[day, "ADX14"]) if "ADX14" in df.columns else np.nan
                            except Exception:
                                adx_val_now = np.nan
                            if np.isfinite(adx_val_now) and adx_val_now >= adx_min_runner:
                                if pos.side == "long":
                                    pos.target = pos.entry_price + runner_R * pos.per_share_risk
                                else:
                                    pos.target = pos.entry_price - runner_R * pos.per_share_risk
                            else:
                                fill_px = fill_px_primary; exit_reason = "target"
                        else:
                            fill_px = pos.target if ((pos.side == "long" and o <= pos.target) or (pos.side == "short" and o >= pos.target)) else o; exit_reason = "target"
                    else:
                        fill_px = pos.target if ((pos.side == "long" and o <= pos.target) or (pos.side == "short" and o >= pos.target)) else o; exit_reason = "target"
                else:
                    # scaling disabled: treat as full target
                    fill_px = pos.target if ((pos.side == "long" and o <= pos.target) or (pos.side == "short" and o >= pos.target)) else o; exit_reason = "target"

            # Runner stays open – update metrics only
            if exit_reason is None and target_hit and pos.scaled and pos.qty > 0:
                pos.bars_held += 1
                if pos.per_share_risk > 0:
                    fav = (h - pos.entry_price) / pos.per_share_risk if pos.side == "long" else (pos.entry_price - l) / pos.per_share_risk
                    pos.max_favorable_R = max(pos.max_favorable_R, fav)
                continue

            # Time-stop tracking
            if exit_reason is None:
                pos.bars_held += 1
                if pos.per_share_risk > 0:
                    fav = (h - pos.entry_price) / pos.per_share_risk if pos.side == "long" else (pos.entry_price - l) / pos.per_share_risk
                    pos.max_favorable_R = max(pos.max_favorable_R, fav)
                    # update MAE_R (adverse excursion)
                    adverse = (pos.entry_price - l)/pos.per_share_risk if pos.side=="long" else (h - pos.entry_price)/pos.per_share_risk
                    if adverse > pos.mae_R:
                        pos.mae_R = adverse
                # Loss guard: exit early if large adverse excursion in initial bars
                if (exit_reason is None and lg_enabled and lg_mae_R>0 and lg_bars>0 and pos.bars_held <= lg_bars and pos.mae_R >= lg_mae_R):
                    c_px = c
                    fill_px = c_px; exit_reason = "loss_guard"
                # Early soft stop check BEFORE time-stop
                if (exit_reason is None and es_enabled and es_bars > 0 and es_min_mfe > 0 and es_adverse_atr > 0 and pos.bars_held >= es_bars):
                    # compute adverse move from entry close vs current close relative to ATR at entry day (approx via df)
                    df_sym = data.get(sym)
                    if df_sym is not None and day in df_sym.index:
                        c_px = float(df_sym.at[day, "Close"])
                        atr_today = float(df_sym.at[day, "ATR14"]) if "ATR14" in df_sym.columns else np.nan
                        if np.isfinite(atr_today) and pos.per_share_risk > 0:
                            adverse_move = (pos.entry_price - c_px) if pos.side == "long" else (c_px - pos.entry_price)
                            if pos.max_favorable_R < es_min_mfe and adverse_move >= es_adverse_atr * atr_today:
                                fill_px = c_px; exit_reason = "early_soft_stop"
                # Time-stop after early soft-stop opportunity
                if ts_enabled and ts_bars > 0 and ts_min_R > 0 and pos.bars_held >= ts_bars and pos.max_favorable_R < ts_min_R:
                    fill_px = c; exit_reason = "time_stop"

            if exit_reason:
                slip = (slip_bps/10000.0)*fill_px
                fill_eff = fill_px - slip if pos.side == "long" else fill_px + slip
                commission = commission_ps * pos.qty
                cash += (pos.qty * fill_eff) if pos.side == "long" else (-pos.qty * fill_eff)
                cash -= commission
                pnl = (fill_eff - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - fill_eff) * pos.qty
                trades.append({
                    "symbol": sym, "side": pos.side, "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                    "exit_date": day, "exit_price": fill_eff, "qty": pos.qty, "pnl": pnl,
                    "r_multiple": pnl / (pos.per_share_risk * pos.qty) if pos.per_share_risk > 0 else np.nan,
                    "commission": commission, "slippage_cost": slip * pos.qty, "reason": exit_reason,
                    "part": ("runner" if pos.scaled else "full"), "bars_held": pos.bars_held, "mfe_R": pos.max_favorable_R,
                    "atr_pct_entry": getattr(pos, "atr_pct_entry", np.nan),
                    "target_R_at_entry": getattr(pos, "target_R_at_entry", np.nan),
                    "adaptive_shrunk": getattr(pos, "_adaptive_shrunk", False),
                    "mae_R": pos.mae_R,
                    "setup_type": pos.setup_type,
                    "signal_ext_pct": pos.signal_ext_pct,
                    "signal_rvol": pos.signal_rvol,
                    "pivot_dist_pct": pos.pivot_dist_pct,
                    "ema20_dist_pct": pos.ema20_dist_pct,
                    "position_id": pos.position_id,
                })
                del open_positions[sym]
                continue

        # 1a) pending market fills
        for pend in list(pending_markets):
            if pend.fill_date != day:
                continue
            df = data.get(pend.symbol)
            if df is None or day not in df.index:
                pending_markets.remove(pend)
                continue
            if len(open_positions) >= max_open or (_current_total_risk() >= max_risk_pct):
                pending_markets.remove(pend)
                continue
            wk = _week_key(day)
            if max_new_week > 0 and week_new_count[wk] >= max_new_week:
                pending_markets.remove(pend)
                continue
            o = float(df.at[day, "Open"])
            slip = (slip_bps / 10000.0) * o
            entry_px = o + slip if pend.direction == "long" else o - slip
            # Compute stop/target: pivot-based in Burns mode, else ATR-based
            if burns_mode:
                try:
                    scfg = (cfg.get("signals", {}) or {}).get("sr_pivots", {}) or {}
                    lookback = int(scfg.get("lookback", 60))
                    left = int(scfg.get("left", 3))
                    right = int(scfg.get("right", 3))
                    # Use history up to the prior bar (today's open not yet a full bar)
                    try:
                        idx_fill = df.index.get_loc(day)
                        df_past = df.iloc[:max(1, idx_fill)]
                    except Exception:
                        df_past = df
                    support, resistance = _nearest_pivot_levels(df_past, lookback, left, right, price=entry_px)
                    pad_atr_mult = float(cfg.get("risk", {}).get("pad_atr_for_pivot_stop", 0.25))
                    min_stop_pct = float(cfg.get("risk", {}).get("min_stop_pct", 0.0))
                    pad = 0.0
                    if np.isfinite(pend.atr) and pend.atr > 0:
                        pad = max(pad, pad_atr_mult * pend.atr)
                    if entry_px > 0 and min_stop_pct > 0:
                        pad = max(pad, min_stop_pct * entry_px)
                    reward_mult = float(cfg.get("risk", {}).get("reward_multiple", 2.0))
                    stop = target = None
                    if pend.direction == "long" and support is not None and support < entry_px:
                        stop = round(max(0.01, support - pad), 2)
                        per_r = entry_px - stop
                        if per_r > 0:
                            target = round(entry_px + reward_mult * per_r, 2)
                    elif pend.direction == "short" and resistance is not None and resistance > entry_px:
                        stop = round(resistance + pad, 2)
                        per_r = stop - entry_px
                        if per_r > 0:
                            target = round(entry_px - reward_mult * per_r, 2)
                    # Fallback to ATR if pivot levels invalid
                    if stop is None or target is None:
                        stop, target = rm_compute_levels(pend.direction, entry_px, pend.atr, float(cfg["risk"]["atr_multiple_stop"]), float(cfg["risk"]["reward_multiple"]))
                except Exception:
                    stop, target = rm_compute_levels(pend.direction, entry_px, pend.atr, float(cfg["risk"]["atr_multiple_stop"]), float(cfg["risk"]["reward_multiple"]))
            else:
                stop, target = rm_compute_levels(pend.direction, entry_px, pend.atr, float(cfg["risk"]["atr_multiple_stop"]), float(cfg["risk"]["reward_multiple"]))
            if stop is None or target is None:
                pending_markets.remove(pend)
                continue
            stop, target = _enforce_floor(pend.direction, entry_px, stop, target)
            if target <= 0:
                pending_markets.remove(pend)
                continue
            cfg_day = copy.deepcopy(cfg)
            cfg_day["risk"]["use_broker_equity"] = False
            cfg_day["risk"]["account_equity"] = float(equity)
            qty = rm_size_position(cfg_day, entry_px, stop)
            qty = min(qty, _qty_bp_cap(entry_px, cash, bp_multiple))
            if qty <= 0:
                pending_markets.remove(pend)
                continue
            psr = abs(entry_px - stop)
            new_total = (_current_total_risk() * max(equity, 1e-9) + psr * qty) / max(equity, 1e-9)
            if new_total > max_risk_pct:
                pending_markets.remove(pend)
                continue
            commission = commission_ps * qty
            cash += (-qty * entry_px) if pend.direction == "long" else (qty * entry_px)
            cash -= commission
            position_id_counter += 1
            open_positions[pend.symbol] = Position(pend.symbol, pend.direction, day, entry_px, qty, stop, target, psr, pend.score, False, position_id=position_id_counter)
            # annotate adaptive diagnostics
            pos_created = open_positions[pend.symbol]
            pos_created.atr_pct_entry = float(pend.atr / entry_px) if entry_px > 0 else np.nan
            pos_created.target_R_at_entry = (abs(pos_created.target - pos_created.entry_price) / pos_created.per_share_risk) if pos_created.per_share_risk > 0 else np.nan
            # Phase0: placeholder setup metadata (market entries treat as 'breakout')
            pos_created.setup_type = getattr(pend, "setup_type", "breakout")
            entries_audit.append({
                "position_id": pos_created.position_id,
                "symbol": pend.symbol,
                "entry_date": day,
                "entry_price": entry_px,
                "qty": qty,
                "entry_commission": commission,
                "entry_cash_flow": ((-qty * entry_px) if pend.direction == "long" else (qty * entry_px)) - commission,
            })
            # same day exit check
            h = float(df.at[day, "High"])
            l = float(df.at[day, "Low"])
            pos = open_positions[pend.symbol]
            exit_reason = None
            fill_px = None
            if pos.side == "long":
                if l <= pos.stop:
                    fill_px = _gap_exit_price("long", o, pos.stop)
                    exit_reason = "stop_same_day"
                elif h >= pos.target:
                    fill_px = pos.target if o <= pos.target else o
                    exit_reason = "target_same_day"
            else:
                if h >= pos.stop:
                    fill_px = _gap_exit_price("short", o, pos.stop)
                    exit_reason = "stop_same_day"
                elif l <= pos.target:
                    fill_px = pos.target if o >= pos.target else o
                    exit_reason = "target_same_day"
            if exit_reason:
                slip2 = (slip_bps / 10000.0) * fill_px
                eff = fill_px - slip2 if pos.side == "long" else fill_px + slip2
                commission = commission_ps * pos.qty
                cash += (pos.qty * eff) if pos.side == "long" else (-pos.qty * eff)
                cash -= commission
                pnl = (eff - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - eff) * pos.qty
                trades.append({
                    "symbol": pend.symbol,
                    "side": pos.side,
                    "entry_date": pos.entry_date,
                    "entry_price": pos.entry_price,
                    "exit_date": day,
                    "exit_price": eff,
                    "qty": pos.qty,
                    "pnl": pnl,
                    "r_multiple": pnl / (pos.per_share_risk * pos.qty) if pos.per_share_risk > 0 else np.nan,
                    "commission": commission,
                    "slippage_cost": slip2 * pos.qty,
                    "reason": exit_reason,
                    "part": "full",
                    "bars_held": pos.bars_held,
                    "mfe_R": pos.max_favorable_R,
                    "atr_pct_entry": float(pend.atr / pos.entry_price) if pos.entry_price > 0 else np.nan,
                    "target_R_at_entry": (abs(pos.target - pos.entry_price) / pos.per_share_risk) if pos.per_share_risk > 0 else np.nan,
                    "adaptive_shrunk": False,
                    "position_id": pos.position_id,
                })
                del open_positions[pend.symbol]
            else:
                # Apply adaptive target adjustment after fill if enabled
                if adapt_enabled and pos.per_share_risk > 0:
                    # approximate ATR% at entry using pend.atr / entry price
                    atr_pct = pend.atr / pos.entry_price if pos.entry_price > 0 else 0.0
                    if atr_pct <= adapt_low_atr_pct:
                        shrink_mult = adapt_primary_mult
                        # shrink only if current target distance exceeds multiple
                        dist_R = (abs(pos.target - pos.entry_price) / pos.per_share_risk) if pos.per_share_risk > 0 else 0.0
                        if dist_R > 0:
                            new_dist_R = dist_R * shrink_mult
                            if pos.side == "long":
                                pos.target = pos.entry_price + new_dist_R * pos.per_share_risk
                            else:
                                pos.target = pos.entry_price - new_dist_R * pos.per_share_risk
                            pos._adaptive_shrunk = True
                        else:
                            pos._adaptive_shrunk = False
                    else:
                        pos._adaptive_shrunk = False
                else:
                    pos._adaptive_shrunk = False
            week_new_count[wk] += 1
            pending_markets.remove(pend)

        # 2) pending limit fills
        for pend in list(pending_limits):
            if equity <= equity_halt_floor:
                pending_limits.clear()
                break
            df = data.get(pend.symbol)
            if df is None or day not in df.index:
                continue
            if day < pend.signal_date or day > pend.expires:
                if day > pend.expires:
                    pending_limits.remove(pend)
                continue
            o = float(df.at[day, "Open"])
            h = float(df.at[day, "High"])
            l = float(df.at[day, "Low"])
            hit = (l <= pend.limit_px) if pend.direction == "long" else (h >= pend.limit_px)
            if not hit:
                if day == pend.expires:
                    pending_limits.remove(pend)
                continue
            if len(open_positions) >= max_open or (_current_total_risk() >= max_risk_pct):
                continue
            wk = _week_key(day)
            if max_new_week > 0 and week_new_count[wk] >= max_new_week:
                continue
            cfg_day = copy.deepcopy(cfg)
            cfg_day["risk"]["use_broker_equity"] = False
            cfg_day["risk"]["account_equity"] = float(equity)
            cfg_day = copy.deepcopy(cfg)
            cfg_day["risk"]["use_broker_equity"] = False
            cfg_day["risk"]["account_equity"] = float(cash)
            raw_entry = min(o, pend.limit_px) if pend.direction == "long" else max(o, pend.limit_px)
            slip = (slip_bps / 10000.0) * raw_entry
            entry_px = raw_entry + slip if pend.direction == "long" else raw_entry - slip
            # Compute stop/target: pivot-based in Burns mode, else ATR-based
            if burns_mode:
                try:
                    scfg = (cfg.get("signals", {}) or {}).get("sr_pivots", {}) or {}
                    lookback = int(scfg.get("lookback", 60))
                    left = int(scfg.get("left", 3))
                    right = int(scfg.get("right", 3))
                    # Use history up to the prior bar (today's bar not complete at hit time)
                    try:
                        idx_fill = df.index.get_loc(day)
                        df_past = df.iloc[:max(1, idx_fill)]
                    except Exception:
                        df_past = df
                    support, resistance = _nearest_pivot_levels(df_past, lookback, left, right, price=entry_px)
                    pad_atr_mult = float(cfg.get("risk", {}).get("pad_atr_for_pivot_stop", 0.25))
                    min_stop_pct = float(cfg.get("risk", {}).get("min_stop_pct", 0.0))
                    pad = 0.0
                    if np.isfinite(pend.atr) and pend.atr > 0:
                        pad = max(pad, pad_atr_mult * pend.atr)
                    if entry_px > 0 and min_stop_pct > 0:
                        pad = max(pad, min_stop_pct * entry_px)
                    reward_mult = float(cfg.get("risk", {}).get("reward_multiple", 2.0))
                    stop = target = None
                    if pend.direction == "long" and support is not None and support < entry_px:
                        stop = round(max(0.01, support - pad), 2)
                        per_r = entry_px - stop
                        if per_r > 0:
                            target = round(entry_px + reward_mult * per_r, 2)
                    elif pend.direction == "short" and resistance is not None and resistance > entry_px:
                        stop = round(resistance + pad, 2)
                        per_r = stop - entry_px
                        if per_r > 0:
                            target = round(entry_px - reward_mult * per_r, 2)
                    if stop is None or target is None:
                        stop, target = rm_compute_levels(pend.direction, entry_px, pend.atr, float(cfg["risk"]["atr_multiple_stop"]), float(cfg["risk"]["reward_multiple"]))
                except Exception:
                    stop, target = rm_compute_levels(pend.direction, entry_px, pend.atr, float(cfg["risk"]["atr_multiple_stop"]), float(cfg["risk"]["reward_multiple"]))
            else:
                stop, target = rm_compute_levels(pend.direction, entry_px, pend.atr, float(cfg["risk"]["atr_multiple_stop"]), float(cfg["risk"]["reward_multiple"]))
            if stop is None or target is None:
                continue
            stop, target = _enforce_floor(pend.direction, entry_px, stop, target)
            if target <= 0:
                continue
            qty = rm_size_position(cfg_day, entry_px, stop)
            qty = min(qty, _qty_bp_cap(entry_px, equity, bp_multiple))
            qty = rm_size_position(cfg_day, entry_px, stop)
            qty = min(qty, _qty_bp_cap(entry_px, cash, bp_multiple))
            if qty <= 0:
                continue
            psr = abs(entry_px - stop)
            new_total = (_current_total_risk() * max(equity, 1e-9) + psr * qty) / max(equity, 1e-9)
            if new_total > max_risk_pct:
                continue
            commission = commission_ps * qty
            cash += (-qty * entry_px) if pend.direction == "long" else (qty * entry_px)
            cash -= commission
            position_id_counter += 1
            open_positions[pend.symbol] = Position(pend.symbol, pend.direction, day, entry_px, qty, stop, target, psr, pend.score, False, position_id=position_id_counter)
            entries_audit.append({
                "position_id": position_id_counter,
                "symbol": pend.symbol,
                "entry_date": day,
                "entry_price": entry_px,
                "qty": qty,
                "entry_commission": commission,
                "entry_cash_flow": ((-qty * entry_px) if pend.direction == "long" else (qty * entry_px)) - commission,
            })
            week_new_count[wk] += 1
            pending_limits.remove(pend)
            pos_created = open_positions[pend.symbol]
            pos_created.atr_pct_entry = float(pend.atr / entry_px) if entry_px > 0 else np.nan
            pos_created.target_R_at_entry = (abs(pos_created.target - pos_created.entry_price) / pos_created.per_share_risk) if pos_created.per_share_risk > 0 else np.nan
            # Limit retrace assumed pullback setup
            pos_created.setup_type = getattr(pend, "setup_type", "pullback")
            pos = open_positions[pend.symbol]
            exit_reason = None
            fill_px = None
            if pos.side == "long":
                if l <= pos.stop:
                    fill_px = _gap_exit_price("long", o, pos.stop)
                    exit_reason = "stop_same_day"
                elif h >= pos.target:
                    fill_px = pos.target if o <= pos.target else o
                    exit_reason = "target_same_day"
            else:
                if h >= pos.stop:
                    fill_px = _gap_exit_price("short", o, pos.stop)
                    exit_reason = "stop_same_day"
                elif l <= pos.target:
                    fill_px = pos.target if o >= pos.target else o
                    exit_reason = "target_same_day"
            if exit_reason:
                slip2 = (slip_bps / 10000.0) * fill_px
                eff = fill_px - slip2 if pos.side == "long" else fill_px + slip2
                commission = commission_ps * pos.qty
                cash += (pos.qty * eff) if pos.side == "long" else (-pos.qty * eff)
                cash -= commission
                pnl = (eff - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - eff) * pos.qty
                trades.append({
                    "symbol": pend.symbol,
                    "side": pos.side,
                    "entry_date": pos.entry_date,
                    "entry_price": pos.entry_price,
                    "exit_date": day,
                    "exit_price": eff,
                    "qty": pos.qty,
                    "pnl": pnl,
                    "r_multiple": pnl / (pos.per_share_risk * pos.qty) if pos.per_share_risk > 0 else np.nan,
                    "commission": commission,
                    "slippage_cost": slip2 * pos.qty,
                    "reason": exit_reason,
                    "part": "full",
                    "bars_held": pos.bars_held,
                    "mfe_R": pos.max_favorable_R,
                    "atr_pct_entry": float(pend.atr / pos.entry_price) if pos.entry_price > 0 else np.nan,
                    "target_R_at_entry": (abs(pos.target - pos.entry_price) / pos.per_share_risk) if pos.per_share_risk > 0 else np.nan,
                    "adaptive_shrunk": False,
                    "position_id": pos.position_id,
                })
                del open_positions[pend.symbol]
            else:
                # Apply adaptive target adjustment for limit fills if enabled
                if adapt_enabled and pos.per_share_risk > 0:
                    atr_pct = pend.atr / pos.entry_price if pos.entry_price > 0 else 0.0
                    if atr_pct <= adapt_low_atr_pct:
                        shrink_mult = adapt_primary_mult
                        dist_R = (abs(pos.target - pos.entry_price) / pos.per_share_risk) if pos.per_share_risk > 0 else 0.0
                        if dist_R > 0:
                            new_dist_R = dist_R * shrink_mult
                            if pos.side == "long":
                                pos.target = pos.entry_price + new_dist_R * pos.per_share_risk
                            else:
                                pos.target = pos.entry_price - new_dist_R * pos.per_share_risk
                            pos._adaptive_shrunk = True
                        else:
                            pos._adaptive_shrunk = False
                    else:
                        pos._adaptive_shrunk = False
                else:
                    pos._adaptive_shrunk = False

        # 3) new signals (queued for future fills)
        candidates_today = sorted(
            candidates_by_day.get(day, []),
            key=lambda s: (-s.get("score_eff", s["score"]), -s.get("adx", 0.0), -s.get("atr", 0.0), -s.get("close", 0.0)),
        )
        for sig in candidates_today:
            if equity <= equity_halt_floor:
                break
            if sig["symbol"] in open_positions:
                continue
            if len(open_positions) >= max_open or (_current_total_risk() >= max_risk_pct):
                break
            df = data[sig["symbol"]]
            if entry_type == "market":
                idx = df.index.get_indexer([sig["date"]])[0]
                if idx == -1 or idx + 1 >= len(df.index):
                    continue
                nd = df.index[idx + 1]
                wk_nd = _week_key(nd)
                if max_new_week > 0 and week_new_count[wk_nd] >= max_new_week:
                    continue
                pending_markets.append(
                    PendingMarket(
                        sig["symbol"],
                        sig["direction"],
                        nd,
                        float(sig["atr"]),
                        int(sig.get("score_eff", sig["score"])) ,
                        setup_type=sig.get("setup_type", ""),
                        ema20_dist_pct=float(sig.get("ema20_dist_pct", np.nan)),
                    )
                )
            elif entry_type == "limit_retrace":
                ema = sig["ema20"] if retrace_ref == "EMA20" else sig["close"]
                limit_px = (ema - atr_frac * sig["atr"]) if sig["direction"] == "long" else (ema + atr_frac * sig["atr"])
                stop, target = rm_compute_levels(
                    sig["direction"], float(limit_px), sig["atr"], float(cfg["risk"]["atr_multiple_stop"]), float(cfg["risk"]["reward_multiple"])
                )
                if stop is None or target is None:
                    continue
                stop, target = _enforce_floor(sig["direction"], limit_px, stop, target)
                if target <= 0:
                    continue
                pending_limits.append(
                    PendingLimit(
                        sig["symbol"],
                        sig["direction"],
                        sig["date"],
                        sig["date"] + pd.tseries.offsets.BDay(horizon),
                        float(limit_px),
                        float(sig["atr"]),
                        int(sig.get("score_eff", sig["score"])) ,
                        setup_type=sig.get("setup_type", ""),
                        ema20_dist_pct=float(sig.get("ema20_dist_pct", np.nan)),
                    )
                )

        # 4) mark-to-market (once per day)
        mv = 0.0
        for sym, pos in open_positions.items():
            df = data.get(sym)
            if df is None or day not in df.index:
                continue
            mv += pos.market_value(float(df.at[day, "Close"]))
        equity_today = cash + mv
        equity_curve.append((day, equity_today))
        equity = float(equity_today)

        # 5) force-close at end (once per day if final day)
        if force_close_on_end and (day == calendar[-1]) and open_positions:
            for sym, pos in list(open_positions.items()):
                df = data[sym]
                if day not in df.index:
                    continue
                c = float(df.at[day, "Close"])  
                slip = (slip_bps / 10000.0) * c
                eff = c - slip if pos.side == "long" else c + slip
                commission = commission_ps * pos.qty
                cash += (pos.qty * eff) if pos.side == "long" else (-pos.qty * eff)
                cash -= commission
                pnl = (eff - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - eff) * pos.qty
                trades.append({
                    "symbol": sym,
                    "side": pos.side,
                    "entry_date": pos.entry_date,
                    "entry_price": pos.entry_price,
                    "exit_date": day,
                    "exit_price": eff,
                    "qty": pos.qty,
                    "pnl": pnl,
                    "r_multiple": pnl / (pos.per_share_risk * pos.qty) if pos.per_share_risk > 0 else np.nan,
                    "commission": commission,
                    "slippage_cost": slip * pos.qty,
                    "reason": "force_end",
                    "part": ("runner" if pos.scaled else "full"),
                    "bars_held": pos.bars_held,
                    "mfe_R": pos.max_favorable_R,
                    "position_id": pos.position_id,
                })
                del open_positions[sym]
            equity_today = cash
            equity_curve[-1] = (day, equity_today)
            equity = float(equity_today)

    # --- Post-loop safeguard: force close any residual open positions lacking data on final calendar day ---
    residual_force_closes = 0
    if open_positions:
        final_day = calendar[-1]
        for sym, pos in list(open_positions.items()):
            df = data.get(sym)
            if df is None or len(df.index)==0:
                # no data at all; skip with zero value (could log)
                del open_positions[sym]; continue
            # find last available date up to final_day
            valid_dates = df.index[df.index <= final_day]
            if len(valid_dates)==0:
                del open_positions[sym]; continue
            ld = valid_dates[-1]
            c = float(df.at[ld, "Close"])
            slip = (slip_bps/10000.0)*c; eff = c - slip if pos.side=="long" else c + slip
            commission = commission_ps * pos.qty
            cash += (pos.qty*eff) if pos.side=="long" else (-pos.qty*eff); cash -= commission
            pnl = (eff-pos.entry_price)*pos.qty if pos.side=="long" else (pos.entry_price-eff)*pos.qty
            trades.append({
                "symbol": sym,
                "side": pos.side,
                "entry_date": pos.entry_date,
                "entry_price": pos.entry_price,
                "exit_date": ld,
                "exit_price": eff,
                "qty": pos.qty,
                "pnl": pnl,
                "r_multiple": pnl/(pos.per_share_risk*pos.qty) if pos.per_share_risk>0 else np.nan,
                "commission": commission,
                "slippage_cost": slip*pos.qty,
                "reason": "force_end_missing_final_bar" if ld != final_day else "force_end",
                "part": ("runner" if pos.scaled else "full"),
                "bars_held": pos.bars_held,
                "mfe_R": pos.max_favorable_R,
                "position_id": pos.position_id,
            })
            residual_force_closes += 1
            del open_positions[sym]
        # overwrite final equity to reflect closures
        equity = cash  # no open positions remain
        if len(equity_curve)>0 and equity_curve[-1][0]==final_day:
            equity_curve[-1] = (final_day, equity)
        else:
            equity_curve.append((final_day, equity))

    # ---------- wrap up ----------
    eq_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    daily_ret = eq_df["equity"].pct_change().fillna(0.0)

    start_eq = float(eq_df["equity"].iloc[0]) if not eq_df.empty else 0.0
    end_eq = float(eq_df["equity"].iloc[-1]) if not eq_df.empty else 0.0
    total_ret = (end_eq / start_eq) - 1.0 if start_eq > 0 else 0.0
    days = len(eq_df)
    years = max((eq_df.index[-1] - eq_df.index[0]).days / 365.25, 1e-9) if not eq_df.empty else 1e-9
    cagr = (end_eq / start_eq) ** (1 / years) - 1 if (start_eq > 0 and end_eq > 0) else float("nan")
    max_dd, _ = _drawdown_stats(eq_df["equity"]) if not eq_df.empty else (0.0, pd.NaT)
    sharpe = _annualized_sharpe(daily_ret) if not eq_df.empty else 0.0

    trades_df = pd.DataFrame(trades)
    entries_df = pd.DataFrame(entries_audit)
    if not entries_df.empty and not trades_df.empty and "position_id" in trades_df.columns:
        exit_qty = trades_df.groupby("position_id")["qty"].sum().rename("exit_qty")
        audit_qty = entries_df.set_index("position_id").join(exit_qty, how="left")
        audit_qty["exit_qty"].fillna(0, inplace=True)
        audit_qty["qty_diff"] = audit_qty["qty"] - audit_qty["exit_qty"]
        unmatched_positions = audit_qty[audit_qty["qty_diff"] != 0]
    else:
        audit_qty = pd.DataFrame(); unmatched_positions = pd.DataFrame()
    # --- Performance stats (gross & net) ---
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]  # strictly negative for PF denominator
        win_rate = len(wins) / len(trades_df) if len(trades_df) else 0.0
        avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0
        gross_profit = float(wins["pnl"].sum())
        gross_loss = float(-losses["pnl"].sum()) if len(losses) else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
        exit_commissions_sum = float(trades_df.get("commission", 0).sum()) if "commission" in trades_df.columns else 0.0
        slippage_sum = float(trades_df.get("slippage_cost", 0).sum()) if "slippage_cost" in trades_df.columns else 0.0
        trade_pnl_net = trades_df["pnl"] - trades_df.get("commission", 0.0) - trades_df.get("slippage_cost", 0.0)
        wins_net = trade_pnl_net[trade_pnl_net > 0]
        losses_net = trade_pnl_net[trade_pnl_net < 0]
        gross_profit_net = float(wins_net.sum())
        gross_loss_net = float(-losses_net.sum()) if len(losses_net) else 0.0
        profit_factor_net = (gross_profit_net / gross_loss_net) if gross_loss_net > 0 else np.inf
        avg_R = float(trades_df["r_multiple"].mean(skipna=True))
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_R = 0.0
        gross_profit = gross_loss = exit_commissions_sum = slippage_sum = profit_factor_net = gross_profit_net = gross_loss_net = 0.0
        trade_pnl_net = pd.Series(dtype=float)

    # --- Reconciliation (equity change vs aggregated trade components) ---
    if start_eq is not None and end_eq is not None:
        equity_delta = end_eq - start_eq
        price_pnl_sum = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
        entry_commissions_sum = float(entries_df.get("entry_commission", 0).sum()) if not entries_df.empty else 0.0
        expected_end_from_trades = start_eq + price_pnl_sum - (entry_commissions_sum + exit_commissions_sum)
        recon_residual = end_eq - expected_end_from_trades
        # cash flow reconstruction (entry + exit flows)
        if not entries_df.empty and "entry_cash_flow" in entries_df.columns:
            entry_cash_flow_sum = float(entries_df["entry_cash_flow"].sum())
        else:
            entry_cash_flow_sum = 0.0
        if not trades_df.empty:
            def _exit_cf(row):
                # long exit adds cash (qty*exit_price - commission); short exit removes ( -qty*exit_price - commission )
                if row.side == "long":
                    return row.exit_price * row.qty - row.commission
                else:
                    return -row.exit_price * row.qty - row.commission
            exit_cash_flow_sum = float(trades_df.apply(_exit_cf, axis=1).sum())
        else:
            exit_cash_flow_sum = 0.0
        cash_end_reconstructed = start_eq + entry_cash_flow_sum + exit_cash_flow_sum
        recon_residual_cashflow = end_eq - cash_end_reconstructed
        # Unrealized PnL (should be zero after force close safeguards)
        unrealized_pnl = 0.0
        if 'mv' in locals() and not open_positions:  # all closed; unrealized zero
            unrealized_pnl = 0.0
    else:
        equity_delta = price_pnl_sum = exit_commissions_sum = expected_end_from_trades = recon_residual = entry_commissions_sum = 0.0
        entry_cash_flow_sum = exit_cash_flow_sum = cash_end_reconstructed = recon_residual_cashflow = 0.0
        unrealized_pnl = 0.0

    # write CSVs
    eval_df = pd.DataFrame(evaluations)
    # Use unique timestamped filenames to avoid cross-run contamination
    eval_path = today_filename("backtest_evaluations", unique=True)
    if not eval_df.empty:
        eval_df = eval_df.sort_values(["date", "symbol"])
        log_dataframe(eval_df, eval_path)

    trades_path = today_filename("backtest_trades", unique=True)
    equity_path = today_filename("backtest_equity", unique=True)
    if not trades_df.empty:
        log_dataframe(trades_df, trades_path, overwrite=True)
    log_dataframe(eq_df.reset_index(), equity_path, overwrite=True)

    # ---- Additional diagnostics ----
    exit_reason_counts = trades_df["reason"].value_counts().to_dict() if not trades_df.empty else {}
    scale_out_count = int(exit_reason_counts.get("scale_out", 0))
    runner_exits = trades_df[trades_df["part"] == "runner"] if not trades_df.empty and "part" in trades_df.columns else pd.DataFrame()
    runner_target_wins = int(len(runner_exits[runner_exits["reason"].isin(["target","force_end"]) & (runner_exits["pnl"]>0)])) if not runner_exits.empty else 0
    time_stop_exits = int(exit_reason_counts.get("time_stop", 0))
    stop_exits = int(exit_reason_counts.get("stop", 0) + exit_reason_counts.get("stop_same_day", 0))
    target_full_exits = int(exit_reason_counts.get("target", 0) + exit_reason_counts.get("target_same_day", 0))
    median_bars_held = float(trades_df["bars_held"].median()) if (not trades_df.empty and "bars_held" in trades_df.columns) else 0.0
    median_mfe_R_wins = float(trades_df.loc[trades_df["pnl"]>0, "mfe_R"].median()) if (not trades_df.empty and "mfe_R" in trades_df.columns and len(trades_df[trades_df["pnl"]>0])>0) else 0.0
    median_mfe_R_losses = float(trades_df.loc[trades_df["pnl"]<=0, "mfe_R"].median()) if (not trades_df.empty and "mfe_R" in trades_df.columns and len(trades_df[trades_df["pnl"]<=0])>0) else 0.0

    summary = {
        "start_equity": start_eq,
        "end_equity": end_eq,
        "total_return": total_ret,
        "CAGR": cagr,
        "max_drawdown": max_dd,
        "daily_sharpe": sharpe,
        "days": days,
        "trades": int(len(trades_df)),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "profit_factor_net": float(profit_factor_net),
        "avg_R": float(avg_R),
        # gross components
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "gross_profit_net": float(gross_profit_net),
        "gross_loss_net": float(gross_loss_net),
    "entry_commissions_sum": float(entry_commissions_sum),
    "exit_commissions_sum": float(exit_commissions_sum),
    "entry_cash_flow_sum": float(entry_cash_flow_sum),
    "exit_cash_flow_sum": float(exit_cash_flow_sum),
    "cash_end_reconstructed": float(cash_end_reconstructed),
    "recon_residual_cashflow": float(recon_residual_cashflow),
        "slippage_sum": float(slippage_sum),
        # reconciliation
        "equity_delta": float(equity_delta),
        "price_pnl_sum": float(price_pnl_sum),
        "expected_end_from_trades": float(expected_end_from_trades),
        "recon_residual": float(recon_residual),
    "unrealized_pnl": float(unrealized_pnl),
    "residual_force_closes": int(residual_force_closes),
    # diagnostics
    "stop_exits": stop_exits,
    "target_exits": target_full_exits,
    "time_stop_exits": time_stop_exits,
    "scale_out_trades": scale_out_count,
    "runner_exits": int(len(runner_exits)),
    "runner_target_wins": runner_target_wins,
    "median_bars_held": median_bars_held,
    "median_mfe_R_wins": median_mfe_R_wins,
    "median_mfe_R_losses": median_mfe_R_losses,
        "files": [str(trades_path.name), str(equity_path.name)],
    }

    print("\n— Backtest Summary —")
    for k in ("start_equity","end_equity","total_return","CAGR","max_drawdown","daily_sharpe","days","trades","win_rate","avg_win","avg_loss","profit_factor","profit_factor_net","avg_R",
              "stop_exits","target_exits","time_stop_exits","scale_out_trades","runner_exits","runner_target_wins","median_bars_held","median_mfe_R_wins","median_mfe_R_losses"):
        v = summary[k]
        if k in {"total_return","CAGR","max_drawdown","win_rate"}:
            if isinstance(v, float) and not np.isnan(v):
                print(f"{k:<15}: {v*100:.2f}%")
            else:
                print(f"{k:<15}: N/A")
        else:
            print(f"{k:<15}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"{k:<15}: {v}")

    # Reconciliation print
    print("\nReconciliation:")
    print(f"  equity_delta            : {summary['equity_delta']:.2f}")
    print(f"  price_pnl_sum (gross)   : {summary['price_pnl_sum']:.2f}")
    print(f"  entry_commissions_sum   : {summary['entry_commissions_sum']:.2f}")
    print(f"  exit_commissions_sum    : {summary['exit_commissions_sum']:.2f}")
    print(f"  expected_end_from_trades: {summary['expected_end_from_trades']:.2f}")
    print(f"  recon_residual          : {summary['recon_residual']:.2f}")
    print(f"  entry_cash_flow_sum     : {summary['entry_cash_flow_sum']:.2f}")
    print(f"  exit_cash_flow_sum      : {summary['exit_cash_flow_sum']:.2f}")
    print(f"  cash_end_reconstructed  : {summary['cash_end_reconstructed']:.2f}")
    print(f"  recon_residual_cashflow : {summary['recon_residual_cashflow']:.2f}")
    print(f"  unrealized_pnl (post-close): {summary['unrealized_pnl']:.2f}")
    print(f"  residual_force_closes   : {summary['residual_force_closes']}")
    if not entries_df.empty:
        if unmatched_positions.empty:
            print("  AUDIT qty               : OK (entry qty == summed exit qty per position)")
        else:
            print(f"  AUDIT qty MISMATCH      : {len(unmatched_positions)} position(s); first 5")
            print(unmatched_positions.head(5).to_string())
    print(f"  gross_profit            : {summary['gross_profit']:.2f}")
    print(f"  gross_loss              : {summary['gross_loss']:.2f}")
    print(f"  PF_gross                : {summary['profit_factor']:.3f}")
    print(f"  PF_net                  : {summary['profit_factor_net']:.3f}")

    if not trades_df.empty:
        t5 = trades_df.sort_values("r_multiple", ascending=False).head(5)
        b5 = trades_df.sort_values("r_multiple", ascending=True).head(5)
        print("\nTop 5 trades by R:")
        print(t5[["symbol","side","entry_date","entry_price","exit_date","exit_price","qty","pnl","r_multiple","commission","slippage_cost"]].to_string(index=False))
        print("\nBottom 5 trades by R:")
        print(b5[["symbol","side","entry_date","entry_price","exit_date","exit_price","qty","pnl","r_multiple","commission","slippage_cost"]].to_string(index=False))

    print("\nFiles written:")
    if not eval_df.empty:
        print(f"  - {eval_path.name}")
    print(f"  - {trades_path.name}")
    print(f"  - {equity_path.name}")
    return summary
