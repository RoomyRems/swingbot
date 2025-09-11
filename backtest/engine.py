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
from pathlib import Path
import hashlib
import yaml
import sys as _sys
import platform as _platform

from risk.manager import compute_levels as rm_compute_levels, size_position as rm_size_position
from utils.config import load_config
from utils.logger import today_filename, log_dataframe
from broker.alpaca import get_daily_bars
from fundamentals.screener import build_fund_ctx, fundamentals_pass_at_fill, earnings_in_window, _min_avg_vol, _get_backtest_network_mode
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

def _derive_mtf_state_from_ctx(weekly_ctx: dict | None) -> str:
    """Return 'up' | 'down' | 'none' from cached weekly MACD context when available."""
    try:
        if weekly_ctx is None:
            return "none"
        wmacd = weekly_ctx.get("wmacd"); wmacds = weekly_ctx.get("wmacds"); wmacdh = weekly_ctx.get("wmacdh")
        if all([np.isfinite(wmacd), np.isfinite(wmacds), np.isfinite(wmacdh)]):
            if (wmacd > wmacds) and (wmacdh > 0):
                return "up"
            elif (wmacd < wmacds) and (wmacdh < 0):
                return "down"
        return "none"
    except Exception:
        return "none"

def _compute_rvol20(df: pd.DataFrame, day: pd.Timestamp) -> float:
    """Return RVOL20 at day if present else fallback Volume/mean(20) from history up to day."""
    try:
        if (day in df.index) and ("RVOL20" in df.columns) and np.isfinite(float(df.at[day, "RVOL20"])):
            return float(df.at[day, "RVOL20"])
    except Exception:
        pass
    try:
        if (day in df.index) and ("Volume" in df.columns):
            idx = df.index.get_loc(day)
            sl = df.iloc[: idx + 1]
            if len(sl) >= 2:
                v = float(sl.iloc[-1]["Volume"]) if np.isfinite(sl.iloc[-1]["Volume"]) else np.nan
                mv = float(sl["Volume"].rolling(20, min_periods=1).mean().iloc[-1])
                if np.isfinite(v) and np.isfinite(mv) and mv > 0:
                    return v / mv
    except Exception:
        pass
    return float("nan")

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

def _intraday_recover_check(symbol: str, day: pd.Timestamp, level_px: float, minutes: int) -> bool:
        """
        Return True if, in the last `minutes` of the session for `day`, price reclaimed `level_px`.
        Default stub returns False (no intraday data wired). If you have a feed, implement:
            - fetch intraday bars for `day` within [close - minutes, close]
            - for long: any bar close >= level_px; for short: close <= level_px
        """
        return False

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
    # Evidence-exit hysteresis state
    mom_fail_streak: int = 0
    cycle_fail_streak: int = 0
    val_fail_streak: int = 0
    prev_evidence_weight: float = 0.0
    prev_evidence_date: pd.Timestamp | None = None
    # Entry diagnostics (persist to exit rows)
    entry_setup_type: str = ""
    entry_adx14: float = float("nan")
    entry_rvol20: float = float("nan")
    entry_ema20_dist_pct: float = float("nan")
    entry_pivot_dist_pct: float = float("nan")
    entry_atr_pct: float = float("nan")
    entry_mtf_state: str = ""
    qty_cap_reason: str = "full_size"
    order_type: str = "market"
    days_to_fill: float = float("nan")
    # Lifecycle flags / timers
    time_to_BE_bars: int | None = None
    time_to_1R_bars: int | None = None
    be_armed_bar: pd.Timestamp | None = None
    trail_active_bar: pd.Timestamp | None = None

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
    # Logging controls (gate outputs to avoid redundancy)
    log_cfg = (bt_cfg.get("logging", {}) or {})
    log_eval_csv      = bool(log_cfg.get("write_evaluations_csv", True))
    log_daily_csv     = bool(log_cfg.get("write_daily_metrics_csv", True))
    log_drawdowns_csv = bool(log_cfg.get("write_drawdowns_csv", True))
    log_meta_files    = bool(log_cfg.get("write_meta", True))
    print_rollups     = bool(log_cfg.get("print_rollups", True))
    print_pending     = bool(log_cfg.get("print_pending_stats", True))
    print_recon       = bool(log_cfg.get("print_reconciliation", True))
    print_topbottom   = bool(log_cfg.get("print_top_bottom_trades", True))
    # Evidence-based exit (Burns: exit when the case weakens)
    evx_cfg = (bt_cfg.get("evidence_exit", {}) or {})
    evx_enabled   = bool(evx_cfg.get("enabled", True))
    evx_pre1R_bars= int(evx_cfg.get("pre1R_max_bars", 8))
    evx_pre1R_mfe = float(evx_cfg.get("pre1R_min_mfe", 0.5))
    evx_mom_fail  = bool(evx_cfg.get("momentum_fail", True))
    evx_cycle_fail= bool(evx_cfg.get("cycle_fail", True))
    evx_val_fail  = bool(evx_cfg.get("value_zone_fail", True))
    # NEW — softness / hysteresis with safe defaults
    evx_min_hold_bars = int(evx_cfg.get("min_hold_bars", 0))
    consec_default = int(evx_cfg.get("consecutive_fails_default", 1))
    mom_consec_default = int(evx_cfg.get("momentum_consecutive", consec_default) or consec_default)
    cyc_consec_default = int(evx_cfg.get("cycle_consecutive", consec_default) or consec_default)
    val_consec_default = int(evx_cfg.get("value_zone_consecutive", consec_default) or consec_default)

    w_cfg = (evx_cfg.get("weights") or {})
    w_mom_default = float(w_cfg.get("momentum", 0.5))
    w_cyc_default = float(w_cfg.get("cycle", 0.5))
    w_val_default = float(w_cfg.get("value_zone", 1.0))
    thr_single_default = float(evx_cfg.get("threshold_single", 1.0))
    thr_two_default = float(evx_cfg.get("threshold_two_bar", 1.5))

    vz_buf_pct_default = float(evx_cfg.get("value_zone_fail_buffer_pct", 0.0))
    cyc_gap_min_default = float(evx_cfg.get("cycle_fail_gap_min", 0.0))
    mom_need_zero_default = bool(evx_cfg.get("momentum_need_zero_cross", False))

    # Setup-aware overrides
    setup_ovr = (evx_cfg.get("setup_overrides") or {})

    def _evx_params_for(pos_setup: str):
        params = {
            "mom_consec": mom_consec_default,
            "cyc_consec": cyc_consec_default,
            "val_consec": val_consec_default,
            "w_mom": w_mom_default,
            "w_cyc": w_cyc_default,
            "w_val": w_val_default,
            "thr_single": thr_single_default,
            "thr_two": thr_two_default,
            "vz_buf_pct": vz_buf_pct_default,
            "cyc_gap_min": cyc_gap_min_default,
            "mom_need_zero": mom_need_zero_default,
        }
        ovr = (setup_ovr.get(pos_setup or "", {}) or {})
        if "momentum_consecutive" in ovr:
            params["mom_consec"] = int(ovr["momentum_consecutive"])  # type: ignore
        if "cycle_consecutive" in ovr:
            params["cyc_consec"] = int(ovr["cycle_consecutive"])  # type: ignore
        if "value_zone_consecutive" in ovr:
            params["val_consec"] = int(ovr["value_zone_consecutive"])  # type: ignore
        if "value_zone_fail_buffer_pct" in ovr:
            params["vz_buf_pct"] = float(ovr["value_zone_fail_buffer_pct"])  # type: ignore
        if "cycle_fail_gap_min" in ovr:
            params["cyc_gap_min"] = float(ovr["cycle_fail_gap_min"])  # type: ignore
        if "momentum_need_zero_cross" in ovr:
            params["mom_need_zero"] = bool(ovr["momentum_need_zero_cross"])  # type: ignore
        return params
    # Trailing after R
    trail_cfg = (bt_cfg.get("trail_after_R", {}) or {})
    trail_enabled = bool(trail_cfg.get("enabled", False))
    trail_start_R = float(trail_cfg.get("start_R", 1.0))
    # Stale data exit
    stale_cfg = (bt_cfg.get("stale_data_exit", {}) or {})
    stale_enabled = bool(stale_cfg.get("enabled", False))
    stale_max_days = int(stale_cfg.get("max_gap_days", 5))
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

    # ---------- fundamentals prefetch/context (backtest-friendly) ----------
    fcfg = (cfg.get("fundamentals", {}) or {})
    fund_ctx = None
    if bool(fcfg.get("enabled", False)):
        try:
            fund_ctx = build_fund_ctx(cfg, list(data.keys()), start_date.date(), end_date.date())
        except Exception:
            fund_ctx = None

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
            # Fundamentals gate at scan-time: support earnings-only blackout mode
            fcfg_scan = (cfg.get("fundamentals", {}) or {})
            fundamentals_ok = True
            reason_fund = "fundamentals"
            if bool(fcfg_scan.get("enabled", False)):
                if bool(fcfg_scan.get("only_earnings_blackout", False)):
                    try:
                        blk = int(fcfg_scan.get("earnings_blackout_days", 0))
                    except Exception:
                        blk = 0
                    if blk > 0 and fund_ctx is not None:
                        if earnings_in_window(sym, pd.to_datetime(d).date(), blk, fund_ctx):
                            fundamentals_ok = False
                            reason_fund = "earnings"
                else:
                    try:
                        price_o = float(row.get("Close", np.nan))
                        avgv_o = float(sl["Volume"].tail(50).mean()) if "Volume" in sl.columns else float("nan")
                    except Exception:
                        price_o = float("nan"); avgv_o = float("nan")
                    min_p = float(fcfg_scan.get("min_price", 0))
                    try:
                        min_vol = float(_min_avg_vol(cfg))
                    except Exception:
                        min_vol = float((cfg.get("trading", {}).get("filters", {}) or {}).get("min_avg_vol50", 300000))
                    fundamentals_ok = (
                        np.isfinite(price_o) and np.isfinite(avgv_o) and price_o > 0 and avgv_o > 0 and
                        (price_o >= min_p) and (avgv_o >= min_vol)
                    )
                    reason_fund = "fundamentals"

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
                if not fundamentals_ok:
                    reason = reason_fund
                    if reason == "earnings":
                        earnings_block_scan += 1
                    accept = False
                elif not regime_ok:
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
                if not fundamentals_ok:
                    reason = reason_fund; accept = False
                    if reason == "earnings":
                        earnings_block_scan += 1
                elif not regime_ok:
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
            try:
                ema20_val = float(row.get("EMA20", np.nan))
                close_val = float(row.get("Close", np.nan))
                ema20_dist_pct = ((close_val - ema20_val)/ema20_val) if np.isfinite(ema20_val) and ema20_val>0 and np.isfinite(close_val) else np.nan
            except Exception:
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
    daily_metrics: List[dict] = []
    limit_signals_count = 0
    limit_expired_count = 0
    # Fundamentals gating counters
    earnings_block_market = 0
    earnings_block_limit_skip = 0
    earnings_block_scan = 0
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
            if df is None:
                continue
            # If today's bar is missing, optionally trigger stale-data exit; otherwise skip processing for this symbol today
            if day not in df.index:
                if stale_enabled:
                    try:
                        prior = df.index[df.index <= day]
                        if len(prior) > 0:
                            last_day = prior[-1]
                            if (day - last_day).days >= stale_max_days:
                                c_last = float(df.at[last_day, "Close"])
                                fill_px = c_last
                                slip = (slip_bps / 10000.0) * fill_px
                                fill_eff = fill_px - slip if pos.side == "long" else fill_px + slip
                                commission = commission_ps * pos.qty
                                cash += (pos.qty * fill_eff) if pos.side == "long" else (-pos.qty * fill_eff)
                                cash -= commission
                                pnl = (fill_eff - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - fill_eff) * pos.qty
                                trades.append({
                                    "symbol": sym,
                                    "side": pos.side,
                                    "entry_date": pos.entry_date,
                                    "entry_price": pos.entry_price,
                                    "exit_date": last_day,
                                    "exit_price": fill_eff,
                                    "qty": pos.qty,
                                    "pnl": pnl,
                                    "r_multiple": pnl / (pos.per_share_risk * pos.qty) if pos.per_share_risk > 0 else np.nan,
                                    "commission": commission,
                                    "slippage_cost": slip * pos.qty,
                                    "reason": "stale_data_exit",
                                    "part": ("runner" if pos.scaled else "full"),
                                    "bars_held": pos.bars_held,
                                    "mfe_R": pos.max_favorable_R,
                                    "position_id": pos.position_id,
                                })
                                del open_positions[sym]
                                continue
                    except Exception:
                        pass
                # no bar and no stale exit – skip other checks for this symbol today
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

            # Breakeven arming (safer): only after >= 1.0R; remove ratchet that pins stop at BE
            if not pos.be_armed and pos.per_share_risk > 0:
                arm_R = max(1.0, float(be_R)) if be_R > 0 else 1.0
                if pos.side == "long":
                    if h >= pos.entry_price + arm_R * pos.per_share_risk and not (be_intrabar == "favor_stop" and l <= pos.stop):
                        pos.stop = max(pos.stop, pos.entry_price); pos.be_armed = True
                        if getattr(pos, "time_to_BE_bars", None) is None:
                            pos.time_to_BE_bars = pos.bars_held + 1
                            pos.be_armed_bar = day
                else:
                    if l <= pos.entry_price - arm_R * pos.per_share_risk and not (be_intrabar == "favor_stop" and h >= pos.stop):
                        pos.stop = min(pos.stop, pos.entry_price); pos.be_armed = True
                        if getattr(pos, "time_to_BE_bars", None) is None:
                            pos.time_to_BE_bars = pos.bars_held + 1
                            pos.be_armed_bar = day

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
                        "position_id": pos.position_id,
                    })
                    del open_positions[sym]
                    continue

            exit_reason = None; fill_px = None; target_hit = False

            # Optional: stale data exit if no bar for too long
            if exit_reason is None and stale_enabled:
                try:
                    # if today not present, find last bar and compute gap in days
                    if day not in df.index:
                        prior = df.index[df.index <= day]
                        if len(prior) > 0:
                            last_day = prior[-1]
                            if (day - last_day).days >= stale_max_days:
                                # exit at last available close
                                c_last = float(df.at[last_day, "Close"])
                                fill_px = c_last; exit_reason = "stale_data_exit"
                    else:
                        # last seen day is today; okay
                        pass
                except Exception:
                    pass

            # Evidence-based exit BEFORE stop/target checks
            if exit_reason is None and evx_enabled:
                # pre-1R no-progress budget
                if pos.per_share_risk > 0 and pos.bars_held >= evx_pre1R_bars and pos.max_favorable_R < evx_pre1R_mfe:
                    fill_px = c; exit_reason = "evidence_no_progress"
                else:
                    try:
                        macd  = float(df.at[day, "MACD"]) if (day in df.index and "MACD" in df.columns) else np.nan
                        macds = float(df.at[day, "MACDs"]) if (day in df.index and "MACDs" in df.columns) else np.nan
                        # Respect configured stochastic mode for cycle evidence
                        ccfg = (cfg.get("cycle", {}) or {})
                        use_burns = str(ccfg.get("stoch_mode","")).lower() in {"burns","5_3_2","burns_5_3_2"}
                        kcol = "SlowK_5_3_2" if (use_burns and "SlowK_5_3_2" in df.columns) else "SlowK"
                        dcol = "SlowD_5_3_2" if (use_burns and "SlowD_5_3_2" in df.columns) else "SlowD"
                        k = float(df.at[day, kcol]) if (day in df.index and kcol in df.columns) else np.nan
                        d = float(df.at[day, dcol]) if (day in df.index and dcol in df.columns) else np.nan
                        ema50_now = float(df.at[day, "EMA50"]) if (day in df.index and "EMA50" in df.columns) else np.nan
                        ema20_now = float(df.at[day, "EMA20"]) if (day in df.index and "EMA20" in df.columns) else np.nan
                    except Exception:
                        macd = macds = k = d = ema50_now = ema20_now = np.nan

                    # Setup-aware parameters
                    params = _evx_params_for(getattr(pos, "setup_type", ""))

                    # Buffered fail checks
                    mom_fail_now = False
                    cyc_fail_now = False
                    val_fail_now = False
                    if evx_mom_fail and np.isfinite(macd) and np.isfinite(macds):
                        if params["mom_need_zero"]:
                            if pos.side == "long":
                                mom_fail_now = (macd < 0.0) and (macd < macds)
                            else:
                                mom_fail_now = (macd > 0.0) and (macd > macds)
                        else:
                            if pos.side == "long":
                                mom_fail_now = (macd < 0.0) or (macd < macds)
                            else:
                                mom_fail_now = (macd > 0.0) or (macd > macds)
                    if evx_cycle_fail and np.isfinite(k) and np.isfinite(d):
                        if pos.side == "long":
                            cyc_fail_now = (d - k) >= params["cyc_gap_min"] if params["cyc_gap_min"] > 0 else (k < d)
                        else:
                            cyc_fail_now = (k - d) >= params["cyc_gap_min"] if params["cyc_gap_min"] > 0 else (k > d)
                    if evx_val_fail and np.isfinite(ema50_now):
                        if pos.side == "long":
                            thr_px = ema50_now * (1.0 - params["vz_buf_pct"]) if params["vz_buf_pct"] > 0 else ema50_now
                            val_fail_now = c <= thr_px if params["vz_buf_pct"] > 0 else (c < ema50_now)
                        else:
                            thr_px = ema50_now * (1.0 + params["vz_buf_pct"]) if params["vz_buf_pct"] > 0 else ema50_now
                            val_fail_now = c >= thr_px if params["vz_buf_pct"] > 0 else (c > ema50_now)

                    # Initialize telemetry defaults (will persist through exit record)
                    ev_w_now = 0.0
                    ev_two_sum = 0.0
                    ev_mom_streak = pos.mom_fail_streak
                    ev_cyc_streak = pos.cycle_fail_streak
                    ev_val_streak = pos.val_fail_streak

                    # Grace period: skip evidence exits during initial bars
                    within_grace = pos.bars_held < evx_min_hold_bars
                    if not within_grace:
                        # Update fail streaks
                        pos.mom_fail_streak  = (pos.mom_fail_streak + 1) if mom_fail_now else 0
                        pos.cycle_fail_streak= (pos.cycle_fail_streak + 1) if cyc_fail_now else 0
                        pos.val_fail_streak  = (pos.val_fail_streak + 1) if val_fail_now else 0
                        ev_mom_streak = pos.mom_fail_streak
                        ev_cyc_streak = pos.cycle_fail_streak
                        ev_val_streak = pos.val_fail_streak

                        # Consecutive trigger
                        consec_trigger = (
                            (mom_fail_now and pos.mom_fail_streak  >= params["mom_consec"]) or
                            (cyc_fail_now and pos.cycle_fail_streak>= params["cyc_consec"]) or
                            (val_fail_now and pos.val_fail_streak  >= params["val_consec"])
                        )

                        # Weighted trigger
                        ev_w_now = (
                            (params["w_mom"] if mom_fail_now else 0.0) +
                            (params["w_cyc"] if cyc_fail_now else 0.0) +
                            (params["w_val"] if val_fail_now else 0.0)
                        )
                        prev_ok = (pos.prev_evidence_date is not None and pos.prev_evidence_date == (day - pd.tseries.offsets.BDay(1)))
                        ev_two_sum = ev_w_now + (pos.prev_evidence_weight if prev_ok else 0.0)
                        weighted_trigger = (ev_w_now >= params["thr_single"]) or (ev_two_sum >= params["thr_two"])

                        if consec_trigger or weighted_trigger:
                            fill_px = c
                            if consec_trigger:
                                exit_reason = "evidence_consecutive"
                            else:
                                # weighted trigger
                                fails = [("momentum", mom_fail_now), ("cycle", cyc_fail_now), ("value_zone", val_fail_now)]
                                fail_names = [name for name, ok in fails if ok]
                                # If exactly one component failed and single-bar threshold met, preserve legacy reason name
                                if (len(fail_names) == 1) and (ev_w_now >= params["thr_single"]):
                                    only = fail_names[0]
                                    if only == "momentum":
                                        exit_reason = "evidence_momentum_fail"
                                    elif only == "cycle":
                                        exit_reason = "evidence_cycle_fail"
                                    else:
                                        exit_reason = "evidence_value_zone_fail"
                                else:
                                    exit_reason = "evidence_weighted"

                    # Save rolling evidence state for 2-bar test
                    pos.prev_evidence_weight = float(ev_w_now)
                    pos.prev_evidence_date = day

                    # Optional intraday recovery veto
                    if exit_reason is not None:
                        idr_cfg = (evx_cfg.get("intraday_recovery") or {})
                        if bool(idr_cfg.get("enabled", False)):
                            look_m = int(idr_cfg.get("minutes", 60))
                            reclaim_line = str(idr_cfg.get("require_reclaim", "EMA20")).upper()
                            reclaim_level = None
                            if reclaim_line == "EMA20" and np.isfinite(ema20_now):
                                reclaim_level = ema20_now
                            elif reclaim_line == "EMA50" and np.isfinite(ema50_now):
                                reclaim_level = ema50_now
                            if reclaim_level is not None:
                                if _intraday_recover_check(sym, day, reclaim_level, look_m):
                                    exit_reason = None
                                    fill_px = None
            # Pre-earnings exit at close: sell the last trading day before an earnings date
            if exit_reason is None and fund_ctx is not None:
                try:
                    fcfg_local = (cfg.get("fundamentals", {}) or {})
                    if bool(fcfg_local.get("exit_before_earnings", False)):
                        cal_map = (fund_ctx.get("earnings_calendar", {}) or {})
                        e_dates = cal_map.get(sym, []) or []
                        if e_dates:
                            # find next trading day in the global backtest calendar
                            try:
                                cal_idx = calendar.index(day)
                                next_trading = calendar[cal_idx + 1] if cal_idx + 1 < len(calendar) else None
                            except Exception:
                                next_trading = None
                            if next_trading is not None:
                                today_d = pd.to_datetime(day).date()
                                next_d  = pd.to_datetime(next_trading).date()
                                # If an earnings date falls between (today, next_trading] → exit today at close
                                if any((today_d < ed <= next_d) for ed in e_dates):
                                    fill_px = c
                                    exit_reason = "pre_earnings"
                except Exception:
                    pass
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
                    if getattr(pos, "time_to_1R_bars", None) is None:
                        intraday_R = ((h - pos.entry_price) / pos.per_share_risk) if pos.side == "long" else ((pos.entry_price - l) / pos.per_share_risk)
                        if intraday_R >= 1.0:
                            pos.time_to_1R_bars = pos.bars_held
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

            # Post-1R trailing to EMA20 (optional)
            if exit_reason is None and trail_enabled and pos.per_share_risk > 0 and (day in df.index):
                reached_R = ((h - pos.entry_price) / pos.per_share_risk) if pos.side == "long" else ((pos.entry_price - l) / pos.per_share_risk)
                if reached_R >= trail_start_R and "EMA20" in df.columns:
                    try:
                        ema20_now = float(df.at[day, "EMA20"]) if np.isfinite(df.at[day, "EMA20"]) else np.nan
                    except Exception:
                        ema20_now = np.nan
                    if np.isfinite(ema20_now):
                        if pos.side == "long":
                            pos.stop = max(pos.stop, round(ema20_now, 2))
                        else:
                            pos.stop = min(pos.stop, round(ema20_now, 2))
                        if getattr(pos, "trail_active_bar", None) is None:
                            pos.trail_active_bar = day
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
                    "evidence_w_now": float(locals().get("ev_w_now", 0.0)),
                    "evidence_two_bar": float(locals().get("ev_two_sum", 0.0)),
                    "mom_fail_now": bool(locals().get("mom_fail_now", False)),
                    "cyc_fail_now": bool(locals().get("cyc_fail_now", False)),
                    "val_fail_now": bool(locals().get("val_fail_now", False)),
                    "mom_streak": int(getattr(pos, "mom_fail_streak", 0)),
                    "cyc_streak": int(getattr(pos, "cycle_fail_streak", 0)),
                    "val_streak": int(getattr(pos, "val_fail_streak", 0)),
                    # Entry diagnostics persisted
                    "entry_setup_type": getattr(pos, "entry_setup_type", pos.setup_type),
                    "entry_adx14": getattr(pos, "entry_adx14", np.nan),
                    "entry_rvol20": getattr(pos, "entry_rvol20", np.nan),
                    "entry_ema20_dist_pct": getattr(pos, "entry_ema20_dist_pct", np.nan),
                    "entry_pivot_dist_pct": getattr(pos, "entry_pivot_dist_pct", np.nan),
                    "entry_atr_pct": getattr(pos, "entry_atr_pct", np.nan),
                    "entry_mtf_state": getattr(pos, "entry_mtf_state", ""),
                    "qty_cap_reason": getattr(pos, "qty_cap_reason", "full_size"),
                    "order_type": getattr(pos, "order_type", "market"),
                    "days_to_fill": getattr(pos, "days_to_fill", np.nan),
                    # Lifecycle timers
                    "time_to_BE_bars": getattr(pos, "time_to_BE_bars", None),
                    "time_to_1R_bars": getattr(pos, "time_to_1R_bars", None),
                    "be_armed_bar": getattr(pos, "be_armed_bar", None),
                    "trail_active_bar": getattr(pos, "trail_active_bar", None),
                    # Evidence exit snapshot (filled for evidence exits, NaN/default otherwise)
                    "exit_macd": float(locals().get("macd", np.nan)),
                    "exit_macds": float(locals().get("macds", np.nan)),
                    "exit_slowk": float(locals().get("k", np.nan)),
                    "exit_slowd": float(locals().get("d", np.nan)),
                    "exit_ema50": float(locals().get("ema50_now", np.nan)),
                    "exit_close": float(locals().get("c", np.nan)),
                    "evidence_mom_fail": int(bool(locals().get("mom_fail_now", False))),
                    "evidence_cycle_fail": int(bool(locals().get("cyc_fail_now", False))),
                    "evidence_value_fail": int(bool(locals().get("val_fail_now", False))),
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
            # Skip if this symbol is already open (avoid overwriting an existing position)
            if pend.symbol in open_positions:
                pending_markets.remove(pend)
                continue
            if len(open_positions) >= max_open or (_current_total_risk() >= max_risk_pct):
                pending_markets.remove(pend)
                continue
            wk = _week_key(day)
            if max_new_week > 0 and week_new_count[wk] >= max_new_week:
                pending_markets.remove(pend)
                continue
            # Fill-time MTF alignment gate: enforce higher-timeframe harmony without double-penalizing in scoring
            mtf_rt_cfg = (cfg.get("trading", {}).get("mtf") or {})
            # accept filter_at_fill from either trading.mtf.filter_at_fill or trading.filter_at_fill
            filter_at_fill = bool(mtf_rt_cfg.get("filter_at_fill", (cfg.get("trading", {}) or {}).get("filter_at_fill", False)))
            if filter_at_fill:
                try:
                    idx_fill = df.index.get_loc(day)
                    sl_fill = df.iloc[: idx_fill + 1]
                    wctx_fill = _wcache.get((pend.symbol, idx_fill))
                    if wctx_fill is None:
                        wctx_fill = weekly_context(sl_fill)
                        _wcache[(pend.symbol, idx_fill)] = wctx_fill
                    mtf_ok_fill, _ = _mtf_ok_for_slice(sl_fill, cfg, pend.direction, weekly_ctx=wctx_fill)
                except Exception:
                    mtf_ok_fill = True  # fail-open if anything unexpected
                if not mtf_ok_fill:
                    pending_markets.remove(pend)
                    continue
            # Fill-time fundamentals earnings blackout (no network; uses prefetched context)
            try:
                f_ok, _f_reason = fundamentals_pass_at_fill(pend.symbol, pd.to_datetime(day).date(), cfg, fund_ctx)
            except Exception:
                # fail-open if anything unexpected
                f_ok = True; _f_reason = "fail_open_error"
            if not f_ok:
                # cancel this market entry; do not reschedule
                pending_markets.remove(pend)
                earnings_block_market += 1
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
            # Compute cap reasons
            psr = abs(entry_px - stop)
            try:
                risk_dollars = float(cfg_day.get("risk", {}).get("risk_per_trade_pct", 0.0)) * float(equity)
                base_qty = int(math.floor(risk_dollars / max(psr, 1e-12))) if risk_dollars > 0 else 0
            except Exception:
                base_qty = 0
            qty_mgr = rm_size_position(cfg_day, entry_px, stop)
            qty_bp = _qty_bp_cap(entry_px, cash, bp_multiple)
            qty = min(qty_mgr, qty_bp)
            cap_reason = "full_size"
            if qty_mgr < base_qty:
                cap_reason = "risk_cap"
            elif qty < qty_mgr:
                cap_reason = "bp_cap"
            if qty <= 0:
                pending_markets.remove(pend)
                continue
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
            # Entry diagnostics
            pos_created.entry_setup_type = pos_created.setup_type
            try:
                pos_created.entry_adx14 = float(df.at[day, "ADX14"]) if "ADX14" in df.columns else float("nan")
            except Exception:
                pos_created.entry_adx14 = float("nan")
            pos_created.entry_rvol20 = _compute_rvol20(df, day)
            pos_created.entry_ema20_dist_pct = getattr(pos_created, "ema20_dist_pct", float("nan"))
            pos_created.entry_pivot_dist_pct = getattr(pos_created, "pivot_dist_pct", float("nan"))
            pos_created.entry_atr_pct = getattr(pos_created, "atr_pct_entry", float("nan"))
            # MTF state at fill
            try:
                idx_fill = df.index.get_loc(day)
                wctx_fill = _wcache.get((pend.symbol, idx_fill))
                if wctx_fill is None:
                    wctx_fill = weekly_context(df.iloc[: idx_fill + 1])
                    _wcache[(pend.symbol, idx_fill)] = wctx_fill
                pos_created.entry_mtf_state = _derive_mtf_state_from_ctx(wctx_fill)
            except Exception:
                pos_created.entry_mtf_state = ""
            pos_created.qty_cap_reason = cap_reason
            pos_created.order_type = "market"
            entries_audit.append({
                "position_id": pos_created.position_id,
                "symbol": pend.symbol,
                "entry_date": day,
                "entry_price": entry_px,
                "qty": qty,
                "entry_commission": commission,
                "entry_cash_flow": ((-qty * entry_px) if pend.direction == "long" else (qty * entry_px)) - commission,
                "order_type": "market",
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
                    # Entry diag copy
                    "entry_setup_type": getattr(pos, "entry_setup_type", pos.setup_type),
                    "entry_adx14": getattr(pos, "entry_adx14", np.nan),
                    "entry_rvol20": getattr(pos, "entry_rvol20", np.nan),
                    "entry_ema20_dist_pct": getattr(pos, "entry_ema20_dist_pct", np.nan),
                    "entry_pivot_dist_pct": getattr(pos, "entry_pivot_dist_pct", np.nan),
                    "entry_atr_pct": getattr(pos, "entry_atr_pct", np.nan),
                    "entry_mtf_state": getattr(pos, "entry_mtf_state", ""),
                    "qty_cap_reason": getattr(pos, "qty_cap_reason", "full_size"),
                    "order_type": getattr(pos, "order_type", "market"),
                    "days_to_fill": getattr(pos, "days_to_fill", np.nan),
                    # Lifecycle
                    "time_to_BE_bars": getattr(pos, "time_to_BE_bars", None),
                    "time_to_1R_bars": getattr(pos, "time_to_1R_bars", None),
                    "be_armed_bar": getattr(pos, "be_armed_bar", None),
                    "trail_active_bar": getattr(pos, "trail_active_bar", None),
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
            # Skip if symbol already has an open position (avoid duplicate entries)
            if pend.symbol in open_positions:
                pending_limits.remove(pend)
                continue
            if day < pend.signal_date or day > pend.expires:
                if day > pend.expires:
                    pending_limits.remove(pend)
                    limit_expired_count += 1
                continue
            o = float(df.at[day, "Open"])
            h = float(df.at[day, "High"])
            l = float(df.at[day, "Low"])
            hit = (l <= pend.limit_px) if pend.direction == "long" else (h >= pend.limit_px)
            if not hit:
                if day == pend.expires:
                    pending_limits.remove(pend)
                    limit_expired_count += 1
                continue
            # Fill-time MTF alignment gate: only applied when the limit is actually hit today.
            # If mismatched, skip filling today but keep the pending order until expiry.
            mtf_rt_cfg = (cfg.get("trading", {}).get("mtf") or {})
            filter_at_fill = bool(mtf_rt_cfg.get("filter_at_fill", (cfg.get("trading", {}) or {}).get("filter_at_fill", False)))
            if filter_at_fill:
                try:
                    idx_fill = df.index.get_loc(day)
                    sl_fill = df.iloc[: idx_fill + 1]
                    wctx_fill = _wcache.get((pend.symbol, idx_fill))
                    if wctx_fill is None:
                        wctx_fill = weekly_context(sl_fill)
                        _wcache[(pend.symbol, idx_fill)] = wctx_fill
                    mtf_ok_fill, _ = _mtf_ok_for_slice(sl_fill, cfg, pend.direction, weekly_ctx=wctx_fill)
                except Exception:
                    mtf_ok_fill = True
                if not mtf_ok_fill:
                    # skip fill today; keep pending for future bars within horizon
                    continue
            # Fill-time fundamentals earnings blackout (no network; uses prefetched context)
            try:
                f_ok, _f_reason = fundamentals_pass_at_fill(pend.symbol, pd.to_datetime(day).date(), cfg, fund_ctx)
            except Exception:
                f_ok = True; _f_reason = "fail_open_error"
            if not f_ok:
                # blocked today; keep pending for future bars within horizon
                earnings_block_limit_skip += 1
                continue
            # Fill-time core revalidation (require N-of-5 core energies at fill); default enabled
            try:
                core_fill_cfg = (cfg.get("trading", {}).get("core_at_fill") or {})
                core_check_enabled = bool(core_fill_cfg.get("enabled", True))
            except Exception:
                core_fill_cfg = {}; core_check_enabled = True
            if core_check_enabled:
                try:
                    idx_fill = df.index.get_loc(day)
                    sl_fill = df.iloc[: idx_fill + 1]
                    # ensure weekly ctx for the slice so Scale energy is valid
                    wctx_fill = _wcache.get((pend.symbol, idx_fill))
                    if wctx_fill is None:
                        wctx_fill = weekly_context(sl_fill)
                        _wcache[(pend.symbol, idx_fill)] = wctx_fill
                    eng_fill = evaluate_five_energies(sl_fill, cfg, weekly_ctx=wctx_fill) or {}
                    core_count_fill = int(eng_fill.get("core_pass_count", int(eng_fill.get("score", 0))))
                    core_min_fill = int(core_fill_cfg.get(
                        "min_core_energies",
                        (cfg.get("trading", {}) or {}).get("min_core_energies", (cfg.get("trading", {}) or {}).get("min_score", 4))
                    ))
                    if bool(core_fill_cfg.get("require_same_direction", True)):
                        dir_ok = (str(eng_fill.get("direction", "")) == pend.direction)
                    else:
                        dir_ok = True
                    if (not dir_ok) or (core_count_fill < core_min_fill):
                        # skip fill today; keep pending for future bars within horizon
                        continue
                except Exception:
                    # Fail-open on unexpected errors to avoid blocking fills due to evaluator issues
                    pass
            if len(open_positions) >= max_open or (_current_total_risk() >= max_risk_pct):
                continue
            wk = _week_key(day)
            if max_new_week > 0 and week_new_count[wk] >= max_new_week:
                continue
            cfg_day = copy.deepcopy(cfg)
            cfg_day["risk"]["use_broker_equity"] = False
            cfg_day["risk"]["account_equity"] = float(equity)
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
            qty_mgr = rm_size_position(cfg_day, entry_px, stop)
            # Cap by available cash buying power only
            qty_bp = _qty_bp_cap(entry_px, cash, bp_multiple)
            qty = min(qty_mgr, qty_bp)
            if qty <= 0:
                continue
            psr = abs(entry_px - stop)
            new_total = (_current_total_risk() * max(equity, 1e-9) + psr * qty) / max(equity, 1e-9)
            if new_total > max_risk_pct:
                continue
            # Derive cap reason (risk vs BP)
            try:
                risk_dollars = float(cfg_day.get("risk", {}).get("risk_per_trade_pct", 0.0)) * float(equity)
                base_qty = int(math.floor(risk_dollars / max(psr, 1e-12))) if risk_dollars > 0 else 0
            except Exception:
                base_qty = 0
            cap_reason = "full_size"
            if qty_mgr < base_qty:
                cap_reason = "risk_cap"
            elif qty < qty_mgr:
                cap_reason = "bp_cap"
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
                "order_type": "limit",
            })
            week_new_count[wk] += 1
            pending_limits.remove(pend)
            pos_created = open_positions[pend.symbol]
            pos_created.atr_pct_entry = float(pend.atr / entry_px) if entry_px > 0 else np.nan
            pos_created.target_R_at_entry = (abs(pos_created.target - pos_created.entry_price) / pos_created.per_share_risk) if pos_created.per_share_risk > 0 else np.nan
            # Limit retrace assumed pullback setup
            pos_created.setup_type = getattr(pend, "setup_type", "pullback")
            pos_created.qty_cap_reason = cap_reason
            # Entry diagnostics
            pos_created.entry_setup_type = pos_created.setup_type
            try:
                pos_created.entry_adx14 = float(df.at[day, "ADX14"]) if "ADX14" in df.columns else float("nan")
            except Exception:
                pos_created.entry_adx14 = float("nan")
            pos_created.entry_rvol20 = _compute_rvol20(df, day)
            pos_created.entry_ema20_dist_pct = getattr(pos_created, "ema20_dist_pct", float("nan"))
            pos_created.entry_pivot_dist_pct = getattr(pos_created, "pivot_dist_pct", float("nan"))
            pos_created.entry_atr_pct = getattr(pos_created, "atr_pct_entry", float("nan"))
            # MTF state at fill
            try:
                idx_fill = df.index.get_loc(day)
                wctx_fill = _wcache.get((pend.symbol, idx_fill))
                if wctx_fill is None:
                    wctx_fill = weekly_context(df.iloc[: idx_fill + 1])
                    _wcache[(pend.symbol, idx_fill)] = wctx_fill
                pos_created.entry_mtf_state = _derive_mtf_state_from_ctx(wctx_fill)
            except Exception:
                pos_created.entry_mtf_state = ""
            # days to fill for limit (0 if same day)
            try:
                pos_created.days_to_fill = float((day - pend.signal_date).days)
            except Exception:
                pos_created.days_to_fill = float("nan")
            pos_created.order_type = "limit"
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
                    # Entry diag copy
                    "entry_setup_type": getattr(pos, "entry_setup_type", pos.setup_type),
                    "entry_adx14": getattr(pos, "entry_adx14", np.nan),
                    "entry_rvol20": getattr(pos, "entry_rvol20", np.nan),
                    "entry_ema20_dist_pct": getattr(pos, "entry_ema20_dist_pct", np.nan),
                    "entry_pivot_dist_pct": getattr(pos, "entry_pivot_dist_pct", np.nan),
                    "entry_atr_pct": getattr(pos, "entry_atr_pct", np.nan),
                    "entry_mtf_state": getattr(pos, "entry_mtf_state", ""),
                    "qty_cap_reason": getattr(pos, "qty_cap_reason", "full_size"),
                    "order_type": getattr(pos, "order_type", "limit"),
                    "days_to_fill": getattr(pos, "days_to_fill", np.nan),
                    # Lifecycle
                    "time_to_BE_bars": getattr(pos, "time_to_BE_bars", None),
                    "time_to_1R_bars": getattr(pos, "time_to_1R_bars", None),
                    "be_armed_bar": getattr(pos, "be_armed_bar", None),
                    "trail_active_bar": getattr(pos, "trail_active_bar", None),
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
                limit_signals_count += 1

        # 4) mark-to-market (once per day)
        mv = 0.0
        for sym, pos in open_positions.items():
            df = data.get(sym)
            if df is None:
                continue
            if day in df.index:
                px = float(df.at[day, "Close"]) if np.isfinite(df.at[day, "Close"]) else np.nan
            else:
                # fallback to last available close <= day to avoid zeroing MV on missing bars
                prior = df.index[df.index <= day]
                if len(prior) > 0:
                    ld = prior[-1]
                    px = float(df.at[ld, "Close"]) if np.isfinite(df.at[ld, "Close"]) else np.nan
                else:
                    px = np.nan
            if np.isfinite(px):
                mv += pos.market_value(px)
        equity_today = cash + mv
        equity_curve.append((day, equity_today))
        equity = float(equity_today)
        # Daily metrics snapshot
        try:
            daily_metrics.append({
                "date": day,
                "equity": equity_today,
                "open_positions_count": int(len(open_positions)),
                "risk_utilization_pct": float(_current_total_risk()),
                "cash": float(cash),
            })
        except Exception:
            pass

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
        # Avoid chained-assignment; assign the filled series back to the DataFrame
        audit_qty["exit_qty"] = audit_qty["exit_qty"].fillna(0)
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

    # ---- Drawdown forensics: top episodes with attribution ----
    def _drawdown_episodes(eq: pd.Series, top_n: int = 3):
        if eq.empty:
            return []
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        troughs = dd.nsmallest(min(len(dd), top_n * 5))
        episodes = []
        used: List[pd.Timestamp] = []
        for tr_idx, depth in troughs.items():
            if any(abs((tr_idx - u).days) <= 5 for u in used):
                continue
            prior = eq.loc[:tr_idx]
            pk_val = float(prior.cummax().iloc[-1])
            pk_idx = prior[prior == pk_val].index[-1]
            after = eq.loc[tr_idx:]
            rc_idx = after[after >= pk_val].index.min() if (after >= pk_val).any() else after.index[-1]
            episodes.append((pk_idx, tr_idx, rc_idx, float(depth)))
            used.append(tr_idx)
            if len(episodes) >= top_n:
                break
        return episodes

    dd_eps = _drawdown_episodes(eq_df["equity"] if not eq_df.empty else pd.Series(dtype=float), top_n=3)
    dd_rows = []
    if not trades_df.empty:
        if "exit_date" in trades_df.columns:
            trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"], errors="coerce")
    for (pk, tr, rc, depth) in dd_eps:
        if not trades_df.empty and "exit_date" in trades_df.columns:
            seg_trades = trades_df[(trades_df["exit_date"] > pk) & (trades_df["exit_date"] <= rc)]
        else:
            seg_trades = pd.DataFrame()
        gross_loss_seg = float(seg_trades.loc[seg_trades["pnl"] < 0, "pnl"].sum()) if (not seg_trades.empty and "pnl" in seg_trades.columns) else 0.0
        worst = seg_trades.nsmallest(1, "pnl") if (not seg_trades.empty and "pnl" in seg_trades.columns) else pd.DataFrame()
        worst_sym = worst["symbol"].iloc[0] if not worst.empty and "symbol" in worst.columns else ""
        worst_reason = worst["reason"].iloc[0] if not worst.empty and "reason" in worst.columns else ""
        dd_rows.append({
            "peak_date": pk, "trough_date": tr, "recovery_date": rc,
            "depth_pct": depth, "gross_loss": gross_loss_seg,
            "worst_symbol": worst_sym, "worst_reason": worst_reason,
            "trades_in_window": int(len(seg_trades))
        })
    dd_df = pd.DataFrame(dd_rows)
    if not dd_df.empty:
        if log_drawdowns_csv:
            dd_path = today_filename("backtest_drawdowns", unique=True)
            log_dataframe(dd_df, dd_path)
        if print_rollups:
            print("\nTop drawdowns:")
            print(dd_df.assign(depth_pct=lambda d: (d["depth_pct"] * 100).round(2)).to_string(index=False))

    # write CSVs
    eval_df = pd.DataFrame(evaluations)
    # Use unique timestamped filenames to avoid cross-run contamination
    eval_path = today_filename("backtest_evaluations", unique=True)
    if log_eval_csv and not eval_df.empty:
        eval_df = eval_df.sort_values(["date", "symbol"])
        log_dataframe(eval_df, eval_path)

    trades_path = today_filename("backtest_trades", unique=True)
    equity_path = today_filename("backtest_equity", unique=True)
    daily_metrics_path = today_filename("backtest_daily_metrics", unique=True)
    if not trades_df.empty:
        log_dataframe(trades_df, trades_path, overwrite=True)
    log_dataframe(eq_df.reset_index(), equity_path, overwrite=True)
    if log_daily_csv and len(daily_metrics) > 0:
        log_dataframe(pd.DataFrame(daily_metrics), daily_metrics_path, overwrite=True)

    # ---- Additional diagnostics ----
    exit_reason_counts = trades_df["reason"].value_counts().to_dict() if not trades_df.empty else {}
    scale_out_count = int(exit_reason_counts.get("scale_out", 0))
    pre_earnings_exits = int(exit_reason_counts.get("pre_earnings", 0))
    runner_exits = trades_df[trades_df["part"] == "runner"] if not trades_df.empty and "part" in trades_df.columns else pd.DataFrame()
    runner_target_wins = int(len(runner_exits[runner_exits["reason"].isin(["target","force_end"]) & (runner_exits["pnl"]>0)])) if not runner_exits.empty else 0
    time_stop_exits = int(exit_reason_counts.get("time_stop", 0))
    stop_exits = int(exit_reason_counts.get("stop", 0) + exit_reason_counts.get("stop_same_day", 0))
    target_full_exits = int(exit_reason_counts.get("target", 0) + exit_reason_counts.get("target_same_day", 0))
    median_bars_held = float(trades_df["bars_held"].median()) if (not trades_df.empty and "bars_held" in trades_df.columns) else 0.0
    median_mfe_R_wins = float(trades_df.loc[trades_df["pnl"]>0, "mfe_R"].median()) if (not trades_df.empty and "mfe_R" in trades_df.columns and len(trades_df[trades_df["pnl"]>0])>0) else 0.0
    median_mfe_R_losses = float(trades_df.loc[trades_df["pnl"]<=0, "mfe_R"].median()) if (not trades_df.empty and "mfe_R" in trades_df.columns and len(trades_df[trades_df["pnl"]<=0])>0) else 0.0

    # Pending/fill diagnostics
    limit_filled = int(entries_df[entries_df.get("order_type","market") == "limit"].shape[0]) if not entries_df.empty else 0
    market_fills = int(entries_df[entries_df.get("order_type","market") == "market"].shape[0]) if not entries_df.empty else 0
    avg_days_to_fill = float(trades_df["days_to_fill"].mean(skipna=True)) if (not trades_df.empty and "days_to_fill" in trades_df.columns) else float("nan")

    # Rollups
    def _safe_rate(n, d):
        return float(n / d) if d else 0.0
    same_day_exit_rate = _safe_rate(int(trades_df[trades_df["reason"].isin(["stop_same_day","target_same_day"])].shape[0]) if not trades_df.empty else 0, int(len(trades_df)))
    one_bar_exit_rate = _safe_rate(int(trades_df[trades_df.get("bars_held", 0) <= 1].shape[0]) if not trades_df.empty else 0, int(len(trades_df)))
    # Buckets
    def _bucket_adx(x):
        try:
            if not np.isfinite(x):
                return "nan"
            if x < 18: return "<18"
            if x < 25: return "18-25"
            return ">=25"
        except Exception:
            return "nan"
    def _bucket_atr_pct(x):
        try:
            if not np.isfinite(x):
                return "nan"
            if x < 0.01: return "<1%"
            if x < 0.015: return "1-1.5%"
            return ">=1.5%"
        except Exception:
            return "nan"

    if not trades_df.empty:
        roll = {}
        # by exit reason
        roll["by_exit_reason"] = trades_df.groupby("reason")["pnl"].agg(["count","sum"]).reset_index().to_dict(orient="records")
        # by entry setup
        if "entry_setup_type" in trades_df.columns:
            roll["by_setup"] = trades_df.groupby("entry_setup_type")["pnl"].agg(["count","sum"]).reset_index().to_dict(orient="records")
        # by MTF state
        if "entry_mtf_state" in trades_df.columns:
            roll["by_mtf"] = trades_df.groupby("entry_mtf_state")["pnl"].agg(["count","sum"]).reset_index().to_dict(orient="records")
        # by ADX bucket (use entry_adx14)
        if "entry_adx14" in trades_df.columns:
            tmp = trades_df.copy()
            tmp["adx_bucket"] = tmp["entry_adx14"].apply(_bucket_adx)
            roll["by_adx_bucket"] = tmp.groupby("adx_bucket")["pnl"].agg(["count","sum"]).reset_index().to_dict(orient="records")
        # by ATR% bucket (use entry_atr_pct)
        if "entry_atr_pct" in trades_df.columns:
            tmp = trades_df.copy()
            tmp["atr_bucket"] = tmp["entry_atr_pct"].apply(_bucket_atr_pct)
            roll["by_atr_bucket"] = tmp.groupby("atr_bucket")["pnl"].agg(["count","sum"]).reset_index().to_dict(orient="records")
    else:
        roll = {}

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
    "pre_earnings_exits": int(pre_earnings_exits),
    "median_bars_held": median_bars_held,
    "median_mfe_R_wins": median_mfe_R_wins,
    "median_mfe_R_losses": median_mfe_R_losses,
    # pending / fills
    "limit_signals": int(limit_signals_count),
    "limit_filled": int(limit_filled),
    "limit_expired": int(limit_expired_count),
    "avg_days_to_fill": float(avg_days_to_fill),
    "market_fills": int(market_fills),
    # rates
    "same_day_exit_rate": float(same_day_exit_rate),
    "one_bar_exit_rate": float(one_bar_exit_rate),
    # rollups
    "rollups": roll,
    "files": [str(trades_path.name), str(equity_path.name)],
    }

    # Add worst-episode markers into summary if computed
    try:
        if 'dd_df' in locals() and not dd_df.empty:
            worst = dd_df.nsmallest(1, "depth_pct").iloc[0]
            summary["max_dd_start"] = worst["peak_date"]
            summary["max_dd_trough"] = worst["trough_date"]
            summary["max_dd_depth"] = float(worst["depth_pct"])
    except Exception:
        pass

    print("\n— Backtest Summary —")
    for k in ("start_equity","end_equity","total_return","CAGR","max_drawdown","daily_sharpe","days","trades","win_rate","avg_win","avg_loss","profit_factor","profit_factor_net","avg_R",
              "stop_exits","target_exits","time_stop_exits","scale_out_trades","runner_exits","runner_target_wins","pre_earnings_exits","median_bars_held","median_mfe_R_wins","median_mfe_R_losses"):
        v = summary[k]
        if k in {"total_return","CAGR","max_drawdown","win_rate"}:
            if isinstance(v, float) and not np.isnan(v):
                print(f"{k:<15}: {v*100:.2f}%")
            else:
                print(f"{k:<15}: N/A")
        else:
            print(f"{k:<15}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"{k:<15}: {v}")

    # Reconciliation print
    if print_recon:
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

    if print_topbottom and not trades_df.empty:
        t5 = trades_df.sort_values("r_multiple", ascending=False).head(5)
        b5 = trades_df.sort_values("r_multiple", ascending=True).head(5)
        print("\nTop 5 trades by R:")
        print(t5[["symbol","side","entry_date","entry_price","exit_date","exit_price","qty","pnl","r_multiple","commission","slippage_cost"]].to_string(index=False))
        print("\nBottom 5 trades by R:")
        print(b5[["symbol","side","entry_date","entry_price","exit_date","exit_price","qty","pnl","r_multiple","commission","slippage_cost"]].to_string(index=False))

    print("\nFiles written:")
    if log_eval_csv and not eval_df.empty:
        print(f"  - {eval_path.name}")
    print(f"  - {trades_path.name}")
    print(f"  - {equity_path.name}")
    if log_daily_csv and len(daily_metrics) > 0:
        print(f"  - {daily_metrics_path.name}")
        summary["files"].append(str(daily_metrics_path.name))

    # Reproducibility snapshot
    try:
        cfg_snap_path = today_filename("config_used", unique=True).with_suffix(".yaml")
        if log_meta_files:
            with open(cfg_snap_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
        from strategies import swing_strategy as _ss
        ss_path = Path(_ss.__file__)
        engine_hash = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
        swing_hash = hashlib.sha1(ss_path.read_bytes()).hexdigest()[:12]
        meta = {
            "universe_size": int(len(tickers)),
            "date_start": str(pd.to_datetime(start_date).date()),
            "date_end": str(pd.to_datetime(end_date).date()),
            "timezone": str(eq_df.index.tz) if hasattr(eq_df.index, "tz") else "naive",
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python": _sys.version.split(" ")[0],
            "platform": _platform.platform(),
            "engine_sha": engine_hash,
            "swing_strategy_sha": swing_hash,
        }
        # fundamentals diagnostics
        try:
            fcfg_meta = (cfg.get("fundamentals", {}) or {})
            btcfg_meta = (fcfg_meta.get("backtest_mode", {}) or {})
            if fund_ctx is not None:
                cal = fund_ctx.get("earnings_calendar", {}) or {}
                total_events = int(sum(len(v or []) for v in cal.values()))
                meta["fundamentals"] = {
                    "enabled": bool(fcfg_meta.get("enabled", False)),
                    "network_mode": str(btcfg_meta.get("network", "prefetch_only")),
                    "symbols_prefetched": int(len(cal)),
                    "earnings_events": total_events,
                    "request_counter": fund_ctx.get("request_counter", {}),
                }
        except Exception:
            pass
        meta_path = today_filename("run_meta", unique=True).with_suffix(".yaml")
        if log_meta_files:
            with open(meta_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(meta, f, sort_keys=False)
            print(f"  - {cfg_snap_path.name}")
            print(f"  - {meta_path.name}")
            summary["files"].extend([str(cfg_snap_path.name), str(meta_path.name)])
        summary["meta"] = meta
    except Exception:
        pass
    # Print grouped rollups and pending/fill stats
    if print_pending:
        print("\nPending/fill stats:")
        print(f"  limit_signals        : {summary['limit_signals']}")
        print(f"  limit_filled         : {summary['limit_filled']}")
        print(f"  limit_expired        : {summary['limit_expired']}")
        print(f"  avg_days_to_fill     : {summary['avg_days_to_fill']:.2f}" if np.isfinite(summary['avg_days_to_fill']) else "  avg_days_to_fill     : N/A")
        print(f"  market_fills         : {summary['market_fills']}")
        if (earnings_block_market + earnings_block_limit_skip + earnings_block_scan) > 0:
            print("  fundamentals blocks  :")
            if earnings_block_scan > 0:
                print(f"    earnings_blocked_scan   : {earnings_block_scan}")
            print(f"    earnings_cancelled_market : {earnings_block_market}")
            print(f"    earnings_skipped_limit    : {earnings_block_limit_skip}")
        print("\nRates:")
        print(f"  same_day_exit_rate   : {summary['same_day_exit_rate']*100:.2f}%")
        print(f"  one_bar_exit_rate    : {summary['one_bar_exit_rate']*100:.2f}%")
    if print_rollups and summary.get("rollups"):
        print("\nRollups:")
        for key, rows in summary["rollups"].items():
            try:
                df_print = pd.DataFrame(rows)
                if not df_print.empty:
                    print(f"  {key}:")
                    cols = [c for c in ["reason","entry_setup_type","entry_mtf_state","adx_bucket","atr_bucket","count","sum"] if c in df_print.columns]
                    print(df_print[cols].to_string(index=False))
            except Exception:
                pass
    return summary
