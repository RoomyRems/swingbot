# backtest/engine.py
from __future__ import annotations
from dataclasses import dataclass
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

def _qty_bp_cap(entry_px: float, equity: float, bp_multiple: float) -> int:
    if entry_px <= 0 or equity <= 0 or bp_multiple <= 0:
        return 0
    return int(math.floor((equity * bp_multiple) / entry_px))

@dataclass
class PendingLimit:
    symbol: str
    direction: str   # "long"/"short"
    signal_date: pd.Timestamp
    expires: pd.Timestamp
    limit_px: float
    atr: float       # <-- keep ATR so we can compute levels at fill
    score: int

@dataclass
class PendingMarket:
    symbol: str
    direction: str    # "long"/"short"
    fill_date: pd.Timestamp
    atr: float
    score: int

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

def _mtf_ok_for_slice(dfslice: pd.DataFrame, cfg: dict, direction: str) -> Tuple[bool, str]:
    mtf_cfg = (cfg.get("trading", {}).get("mtf") or {})
    if not bool(mtf_cfg.get("enabled", False)):
        return True, "mtf-disabled"
    try:
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

    allow_short = bool(bt_cfg.get("allow_short", True))
    slip_bps = float(bt_cfg.get("slippage_bps", 0.0))
    commission_ps = float(bt_cfg.get("commission_per_share", 0.0))

    # BREAKEVEN
    be_R = float(bt_cfg.get("breakeven_at_R", 0.0))
    be_intrabar = str(bt_cfg.get("breakeven_intrabar", "favor_be")).lower()

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
    for sym, df in tqdm(data.items(), desc="Scan signals", leave=False):
        mask = (df.index >= start_date) & (df.index <= end_date)
        for d in df.index[mask]:
            i = df.index.get_loc(d)
            if i < 60:
                continue
            sl = df.iloc[: i + 1]
            row = sl.iloc[-1]

            # Regime
            regime_ok, _ = _regime_check(sl, cfg)

            # Energies/components
            eng = evaluate_five_energies(sl, cfg) or {}
            direction = eng.get("direction")
            score = int(eng.get("score", 0))
            trend_ok = bool(eng.get("trend", False))
            mom_ok   = bool(eng.get("momentum", False))
            cycle_ok = bool(eng.get("cycle", False))
            sr_ok    = bool(eng.get("sr", False))
            fractal_ok = bool(eng.get("fractal", False))
            vol_ok   = bool(eng.get("volume", False))  # confirmation only

            # MTF bonus
            if direction in ("long", "short"):
                mtf_ok, _mtf_reason = _mtf_ok_for_slice(sl, cfg, direction)
            else:
                mtf_ok, _mtf_reason = False, "no-dir"
            mtf_cfg = (cfg.get("trading", {}).get("mtf") or {})
            mtf_counts = bool(mtf_cfg.get("counts_as_energy", True))
            core_min = int(cfg.get("trading", {}).get("min_core_energies", cfg.get("trading", {}).get("min_score", 4)))
            # enforce core gate before bonus
            score_eff = score + (1 if (mtf_counts and mtf_ok) else 0)

            min_score = int(cfg["trading"]["min_score"])  # effective threshold (post bonus)
            allowed_dir = (direction != "short") or allow_short
            mtf_cfg = (cfg.get("trading", {}).get("mtf") or {})
            mtf_veto = bool(mtf_cfg.get("reject_on_mismatch", False))
            accept = regime_ok and (direction in ("long", "short")) and (score >= core_min) and (score_eff >= min_score) and allowed_dir
            if accept and mtf_veto and (not mtf_ok):
                accept = False
                reason = "mtf_mismatch"

            # First failing reason
            if not regime_ok:
                reason = "regime"
            elif direction not in ("long","short"):
                reason = "no_dir"
            elif mtf_veto and not mtf_ok:
                reason = "mtf_mismatch"
            elif score < core_min:
                reason = "below_core_min"
            elif score_eff < min_score:
                reason = "below_score"
            elif not allowed_dir:
                reason = "short_blocked"
            else:
                reason = "pass"

            accept = (reason == "pass")

            # Evaluation row
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
                "fractal": fractal_ok,
                "volume_confirm": vol_ok,
                "regime_ok": bool(regime_ok),
                "mtf_ok": bool(mtf_ok),
                "reject_reason": reason,
                "close": float(row.get("Close", np.nan)),
                "ema20": float(row.get("EMA20", np.nan)),
                    "atr14": float(row.get("ATR14", np.nan)),
                    "adx14": float(row.get("ADX14", np.nan)) if "ADX14" in row.index else np.nan,
            })

            if accept:
                candidates_by_day[d].append({
                    "symbol": sym,
                    "date": d,
                    "direction": direction,
                    "score": int(score),
                    "score_eff": int(score_eff),
                    "close": float(row["Close"]),
                    "ema20": float(row.get("EMA20", np.nan)),
                    "atr": float(row.get("ATR14", np.nan)),
                })

    # --- DIAGNOSTIC ---
    total_sigs = sum(len(v) for v in candidates_by_day.values())
    uniq_syms = len({s["symbol"] for v in candidates_by_day.values() for s in v})
    from tqdm import tqdm as _tqdm
    _tqdm.write(f"Found {total_sigs} daily signal candidates across the test window from {uniq_syms} symbols.")

    # ---------- portfolio state ----------
    start_equity_cfg = float(bt_cfg.get("initial_equity", cfg.get("risk", {}).get("account_equity", 50000)))
    equity = float(start_equity_cfg)
    cash = equity
    open_positions: Dict[str, Position] = {}
    pending_limits: List[PendingLimit] = []
    pending_markets: List[PendingMarket] = []
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
    for day in tqdm(calendar, desc="Backtest", leave=True):
        # 1) exits first
        for sym, pos in list(open_positions.items()):
            df = data.get(sym)
            if df is None or day not in df.index:
                continue
            o = float(df.at[day, "Open"])
            h = float(df.at[day, "High"])
            l = float(df.at[day, "Low"])
            c = float(df.at[day, "Close"])

            # BREAKEVEN
            orig_stop = pos.stop
            if be_R > 0 and not pos.be_armed:
                if pos.side == "long":
                    be_level = pos.entry_price + be_R * pos.per_share_risk
                    if h >= be_level:
                        if not (be_intrabar == "favor_stop" and l <= orig_stop):
                            pos.stop = max(pos.stop, pos.entry_price)
                            pos.be_armed = True
                else:
                    be_level = pos.entry_price - be_R * pos.per_share_risk
                    if l <= be_level:
                        if not (be_intrabar == "favor_stop" and h >= orig_stop):
                            pos.stop = min(pos.stop, pos.entry_price)
                            pos.be_armed = True

            exit_reason = None
            fill_px = None

            if pos.side == "long":
                if l <= pos.stop:
                    fill_px = _gap_exit_price("long", o, pos.stop)
                    exit_reason = "stop"
                elif h >= pos.target:
                    fill_px = pos.target if o <= pos.target else o
                    exit_reason = "target"
            else:
                if h >= pos.stop:
                    fill_px = _gap_exit_price("short", o, pos.stop)
                    exit_reason = "stop"
                elif l <= pos.target:
                    fill_px = pos.target if o >= pos.target else o
                    exit_reason = "target"

            if exit_reason:
                slip = (slip_bps / 10000.0) * fill_px
                fill_px_eff = fill_px - slip if pos.side == "long" else fill_px + slip
                commission = commission_ps * pos.qty
                if pos.side == "long":
                    cash += pos.qty * fill_px_eff
                else:
                    cash -= pos.qty * fill_px_eff
                cash -= commission

                pnl = (fill_px_eff - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - fill_px_eff) * pos.qty
                trades.append({
                    "symbol": sym,
                    "side": pos.side,
                    "entry_date": pos.entry_date,
                    "entry_price": pos.entry_price,
                    "exit_date": day,
                    "exit_price": fill_px_eff,
                    "qty": pos.qty,
                    "pnl": pnl,
                    "r_multiple": pnl / (pos.per_share_risk * pos.qty) if pos.per_share_risk > 0 else np.nan,
                    "commission": commission_ps * pos.qty * 2.0,
                    "slippage_cost": slip * pos.qty,
                    "reason": exit_reason,
                })
                del open_positions[sym]
        # 1a) fill any scheduled MARKET entries for today at the open
        for pend in list(pending_markets):
            if pend.fill_date != day:
                continue

            df = data.get(pend.symbol)
            if df is None or day not in df.index:
                pending_markets.remove(pend)
                continue

            # caps before sizing
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

            # compute levels at actual entry
            stop, target = rm_compute_levels(
                direction=pend.direction,
                entry=entry_px,
                atr=pend.atr,
                atr_mult=float(cfg["risk"]["atr_multiple_stop"]),
                reward_mult=float(cfg["risk"]["reward_multiple"]),
            )
            if stop is None or target is None:
                pending_markets.remove(pend)
                continue

            # per-day cfg and sizing
            cfg_day = copy.deepcopy(cfg)
            cfg_day["risk"]["use_broker_equity"] = False
            cfg_day["risk"]["account_equity"] = float(equity)

            qty = rm_size_position(cfg_day, entry=entry_px, stop=stop)
            qty_bp = _qty_bp_cap(entry_px, equity, bp_multiple)
            qty = min(qty, qty_bp)
            if qty <= 0:
                pending_markets.remove(pend)
                continue

            # portfolio risk cap
            psr = abs(entry_px - stop)
            new_total_risk = (_current_total_risk() * max(equity, 1e-9) + psr * qty) / max(equity, 1e-9)
            if new_total_risk > max_risk_pct:
                pending_markets.remove(pend)
                continue

            commission = commission_ps * qty
            if pend.direction == "long":
                cash -= qty * entry_px
            else:
                cash += qty * entry_px
            cash -= commission

            open_positions[pend.symbol] = Position(
                symbol=pend.symbol,
                side=pend.direction,
                entry_date=day,
                entry_price=entry_px,
                qty=qty,
                stop=stop,
                target=target,
                per_share_risk=psr,
                score=pend.score,
                be_armed=False,
            )

            # Immediately check same-day exit for newly opened market positions
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
                slip_exit = (slip_bps / 10000.0) * fill_px
                fill_px_eff = fill_px - slip_exit if pos.side == "long" else fill_px + slip_exit
                commission = commission_ps * pos.qty
                if pos.side == "long":
                    cash += pos.qty * fill_px_eff
                else:
                    cash -= pos.qty * fill_px_eff
                cash -= commission

                pnl = (fill_px_eff - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - fill_px_eff) * pos.qty
                trades.append({
                    "symbol": pend.symbol, "side": pos.side,
                    "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                    "exit_date": day, "exit_price": fill_px_eff, "qty": pos.qty,
                    "pnl": pnl,
                    "r_multiple": pnl / (pos.per_share_risk * pos.qty) if pos.per_share_risk > 0 else np.nan,
                    "commission": commission_ps * pos.qty * 2.0,
                    "slippage_cost": slip_exit * pos.qty,
                    "reason": exit_reason,
                })
                del open_positions[pend.symbol]



            week_new_count[wk] += 1
            pending_markets.remove(pend)



        # 2) try to place/fill pending limit orders
        for pend in list(pending_limits):
            if equity <= equity_halt_floor:
                pending_limits.clear()
                break
            df = data.get(pend.symbol)
            if df is None or day < pend.signal_date or day > pend.expires or day not in df.index:
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

            # per-day cfg: override equity (no broker BP)
            cfg_day = copy.deepcopy(cfg)
            cfg_day["risk"]["use_broker_equity"] = False
            cfg_day["risk"]["account_equity"] = float(equity)

            # effective entry with slippage
            raw_entry = min(o, pend.limit_px) if pend.direction == "long" else max(o, pend.limit_px)
            slip = (slip_bps / 10000.0) * raw_entry
            entry_px = raw_entry + slip if pend.direction == "long" else raw_entry - slip

            stop, target = rm_compute_levels(
                direction=pend.direction,
                entry=entry_px,
                atr=pend.atr,                                  # <-- use stored ATR
                atr_mult=float(cfg["risk"]["atr_multiple_stop"]),
                reward_mult=float(cfg["risk"]["reward_multiple"]),
            )
            if stop is None or target is None:
                continue

            # size from live helper, then apply BP cap
            qty = rm_size_position(cfg_day, entry=entry_px, stop=stop)
            qty_bp = _qty_bp_cap(entry_px, equity, bp_multiple)
            qty = min(qty, qty_bp)
            if qty <= 0:
                continue

            # portfolio risk cap check
            psr = abs(entry_px - stop)
            new_total_risk = (_current_total_risk() * max(equity, 1e-9) + psr * qty) / max(equity, 1e-9)
            if new_total_risk > max_risk_pct:
                continue

            commission = commission_ps * qty
            if pend.direction == "long":
                cash -= qty * entry_px
            else:
                cash += qty * entry_px
            cash -= commission

            open_positions[pend.symbol] = Position(
                symbol=pend.symbol,
                side=pend.direction,
                entry_date=day,
                entry_price=entry_px,
                qty=qty,
                stop=stop,
                target=target,
                per_share_risk=psr,
                score=pend.score,
                be_armed=False,
            )
            week_new_count[wk] += 1
            pending_limits.remove(pend)

            # Immediately check same-day exit for newly filled limit orders
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
                slip_exit = (slip_bps / 10000.0) * fill_px
                fill_px_eff = fill_px - slip_exit if pos.side == "long" else fill_px + slip_exit
                commission = commission_ps * pos.qty
                if pos.side == "long":
                    cash += pos.qty * fill_px_eff
                else:
                    cash -= pos.qty * fill_px_eff
                cash -= commission

                pnl = (fill_px_eff - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - fill_px_eff) * pos.qty
                trades.append({
                    "symbol": pend.symbol, "side": pos.side,
                    "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                    "exit_date": day, "exit_price": fill_px_eff, "qty": pos.qty,
                    "pnl": pnl,
                    "r_multiple": pnl / (pos.per_share_risk * pos.qty) if pos.per_share_risk > 0 else np.nan,
                    "commission": commission_ps * pos.qty * 2.0,
                    "slippage_cost": slip_exit * pos.qty,
                    "reason": exit_reason,
                })
                del open_positions[pend.symbol]


        # 3) new signals for this day (respect caps; market enters at next open)
        day_candidates = sorted(
            candidates_by_day.get(day, []),
            key=lambda s: (-s.get("score_eff", s["score"]), s["close"])
        )

        for sig in day_candidates:
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

                pending_markets.append(PendingMarket(
                    symbol=sig["symbol"],
                    direction=sig["direction"],
                    fill_date=nd,
                    atr=float(sig["atr"]),
                    score=int(sig.get("score_eff", sig["score"])),
                ))


            elif entry_type == "limit_retrace":
                ema = sig["ema20"] if retrace_ref == "EMA20" else sig["close"]
                limit_px = (ema - atr_frac * sig["atr"]) if sig["direction"] == "long" else (ema + atr_frac * sig["atr"])
                stop, target = rm_compute_levels(
                    direction=sig["direction"],
                    entry=float(limit_px),
                    atr=sig["atr"],
                    atr_mult=float(cfg["risk"]["atr_multiple_stop"]),
                    reward_mult=float(cfg["risk"]["reward_multiple"]),
                )
                if stop is None or target is None:
                    continue

                pending_limits.append(PendingLimit(
                    symbol=sig["symbol"],
                    direction=sig["direction"],
                    signal_date=sig["date"],
                    expires=sig["date"] + pd.tseries.offsets.BDay(horizon),
                    limit_px=float(limit_px),
                    atr=float(sig["atr"]),        # <-- NEW
                    score=int(sig.get("score_eff", sig["score"])),
                ))

        # 4) mark-to-market equity for the day (close)
        mv = 0.0
        for sym, pos in open_positions.items():
            df = data.get(sym)
            if df is None or day not in df.index:
                continue
            c = float(df.at[day, "Close"])
            mv += pos.market_value(c)
        equity_today = cash + mv
        equity_curve.append((day, equity_today))
        equity = float(equity_today)

        # 5) Force-close all open positions on last day (so PnL reconciles)
        if force_close_on_end and (day == calendar[-1]) and open_positions:
            for sym, pos in list(open_positions.items()):
                df = data[sym]
                if day not in df.index:
                    continue
                c = float(df.at[day, "Close"])
                slip = (slip_bps / 10000.0) * c
                fill_px_eff = c - slip if pos.side == "long" else c + slip
                commission = commission_ps * pos.qty
                if pos.side == "long":
                    cash += pos.qty * fill_px_eff
                else:
                    cash -= pos.qty * fill_px_eff
                cash -= commission
                pnl = (fill_px_eff - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - fill_px_eff) * pos.qty
                trades.append({
                    "symbol": sym,
                    "side": pos.side,
                    "entry_date": pos.entry_date,
                    "entry_price": pos.entry_price,
                    "exit_date": day,
                    "exit_price": fill_px_eff,
                    "qty": pos.qty,
                    "pnl": pnl,
                    "r_multiple": pnl / (pos.per_share_risk * pos.qty) if pos.per_share_risk > 0 else np.nan,
                    "commission": commission_ps * pos.qty * 2.0,
                    "slippage_cost": slip * pos.qty,
                    "reason": "force_end",
                })
                del open_positions[sym]
            # mark equity after forced liquidation
            equity_today = cash
            equity_curve[-1] = (day, equity_today)
            equity = float(equity_today)

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
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(wins) / len(trades_df) if len(trades_df) else 0.0
        avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0
        profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and abs(losses["pnl"].sum()) > 0 else np.inf
        avg_R = float(trades_df["r_multiple"].mean(skipna=True))
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_R = 0.0

    # write CSVs
    eval_df = pd.DataFrame(evaluations)
    eval_path = today_filename("backtest_evaluations")
    if not eval_df.empty:
        eval_df = eval_df.sort_values(["date", "symbol"])
        log_dataframe(eval_df, eval_path)

    trades_path = today_filename("backtest_trades")
    equity_path = today_filename("backtest_equity")
    if not trades_df.empty:
        log_dataframe(trades_df, trades_path)
    log_dataframe(eq_df.reset_index(), equity_path)

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
        "avg_R": float(avg_R),
        "files": [str(trades_path.name), str(equity_path.name)],
    }

    print("\n— Backtest Summary —")
    for k in ("start_equity","end_equity","total_return","CAGR","max_drawdown","daily_sharpe","days","trades","win_rate","avg_win","avg_loss","profit_factor","avg_R"):
        v = summary[k]
        if k in {"total_return","CAGR","max_drawdown","win_rate"}:
            if isinstance(v, float) and not np.isnan(v):
                print(f"{k:<15}: {v*100:.2f}%")
            else:
                print(f"{k:<15}: N/A")
        else:
            print(f"{k:<15}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"{k:<15}: {v}")

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
