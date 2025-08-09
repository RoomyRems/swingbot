# backtest/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

# Reuse your live logic so the rules match exactly
from strategies.swing_strategy import (
    add_indicators,
    evaluate_five_energies,
    _regime_check,
    _resample_weekly_ohlcv,
    _weekly_trend_view,
    _mtf_alignment_ok,
)
from risk.manager import compute_levels  # we'll size inside the backtester

# ---------- Data containers ----------

@dataclass
class Position:
    symbol: str
    side: str            # "long" or "short"
    qty: int
    entry_date: pd.Timestamp
    entry_price: float
    stop: float
    target: float
    planned_r_per_share: float   # per-share risk based on planned entry (used for R calc)
    reward_multiple: float

@dataclass
class Trade:
    symbol: str
    side: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    qty: int
    pnl: float
    r_multiple: float
    commission: float
    slippage_cost: float


# ---------- Helpers ----------

def _cfg_backtest(cfg: dict, key: str, default):
    return ((cfg or {}).get("backtest") or {}).get(key, default)

def _slip(price: float, action: str, bps: float) -> Tuple[float, float]:
    """
    Apply slippage in basis points (e.g., 2 bps = 0.02%).
    action: "buy" or "sell" (we assume worse price for you).
    Returns (slipped_price, slippage_cost_per_share).
    """
    frac = float(bps) / 10_000.0
    if frac <= 0:
        return price, 0.0
    if action == "buy":
        p2 = price * (1.0 + frac)
        return p2, p2 - price
    else:  # sell
        p2 = price * (1.0 - frac)
        return p2, price - p2

def _size_qty(current_equity: float, risk_per_trade_pct: float, per_share_risk: float, min_shares: int = 1) -> int:
    if per_share_risk <= 0 or current_equity <= 0 or risk_per_trade_pct <= 0:
        return 0
    raw = math.floor((current_equity * risk_per_trade_pct) / per_share_risk)
    return int(max(raw, int(min_shares)))

def _anchor_bracket_at_fill(planned_entry: float, stop: float, target: float, fill: float, side: str, reward_mult: float) -> Tuple[float, float]:
    """
    Re-anchor stop/target to the actual fill, preserving per-share risk and reward multiple.
    """
    if side == "long":
        per_risk = max(planned_entry - stop, 0.0)
        if per_risk <= 0:
            return stop, target
        new_stop = fill - per_risk
        new_target = fill + reward_mult * per_risk
        return new_stop, new_target
    else:
        per_risk = max(stop - planned_entry, 0.0)
        if per_risk <= 0:
            return stop, target
        new_stop = fill + per_risk
        new_target = fill - reward_mult * per_risk
        return new_stop, new_target

def _touch_priority_exit(bar_open: float, day_high: float, day_low: float, stop: float, target: float, side: str, touch_priority: str) -> Optional[Tuple[str, float]]:
    """
    Decide exit for a bracket given daily OHLC and stop/target.
    Returns (which, price) where which in {"stop","target"}, price is the level hit.
    Priority rules:
      1) gap at open through a level -> fill at open
      2) else both levels inside day range -> pick per 'touch_priority'
      3) else whichever is touched
    """
    # Gap at open
    if side == "long":
        if bar_open <= stop:
            return ("stop", bar_open)
        if bar_open >= target:
            return ("target", bar_open)
    else:  # short
        if bar_open >= stop:
            return ("stop", bar_open)
        if bar_open <= target:
            return ("target", bar_open)

    # Intraday touches
    hit_target = (day_high >= target) if side == "long" else (day_low <= target)
    hit_stop   = (day_low  <= stop)   if side == "long" else (day_high >= stop)

    if hit_target and hit_stop:
        if touch_priority == "target_first":
            return ("target", target)
        else:
            return ("stop", stop)
    elif hit_target:
        return ("target", target)
    elif hit_stop:
        return ("stop", stop)
    else:
        return None

def _trend_direction(row: pd.Series) -> str:
    if (row["EMA20"] > row["EMA50"]) and (row["Close"] > row["EMA50"]):
        return "long"
    if (row["EMA20"] < row["EMA50"]) and (row["Close"] < row["EMA50"]):
        return "short"
    return "none"


# ---------- Core backtest ----------

def run_backtest(
    universe: List[str],
    cfg: dict,
    load_bars: Callable[[str, int], pd.DataFrame],
) -> Dict[str, pd.DataFrame | Dict]:
    """
    Minimal but solid daily backtester:
      - Walk-forward per symbol
      - Signals evaluated EOD (day t), entries at next open (t+1)
      - Bracket exits using daily OHLC (gap logic + touch priority)
      - Risk-based sizing, commission, slippage, unlimited concurrent positions (v1)
    Outputs:
      - "trades": DataFrame of all closed trades
      - "equity": DataFrame of daily equity
      - "metrics": dict of summary stats
    """
    # --- Config defaults ---
    lookback_days     = int(_cfg_backtest(cfg, "start_days_ago", 300))
    initial_equity    = float(_cfg_backtest(cfg, "initial_equity", 50_000))
    commission_ps     = float(_cfg_backtest(cfg, "commission_per_share", 0.0))
    slippage_bps      = float(_cfg_backtest(cfg, "slippage_bps", 2.0))
    allow_short       = bool(_cfg_backtest(cfg, "allow_short", True))
    max_positions     = int(_cfg_backtest(cfg, "max_positions", 0))  # 0 = unlimited (v1 ignores anyway)
    max_new_pw        = int(_cfg_backtest(cfg, "max_new_trades_per_week", 0))  # v1 ignore
    entry_price_mode  = str(_cfg_backtest(cfg, "entry_price", "next_open")).lower()
    touch_priority    = str(_cfg_backtest(cfg, "touch_priority", "stop_first")).lower()

    # Risk params (mirror live)
    rcfg = (cfg.get("risk") or {})
    risk_per_trade_pct = float(rcfg.get("risk_per_trade_pct", 0.02))
    atr_multiple_stop  = float(rcfg.get("atr_multiple_stop", 1.5))
    reward_multiple    = float(rcfg.get("reward_multiple", 2.0))
    min_shares         = int(rcfg.get("min_shares", 1))

    # MTF config (mirror live)
    mtf_cfg = (cfg.get("trading", {}).get("mtf") or {})
    mtf_enabled = bool(mtf_cfg.get("enabled", False))

    # Warmup bars for indicators/regime
    min_bars = 60

    # Storage
    trades: List[Trade] = []
    positions: Dict[str, Position] = {}
    # equity series as dict[date] -> equity (update when trades close; carry forward later)
    equity_line: Dict[pd.Timestamp, float] = {}

    current_equity = initial_equity

    # --- Sim loop per symbol (v1: independent) ---
    for sym in tqdm(universe, desc="Backtest", leave=True):
        try:
            raw = load_bars(sym, lookback_days + min_bars + 5)  # tiny buffer
            if raw is None or raw.empty or len(raw) < (min_bars + 2):
                continue

            # Indicators once (safe; EMAs etc. don't peek forward)
            ind = add_indicators(raw.copy())

            # Rolling day index (up to second-last, since we enter on next day)
            for i in range(min_bars, len(ind) - 1):
                today = ind.index[i]
                nxt   = ind.index[i + 1]

                # ---------- 1) manage open position on 'today' bar ----------
                if sym in positions:
                    pos = positions[sym]
                    bar = ind.iloc[i]  # today's OHLC for exit checks
                    which = _touch_priority_exit(
                        bar_open = float(bar["Open"]),
                        day_high = float(bar["High"]),
                        day_low  = float(bar["Low"]),
                        stop     = float(pos.stop),
                        target   = float(pos.target),
                        side     = pos.side,
                        touch_priority = touch_priority,
                    )
                    if which is not None:
                        exit_kind, raw_px = which
                        action = "buy" if (pos.side == "short") else "sell"
                        fill_px, slip_cost_ps = _slip(raw_px, action, slippage_bps)

                        # PnL (before fees)
                        if pos.side == "long":
                            pnl = (fill_px - pos.entry_price) * pos.qty
                            r_ps = pos.entry_price - pos.stop  # planned risk
                        else:
                            pnl = (pos.entry_price - fill_px) * pos.qty
                            r_ps = pos.stop - pos.entry_price

                        r_ps = max(r_ps, 1e-8)
                        r_mult = pnl / (r_ps * pos.qty)

                        # Fees
                        commission = commission_ps * pos.qty
                        slippage_cost = slip_cost_ps * pos.qty

                        trades.append(Trade(
                            symbol=sym,
                            side=pos.side,
                            entry_date=pos.entry_date,
                            entry_price=pos.entry_price,
                            exit_date=today,
                            exit_price=fill_px,
                            qty=pos.qty,
                            pnl=pnl - commission - slippage_cost,
                            r_multiple=r_mult,
                            commission=commission,
                            slippage_cost=slippage_cost,
                        ))

                        # Update equity on close
                        current_equity += (pnl - commission - slippage_cost)
                        equity_line[today] = current_equity

                        # Flat now
                        del positions[sym]

                # ---------- 2) generate new signal for entry at next open ----------
                # Use slice up to 'today' to avoid look-ahead
                hist = ind.iloc[: i + 1]
                if len(hist) < min_bars:
                    continue

                # Regime filter first (same as live)
                try:
                    if not _regime_pass(hist, cfg):
                        continue
                except Exception:
                    # if anything weird → don't block
                    pass

                # Five energies (same as live)
                try:
                    energies = evaluate_five_energies(hist)
                except Exception:
                    continue

                direction = energies.get("direction", "none")
                score = int(energies.get("score", 0))
                if score < int(cfg.get("trading", {}).get("min_score", 4)):
                    continue
                if direction not in {"long", "short"}:
                    continue
                if direction == "short" and not allow_short:
                    continue

                # MTF weekly confirmation (same rules as live)
                if mtf_enabled:
                    try:
                        wk = _resample_weekly_ohlcv(hist)
                        wk_ind = add_indicators(wk)
                        w_trend = _weekly_trend_view(wk_ind, mtf_cfg)
                        mtf_ok, _ = _mtf_alignment_ok(direction, w_trend, mtf_cfg)
                        if not mtf_ok:
                            continue
                    except Exception:
                        # if MTF fails → don't block
                        pass

                # Plan bracket from today's close (like live)
                row = hist.iloc[-1]
                close = float(row["Close"])
                atr14 = float(row["ATR14"]) if not pd.isna(row["ATR14"]) else None
                if (atr14 is None) or (atr14 <= 0):
                    continue

                stop_planned, target_planned = compute_levels(
                    direction=direction,
                    entry=close,
                    atr=atr14,
                    atr_mult=atr_multiple_stop,
                    reward_mult=reward_multiple,
                )
                if stop_planned is None or target_planned is None:
                    continue

                # Size on today's close (like live). Equity-based.
                if direction == "long":
                    per_risk = max(close - stop_planned, 0.0)
                else:
                    per_risk = max(stop_planned - close, 0.0)
                qty = _size_qty(current_equity, risk_per_trade_pct, per_risk, min_shares=min_shares)
                if qty <= 0:
                    continue

                # Entry at next bar open
                nxt_bar = ind.loc[nxt]
                raw_entry = float(nxt_bar["Open"]) if entry_price_mode == "next_open" else float(row["Close"])
                # Trade action for slippage on entry
                entry_action = "sell" if direction == "short" else "buy"
                entry_fill, entry_slip_ps = _slip(raw_entry, entry_action, slippage_bps)

                # Re-anchor stop/target to actual entry
                stop, target = _anchor_bracket_at_fill(
                    planned_entry=close,
                    stop=stop_planned,
                    target=target_planned,
                    fill=entry_fill,
                    side=direction,
                    reward_mult=reward_multiple,
                )

                # Commission & slippage (entry only for now; exit added when we close)
                _ = commission_ps * qty
                _ = entry_slip_ps * qty
                # We don't deduct at entry; we record realized on exit, equity changes on exit (v1).
                # (If you want to deduct at entry too, uncomment the line below and also record in equity_line[nxt])
                # current_equity -= (commission_ps * qty + entry_slip_ps * qty)

                # Open position
                positions[sym] = Position(
                    symbol=sym,
                    side=direction,
                    qty=qty,
                    entry_date=nxt,
                    entry_price=entry_fill,
                    stop=stop,
                    target=target,
                    planned_r_per_share=per_risk if per_risk > 0 else 1e-8,
                    reward_multiple=reward_multiple,
                )

        except Exception:
            # Fail-soft per symbol
            continue

    # ---------- Wrap up equity series ----------
    if trades:
        dates = sorted(set([t.entry_date for t in trades] + [t.exit_date for t in trades]))
    else:
        dates = []
    if dates:
        # carry-forward equity
        eq = []
        running = initial_equity
        last_known_by_date = dict(sorted(equity_line.items()))
        for d in dates:
            if d in last_known_by_date:
                running = last_known_by_date[d]
            eq.append((d, running))
        equity_df = pd.DataFrame(eq, columns=["date", "equity"]).set_index("date")
    else:
        equity_df = pd.DataFrame(columns=["equity"])

    # Trades DF
    if trades:
        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        trades_df = trades_df.sort_values(["exit_date", "symbol"]).reset_index(drop=True)
    else:
        trades_df = pd.DataFrame(columns=[
            "symbol","side","entry_date","entry_price","exit_date","exit_price",
            "qty","pnl","r_multiple","commission","slippage_cost"
        ])

    # ---------- Metrics ----------
    metrics = {}
    if not equity_df.empty:
        eq = equity_df["equity"].astype(float)
        ret = eq.pct_change().fillna(0.0)
        total_return = (eq.iloc[-1] / eq.iloc[0]) - 1.0 if eq.iloc[0] > 0 else 0.0
        days = len(eq)
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252.0 / max(days, 1)) - 1.0 if days > 1 and eq.iloc[0] > 0 else 0.0
        dd = (eq / eq.cummax() - 1.0).min()
        sharpe = (ret.mean() / (ret.std() + 1e-12)) * math.sqrt(252.0) if ret.std() > 0 else 0.0
        metrics.update({
            "start_equity": float(eq.iloc[0]),
            "end_equity": float(eq.iloc[-1]),
            "total_return": float(total_return),
            "CAGR": float(cagr),
            "max_drawdown": float(dd),
            "daily_sharpe": float(sharpe),
            "days": int(days),
        })

    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        metrics.update({
            "trades": int(len(trades_df)),
            "win_rate": float(len(wins) / len(trades_df)) if len(trades_df) else 0.0,
            "avg_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
            "avg_loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
            "profit_factor": float(wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and abs(losses["pnl"].sum()) > 0 else np.inf if len(wins) else 0.0,
            "avg_R": float(trades_df["r_multiple"].mean()),
        })

    return {
        "trades": trades_df,
        "equity": equity_df,
        "metrics": metrics,
    }
