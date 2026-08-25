#!/usr/bin/env python
"""Drawdown improvement experiment — test circuit breakers that reduce DD risk.

Tests 3 improvements individually and combined:
1. Daily loss circuit breaker (stop new trades if day PnL < -X%)
2. Faster deleveraging ramp (start at 3% instead of 5%)
3. Cross-asset correlation throttle (cut sizing when signals are too correlated)

Uses a modified simulate_with_breaker that adds these portfolio-level protections.

Usage:
    PYTHONPATH="src;." python experiments/drawdown_improvements.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from experiments.circuit_breaker_v2_experiment import (
    NEEDED_TFS, FULL5, CAP, BASELINE_WR, SPREAD_BPS,
    apply_vol, get_vip_tier,
)
from experiments.dynamic_asset_v2_experiment import (
    assemble_signals, OBVIOUS_CORE,
)
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig

CACHE_DIR = Path("experiments/.cache")
N_MC = 1000
SEED = 42


def _edge_from_probability(prob, sig):
    if np.isfinite(prob):
        return float(np.clip(abs(2.0 * prob - 1.0), 0.0, 1.0))
    return float(np.clip(abs(sig), 0.0, 1.0))


TF_RESAMPLE = {"15m": "15min", "1h": "1h", "3h": "3h", "4h": "4h", "24h": "1D"}


def apply_bar_level_sl(signals_df, close_cache, max_risk_per_trade=0.03,
                       sl_cap=0.75, base_sizing_pct=0.06):
    """Replace fwd_return with SL-capped return when price breaches SL intra-hold.

    Walks forward bar-by-bar through each signal's hold period. If the
    unrealized return crosses the dynamic SL threshold, replaces fwd_return
    with the actual return at the SL bar and reduces hold_bars accordingly.

    This matches the live engine's conditional SL order behavior, where the
    exchange exits the position as soon as price touches the SL level.

    Args:
        signals_df: DataFrame with timestamp, asset, direction, fwd_return,
                    hold_bars, probability, signal_strength, strategy (= TF label)
        close_cache: dict[(asset, tf_label)] -> pd.Series of close prices
        max_risk_per_trade: risk budget per trade (fraction of equity)
        sl_cap: maximum SL width (fraction of position)
        base_sizing_pct: for estimating position size → dynamic_sl

    Returns:
        (modified_df, sl_trigger_count)
    """
    out = signals_df.copy()
    fwd = out["fwd_return"].values.copy()
    holds = out["hold_bars"].values.copy()
    timestamps = pd.to_datetime(out["timestamp"])
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_localize(None)

    # Ensure close_cache indices are tz-naive
    close_cache = {k: (v.tz_localize(None) if hasattr(v.index, 'tz') and v.index.tz else v)
                   for k, v in close_cache.items()}

    assets = out["asset"].values
    directions = out["direction"].values.astype(np.float64)
    probs = out["probability"].values.astype(np.float64)
    strategies = out["strategy"].values  # TF labels: "15m", "1h", etc.

    sl_count = 0
    for i in range(len(out)):
        asset = assets[i]
        tf = strategies[i]
        key = (asset, tf)
        if key not in close_cache:
            continue

        closes = close_cache[key]
        entry_ts = timestamps.iloc[i]
        entry_idx = closes.index.searchsorted(entry_ts)
        if entry_idx >= len(closes):
            continue
        entry_price = closes.values[entry_idx]
        if entry_price <= 0:
            continue

        # Compute dynamic SL from estimated sizing
        edge = abs(2.0 * probs[i] - 1.0)
        edge = max(edge, 0.01)  # floor to avoid division by zero
        size_pct = base_sizing_pct * edge
        dynamic_sl = min(max_risk_per_trade / size_pct, sl_cap)

        direction = directions[i]
        n_bars = int(holds[i])

        for b in range(1, n_bars + 1):
            check_idx = entry_idx + b
            if check_idx >= len(closes):
                break
            bar_price = closes.values[check_idx]
            unrealized = direction * (bar_price - entry_price) / entry_price
            if unrealized < -dynamic_sl:
                # SL triggered — use actual return at this bar
                fwd[i] = (bar_price - entry_price) / entry_price
                holds[i] = b
                sl_count += 1
                break

    out["fwd_return"] = fwd
    out["hold_bars"] = holds
    return out, sl_count


def simulate_with_improvements(signals_df, config, breaker_params, improvements,
                               slippage_model=None, asset_volumes=None):
    """Modified simulate_with_breaker with drawdown improvements.

    Optional realistic slippage:
        slippage_model: dict with keys calibration, vip_tier, max_participation_rate
        asset_volumes: dict[str, pd.Series] of daily dollar volumes per asset

    Improvements dict:
        daily_loss_limit: float or None — stop new trades if day PnL < -X (as fraction)
        delever_start: float — override config.delever_start
        correlation_throttle: float or None — cut sizing by 50% when >X% signals same direction
        correlation_window: int — number of recent signals to check for correlation
    """
    if signals_df is None or len(signals_df) < 10:
        return None

    # Deterministic order — must match the live aggregator
    # (src/live/core/aggregator.py: sort by (asset, timeframe)).
    # Fallback: if upstream hasn't tagged timeframe, sort by (timestamp, asset).
    _sort_cols = ["timestamp", "asset"]
    if "timeframe" in signals_df.columns:
        _sort_cols.append("timeframe")
    sub = signals_df.sort_values(
        _sort_cols, kind="stable"
    ).reset_index(drop=True)
    n = len(sub)

    timestamps = sub["timestamp"].values
    assets_arr = sub["asset"].values
    directions = sub["direction"].values.astype(np.float64)
    fwd_returns = sub["fwd_return"].values.astype(np.float64)
    signal_strengths = sub["signal_strength"].values.astype(np.float64)
    probabilities = sub["probability"].values.astype(np.float64)
    regime_mults = sub["regime_exposure_mult"].values.astype(np.float64) if "regime_exposure_mult" in sub.columns else np.ones(n)
    sizing_mults = sub["sizing_mult"].values.astype(np.float64) if "sizing_mult" in sub.columns else np.ones(n)
    hold_bars_arr = sub["hold_bars"].values.astype(np.int64) if "hold_bars" in sub.columns else np.full(n, 4, dtype=np.int64)
    strategies_arr = sub["strategy"].values if "strategy" in sub.columns else np.full(n, "default")

    # Improvement params
    daily_loss_limit = improvements.get("daily_loss_limit", None)
    delever_start = improvements.get("delever_start", config.delever_start)
    corr_throttle = improvements.get("correlation_throttle", None)
    corr_window = improvements.get("correlation_window", 20)
    corr_direction = improvements.get("correlation_direction", "both")
    corr_scale = improvements.get("correlation_scale", 0.5)
    max_signals_per_bar = improvements.get("max_signals_per_bar", None)
    vol_regime_cut = improvements.get("vol_regime_cut", None)
    rolling_sharpe_gate = improvements.get("rolling_sharpe_gate", False)
    max_daily_notional_pct = improvements.get("max_daily_notional_pct", None)
    net_exposure_limit_pct = improvements.get("net_exposure_limit_pct", None)
    cppi_multiplier = improvements.get("cppi_multiplier", None)  # CPPI: M param (e.g. 2.0)
    tuw_threshold_days = improvements.get("tuw_threshold_days", None)  # TUW: days below peak before decay
    tuw_decay_rate = improvements.get("tuw_decay_rate", 0.05)  # TUW: sizing reduction per day past threshold
    tuw_floor = improvements.get("tuw_floor", 0.3)  # TUW: minimum sizing mult
    kurtosis_threshold = improvements.get("kurtosis_threshold", None)  # Kurtosis: threshold to trigger cut
    kurtosis_scale = improvements.get("kurtosis_scale", 0.6)  # Kurtosis: sizing mult when triggered
    compliance_sl_pct = improvements.get("compliance_sl_pct", None)  # Fixed SL (backward compat)
    max_risk_per_trade = improvements.get("max_risk_per_trade", None)  # Dynamic SL: 0.03 = 3% risk cap
    sl_cap = improvements.get("sl_cap", 0.75)  # Max SL distance (cap for dynamic)

    # ── HyroTrader compliance rules ──
    daily_dd_limit_abs = improvements.get("daily_dd_limit", None)      # 0.05 = 5% of initial
    hyro_leverage = improvements.get("leverage", 1)                     # for margin calc
    hyro_max_margin_pct = improvements.get("max_margin_pct", None)      # 0.25 = 25% of initial
    hyro_max_notional_mult = improvements.get("max_notional_mult", None)  # 2.0 = 2× initial
    hyro_risk_basis = improvements.get("risk_basis", "equity")          # "initial" or "equity"
    hyro_max_loss_initial = improvements.get("max_loss_from_initial", None)  # 0.10 = 10% from initial

    equity_arr = np.empty(n)
    pnl_arr = np.zeros(n)
    size_arr = np.zeros(n)
    skipped = np.ones(n, dtype=bool)
    sl_triggers = 0

    equity = config.starting_capital
    initial_balance = config.starting_capital  # HyroTrader: never changes
    peak_equity = equity
    halted = False
    active_positions = []

    unique_ts = pd.Series(timestamps).unique()
    ts_to_idx = {ts: idx for idx, ts in enumerate(unique_ts)}

    # Per-asset breaker tracking (F1 symmetric)
    asset_recent_pnl = defaultdict(lambda: deque(maxlen=breaker_params.get("window", 10)))
    asset_recent_notional = defaultdict(lambda: deque(maxlen=breaker_params.get("window", 10)))
    cb_triggers = 0

    # Daily PnL tracking
    current_day = None
    day_pnl = 0.0
    day_halted = False
    day_notional = 0.0
    day_start_equity = equity  # HyroTrader swing DD: equity at start of day

    # Correlation tracking
    recent_directions = deque(maxlen=corr_window)

    # Rolling Sharpe tracking
    recent_daily_rets = deque(maxlen=20)
    sharpe_halted = False

    # Vol regime: precompute BTC vol percentiles from sizing_mults
    # (vol_series is baked into sizing_mults via apply_vol, but we need raw BTC vol)
    # We'll use a proxy: track rolling volatility of equity returns
    recent_eq_rets = deque(maxlen=20)
    prev_equity = equity

    # Per-bar signal counting
    current_bar_ts = None
    bar_signal_count = 0

    # CPPI tracking
    cppi_floor = equity * (1 - config.max_dd) if cppi_multiplier else 0

    # TUW tracking
    days_below_peak = 0

    # Kurtosis tracking
    kurtosis_daily_rets = deque(maxlen=20)

    for i in range(n):
        current_ts = timestamps[i]
        current_ts_idx = ts_to_idx.get(current_ts, 0)

        # Day boundary check
        ts_date = pd.Timestamp(current_ts)
        if hasattr(ts_date, 'date'):
            trade_day = ts_date.date()
        else:
            trade_day = current_day
        if trade_day != current_day:
            # End-of-day: update trackers
            if current_day is not None and equity > 0:
                eq_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
                recent_daily_rets.append(eq_ret)
                recent_eq_rets.append(eq_ret)
                kurtosis_daily_rets.append(eq_ret)
                prev_equity = equity
                if rolling_sharpe_gate and len(recent_daily_rets) >= 20:
                    arr = np.array(recent_daily_rets)
                    rs = arr.mean() / arr.std() if arr.std() > 0 else 0
                    sharpe_halted = rs < 0
                # TUW: count days below peak
                if equity < peak_equity:
                    days_below_peak += 1
                else:
                    days_below_peak = 0
                # CPPI: ratchet floor up with new peaks
                if cppi_multiplier and equity > peak_equity:
                    cppi_floor = equity * (1 - config.max_dd)
            current_day = trade_day
            day_pnl = 0.0
            day_halted = False
            day_notional = 0.0
            day_start_equity = equity  # HyroTrader swing DD: reset at day boundary

        active_positions = [(e, sz, a, st) for e, sz, a, st in active_positions if e > current_ts_idx]
        current_exposure = sum(sz for _, sz, _, _ in active_positions)
        asset_exposure = {}
        strategy_exposure = {}
        for _, sz, a, st in active_positions:
            asset_exposure[a] = asset_exposure.get(a, 0.0) + sz
            strategy_exposure[st] = strategy_exposure.get(st, 0.0) + sz

        if halted:
            equity_arr[i] = equity
            continue

        # Reality: can't trade with no money
        if equity < 1.0:
            halted = True
            equity_arr[i] = equity
            continue

        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd >= config.max_dd:
            halted = True
            equity_arr[i] = equity
            continue

        # ── HyroTrader: absolute floor from initial balance ──
        if hyro_max_loss_initial is not None:
            abs_floor = initial_balance * (1 - hyro_max_loss_initial)
            if equity <= abs_floor:
                halted = True
                equity_arr[i] = equity
                continue

        # ── IMPROVEMENT 1: Daily loss circuit breaker ──
        if daily_loss_limit is not None and day_halted:
            equity_arr[i] = equity
            continue

        # ── HyroTrader: Swing daily DD (from day-start equity, % of initial) ──
        if daily_dd_limit_abs is not None and day_halted:
            equity_arr[i] = equity
            continue

        # ── IMPROVEMENT 2: Deleveraging (linear or CPPI) ──
        if cppi_multiplier is not None:
            # CPPI: size = M * cushion / equity, convex derisking
            cushion = max(equity - cppi_floor, 0.0)
            size_mult = min(cppi_multiplier * cushion / equity, 1.0) if equity > 0 else 0.0
        elif dd > delever_start:
            size_mult = max(1.0 - (dd - delever_start) / (config.max_dd - delever_start), 0.0)
        else:
            size_mult = 1.0

        # Base sizing
        sig_abs = _edge_from_probability(probabilities[i], signal_strengths[i])
        sig_abs *= float(np.clip(regime_mults[i], config.regime_exposure_floor, 1.0))
        raw_size = equity * config.base_sizing_pct * sig_abs * size_mult * sizing_mults[i]
        raw_size = min(raw_size, equity * config.base_sizing_pct * config.max_signal_mult * size_mult)

        # ── IMPROVEMENT 3: Correlation throttle ──
        if corr_throttle is not None and len(recent_directions) >= corr_window:
            dir_array = np.array(recent_directions)
            pct_short = (dir_array < 0).mean()
            pct_long = (dir_array > 0).mean()

            throttle = False
            if corr_direction == "both":
                throttle = max(pct_long, pct_short) >= corr_throttle
            elif corr_direction == "short":
                throttle = pct_short >= corr_throttle  # only when bearish consensus
            elif corr_direction == "long":
                throttle = pct_long >= corr_throttle

            if throttle:
                raw_size *= corr_scale

        # ── NEW: Max signals per bar ──
        if max_signals_per_bar is not None:
            if current_ts != current_bar_ts:
                current_bar_ts = current_ts
                bar_signal_count = 0
            bar_signal_count += 1
            if bar_signal_count > max_signals_per_bar:
                equity_arr[i] = equity
                continue

        # ── NEW: Rolling Sharpe gate ──
        if rolling_sharpe_gate and sharpe_halted:
            equity_arr[i] = equity
            continue

        # ── NEW: Vol regime hard cut ──
        if vol_regime_cut is not None and len(recent_eq_rets) >= 10:
            eq_vol = np.std(list(recent_eq_rets))
            # High portfolio vol = high market vol; cut sizing
            if eq_vol > vol_regime_cut:
                raw_size *= 0.5

        # ── NEW: Max daily notional ──
        if max_daily_notional_pct is not None:
            max_daily = equity * max_daily_notional_pct
            if day_notional + raw_size > max_daily:
                raw_size = max(max_daily - day_notional, 0.0)

        # ── NEW: Net exposure limit ──
        if net_exposure_limit_pct is not None:
            net_dir = sum(sz * (1 if st == "default" else 1) for _, sz, a, st in active_positions)
            # Approximate: track signed exposure
            long_exp = sum(sz for _, sz, a, st in active_positions
                          if any(d > 0 for d_idx, d in enumerate(directions) if assets_arr[d_idx] == a and not skipped[d_idx]))
            short_exp = current_exposure - long_exp
            net = abs(long_exp - short_exp)
            if net > equity * net_exposure_limit_pct:
                raw_size *= 0.5

        # ── NEW: Time-under-water decay ──
        if tuw_threshold_days is not None and days_below_peak > tuw_threshold_days:
            excess_days = days_below_peak - tuw_threshold_days
            tuw_mult = max(1.0 - tuw_decay_rate * excess_days, tuw_floor)
            raw_size *= tuw_mult

        # ── NEW: Realized kurtosis overlay ──
        if kurtosis_threshold is not None and len(kurtosis_daily_rets) >= 15:
            arr = np.array(kurtosis_daily_rets)
            mu = arr.mean()
            std = arr.std()
            if std > 0:
                kurt = ((arr - mu) ** 4).mean() / (std ** 4) - 3.0  # excess kurtosis
                if kurt > kurtosis_threshold:
                    raw_size *= kurtosis_scale

        # F1 symmetric breaker
        asset = assets_arr[i]
        cb_mult = 1.0
        recent_pnl = asset_recent_pnl[asset]
        recent_not = asset_recent_notional[asset]
        if len(recent_pnl) >= breaker_params.get("window", 10):
            total_pnl = sum(recent_pnl)
            total_not = sum(recent_not)
            if total_not > 0:
                pnl_ratio = total_pnl / total_not
                if pnl_ratio < breaker_params["threshold"]:
                    t = breaker_params["threshold"]
                    cb_mult = max(1.0 + pnl_ratio / abs(t), breaker_params.get("floor", 0.10))
                    cb_triggers += 1
                elif pnl_ratio > breaker_params.get("up_threshold", 0.01):
                    up_t = breaker_params["up_threshold"]
                    cb_mult = min(1.0 + (pnl_ratio - up_t) / up_t, breaker_params.get("cap", 1.5))

        raw_size *= cb_mult

        # Exposure limits
        asset_exp = asset_exposure.get(asset, 0.0)
        if asset_exp + raw_size > equity * config.max_per_asset:
            raw_size = max(equity * config.max_per_asset - asset_exp, 0.0)
        strategy = strategies_arr[i]
        mps = getattr(config, "max_per_strategy", 1.0)
        strat_exp = strategy_exposure.get(strategy, 0.0)
        if mps < 1.0 and strat_exp + raw_size > equity * mps:
            raw_size = max(equity * mps - strat_exp, 0.0)
        if current_exposure + raw_size > equity * config.max_exposure:
            raw_size = max(equity * config.max_exposure - current_exposure, 0.0)

        # ── HyroTrader: margin limit (25% of initial) ──
        if hyro_max_margin_pct is not None:
            total_margin = current_exposure / hyro_leverage
            margin_limit = initial_balance * hyro_max_margin_pct
            new_margin = raw_size / hyro_leverage
            if total_margin + new_margin > margin_limit:
                raw_size = max((margin_limit - total_margin) * hyro_leverage, 0.0)

        # ── HyroTrader: notional limit (2× initial) ──
        if hyro_max_notional_mult is not None:
            notional_limit = initial_balance * hyro_max_notional_mult
            if current_exposure + raw_size > notional_limit:
                raw_size = max(notional_limit - current_exposure, 0.0)

        # Volume-aware slippage: participation rate cap + max-profit guard
        if slippage_model is not None and asset_volumes is not None:
            from signal_engine.slippage import compute_slippage, lookup_volume, max_profitable_size
            _daily_vol = lookup_volume(asset_volumes, asset, timestamps[i])

            # 1. Participation rate cap
            _max_pr = slippage_model.get("max_participation_rate", 0.02)
            if _daily_vol > 0 and _max_pr > 0:
                _max_notional = _daily_vol * _max_pr
                if raw_size > _max_notional:
                    raw_size = _max_notional

            # 2. Max-profitable-size guard (no lookahead: uses model probability)
            _guard_pct = slippage_model.get("guard_pct", 0.50)
            if _guard_pct > 0 and _daily_vol > 0:
                _expected_edge = abs(2.0 * probabilities[i] - 1.0)
                _s_max = max_profitable_size(
                    asset, _expected_edge, _daily_vol,
                    per_symbol_cal=slippage_model.get("calibration"),
                    vip_tier=slippage_model.get("vip_tier", ("VIP0", 2.0, 5.5)),
                    guard_pct=_guard_pct,
                )
                if raw_size > _s_max > 0:
                    raw_size = _s_max

        if raw_size < 1.0:
            equity_arr[i] = equity
            continue

        # Slippage: realistic model or flat
        if slippage_model is not None and asset_volumes is not None:
            from signal_engine.slippage import compute_slippage, lookup_volume
            _daily_vol = lookup_volume(asset_volumes, asset, timestamps[i])
            _slip = compute_slippage(
                asset, raw_size, _daily_vol,
                per_symbol_cal=slippage_model.get("calibration"),
                vip_tier=slippage_model.get("vip_tier", ("VIP0", 2.0, 5.5)),
            )
            trade_return = directions[i] * fwd_returns[i] - 2 * _slip  # round-trip
        else:
            trade_return = directions[i] * fwd_returns[i] - 2 * config.slippage  # round-trip
        # Reality: can't lose more than position size (no leverage) or 100%
        trade_return = max(trade_return, -1.0)
        # Dynamic compliance SL: widest SL per trade within risk budget
        if max_risk_per_trade is not None and raw_size > 0:
            risk_base = initial_balance if hyro_risk_basis == "initial" else equity
            dynamic_sl = min(max_risk_per_trade * risk_base / raw_size, sl_cap)
            if trade_return < -dynamic_sl:
                trade_return = -dynamic_sl
                sl_triggers += 1
        # Fixed compliance SL (backward compat)
        elif compliance_sl_pct is not None and trade_return < -compliance_sl_pct:
            trade_return = -compliance_sl_pct
            sl_triggers += 1
        pnl = raw_size * trade_return
        equity += pnl
        # Reality: equity can't go below zero
        if equity < 0:
            equity = 0.0
        if equity > peak_equity:
            peak_equity = equity

        active_positions.append((current_ts_idx + int(hold_bars_arr[i]), raw_size, asset, strategy))
        equity_arr[i] = equity
        pnl_arr[i] = pnl
        size_arr[i] = raw_size
        skipped[i] = False

        # Update daily tracking
        day_notional += raw_size
        day_pnl += pnl / equity if equity > 0 else 0
        if daily_loss_limit is not None and day_pnl < -daily_loss_limit:
            day_halted = True
        # HyroTrader swing daily DD: floor = day_start - (initial × limit%)
        if daily_dd_limit_abs is not None:
            daily_floor = day_start_equity - (initial_balance * daily_dd_limit_abs)
            if equity <= daily_floor:
                day_halted = True

        # Update correlation tracker
        recent_directions.append(directions[i])

        # Update F1 breaker
        asset_recent_pnl[asset].append(pnl)
        asset_recent_notional[asset].append(raw_size)

    return {
        "equity": equity_arr, "pnl": pnl_arr, "sizes": size_arr,
        "skipped": skipped, "timestamps": timestamps, "halted": halted,
        "final_equity": equity, "n_trades": int((~skipped).sum()),
        "cb_triggers": cb_triggers, "sl_triggers": sl_triggers,
    }


def run_variant(merged, vol_series, improvements):
    """Run one variant and compute stats."""
    port = PortfolioConfig(
        starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
        base_sizing_pct=0.045, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    prepared = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = apply_vol(prepared.sort_values("timestamp").reset_index(drop=True), vol_series)

    breaker_params = {
        "window": 10, "threshold": -0.01, "floor": 0.10,
        "up_threshold": 0.01, "cap": 1.5,
    }

    sim = simulate_with_improvements(prepared, port, breaker_params, improvements)
    if sim is None:
        return None

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    dr = dpnl / CAP
    if dr.std() == 0 or len(dr) < 30:
        return None

    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    eq_adj = (1 + dr).cumprod()
    dd_s = (eq_adj.cummax() - eq_adj) / eq_adj.cummax()
    max_dd = float(dd_s.max() * 100)
    pnl = float(dpnl.sum())
    daily_dd_pct = dpnl / (CAP + dpnl.cumsum()).shift(1).fillna(CAP) * 100
    max_daily_dd = float(daily_dd_pct.min())
    monthly = dpnl.resample("MS").sum()

    return {
        "sr": sr, "max_dd": max_dd, "pnl": pnl,
        "max_daily_dd": max_daily_dd,
        "n_trades": sim["n_trades"],
        "pnl_mo": pnl / max(len(monthly), 1),
        "daily_returns": dr,
    }


def run_monte_carlo(daily_returns, n_sims=N_MC):
    """Block bootstrap Monte Carlo on daily returns."""
    rng = np.random.default_rng(SEED)
    dr = daily_returns.values
    n_days = len(dr)
    block_size = 5
    n_blocks = n_days // block_size

    block_starts = rng.integers(0, n_days - block_size, size=(n_sims, n_blocks))
    boot = np.zeros((n_sims, n_blocks * block_size))
    for b in range(n_blocks):
        for d in range(block_size):
            boot[:, b * block_size + d] = dr[block_starts[:, b] + d]

    cum = np.cumprod(1.0 + boot, axis=1)
    rmax = np.maximum.accumulate(cum, axis=1)
    dds = (rmax - cum) / rmax
    max_dds = dds.max(axis=1) * 100
    worst_days = boot.min(axis=1) * 100

    return {
        "breach_10": (max_dds > 10).mean() * 100,
        "breach_8": (max_dds > 8).mean() * 100,
        "breach_daily_5": (worst_days < -5).mean() * 100,
        "breach_daily_3": (worst_days < -3).mean() * 100,
        "dd_p50": np.percentile(max_dds, 50),
        "dd_p95": np.percentile(max_dds, 95),
        "dd_p99": np.percentile(max_dds, 99),
        "day_p1": np.percentile(worst_days, 1),
        "day_p5": np.percentile(worst_days, 5),
    }


def main():
    T0 = time.time()
    print("=" * 110)
    print("  DRAWDOWN IMPROVEMENT EXPERIMENT")
    print(f"  Each variant: backtest + {N_MC} Monte Carlo simulations")
    print("=" * 110)

    with open(CACHE_DIR / "v2_prep_cache.pkl", "rb") as f:
        cached = pickle.load(f)
    asset_volumes = cached["asset_volumes"]
    hold_sigs = cached["hold_sigs"]
    vol_series = cached["vol_series"]

    months = pd.period_range("2020-07", "2026-02", freq="M")
    CORE = OBVIOUS_CORE

    def _regime_sel(ms):
        btc_vol = vol_series.get("BTC")
        n_assets = 30
        if btc_vol is not None:
            avail = btc_vol.loc[:ms - pd.Timedelta(days=1)].dropna()
            if len(avail) > 60:
                current = avail.iloc[-1]
                if current > avail.quantile(0.80): n_assets = 15
                elif current < avail.quantile(0.20): n_assets = 40
        cutoff = ms - pd.DateOffset(months=3)
        scored = []
        for name in hold_sigs:
            confs = []
            for tf in hold_sigs[name]:
                s = hold_sigs[name][tf]
                ts = pd.to_datetime(s["timestamp"])
                if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
                w = s[((ts >= cutoff) & (ts < ms)).values]
                if "confidence" in w.columns and len(w) > 5:
                    confs.extend(w["confidence"].values)
            if not confs: continue
            vol = asset_volumes.get(name)
            if vol is None: continue
            avail_v = vol.loc[:ms - pd.Timedelta(days=1)]
            if len(avail_v) == 0: continue
            scored.append((name, float(np.mean(confs)) * float(avail_v.iloc[-1])))
        scored.sort(key=lambda x: -x[1])
        return [a for a, _ in scored[:n_assets]]

    def _s8_sel(ms):
        active_core = [a for a in CORE if a in hold_sigs]
        full = _regime_sel(ms)
        return active_core + [a for a in full if a not in CORE]

    merged = assemble_signals(hold_sigs, _s8_sel, months)
    print(f"  {len(merged):,} signals loaded")

    # ── Variants ──
    variants = [
        ("Baseline (no improvements)", {}),
        # Individual improvements
        ("DailyLimit -1.5%",     {"daily_loss_limit": 0.015}),
        ("DailyLimit -2.0%",     {"daily_loss_limit": 0.020}),
        ("DailyLimit -2.5%",     {"daily_loss_limit": 0.025}),
        ("Delever@3%",           {"delever_start": 0.03}),
        ("Delever@4%",           {"delever_start": 0.04}),
        ("Delever@2%",           {"delever_start": 0.02}),
        ("CorrThrottle 80%",    {"correlation_throttle": 0.80}),
        ("CorrThrottle 70%",    {"correlation_throttle": 0.70}),
        ("CorrThrottle 90%",    {"correlation_throttle": 0.90}),
        # Combinations
        ("DL-2% + Dlv@3%",      {"daily_loss_limit": 0.020, "delever_start": 0.03}),
        ("DL-2% + Dlv@4%",      {"daily_loss_limit": 0.020, "delever_start": 0.04}),
        ("DL-2% + Corr80%",     {"daily_loss_limit": 0.020, "correlation_throttle": 0.80}),
        ("Dlv@3% + Corr80%",    {"delever_start": 0.03, "correlation_throttle": 0.80}),
        ("All3: DL2+Dlv3+C80",  {"daily_loss_limit": 0.020, "delever_start": 0.03, "correlation_throttle": 0.80}),
        ("All3: DL2+Dlv3+C90",  {"daily_loss_limit": 0.020, "delever_start": 0.03, "correlation_throttle": 0.90}),
        ("All3: DL2.5+Dlv3+C80", {"daily_loss_limit": 0.025, "delever_start": 0.03, "correlation_throttle": 0.80}),
    ]

    print(f"\n  Running {len(variants)} variants (backtest + {N_MC} MC each)...")
    print(f"\n  {'Variant':<26} {'Sharpe':>6} {'PnL':>10} {'MaxDD':>5} {'DayDD':>6} "
          f"{'MC>10%':>6} {'MC>8%':>5} {'MC d<-5':>7} {'MC d<-3':>7} "
          f"{'DDp95':>5} {'DDp99':>5}")
    print("  " + "-" * 110)

    for name, improvements in variants:
        t1 = time.time()
        r = run_variant(merged, vol_series, improvements)
        if r is None:
            print(f"  {name:<26} FAILED")
            continue

        mc = run_monte_carlo(r["daily_returns"])
        elapsed = time.time() - t1

        print(f"  {name:<26} {r['sr']:>6.2f} ${r['pnl']:>8,.0f} {r['max_dd']:>4.1f}% "
              f"{r['max_daily_dd']:>5.2f}% "
              f"{mc['breach_10']:>5.1f}% {mc['breach_8']:>4.1f}% "
              f"{mc['breach_daily_5']:>6.1f}% {mc['breach_daily_3']:>6.1f}% "
              f"{mc['dd_p95']:>4.1f}% {mc['dd_p99']:>4.1f}%")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
