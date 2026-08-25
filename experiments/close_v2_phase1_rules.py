#!/usr/bin/env python
"""Close Model V2 — Phase 1: Rule-Based Exit Baseline.

Tests simple exit rules against fixed_T4 (1h hold) on the optimized baseline
(conf >= 0.08, inverted vol, 4% sizing). Each early exit costs 4.5 bps extra.

Uses cached close decision examples which have bar-by-bar returns already computed.
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.pipeline import apply_vol_overlay
from signal_engine.config import VolOverlayConfig
from signal_engine.data import load_binance_1m, resample_ohlcv

EXIT_COST_BPS = 4.5
EXIT_COST = EXIT_COST_BPS / 10000
CAPITAL = 100_000.0
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]

VOL_INVERTED = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                 low_pct=0.20, high_pct=0.80,
                                 low_mult=1.25, mid_mult=1.00, high_mult=0.50)


def load_examples():
    """Load cached close decision examples with bar-by-bar data."""
    path = Path("results/close_cache/examples_logreg_15m.parquet")
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows, {df['trade_id'].nunique():,} trades")
    return df


def build_trade_matrix(examples):
    """Build per-trade matrix: returns at each decision step.

    Returns DataFrame indexed by trade_id with columns:
    ret_0..ret_3 (exit return at each step), peak_0..peak_3 (MFE), expiry_ret
    """
    # Group by trade_id, pivot decision steps
    trades = []
    for trade_id, group in examples.groupby("trade_id"):
        group = group.sort_values("bars_since_entry")
        row = {
            "trade_id": trade_id,
            "asset": group.iloc[0]["asset"],
            "timestamp": group.iloc[0]["timestamp"],
            "direction": group.iloc[0]["direction"],
            "probability": group.iloc[0]["entry_probability"],
            "confidence": group.iloc[0]["entry_confidence"],
            "signal_strength": group.iloc[0]["entry_signal_strength"],
            "expiry_ret": group.iloc[-1]["exit_return"],
        }
        if "regime_exposure_mult" in group.columns:
            row["regime_exposure_mult"] = group.iloc[0]["regime_exposure_mult"]

        for i, (_, r) in enumerate(group.iterrows()):
            row[f"ret_{i}"] = r["exit_return"]
            row[f"mfe_{i}"] = r["max_favorable_return_since_entry"]
            row[f"mae_{i}"] = r["max_adverse_return_since_entry"]

        trades.append(row)

    df = pd.DataFrame(trades)
    df["strategy"] = "logreg_15m"
    return df


def apply_exit_rule(trade_matrix, rule_name, rule_fn):
    """Apply an exit rule. Returns modified signals compatible with bybit_sim.

    rule_fn(trade_row) -> exit_step (0-based), or -1 for hold-to-expiry.
    Early exit (step < max_step) costs extra EXIT_COST.
    """
    n_steps = sum(1 for c in trade_matrix.columns if c.startswith("ret_"))
    max_step = n_steps - 1

    out = trade_matrix.copy()
    exit_returns = np.zeros(len(out))
    hold_bars = np.full(len(out), n_steps, dtype=int)

    for i in range(len(out)):
        row = out.iloc[i]
        step = rule_fn(row, n_steps)
        if step < 0 or step >= n_steps:
            step = max_step  # hold to expiry

        ret_col = f"ret_{step}"
        if ret_col in row.index and np.isfinite(row[ret_col]):
            ret = row[ret_col]
        else:
            ret = row["expiry_ret"] if np.isfinite(row["expiry_ret"]) else 0.0

        # Early exit costs extra
        if step < max_step:
            ret -= EXIT_COST

        exit_returns[i] = ret
        hold_bars[i] = step + 1

    out["fwd_return"] = exit_returns
    out["hold_bars"] = hold_bars
    return out


# ============================================================
# EXIT RULES
# ============================================================

def rule_fixed_t4(row, n_steps):
    """Hold to expiry (last step)."""
    return n_steps - 1

def rule_decay_t3(row, n_steps):
    """Close at step 2 (T+3 in 1-indexed) if losing."""
    if n_steps <= 2:
        return n_steps - 1
    ret_2 = row.get("ret_2", np.nan)
    if np.isfinite(ret_2) and ret_2 < 0:
        return 2
    return n_steps - 1

def rule_decay_t2(row, n_steps):
    """Close at step 1 (T+2) if losing."""
    if n_steps <= 1:
        return n_steps - 1
    ret_1 = row.get("ret_1", np.nan)
    if np.isfinite(ret_1) and ret_1 < 0:
        return 1
    return n_steps - 1

def make_trail_rule(trail_bps):
    """Trailing stop: exit when return drops trail_bps from MFE."""
    trail = trail_bps / 10000
    def rule(row, n_steps):
        for step in range(n_steps):
            mfe = row.get(f"mfe_{step}", 0)
            ret = row.get(f"ret_{step}", np.nan)
            if np.isfinite(ret) and np.isfinite(mfe) and (mfe - ret) > trail:
                return step
        return n_steps - 1
    return rule

def rule_breakeven_t2(row, n_steps):
    """If was profitable at step 0, close at step 1 if now negative."""
    if n_steps <= 1:
        return n_steps - 1
    ret_0 = row.get("ret_0", np.nan)
    ret_1 = row.get("ret_1", np.nan)
    if np.isfinite(ret_0) and ret_0 > 0 and np.isfinite(ret_1) and ret_1 < 0:
        return 1
    return n_steps - 1

def rule_time_decay_losing(row, n_steps):
    """Close at step 1 if losing >10bps, step 2 if losing >0."""
    if n_steps <= 1:
        return n_steps - 1
    ret_1 = row.get("ret_1", np.nan)
    if np.isfinite(ret_1) and ret_1 < -0.001:  # losing > 10bps
        return 1
    if n_steps <= 2:
        return n_steps - 1
    ret_2 = row.get("ret_2", np.nan)
    if np.isfinite(ret_2) and ret_2 < 0:
        return 2
    return n_steps - 1

def rule_cut_losers_hold_winners(row, n_steps):
    """At each step, if losing AND getting worse, exit. Otherwise hold."""
    for step in range(1, n_steps):
        ret_prev = row.get(f"ret_{step-1}", np.nan)
        ret_curr = row.get(f"ret_{step}", np.nan)
        if np.isfinite(ret_curr) and ret_curr < 0 and np.isfinite(ret_prev) and ret_curr < ret_prev:
            return step
    return n_steps - 1


# ============================================================
# SIMULATION
# ============================================================

def bybit_sim(signals, base_pct=0.04, max_mult=2.0):
    """Quick Bybit sim with configurable sizing."""
    sub = signals.sort_values("timestamp").reset_index(drop=True)
    rt = EXIT_COST_BPS / 10000  # Base roundtrip cost (applied on top of fwd_return)
    # fwd_return already includes EXIT_COST for early exits, so only apply base RT cost
    gross = sub["fwd_return"].values  # already direction-signed
    net = gross - rt  # base Bybit cost

    prob = sub["probability"].values if "probability" in sub.columns else np.full(len(sub), 0.55)
    sizing_mult = sub["sizing_mult"].values if "sizing_mult" in sub.columns else np.ones(len(sub))
    edge = np.where(np.isfinite(prob), np.abs(2 * prob - 1), 0.1)
    edge = np.clip(edge, 0, 1)

    eq = CAPITAL
    pnl = np.zeros(len(sub))
    for i in range(len(sub)):
        sz = min(eq * base_pct * edge[i] * sizing_mult[i], eq * base_pct * max_mult)
        pnl[i] = sz * net[i]
        eq += pnl[i]

    ts = pd.to_datetime(sub["timestamp"].values)
    if hasattr(ts, 'tz') and ts.tz is not None:
        ts = ts.tz_localize(None)
    daily = pd.Series(pnl, index=ts).resample("1D").sum()
    daily = daily[daily != 0]
    sr = float(daily.mean() / daily.std() * np.sqrt(365.25)) if len(daily) > 30 and daily.std() > 0 else 0
    cum_eq = CAPITAL + daily.cumsum()
    dd = float(((cum_eq.cummax() - cum_eq) / cum_eq.cummax()).max() * 100) if len(cum_eq) > 0 else 0

    return {"sr": sr, "pnl": float(pnl.sum()), "dd": dd, "daily": daily,
            "n": len(sub), "n_early": int((sub["hold_bars"] < sub["hold_bars"].max()).sum()),
            "mean_ret_bps": float(gross.mean() * 10000), "mean_net_bps": float(net.mean() * 10000)}


def main():
    T0 = time.time()
    print("=" * 80)
    print("  CLOSE V2 PHASE 1: Rule-Based Exit Baseline")
    print("=" * 80)

    # Load examples
    print("\nLoading cached examples...")
    examples = load_examples()
    if examples is None:
        return

    # Build trade matrix
    print("Building trade matrix...")
    trade_matrix = build_trade_matrix(examples)
    print(f"  {len(trade_matrix):,} trades with "
          f"{sum(1 for c in trade_matrix.columns if c.startswith('ret_'))} steps")

    # Apply confidence filter
    if "confidence" in trade_matrix.columns:
        trade_matrix = trade_matrix[trade_matrix["confidence"] >= 0.08].reset_index(drop=True)
        print(f"  After conf >= 0.08: {len(trade_matrix):,} trades")

    # Load other strategies for combined portfolio
    sigs_4h = pd.read_parquet("results/signals_cache/logreg_4h.parquet") if Path("results/signals_cache/logreg_4h.parquet").exists() else pd.DataFrame()
    sigs_cs = pd.read_parquet("results/signals_cache/cs_momentum.parquet") if Path("results/signals_cache/cs_momentum.parquet").exists() else pd.DataFrame()

    # Apply inverted vol overlay
    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
        has_vol = True
    except Exception:
        has_vol = False

    # Define rules
    rules = {
        "fixed_T4": rule_fixed_t4,
        "decay_T3": rule_decay_t3,
        "decay_T2": rule_decay_t2,
        "trail_20bps": make_trail_rule(20),
        "trail_30bps": make_trail_rule(30),
        "trail_50bps": make_trail_rule(50),
        "breakeven_T2": rule_breakeven_t2,
        "time_decay": rule_time_decay_losing,
        "cut_losers": rule_cut_losers_hold_winners,
    }

    # Test each rule
    print(f"\n{'='*80}")
    print("  RULE COMPARISON (bybit_sim, conf>=0.08, inverted vol, 4% sizing)")
    print(f"{'='*80}")
    print(f"  {'Rule':<20} {'SR':>7} {'DD%':>6} {'PnL':>12} {'Early%':>7} {'Net bps':>8}")
    print("-" * 70)

    results = []
    for name, rule_fn in rules.items():
        # Apply rule to get exit returns
        ruled = apply_exit_rule(trade_matrix, name, rule_fn)

        # Merge with 4h + cs
        parts = [ruled]
        for other in [sigs_4h, sigs_cs]:
            if len(other) > 0:
                o = other.copy()
                for col in ["regime_exposure_mult", "sizing_mult", "hold_bars"]:
                    if col not in o.columns:
                        o[col] = 4 if col == "hold_bars" else 1.0
                o["probability"] = o["probability"].astype(float)
                parts.append(o)

        merged = pd.concat(parts, ignore_index=True)
        if "regime_exposure_mult" not in merged.columns:
            merged["regime_exposure_mult"] = 1.0
        merged["regime_exposure_mult"] = merged["regime_exposure_mult"].fillna(1.0)
        if "sizing_mult" not in merged.columns:
            merged["sizing_mult"] = 1.0
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        # Apply vol overlay
        if has_vol:
            merged = apply_vol_overlay(merged, btc_daily, VOL_INVERTED)

        r = bybit_sim(merged)
        n_15m = len(ruled)
        early_pct = r["n_early"] / max(n_15m, 1) * 100
        results.append({"rule": name, **r, "early_pct": early_pct})
        print(f"  {name:<20} {r['sr']:>7.3f} {r['dd']:>5.1f}% ${r['pnl']:>10,.0f} {early_pct:>6.1f}% {r['mean_net_bps']:>7.2f}")

    # Sort by SR
    df = pd.DataFrame(results).sort_values("sr", ascending=False)
    best = df.iloc[0]
    print(f"\n  BEST: {best['rule']} SR={best['sr']:.3f}")

    # Year-by-year for best rule
    print(f"\n{'='*80}")
    print(f"  YEAR-BY-YEAR: {best['rule']} vs fixed_T4")
    print(f"{'='*80}")

    baseline_sr = df[df["rule"] == "fixed_T4"]["sr"].values[0]
    best_daily = best["daily"]
    fixed_daily = df[df["rule"] == "fixed_T4"]["daily"].values[0]

    for year in sorted(best_daily.index.year.unique()):
        y_best = best_daily[best_daily.index.year == year]
        y_fixed = fixed_daily[fixed_daily.index.year == year]
        if len(y_best) > 20:
            sr_b = float(y_best.mean() / y_best.std() * np.sqrt(365.25)) if y_best.std() > 0 else 0
            sr_f = float(y_fixed.mean() / y_fixed.std() * np.sqrt(365.25)) if y_fixed.std() > 0 else 0
            print(f"  {year}: best={sr_b:.3f}  fixed_T4={sr_f:.3f}  delta={sr_b-sr_f:+.3f}")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")

    # Return results for Phase 2
    return df


if __name__ == "__main__":
    main()
