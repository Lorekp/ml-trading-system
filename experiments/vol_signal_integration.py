#!/usr/bin/env python
"""Integrate vol prediction model into logreg perp trader.

Three experiments:
  A: Replace backward-looking vol sizing with predicted vol (BTC only)
  B: Vol prediction as BTC confidence/sizing modifier
  C: Vol prediction as standalone BTC direction signal

Usage:
    PYTHONPATH="src;." python experiments/vol_signal_integration.py
"""

import sys, os, time, warnings, pickle, logging
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

from experiments.drawdown_improvements import (
    simulate_with_improvements, apply_bar_level_sl, CACHE_DIR, TF_RESAMPLE,
)
from experiments.circuit_breaker_v2_experiment import apply_vol
from experiments.dynamic_asset_v2_experiment import assemble_signals, OBVIOUS_CORE
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig
from signal_engine.slippage import load_per_symbol_calibration
from signal_engine.execution import get_vip_tier

CAP = 200_000.0


def load_vol_predictions():
    """Load vol model predictions and convert to realized vol."""
    pred = pd.read_parquet("vol_model/output/BTCUSDT_pruned_svi_predictions.parquet")
    pred["rv_pred"] = np.exp(pred["y_pred"])  # decimal RV
    pred["rv_true"] = np.exp(pred["y_true"])
    # Make timezone-naive
    if pred.index.tz is not None:
        pred.index = pred.index.tz_localize(None)
    return pred


def load_market_iv():
    """Load market IV for RV vs IV comparison."""
    bars = pd.read_parquet("vol_arb/output/BTC_vol_arb_svi.parquet")
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars[["iv", "vol_ann"]].dropna()


# ── Experiment A: Predicted vol sizing ────────────────────────

def apply_predicted_vol_sizing(signals, vol_pred):
    """Replace backward-looking vol sizing with predicted vol for BTC signals.

    Same formula as apply_vol: sizing_mult *= clip(0.025 / vol, 0.3, 1.5)
    But using PREDICTED future vol instead of trailing 20d vol.
    """
    out = signals.copy()
    if "sizing_mult" not in out.columns:
        out["sizing_mult"] = 1.0

    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)

    btc_mask = out["asset"] == "BTC"
    btc_idx = out.index[btc_mask]

    for i in btc_idx:
        sig_ts = ts.iloc[i]
        # Find nearest prediction (backward — no lookahead)
        idx = vol_pred.index.searchsorted(sig_ts, side="right") - 1
        if 0 <= idx < len(vol_pred):
            rv = vol_pred["rv_pred"].iloc[idx]
            if pd.notna(rv) and rv > 0:
                mult = np.clip(0.025 / rv, 0.3, 1.5)
                out.at[i, "sizing_mult"] *= mult

    return out


# ── Experiment B: Vol confidence modifier ─────────────────────

def apply_vol_confidence_modifier(signals, vol_pred):
    """Boost/reduce BTC sizing based on predicted vol regime.

    High predicted vol → trending market → boost sizing.
    Low predicted vol → choppy → reduce sizing.
    """
    out = signals.copy()
    if "sizing_mult" not in out.columns:
        out["sizing_mult"] = 1.0

    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)

    # Rolling stats for z-scoring (96 bars = 1 day of 15m)
    rv_mean = vol_pred["rv_pred"].rolling(96 * 7, min_periods=96).mean()
    rv_std = vol_pred["rv_pred"].rolling(96 * 7, min_periods=96).std()

    btc_mask = out["asset"] == "BTC"
    btc_idx = out.index[btc_mask]

    for i in btc_idx:
        sig_ts = ts.iloc[i]
        idx = vol_pred.index.searchsorted(sig_ts, side="right") - 1
        if 0 <= idx < len(vol_pred):
            rv = vol_pred["rv_pred"].iloc[idx]
            mu = rv_mean.iloc[idx] if idx < len(rv_mean) else rv
            std = rv_std.iloc[idx] if idx < len(rv_std) else 0.1
            if pd.notna(rv) and pd.notna(mu) and std > 0.001:
                z = (rv - mu) / std
                # High predicted vol → boost (trending), low → reduce (choppy)
                conf_mult = np.clip(1.0 + 0.3 * z, 0.5, 2.0)
                out.at[i, "sizing_mult"] *= conf_mult

    return out


# ── Experiment C: Vol direction signal ────────────────────────

def create_vol_direction_signals(vol_pred, market_iv_df):
    """Create BTC direction signals from RV vs IV divergence.

    When predicted RV > market IV → vol underpriced → expect trend → long BTC
    When predicted RV < market IV → vol overpriced → expect mean reversion → short BTC
    """
    # Align vol predictions with market IV
    merged = vol_pred[["rv_pred"]].join(market_iv_df[["iv"]], how="inner")
    merged = merged.dropna()

    signals = []
    for i in range(len(merged)):
        ts = merged.index[i]
        rv = merged["rv_pred"].iloc[i] * 100  # convert to pct for comparison
        iv = merged["iv"].iloc[i]

        spread = rv - iv  # positive = vol underpriced, negative = overpriced

        # Only signal when spread is significant (> 5 vol points)
        if abs(spread) < 5.0:
            continue

        direction = 1.0 if spread > 0 else -1.0
        # Probability: map spread to [0.5, 0.8] range
        prob = 0.5 + np.clip(abs(spread) / 40.0, 0, 0.3)

        signals.append({
            "timestamp": ts,
            "asset": "BTC",
            "direction": direction,
            "probability": prob if direction > 0 else 1.0 - prob,
            "confidence": abs(prob - 0.5),
            "signal_strength": np.clip(abs(spread) / 20.0, 0, 1),
            "strategy": "vol_arb_15m",
            "hold_bars": 4,  # 1 hour hold
            "sizing_mult": 1.0,
            "regime_exposure_mult": 1.0,
            "fwd_return": 0.0,  # will be filled by simulator from price data
        })

    df = pd.DataFrame(signals)
    print(f"  Vol direction signals: {len(df):,} ({(df['direction'] == 1).sum()} long, "
          f"{(df['direction'] == -1).sum()} short)")
    return df


# ── Runner (reuses yolo_compound logic) ───────────────────────

def run_experiment(prepared, label, close_cache, slippage_model, asset_volumes):
    """Run one experiment config through the simulator."""
    port = PortfolioConfig(
        starting_capital=CAP, max_dd=0.30, delever_start=0.20, slippage=0.0001,
        max_exposure=0.95, max_per_asset=0.25, max_per_strategy=0.50,
        base_sizing_pct=0.200, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
    )
    breaker = {"window": 10, "threshold": -0.01, "floor": 0.10,
               "up_threshold": 0.01, "cap": 1.5}
    imp = {"max_signals_per_bar": 20, "max_risk_per_trade": 0.03, "sl_cap": 0.75}

    # Apply bar-level SL if close cache available
    if close_cache:
        prepared, sl_count = apply_bar_level_sl(
            prepared, close_cache,
            max_risk_per_trade=imp.get("max_risk_per_trade", 0.03),
            sl_cap=imp.get("sl_cap", 0.75),
            base_sizing_pct=port.base_sizing_pct,
        )

    sim = simulate_with_improvements(
        prepared, port, breaker, imp,
        slippage_model=slippage_model, asset_volumes=asset_volumes,
    )
    if sim is None:
        return None

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 10:
        return None
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)

    # Compound equity
    running_eq = CAP
    daily_rets = []
    eq_series = []
    for dp in dpnl.values:
        r = dp / running_eq if running_eq > 0 else 0.0
        daily_rets.append(r)
        running_eq = max(running_eq + dp, 0.0)
        eq_series.append(running_eq)
    daily_rets = np.array(daily_rets)
    eq_arr = np.array(eq_series)

    sr = float(daily_rets.mean() / daily_rets.std() * np.sqrt(365.25)) if daily_rets.std() > 0 else 0
    peak = np.maximum.accumulate(eq_arr)
    dd = np.where(peak > 0, (peak - eq_arr) / peak, 0)
    max_dd = float(dd.max() * 100)
    final_eq = eq_arr[-1]
    cagr = ((final_eq / CAP) ** (365.25 / len(dpnl)) - 1) * 100
    n_trades = sim["n_trades"]

    return {
        "label": label, "sr": sr, "final_eq": final_eq,
        "cagr": cagr, "max_dd": max_dd, "n_trades": n_trades,
        "n_days": len(dpnl),
        "dpnl": dpnl, "eq_arr": eq_arr,
    }


def main():
    T0 = time.time()

    # ── Load data ──
    print("Loading data...")
    with open(CACHE_DIR / "v2_prep_cache.pkl", "rb") as f:
        cached = pickle.load(f)
    hold_sigs = cached["hold_sigs"]
    vol_series = cached["vol_series"]
    asset_volumes = cached["asset_volumes"]

    vol_pred = load_vol_predictions()
    market_iv = load_market_iv()
    print(f"  Vol predictions: {len(vol_pred):,} bars ({vol_pred.index[0]} to {vol_pred.index[-1]})")
    print(f"  Market IV: {len(market_iv):,} bars")

    # ── Assemble baseline signals ──
    months = pd.period_range("2020-07", "2026-04", freq="M")

    def _s8_sel(ms):
        active_core = [a for a in OBVIOUS_CORE if a in hold_sigs]
        return active_core + [a for a in hold_sigs if a not in OBVIOUS_CORE][:21]

    merged = assemble_signals(hold_sigs, _s8_sel, months)
    prepared_base = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared_base.columns:
            prepared_base[c] = 1.0
        prepared_base[c] = prepared_base[c].fillna(1.0)
    prepared_base["probability"] = prepared_base["probability"].astype(float)
    prepared_base = apply_vol(
        prepared_base.sort_values("timestamp").reset_index(drop=True), vol_series
    )

    print(f"  Baseline signals: {len(prepared_base):,}")
    btc_count = (prepared_base["asset"] == "BTC").sum()
    print(f"  BTC signals: {btc_count:,} ({100*btc_count/len(prepared_base):.1f}%)")

    # ── Load close cache for bar-level SL ──
    close_cache_path = CACHE_DIR / "close_cache.pkl"
    close_cache = None
    if close_cache_path.exists():
        with open(close_cache_path, "rb") as f:
            close_cache = pickle.load(f)
        print(f"  Close cache: {len(close_cache)} pairs")

    # ── Slippage model ──
    per_symbol_cal = load_per_symbol_calibration()
    vip_name, maker_bps, taker_bps = get_vip_tier(CAP)
    slippage_model = {
        "calibration": per_symbol_cal,
        "vip_tier": (vip_name, maker_bps, taker_bps),
        "max_participation_rate": 0.02,
        "guard_pct": 0.50,
    }

    print(f"\n{'='*100}")
    print("  VOL MODEL INTEGRATION EXPERIMENTS")
    print(f"{'='*100}\n")

    results = {}

    # ── Baseline ──
    print("Running BASELINE (no vol model)...")
    r = run_experiment(prepared_base.copy(), "BASELINE", close_cache, slippage_model, asset_volumes)
    if r:
        results["BASELINE"] = r
        print(f"  BASELINE: ${r['final_eq']:,.0f} | SR {r['sr']:.2f} | CAGR {r['cagr']:.0f}% | DD {r['max_dd']:.1f}%")

    # ── Exp A: Predicted vol sizing ──
    print("\nRunning EXP A: Predicted vol sizing (BTC only)...")
    prep_a = apply_predicted_vol_sizing(prepared_base.copy(), vol_pred)
    r = run_experiment(prep_a, "EXP_A_PRED_VOL", close_cache, slippage_model, asset_volumes)
    if r:
        results["EXP_A"] = r
        print(f"  EXP A: ${r['final_eq']:,.0f} | SR {r['sr']:.2f} | CAGR {r['cagr']:.0f}% | DD {r['max_dd']:.1f}%")

    # ── Exp B: Vol confidence modifier ──
    print("\nRunning EXP B: Vol confidence modifier (BTC only)...")
    prep_b = apply_vol_confidence_modifier(prepared_base.copy(), vol_pred)
    r = run_experiment(prep_b, "EXP_B_VOL_CONF", close_cache, slippage_model, asset_volumes)
    if r:
        results["EXP_B"] = r
        print(f"  EXP B: ${r['final_eq']:,.0f} | SR {r['sr']:.2f} | CAGR {r['cagr']:.0f}% | DD {r['max_dd']:.1f}%")

    # ── Exp C: Vol direction signal ──
    print("\nRunning EXP C: Vol direction signal (added to BTC)...")
    vol_sigs = create_vol_direction_signals(vol_pred, market_iv)
    if len(vol_sigs) > 0:
        # Merge vol signals with baseline — ensure consistent timezone
        base_c = prepared_base.copy()
        base_ts = pd.to_datetime(base_c["timestamp"])
        if base_ts.dt.tz is not None:
            base_c["timestamp"] = base_ts.dt.tz_localize(None)
        vol_ts = pd.to_datetime(vol_sigs["timestamp"])
        if vol_ts.dt.tz is not None:
            vol_sigs["timestamp"] = vol_ts.dt.tz_localize(None)
        prep_c = pd.concat([base_c, vol_sigs], ignore_index=True)
        prep_c = prep_c.sort_values("timestamp").reset_index(drop=True)
        r = run_experiment(prep_c, "EXP_C_VOL_SIG", close_cache, slippage_model, asset_volumes)
        if r:
            results["EXP_C"] = r
            print(f"  EXP C: ${r['final_eq']:,.0f} | SR {r['sr']:.2f} | CAGR {r['cagr']:.0f}% | DD {r['max_dd']:.1f}%")

    # ── Exp A+B combined ──
    print("\nRunning EXP A+B: Predicted vol sizing + confidence modifier...")
    prep_ab = apply_predicted_vol_sizing(prepared_base.copy(), vol_pred)
    prep_ab = apply_vol_confidence_modifier(prep_ab, vol_pred)
    r = run_experiment(prep_ab, "EXP_AB_COMBINED", close_cache, slippage_model, asset_volumes)
    if r:
        results["EXP_AB"] = r
        print(f"  EXP AB: ${r['final_eq']:,.0f} | SR {r['sr']:.2f} | CAGR {r['cagr']:.0f}% | DD {r['max_dd']:.1f}%")

    # ── Summary ──
    print(f"\n{'='*100}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Config':<25s} {'Final Eq':>14s} {'Sharpe':>8s} {'CAGR':>8s} {'MaxDD':>8s} {'Trades':>8s} {'vs Base':>12s}")
    print(f"  {'-'*90}")

    base_eq = results.get("BASELINE", {}).get("final_eq", CAP)
    for name, r in results.items():
        vs = f"${r['final_eq'] - base_eq:>+10,.0f}" if name != "BASELINE" else "---"
        print(f"  {name:<25s} ${r['final_eq']:>13,.0f} {r['sr']:>8.2f} {r['cagr']:>7.0f}% "
              f"{r['max_dd']:>7.1f}% {r['n_trades']:>8,} {vs:>12s}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
