#!/usr/bin/env python -u
"""Feature ablation: remove features one by one, find which ones matter.

Strategy: Start with all 46 features. Remove one at a time. Measure SR drop.
Then do cumulative removal of least-important features until SR drops sharply.
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
os.environ["XGB_DEVICE"] = "cpu"
warnings.filterwarnings("ignore")

from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.config import PortfolioConfig, VolOverlayConfig
from signal_engine.pipeline import apply_vol_overlay
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.close_model import close_feature_columns
from signal_engine.parallel_close_wf import parallel_walk_forward_close
from pathlib import Path

CAPITAL = 100_000.0
PORT = PortfolioConfig(
    starting_capital=CAPITAL,
    max_dd=0.15, delever_start=0.075,
    slippage=0.0001,
    max_exposure=0.90, max_per_asset=0.06,
    max_per_strategy=0.40,
    base_sizing_pct=0.02, max_signal_mult=2.0,
    sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
)
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
CACHE = Path("results/close_cache")
SIGNAL_CACHE = Path("results/signals_cache")
FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]


def load_examples():
    path = CACHE / "examples_logreg_15m.parquet"
    return pd.read_parquet(path)


def load_btc_daily():
    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
        return btc_daily
    except Exception:
        return None


def load_other_strategies():
    parts = []
    for name in ["logreg_4h", "cs_momentum"]:
        path = SIGNAL_CACHE / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            for col in ["regime_exposure_mult", "sizing_mult", "strategy", "hold_bars"]:
                if col not in df.columns:
                    if col == "strategy":
                        df[col] = name
                    elif col == "hold_bars":
                        df[col] = 4
                    else:
                        df[col] = 1.0
            df["probability"] = df["probability"].astype(float)
            parts.append(df)
    return parts


def run_with_features(examples, feat_cols, btc_daily, other_strats):
    """Retrain WF close model with given features, run portfolio sim, return SR."""
    ex = examples.copy()
    ex_pred = parallel_walk_forward_close(
        ex, feat_cols, lookback_months=10,
        test_start="2021-01", max_workers=os.cpu_count(),
        use_gpu=False,
    )

    # Select exits
    ex_pred["_row_idx"] = np.arange(len(ex_pred))
    ex_pred["_triggered"] = ex_pred["close_probability"] > 0.6
    triggered = ex_pred[ex_pred["_triggered"]].groupby("trade_id")["_row_idx"].first()
    last_bar = ex_pred.groupby("trade_id")["_row_idx"].last()
    exit_idx = last_bar.copy()
    exit_idx.loc[triggered.index] = triggered
    selected = ex_pred.iloc[exit_idx.values].copy()

    # Build signals
    out = pd.DataFrame({
        "timestamp": selected["timestamp"],
        "asset": selected["asset"],
        "direction": selected["direction"],
        "fwd_return": selected["exit_return"] / selected["direction"],
        "probability": selected["entry_probability"],
        "confidence": selected["entry_confidence"],
        "signal_strength": selected["entry_signal_strength"].abs(),
        "strategy": "logreg_15m",
        "hold_bars": 4,
        "regime_exposure_mult": selected.get("regime_exposure_mult", 1.0),
        "sizing_mult": 1.0,
    })

    merged = pd.concat([out] + other_strats, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    if btc_daily is not None:
        try:
            vol_cfg = VolOverlayConfig(
                enabled=True, vol_window=20, pct_window=60,
                low_pct=0.20, high_pct=0.80,
                low_mult=0.50, mid_mult=1.00, high_mult=1.25,
            )
            merged = apply_vol_overlay(merged, btc_daily, vol_cfg)
        except Exception:
            pass

    sim = simulate_portfolio(merged, PORT)
    if sim is None:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 30:
        return None
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)

    dr = dpnl / CAPITAL
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
    pnl = float(dpnl.sum())
    n_early = int(triggered.notna().sum())
    n_total = len(selected)
    return {"sr": sr, "pnl": pnl, "early_pct": n_early / max(n_total, 1) * 100}


def main():
    t0 = time.time()
    print("=" * 90)
    print("  FEATURE ABLATION — Remove features one by one")
    print("=" * 90)

    examples = load_examples()
    btc_daily = load_btc_daily()
    other_strats = load_other_strategies()
    all_feat_cols = close_feature_columns(FEATURES, regime_timeframes=["1h"])

    print(f"  {len(examples):,} examples, {len(all_feat_cols)} features")

    # Categorize features
    state_features = [f for f in all_feat_cols if f in [
        "entry_direction", "entry_probability", "entry_confidence",
        "entry_signal_strength", "bars_since_entry", "minutes_since_entry",
        "time_to_expiry_minutes", "remaining_decision_steps",
        "current_unrealized_return", "max_favorable_return_since_entry",
        "max_adverse_return_since_entry", "realized_vol_since_entry",
        "return_since_last_decision", "rolling_2bar", "rolling_3bar",
        "distance_from_entry_mfe", "distance_from_entry_mae",
        "mfe_minus_mae", "entry_to_now_trend_slope",
    ]]
    market_features = [f for f in all_feat_cols if f in FEATURES]
    regime_features = [f for f in all_feat_cols if f.startswith("regime_")]

    print(f"  State: {len(state_features)}, Market: {len(market_features)}, Regime: {len(regime_features)}")

    # ================================================================
    # Phase 1: Baseline with all features
    # ================================================================
    print(f"\n--- BASELINE (all {len(all_feat_cols)} features) ---")
    baseline = run_with_features(examples, all_feat_cols, btc_daily, other_strats)
    if baseline is None:
        print("  FAILED")
        return
    print(f"  SR={baseline['sr']:.3f}  PnL=${baseline['pnl']:,.0f}  Early={baseline['early_pct']:.0f}%")

    # ================================================================
    # Phase 2: Remove feature GROUPS first (fast screening)
    # ================================================================
    print(f"\n--- GROUP ABLATION ---")
    groups = {
        "state (trade)": state_features,
        "market (OHLCV)": market_features,
        "regime (1h)": regime_features,
    }

    for group_name, group_cols in groups.items():
        remaining = [f for f in all_feat_cols if f not in group_cols]
        print(f"\n  Without {group_name} ({len(group_cols)} features, {len(remaining)} remaining):")
        r = run_with_features(examples, remaining, btc_daily, other_strats)
        if r:
            delta = r["sr"] - baseline["sr"]
            print(f"    SR={r['sr']:.3f} (delta={delta:+.3f})  PnL=${r['pnl']:,.0f}  Early={r['early_pct']:.0f}%")
        else:
            print(f"    FAILED")

    # ================================================================
    # Phase 3: Remove individual state features (most likely to matter)
    # ================================================================
    print(f"\n--- INDIVIDUAL STATE FEATURE ABLATION ---")
    print(f"  {'Feature':<42} {'SR':>7} {'Delta':>8} {'PnL':>12} {'Early%':>7}")
    print(f"  {'-'*78}")

    individual_results = {}
    for feat in state_features:
        remaining = [f for f in all_feat_cols if f != feat]
        r = run_with_features(examples, remaining, btc_daily, other_strats)
        if r:
            delta = r["sr"] - baseline["sr"]
            individual_results[feat] = {"sr": r["sr"], "delta": delta, "pnl": r["pnl"], "early": r["early_pct"]}
            print(f"  {feat:<42} {r['sr']:>7.3f} {delta:>+8.3f} ${r['pnl']:>11,.0f} {r['early_pct']:>6.0f}%")
        else:
            print(f"  {feat:<42} FAILED")

    # ================================================================
    # Phase 4: Cumulative removal (least important first)
    # ================================================================
    if individual_results:
        print(f"\n--- CUMULATIVE REMOVAL (least important first) ---")
        # Sort by delta (least negative = least important)
        sorted_feats = sorted(individual_results.items(), key=lambda x: x[1]["delta"], reverse=True)

        print(f"\n  Feature importance ranking (by SR drop when removed):")
        for feat, info in sorted(individual_results.items(), key=lambda x: x[1]["delta"]):
            print(f"    {feat:<42} delta={info['delta']:+.3f}")

        removed = []
        current_feats = list(all_feat_cols)
        print(f"\n  {'Removed':<42} {'Remaining':>4} {'SR':>7} {'Delta':>8}")
        print(f"  {'-'*65}")
        print(f"  {'(none)':<42} {len(current_feats):>4} {baseline['sr']:>7.3f} {0:>+8.3f}")

        for feat, info in sorted_feats:
            if feat not in current_feats:
                continue
            current_feats = [f for f in current_feats if f != feat]
            removed.append(feat)
            r = run_with_features(examples, current_feats, btc_daily, other_strats)
            if r:
                delta = r["sr"] - baseline["sr"]
                print(f"  -{feat:<41} {len(current_feats):>4} {r['sr']:>7.3f} {delta:>+8.3f}")
                # Stop if SR dropped sharply
                if r["sr"] < baseline["sr"] * 0.5:
                    print(f"\n  SHARP DROP detected after removing {feat}")
                    print(f"  Remaining features: {current_feats}")
                    break
            else:
                print(f"  -{feat:<41} FAILED")
                break

    print(f"\n{'='*90}")
    print(f"  FEATURE ABLATION COMPLETE -- Total: {time.time()-t0:.0f}s")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
