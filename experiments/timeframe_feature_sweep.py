#!/usr/bin/env python
"""Multi-timeframe feature sweep.

For each timeframe (1m, 5m, 10m, 15m, 1h, 4h), test:
- OHLCV features only
- OHLCV + micro top 3
- All combos

Uses walk-forward LogReg, 2-month lookback, production config.
Restricted to May 2023 — Mar 2024 (L2 data period).
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_regime_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.pipeline import apply_vol_overlay
from signal_engine.config import *

CAPITAL = 100_000.0
BYBIT_EXTRA = 3.5
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
L2_DIR = Path("data/l2-features")

# 25 assets with micro data
ASSETS = {}
for f in L2_DIR.glob("*_micro_15m.parquet"):
    sym = f.stem.replace("_micro_15m", "")
    ASSETS[sym.replace("USDT", "")] = sym

OHLCV_13 = [
    "ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore",
    "rsi_14","stoch_rsi","macd_hist","bb_pct_b","atr_14",
    "dist_swing_high","dist_swing_low",
]
MICRO_3 = ["ms_depth_imb_1pct","ms_depth_imb_2pct","ms_trade_imbalance"]

PORT = PortfolioConfig(
    starting_capital=CAPITAL, max_dd=0.15, delever_start=0.075,
    slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
    max_per_strategy=0.50, base_sizing_pct=0.04, max_signal_mult=2.0,
    sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
)
VOL_INV = VolOverlayConfig(
    enabled=True, vol_window=20, pct_window=60,
    low_pct=0.20, high_pct=0.80, low_mult=1.25, mid_mult=1.00, high_mult=0.50,
)

# Timeframe → fwd_bars mapping
# Skip 1min (impractical: too slow to train, can't cover costs at 1m frequency)
TF_CONFIG = {
    "5min":  {"tf": "5min",  "fwd_bars": 12},   # 12 × 5m = 1h
    "15min": {"tf": "15min", "fwd_bars": 4},    # 4 × 15m = 1h (production)
    "1h":    {"tf": "1h",    "fwd_bars": 4},    # 4 × 1h = 4h
    "4h":    {"tf": "4h",    "fwd_bars": 4},    # 4 × 4h = 16h
}


def _process_asset(args):
    """Generate signals for one asset at one timeframe with given features."""
    name, sym, timeframe, fwd_bars, feature_list, use_micro, micro_tf = args
    warnings.filterwarnings("ignore")

    config = SystemConfig(
        model=ModelConfig(C=0.1, penalty="l1", solver="saga", max_iter=1000, seed=42,
                          fwd_bars=fwd_bars, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=feature_list, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe=timeframe,
                        regime_timeframe="4h"),
        assets=ASSETS,
    )

    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
    except FileNotFoundError:
        return None

    ohlcv_tf = resample_ohlcv(ohlcv_1m, timeframe)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    feat = compute_features(ohlcv_tf, config.features)

    if use_micro:
        # Load micro features at 15m resolution, resample to target timeframe
        micro_path = L2_DIR / f"{sym}_micro_15m.parquet"
        if micro_path.exists():
            micro = pd.read_parquet(micro_path)
            if micro.index.tz is not None and feat.index.tz is not None:
                pass  # both UTC
            elif micro.index.tz is not None:
                micro.index = micro.index.tz_localize(None)
            elif feat.index.tz is not None:
                micro.index = micro.index.tz_localize(feat.index.tz)

            # Resample micro to target timeframe if different from 15m
            if timeframe != "15min":
                micro_resampled = micro[MICRO_3].resample(timeframe).mean()
            else:
                micro_resampled = micro[MICRO_3]

            orig_idx = feat.index
            feat = feat.join(micro_resampled, how="left")
            feat.index = orig_idx
            for c in MICRO_3:
                if c in feat.columns:
                    feat[c] = feat[c].fillna(0)

    # Monthly retrain for 15m+, quarterly for faster TFs (too much data otherwise)
    force_monthly = timeframe in ("15min", "1h", "4h")
    wf = WalkForwardModel(config, force_monthly_retrain=force_monthly)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0:
        return None
    if config.regime_filter.enabled:
        sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    if len(sigs) > 0:
        sigs["strategy"] = f"logreg_{timeframe}"
        sigs["hold_bars"] = fwd_bars
        return sigs.to_dict("list")
    return None


def run_config(label, timeframe, fwd_bars, feature_list, use_micro):
    """Run one timeframe × feature config."""
    t0 = time.time()
    tasks = [(n, s, timeframe, fwd_bars, feature_list, use_micro, timeframe)
             for n, s in ASSETS.items()]
    all_sigs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_process_asset, t): t[0] for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            if r:
                all_sigs.append(pd.DataFrame(r))

    if not all_sigs:
        return None

    merged = pd.concat(all_sigs, ignore_index=True)
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in merged.columns:
            merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    if "confidence" in merged.columns:
        merged = merged[merged["confidence"] >= 0.08].reset_index(drop=True)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
        merged = apply_vol_overlay(merged, btc_daily, VOL_INV)
    except Exception:
        pass

    sim = simulate_portfolio(merged, PORT)
    if sim is None:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    n_trades = sim["n_trades"]
    avg_sz = float(sim["sizes"][~sim["skipped"]].mean()) if (~sim["skipped"]).any() else 0
    total_extra = n_trades * avg_sz * BYBIT_EXTRA / 10000
    dpnl_adj = dpnl - total_extra / max(len(dpnl), 1)

    dr = dpnl_adj / CAPITAL
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq = (1 + dr).cumprod()
    dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)

    elapsed = time.time() - t0
    print(f"  {label:<35} SR={sr:.3f}  Sortino={sortino:.3f}  DD={dd:.1f}%  "
          f"PnL=${dpnl_adj.sum():,.0f}  sigs={len(merged):,}  ({elapsed:.0f}s)")
    return {"sr": sr, "sortino": sortino, "dd": dd, "pnl": float(dpnl_adj.sum()),
            "n_signals": len(merged), "n_trades": n_trades}


def main():
    T0 = time.time()
    print("=" * 90)
    print("  MULTI-TIMEFRAME × FEATURE SWEEP (25 assets, Bybit VIP0)")
    print("=" * 90)

    results = []

    for tf_name, tf_cfg in TF_CONFIG.items():
        tf = tf_cfg["tf"]
        fb = tf_cfg["fwd_bars"]
        print(f"\n--- Timeframe: {tf_name} (fwd_bars={fb}) ---")

        # A: OHLCV only
        r = run_config(f"{tf_name} OHLCV", tf, fb, OHLCV_13, False)
        if r:
            results.append({"timeframe": tf_name, "features": "OHLCV", **r})

        # B: OHLCV + micro
        r = run_config(f"{tf_name} OHLCV+micro", tf, fb, OHLCV_13 + MICRO_3, True)
        if r:
            results.append({"timeframe": tf_name, "features": "OHLCV+micro", **r})

    # Summary table
    print(f"\n{'='*90}")
    print("  SUMMARY")
    print(f"{'='*90}")
    df = pd.DataFrame(results)
    if len(df) > 0:
        print(f"\n{'Config':<35} {'SR':>7} {'Sortino':>8} {'DD%':>6} {'PnL':>12} {'Signals':>9}")
        print("-" * 80)
        for _, row in df.sort_values("sr", ascending=False).iterrows():
            print(f"{row['timeframe']+' '+row['features']:<35} {row['sr']:>7.3f} {row['sortino']:>8.3f} "
                  f"{row['dd']:>5.1f}% ${row['pnl']:>10,.0f} {row['n_signals']:>9,}")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
