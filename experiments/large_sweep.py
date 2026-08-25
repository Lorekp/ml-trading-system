#!/usr/bin/env python
"""Large parameter sweep across timeframes, lookbacks, sizing, and confidence.

Timeframes: 1h, 4h, 8h, 12h, 1d
Lookbacks: 2, 4, 6, 10 months
Confidence: 0.04, 0.06, 0.08, 0.10, 0.12
Sizing: 0.02, 0.04, 0.06
C: 0.05, 0.1, 0.2
fwd_bars: 2, 4, 6

All through simulate_portfolio + Bybit VIP0.
"""
import sys, os, time, warnings, itertools
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

# Full 30-asset universe (exclude ENA, TAO — no L2, but irrelevant for OHLCV-only)
ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "ADA": "ADAUSDT", "AVAX": "AVAXUSDT", "DOT": "DOTUSDT",
    "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "UNI": "UNIUSDT", "AAVE": "AAVEUSDT",
    "LTC": "LTCUSDT", "BCH": "BCHUSDT", "HBAR": "HBARUSDT", "TRX": "TRXUSDT",
    "SUI": "SUIUSDT", "APT": "APTUSDT", "ENJ": "ENJUSDT", "SXP": "SXPUSDT",
    "ZEC": "ZECUSDT", "XLM": "XLMUSDT", "FET": "FETUSDT", "WLD": "WLDUSDT",
    "ANKR": "ANKRUSDT", "TAO": "TAOUSDT", "ENA": "ENAUSDT",
    "FTM": "FTMUSDT", "FIL": "FILUSDT", "BNX": "BNXUSDT",
}

OHLCV = ["ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
         "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]

VOL_INV = VolOverlayConfig(
    enabled=True, vol_window=20, pct_window=60,
    low_pct=0.20, high_pct=0.80, low_mult=1.25, mid_mult=1.00, high_mult=0.50,
)

# ============================================================
# SWEEP DIMENSIONS
# ============================================================
TIMEFRAMES = ["1h", "4h", "8h", "12h", "1d"]
LOOKBACKS = [2, 4, 6, 10]
FWD_BARS = [2, 4, 6]
C_VALUES = [0.05, 0.1, 0.2]
CONFIDENCE = [0.04, 0.06, 0.08, 0.10, 0.12]
SIZING = [0.02, 0.04, 0.06]

# Pre-generate signals for each (timeframe × lookback × C × fwd_bars) combo,
# then sweep confidence × sizing on the cached signals (instant).
# This avoids re-generating signals for each confidence/sizing combo.


def _gen_asset_signals(args):
    """Generate signals for one asset with specific config."""
    name, sym, tf, fwd_bars, C_val = args
    warnings.filterwarnings("ignore")

    config = SystemConfig(
        model=ModelConfig(C=C_val, penalty="l1", solver="saga", max_iter=1000, seed=42,
                          fwd_bars=fwd_bars, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=OHLCV, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe=tf, regime_timeframe="4h"),
        assets=ASSETS,
    )

    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
    except FileNotFoundError:
        return None
    ohlcv_tf = resample_ohlcv(ohlcv_1m, tf)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    feat = compute_features(ohlcv_tf, config.features)

    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0:
        return None
    if config.regime_filter.enabled:
        sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    if len(sigs) > 0:
        sigs["strategy"] = f"logreg_{tf}"
        sigs["hold_bars"] = fwd_bars
        return sigs
    return None


def generate_signals_batch(tf, fwd_bars, C_val):
    """Generate signals for all assets with one config."""
    tasks = [(n, s, tf, fwd_bars, C_val) for n, s in ASSETS.items()]
    all_sigs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_gen_asset_signals, t): t[0] for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            if r is not None and len(r) > 0:
                all_sigs.append(r)
    if all_sigs:
        return pd.concat(all_sigs, ignore_index=True)
    return pd.DataFrame()


_btc_daily_cache = None

def evaluate(signals, conf_thresh, base_sizing):
    """Apply confidence filter + sizing + portfolio sim + Bybit costs."""
    if len(signals) == 0:
        return None

    merged = signals.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in merged.columns:
            merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)

    # Confidence filter
    if "confidence" in merged.columns:
        merged = merged[merged["confidence"] >= conf_thresh].reset_index(drop=True)
    if len(merged) < 50:
        return None
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Vol overlay (use pre-loaded cache)
    if _btc_daily_cache is not None:
        merged = apply_vol_overlay(merged, _btc_daily_cache, VOL_INV)

    port = PortfolioConfig(
        starting_capital=CAPITAL, max_dd=0.15, delever_start=0.075,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
        max_per_strategy=0.50, base_sizing_pct=base_sizing, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
    )

    sim = simulate_portfolio(merged, port)
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
    if dr.std() == 0 or len(dr) < 30:
        return None
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq = (1 + dr).cumprod()
    dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)

    return {"sr": sr, "sortino": sortino, "dd": dd,
            "pnl": float(dpnl_adj.sum()), "n_trades": n_trades,
            "n_signals": len(merged)}


def main():
    T0 = time.time()
    print("=" * 90)
    print("  LARGE PARAMETER SWEEP")
    print(f"  {len(TIMEFRAMES)} TFs × {len(FWD_BARS)} fwd × {len(C_VALUES)} C × "
          f"{len(CONFIDENCE)} conf × {len(SIZING)} sizing = "
          f"{len(TIMEFRAMES)*len(FWD_BARS)*len(C_VALUES)*len(CONFIDENCE)*len(SIZING)} combos")
    print("=" * 90)

    results = []

    # Phase 1: Generate signals for each (TF × fwd × C) combo
    # Then sweep (conf × sizing) on cached signals (instant)
    signal_cache = {}
    gen_combos = list(itertools.product(TIMEFRAMES, FWD_BARS, C_VALUES))
    print(f"\nPhase 1: Generating signals for {len(gen_combos)} (TF × fwd × C) combos...")

    for i, (tf, fb, C) in enumerate(gen_combos):
        key = (tf, fb, C)
        t0 = time.time()
        sigs = generate_signals_batch(tf, fb, C)
        elapsed = time.time() - t0
        n = len(sigs) if len(sigs) > 0 else 0
        signal_cache[key] = sigs
        print(f"  [{i+1}/{len(gen_combos)}] tf={tf} fwd={fb} C={C}: "
              f"{n:,} signals ({elapsed:.0f}s)")

    # Pre-load BTC daily ONCE for vol overlay (was loading 675 times!)
    print("\nLoading BTC daily for vol overlay (once)...")
    global _btc_daily_cache
    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        _btc_daily_cache = resample_ohlcv(btc_1m, "1D")["close"]
        if _btc_daily_cache.index.tz is not None:
            _btc_daily_cache.index = _btc_daily_cache.index.tz_localize(None)
        print(f"  Cached BTC daily: {len(_btc_daily_cache)} days")
    except Exception:
        _btc_daily_cache = None

    # Phase 2: Sweep confidence × sizing on cached signals
    print(f"\nPhase 2: Sweeping {len(CONFIDENCE)} conf × {len(SIZING)} sizing on each...")
    total_evals = sum(1 for s in signal_cache.values() if len(s) > 0) * len(CONFIDENCE) * len(SIZING)
    done_evals = 0

    for (tf, fb, C), sigs in signal_cache.items():
        if len(sigs) == 0:
            continue
        for conf in CONFIDENCE:
            for sizing in SIZING:
                r = evaluate(sigs, conf, sizing)
                if r:
                    results.append({
                        "tf": tf, "fwd_bars": fb, "C": C,
                        "conf": conf, "sizing": sizing, **r,
                    })
                done_evals += 1
                if done_evals % 50 == 0:
                    print(f"  {done_evals}/{total_evals} evaluations done...")

    # Summary
    df = pd.DataFrame(results)
    if len(df) == 0:
        print("No results!")
        return

    print(f"\n{'='*90}")
    print(f"  RESULTS: {len(df)} valid configs")
    print(f"{'='*90}")

    # Top 20 by SR
    print(f"\n  TOP 20 BY SR:")
    print(f"  {'TF':<5} {'fwd':>3} {'C':>5} {'conf':>5} {'sz%':>4} {'SR':>7} {'Sort':>7} "
          f"{'DD%':>5} {'PnL':>11} {'Trades':>8}")
    print("  " + "-" * 70)
    for _, row in df.nlargest(20, "sr").iterrows():
        print(f"  {row['tf']:<5} {row['fwd_bars']:>3} {row['C']:>5.2f} {row['conf']:>5.2f} "
              f"{row['sizing']:>4.0%} {row['sr']:>7.3f} {row['sortino']:>7.3f} "
              f"{row['dd']:>4.1f}% ${row['pnl']:>9,.0f} {row['n_trades']:>8,}")

    # Top 20 by PnL (with SR > 2.0)
    viable = df[df["sr"] > 2.0]
    print(f"\n  TOP 20 BY PnL (SR > 2.0):")
    print(f"  {'TF':<5} {'fwd':>3} {'C':>5} {'conf':>5} {'sz%':>4} {'SR':>7} {'Sort':>7} "
          f"{'DD%':>5} {'PnL':>11} {'Trades':>8}")
    print("  " + "-" * 70)
    for _, row in viable.nlargest(20, "pnl").iterrows():
        print(f"  {row['tf']:<5} {row['fwd_bars']:>3} {row['C']:>5.2f} {row['conf']:>5.2f} "
              f"{row['sizing']:>4.0%} {row['sr']:>7.3f} {row['sortino']:>7.3f} "
              f"{row['dd']:>4.1f}% ${row['pnl']:>9,.0f} {row['n_trades']:>8,}")

    # Best per timeframe
    print(f"\n  BEST SR PER TIMEFRAME:")
    for tf in TIMEFRAMES:
        tf_df = df[df["tf"] == tf]
        if len(tf_df) > 0:
            best = tf_df.loc[tf_df["sr"].idxmax()]
            print(f"  {tf:<5}: SR={best['sr']:.3f} (fwd={best['fwd_bars']}, C={best['C']:.2f}, "
                  f"conf={best['conf']:.2f}, sz={best['sizing']:.0%}) "
                  f"DD={best['dd']:.1f}% PnL=${best['pnl']:,.0f}")

    # Save full results
    df.to_csv("experiments/large_sweep_results.csv", index=False)
    print(f"\n  Full results saved to experiments/large_sweep_results.csv")
    print(f"  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
