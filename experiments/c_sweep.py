#!/usr/bin/env python
"""Sweep LogReg C parameter with 13 features.

For each C value: regenerate signals -> apply confidence filter -> bybit_sim.
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.config import (
    SystemConfig, ModelConfig, FeatureConfig, PortfolioConfig,
    RegimeFilterConfig, ConfidenceFilterConfig, DataConfig, SignalConfig,
)
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_regime_filter
from signal_engine.pipeline import apply_vol_overlay
from signal_engine.config import VolOverlayConfig

ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "SOL": "SOLUSDT", "ADA": "ADAUSDT",
    "AVAX": "AVAXUSDT", "DOT": "DOTUSDT", "LINK": "LINKUSDT",
    "NEAR": "NEARUSDT", "UNI": "UNIUSDT", "AAVE": "AAVEUSDT",
    "LTC": "LTCUSDT", "BCH": "BCHUSDT", "FIL": "FILUSDT",
    "FTM": "FTMUSDT", "HBAR": "HBARUSDT", "TRX": "TRXUSDT",
    "SUI": "SUIUSDT", "APT": "APTUSDT", "ENJ": "ENJUSDT",
    "SXP": "SXPUSDT", "ZEC": "ZECUSDT", "XLM": "XLMUSDT",
    "FET": "FETUSDT", "WLD": "WLDUSDT", "BNX": "BNXUSDT",
    "ANKR": "ANKRUSDT", "TAO": "TAOUSDT", "ENA": "ENAUSDT",
}
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]
N_WORKERS = min(os.cpu_count() or 4, 11)


def _process_asset(asset_name, symbol, config):
    """Generate signals for one asset."""
    warnings.filterwarnings("ignore")
    try:
        ohlcv_1m = load_binance_1m(symbol, DATA_DIRS)
    except FileNotFoundError:
        return None

    ohlcv_tf = resample_ohlcv(ohlcv_1m, config.data.timeframe)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    features = compute_features(ohlcv_tf, config.features)

    wf = WalkForwardModel(config, force_monthly_retrain=True)
    signals = wf.generate_signals_batch(ohlcv_tf, features, asset_name)

    if len(signals) == 0:
        return None

    if config.regime_filter.enabled:
        filtered = apply_regime_filter(signals, ohlcv_4h, config.regime_filter)
    else:
        filtered = signals

    return filtered if len(filtered) > 0 else None


def generate_signals(C_val):
    """Generate all signals with given C value."""
    config = SystemConfig(
        model=ModelConfig(C=C_val, penalty="l1", solver="saga",
                          max_iter=1000, seed=42, fwd_bars=4,
                          signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=FEATURES, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        portfolio=PortfolioConfig(starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
                                  slippage=0.0001, max_exposure=0.80, max_per_asset=0.06),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe="15min", regime_timeframe="4h"),
        assets=ASSETS,
    )

    all_signals = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(_process_asset, name, sym, config): name
            for name, sym in ASSETS.items()
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_signals.append(result)

    if not all_signals:
        return pd.DataFrame()

    signals = pd.concat(all_signals, ignore_index=True)
    signals["strategy"] = "logreg_15m"
    signals["hold_bars"] = 4
    return signals


def bybit_sim(signals, rt_bps=4.5, capital=100_000, base_pct=0.02, max_mult=2.0):
    """Quick Bybit sim."""
    sub = signals.sort_values("timestamp").reset_index(drop=True)
    rt = rt_bps / 10000.0
    gross = sub["direction"].values * sub["fwd_return"].values
    net = gross - rt
    prob = sub["probability"].values
    edge = np.where(np.isfinite(prob), np.abs(2 * prob - 1), 0.1)
    edge = np.clip(edge, 0, 1)
    sizing_mult = sub["sizing_mult"].values if "sizing_mult" in sub.columns else np.ones(len(sub))

    eq = capital
    pnl = np.zeros(len(sub))
    for i in range(len(sub)):
        sz = min(eq * base_pct * edge[i] * sizing_mult[i], eq * base_pct * max_mult)
        pnl[i] = sz * net[i]
        eq += pnl[i]

    ts = pd.to_datetime(sub["timestamp"].values)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    daily = pd.Series(pnl, index=ts).resample("1D").sum()
    daily = daily[daily != 0]
    sr = float(daily.mean() / daily.std() * np.sqrt(365.25)) if len(daily) > 30 and daily.std() > 0 else 0
    cum_eq = capital + daily.cumsum()
    dd = float(((cum_eq.cummax() - cum_eq) / cum_eq.cummax()).max() * 100) if len(cum_eq) > 0 else 0

    return {"sr": sr, "pnl": float(pnl.sum()), "dd": dd, "daily": daily,
            "n": len(sub), "net_bps": float(net.mean() * 10000)}


def main():
    T0 = time.time()
    print("=" * 80)
    print("  C PARAMETER SWEEP — 13 features, logreg_15m only")
    print("=" * 80)

    # Load BTC daily for vol overlay
    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
    except Exception as e:
        print(f"BTC load failed: {e}")
        btc_daily = None

    vol_inv = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                low_pct=0.20, high_pct=0.80,
                                low_mult=1.25, mid_mult=1.00, high_mult=0.50)

    # Also load cached 4h and cs_momentum for combined results
    sigs_4h = pd.read_parquet("results/signals_cache/logreg_4h.parquet") if Path("results/signals_cache/logreg_4h.parquet").exists() else pd.DataFrame()
    sigs_cs = pd.read_parquet("results/signals_cache/cs_momentum.parquet") if Path("results/signals_cache/cs_momentum.parquet").exists() else pd.DataFrame()

    results = []
    for C_val in [0.01, 0.05, 0.1, 0.2, 0.5]:
        print(f"\n--- C={C_val} ---")
        t0 = time.time()
        sigs_15m = generate_signals(C_val)
        gen_time = time.time() - t0
        print(f"  Generated {len(sigs_15m):,} logreg_15m signals in {gen_time:.0f}s")

        if len(sigs_15m) < 100:
            print("  TOO FEW signals, skipping")
            continue

        # Save C=0.1 to cache (it's the baseline)
        if abs(C_val - 0.1) < 0.001:
            Path("results/signals_cache").mkdir(parents=True, exist_ok=True)
            sigs_15m.to_parquet("results/signals_cache/logreg_15m.parquet", index=False)
            print("  Saved to signals_cache/logreg_15m.parquet")

        # Merge with 4h + cs
        parts = [sigs_15m]
        if len(sigs_4h) > 0:
            parts.append(sigs_4h.copy())
        if len(sigs_cs) > 0:
            parts.append(sigs_cs.copy())
        merged = pd.concat(parts, ignore_index=True)
        for col in ["regime_exposure_mult", "sizing_mult", "hold_bars"]:
            if col not in merged.columns:
                merged[col] = 4 if col == "hold_bars" else 1.0
            merged[col] = merged[col].fillna(4 if col == "hold_bars" else 1.0)
        merged["probability"] = merged["probability"].astype(float)
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        # Apply inverted vol overlay
        if btc_daily is not None:
            merged = apply_vol_overlay(merged, btc_daily, vol_inv)

        # Test with different confidence filters
        for min_conf in [0.00, 0.06, 0.08, 0.10]:
            if "confidence" in merged.columns:
                filtered = merged[merged["confidence"] >= min_conf].sort_values("timestamp")
            else:
                filtered = merged
            if len(filtered) < 100:
                continue
            r = bybit_sim(filtered)
            results.append({
                "C": C_val, "min_conf": min_conf,
                "n_signals": len(filtered), "sr": r["sr"],
                "dd": r["dd"], "pnl": r["pnl"], "net_bps": r["net_bps"],
            })
            print(f"  C={C_val} conf>={min_conf:.2f}: "
                  f"{len(filtered):>7,} signals  SR={r['sr']:.3f}  DD={r['dd']:.1f}%  "
                  f"PnL=${r['pnl']:,.0f}  net={r['net_bps']:.2f}bps")

    # Summary
    df = pd.DataFrame(results)
    if len(df) > 0:
        print(f"\n{'='*80}")
        print("  SUMMARY (sorted by SR, conf>=0.08)")
        print(f"{'='*80}")
        sub = df[df["min_conf"] == 0.08].sort_values("sr", ascending=False)
        for _, row in sub.iterrows():
            print(f"  C={row['C']:<5}  SR={row['sr']:.3f}  DD={row['dd']:.1f}%  "
                  f"PnL=${row['pnl']:,.0f}  signals={row['n_signals']:,}")

    print(f"\nTotal runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
