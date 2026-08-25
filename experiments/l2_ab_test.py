#!/usr/bin/env python
"""L2 A/B Test — OHLCV-only vs OHLCV+L2 depth features.

Uses locally cached L2 features (data/l2-features/*.parquet).
Parallel signal generation across assets.
simulate_portfolio + Bybit VIP0 costs.
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
from signal_engine.config import (
    SystemConfig, ModelConfig, FeatureConfig, PortfolioConfig,
    RegimeFilterConfig, ConfidenceFilterConfig, DataConfig, SignalConfig,
    VolOverlayConfig,
)

CAPITAL = 100_000.0
BYBIT_EXTRA = 3.5
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
L2_DIR = Path("data/l2-features")

ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "ADA": "ADAUSDT", "AVAX": "AVAXUSDT", "DOT": "DOTUSDT",
    "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "UNI": "UNIUSDT", "AAVE": "AAVEUSDT",
    "LTC": "LTCUSDT", "BCH": "BCHUSDT", "HBAR": "HBARUSDT", "TRX": "TRXUSDT",
    "SUI": "SUIUSDT", "APT": "APTUSDT", "ENJ": "ENJUSDT", "SXP": "SXPUSDT",
    "ZEC": "ZECUSDT", "XLM": "XLMUSDT", "FET": "FETUSDT", "WLD": "WLDUSDT",
    "ANKR": "ANKRUSDT", "TAO": "TAOUSDT", "ENA": "ENAUSDT",
}

OHLCV_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]
L2_FEATURES = [
    "l2_depth_imbalance_1pct", "l2_depth_imbalance_2pct",
    "l2_total_bid_depth_2pct", "l2_depth_change_1pct",
]

PORT = PortfolioConfig(
    starting_capital=CAPITAL, max_dd=0.15, delever_start=0.075,
    slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
    max_per_strategy=0.50, base_sizing_pct=0.04, max_signal_mult=2.0,
    sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
)
VOL_INV = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                            low_pct=0.20, high_pct=0.80,
                            low_mult=1.25, mid_mult=1.00, high_mult=0.50)


def _process_asset(asset_name, symbol, feature_list, use_l2):
    """Generate signals for one asset. Runs in worker process."""
    warnings.filterwarnings("ignore")

    config = SystemConfig(
        model=ModelConfig(C=0.1, penalty="l1", solver="saga", max_iter=1000, seed=42, fwd_bars=4,
                          signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=feature_list, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        portfolio=PortfolioConfig(starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
                                  slippage=0.0001, max_exposure=0.80, max_per_asset=0.06),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe="15min", regime_timeframe="4h"),
        assets=ASSETS,
    )

    try:
        ohlcv_1m = load_binance_1m(symbol, DATA_DIRS)
    except FileNotFoundError:
        return None

    ohlcv_tf = resample_ohlcv(ohlcv_1m, "15min")
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    ohlcv_feat = compute_features(ohlcv_tf, config.features)

    # Merge L2 features if requested
    if use_l2:
        l2_path = L2_DIR / f"{symbol}_bookDepth_15m.parquet"
        if l2_path.exists():
            l2 = pd.read_parquet(l2_path)
            # Align timezones: convert L2 to match OHLCV (preserve OHLCV tz)
            if ohlcv_feat.index.tz is not None and l2.index.tz is None:
                l2.index = l2.index.tz_localize(ohlcv_feat.index.tz)
            elif ohlcv_feat.index.tz is None and l2.index.tz is not None:
                l2.index = l2.index.tz_localize(None)
            orig_index = ohlcv_feat.index
            ohlcv_feat = ohlcv_feat.join(l2, how="left")
            ohlcv_feat.index = orig_index  # preserve DatetimeIndex type
            l2_cols = [c for c in ohlcv_feat.columns if c.startswith("l2_")]
            ohlcv_feat[l2_cols] = ohlcv_feat[l2_cols].fillna(0)
        else:
            for col in L2_FEATURES:
                ohlcv_feat[col] = 0.0

    wf = WalkForwardModel(config, force_monthly_retrain=True)
    signals = wf.generate_signals_batch(ohlcv_tf, ohlcv_feat, asset_name)

    if len(signals) == 0:
        return None

    if config.regime_filter.enabled:
        filtered = apply_regime_filter(signals, ohlcv_4h, config.regime_filter)
    else:
        filtered = signals

    if len(filtered) > 0:
        filtered["strategy"] = "logreg_15m"
        filtered["hold_bars"] = 4
        return filtered.to_dict("list")
    return None


def run_variant(label, feature_list, use_l2, n_workers=8):
    """Generate signals for all assets in parallel, run backtest."""
    print(f"\n  Generating signals ({label}, {len(feature_list)} features)...")
    t0 = time.time()

    all_signals = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_asset, name, sym, feature_list, use_l2): name
            for name, sym in ASSETS.items()
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is not None:
                all_signals.append(pd.DataFrame(result))
            if done % 10 == 0:
                print(f"    {done}/{len(ASSETS)} assets...")

    gen_time = time.time() - t0
    if not all_signals:
        return None

    signals = pd.concat(all_signals, ignore_index=True)

    # Add 4h and cs_momentum from cache
    for strat in ["logreg_4h", "cs_momentum"]:
        p = Path(f"results/signals_cache/{strat}.parquet")
        if p.exists():
            o = pd.read_parquet(p)
            for c in ["regime_exposure_mult", "sizing_mult", "hold_bars"]:
                if c not in o.columns:
                    o[c] = 4 if c == "hold_bars" else 1.0
            o["probability"] = o["probability"].astype(float)
            all_signals.append(o)

    merged = pd.concat(all_signals, ignore_index=True)
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in merged.columns:
            merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)

    # Confidence filter
    if "confidence" in merged.columns:
        merged = merged[merged["confidence"] >= 0.08].reset_index(drop=True)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Vol overlay
    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
        merged = apply_vol_overlay(merged, btc_daily, VOL_INV)
    except Exception:
        pass

    total_sigs = len(merged)
    print(f"    {total_sigs:,} signals after filters ({gen_time:.0f}s generation)")

    # Simulate
    sim = simulate_portfolio(merged, PORT)
    if sim is None:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    # Bybit cost
    n_trades = sim["n_trades"]
    avg_sz = float(sim["sizes"][~sim["skipped"]].mean()) if (~sim["skipped"]).any() else 0
    total_extra = n_trades * avg_sz * BYBIT_EXTRA / 10000
    daily_cost = total_extra / max(len(dpnl), 1)
    dpnl_adj = dpnl - daily_cost

    dr = dpnl_adj / CAPITAL
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq = (1 + dr).cumprod()
    dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)

    rng = np.random.RandomState(42)
    bsr = np.empty(3000)
    for b in range(3000):
        s = rng.choice(dr.values, size=len(dr), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

    return {"sr": sr, "ci": ci, "sortino": sortino, "max_dd": dd,
            "pnl": float(dpnl_adj.sum()), "n_trades": n_trades,
            "n_signals": total_sigs, "daily": dpnl_adj}


def main():
    T0 = time.time()
    print("=" * 80)
    print("  L2 A/B TEST — OHLCV vs OHLCV+L2 (parallel, Bybit VIP0)")
    print("=" * 80)

    # A: Baseline (OHLCV only, 13 features)
    r_base = run_variant("OHLCV only", OHLCV_FEATURES, use_l2=False)

    # B: OHLCV + L2 depth (17 features)
    r_l2 = run_variant("OHLCV + L2", OHLCV_FEATURES + L2_FEATURES, use_l2=True)

    # Results
    print(f"\n{'='*80}")
    print("  A/B RESULTS (simulate_portfolio + Bybit VIP0)")
    print(f"{'='*80}")

    for label, r in [("OHLCV only (13 feat)", r_base), ("OHLCV + L2 depth (17 feat)", r_l2)]:
        if r:
            print(f"\n  {label}:")
            print(f"    SR={r['sr']:.3f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]  "
                  f"Sortino={r['sortino']:.3f}  DD={r['max_dd']:.1f}%  "
                  f"PnL=${r['pnl']:,.0f}  trades={r['n_trades']:,}  signals={r['n_signals']:,}")

    if r_base and r_l2:
        delta = r_l2["sr"] - r_base["sr"]
        print(f"\n  Delta: {delta:+.3f} SR")

        print(f"\n  Year-by-year:")
        for year in sorted(set(r_base["daily"].index.year) | set(r_l2["daily"].index.year)):
            yb = r_base["daily"][r_base["daily"].index.year == year]
            yl = r_l2["daily"][r_l2["daily"].index.year == year]
            if len(yb) > 20 and len(yl) > 20:
                sr_b = float(yb.mean() / yb.std() * np.sqrt(365.25)) if yb.std() > 0 else 0
                sr_l = float(yl.mean() / yl.std() * np.sqrt(365.25)) if yl.std() > 0 else 0
                print(f"    {year}: base={sr_b:.3f}  +L2={sr_l:.3f}  delta={sr_l-sr_b:+.3f}")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
