#!/usr/bin/env python
"""L2 Integration — Phase 3: Compute, cache, and backtest with L2 features.

1. Compute bookDepth L2 features for all 27 symbols, cache on homeserver
2. Run walk-forward LogReg with 13 OHLCV + 4 L2 depth features = 17 features
3. A/B compare: OHLCV-only vs OHLCV+L2 using simulate_portfolio + Bybit costs
4. Apply optimized config (conf >= 0.08, inverted vol, 4% sizing)
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
from signal_engine.l2_data import load_book_depth, list_available_dates
from signal_engine.l2_features import (
    aggregate_book_depth_to_15m, merge_l2_with_ohlcv,
    ALL_BOOK_DEPTH_FEATURES,
)

CAPITAL = 100_000.0
BYBIT_EXTRA = 3.5
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
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
L2_DEPTH_FEATURES = ALL_BOOK_DEPTH_FEATURES  # 4 depth features

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


def compute_l2_depth_for_symbol(symbol):
    """Compute all bookDepth L2 features for a symbol, return 15m DataFrame."""
    dates = list_available_dates(symbol, "bookDepth")
    if not dates:
        return pd.DataFrame()

    all_bars = []
    for date in dates:
        try:
            raw = load_book_depth(symbol, date)
            bars = aggregate_book_depth_to_15m(raw)
            all_bars.append(bars)
        except Exception:
            pass

    if not all_bars:
        return pd.DataFrame()
    combined = pd.concat(all_bars).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def generate_signals_with_l2(asset_name, symbol, config, l2_features):
    """Generate signals for one asset using OHLCV + L2 features."""
    try:
        ohlcv_1m = load_binance_1m(symbol, DATA_DIRS)
    except FileNotFoundError:
        return None

    ohlcv_tf = resample_ohlcv(ohlcv_1m, "15min")
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    ohlcv_feat = compute_features(ohlcv_tf, config.features)

    # Merge L2 features
    if not l2_features.empty:
        merged_feat = merge_l2_with_ohlcv(ohlcv_feat, l2_features)
    else:
        merged_feat = ohlcv_feat
        # Add zero-filled L2 columns so model has consistent feature count
        for col in L2_DEPTH_FEATURES:
            merged_feat[col] = 0.0

    wf = WalkForwardModel(config, force_monthly_retrain=True)
    signals = wf.generate_signals_batch(ohlcv_tf, merged_feat, asset_name)

    if len(signals) == 0:
        return None

    if config.regime_filter.enabled:
        filtered = apply_regime_filter(signals, ohlcv_4h, config.regime_filter)
    else:
        filtered = signals

    return filtered if len(filtered) > 0 else None


def run_backtest(signals_dict, label):
    """Run full backtest: merge signals, apply filters, simulate_portfolio + Bybit."""
    all_sigs = []
    for name, sigs in signals_dict.items():
        if sigs is not None and len(sigs) > 0:
            sigs = sigs.copy()
            sigs["strategy"] = "logreg_15m"
            sigs["hold_bars"] = 4
            all_sigs.append(sigs)

    # Add 4h and cs_momentum from cache
    for strat in ["logreg_4h", "cs_momentum"]:
        p = Path(f"results/signals_cache/{strat}.parquet")
        if p.exists():
            o = pd.read_parquet(p)
            for c in ["regime_exposure_mult", "sizing_mult", "hold_bars"]:
                if c not in o.columns:
                    o[c] = 4 if c == "hold_bars" else 1.0
            o["probability"] = o["probability"].astype(float)
            all_sigs.append(o)

    if not all_sigs:
        return None

    merged = pd.concat(all_sigs, ignore_index=True)
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

    # Bootstrap CI
    rng = np.random.RandomState(42)
    bsr = np.empty(3000)
    for b in range(3000):
        s = rng.choice(dr.values, size=len(dr), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

    return {
        "sr": sr, "ci": ci, "sortino": sortino, "max_dd": dd,
        "pnl": float(dpnl_adj.sum()), "n_trades": n_trades,
        "n_signals": len(merged), "daily": dpnl_adj,
    }


def main():
    T0 = time.time()
    print("=" * 80)
    print("  L2 INTEGRATION — Phase 3: OHLCV vs OHLCV+L2 A/B Test")
    print("=" * 80)

    # ================================================================
    # Step 1: Compute L2 depth features for all symbols
    # ================================================================
    print("\n[Step 1] Computing bookDepth L2 features for all symbols...")
    l2_cache = {}
    for name, symbol in sorted(ASSETS.items()):
        t0 = time.time()
        l2 = compute_l2_depth_for_symbol(symbol)
        if not l2.empty:
            l2_cache[symbol] = l2
            print(f"  {name}: {len(l2)} bars ({l2.index[0].date()} to {l2.index[-1].date()}) in {time.time()-t0:.0f}s")
        else:
            print(f"  {name}: no bookDepth data")
    print(f"  L2 data for {len(l2_cache)}/{len(ASSETS)} symbols")

    # ================================================================
    # Step 2: Generate signals — OHLCV only (baseline)
    # ================================================================
    print(f"\n[Step 2a] Generating signals: OHLCV only (13 features)...")
    ohlcv_config = SystemConfig(
        model=ModelConfig(C=0.1, penalty="l1", solver="saga", max_iter=1000, seed=42, fwd_bars=4,
                          signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=OHLCV_FEATURES, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        portfolio=PortfolioConfig(starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
                                  slippage=0.0001, max_exposure=0.80, max_per_asset=0.06),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe="15min", regime_timeframe="4h"),
        assets=ASSETS,
    )

    baseline_signals = {}
    for name, symbol in sorted(ASSETS.items()):
        empty_l2 = pd.DataFrame()  # no L2 features
        sigs = generate_signals_with_l2(name, symbol, ohlcv_config, empty_l2)
        if sigs is not None:
            baseline_signals[name] = sigs
        if len(baseline_signals) % 5 == 0:
            print(f"  {len(baseline_signals)} assets done...")

    total_baseline = sum(len(s) for s in baseline_signals.values())
    print(f"  Baseline: {total_baseline:,} signals from {len(baseline_signals)} assets")

    # ================================================================
    # Step 3: Generate signals — OHLCV + L2 (17 features)
    # ================================================================
    print(f"\n[Step 2b] Generating signals: OHLCV + L2 depth (17 features)...")
    l2_config = SystemConfig(
        model=ModelConfig(C=0.1, penalty="l1", solver="saga", max_iter=1000, seed=42, fwd_bars=4,
                          signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=OHLCV_FEATURES + L2_DEPTH_FEATURES,
                               winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        portfolio=PortfolioConfig(starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
                                  slippage=0.0001, max_exposure=0.80, max_per_asset=0.06),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe="15min", regime_timeframe="4h"),
        assets=ASSETS,
    )

    l2_signals = {}
    for name, symbol in sorted(ASSETS.items()):
        l2_feat = l2_cache.get(symbol, pd.DataFrame())
        sigs = generate_signals_with_l2(name, symbol, l2_config, l2_feat)
        if sigs is not None:
            l2_signals[name] = sigs
        if len(l2_signals) % 5 == 0:
            print(f"  {len(l2_signals)} assets done...")

    total_l2 = sum(len(s) for s in l2_signals.values())
    print(f"  OHLCV+L2: {total_l2:,} signals from {len(l2_signals)} assets")

    # ================================================================
    # Step 4: A/B Backtest
    # ================================================================
    print(f"\n[Step 3] A/B Backtest (simulate_portfolio + Bybit)...")

    r_base = run_backtest(baseline_signals, "OHLCV only")
    r_l2 = run_backtest(l2_signals, "OHLCV + L2")

    print(f"\n{'='*80}")
    print("  A/B COMPARISON (simulate_portfolio + Bybit VIP0)")
    print(f"{'='*80}")

    for label, r in [("OHLCV only (baseline)", r_base), ("OHLCV + L2 depth", r_l2)]:
        if r:
            print(f"  {label}:")
            print(f"    SR={r['sr']:.3f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]  "
                  f"Sortino={r['sortino']:.3f}  DD={r['max_dd']:.1f}%  "
                  f"PnL=${r['pnl']:,.0f}  trades={r['n_trades']:,}")

    if r_base and r_l2:
        delta = r_l2["sr"] - r_base["sr"]
        print(f"\n  Delta: {delta:+.3f} SR")

        # Year-by-year
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
