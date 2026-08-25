#!/usr/bin/env python
"""Multi-timeframe production comparison.

Production config: C=0.1, 2-month lookback, conf>=0.08, inverted vol, 4% sizing.
Tests: 5m+micro, 15m, 1h, 4h, and multi-TF combos.
All through simulate_portfolio + Bybit VIP0.
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

ASSETS = {}
for f in L2_DIR.glob("*_micro_15m.parquet"):
    sym = f.stem.replace("_micro_15m", "")
    ASSETS[sym.replace("USDT", "")] = sym

OHLCV = ["ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
         "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]
MICRO = ["ms_depth_imb_1pct","ms_depth_imb_2pct","ms_trade_imbalance"]

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


def _gen_signals(args):
    name, sym, tf, fwd_bars, feat_list, use_micro, strategy_name = args
    warnings.filterwarnings("ignore")

    # PRODUCTION CONFIG: 2-month lookback
    config = SystemConfig(
        model=ModelConfig(C=0.1, penalty="l1", solver="saga", max_iter=1000, seed=42,
                          fwd_bars=fwd_bars, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=feat_list, winsor_lo=0.01, winsor_hi=0.99,
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

    if use_micro:
        micro_path = L2_DIR / f"{sym}_micro_15m.parquet"
        if micro_path.exists():
            micro = pd.read_parquet(micro_path)
            if micro.index.tz is not None and feat.index.tz is None:
                micro.index = micro.index.tz_localize(None)
            elif micro.index.tz is None and feat.index.tz is not None:
                micro.index = micro.index.tz_localize(feat.index.tz)
            micro_rs = micro[MICRO].resample(tf).mean() if tf != "15min" else micro[MICRO]
            orig = feat.index
            feat = feat.join(micro_rs, how="left")
            feat.index = orig
            for c in MICRO:
                if c in feat.columns:
                    feat[c] = feat[c].fillna(0)

    wf = WalkForwardModel(config, force_monthly_retrain=(tf in ("15min", "1h", "4h")))
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0:
        return None
    if config.regime_filter.enabled:
        sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    if len(sigs) > 0:
        sigs["strategy"] = strategy_name
        sigs["hold_bars"] = fwd_bars
        return sigs
    return None


def generate_strategy(label, tf, fwd_bars, feat_list, use_micro, strategy_name):
    """Generate signals for all assets for one strategy."""
    print(f"  Generating {label}...")
    t0 = time.time()
    tasks = [(n, s, tf, fwd_bars, feat_list, use_micro, strategy_name)
             for n, s in ASSETS.items()]
    all_sigs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_gen_signals, t): t[0] for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            if r is not None and len(r) > 0:
                all_sigs.append(r)
    if all_sigs:
        merged = pd.concat(all_sigs, ignore_index=True)
        print(f"    {len(merged):,} signals in {time.time()-t0:.0f}s")
        return merged
    print(f"    No signals ({time.time()-t0:.0f}s)")
    return pd.DataFrame()


def run_portfolio(signals, label):
    """Full portfolio sim + Bybit costs."""
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
    rng = np.random.RandomState(42)
    bsr = np.empty(3000)
    for b in range(3000):
        s = rng.choice(dr.values, size=len(dr), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

    print(f"  {label:<40} SR={sr:.3f} [{ci[0]:.2f},{ci[1]:.2f}] "
          f"Sortino={sortino:.3f} DD={dd:.1f}% PnL=${dpnl_adj.sum():,.0f} "
          f"trades={n_trades:,}")
    return {"sr": sr, "ci": ci, "sortino": sortino, "dd": dd,
            "pnl": float(dpnl_adj.sum()), "n_trades": n_trades, "daily": dpnl_adj}


def main():
    T0 = time.time()
    print("=" * 90)
    print("  PRODUCTION MULTI-TF COMPARISON (2-month lookback, conf>=0.08, Bybit VIP0)")
    print(f"  {len(ASSETS)} assets")
    print("=" * 90)

    # Generate signals for each strategy
    strats = {}

    strats["5m_micro"] = generate_strategy(
        "5min OHLCV+micro", "5min", 12, OHLCV + MICRO, True, "logreg_5m_micro")
    strats["15m"] = generate_strategy(
        "15min OHLCV (production)", "15min", 4, OHLCV, False, "logreg_15m")
    strats["1h"] = generate_strategy(
        "1h OHLCV", "1h", 4, OHLCV, False, "logreg_1h")
    strats["1h_micro"] = generate_strategy(
        "1h OHLCV+micro", "1h", 4, OHLCV + MICRO, True, "logreg_1h_micro")
    strats["4h"] = generate_strategy(
        "4h OHLCV", "4h", 4, OHLCV, False, "logreg_4h")

    # Individual strategy backtests
    print(f"\n{'='*90}")
    print("  INDIVIDUAL STRATEGIES")
    print(f"{'='*90}")
    results = {}
    for name, sigs in strats.items():
        r = run_portfolio(sigs, name)
        if r:
            results[name] = r

    # Multi-strategy combos
    print(f"\n{'='*90}")
    print("  MULTI-STRATEGY PORTFOLIOS")
    print(f"{'='*90}")

    combos = {
        "15m + 4h (current prod)": ["15m", "4h"],
        "15m + 1h + 4h": ["15m", "1h", "4h"],
        "5m_micro + 4h": ["5m_micro", "4h"],
        "5m_micro + 1h + 4h": ["5m_micro", "1h", "4h"],
        "5m_micro + 15m + 4h": ["5m_micro", "15m", "4h"],
        "1h_micro + 4h": ["1h_micro", "4h"],
        "ALL (5m_micro+15m+1h+4h)": ["5m_micro", "15m", "1h", "4h"],
    }

    for combo_name, strat_keys in combos.items():
        parts = [strats[k] for k in strat_keys if k in strats and len(strats[k]) > 0]
        if not parts:
            continue
        combined = pd.concat(parts, ignore_index=True)
        r = run_portfolio(combined, combo_name)
        if r:
            results[combo_name] = r

    # Summary
    print(f"\n{'='*90}")
    print("  SUMMARY (sorted by SR)")
    print(f"{'='*90}")
    print(f"{'Config':<45} {'SR':>6} {'Sortino':>8} {'DD%':>6} {'PnL':>12} {'Trades':>8}")
    print("-" * 90)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["sr"]):
        print(f"{name:<45} {r['sr']:>6.3f} {r['sortino']:>8.3f} {r['dd']:>5.1f}% "
              f"${r['pnl']:>10,.0f} {r['n_trades']:>8,}")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
