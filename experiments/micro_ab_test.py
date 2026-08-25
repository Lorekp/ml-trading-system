#!/usr/bin/env python
"""Microstructure A/B Test: OHLCV vs OHLCV + top 3 micro features.
Uses simulate_portfolio + Bybit VIP0 costs. Production config.
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

OHLCV_FEATURES = [
    "ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore",
    "rsi_14","stoch_rsi","macd_hist","bb_pct_b","atr_14",
    "dist_swing_high","dist_swing_low",
]
MICRO_TOP3 = ["ms_depth_imb_1pct","ms_depth_imb_2pct","ms_trade_imbalance"]

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


def _process_asset(args):
    name, sym, feature_list, use_micro = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=0.1, penalty="l1", solver="saga", max_iter=1000, seed=42, fwd_bars=4,
                          signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=feature_list, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe="15min", regime_timeframe="4h"),
        assets=ASSETS,
    )
    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
    except FileNotFoundError:
        return None
    ohlcv_tf = resample_ohlcv(ohlcv_1m, "15min")
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    feat = compute_features(ohlcv_tf, config.features)

    if use_micro:
        micro_path = L2_DIR / f"{sym}_micro_15m.parquet"
        if micro_path.exists():
            micro = pd.read_parquet(micro_path)
            # Align timezones: match micro to feat (both should be UTC)
            if feat.index.tz is not None and micro.index.tz is None:
                micro.index = micro.index.tz_localize(feat.index.tz)
            elif feat.index.tz is None and micro.index.tz is not None:
                micro.index = micro.index.tz_localize(None)
            orig_idx = feat.index
            feat = feat.join(micro[MICRO_TOP3], how="left")
            feat.index = orig_idx  # preserve DatetimeIndex type
            for c in MICRO_TOP3:
                if c in feat.columns:
                    feat[c] = feat[c].fillna(0)

    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0:
        return None
    if config.regime_filter.enabled:
        sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    if len(sigs) > 0:
        sigs["strategy"] = "logreg_15m"
        sigs["hold_bars"] = 4
        return sigs.to_dict("list")
    return None


def run_variant(label, feature_list, use_micro):
    print(f"\n  {label} ({len(feature_list)} features)...")
    t0 = time.time()
    tasks = [(n, s, feature_list, use_micro) for n, s in ASSETS.items()]
    all_sigs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_process_asset, t): t[0] for t in tasks}
        done = 0
        for f in as_completed(futs):
            done += 1
            r = f.result()
            if r:
                all_sigs.append(pd.DataFrame(r))
            if done % 10 == 0:
                print(f"    {done}/{len(tasks)} assets...")
    if not all_sigs:
        print("    No signals!")
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

    rng = np.random.RandomState(42)
    bsr = np.empty(3000)
    for b in range(3000):
        s = rng.choice(dr.values, size=len(dr), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

    print(f"    SR={sr:.3f} [{ci[0]:.2f},{ci[1]:.2f}] Sortino={sortino:.3f} DD={dd:.1f}%"
          f" PnL=${dpnl_adj.sum():,.0f} signals={len(merged):,} ({time.time()-t0:.0f}s)")
    return {"sr": sr, "ci": ci, "sortino": sortino, "dd": dd, "pnl": float(dpnl_adj.sum()), "daily": dpnl_adj}


def main():
    print("=" * 70)
    print(f"  MICROSTRUCTURE A/B TEST ({len(ASSETS)} assets, Bybit VIP0)")
    print("=" * 70)

    r_base = run_variant("A: OHLCV only", OHLCV_FEATURES, False)
    r_micro = run_variant("B: OHLCV + top 3 micro", OHLCV_FEATURES + MICRO_TOP3, True)

    if r_base and r_micro:
        delta = r_micro["sr"] - r_base["sr"]
        print(f"\n  Delta: {delta:+.3f} SR")

        # Year-by-year
        print(f"\n  Year-by-year:")
        for year in sorted(set(r_base["daily"].index.year) | set(r_micro["daily"].index.year)):
            yb = r_base["daily"][r_base["daily"].index.year == year]
            ym = r_micro["daily"][r_micro["daily"].index.year == year]
            if len(yb) > 20 and len(ym) > 20:
                sr_b = float(yb.mean() / yb.std() * np.sqrt(365.25)) if yb.std() > 0 else 0
                sr_m = float(ym.mean() / ym.std() * np.sqrt(365.25)) if ym.std() > 0 else 0
                print(f"    {year}: base={sr_b:.3f}  +micro={sr_m:.3f}  delta={sr_m-sr_b:+.3f}")


if __name__ == "__main__":
    main()
