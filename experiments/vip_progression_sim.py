#!/usr/bin/env python
"""Simulate Bybit VIP tier progression as volume/equity grows.

Uses the best production config signals, applies dynamic fees based on
rolling 30-day volume and current equity. Tests multiple starting capitals.
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
from signal_engine.config import *

DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
L2_DIR = Path("data/l2-features")
ASSETS = {
    "BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT",
    "BNB":"BNBUSDT","ADA":"ADAUSDT","AVAX":"AVAXUSDT","DOT":"DOTUSDT",
    "LINK":"LINKUSDT","NEAR":"NEARUSDT","UNI":"UNIUSDT","AAVE":"AAVEUSDT",
    "LTC":"LTCUSDT","BCH":"BCHUSDT","HBAR":"HBARUSDT","TRX":"TRXUSDT",
    "SUI":"SUIUSDT","APT":"APTUSDT","ENJ":"ENJUSDT","SXP":"SXPUSDT",
    "ZEC":"ZECUSDT","XLM":"XLMUSDT","FET":"FETUSDT","WLD":"WLDUSDT",
    "ANKR":"ANKRUSDT","TAO":"TAOUSDT","ENA":"ENAUSDT",
    "FTM":"FTMUSDT","FIL":"FILUSDT","BNX":"BNXUSDT",
}
OHLCV = ["ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
         "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]

# Bybit USDT Perpetual VIP Tiers (confirmed)
VIP_TIERS = [
    # (name, min_assets, min_30d_vol, maker_bps, taker_bps)
    ("VIP0",    0,           0,             2.00, 5.50),
    ("VIP1",    100_000,     10_000_000,    1.80, 4.00),
    ("VIP2",    250_000,     25_000_000,    1.60, 3.75),
    ("VIP3",    500_000,     50_000_000,    1.40, 3.50),
    ("VIP4",    1_000_000,   100_000_000,   1.20, 3.20),
    ("VIP5",    2_000_000,   250_000_000,   1.00, 3.20),
    ("Supreme", 5_000_000,   500_000_000,   0.00, 3.00),
]
SPREAD_BPS = 0.5  # estimated half-spread


def get_vip_tier(equity, volume_30d):
    """Return (tier_name, maker_bps, taker_bps) based on equity OR volume."""
    result = VIP_TIERS[0]
    for tier in VIP_TIERS:
        name, min_assets, min_vol, maker, taker = tier
        if equity >= min_assets or volume_30d >= min_vol:
            result = tier
    return result[0], result[3], result[4]


def get_rt_cost_bps(maker_bps):
    """Roundtrip cost using maker orders: 2 * maker + spread."""
    return 2 * maker_bps + SPREAD_BPS


def _gen(args):
    name, sym, tf, fwd, C = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=C, penalty="l1", solver="saga", max_iter=1000, seed=42,
                          fwd_bars=fwd, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=OHLCV, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe=tf, regime_timeframe="4h"),
        assets=ASSETS)
    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
    except FileNotFoundError:
        return None
    ohlcv_tf = resample_ohlcv(ohlcv_1m, tf)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    feat = compute_features(ohlcv_tf, config.features)
    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0: return None
    sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    return sigs if len(sigs) > 0 else None


def simulate_with_vip(signals, starting_capital):
    """Simulate portfolio with dynamic VIP tier progression.

    Fast: precompute daily trade volumes, then rolling 30d sum.
    """
    sigs = signals.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(sigs["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)

    prob = sigs["probability"].values.astype(float)
    direction = sigs["direction"].values.astype(float)
    fwd_return = sigs["fwd_return"].values.astype(float)
    sizing_mult = sigs["sizing_mult"].values.astype(float) if "sizing_mult" in sigs.columns else np.ones(len(sigs))
    edge = np.where(np.isfinite(prob), np.abs(2 * prob - 1), 0.1)
    edge = np.clip(edge, 0, 1)

    base_sizing = 0.045
    dates = ts.dt.date.values

    # Phase 1: compute daily PnL at VIP0 to get equity curve, then daily volumes
    equity = starting_capital
    daily_pnl_flat = {}
    daily_vol = {}
    for i in range(len(sigs)):
        d = dates[i]
        sz = min(equity * base_sizing * edge[i] * sizing_mult[i],
                 equity * base_sizing * 2.0)
        gross = direction[i] * fwd_return[i]
        # Use VIP0 for equity tracking (conservative)
        pnl = sz * (gross - 4.5 / 10000)
        equity += pnl
        daily_pnl_flat[d] = daily_pnl_flat.get(d, 0) + pnl
        daily_vol[d] = daily_vol.get(d, 0) + abs(sz) * 2  # roundtrip

    # Phase 2: compute rolling 30d volume and equity per day
    vol_series = pd.Series(daily_vol).sort_index()
    vol_30d = vol_series.rolling(30, min_periods=1).sum()
    eq_series = starting_capital + pd.Series(daily_pnl_flat).sort_index().cumsum()

    # Phase 3: determine tier per day
    tier_per_day = {}
    tier_history = []
    for d in vol_30d.index:
        eq = eq_series.get(d, starting_capital)
        v30 = vol_30d.get(d, 0)
        tier_name, maker_bps, taker_bps = get_vip_tier(eq, v30)
        rt = get_rt_cost_bps(maker_bps)
        tier_per_day[d] = rt
        tier_history.append((d, tier_name, rt, eq, v30))

    # Phase 4: recompute PnL with actual tier fees
    equity = starting_capital
    daily_pnl = {}
    for i in range(len(sigs)):
        d = dates[i]
        rt_bps = tier_per_day.get(d, 4.5)
        sz = min(equity * base_sizing * edge[i] * sizing_mult[i],
                 equity * base_sizing * 2.0)
        gross = direction[i] * fwd_return[i]
        pnl = sz * (gross - rt_bps / 10000)
        equity += pnl
        daily_pnl[d] = daily_pnl.get(d, 0) + pnl

    dpnl = pd.Series(daily_pnl).sort_index()
    dpnl.index = pd.to_datetime(dpnl.index)
    dpnl = dpnl[dpnl != 0]

    return dpnl, tier_history, equity


def main():
    T0 = time.time()
    print("=" * 80)
    print("  BYBIT VIP TIER PROGRESSION SIMULATION")
    print("=" * 80)

    # Generate signals (same as best config)
    print("\nGenerating 15m+1h+4h+cs signals...")
    strat_sigs = {}
    strat_configs = {"15m": ("15min", 4, 0.10, 0.08), "1h": ("1h", 4, 0.10, 0.08), "4h": ("4h", 6, 0.20, 0.08)}
    for sname, (tf, fwd, C, conf) in strat_configs.items():
        tasks = [(n, s, tf, fwd, C) for n, s in ASSETS.items()]
        all_sigs = []
        with ProcessPoolExecutor(max_workers=8) as pool:
            for f in as_completed({pool.submit(_gen, t): t for t in tasks}):
                r = f.result()
                if r is not None and len(r) > 0: all_sigs.append(r)
        if all_sigs:
            m = pd.concat(all_sigs, ignore_index=True)
            m["strategy"] = sname; m["hold_bars"] = fwd
            if "confidence" in m.columns:
                m = m[m["confidence"] >= conf].reset_index(drop=True)
            strat_sigs[sname] = m
            print(f"  {sname}: {len(m):,}")

    scs = pd.read_parquet("results/signals_cache/cs_momentum.parquet")
    combined = pd.concat(list(strat_sigs.values()) + [scs], ignore_index=True)
    combined["probability"] = combined["probability"].astype(float)
    if "sizing_mult" not in combined.columns:
        combined["sizing_mult"] = 1.0
    combined["sizing_mult"] = combined["sizing_mult"].fillna(1.0)
    print(f"  Total: {len(combined):,}")

    # Run at different starting capitals
    capitals = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000]

    print(f"\n{'='*80}")
    print(f"  RESULTS BY STARTING CAPITAL")
    print(f"{'='*80}")
    print(f"\n  {'Capital':>12} {'Final':>12} {'PnL':>12} {'SR':>7} {'Max Tier':>10} "
          f"{'Fee Saved':>10} {'SR vs VIP0':>10}")
    print("  " + "-" * 75)

    # First compute VIP0 baseline for comparison
    for capital in capitals:
        dpnl, tier_hist, final_eq = simulate_with_vip(combined, capital)

        if len(dpnl) < 30:
            print(f"  ${capital:>11,}: insufficient data")
            continue

        dr = dpnl / capital
        sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
        pnl = float(dpnl.sum())

        # Get max tier reached
        max_tier = "VIP0"
        min_rt = 4.5
        for date, tier, rt, eq, vol in tier_hist:
            if tier > max_tier or (tier == max_tier and rt < min_rt):
                max_tier = tier
                min_rt = rt

        # Fee savings vs flat VIP0
        flat_vip0_fees = len(combined) * capital * 0.045 * 0.001 * 4.5 / 10000  # rough
        # Just report the tier progression
        tier_changes = []
        prev_tier = "VIP0"
        for date, tier, rt, eq, vol in tier_hist:
            if tier != prev_tier:
                tier_changes.append(f"{date}:{tier}")
                prev_tier = tier

        print(f"  ${capital:>11,} ${final_eq:>11,.0f} ${pnl:>11,.0f} {sr:>7.3f} {max_tier:>10} "
              f"{min_rt:>9.1f}bp")

        # Print tier progression for larger capitals
        if capital >= 100_000 and tier_changes:
            print(f"    Tier progression: {' -> '.join(tier_changes[:5])}")

    # Detailed report for $100K and $1M
    for cap_label, capital in [("$100K", 100_000), ("$1M", 1_000_000)]:
        print(f"\n  --- Detailed: {cap_label} ---")
        dpnl, tier_hist, final_eq = simulate_with_vip(combined, capital)
        dr = dpnl / capital
        sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
        eq = (1 + dr).cumprod()
        dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)

        print(f"  SR={sr:.3f}  DD={dd:.1f}%  PnL=${dpnl.sum():,.0f}  Final=${final_eq:,.0f}")

        # Year-by-year
        for year in sorted(dpnl.index.year.unique()):
            y = dpnl[dpnl.index.year == year]
            if len(y) > 20:
                sr_y = float(y.mean() / y.std() * np.sqrt(365.25)) if y.std() > 0 else 0
                # Find tier at end of year
                year_tiers = [(d, t, rt) for d, t, rt, e, v in tier_hist if d.year == year]
                end_tier = year_tiers[-1][1] if year_tiers else "VIP0"
                end_rt = year_tiers[-1][2] if year_tiers else 4.5
                print(f"    {year}: SR={sr_y:.3f} PnL=${y.sum():,.0f} tier={end_tier} RT={end_rt:.1f}bp")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
