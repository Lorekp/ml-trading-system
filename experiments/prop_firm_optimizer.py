#!/usr/bin/env python
"""Prop firm optimizer: max PnL under 10% DD constraint.

Sweeps sizing per strategy and per-asset vs BTC vol overlay.
Uses the 15m + 1h + 4h combo with per-TF optimized params.
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

# Pre-computed strategy configs (from sweep)
STRAT_CONFIGS = {
    "15m": ("15min", 4, 0.10, 0.08),
    "1h":  ("1h",    4, 0.10, 0.08),
    "4h":  ("4h",    6, 0.20, 0.08),
}


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


def gen_strategy(sname, tf, fwd, C, conf):
    tasks = [(n, s, tf, fwd, C) for n, s in ASSETS.items()]
    all_sigs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed({pool.submit(_gen, t): t for t in tasks}):
            r = f.result()
            if r is not None and len(r) > 0:
                all_sigs.append(r)
    if not all_sigs:
        return pd.DataFrame()
    merged = pd.concat(all_sigs, ignore_index=True)
    merged["strategy"] = sname
    merged["hold_bars"] = fwd
    if "confidence" in merged.columns:
        merged = merged[merged["confidence"] >= conf].reset_index(drop=True)
    return merged


def compute_per_asset_vol(assets_dict, window=20, pct_window=60):
    """Compute per-asset vol percentile (alternative to BTC-only vol overlay)."""
    vol_data = {}
    for name, sym in assets_dict.items():
        try:
            ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
            daily = resample_ohlcv(ohlcv_1m, "1D")["close"]
            if daily.index.tz is not None:
                daily.index = daily.index.tz_localize(None)
            vol = daily.pct_change().rolling(window).std()
            vol_pct = vol.rolling(pct_window).rank(pct=True)
            vol_data[sym] = vol_pct
        except Exception:
            pass
    return vol_data


def apply_per_asset_vol_overlay(signals, vol_data, low_mult=1.25, mid_mult=1.0, high_mult=0.50,
                                 low_pct=0.20, high_pct=0.80):
    """Apply vol overlay using each asset's OWN vol instead of BTC."""
    out = signals.copy()
    if "sizing_mult" not in out.columns:
        out["sizing_mult"] = 1.0

    if "asset" not in out.columns:
        return out

    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)

    for i in range(len(out)):
        asset_sym = out.iloc[i].get("asset", "")
        # Map asset name to symbol
        sym = asset_sym + "USDT" if not asset_sym.endswith("USDT") else asset_sym
        if sym not in vol_data:
            continue
        vol_pct = vol_data[sym]
        t = ts.iloc[i]
        # Find nearest vol percentile
        idx = vol_pct.index.searchsorted(t)
        if idx == 0 or idx >= len(vol_pct):
            continue
        pct_val = vol_pct.iloc[idx - 1]
        if pd.isna(pct_val):
            continue
        if pct_val < low_pct:
            out.iat[i, out.columns.get_loc("sizing_mult")] = low_mult
        elif pct_val > high_pct:
            out.iat[i, out.columns.get_loc("sizing_mult")] = high_mult
        else:
            out.iat[i, out.columns.get_loc("sizing_mult")] = mid_mult

    return out


def run_portfolio(signals, label, sizing, max_dd, vol_mode, btc_daily, vol_data):
    if len(signals) == 0: return None
    merged = signals.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in merged.columns: merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Vol overlay
    if vol_mode == "btc":
        vol_cfg = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
            low_pct=0.20, high_pct=0.80, low_mult=1.25, mid_mult=1.00, high_mult=0.50)
        merged = apply_vol_overlay(merged, btc_daily, vol_cfg)
    elif vol_mode == "per_asset":
        merged = apply_per_asset_vol_overlay(merged, vol_data)
    # else: no vol overlay

    port = PortfolioConfig(
        starting_capital=CAPITAL, max_dd=max_dd, delever_start=max_dd * 0.5,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
        max_per_strategy=0.50, base_sizing_pct=sizing, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    sim = simulate_portfolio(merged, port)
    if sim is None: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None
    if dpnl.index.tz is not None: dpnl = dpnl.copy(); dpnl.index = dpnl.index.tz_localize(None)

    n_trades = sim["n_trades"]
    avg_sz = float(sim["sizes"][~sim["skipped"]].mean()) if (~sim["skipped"]).any() else 0
    total_extra = n_trades * avg_sz * BYBIT_EXTRA / 10000
    dpnl_adj = dpnl - total_extra / max(len(dpnl), 1)

    dr = dpnl_adj / CAPITAL
    if dr.std() == 0 or len(dr) < 30: return None
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    eq = (1 + dr).cumprod()
    dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    worst_daily_dd = float(dr.min() * 100)
    pnl = float(dpnl_adj.sum())

    return {"sr": sr, "dd": dd, "worst_daily": worst_daily_dd, "pnl": pnl,
            "n_trades": n_trades, "daily": dpnl_adj}


def main():
    T0 = time.time()
    print("=" * 90)
    print("  PROP FIRM OPTIMIZER: Max PnL under 10% DD, <5% daily DD")
    print("=" * 90)

    # Generate strategies
    print("\nGenerating strategies...")
    strat_sigs = {}
    for sname, (tf, fwd, C, conf) in STRAT_CONFIGS.items():
        print(f"  {sname}...", end=" ", flush=True)
        strat_sigs[sname] = gen_strategy(sname, tf, fwd, C, conf)
        print(f"{len(strat_sigs[sname]):,} signals")

    # Load BTC daily (once)
    btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
    btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
    if btc_daily.index.tz is not None:
        btc_daily.index = btc_daily.index.tz_localize(None)

    # Compute per-asset vol
    print("\nComputing per-asset vol percentiles...")
    vol_data = compute_per_asset_vol(ASSETS)
    print(f"  {len(vol_data)} assets with vol data")

    # Strategy combinations
    combos = {
        "15m only":     ["15m"],
        "1h only":      ["1h"],
        "4h only":      ["4h"],
        "15m+4h":       ["15m", "4h"],
        "1h+4h":        ["1h", "4h"],
        "15m+1h":       ["15m", "1h"],
        "15m+1h+4h":    ["15m", "1h", "4h"],
    }

    # Sweep: sizing × vol_mode × combo
    sizings = [0.02, 0.03, 0.04, 0.05, 0.06]
    vol_modes = ["btc", "per_asset", "none"]

    print(f"\nSweeping {len(combos)} combos × {len(sizings)} sizings × {len(vol_modes)} vol modes = "
          f"{len(combos)*len(sizings)*len(vol_modes)} configs")

    results = []
    for combo_name, strat_keys in combos.items():
        parts = [strat_sigs[k] for k in strat_keys if len(strat_sigs[k]) > 0]
        if not parts: continue
        combined = pd.concat(parts, ignore_index=True)

        for sizing in sizings:
            for vm in vol_modes:
                r = run_portfolio(combined, f"{combo_name} sz={sizing} vol={vm}",
                                  sizing, 0.10, vm, btc_daily, vol_data)
                if r:
                    results.append({
                        "combo": combo_name, "sizing": sizing, "vol_mode": vm,
                        **{k: v for k, v in r.items() if k != "daily"},
                    })

    df = pd.DataFrame(results)

    # Filter: DD < 10%
    viable = df[df["dd"] < 10.0].copy()
    print(f"\n{'='*90}")
    print(f"  RESULTS: {len(viable)} configs with DD < 10% (of {len(df)} total)")
    print(f"{'='*90}")

    # Top 20 by PnL
    print(f"\n  TOP 20 BY PnL (DD < 10%):")
    print(f"  {'Combo':<18} {'sz':>4} {'vol':>10} {'SR':>6} {'DD%':>5} {'dDD%':>5} {'PnL':>11} {'Trades':>7}")
    print("  " + "-" * 70)
    for _, r in viable.nlargest(20, "pnl").iterrows():
        print(f"  {r['combo']:<18} {r['sizing']:>4.0%} {r['vol_mode']:>10} {r['sr']:>6.3f} "
              f"{r['dd']:>4.1f}% {r['worst_daily']:>5.2f}% ${r['pnl']:>9,.0f} {r['n_trades']:>7,}")

    # Best per combo
    print(f"\n  BEST PnL PER COMBO (DD < 10%):")
    for combo in combos.keys():
        c_df = viable[viable["combo"] == combo]
        if len(c_df) > 0:
            best = c_df.loc[c_df["pnl"].idxmax()]
            print(f"  {combo:<18}: sz={best['sizing']:.0%} vol={best['vol_mode']:<10} "
                  f"SR={best['sr']:.3f} DD={best['dd']:.1f}% PnL=${best['pnl']:,.0f}")

    # BTC vol vs per-asset vol comparison
    print(f"\n  VOL OVERLAY COMPARISON (15m+1h+4h, sz=4%):")
    for vm in vol_modes:
        row = df[(df["combo"] == "15m+1h+4h") & (df["sizing"] == 0.04) & (df["vol_mode"] == vm)]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"  {vm:<12}: SR={r['sr']:.3f} DD={r['dd']:.1f}% PnL=${r['pnl']:,.0f}")

    df.to_csv("experiments/prop_firm_results.csv", index=False)
    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
