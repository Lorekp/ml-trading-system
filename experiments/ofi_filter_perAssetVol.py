#!/usr/bin/env python
"""OFI as trade filter/scaler + per-asset vol sizing on 15m+1h+4h.

Tests:
1. Per-asset vol sizing (baseline for this experiment)
2. OFI confirmation filter: only trade when OFI agrees with signal direction
3. OFI scaling: scale position size by OFI strength
4. OFI + per-asset vol combined
5. All above with different leverage multipliers to size back up
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


def load_ofi_for_asset(sym):
    """Load trade imbalance (OFI proxy) for an asset at 15m resolution."""
    p = L2_DIR / f"{sym}_micro_15m.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["ms_trade_imbalance", "ms_depth_imb_2pct"]].copy()


def compute_per_asset_vol():
    """Compute per-asset vol scaling factors."""
    vol_series = {}
    for name, sym in ASSETS.items():
        try:
            ohlcv = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
            if ohlcv.index.tz is not None: ohlcv.index = ohlcv.index.tz_localize(None)
            vol = ohlcv.pct_change().rolling(20).std()
            vol_series[name] = vol
        except Exception:
            pass
    return vol_series


def apply_per_asset_vol(signals, vol_series):
    """Scale sizing_mult by inverse of per-asset vol."""
    out = signals.copy()
    if "sizing_mult" not in out.columns:
        out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)

    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        if asset not in vol_series:
            continue
        vs = vol_series[asset]
        t = ts.iloc[i]
        idx = vs.index.searchsorted(t)
        if 0 < idx < len(vs):
            v = vs.iloc[idx - 1]
            if pd.notna(v) and v > 0:
                mult = np.clip(0.025 / v, 0.3, 1.5)
                out.iat[i, out.columns.get_loc("sizing_mult")] *= mult
    return out


def apply_ofi_filter(signals, ofi_data, mode="filter", threshold=0.0):
    """Apply OFI as filter or scaler.

    mode='filter': drop signals where OFI disagrees with direction
    mode='scale': scale sizing_mult by OFI alignment strength
    """
    out = signals.copy()
    if "sizing_mult" not in out.columns:
        out["sizing_mult"] = 1.0

    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)

    keep = np.ones(len(out), dtype=bool)

    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        sym = asset + "USDT" if not asset.endswith("USDT") else asset
        if sym not in ofi_data or ofi_data[sym] is None:
            continue

        ofi_df = ofi_data[sym]
        t = ts.iloc[i]
        idx = ofi_df.index.searchsorted(t)
        if idx == 0 or idx > len(ofi_df):
            continue

        # Use PREVIOUS bar's OFI (no look-ahead)
        ofi_val = ofi_df.iloc[idx - 1]["ms_trade_imbalance"]
        depth_val = ofi_df.iloc[idx - 1]["ms_depth_imb_2pct"]
        if pd.isna(ofi_val):
            continue

        direction = out.iloc[i]["direction"]
        # OFI agreement: positive OFI + long = agree, negative OFI + short = agree
        ofi_agrees = (ofi_val * direction) > threshold
        depth_agrees = (depth_val * direction) > 0

        if mode == "filter":
            # Only keep if OFI agrees with direction
            if not ofi_agrees:
                keep[i] = False
        elif mode == "filter_both":
            # Only keep if BOTH OFI and depth agree
            if not (ofi_agrees and depth_agrees):
                keep[i] = False
        elif mode == "scale":
            # Scale sizing by OFI agreement strength
            agreement = ofi_val * direction  # positive = agrees
            # Map to multiplier: strong agree -> 1.5x, disagree -> 0.5x
            mult = np.clip(0.5 + agreement * 10, 0.3, 1.5)
            out.iat[i, out.columns.get_loc("sizing_mult")] *= mult
        elif mode == "scale_depth":
            # Scale by depth imbalance agreement
            agreement = depth_val * direction
            mult = np.clip(0.5 + agreement * 5, 0.3, 1.5)
            out.iat[i, out.columns.get_loc("sizing_mult")] *= mult

    if mode in ("filter", "filter_both"):
        out = out[keep].reset_index(drop=True)
    return out


def run_portfolio(signals, label, sizing, leverage=1.0):
    if len(signals) == 0:
        print(f"  {label:<50} NO SIGNALS")
        return None
    merged = signals.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in merged.columns: merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    # Apply leverage multiplier to sizing_mult
    merged["sizing_mult"] = merged["sizing_mult"] * leverage
    merged["probability"] = merged["probability"].astype(float)
    if "confidence" in merged.columns:
        merged = merged[merged["confidence"] >= 0.08].reset_index(drop=True)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    port = PortfolioConfig(
        starting_capital=CAPITAL, max_dd=0.10, delever_start=0.05,
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
    worst_day = float(dr.min() * 100)
    cvar5 = float(-dr[dr <= dr.quantile(0.05)].mean() * 10000) if len(dr) > 20 else 0
    pnl = float(dpnl_adj.sum())

    print(f"  {label:<50} SR={sr:.3f} DD={dd:.1f}% dDD={worst_day:.2f}% "
          f"CVaR5={cvar5:.0f}bp PnL=${pnl:,.0f} trades={n_trades:,}")
    return {"sr": sr, "dd": dd, "worst_day": worst_day, "cvar5": cvar5, "pnl": pnl, "n_trades": n_trades}


def main():
    T0 = time.time()
    print("=" * 100)
    print("  OFI FILTER + PER-ASSET VOL on 15m+1h+4h")
    print("=" * 100)

    # Generate strategies
    print("\nGenerating 15m+1h+4h signals...")
    strat_sigs = {}
    for sname, (tf, fwd, C, conf) in STRAT_CONFIGS.items():
        tasks = [(n, s, tf, fwd, C) for n, s in ASSETS.items()]
        all_sigs = []
        with ProcessPoolExecutor(max_workers=8) as pool:
            for f in as_completed({pool.submit(_gen, t): t for t in tasks}):
                r = f.result()
                if r is not None and len(r) > 0: all_sigs.append(r)
        if all_sigs:
            m = pd.concat(all_sigs, ignore_index=True)
            m["strategy"] = sname
            m["hold_bars"] = fwd
            strat_sigs[sname] = m
            print(f"  {sname}: {len(m):,} signals")
    combined_raw = pd.concat(strat_sigs.values(), ignore_index=True)

    # Load OFI data
    print("\nLoading OFI data...")
    ofi_data = {}
    for name, sym in ASSETS.items():
        ofi = load_ofi_for_asset(sym)
        if ofi is not None:
            ofi_data[sym] = ofi
    print(f"  {len(ofi_data)} assets with OFI")

    # Compute per-asset vol
    print("Computing per-asset vol...")
    vol_series = compute_per_asset_vol()
    print(f"  {len(vol_series)} assets with vol data")

    results = []

    # ================================================================
    print(f"\n{'='*100}")
    print("  EXPERIMENT MATRIX")
    print(f"{'='*100}")

    configs = [
        # (label, use_per_asset_vol, ofi_mode, ofi_threshold, leverage, sizing)
        ("A: Baseline (no vol/ofi adjust)", False, None, 0, 1.0, 0.04),
        ("B: Per-asset vol only", True, None, 0, 1.0, 0.04),
        ("B2: Per-asset vol, lev=2x", True, None, 0, 2.0, 0.04),
        ("B3: Per-asset vol, lev=3x", True, None, 0, 3.0, 0.04),
        ("B4: Per-asset vol, sz=6%", True, None, 0, 1.0, 0.06),
        ("B5: Per-asset vol, sz=8%", True, None, 0, 1.0, 0.08),
        ("C: OFI filter (agree only)", False, "filter", 0, 1.0, 0.04),
        ("D: OFI+depth filter (both agree)", False, "filter_both", 0, 1.0, 0.04),
        ("E: OFI scale (size by agreement)", False, "scale", 0, 1.0, 0.04),
        ("F: Depth scale", False, "scale_depth", 0, 1.0, 0.04),
        ("G: Per-asset vol + OFI filter", True, "filter", 0, 1.0, 0.04),
        ("G2: Per-asset vol + OFI filter, lev=2x", True, "filter", 0, 2.0, 0.04),
        ("H: Per-asset vol + OFI scale", True, "scale", 0, 1.0, 0.04),
        ("H2: Per-asset vol + OFI scale, lev=2x", True, "scale", 0, 2.0, 0.04),
        ("I: Per-asset vol + depth scale", True, "scale_depth", 0, 1.0, 0.04),
        ("I2: Per-asset vol + depth scale, lev=2x", True, "scale_depth", 0, 2.0, 0.04),
        ("J: Per-asset vol + depth scale, sz=6%", True, "scale_depth", 0, 1.0, 0.06),
        ("K: Per-asset vol + OFI+depth filter, lev=2x", True, "filter_both", 0, 2.0, 0.04),
    ]

    for label, use_vol, ofi_mode, ofi_thresh, lev, sz in configs:
        sigs = combined_raw.copy()

        if use_vol:
            sigs = apply_per_asset_vol(sigs, vol_series)

        if ofi_mode:
            sigs = apply_ofi_filter(sigs, ofi_data, mode=ofi_mode, threshold=ofi_thresh)

        r = run_portfolio(sigs, label, sz, leverage=lev)
        if r:
            results.append({"label": label, **r})

    # Summary
    df = pd.DataFrame(results)
    viable = df[(df["dd"] < 10) & (df["worst_day"] > -5)]
    print(f"\n{'='*100}")
    print(f"  VIABLE (DD<10%, daily<5%), sorted by PnL")
    print(f"{'='*100}")
    for _, r in viable.sort_values("pnl", ascending=False).iterrows():
        print(f"  {r['label']:<50} PnL=${r['pnl']:>9,.0f} SR={r['sr']:.3f} "
              f"DD={r['dd']:.1f}% dDD={r['worst_day']:.2f}% CVaR5={r['cvar5']:.0f}bp")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
