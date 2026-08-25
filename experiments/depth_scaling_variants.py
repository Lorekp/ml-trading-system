#!/usr/bin/env python
"""Test different depth scaling functions on config I (per-asset vol + depth scale).

Current: linear clip: mult = clip(0.5 + agreement * 5, 0.3, 1.5)
Test: linear, quadratic, exponential, log, sigmoid, step
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
from signal_engine.config import *

CAPITAL = 100_000.0
BYBIT_EXTRA = 3.5
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


def gen_strat(sname, tf, fwd, C, conf):
    tasks = [(n, s, tf, fwd, C) for n, s in ASSETS.items()]
    all_sigs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed({pool.submit(_gen, t): t for t in tasks}):
            r = f.result()
            if r is not None and len(r) > 0: all_sigs.append(r)
    if not all_sigs: return pd.DataFrame()
    m = pd.concat(all_sigs, ignore_index=True)
    m["strategy"] = sname; m["hold_bars"] = fwd
    if "confidence" in m.columns:
        m = m[m["confidence"] >= conf].reset_index(drop=True)
    return m


def compute_per_asset_vol():
    vol_series = {}
    for name, sym in ASSETS.items():
        try:
            ohlcv = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
            if ohlcv.index.tz is not None: ohlcv.index = ohlcv.index.tz_localize(None)
            vol_series[name] = ohlcv.pct_change().rolling(20).std()
        except: pass
    return vol_series


def apply_per_asset_vol(signals, vol_series):
    out = signals.copy()
    if "sizing_mult" not in out.columns: out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        if asset not in vol_series: continue
        vs = vol_series[asset]
        t = ts.iloc[i]
        idx = vs.index.searchsorted(t)
        if 0 < idx < len(vs):
            v = vs.iloc[idx - 1]
            if pd.notna(v) and v > 0:
                out.iat[i, out.columns.get_loc("sizing_mult")] *= np.clip(0.025 / v, 0.3, 1.5)
    return out


def apply_depth_scale(signals, ofi_data, scale_fn):
    """Apply depth scaling with a custom function.

    scale_fn(agreement) -> multiplier, where agreement = depth_imb * direction
    """
    out = signals.copy()
    if "sizing_mult" not in out.columns: out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        sym = asset + "USDT" if not asset.endswith("USDT") else asset
        if sym not in ofi_data or ofi_data[sym] is None: continue
        ofi_df = ofi_data[sym]
        t = ts.iloc[i]
        idx = ofi_df.index.searchsorted(t)
        if idx == 0 or idx > len(ofi_df): continue
        depth_val = ofi_df.iloc[idx - 1]["ms_depth_imb_2pct"]
        if pd.isna(depth_val): continue
        direction = out.iloc[i]["direction"]
        agreement = depth_val * direction  # positive = depth agrees with signal
        mult = scale_fn(agreement)
        out.iat[i, out.columns.get_loc("sizing_mult")] *= mult
    return out


# Scaling functions: agreement in [-1, 1] -> multiplier in [0.3, 1.5]
def linear_scale(a):
    return np.clip(0.5 + a * 5, 0.3, 1.5)

def quadratic_scale(a):
    # Amplify strong agreement, dampen weak
    sign = np.sign(a)
    return np.clip(0.5 + sign * (a ** 2) * 50, 0.3, 1.5)

def exponential_scale(a):
    # Exponential response to agreement
    return np.clip(np.exp(a * 3) * 0.5, 0.3, 1.5)

def log_scale(a):
    # Logarithmic — big response to small agreement, saturates
    sign = np.sign(a)
    return np.clip(1.0 + sign * np.log1p(abs(a) * 10) * 0.3, 0.3, 1.5)

def sigmoid_scale(a):
    # Smooth S-curve centered at agreement=0
    return np.clip(0.3 + 1.2 / (1 + np.exp(-a * 15)), 0.3, 1.5)

def step_scale(a):
    # Binary: agree -> 1.3x, disagree -> 0.5x, neutral -> 1.0x
    if a > 0.02: return 1.3
    elif a < -0.02: return 0.5
    else: return 1.0

def aggressive_linear(a):
    return np.clip(0.3 + a * 8, 0.2, 2.0)

def conservative_linear(a):
    return np.clip(0.7 + a * 3, 0.5, 1.3)


def run(signals, label, sizing=0.04):
    if len(signals) == 0: return None
    merged = signals.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in merged.columns: merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
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
    dpnl_adj = dpnl - n_trades * avg_sz * BYBIT_EXTRA / 10000 / max(len(dpnl), 1)
    dr = dpnl_adj / CAPITAL
    if dr.std() == 0: return None
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    eq = (1 + dr).cumprod()
    dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    worst_day = float(dr.min() * 100)
    cvar5 = float(-dr[dr <= dr.quantile(0.05)].mean() * 10000) if len(dr) > 20 else 0
    pnl = float(dpnl_adj.sum())
    print(f"  {label:<45} SR={sr:.3f} DD={dd:.1f}% dDD={worst_day:.2f}% "
          f"CVaR5={cvar5:.0f}bp PnL=${pnl:,.0f}")
    return {"sr": sr, "dd": dd, "worst_day": worst_day, "cvar5": cvar5, "pnl": pnl}


def main():
    T0 = time.time()
    print("=" * 90)
    print("  DEPTH SCALING VARIANTS on config I (per-asset vol + depth)")
    print("=" * 90)

    print("\nGenerating 15m+1h+4h...")
    s15m = gen_strat("15m", "15min", 4, 0.10, 0.08)
    s1h = gen_strat("1h", "1h", 4, 0.10, 0.08)
    s4h = gen_strat("4h", "4h", 6, 0.20, 0.08)
    combined = pd.concat([s15m, s1h, s4h], ignore_index=True)
    print(f"  Total: {len(combined):,} signals")

    ofi_data = {}
    for name, sym in ASSETS.items():
        p = L2_DIR / f"{sym}_micro_15m.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            ofi_data[sym] = df[["ms_trade_imbalance", "ms_depth_imb_2pct"]]

    vol_series = compute_per_asset_vol()

    scale_fns = {
        "No depth (vol only)":      None,
        "Linear (current)":         linear_scale,
        "Quadratic":                quadratic_scale,
        "Exponential":              exponential_scale,
        "Logarithmic":              log_scale,
        "Sigmoid":                  sigmoid_scale,
        "Step (binary)":            step_scale,
        "Aggressive linear":        aggressive_linear,
        "Conservative linear":      conservative_linear,
    }

    print(f"\n{'='*90}")
    print(f"  SCALING FUNCTION COMPARISON (sz=4%)")
    print(f"{'='*90}")

    for name, fn in scale_fns.items():
        sigs = apply_per_asset_vol(combined.copy(), vol_series)
        if fn is not None:
            sigs = apply_depth_scale(sigs, ofi_data, fn)
        run(sigs, name)

    # Best scaling function at higher sizing
    print(f"\n{'='*90}")
    print(f"  BEST FUNCTIONS AT sz=6%")
    print(f"{'='*90}")
    for name, fn in [("Linear", linear_scale), ("Sigmoid", sigmoid_scale),
                      ("Exponential", exponential_scale), ("Log", log_scale)]:
        sigs = apply_per_asset_vol(combined.copy(), vol_series)
        sigs = apply_depth_scale(sigs, ofi_data, fn)
        run(sigs, f"{name} sz=6%", 0.06)

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
