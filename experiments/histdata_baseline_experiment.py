#!/usr/bin/env python
"""HistData Baseline Experiment — Test production config on non-crypto assets.

Runs the exact same pipeline (Elastic Net, 15m+1h+4h, vol scaling, MkrTkr60)
on 13 forex/commodity/index instruments from histdata.com (2019-2026, 1m bars).

Key differences from crypto:
  - Volume is 0 (tick data without volume). vol_zscore feature = 0 (harmless).
  - Market hours vary: forex 24/5, indices ~23/5, commodities ~23/5.
  - Prices are in USD (e.g., EURUSD=1.18, XAUUSD=2000, SPXUSD=5000).

Usage:
    PYTHONPATH=src python experiments/histdata_baseline_experiment.py [--smoke]
"""
import sys, os, time, warnings, gc, glob
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.data import resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_regime_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.config import *

SMOKE = "--smoke" in sys.argv
HISTDATA_DIR = Path("data/histdata-1m")

ALL_ASSETS = {
    # Forex
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD", "NZDUSD": "NZDUSD", "USDCAD": "USDCAD",
    "USDCHF": "USDCHF", "EURJPY": "EURJPY",
    # Commodities
    "XAUUSD": "XAUUSD", "XAGUSD": "XAGUSD", "BCOUSD": "BCOUSD",
    # Indices
    "SPXUSD": "SPXUSD", "NSXUSD": "NSXUSD",
}
SMOKE_ASSETS = {"EURUSD": "EURUSD", "XAUUSD": "XAUUSD", "SPXUSD": "SPXUSD"}
ASSETS = SMOKE_ASSETS if SMOKE else ALL_ASSETS

OHLCV_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore", "rsi_14",
    "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14", "dist_swing_high", "dist_swing_low"]

VIP_TIERS = [
    ("VIP0",    0,           0,             2.00, 5.50),
    ("VIP1",    100_000,     10_000_000,    1.80, 4.00),
    ("VIP2",    250_000,     25_000_000,    1.60, 3.75),
    ("VIP3",    500_000,     50_000_000,    1.40, 3.50),
    ("VIP4",    1_000_000,   100_000_000,   1.20, 3.20),
    ("VIP5",    2_000_000,   250_000_000,   1.00, 3.20),
    ("Supreme", 5_000_000,   500_000_000,   0.00, 3.00),
]
SPREAD_BPS = 0.5
CAP = 250_000


def get_vip_tier(equity, volume_30d):
    result = VIP_TIERS[0]
    for tier in VIP_TIERS:
        if equity >= tier[1] or volume_30d >= tier[2]:
            result = tier
    return result[0], result[3], result[4]


# ── HistData loader ──

def load_histdata_1m(symbol):
    """Load histdata 1m CSVs for a symbol. Returns OHLCV DataFrame matching Binance format."""
    sym_dir = HISTDATA_DIR / symbol
    if not sym_dir.exists():
        raise FileNotFoundError(f"No histdata directory for {symbol}")

    csv_files = sorted(sym_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {sym_dir}")

    frames = []
    for fp in csv_files:
        df = pd.read_csv(fp, sep=";", header=None,
                         names=["timestamp", "open", "high", "low", "close", "volume"])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], format="%Y%m%d %H%M%S")
    combined = combined.set_index("timestamp").sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined = combined.dropna(subset=["close"])

    return combined


# ── Signal generation ──

def _gen(args):
    """Generate signals for one asset at one timeframe config."""
    name, tf_resample, fwd_bars, C = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=C, penalty="elasticnet", solver="saga", max_iter=1000,
                          seed=42, fwd_bars=fwd_bars, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=list(OHLCV_FEATURES), winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(HISTDATA_DIR)], timeframe=tf_resample,
                        regime_timeframe="4h"),
        assets=ASSETS)
    try:
        ohlcv_1m = load_histdata_1m(name)
    except FileNotFoundError:
        return None
    ohlcv_tf = resample_ohlcv(ohlcv_1m, tf_resample)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    feat = compute_features(ohlcv_tf, config.features)
    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0:
        return None
    sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    return sigs if len(sigs) > 0 else None


def generate_all_signals():
    """Generate multi-timeframe signals identical to production config."""
    strat_sigs = {}
    tfs = [("15m", "15min", 4, 0.10)] if SMOKE else [
        ("15m", "15min", 4, 0.10), ("1h", "1h", 4, 0.10), ("4h", "4h", 6, 0.20)]
    for sname, tf, fwd, C in tfs:
        tasks = [(n, tf, fwd, C) for n in ASSETS.keys()]
        all_sigs = []
        with ProcessPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(_gen, t): t for t in tasks}
            for f in as_completed(futs):
                r = f.result()
                if r is not None and len(r) > 0:
                    all_sigs.append(r)
        if all_sigs:
            m = pd.concat(all_sigs, ignore_index=True)
            m["strategy"] = sname
            m["hold_bars"] = fwd
            if "confidence" in m.columns:
                m = m[m["confidence"] >= 0.08].reset_index(drop=True)
            strat_sigs[sname] = m
            print(f"    {sname}: {len(m):,} signals", flush=True)
    return strat_sigs


# ── Vol scaling ──

def _compute_vol_one(name):
    warnings.filterwarnings("ignore")
    try:
        ohlcv = resample_ohlcv(load_histdata_1m(name), "1D")["close"]
        if ohlcv.index.tz is not None:
            ohlcv.index = ohlcv.index.tz_localize(None)
        return name, ohlcv.pct_change().rolling(20).std()
    except Exception:
        return name, None


def compute_vol_parallel():
    vol = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for name, vs in pool.map(_compute_vol_one, list(ASSETS.keys())):
            if vs is not None:
                vol[name] = vs
    return vol


def apply_vol(signals, vol_series):
    out = signals.copy()
    if "sizing_mult" not in out.columns:
        out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        if asset not in vol_series:
            continue
        vs = vol_series[asset]
        idx = vs.index.searchsorted(ts.iloc[i])
        if 0 < idx < len(vs):
            v = vs.iloc[idx - 1]
            if pd.notna(v) and v > 0:
                out.iat[i, out.columns.get_loc("sizing_mult")] *= np.clip(0.025 / v, 0.3, 1.5)
    return out


# ── Portfolio sim with MkrTkr60 VIP fees ──

def run_sim(signals_prepared, starting_capital):
    """Run portfolio sim with MkrTkr60 VIP fee overlay."""
    port = PortfolioConfig(
        starting_capital=starting_capital, max_dd=0.10, delever_start=0.05,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
        max_per_strategy=0.50, base_sizing_pct=0.045, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
    merged = signals_prepared.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in merged.columns:
            merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    sim = simulate_portfolio(merged, port)
    if sim is None:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    # MkrTkr60 VIP fee overlay
    maker_fill_pct = 0.90
    active = ~sim["skipped"]
    sizes = sim["sizes"]
    ts_arr = pd.to_datetime(merged["timestamp"].values)
    if ts_arr.tz is not None:
        ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date
    daily_vol = {}
    for i in range(len(merged)):
        if active[i]:
            daily_vol[dates_arr[i]] = daily_vol.get(dates_arr[i], 0) + abs(sizes[i]) * 2
    vol_s = pd.Series(daily_vol).sort_index()
    vol_30d = vol_s.rolling(30, min_periods=1).sum()
    eq_curve = starting_capital + dpnl.cumsum()

    dpnl_adj = dpnl.copy()
    for d in dpnl.index:
        d_date = d.date()
        eq = eq_curve.get(d, starting_capital)
        v30 = vol_30d.get(d_date, 0)
        _, maker_bps, taker_bps = get_vip_tier(eq, v30)
        rt_bps = (maker_fill_pct * 2 * maker_bps +
                  (1 - maker_fill_pct) * 2 * taker_bps + SPREAD_BPS)
        extra_bps = rt_bps - 1.0
        n_day = sum(1 for i in range(len(merged)) if active[i] and dates_arr[i] == d_date)
        avg_sz = daily_vol.get(d_date, 0) / max(n_day * 2, 1)
        dpnl_adj[d] -= n_day * avg_sz * extra_bps / 10000

    dr = dpnl_adj / starting_capital
    if dr.std() == 0 or len(dr) < 30:
        return None
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq_adj = (1 + dr).cumprod()
    dd = float(((eq_adj.cummax() - eq_adj) / eq_adj.cummax()).max() * 100)
    calmar = float(dr.mean() * 365.25 / (dd / 100)) if dd > 0 else 0
    pnl = float(dpnl_adj.sum())
    pf = float(dpnl_adj[dpnl_adj > 0].sum() / abs(dpnl_adj[dpnl_adj < 0].sum())) if (dpnl_adj < 0).any() else 99
    monthly = dpnl_adj.resample("MS").sum()
    pos_months = int((monthly > 0).sum())
    tot_months = len(monthly)
    return {"sr": sr, "sortino": sortino, "dd": dd, "calmar": calmar,
            "pnl": pnl, "pf": pf, "n_trades": sim["n_trades"], "monthly": monthly,
            "pos_months": pos_months, "tot_months": tot_months,
            "pnl_mo": pnl / max(tot_months, 1)}


# ── Main ──

def main():
    T0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print("=" * 80)
    print(f"  HISTDATA BASELINE EXPERIMENT ({mode})")
    print(f"  {len(ASSETS)} assets, production config (EN, vol scaling, MkrTkr60)")
    print(f"  Data: histdata.com 1m bars, 2019-2026")
    print("=" * 80)

    # Data check
    print(f"\n  Loading 1m data...", flush=True)
    for name in ASSETS:
        try:
            df = load_histdata_1m(name)
            print(f"    {name}: {len(df):>10,} bars  {df.index[0].date()} to {df.index[-1].date()}")
        except Exception as e:
            print(f"    {name}: FAILED ({e})")

    # Generate signals
    print(f"\n  Generating signals (production config)...", flush=True)
    t0 = time.time()
    strat_sigs = generate_all_signals()
    if not strat_sigs:
        print("  ERROR: No signals generated. Exiting.")
        return
    combined = pd.concat(list(strat_sigs.values()), ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    total = len(combined)
    print(f"  Total: {total:,} signals ({time.time()-t0:.0f}s)", flush=True)

    # Per-asset signal breakdown
    print(f"\n  Signals per asset:")
    for name in sorted(ASSETS.keys()):
        n = len(combined[combined["asset"] == name])
        if n > 0:
            print(f"    {name:<10} {n:>6,}")

    # Per-strategy breakdown
    print(f"\n  Signals per timeframe:")
    for strat in combined["strategy"].unique():
        n = len(combined[combined["strategy"] == strat])
        print(f"    {strat:<6} {n:>6,}")

    # Vol scaling
    print(f"\n  Computing vol scaling...", flush=True)
    t0 = time.time()
    vol_series = compute_vol_parallel()
    combined = apply_vol(combined, vol_series)
    print(f"  Done ({time.time()-t0:.0f}s)")

    # Portfolio sim
    print(f"\n{'='*80}")
    print("  PORTFOLIO RESULTS (MkrTkr60, VIP fees, $250K)")
    print(f"{'='*80}")

    result = run_sim(combined, CAP)
    if result is None:
        print("  Portfolio sim returned None — not enough data or trades.")
        return

    print(f"\n  Sharpe:      {result['sr']:.3f}")
    print(f"  Sortino:     {result['sortino']:.3f}")
    print(f"  PnL:         ${result['pnl']:,.0f}")
    print(f"  MaxDD:       {result['dd']:.2f}%")
    print(f"  Calmar:      {result['calmar']:.2f}")
    print(f"  PF:          {result['pf']:.2f}")
    print(f"  Trades:      {result['n_trades']:,}")
    print(f"  $/month:     ${result['pnl_mo']:,.0f}")
    print(f"  Pos months:  {result['pos_months']}/{result['tot_months']}")

    # Monthly PnL breakdown
    print(f"\n{'='*80}")
    print("  MONTHLY PnL BREAKDOWN")
    print(f"{'='*80}")

    monthly = result["monthly"]
    yearly = {}
    print(f"\n  {'Month':<10} {'PnL':>14}")
    print("  " + "-" * 26)
    for m in monthly.index:
        val = monthly[m]
        yearly[m.year] = yearly.get(m.year, 0) + val
        print(f"  {m.strftime('%Y-%m'):<10} ${val:>12,.0f}")

    print("  " + "-" * 26)
    for yr in sorted(yearly.keys()):
        print(f"  {yr:<10} ${yearly[yr]:>12,.0f}")
    print("  " + "=" * 26)
    print(f"  {'TOTAL':<10} ${result['pnl']:>12,.0f}")

    # Compare with crypto baseline
    print(f"\n{'='*80}")
    print("  COMPARISON vs CRYPTO BASELINE")
    print(f"{'='*80}")
    print(f"\n  {'Metric':<16} {'HistData':>12} {'Crypto':>12}")
    print("  " + "-" * 42)
    print(f"  {'Sharpe':<16} {result['sr']:>12.3f} {'2.929':>12}")
    print(f"  {'Sortino':<16} {result['sortino']:>12.3f} {'3.266':>12}")
    print(f"  {'PnL':<16} ${result['pnl']:>10,.0f} {'$321,899':>12}")
    print(f"  {'MaxDD':<16} {result['dd']:>11.2f}% {'7.28%':>12}")
    print(f"  {'Calmar':<16} {result['calmar']:>12.2f} {'3.38':>12}")
    print(f"  {'PF':<16} {result['pf']:>12.2f} {'2.10':>12}")
    print(f"  {'$/month':<16} ${result['pnl_mo']:>10,.0f} {'$4,734':>12}")
    print(f"  {'Assets':<16} {len(ASSETS):>12} {'30':>12}")
    print(f"  {'Trades':<16} {result['n_trades']:>12,} {'68,268':>12}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
