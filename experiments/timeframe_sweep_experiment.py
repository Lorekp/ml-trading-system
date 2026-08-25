#!/usr/bin/env python
"""Timeframe Sweep — Test 15 timeframes solo + correlation matrix.

Tests: 1m, 5m, 10m, 15m, 1h, 2h, 3h, 4h, 5h, 6h, 7h, 8h, 12h, 16h, 24h
All with fwd_bars=4, C=0.10, Elastic Net, MkrTkr60.

Usage:
    PYTHONPATH=src python experiments/timeframe_sweep_experiment.py [--smoke]
"""
import sys, os, time, warnings, gc
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

SMOKE = "--smoke" in sys.argv
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]

ALL_ASSETS = {
    "BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT",
    "BNB":"BNBUSDT","ADA":"ADAUSDT","AVAX":"AVAXUSDT","DOT":"DOTUSDT",
    "LINK":"LINKUSDT","NEAR":"NEARUSDT","UNI":"UNIUSDT","AAVE":"AAVEUSDT",
    "LTC":"LTCUSDT","BCH":"BCHUSDT","HBAR":"HBARUSDT","TRX":"TRXUSDT",
    "SUI":"SUIUSDT","APT":"APTUSDT","ENJ":"ENJUSDT","SXP":"SXPUSDT",
    "ZEC":"ZECUSDT","XLM":"XLMUSDT","FET":"FETUSDT","WLD":"WLDUSDT",
    "ANKR":"ANKRUSDT","TAO":"TAOUSDT","ENA":"ENAUSDT",
    "FTM":"FTMUSDT","FIL":"FILUSDT","BNX":"BNXUSDT",
}
ASSETS = {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT"} if SMOKE else ALL_ASSETS

OHLCV_FEATURES = [
    "ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
    "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]

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

# (label, resample_str, fwd_bars, C, hold_desc)
ALL_TFS = [
    ("1m",  "1min",  4, 0.10, "4min"),
    ("5m",  "5min",  4, 0.10, "20min"),
    ("10m", "10min", 4, 0.10, "40min"),
    ("15m", "15min", 4, 0.10, "1h"),
    ("1h",  "1h",    4, 0.10, "4h"),
    ("2h",  "2h",    4, 0.10, "8h"),
    ("3h",  "3h",    4, 0.10, "12h"),
    ("4h",  "4h",    4, 0.10, "16h"),
    ("5h",  "5h",    4, 0.10, "20h"),
    ("6h",  "6h",    4, 0.10, "24h"),
    ("7h",  "7h",    4, 0.10, "28h"),
    ("8h",  "8h",    4, 0.10, "32h"),
    ("12h", "12h",   4, 0.10, "48h"),
    ("16h", "16h",   4, 0.10, "64h"),
    ("24h", "1D",    4, 0.10, "4d"),
]
SMOKE_TFS = [
    ("15m", "15min", 4, 0.10, "1h"),
    ("1h",  "1h",    4, 0.10, "4h"),
    ("4h",  "4h",    4, 0.10, "16h"),
]
TFS = SMOKE_TFS if SMOKE else ALL_TFS


def get_vip_tier(equity, volume_30d):
    result = VIP_TIERS[0]
    for tier in VIP_TIERS:
        if equity >= tier[1] or volume_30d >= tier[2]:
            result = tier
    return result[0], result[3], result[4]


# ── Signal generation ──

def _gen_one(args):
    name, sym, tf_resample, fwd_bars, C = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=C, penalty="elasticnet", solver="saga", max_iter=1000,
                          seed=42, fwd_bars=fwd_bars, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=list(OHLCV_FEATURES), winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe=tf_resample,
                        regime_timeframe="4h"),
        assets=ASSETS)
    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
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


def generate_signals(label, tf_resample, fwd_bars, C):
    # Fewer workers for sub-5m to manage memory (huge bar counts)
    workers = 4 if tf_resample in ("1min", "5min") else 8
    tasks = [(n, s, tf_resample, fwd_bars, C) for n, s in ASSETS.items()]
    all_sigs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_gen_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            if r is not None and len(r) > 0:
                all_sigs.append(r)
    if not all_sigs:
        return None
    merged = pd.concat(all_sigs, ignore_index=True)
    merged["strategy"] = label
    merged["hold_bars"] = fwd_bars
    if "confidence" in merged.columns:
        merged = merged[merged["confidence"] >= 0.08].reset_index(drop=True)
    return merged if len(merged) > 0 else None


# ── Vol scaling ──

def _compute_vol_one(args):
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        ohlcv = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
        if ohlcv.index.tz is not None:
            ohlcv.index = ohlcv.index.tz_localize(None)
        return name, ohlcv.pct_change().rolling(20).std()
    except Exception:
        return name, None


def compute_vol_parallel():
    vol = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for name, vs in pool.map(_compute_vol_one, list(ASSETS.items())):
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
    """Returns dict with metrics + daily_pnl Series for correlation."""
    port = PortfolioConfig(
        starting_capital=starting_capital, max_dd=0.10, delever_start=0.05,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
        max_per_strategy=1.0, base_sizing_pct=0.045, max_signal_mult=2.0,
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
    wr = float((dpnl_adj > 0).sum() / len(dpnl_adj)) if len(dpnl_adj) > 0 else 0
    monthly = dpnl_adj.resample("MS").sum()
    pos_months = int((monthly > 0).sum())
    tot_months = len(monthly)
    return {"sr": sr, "sortino": sortino, "dd": dd, "calmar": calmar,
            "pnl": pnl, "pf": pf, "wr": wr, "n_trades": sim["n_trades"],
            "pnl_mo": pnl / max(tot_months, 1),
            "pos_months": pos_months, "tot_months": tot_months,
            "daily_pnl": dpnl_adj}


# ── Main ──

def main():
    T0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print("=" * 100)
    print(f"  TIMEFRAME SWEEP EXPERIMENT ({mode})")
    print(f"  {len(ASSETS)} assets, {len(TFS)} timeframes, fwd=4, C=0.10, MkrTkr60")
    print("=" * 100)

    # Vol scaling (once)
    t0 = time.time()
    vol_series = compute_vol_parallel()
    print(f"  Vol series computed ({time.time()-t0:.0f}s)\n")

    # ── Generate & evaluate each timeframe ──
    results = {}
    daily_pnls = {}

    print(f"  {'TF':<6} {'Hold':<6} {'Signals':>8} {'Sharpe':>8} {'Sortino':>8} {'PnL':>12}"
          f" {'MaxDD':>7} {'Calmar':>7} {'PF':>6} {'WR':>6} {'$/mo':>8} {'Time':>6}")
    print("  " + "-" * 95)

    for label, resample, fwd, C, hold_desc in TFS:
        t1 = time.time()
        sigs = generate_signals(label, resample, fwd, C)
        if sigs is None or len(sigs) == 0:
            print(f"  {label:<6} {hold_desc:<6} {'0':>8} {'---':>8}")
            gc.collect()
            continue

        sigs_vol = apply_vol(sigs, vol_series)
        n_sigs = len(sigs_vol)
        r = run_sim(sigs_vol, CAP)
        elapsed = time.time() - t1

        if r is not None:
            results[label] = r
            daily_pnls[label] = r["daily_pnl"]
            print(f"  {label:<6} {hold_desc:<6} {n_sigs:>8,} {r['sr']:>8.3f} {r['sortino']:>8.3f}"
                  f" ${r['pnl']:>10,.0f} {r['dd']:>6.2f}% {r['calmar']:>7.2f}"
                  f" {r['pf']:>5.2f} {r['wr']:>5.1%} ${r['pnl_mo']:>7,.0f} {elapsed:>5.0f}s")
        else:
            print(f"  {label:<6} {hold_desc:<6} {n_sigs:>8,} {'FAIL':>8} {elapsed:>68.0f}s")

        del sigs, sigs_vol
        gc.collect()

    # ── Ranking ──
    print(f"\n{'='*100}")
    print("  RANKING BY SHARPE")
    print(f"{'='*100}")
    ranked = sorted(results.items(), key=lambda x: x[1]["sr"], reverse=True)
    print(f"\n  {'#':>3} {'TF':<6} {'Sharpe':>8} {'Sortino':>8} {'PnL':>12} {'MaxDD':>7}"
          f" {'Calmar':>7} {'PF':>6} {'$/mo':>8} {'Trades':>8}")
    print("  " + "-" * 80)
    for i, (label, r) in enumerate(ranked, 1):
        print(f"  {i:>3} {label:<6} {r['sr']:>8.3f} {r['sortino']:>8.3f}"
              f" ${r['pnl']:>10,.0f} {r['dd']:>6.2f}% {r['calmar']:>7.2f}"
              f" {r['pf']:>5.2f} ${r['pnl_mo']:>7,.0f} {r['n_trades']:>8,}")

    # ── Correlation matrix ──
    if len(daily_pnls) >= 2:
        print(f"\n{'='*100}")
        print("  DAILY PnL CORRELATION MATRIX")
        print(f"{'='*100}")

        # Align all daily PnL series to common date index
        pnl_df = pd.DataFrame(daily_pnls)
        pnl_df = pnl_df.fillna(0)
        corr = pnl_df.corr()

        # Order by Sharpe ranking
        ordered = [label for label, _ in ranked if label in corr.columns]
        corr = corr.loc[ordered, ordered]

        # Print matrix
        col_w = 6
        header = f"  {'':>{col_w}}"
        for c in ordered:
            header += f" {c:>{col_w}}"
        print(f"\n{header}")
        print("  " + "-" * (col_w + 1 + (col_w + 1) * len(ordered)))

        for row in ordered:
            line = f"  {row:>{col_w}}"
            for col in ordered:
                v = corr.loc[row, col]
                if row == col:
                    line += f" {'1.00':>{col_w}}"
                else:
                    line += f" {v:>{col_w}.2f}"
            print(line)

        # Average pairwise correlation
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        avg_corr = corr.values[mask].mean()
        print(f"\n  Average pairwise correlation: {avg_corr:.3f}")

        # Highlight lowest correlation pairs (best diversification)
        pairs = []
        for i, r in enumerate(ordered):
            for j, c in enumerate(ordered):
                if i < j:
                    pairs.append((r, c, corr.loc[r, c]))
        pairs.sort(key=lambda x: x[2])

        print(f"\n  Lowest correlation pairs (best diversification):")
        for a, b, v in pairs[:5]:
            print(f"    {a:>6} + {b:<6} = {v:+.3f}")

        print(f"\n  Highest correlation pairs (most redundant):")
        for a, b, v in pairs[-5:]:
            print(f"    {a:>6} + {b:<6} = {v:+.3f}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
