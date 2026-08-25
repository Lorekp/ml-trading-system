#!/usr/bin/env python
"""Hold Period Sweep — Vary hold length without retraining.

Model is trained once at fwd_bars=4 per timeframe. Then actual holding period
is varied by recomputing fwd_return from OHLCV at different bar offsets.

Tests: "model trained on 4-bar horizon, but held for N bars"

Usage:
    PYTHONPATH=src python experiments/hold_sweep_experiment.py [--smoke]
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

# Timeframes to test (skip 1m — Sharpe 0.95, not worth 2h compute)
ALL_TFS = [
    ("5m",  "5min"),  ("10m", "10min"), ("15m", "15min"),
    ("1h",  "1h"),    ("2h",  "2h"),    ("3h",  "3h"),
    ("4h",  "4h"),    ("5h",  "5h"),    ("6h",  "6h"),
    ("7h",  "7h"),    ("8h",  "8h"),    ("12h", "12h"),
    ("16h", "16h"),   ("24h", "1D"),
]
SMOKE_TFS = [("15m", "15min"), ("1h", "1h"), ("4h", "4h")]
TFS = SMOKE_TFS if SMOKE else ALL_TFS

HOLD_BARS = [1, 2, 3, 4, 6, 8, 12, 16]
SMOKE_HOLDS = [2, 4, 8]
HOLDS = SMOKE_HOLDS if SMOKE else HOLD_BARS


def get_vip_tier(equity, volume_30d):
    result = VIP_TIERS[0]
    for tier in VIP_TIERS:
        if equity >= tier[1] or volume_30d >= tier[2]:
            result = tier
    return result[0], result[3], result[4]


# ── Signal generation (fwd=4 fixed) ──

def _gen_one(args):
    name, sym, tf_resample = args
    warnings.filterwarnings("ignore")
    fwd_bars, C = 4, 0.10
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


def generate_signals(label, tf_resample):
    workers = 4 if tf_resample in ("1min", "5min") else 8
    tasks = [(n, s, tf_resample) for n, s in ASSETS.items()]
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
    merged["hold_bars"] = 4
    if "confidence" in merged.columns:
        merged = merged[merged["confidence"] >= 0.08].reset_index(drop=True)
    return merged if len(merged) > 0 else None


# ── Recompute returns at different hold periods ──

def recompute_returns(signals, tf_resample, hold_bars_list):
    """Recompute fwd_return from OHLCV at each hold_bars. Returns dict: hold -> DataFrame."""
    # Group signals by asset
    assets_in_signals = signals["asset"].unique()

    # Preload close prices per asset at native TF
    close_cache = {}
    for asset_name in assets_in_signals:
        sym = ASSETS.get(asset_name)
        if sym is None:
            continue
        try:
            ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
            ohlcv_tf = resample_ohlcv(ohlcv_1m, tf_resample)
            close = ohlcv_tf["close"]
            if close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            close_cache[asset_name] = close
            del ohlcv_1m, ohlcv_tf
        except Exception:
            continue
    gc.collect()

    # Normalize signal timestamps
    sig_ts = pd.to_datetime(signals["timestamp"])
    if sig_ts.dt.tz is not None:
        sig_ts = sig_ts.dt.tz_localize(None)

    # Precompute bar indices per asset (once)
    bar_indices = np.full(len(signals), -1, dtype=np.int64)
    entry_prices = np.full(len(signals), np.nan)
    for asset_name, close in close_cache.items():
        mask = (signals["asset"] == asset_name).values
        if not mask.any():
            continue
        idx = close.index.searchsorted(sig_ts[mask].values)
        # Verify timestamps match (should be exact bar boundaries)
        valid = idx < len(close)
        positions = np.where(mask)[0]
        bar_indices[positions[valid]] = idx[valid]
        entry_prices[positions[valid]] = close.values[idx[valid]]

    # For each hold_bars, compute exit prices and fwd_return
    result = {}
    for h in hold_bars_list:
        sigs_h = signals.copy()
        exit_idx = bar_indices + h
        new_fwd = np.full(len(signals), np.nan)

        for asset_name, close in close_cache.items():
            mask = (signals["asset"] == asset_name).values
            positions = np.where(mask)[0]
            for pos in positions:
                bi = bar_indices[pos]
                ei = bi + h
                if bi >= 0 and ei < len(close):
                    new_fwd[pos] = (close.values[ei] - close.values[bi]) / close.values[bi]

        sigs_h["fwd_return"] = new_fwd
        sigs_h["hold_bars"] = h
        sigs_h = sigs_h.dropna(subset=["fwd_return"]).reset_index(drop=True)
        result[h] = sigs_h

    return result


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
    return {"sr": sr, "sortino": sortino, "dd": dd, "calmar": calmar,
            "pnl": pnl, "pf": pf, "n_trades": sim["n_trades"],
            "daily_pnl": dpnl_adj}


# ── Main ──

def main():
    T0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print("=" * 110)
    print(f"  HOLD PERIOD SWEEP ({mode})")
    print(f"  {len(ASSETS)} assets, {len(TFS)} timeframes, hold_bars={HOLDS}")
    print(f"  Model trained at fwd=4, actual hold varied via OHLCV return lookup")
    print("=" * 110)

    # Vol scaling (once)
    t0 = time.time()
    vol_series = compute_vol_parallel()
    print(f"  Vol series computed ({time.time()-t0:.0f}s)")

    # ── Phase 1: Generate signals (once per TF) ──
    print(f"\n{'='*110}")
    print("  PHASE 1: Signal Generation (fwd=4, cached)")
    print(f"{'='*110}")

    signal_cache = {}
    for label, resample in TFS:
        t1 = time.time()
        sigs = generate_signals(label, resample)
        if sigs is not None and len(sigs) > 0:
            signal_cache[label] = sigs
            print(f"  {label:<6} {len(sigs):>8,} signals ({time.time()-t1:.0f}s)", flush=True)
        else:
            print(f"  {label:<6}        0 signals (skipped)", flush=True)
        gc.collect()

    print(f"\n  Cached: {len(signal_cache)} timeframes, "
          f"{sum(len(s) for s in signal_cache.values()):,} total signals")

    # ── Phase 2: Recompute returns + run sims ──
    print(f"\n{'='*110}")
    print("  PHASE 2: Hold Period Variation")
    print(f"{'='*110}")

    # Store results: results[tf][hold] = sim result
    all_results = {}
    daily_pnls = {}  # (tf, hold) -> daily pnl for correlation

    for label, resample in TFS:
        if label not in signal_cache:
            continue
        sigs = signal_cache[label]

        t1 = time.time()
        hold_variants = recompute_returns(sigs, resample, HOLDS)
        load_time = time.time() - t1

        all_results[label] = {}
        for h in HOLDS:
            if h not in hold_variants or len(hold_variants[h]) == 0:
                continue
            sigs_h = apply_vol(hold_variants[h], vol_series)
            r = run_sim(sigs_h, CAP)
            if r is not None:
                all_results[label][h] = r
                daily_pnls[(label, h)] = r["daily_pnl"]

        del hold_variants
        gc.collect()

        # Print row
        n_ok = sum(1 for h in HOLDS if h in all_results[label])
        print(f"  {label:<6} {n_ok}/{len(HOLDS)} holds computed "
              f"(OHLCV load: {load_time:.0f}s, sims: {time.time()-t1-load_time:.0f}s)", flush=True)

    # ── Phase 3: Sharpe heatmap ──
    print(f"\n{'='*110}")
    print("  SHARPE HEATMAP (rows=TF, cols=hold_bars)")
    print(f"{'='*110}")

    # Header
    h_width = 8
    header = f"\n  {'TF':<6}"
    for h in HOLDS:
        header += f" {'h=' + str(h):>{h_width}}"
    header += f" {'BEST':>{h_width}} {'@hold':>{h_width}}"
    print(header)
    print("  " + "-" * (6 + (h_width + 1) * (len(HOLDS) + 2)))

    # Rows sorted by best Sharpe across holds
    tf_best = {}
    for label in all_results:
        best_sr, best_h = -999, 4
        for h, r in all_results[label].items():
            if r["sr"] > best_sr:
                best_sr, best_h = r["sr"], h
        tf_best[label] = (best_sr, best_h)

    ranked_tfs = sorted(tf_best.keys(), key=lambda x: tf_best[x][0], reverse=True)

    for label in ranked_tfs:
        best_sr, best_h = tf_best[label]
        row = f"  {label:<6}"
        for h in HOLDS:
            r = all_results[label].get(h)
            if r is not None:
                marker = "*" if h == best_h else " "
                row += f" {r['sr']:>{h_width-1}.3f}{marker}"
            else:
                row += f" {'---':>{h_width}}"
        row += f" {best_sr:>{h_width}.3f} {'h=' + str(best_h):>{h_width}}"
        print(row)

    # ── PnL heatmap ──
    print(f"\n{'='*110}")
    print("  PnL HEATMAP ($K)")
    print(f"{'='*110}")

    header = f"\n  {'TF':<6}"
    for h in HOLDS:
        header += f" {'h=' + str(h):>{h_width}}"
    print(header)
    print("  " + "-" * (6 + (h_width + 1) * len(HOLDS)))

    for label in ranked_tfs:
        row = f"  {label:<6}"
        for h in HOLDS:
            r = all_results[label].get(h)
            if r is not None:
                row += f" {r['pnl']/1000:>{h_width-1}.0f}K"
            else:
                row += f" {'---':>{h_width}}"
        print(row)

    # ── MaxDD heatmap ──
    print(f"\n{'='*110}")
    print("  MaxDD HEATMAP (%)")
    print(f"{'='*110}")

    header = f"\n  {'TF':<6}"
    for h in HOLDS:
        header += f" {'h=' + str(h):>{h_width}}"
    print(header)
    print("  " + "-" * (6 + (h_width + 1) * len(HOLDS)))

    for label in ranked_tfs:
        row = f"  {label:<6}"
        for h in HOLDS:
            r = all_results[label].get(h)
            if r is not None:
                row += f" {r['dd']:>{h_width-1}.1f}%"
            else:
                row += f" {'---':>{h_width}}"
        print(row)

    # ── Optimal configs summary ──
    print(f"\n{'='*110}")
    print("  OPTIMAL HOLD PER TIMEFRAME")
    print(f"{'='*110}")

    print(f"\n  {'TF':<6} {'Best Hold':>10} {'Sharpe':>8} {'Sortino':>8} {'PnL':>12}"
          f" {'MaxDD':>7} {'Calmar':>7} {'PF':>6} {'Trades':>8}")
    print("  " + "-" * 80)

    for label in ranked_tfs:
        best_sr, best_h = tf_best[label]
        r = all_results[label][best_h]
        print(f"  {label:<6} {'h=' + str(best_h):>10} {r['sr']:>8.3f} {r['sortino']:>8.3f}"
              f" ${r['pnl']:>10,.0f} {r['dd']:>6.2f}% {r['calmar']:>7.2f}"
              f" {r['pf']:>5.2f} {r['n_trades']:>8,}")

    # ── Correlation matrix (best hold per TF) ──
    if len(daily_pnls) >= 2:
        print(f"\n{'='*110}")
        print("  DAILY PnL CORRELATION MATRIX (each TF at its optimal hold)")
        print(f"{'='*110}")

        best_pnls = {}
        for label in ranked_tfs:
            best_h = tf_best[label][1]
            key = (label, best_h)
            if key in daily_pnls:
                best_pnls[label] = daily_pnls[key]

        pnl_df = pd.DataFrame(best_pnls).fillna(0)
        corr = pnl_df.corr()
        ordered = [l for l in ranked_tfs if l in corr.columns]
        corr = corr.loc[ordered, ordered]

        col_w = 6
        header = f"  {'':>{col_w}}"
        for c in ordered:
            header += f" {c:>{col_w}}"
        print(f"\n{header}")
        print("  " + "-" * (col_w + 1 + (col_w + 1) * len(ordered)))

        for row_label in ordered:
            line = f"  {row_label:>{col_w}}"
            for col_label in ordered:
                v = corr.loc[row_label, col_label]
                if row_label == col_label:
                    line += f" {'1.00':>{col_w}}"
                else:
                    line += f" {v:>{col_w}.2f}"
            print(line)

        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        avg_corr = corr.values[mask].mean()
        print(f"\n  Average pairwise correlation: {avg_corr:.3f}")

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
