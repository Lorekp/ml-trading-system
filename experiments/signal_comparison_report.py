"""Full multi-asset signal comparison report — vectorized with caching.

Vectorized walk-forward: train once per month, predict ALL test bars at once.
Per-asset per-lookback predictions cached to parquet.

Usage:
    python experiments/signal_comparison_report.py
    ML_SIGNAL_ASSETS=BTC,ETH,SOL python experiments/signal_comparison_report.py
"""

import sys
sys.path.insert(0, "src")

import hashlib
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)

# ============================================================
# Configuration
# ============================================================

ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "ZEC": "ZECUSDT", "BNX": "BNXUSDT", "BNB": "BNBUSDT", "ENJ": "ENJUSDT",
    "TAO": "TAOUSDT", "ADA": "ADAUSDT", "ANKR": "ANKRUSDT", "SUI": "SUIUSDT",
    "AVAX": "AVAXUSDT", "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "FIL": "FILUSDT",
    "DOT": "DOTUSDT", "SXP": "SXPUSDT", "ENA": "ENAUSDT", "BCH": "BCHUSDT",
    "LTC": "LTCUSDT", "UNI": "UNIUSDT", "WLD": "WLDUSDT", "TRX": "TRXUSDT",
    "FTM": "FTMUSDT", "AAVE": "AAVEUSDT", "FET": "FETUSDT", "APT": "APTUSDT",
    "HBAR": "HBARUSDT", "XLM": "XLMUSDT",
}

LOOKBACKS = [2, 4, 6, 8, 10, 12]
FWD_BARS = 4
SLIPPAGE = 0.0001
WINSOR_COLS = ["ret_1", "ret_3", "ret_7", "ret_14"]
VOL_FILTER_QUANTILE = 0.50

SEARCH_START = "2022-01"
SEARCH_END = "2024-12"
HOLDOUT_START = "2025-01"

CACHE_DIR = Path("experiments/cache/signal_comparison")
CACHE_VERSION = "v2"  # v1 had regime feature leakage (unlagged ffill)

WORKERS = min(8, max(1, os.cpu_count() or 4))
XGB_NJOBS = max(1, (os.cpu_count() or 4) // WORKERS)


# ============================================================
# Vectorized walk-forward per asset
# ============================================================

def data_fingerprint(close_arr, n_features):
    """Quick hash of data shape + first/last values for cache invalidation."""
    h = hashlib.md5()
    h.update(f"{len(close_arr)}_{n_features}_{close_arr[0]:.6f}_{close_arr[-1]:.6f}".encode())
    return h.hexdigest()[:12]


def generate_asset_signals(asset, symbol, lookbacks, cfg):
    """Vectorized walk-forward signal generation for one asset, all lookbacks.

    Returns dict[lookback] -> DataFrame with columns:
        timestamp, direction, fwd_return, pnl, confidence, probability, asset
    """
    data_dirs = [Path(p) for p in cfg.data.data_dirs] or [Path("data/binance-1m")]

    try:
        ohlcv_1m = load_binance_1m(symbol, data_dirs)
    except Exception as e:
        print(f"  SKIP {asset}: {e}", flush=True)
        return {}

    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    if len(ohlcv_15m) < 5000:
        print(f"  SKIP {asset}: only {len(ohlcv_15m)} bars", flush=True)
        return {}

    # Compute features once
    base_features = compute_features(ohlcv_15m, cfg.features)
    regime_parts = []
    for rtf in cfg.close_model.regime_timeframes:
        rtf_ohlcv = resample_ohlcv(ohlcv_1m, rtf)
        rtf_feat = compute_features(rtf_ohlcv, cfg.features)
        # CRITICAL: shift by 1 period so features only use COMPLETED bars (no future leakage)
        aligned = rtf_feat.shift(1).reindex(ohlcv_15m.index, method="ffill")
        aligned = aligned.add_prefix(f"signal_regime_{rtf}_")
        regime_parts.append(aligned)
    all_features = pd.concat([base_features] + regime_parts, axis=1)

    close = ohlcv_15m["close"].values.astype(np.float64)
    vol = ohlcv_15m["close"].pct_change().ewm(span=50, min_periods=20).std().values.astype(np.float64)
    timestamps = ohlcv_15m.index
    n_total = len(close)

    # Pre-compute forward returns for ALL bars
    fwd_ret = np.full(n_total, np.nan, dtype=np.float64)
    valid = np.arange(n_total - FWD_BARS)
    mask = close[valid] > 0
    fwd_ret[valid[mask]] = (close[valid[mask] + FWD_BARS] - close[valid[mask]]) / close[valid[mask]]

    # Pre-compute labels
    labels_all = np.full(n_total, -1, dtype=np.int8)
    finite_mask = np.isfinite(fwd_ret)
    labels_all[finite_mask] = (fwd_ret[finite_mask] > 0).astype(np.int8)

    # Feature array (float32 for XGBoost)
    feat_arr = all_features.values.astype(np.float32)
    feat_cols = list(all_features.columns)
    n_features = len(feat_cols)

    # Month boundaries
    if hasattr(timestamps, 'tz') and timestamps.tz is not None:
        month_keys = (timestamps.year * 12 + timestamps.month).values
    else:
        ts_pd = pd.DatetimeIndex(timestamps)
        month_keys = (ts_pd.year * 12 + ts_pd.month).values

    unique_months = np.unique(month_keys)

    # Data fingerprint for caching
    dfp = data_fingerprint(close, n_features)

    results = {}
    for lb in lookbacks:
        # Check cache
        cache_path = CACHE_DIR / f"{CACHE_VERSION}_{asset}_lb{lb}_{dfp}.parquet"
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                results[lb] = df
                continue
            except Exception:
                pass

        all_month_trades = []

        for mk in unique_months:
            # Test bars: this month
            test_mask = month_keys == mk
            test_indices = np.where(test_mask)[0]
            if len(test_indices) < 10:
                continue

            # Training window: lookback months before this month
            train_end_mk = mk - 1  # previous month
            train_start_mk = mk - lb
            train_mask = (month_keys >= train_start_mk) & (month_keys <= train_end_mk)

            # Embargo: exclude last fwd_bars+1 bars of training
            train_indices = np.where(train_mask)[0]
            if len(train_indices) < 200:
                continue
            embargo = FWD_BARS + 1
            safe_train_indices = train_indices[:-embargo] if len(train_indices) > embargo else train_indices

            # Filter to valid labels
            valid_train = safe_train_indices[labels_all[safe_train_indices] >= 0]
            if len(valid_train) < 200:
                continue

            # Training data
            train_X = feat_arr[valid_train].copy()
            train_y = labels_all[valid_train]

            # Winsorize training features
            winsor_bounds = {}
            for col_name in WINSOR_COLS:
                if col_name in feat_cols:
                    ci = feat_cols.index(col_name)
                    lo, hi = np.nanquantile(train_X[:, ci], [0.01, 0.99])
                    train_X[:, ci] = np.clip(train_X[:, ci], lo, hi)
                    winsor_bounds[ci] = (lo, hi)

            # Sample weights
            train_sw = compute_sample_weights(
                np.arange(len(valid_train), dtype=np.int64), fwd_bars=FWD_BARS)

            # Val split: last month of training
            train_months = month_keys[valid_train]
            val_mk = train_months.max()
            val_mask_t = train_months == val_mk
            fit_mask_t = ~val_mask_t
            use_val = fit_mask_t.sum() >= 200 and val_mask_t.sum() >= 50

            if not use_val:
                fit_mask_t = np.ones(len(valid_train), dtype=bool)
                val_mask_t = np.zeros(len(valid_train), dtype=bool)

            # Train XGBoost
            model = xgb.XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                reg_alpha=0.1, reg_lambda=1.0,
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=42, n_jobs=XGB_NJOBS, verbosity=0,
                early_stopping_rounds=25 if use_val else None,
            )
            fit_kwargs = {"sample_weight": train_sw[fit_mask_t], "verbose": False}
            if use_val:
                fit_kwargs["eval_set"] = [(train_X[val_mask_t], train_y[val_mask_t])]
            model.fit(train_X[fit_mask_t], train_y[fit_mask_t], **fit_kwargs)

            # Vol threshold from training period (3-month window)
            vol_window = train_indices[-min(len(train_indices), 3 * 30 * 24 * 4):]
            vol_vals = vol[vol_window]
            vol_vals = vol_vals[np.isfinite(vol_vals)]
            vol_threshold = np.quantile(vol_vals, VOL_FILTER_QUANTILE) if len(vol_vals) > 50 else 0.0

            # VECTORIZED PREDICTION: predict ALL test bars at once
            test_X = feat_arr[test_indices].copy()
            for ci, (lo, hi) in winsor_bounds.items():
                test_X[:, ci] = np.clip(test_X[:, ci], lo, hi)

            # Handle NaN/inf in test features
            finite_test = np.isfinite(test_X).all(axis=1)
            vol_test = vol[test_indices]
            vol_pass = vol_test >= vol_threshold

            # Combined mask: finite features + vol filter + valid forward return
            fwd_valid = np.isfinite(fwd_ret[test_indices])
            predict_mask = finite_test & vol_pass & fwd_valid

            if predict_mask.sum() == 0:
                continue

            pred_indices = test_indices[predict_mask]
            pred_X = test_X[predict_mask]

            probs = model.predict_proba(pred_X)[:, 1]
            directions = np.where(probs > 0.5, 1.0, -1.0)
            confidences = np.abs(probs - 0.5)
            fwd_rets = fwd_ret[pred_indices]
            pnls = directions * fwd_rets - SLIPPAGE

            month_df = pd.DataFrame({
                "timestamp": timestamps[pred_indices],
                "direction": directions,
                "fwd_return": fwd_rets,
                "pnl": pnls,
                "confidence": confidences,
                "probability": probs,
                "entry_price": close[pred_indices],
                "asset": asset,
            })
            all_month_trades.append(month_df)

        if all_month_trades:
            df = pd.concat(all_month_trades, ignore_index=True)
            # Cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                df.to_parquet(cache_path)
            except Exception:
                pass
            results[lb] = df
        else:
            results[lb] = pd.DataFrame()

    n_total_trades = sum(len(v) for v in results.values())
    print(f"  {asset:<6s} done: {n_total_trades:>7d} trades across {len(lookbacks)} lookbacks", flush=True)
    return results


# ============================================================
# Stats
# ============================================================

def compute_period_stats(trades_df, period_start=None, period_end=None, bet_size=5.0):
    if trades_df is None or len(trades_df) == 0:
        return None

    df = trades_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ym"] = df["ts"].dt.strftime("%Y-%m")

    if period_start:
        df = df[df["ym"] >= period_start]
    if period_end:
        df = df[df["ym"] <= period_end]
    if len(df) < 10:
        return None

    pnls = df["pnl"].values
    daily = pd.Series(bet_size * pnls, index=df["ts"].values).resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 5:
        return None

    capital = 10000.0
    cumulative = daily.cumsum()
    equity = capital + cumulative
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = float(dd.min()) * 100

    std = daily.std()
    sharpe = float(daily.mean() / std * np.sqrt(365.25)) if std > 0 else 0.0
    total_pnl = float(daily.sum())
    n_trades = len(df)
    n_days = len(daily)
    pnl_day = total_pnl / max(n_days, 1)
    win_rate = float((pnls > 0).mean()) * 100
    mean_bps = float(pnls.mean() * 10000)
    long_pct = float((df["direction"].values > 0).mean()) * 100
    n_assets = df["asset"].nunique()

    return {
        "sharpe": sharpe, "total_return_pct": total_pnl / capital * 100,
        "max_dd_pct": max_dd, "n_trades": n_trades, "n_days": n_days,
        "pnl_day": pnl_day, "win_rate": win_rate, "mean_bps": mean_bps,
        "long_pct": long_pct, "n_assets": n_assets, "status": "ok",
    }


# ============================================================
# Reporting
# ============================================================

def print_table(title, rows):
    print(f"\n{'=' * 100}")
    print(title)
    print(f"{'=' * 100}")
    print(f"  {'LB':<5s} {'Sharpe':>7s} {'Ret%':>8s} {'MaxDD%':>8s} "
          f"{'Trades':>8s} {'Days':>6s} {'PnL/day':>9s} {'Win%':>6s} {'bps':>6s} "
          f"{'Long%':>6s} {'Assets':>6s}")
    print("  " + "-" * 93)
    for lb, row in rows:
        if row is None or row.get("status") != "ok":
            print(f"  {lb:>2d}mo   NO RESULT")
            continue
        print(f"  {lb:>2d}mo  {row['sharpe']:>7.3f} {row['total_return_pct']:>8.2f} "
              f"{row['max_dd_pct']:>8.2f} {row['n_trades']:>8d} {row['n_days']:>6d} "
              f"${row['pnl_day']:>8.2f} {row['win_rate']:>5.1f}% {row['mean_bps']:>5.1f} "
              f"{row['long_pct']:>5.1f}% {row['n_assets']:>5d}")


def print_asset_table(title, asset_stats):
    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")
    print(f"  {'Asset':<8s} {'Sharpe':>7s} {'Win%':>6s} {'bps':>6s} {'Trades':>8s} {'Long%':>6s}")
    print("  " + "-" * 45)
    for asset, row in sorted(asset_stats.items(), key=lambda x: -x[1].get("sharpe", 0)):
        if row and row.get("status") == "ok":
            print(f"  {asset:<8s} {row['sharpe']:>7.3f} {row['win_rate']:>5.1f}% "
                  f"{row['mean_bps']:>5.1f} {row['n_trades']:>8d} {row['long_pct']:>5.1f}%")


# ============================================================
# Main
# ============================================================

def main():
    t_global = time.time()
    cfg = load_config(Path("configs/default.yaml"))

    selected = os.getenv("ML_SIGNAL_ASSETS", "").strip()
    if selected:
        asset_list = {k: v for k, v in ASSETS.items() if k in selected.split(",")}
    else:
        asset_list = ASSETS

    print(f"Signal Comparison Report — 15min, XGBoost, {len(asset_list)} assets")
    print(f"  Lookbacks: {LOOKBACKS}")
    print(f"  Workers: {WORKERS}, XGB n_jobs={XGB_NJOBS}")
    print(f"  Cache: {CACHE_DIR}")

    # ============================================================
    # Phase 1: Generate signals (parallelized across assets)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"PHASE 1: SIGNAL GENERATION ({len(asset_list)} assets × {len(LOOKBACKS)} lookbacks)")
    print(f"{'='*70}")

    # all_signals[asset][lookback] = DataFrame
    all_signals = {}
    t1 = time.time()

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}
        for asset, symbol in asset_list.items():
            f = pool.submit(generate_asset_signals, asset, symbol, LOOKBACKS, cfg)
            futures[f] = asset

        for f in as_completed(futures):
            asset = futures[f]
            try:
                all_signals[asset] = f.result()
            except Exception as e:
                print(f"  ERROR {asset}: {e}", flush=True)
                all_signals[asset] = {}

    print(f"\n  Phase 1 done in {time.time()-t1:.0f}s")

    # ============================================================
    # Phase 2: Aggregate by lookback, compute stats
    # ============================================================
    print(f"\n{'='*70}")
    print("PHASE 2: AGGREGATION & STATS")
    print(f"{'='*70}")

    for lb in LOOKBACKS:
        # Combine all assets for this lookback
        parts = []
        for asset, sigs in all_signals.items():
            if lb in sigs and len(sigs[lb]) > 0:
                parts.append(sigs[lb])
        if parts:
            combined = pd.concat(parts, ignore_index=True)
        else:
            combined = pd.DataFrame()

        search = compute_period_stats(combined, SEARCH_START, SEARCH_END)
        holdout = compute_period_stats(combined, HOLDOUT_START)

        s_str = f"SR={search['sharpe']:.2f} Win={search['win_rate']:.1f}% n={search['n_trades']}" if search else "NO DATA"
        h_str = f"SR={holdout['sharpe']:.2f} Win={holdout['win_rate']:.1f}% n={holdout['n_trades']}" if holdout else "NO DATA"
        print(f"  lb={lb:>2d}mo: search={s_str}  holdout={h_str}")

    # ============================================================
    # Phase 3: Full reports
    # ============================================================

    # SEARCH period table
    search_rows = []
    for lb in LOOKBACKS:
        parts = [sigs[lb] for sigs in all_signals.values() if lb in sigs and len(sigs.get(lb, pd.DataFrame())) > 0]
        combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        search_rows.append((lb, compute_period_stats(combined, SEARCH_START, SEARCH_END)))
    print_table(f"SEARCH PERIOD ({SEARCH_START} to {SEARCH_END}) — 15min, 30 assets", search_rows)

    # HOLDOUT period table
    holdout_rows = []
    for lb in LOOKBACKS:
        parts = [sigs[lb] for sigs in all_signals.values() if lb in sigs and len(sigs.get(lb, pd.DataFrame())) > 0]
        combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        holdout_rows.append((lb, compute_period_stats(combined, HOLDOUT_START)))
    print_table(f"HOLDOUT PERIOD ({HOLDOUT_START}+) — 15min, 30 assets", holdout_rows)

    # FULL period table
    full_rows = []
    for lb in LOOKBACKS:
        parts = [sigs[lb] for sigs in all_signals.values() if lb in sigs and len(sigs.get(lb, pd.DataFrame())) > 0]
        combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        full_rows.append((lb, compute_period_stats(combined)))
    print_table("FULL PERIOD — 15min, 30 assets", full_rows)

    # ============================================================
    # Per-asset breakdown (best lookback)
    # ============================================================
    best_lb = max(search_rows, key=lambda x: x[1]["sharpe"] if x[1] else -999)[0]
    print(f"\n  Best search lookback: {best_lb}mo")

    asset_search_stats = {}
    for asset, sigs in all_signals.items():
        if best_lb in sigs and len(sigs[best_lb]) > 0:
            st = compute_period_stats(sigs[best_lb], SEARCH_START, SEARCH_END)
            if st:
                asset_search_stats[asset] = st

    print_asset_table(f"PER-ASSET BREAKDOWN — SEARCH, lb={best_lb}mo", asset_search_stats)

    # Holdout per-asset
    asset_holdout_stats = {}
    for asset, sigs in all_signals.items():
        if best_lb in sigs and len(sigs[best_lb]) > 0:
            st = compute_period_stats(sigs[best_lb], HOLDOUT_START)
            if st:
                asset_holdout_stats[asset] = st

    if asset_holdout_stats:
        print_asset_table(f"PER-ASSET BREAKDOWN — HOLDOUT, lb={best_lb}mo", asset_holdout_stats)

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    best_search = max(search_rows, key=lambda x: x[1]["sharpe"] if x[1] else -999)
    best_holdout = max(holdout_rows, key=lambda x: x[1]["sharpe"] if x[1] else -999)

    if best_search[1]:
        s = best_search[1]
        print(f"  Best SEARCH:  lb={best_search[0]}mo  Sharpe={s['sharpe']:.3f}  "
              f"Win={s['win_rate']:.1f}%  {s['n_trades']} trades  ${s['pnl_day']:.2f}/day")
    if best_holdout[1]:
        h = best_holdout[1]
        print(f"  Best HOLDOUT: lb={best_holdout[0]}mo  Sharpe={h['sharpe']:.3f}  "
              f"Win={h['win_rate']:.1f}%  {h['n_trades']} trades  ${h['pnl_day']:.2f}/day")

    # Count positive/negative assets
    pos_search = sum(1 for s in asset_search_stats.values() if s and s["sharpe"] > 0)
    neg_search = sum(1 for s in asset_search_stats.values() if s and s["sharpe"] <= 0)
    pos_holdout = sum(1 for s in asset_holdout_stats.values() if s and s["sharpe"] > 0)
    neg_holdout = sum(1 for s in asset_holdout_stats.values() if s and s["sharpe"] <= 0)

    print(f"\n  Search:  {pos_search}/{pos_search+neg_search} assets positive Sharpe")
    print(f"  Holdout: {pos_holdout}/{pos_holdout+neg_holdout} assets positive Sharpe")

    # ============================================================
    # Phase 4: Brownian noise null hypothesis (lb=8)
    # ============================================================
    print(f"\n{'='*70}")
    print("PHASE 4: BROWNIAN NOISE NULL HYPOTHESIS (lb=8)")
    print(f"{'='*70}")
    print("  Generating 30 synthetic GBM price series with BTC-like vol...")

    noise_signals = {}
    rng = np.random.RandomState(42)

    # Use BTC's timestamps and feature structure as template
    btc_sigs = all_signals.get("BTC", {})
    if btc_sigs and 8 in btc_sigs and len(btc_sigs[8]) > 0:
        # Get BTC's OHLCV structure as template
        data_dirs = [Path(p) for p in cfg.data.data_dirs] or [Path("data/binance-1m")]
        ohlcv_1m_btc = load_binance_1m("BTCUSDT", data_dirs)
        ohlcv_15m_btc = resample_ohlcv(ohlcv_1m_btc, "15min")
        n_bars = len(ohlcv_15m_btc)
        timestamps_btc = ohlcv_15m_btc.index

        # BTC-like parameters: ~60% annual vol, ~0% drift
        dt = 15 / (365.25 * 24 * 60)  # 15min in years
        annual_vol = 0.60
        sigma = annual_vol * np.sqrt(dt)

        for i in range(30):
            asset_name = f"NOISE_{i:02d}"
            # Generate GBM close prices
            log_rets = rng.normal(0, sigma, n_bars)
            log_prices = np.cumsum(log_rets) + np.log(40000)  # start near BTC price
            close_synth = np.exp(log_prices)

            # Build synthetic OHLCV
            noise_high = close_synth * (1 + np.abs(rng.normal(0, sigma * 0.5, n_bars)))
            noise_low = close_synth * (1 - np.abs(rng.normal(0, sigma * 0.5, n_bars)))
            noise_open = close_synth * (1 + rng.normal(0, sigma * 0.3, n_bars))
            noise_vol = np.abs(rng.normal(1000, 500, n_bars))

            ohlcv_synth = pd.DataFrame({
                "open": noise_open, "high": noise_high,
                "low": noise_low, "close": close_synth,
                "volume": noise_vol,
            }, index=timestamps_btc)

            # Compute features on synthetic data
            base_feat = compute_features(ohlcv_synth, cfg.features)
            # Synthetic regime features: compute on synthetic 1h/4h resampled
            regime_parts_s = []
            for rtf in cfg.close_model.regime_timeframes:
                rtf_ohlcv = ohlcv_synth.resample(rtf).agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"
                }).dropna()
                rtf_feat = compute_features(rtf_ohlcv, cfg.features)
                aligned = rtf_feat.shift(1).reindex(ohlcv_synth.index, method="ffill")
                aligned = aligned.add_prefix(f"signal_regime_{rtf}_")
                regime_parts_s.append(aligned)
            all_feat_s = pd.concat([base_feat] + regime_parts_s, axis=1)

            vol_s = ohlcv_synth["close"].pct_change().ewm(span=50, min_periods=20).std().values.astype(np.float64)
            close_s = close_synth.astype(np.float64)

            # Run vectorized walk-forward (inline, single lookback)
            feat_arr = all_feat_s.values.astype(np.float32)
            feat_cols_s = list(all_feat_s.columns)
            fwd_ret_s = np.full(n_bars, np.nan, dtype=np.float64)
            valid_s = np.arange(n_bars - FWD_BARS)
            mask_s = close_s[valid_s] > 0
            fwd_ret_s[valid_s[mask_s]] = (close_s[valid_s[mask_s] + FWD_BARS] - close_s[valid_s[mask_s]]) / close_s[valid_s[mask_s]]
            labels_s = np.full(n_bars, -1, dtype=np.int8)
            fin_s = np.isfinite(fwd_ret_s)
            labels_s[fin_s] = (fwd_ret_s[fin_s] > 0).astype(np.int8)

            ts_pd = pd.DatetimeIndex(timestamps_btc)
            mk_s = (ts_pd.year * 12 + ts_pd.month).values
            unique_mk = np.unique(mk_s)

            all_month_t = []
            lb = 8
            for mk in unique_mk:
                test_mask = mk_s == mk
                test_idx = np.where(test_mask)[0]
                if len(test_idx) < 10:
                    continue
                tr_mask = (mk_s >= mk - lb) & (mk_s <= mk - 1)
                tr_idx = np.where(tr_mask)[0]
                if len(tr_idx) < 200:
                    continue
                embargo = FWD_BARS + 1
                safe_tr = tr_idx[:-embargo] if len(tr_idx) > embargo else tr_idx
                valid_tr = safe_tr[labels_s[safe_tr] >= 0]
                if len(valid_tr) < 200:
                    continue

                tr_X = feat_arr[valid_tr].copy()
                tr_y = labels_s[valid_tr]
                wb = {}
                for cn in WINSOR_COLS:
                    if cn in feat_cols_s:
                        ci = feat_cols_s.index(cn)
                        lo, hi = np.nanquantile(tr_X[:, ci], [0.01, 0.99])
                        tr_X[:, ci] = np.clip(tr_X[:, ci], lo, hi)
                        wb[ci] = (lo, hi)
                tr_sw = compute_sample_weights(np.arange(len(valid_tr), dtype=np.int64), fwd_bars=FWD_BARS)

                tr_months = mk_s[valid_tr]
                val_mk = tr_months.max()
                vm = tr_months == val_mk
                fm = ~vm
                use_v = fm.sum() >= 200 and vm.sum() >= 50
                if not use_v:
                    fm[:] = True; vm[:] = False

                model = xgb.XGBClassifier(
                    n_estimators=250, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                    reg_alpha=0.1, reg_lambda=1.0, objective="binary:logistic",
                    eval_metric="logloss", tree_method="hist", random_state=42,
                    n_jobs=XGB_NJOBS, verbosity=0,
                    early_stopping_rounds=25 if use_v else None,
                )
                kw = {"sample_weight": tr_sw[fm], "verbose": False}
                if use_v:
                    kw["eval_set"] = [(tr_X[vm], tr_y[vm])]
                model.fit(tr_X[fm], tr_y[fm], **kw)

                vol_w = tr_idx[-min(len(tr_idx), 3*30*24*4):]
                vv = vol_s[vol_w]; vv = vv[np.isfinite(vv)]
                vt = np.quantile(vv, VOL_FILTER_QUANTILE) if len(vv) > 50 else 0.0

                te_X = feat_arr[test_idx].copy()
                for ci, (lo, hi) in wb.items():
                    te_X[:, ci] = np.clip(te_X[:, ci], lo, hi)
                fin_te = np.isfinite(te_X).all(axis=1)
                vp = vol_s[test_idx] >= vt
                fv = np.isfinite(fwd_ret_s[test_idx])
                pm = fin_te & vp & fv
                if pm.sum() == 0:
                    continue

                pi = test_idx[pm]
                probs = model.predict_proba(te_X[pm])[:, 1]
                dirs = np.where(probs > 0.5, 1.0, -1.0)
                frs = fwd_ret_s[pi]
                pnls = dirs * frs - SLIPPAGE

                mdf = pd.DataFrame({
                    "timestamp": timestamps_btc[pi], "direction": dirs,
                    "fwd_return": frs, "pnl": pnls, "confidence": np.abs(probs - 0.5),
                    "probability": probs, "entry_price": close_s[pi], "asset": asset_name,
                })
                all_month_t.append(mdf)

            if all_month_t:
                noise_signals[asset_name] = {lb: pd.concat(all_month_t, ignore_index=True)}
            print(f"  {asset_name} done: {sum(len(d) for d in all_month_t)} trades", flush=True)

        # Aggregate noise results
        noise_parts = [sigs[8] for sigs in noise_signals.values() if 8 in sigs and len(sigs[8]) > 0]
        if noise_parts:
            noise_combined = pd.concat(noise_parts, ignore_index=True)
            noise_search = compute_period_stats(noise_combined, SEARCH_START, SEARCH_END)
            noise_full = compute_period_stats(noise_combined)

            print(f"\n  BROWNIAN NOISE RESULTS (30 synthetic assets, lb=8):")
            if noise_search:
                print(f"    Search: Sharpe={noise_search['sharpe']:.3f}  Win={noise_search['win_rate']:.1f}%  "
                      f"bps={noise_search['mean_bps']:.1f}  Trades={noise_search['n_trades']}")
            if noise_full:
                print(f"    Full:   Sharpe={noise_full['sharpe']:.3f}  Win={noise_full['win_rate']:.1f}%  "
                      f"bps={noise_full['mean_bps']:.1f}  Trades={noise_full['n_trades']}")

            # Compare to real
            real_s = next((r for lb, r in search_rows if lb == 8 and r), None)
            if real_s and noise_search:
                print(f"\n    Real vs Noise:")
                print(f"      Real:  Sharpe={real_s['sharpe']:.3f}  Win={real_s['win_rate']:.1f}%  bps={real_s['mean_bps']:.1f}")
                print(f"      Noise: Sharpe={noise_search['sharpe']:.3f}  Win={noise_search['win_rate']:.1f}%  bps={noise_search['mean_bps']:.1f}")
    else:
        print("  SKIP: BTC data not available for template")

    print(f"\n  Total runtime: {time.time()-t_global:.0f}s")


if __name__ == "__main__":
    main()
