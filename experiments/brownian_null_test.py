"""Brownian noise null hypothesis test.

Generates 30 synthetic GBM price series with BTC-like volatility,
runs the same walk-forward XGBoost pipeline (lb=8, 15min, regime features),
and reports Sharpe. If the algorithm produces high Sharpe on pure noise,
the signal is spurious.
"""

import sys
sys.path.insert(0, "src")

import logging
import os
import time
import warnings
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

FWD_BARS = 4
SLIPPAGE = 0.0001
WINSOR_COLS = ["ret_1", "ret_3", "ret_7", "ret_14"]
VOL_FILTER_QUANTILE = 0.50
LB = 8
N_SYNTHETIC = 3
XGB_NJOBS = max(1, os.cpu_count() or 4)


def run_walkforward_on_ohlcv(ohlcv, timestamps, cfg, asset_name):
    """Run vectorized walk-forward on a single OHLCV DataFrame."""
    base_feat = compute_features(ohlcv, cfg.features)
    regime_parts = []
    for rtf in cfg.close_model.regime_timeframes:
        rtf_ohlcv = ohlcv.resample(rtf).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()
        if len(rtf_ohlcv) < 100:
            continue
        rtf_feat = compute_features(rtf_ohlcv, cfg.features)
        # CRITICAL: shift by 1 period so features only use COMPLETED bars (no future leakage)
        aligned = rtf_feat.shift(1).reindex(ohlcv.index, method="ffill")
        aligned = aligned.add_prefix(f"signal_regime_{rtf}_")
        regime_parts.append(aligned)

    all_feat = pd.concat([base_feat] + regime_parts, axis=1)
    feat_arr = all_feat.values.astype(np.float32)
    feat_cols = list(all_feat.columns)
    close = ohlcv["close"].values.astype(np.float64)
    vol = ohlcv["close"].pct_change().ewm(span=50, min_periods=20).std().values.astype(np.float64)
    n = len(close)

    fwd_ret = np.full(n, np.nan, dtype=np.float64)
    valid = np.arange(n - FWD_BARS)
    mask = close[valid] > 0
    fwd_ret[valid[mask]] = (close[valid[mask] + FWD_BARS] - close[valid[mask]]) / close[valid[mask]]

    labels = np.full(n, -1, dtype=np.int8)
    fin = np.isfinite(fwd_ret)
    labels[fin] = (fwd_ret[fin] > 0).astype(np.int8)

    ts_pd = pd.DatetimeIndex(timestamps)
    mk = (ts_pd.year * 12 + ts_pd.month).values
    unique_mk = np.unique(mk)

    all_trades = []
    for month in unique_mk:
        test_idx = np.where(mk == month)[0]
        if len(test_idx) < 10:
            continue
        tr_mask = (mk >= month - LB) & (mk <= month - 1)
        tr_idx = np.where(tr_mask)[0]
        if len(tr_idx) < 200:
            continue
        embargo = FWD_BARS + 1
        safe_tr = tr_idx[:-embargo] if len(tr_idx) > embargo else tr_idx
        valid_tr = safe_tr[labels[safe_tr] >= 0]
        if len(valid_tr) < 200:
            continue

        tr_X = feat_arr[valid_tr].copy()
        tr_y = labels[valid_tr]
        wb = {}
        for cn in WINSOR_COLS:
            if cn in feat_cols:
                ci = feat_cols.index(cn)
                lo, hi = np.nanquantile(tr_X[:, ci], [0.01, 0.99])
                tr_X[:, ci] = np.clip(tr_X[:, ci], lo, hi)
                wb[ci] = (lo, hi)
        tr_sw = compute_sample_weights(np.arange(len(valid_tr), dtype=np.int64), fwd_bars=FWD_BARS)

        tr_months = mk[valid_tr]
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
        vv = vol[vol_w]; vv = vv[np.isfinite(vv)]
        vt = np.quantile(vv, VOL_FILTER_QUANTILE) if len(vv) > 50 else 0.0

        te_X = feat_arr[test_idx].copy()
        for ci, (lo, hi) in wb.items():
            te_X[:, ci] = np.clip(te_X[:, ci], lo, hi)
        fin_te = np.isfinite(te_X).all(axis=1)
        vp = vol[test_idx] >= vt
        fv = np.isfinite(fwd_ret[test_idx])
        pm = fin_te & vp & fv
        if pm.sum() == 0:
            continue

        pi = test_idx[pm]
        probs = model.predict_proba(te_X[pm])[:, 1]
        dirs = np.where(probs > 0.5, 1.0, -1.0)
        frs = fwd_ret[pi]
        pnls = dirs * frs - SLIPPAGE

        mdf = pd.DataFrame({
            "timestamp": timestamps[pi], "direction": dirs,
            "fwd_return": frs, "pnl": pnls, "asset": asset_name,
        })
        all_trades.append(mdf)

    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


def compute_stats(df):
    if df is None or len(df) < 10:
        return None
    pnls = df["pnl"].values
    ts = pd.to_datetime(df["timestamp"], utc=True)
    daily = pd.Series(5.0 * pnls, index=ts.values).resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 5:
        return None
    std = daily.std()
    sharpe = float(daily.mean() / std * np.sqrt(365.25)) if std > 0 else 0.0
    return {
        "sharpe": sharpe,
        "win_rate": float((pnls > 0).mean()) * 100,
        "mean_bps": float(pnls.mean() * 10000),
        "n_trades": len(df),
    }


def main():
    t0 = time.time()
    cfg = load_config(Path("configs/default.yaml"))
    data_dirs = [Path(p) for p in cfg.data.data_dirs] or [Path("data/binance-1m")]

    print("Loading BTC as template...")
    ohlcv_1m = load_binance_1m("BTCUSDT", data_dirs)
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    n_bars = len(ohlcv_15m)
    timestamps = ohlcv_15m.index

    # Run REAL BTC first as reference
    print(f"\nRunning REAL BTC (lb={LB}, {n_bars} bars)...")
    real_df = run_walkforward_on_ohlcv(ohlcv_15m, timestamps, cfg, "BTC_REAL")
    real_stats = compute_stats(real_df)
    if real_stats:
        print(f"  REAL BTC: Sharpe={real_stats['sharpe']:.3f}  Win={real_stats['win_rate']:.1f}%  "
              f"bps={real_stats['mean_bps']:.1f}  n={real_stats['n_trades']}")

    # Generate and test synthetic series
    print(f"\nGenerating {N_SYNTHETIC} Brownian motion series...")
    rng = np.random.RandomState(42)
    dt = 15 / (365.25 * 24 * 60)
    annual_vol = 0.60
    sigma = annual_vol * np.sqrt(dt)

    noise_results = []
    all_noise_trades = []

    for i in range(N_SYNTHETIC):
        log_rets = rng.normal(0, sigma, n_bars)
        log_prices = np.cumsum(log_rets) + np.log(40000)
        close_synth = np.exp(log_prices)

        noise_high = close_synth * (1 + np.abs(rng.normal(0, sigma * 0.5, n_bars)))
        noise_low = close_synth * (1 - np.abs(rng.normal(0, sigma * 0.5, n_bars)))
        noise_open = close_synth * (1 + rng.normal(0, sigma * 0.3, n_bars))
        noise_vol = np.abs(rng.normal(1000, 500, n_bars))

        ohlcv_synth = pd.DataFrame({
            "open": noise_open, "high": np.maximum(noise_high, np.maximum(noise_open, close_synth)),
            "low": np.minimum(noise_low, np.minimum(noise_open, close_synth)),
            "close": close_synth, "volume": noise_vol,
        }, index=timestamps)

        asset_name = f"NOISE_{i:02d}"
        df = run_walkforward_on_ohlcv(ohlcv_synth, timestamps, cfg, asset_name)
        stats = compute_stats(df)
        if stats:
            noise_results.append(stats)
            print(f"  {asset_name}: Sharpe={stats['sharpe']:>7.3f}  Win={stats['win_rate']:.1f}%  "
                  f"bps={stats['mean_bps']:>5.1f}  n={stats['n_trades']}", flush=True)
        if len(df) > 0:
            all_noise_trades.append(df)

    # Aggregate noise
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    if real_stats:
        print(f"  REAL BTC:       Sharpe={real_stats['sharpe']:>7.3f}  Win={real_stats['win_rate']:.1f}%  "
              f"bps={real_stats['mean_bps']:>5.1f}  n={real_stats['n_trades']}")

    if noise_results:
        sharpes = [r["sharpe"] for r in noise_results]
        wins = [r["win_rate"] for r in noise_results]
        bps_vals = [r["mean_bps"] for r in noise_results]
        print(f"\n  BROWNIAN NOISE ({len(noise_results)} series):")
        print(f"    Sharpe: mean={np.mean(sharpes):.3f}  std={np.std(sharpes):.3f}  "
              f"min={np.min(sharpes):.3f}  max={np.max(sharpes):.3f}")
        print(f"    Win%:   mean={np.mean(wins):.1f}%  std={np.std(wins):.1f}%")
        print(f"    bps:    mean={np.mean(bps_vals):.1f}  std={np.std(bps_vals):.1f}")
        print(f"    Positive Sharpe: {sum(1 for s in sharpes if s > 0)}/{len(sharpes)}")

    # Combined noise portfolio
    if all_noise_trades:
        combined = pd.concat(all_noise_trades, ignore_index=True)
        combined_stats = compute_stats(combined)
        if combined_stats:
            print(f"\n  NOISE PORTFOLIO (30 combined):")
            print(f"    Sharpe={combined_stats['sharpe']:.3f}  Win={combined_stats['win_rate']:.1f}%  "
                  f"bps={combined_stats['mean_bps']:.1f}  n={combined_stats['n_trades']}")

    if real_stats and noise_results:
        z_score = (real_stats["sharpe"] - np.mean(sharpes)) / max(np.std(sharpes), 0.01)
        print(f"\n  Z-score (real vs noise): {z_score:.2f}")
        print(f"  {'PASS' if z_score > 3 else 'FAIL'}: real signal is {'significantly' if z_score > 3 else 'NOT significantly'} "
              f"different from noise (threshold: Z > 3)")

    print(f"\n  Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
