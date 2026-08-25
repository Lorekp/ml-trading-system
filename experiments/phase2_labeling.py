"""Phase 2: Triple-Barrier + CUSUM Labeling Pipeline.

Replaces fixed-horizon labels with AFML-recommended triple-barrier + CUSUM.
Uses existing implementations from src/afml/.

Chunks 2.1-2.5: Triple-barrier, CUSUM, return-attributed weights, integration test.
"""

import sys
sys.path.insert(0, "src")

import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_engine.config import load_config, ConfidenceFilterConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.allocation import compute_nco_weights, compute_kelly_weights
from afml.labeling import get_events, get_bins, add_vertical_barrier
from afml.filters import cusum_filter, get_daily_vol
from afml.sampling import (
    get_num_concurrent_events, get_avg_uniqueness,
    get_sample_weights, get_time_decay,
)
from afml.bet_sizing import get_signal, discretize_signal

warnings.filterwarnings("ignore")

NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")

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

# Triple-barrier params for 15min bars
VERTICAL_BARS = 16       # 4 hours vertical barrier
PT_SL = [1.0, 1.0]       # Symmetric profit-take / stop-loss
VOL_SPAN = 100            # EWM span for dynamic threshold
CUSUM_MULTIPLIER = 1.5    # CUSUM threshold = multiplier * daily_vol (on log prices)
TIME_DECAY_LAST_W = 0.5   # Oldest sample = 50% of newest


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def process_asset(args):
    """Generate triple-barrier + CUSUM signals for one asset."""
    asset, symbol, mode = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    from signal_engine.config import load_config, ConfidenceFilterConfig
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.filters import apply_confidence_filter
    from afml.labeling import get_events, get_bins
    from afml.filters import cusum_filter, get_daily_vol
    from afml.sampling import (
        get_num_concurrent_events, get_avg_uniqueness,
        get_sample_weights, get_time_decay,
    )
    from afml.bet_sizing import get_signal, discretize_signal

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, {}

    close = ohlcv_15m["close"]
    features = compute_features(ohlcv_15m, cfg.features)
    feat = features[cfg.features.feature_list].copy()

    # Winsorize (global — matches baseline)
    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in feat.columns:
            feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))

    fc = feat.dropna()
    if len(fc) < 3000:
        return asset, {}

    # Vol for filtering
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    # Dynamic volatility target for triple-barrier
    daily_vol = get_daily_vol(close, span=VOL_SPAN)

    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    results_baseline = {8: []}
    results_tb = {8: []}
    results_cusum_fixedlabel = {8: []}  # CUSUM events + improved weights, but FIXED-HORIZON labels

    all_months = pd.period_range(
        start=(start_date + pd.DateOffset(months=2)).to_period("M"),
        end=end_date.to_period("M"), freq="M",
    )

    for month in all_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

        # Test bars (this month, vol-filtered)
        valid = fc.index.intersection(close.index)
        te = valid[(valid >= ms) & (valid <= me)]
        vol_train_start = ms - pd.DateOffset(months=3)
        vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms)].dropna()
        if len(vol_train) < 50:
            continue
        vt = vol_train.quantile(0.50)
        test_vol = vol.reindex(te)
        hv = (test_vol > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5:
            continue

        month_str = str(month)
        lb = 8
        train_start = ms - pd.DateOffset(months=lb)
        if train_start < start_date:
            continue
        train_cutoff = ms - embargo_td
        tr = valid[(valid >= train_start) & (valid < train_cutoff)]
        if len(tr) < 200:
            continue

        # ============================================================
        # MODE A: Baseline (fixed-horizon labels) — for comparison
        # ============================================================
        if mode in ("baseline", "both"):
            fwd_ret = close.pct_change(4).shift(-4)
            y_all_base = fwd_ret.reindex(fc.index).dropna()
            tr_base = tr.intersection(y_all_base.index)
            if len(tr_base) >= 200:
                try:
                    from signal_engine.weights import compute_sample_weights
                    sw = compute_sample_weights(fc.loc[tr_base].index.values.astype(np.int64), fwd_bars=4)
                    scaler = StandardScaler()
                    Xtr = scaler.fit_transform(fc.loc[tr_base])
                    model = make_logreg()
                    model.fit(Xtr, (y_all_base.loc[tr_base] > 0).astype(int), sample_weight=sw)
                    Xte = scaler.transform(fc.loc[te])
                    yp = model.predict_proba(Xte)[:, 1]
                    conf = np.abs(yp - 0.5)
                    dirs = np.where(yp > 0.5, 1.0, -1.0)
                    sigs = get_signal(yp, num_classes=2)
                    sd = discretize_signal(sigs, step_size=0.1)
                    ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
                    fv = fwd_ret.reindex(te).values
                    finite = np.isfinite(fv)
                    if finite.any():
                        signals = pd.DataFrame({
                            "timestamp": te[finite], "direction": dirs[finite],
                            "fwd_return": fv[finite], "confidence": conf[finite],
                            "probability": yp[finite], "signal_strength": ss[finite],
                        })
                        signals["asset"] = asset
                        filtered = apply_confidence_filter(
                            signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
                        if len(filtered) > 0:
                            results_baseline[8].append((month_str, filtered.to_dict("list")))
                except Exception:
                    pass

        # ============================================================
        # MODE B: Triple-barrier + CUSUM labels
        # ============================================================
        if mode in ("tb", "both"):
            try:
                # Get training close for CUSUM + triple-barrier
                train_close = close.loc[train_start:train_cutoff]
                if len(train_close) < 200:
                    continue

                # CUSUM events on LOG prices (so diffs are returns, matching vol scale)
                train_log = np.log(train_close)
                train_vol = daily_vol.reindex(train_close.index).dropna()
                if len(train_vol) < 50:
                    continue
                cusum_h = train_vol * CUSUM_MULTIPLIER
                t_events = cusum_filter(train_log, cusum_h)
                t_events = t_events[t_events.isin(valid)]  # must have features
                if len(t_events) < 50:
                    # Fallback: use all training bars if CUSUM too restrictive
                    t_events = pd.DatetimeIndex(tr)

                # Triple-barrier labeling
                target = daily_vol.reindex(t_events).dropna()
                t_events = target.index
                if len(t_events) < 50:
                    continue

                events = get_events(
                    close, t_events, PT_SL, target,
                    min_ret=0.0, num_bars_vertical=VERTICAL_BARS,
                )
                if len(events) < 50:
                    continue

                # Drop 'side' column for standard labeling (not meta-labeling)
                events_std = events.drop(columns=["side"], errors="ignore")
                bins = get_bins(events_std, close)
                # Standard labeling: keep -1, +1 (drop zeros if any)
                valid_labels = bins.dropna(subset=["bin"])
                valid_labels = valid_labels[valid_labels["bin"] != 0]
                if len(valid_labels) < 50:
                    continue

                # Return-attributed sample weights + time decay
                t1 = events.loc[valid_labels.index, "t1"]
                num_conc = get_num_concurrent_events(close.index, t1)
                sw_ret = get_sample_weights(t1, num_conc, close)
                avg_u = get_avg_uniqueness(t1, num_conc)
                sw_decay = get_time_decay(avg_u, clf_last_w=TIME_DECAY_LAST_W)

                # Combine: return-attributed * time-decay
                sw_combined = sw_ret * sw_decay.reindex(sw_ret.index).fillna(1.0)
                sw_combined = sw_combined / sw_combined.mean()  # normalize

                # Align features with labels
                train_feat_idx = valid_labels.index.intersection(fc.index)
                if len(train_feat_idx) < 50:
                    continue

                y_train = valid_labels.loc[train_feat_idx, "bin"]
                y_binary = (y_train > 0).astype(int)  # convert {-1, +1} → {0, 1}
                sw_final = sw_combined.reindex(train_feat_idx).fillna(1.0).values

                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[train_feat_idx])
                model = make_logreg()
                model.fit(Xtr, y_binary, sample_weight=sw_final)

                # Predict on test
                Xte = scaler.transform(fc.loc[te])
                yp = model.predict_proba(Xte)[:, 1]

                conf = np.abs(yp - 0.5)
                dirs = np.where(yp > 0.5, 1.0, -1.0)
                sigs = get_signal(yp, num_classes=2)
                sd = discretize_signal(sigs, step_size=0.1)
                ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))

                # Use fixed-horizon fwd_return for fair comparison
                fwd_ret = close.pct_change(4).shift(-4)
                fv = fwd_ret.reindex(te).values
                finite = np.isfinite(fv)
                if finite.any():
                    signals = pd.DataFrame({
                        "timestamp": te[finite], "direction": dirs[finite],
                        "fwd_return": fv[finite], "confidence": conf[finite],
                        "probability": yp[finite], "signal_strength": ss[finite],
                    })
                    signals["asset"] = asset
                    filtered = apply_confidence_filter(
                        signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
                    if len(filtered) > 0:
                        results_tb[8].append((month_str, filtered.to_dict("list")))
            except Exception as e:
                pass

    return asset, {"baseline": results_baseline, "tb": results_tb}


def run_portfolio(all_results, key, start_after=None):
    """Aggregate signals, apply NCO, simulate portfolio."""
    sigs = []
    for ar in all_results.values():
        bucket = ar.get(key, {})
        for ms, sd in bucket.get(8, []):
            if start_after and ms < start_after:
                continue
            sigs.append(pd.DataFrame(sd))
    if not sigs:
        return None
    combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
    if len(combined) < 50:
        return None

    # NCO allocation (same as baseline) — import via importlib to avoid module path issue
    import importlib.util
    spec = importlib.util.spec_from_file_location("orig", "experiments/test_original_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    combined = mod.apply_nco_wf(combined)

    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.03,
        base_sizing_pct=0.04, max_signal_mult=2.0,
    )
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                               sim["n_trades"] / len(dpnl))
    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
    return {**stats, "n_trades": sim["n_trades"], "pnl_day": pnl_day,
            "combined": combined}


def flat_bet_stats(all_results, key, start=None, end=None):
    """Simple flat-bet stats."""
    sigs = []
    for ar in all_results.values():
        bucket = ar.get(key, {})
        for ms, sd in bucket.get(8, []):
            if start and ms < start:
                continue
            if end and ms > end:
                continue
            sigs.append(pd.DataFrame(sd))
    if not sigs:
        return None
    df = pd.concat(sigs)
    pnls = df["direction"].values * df["fwd_return"].values - 0.0001
    ts_arr = pd.to_datetime(df["timestamp"], utc=True)
    daily = pd.Series(5.0 * pnls, index=ts_arr.values).resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 5:
        return None
    std = daily.std()
    sr = daily.mean() / std * np.sqrt(365.25) if std > 0 else 0
    return {
        "sharpe": sr, "win_rate": (pnls > 0).mean() * 100,
        "mean_bps": pnls.mean() * 10000, "n_trades": len(df),
        "n_assets": df["asset"].nunique(),
    }


def main():
    t0 = time.time()

    # Allow BTC-only quick test
    selected = os.getenv("ML_SIGNAL_ASSETS", "").strip()
    if selected:
        asset_list = {k: v for k, v in ASSETS.items() if k in selected.split(",")}
    else:
        asset_list = ASSETS

    print("=" * 70)
    print("PHASE 2: TRIPLE-BARRIER + CUSUM LABELING")
    print(f"  {len(asset_list)} assets, {W} workers")
    print(f"  Triple-barrier: pt_sl={PT_SL}, vertical={VERTICAL_BARS} bars")
    print(f"  CUSUM: threshold={CUSUM_MULTIPLIER}x daily_vol")
    print(f"  Weights: return-attributed + time_decay(last_w={TIME_DECAY_LAST_W})")
    print("=" * 70, flush=True)

    # Generate signals for both baseline and triple-barrier
    print(f"\nGenerating signals ({W} workers)...", flush=True)
    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s, "both")): a for a, s in asset_list.items()}
        done = 0
        for f in as_completed(futures):
            asset, results = f.result()
            done += 1
            all_results[asset] = results
            n_base = sum(len(v) for v in results.get("baseline", {}).values())
            n_tb = sum(len(v) for v in results.get("tb", {}).values())
            print(f"  [{done:>2d}/{len(asset_list)}] {asset}: baseline={n_base} tb={n_tb}", flush=True)

    print(f"\n  Signals done in {time.time()-t0:.0f}s")

    # ============================================================
    # COMPARISON: FLAT-BET
    # ============================================================
    print(f"\n{'='*70}")
    print("FLAT-BET COMPARISON (lb=8, $5/trade)")
    print(f"{'='*70}")

    for period, start, end in [("SEARCH", "2022-01", "2024-12"), ("HOLDOUT", "2025-01", None), ("FULL", None, None)]:
        base = flat_bet_stats(all_results, "baseline", start, end)
        tb = flat_bet_stats(all_results, "tb", start, end)
        print(f"\n  {period}:")
        for name, r in [("Baseline (fixed-horizon)", base), ("Triple-barrier + CUSUM", tb)]:
            if r:
                print(f"    {name:<30s} SR={r['sharpe']:>6.3f}  Win={r['win_rate']:>5.1f}%  "
                      f"bps={r['mean_bps']:>5.2f}  n={r['n_trades']:>6d}  assets={r['n_assets']}")
            else:
                print(f"    {name:<30s} NO DATA")

    # ============================================================
    # PORTFOLIO COMPARISON (NCO)
    # ============================================================
    print(f"\n{'='*70}")
    print("NCO PORTFOLIO COMPARISON (lb=8)")
    print(f"{'='*70}")

    sample_1m = load_binance_1m("BTCUSDT", [DATA_DIR])
    sample_15m = resample_ohlcv(sample_1m, "15min")
    data_start = sample_15m.index.min()
    fixed_start = str((data_start + pd.DateOffset(months=9)).to_period("M"))

    for name, key in [("Baseline (fixed-horizon)", "baseline"), ("Triple-barrier + CUSUM", "tb")]:
        r = run_portfolio(all_results, key, fixed_start)
        if r:
            print(f"  {name:<30s} SR={r['sharpe']:>6.3f}  Ret={r['total_return_pct']:>8.1f}%  "
                  f"MaxDD={r['max_dd_pct']:>6.2f}%  PnL/day=${r['pnl_day']:>7.2f}  "
                  f"Trades={r['n_trades']}")
        else:
            print(f"  {name:<30s} NO DATA")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
