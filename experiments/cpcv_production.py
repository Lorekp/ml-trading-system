"""Tier 0C: CPCV on production config (8mo, 13 features, Kelly).

Architecture: generate ALL signals in parallel (like lookback_fair),
then split into CPCV folds. Signal generation is the bottleneck —
fully parallelized across assets with all CPU cores.
"""
import sys
sys.path.insert(0, "src")

import copy, os, time, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from signal_engine.config import load_config, ConfidenceFilterConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from afml.bet_sizing import get_signal, discretize_signal, kelly_size

import sklearn
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)

W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
LOOKBACK = 8
N_FOLDS = 6

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


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def _process_asset(args):
    """Generate signals for ALL months for one asset. Returns {month_str: sig_dict}."""
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    from signal_engine.config import load_config, ConfidenceFilterConfig
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights
    from signal_engine.filters import apply_confidence_filter
    from afml.bet_sizing import get_signal, discretize_signal

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [Path("data/binance-1m")])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, []

    features = compute_features(ohlcv_15m, cfg.features)
    feat = features[cfg.features.feature_list].copy()
    close = ohlcv_15m["close"]
    fwd_ret = close.pct_change(4).shift(-4)
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in feat.columns:
            feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))

    fc = feat.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)
    if len(valid) < 3000:
        return asset, []

    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    results = []
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

        train_start = ms - pd.DateOffset(months=LOOKBACK)
        if train_start < start_date:
            continue
        train_cutoff = ms - embargo_td
        tr = valid[(valid >= train_start) & (valid < train_cutoff)]
        if len(tr) < 200:
            continue

        try:
            sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(fc.loc[tr])
            model = make_logreg()
            model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
            Xte = scaler.transform(fc.loc[te])
            yp = model.predict_proba(Xte)[:, 1]
        except Exception:
            continue

        conf = np.abs(yp - 0.5)
        dirs = np.where(yp > 0.5, 1.0, -1.0)
        sigs = get_signal(yp, num_classes=2)
        sd = discretize_signal(sigs, step_size=0.1)
        ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
        fv = fwd_ret.reindex(te).values
        finite = np.isfinite(fv)
        if not finite.any():
            continue

        signals = pd.DataFrame({
            "timestamp": te[finite], "direction": dirs[finite],
            "fwd_return": fv[finite], "confidence": conf[finite],
            "probability": yp[finite], "signal_strength": ss[finite],
        })
        signals["asset"] = asset
        filtered = apply_confidence_filter(
            signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
        if len(filtered) > 0:
            results.append((str(month), filtered.to_dict("list")))

    return asset, results


def apply_kelly(combined):
    """Walk-forward Kelly."""
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(combined["timestamp"])
    months = sorted(ts.dt.to_period("M").unique())
    parts = []
    for km in months:
        km_sigs = combined[ts.dt.to_period("M") == km]
        past_sigs = combined[ts.dt.to_period("M") < km]
        if len(past_sigs) < 50:
            parts.append(km_sigs)
            continue
        kw = {}
        for an in km_sigs["asset"].unique():
            pa = past_sigs[past_sigs["asset"] == an]
            if len(pa) < 10:
                kw[an] = 1.0
                continue
            tr = pa["direction"].values * pa["fwd_return"].values
            w = tr[tr > 0]; lo = tr[tr <= 0]
            if len(w) < 3 or len(lo) < 3:
                kw[an] = 1.0
                continue
            wp, wr, lr = len(w)/len(tr), float(w.mean()), float(abs(lo.mean()))
            if lr <= 0:
                kw[an] = 1.0
                continue
            kw[an] = max(0.2, min(3.0, kelly_size(wp, wr, lr) * 7.5))
        if kw:
            mw = np.mean(list(kw.values()))
            if mw > 0:
                kw = {k: v/mw for k, v in kw.items()}
        kmc = km_sigs.copy()
        kmc["signal_strength"] = kmc.apply(
            lambda r: r["signal_strength"] * kw.get(r["asset"], 1.0), axis=1)
        parts.append(kmc)
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def run_fold(test_months_set, all_results):
    """Run portfolio sim on signals from test_months only, with Kelly from all past."""
    sigs = []
    for asset_results in all_results.values():
        for month_str, sig_dict in asset_results:
            if month_str in test_months_set:
                sigs.append(pd.DataFrame(sig_dict))
    if not sigs:
        return None

    combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
    if len(combined) < 50:
        return None

    combined = apply_kelly(combined)

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
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"], sim["n_trades"]/len(dpnl))
    return stats["sharpe"]


def main():
    print("=" * 70)
    print("TIER 0C: CPCV ON PRODUCTION CONFIG")
    print(f"  {N_FOLDS}-fold temporal CPCV, {W} workers")
    print(f"  8-month lookback, 13 features, 30 assets, Kelly")
    print("=" * 70, flush=True)

    t0 = time.time()

    # Phase 1: Generate ALL signals (parallel across 30 assets)
    print(f"\n[1] Generating signals for all months ({W} workers)...", flush=True)
    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_process_asset, (a, s)): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, results = f.result()
            done += 1
            all_results[asset] = results
            n_months = len(results)
            print(f"  [{done:>2d}/30] {asset}: {n_months} months", flush=True)

    # Collect all unique months with signals
    all_months_set = set()
    for asset_results in all_results.values():
        for month_str, _ in asset_results:
            all_months_set.add(month_str)
    all_months_sorted = sorted(all_months_set)
    print(f"  Total: {len(all_months_sorted)} unique months with signals ({time.time()-t0:.0f}s)", flush=True)

    # Phase 2: Split into N_FOLDS temporal folds
    fold_size = len(all_months_sorted) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        start = i * fold_size
        end = start + fold_size if i < N_FOLDS - 1 else len(all_months_sorted)
        folds.append(set(all_months_sorted[start:end]))

    print(f"\n[2] {N_FOLDS} folds:")
    for i, fold in enumerate(folds):
        s = sorted(fold)
        print(f"  Fold {i}: {s[0]} to {s[-1]} ({len(fold)} months)")

    # Phase 3: Run CPCV — each fold is test, rest is available for Kelly
    print(f"\n[3] Running {N_FOLDS} CPCV paths...", flush=True)
    sharpes = []
    for test_idx in range(N_FOLDS):
        test_months = folds[test_idx]
        sharpe = run_fold(test_months, all_results)
        if sharpe is not None:
            sharpes.append(sharpe)
            print(f"  Path {test_idx} (test fold {test_idx}): Sharpe = {sharpe:.3f}", flush=True)
        else:
            print(f"  Path {test_idx}: NO RESULT", flush=True)

    if not sharpes:
        print("ERROR: No valid paths!")
        return

    # Phase 4: Bootstrap 95% CI
    print(f"\n[4] Bootstrap CI (5000 resamples)...", flush=True)
    rng = np.random.RandomState(42)
    boot_means = [np.mean(rng.choice(sharpes, size=len(sharpes), replace=True)) for _ in range(5000)]
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)

    # Also: full pipeline run (all months, no fold restriction)
    print(f"\n[5] Full pipeline (all months, for reference)...", flush=True)
    all_month_set = set(all_months_sorted)
    full_sharpe = run_fold(all_month_set, all_results)

    # Results
    print(f"\n{'='*70}")
    print("CPCV RESULTS")
    print(f"{'='*70}")
    print(f"  Paths tested:      {len(sharpes)}/{N_FOLDS}")
    print(f"  Sharpe mean:       {np.mean(sharpes):.3f}")
    print(f"  Sharpe median:     {np.median(sharpes):.3f}")
    print(f"  Sharpe std:        {np.std(sharpes):.3f}")
    print(f"  Sharpe min:        {np.min(sharpes):.3f}")
    print(f"  Sharpe max:        {np.max(sharpes):.3f}")
    print(f"  % positive:        {100 * sum(1 for s in sharpes if s > 0) / len(sharpes):.1f}%")
    print(f"  95% CI:            [{ci_lo:.3f}, {ci_hi:.3f}]")
    if full_sharpe:
        print(f"  Full pipeline:     {full_sharpe:.3f}")
    print(f"\n  Elapsed: {time.time()-t0:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
