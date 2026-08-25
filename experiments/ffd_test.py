"""Tier 1A: Fractional Differentiation (AFML Ch5).

Tests FFD(d) of log-close as additional feature.
Finds minimum d for ADF stationarity per asset, then adds ffd_close
to the existing 13 features.

Fully parallelized across 30 assets.
"""
import sys
sys.path.insert(0, "src")

import copy, os, time, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

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
FIXED_START = "2023-08"

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


def ffd_weights(d, threshold=1e-5):
    """Compute FFD weights for fractional differentiation of order d."""
    w = [1.0]
    k = 1
    while True:
        w_new = -w[-1] * (d - k + 1) / k
        if abs(w_new) < threshold:
            break
        w.append(w_new)
        k += 1
    return np.array(w[::-1])


def ffd(series, d):
    """Apply fixed-width fractional differentiation of order d to a series."""
    w = ffd_weights(d)
    width = len(w)
    result = np.full(len(series), np.nan)
    vals = series.values
    for i in range(width - 1, len(vals)):
        result[i] = np.dot(w, vals[i - width + 1:i + 1])
    return pd.Series(result, index=series.index)


def find_min_d(log_close, d_range=np.arange(0.1, 1.01, 0.1), p_threshold=0.05):
    """Find minimum d such that ADF test rejects non-stationarity."""
    for d in d_range:
        ffd_series = ffd(log_close, d).dropna()
        if len(ffd_series) < 100:
            continue
        try:
            adf_stat, p_value, *_ = adfuller(ffd_series, maxlag=1)
            if p_value < p_threshold:
                return round(float(d), 2), float(p_value)
        except Exception:
            continue
    return 1.0, 1.0  # fallback to standard differentiation


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def _process_asset(args):
    """Generate signals with and without FFD feature for one asset."""
    asset, symbol, use_ffd = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    from signal_engine.config import load_config, ConfidenceFilterConfig
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights
    from signal_engine.filters import apply_confidence_filter
    from afml.bet_sizing import get_signal, discretize_signal
    from statsmodels.tsa.stattools import adfuller

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [Path("data/binance-1m")])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, [], None

    features = compute_features(ohlcv_15m, cfg.features)
    feat = features[cfg.features.feature_list].copy()

    # Add FFD feature if requested
    optimal_d = None
    if use_ffd:
        log_close = np.log(ohlcv_15m["close"])
        d_vals = np.arange(0.1, 1.01, 0.1)
        optimal_d, p_val = find_min_d(log_close, d_vals)
        ffd_series = ffd(log_close, optimal_d)
        feat["ffd_close"] = ffd_series.reindex(feat.index).astype(np.float32)

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
        return asset, [], optimal_d

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

    return asset, results, optimal_d


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


def run_test(label, all_results):
    """Aggregate, Kelly, portfolio sim."""
    sigs = []
    for asset_results in all_results.values():
        for month_str, sig_dict in asset_results:
            if month_str < FIXED_START:
                continue
            sigs.append(pd.DataFrame(sig_dict))
    if not sigs:
        return None
    combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
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
    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
    return {"sharpe": stats["sharpe"], "return_pct": stats["total_return_pct"],
            "max_dd_pct": stats["max_dd_pct"], "n_trades": sim["n_trades"], "pnl_day": pnl_day}


def main():
    print("=" * 75)
    print("TIER 1A: FRACTIONAL DIFFERENTIATION (AFML Ch5)")
    print(f"  8mo lookback, 30 assets, Kelly, {W} workers")
    print(f"  Tests: 13 features (baseline) vs 14 features (13 + ffd_close)")
    print("=" * 75, flush=True)

    t0 = time.time()

    # Phase 1: Baseline (no FFD) — parallel
    print(f"\n[1] Generating baseline signals (13 features)...", flush=True)
    baseline_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_process_asset, (a, s, False)): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, results, _ = f.result()
            done += 1
            baseline_results[asset] = results
            print(f"  [{done:>2d}/30] {asset}: {len(results)} months", flush=True)

    # Phase 2: With FFD — parallel
    print(f"\n[2] Generating FFD signals (13 + ffd_close)...", flush=True)
    ffd_results = {}
    d_values = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_process_asset, (a, s, True)): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, results, d = f.result()
            done += 1
            ffd_results[asset] = results
            d_values[asset] = d
            print(f"  [{done:>2d}/30] {asset}: d={d}, {len(results)} months", flush=True)

    # Optimal d per asset
    print(f"\n  Optimal d values:")
    for asset in sorted(d_values, key=lambda x: d_values[x] or 0):
        d = d_values[asset]
        if d is not None:
            print(f"    {asset}: d={d:.1f}")

    # Phase 3: Portfolio sims
    print(f"\n{'='*75}")
    print("RESULTS")
    print(f"{'='*75}", flush=True)

    base_stats = run_test("Baseline (13 feat)", baseline_results)
    ffd_stats = run_test("FFD (14 feat)", ffd_results)

    if base_stats:
        print(f"  Baseline (13):  Sharpe {base_stats['sharpe']:.3f}  Ret {base_stats['return_pct']:.2f}%  "
              f"MaxDD {base_stats['max_dd_pct']:.2f}%  Trades {base_stats['n_trades']}  PnL/day ${base_stats['pnl_day']:.2f}")
    if ffd_stats and base_stats:
        delta = ffd_stats["sharpe"] - base_stats["sharpe"]
        print(f"  FFD (14):       Sharpe {ffd_stats['sharpe']:.3f} ({delta:+.3f})  Ret {ffd_stats['return_pct']:.2f}%  "
              f"MaxDD {ffd_stats['max_dd_pct']:.2f}%  Trades {ffd_stats['n_trades']}  PnL/day ${ffd_stats['pnl_day']:.2f}")
    elif ffd_stats:
        print(f"  FFD (14):       Sharpe {ffd_stats['sharpe']:.3f}  (no baseline to compare)")

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
