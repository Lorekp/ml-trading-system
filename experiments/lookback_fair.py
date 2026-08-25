"""Fair lookback comparison — two versions.

Version A: Each lookback starts ASAP (when it has enough data). No expanding fallback.
Version B: ALL lookbacks tested on SAME months (start at month 14). Apples-to-apples.

Walk-forward Kelly weighting applied per-asset (uses only past signals).
"""
import sys
sys.path.insert(0, "src")

import copy, warnings, os, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from signal_engine.config import load_config, ConfidenceFilterConfig
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

LOOKBACKS = [1, 2, 3, 4, 6, 8, 10, 12, 14, 18]


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def _process_asset(args):
    """Generate signals for ALL lookbacks. Returns {lookback: {month_str: signals_dict}}."""
    asset, symbol, fixed_start_month = args
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
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except:
        return asset, {}

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
        return asset, {}

    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    results = {lb: [] for lb in LOOKBACKS}

    # Generate all possible test months
    all_months = pd.period_range(
        start=(start_date + pd.DateOffset(months=2)).to_period("M"),
        end=end_date.to_period("M"), freq="M")

    for month in all_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

        # Test set
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

        for lb in LOOKBACKS:
            # Check if this lookback has enough history
            train_start = ms - pd.DateOffset(months=lb)
            if train_start < start_date:
                continue  # NOT ENOUGH DATA — skip, NO fallback

            # For Version B: skip if before the fixed start month
            # (handled in aggregation, not here — generate all, filter later)

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
            except:
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
                results[lb].append((month_str, filtered.to_dict("list")))

    return asset, results


def aggregate_and_report(label, all_results, start_after_month=None):
    """Combine signals and run portfolio sim."""
    cfg = load_config(Path("configs/default.yaml"))
    pcfg = copy.deepcopy(cfg.portfolio)
    pcfg.base_sizing_pct = 0.04
    pcfg.slippage = 0.0001
    pcfg.max_exposure = 0.90
    pcfg.max_per_asset = 0.03

    print(f"\n{label}")
    print(f"{'Lookback':<10s} {'Sharpe':>7s} {'Ret%':>8s} {'MaxDD%':>7s} {'Trades':>7s} {'Days':>5s} {'PnL/day':>8s}")
    print("-" * 55)

    for lb in LOOKBACKS:
        sigs = []
        for asset_results in all_results.values():
            for month_str, sig_dict in asset_results.get(lb, []):
                if start_after_month and month_str < start_after_month:
                    continue
                sigs.append(sig_dict)

        if not sigs:
            print(f"  {lb:>2d}mo      NO SIGNALS", flush=True)
            continue

        frames = [pd.DataFrame(d) for d in sigs]
        combined = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)

        # Walk-forward Kelly weighting (uses only PAST signals per month)
        timestamps_pd = pd.to_datetime(combined["timestamp"])
        kelly_months = sorted(timestamps_pd.dt.to_period("M").unique())
        kelly_parts = []
        for km in kelly_months:
            km_mask = timestamps_pd.dt.to_period("M") == km
            km_sigs = combined[km_mask]
            past_mask = timestamps_pd.dt.to_period("M") < km
            past_sigs = combined[past_mask]
            if len(past_sigs) < 50:
                kelly_parts.append(km_sigs)
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
                wp = len(w) / len(tr)
                wr = float(w.mean())
                lr = float(abs(lo.mean()))
                if lr <= 0:
                    kw[an] = 1.0
                    continue
                fs = kelly_size(wp, wr, lr)
                kw[an] = max(0.2, min(3.0, fs * 7.5))
            if kw:
                mw = np.mean(list(kw.values()))
                if mw > 0:
                    kw = {k: v / mw for k, v in kw.items()}
            kmc = km_sigs.copy()
            kmc["signal_strength"] = kmc.apply(
                lambda r: r["signal_strength"] * kw.get(r["asset"], 1.0), axis=1)
            kelly_parts.append(kmc)
        combined = pd.concat(kelly_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 10:
            print(f"  {lb:>2d}mo      TOO FEW", flush=True)
            continue

        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 5:
            continue

        stats = compute_full_stats(dpnl, pcfg.starting_capital,
                                    sim["n_trades"], sim["n_trades"] / len(dpnl))
        pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)

        print(
            f"  {lb:>2d}mo      {stats['sharpe']:>7.3f} {stats['total_return_pct']:>8.2f} "
            f"{stats['max_dd_pct']:>7.2f} {sim['n_trades']:>7d} {stats['n_days']:>5d} "
            f"${pnl_day:>7.2f}",
            flush=True,
        )


def main():
    print("=" * 70)
    print(f"FAIR LOOKBACK COMPARISON ({W} workers)")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"NO expanding fallback — each lookback uses EXACTLY its window")
    print("=" * 70, flush=True)

    t0 = time.time()
    all_results = {}

    # Find the fixed start month for Version B (month 13 = max(LOOKBACKS) + 1)
    # We need to know the data start date first
    cfg = load_config(Path("configs/default.yaml"))
    sample_1m = load_binance_1m("BTCUSDT", [DATA_DIR])
    sample_15m = resample_ohlcv(sample_1m, "15min")
    data_start = sample_15m.index.min()
    fixed_start = (data_start + pd.DateOffset(months=max(LOOKBACKS) + 1)).to_period("M")
    print(f"Data starts: {data_start.date()}")
    print(f"Version B fixed start: {fixed_start} (month {max(LOOKBACKS)+1})", flush=True)

    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_process_asset, (a, s, str(fixed_start))): a for a, s in ASSETS.items()}
        done = 0
        for future in as_completed(futures):
            asset, results = future.result()
            done += 1
            all_results[asset] = results
            total = sum(len(v) for v in results.values())
            print(f"  [{done:>2d}/30] {asset}: {total} month-lookback combos", flush=True)

    print(f"\nGenerated in {time.time()-t0:.0f}s")

    # Version A: Start ASAP (each lookback has different test period)
    aggregate_and_report(
        "VERSION A: Start ASAP (each lookback starts when it has enough data)",
        all_results,
        start_after_month=None,
    )

    # Version B: Same test period (all start at month 13+)
    aggregate_and_report(
        f"VERSION B: Same test period (all start at {fixed_start}, apples-to-apples)",
        all_results,
        start_after_month=str(fixed_start),
    )

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == "__main__":
    main()
