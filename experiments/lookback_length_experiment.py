"""Training lookback length experiment.

For each lookback X months: retrain monthly on last X months, test next month.
Sweep X = 1,2,3,4,6,8,10,12,14,18 + expanding (baseline).
Parallelized per-asset.
"""
import sys
sys.path.insert(0, "src")

import warnings, os, time, copy
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
from afml.bet_sizing import get_signal, discretize_signal

import sklearn
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)

W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
CONF_THRESH = 0.08
SLIPPAGE = 0.0001

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

LOOKBACKS = [1, 2, 3, 4, 6, 8, 10, 12, 14, 18, "expanding"]


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def _process_asset_all_lookbacks(args):
    """Run all lookback lengths for one asset. Returns {lookback: signals_dict}."""
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    cfg = load_config(Path("configs/default.yaml"))

    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, {}

    features = compute_features(ohlcv_15m, cfg.features)
    feature_cols = cfg.features.feature_list
    feat = features[feature_cols].copy()

    close = ohlcv_15m["close"]
    fwd_ret = close.pct_change(4).shift(-4)
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    # Winsorize globally (conservative — same bounds for all windows)
    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in feat.columns:
            lo = feat[col].quantile(0.01)
            hi = feat[col].quantile(0.99)
            feat[col] = feat[col].clip(lo, hi)

    fc = feat.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)

    if len(valid) < 5000:
        return asset, {}

    # Determine monthly periods
    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    all_months = pd.period_range(
        start=(start_date + pd.DateOffset(months=1)).to_period("M"),
        end=end_date.to_period("M"),
        freq="M",
    )

    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    results_per_lookback = {lb: [] for lb in LOOKBACKS}

    for month in all_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

        # Test set for this month
        te = valid[(valid >= ms) & (valid <= me)]

        # Vol filter (need some training data for threshold)
        # Use last 3 months for vol threshold regardless of lookback
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

        X_test = fc.loc[te]
        fwd_test = fwd_ret.reindex(te)

        for lb in LOOKBACKS:
            if lb == "expanding":
                # All data before this month
                train_cutoff = ms - embargo_td
                tr = valid[valid < train_cutoff]
            else:
                # Rolling window: last lb months before this month
                train_start = ms - pd.DateOffset(months=lb)
                train_cutoff = ms - embargo_td
                tr = valid[(valid >= train_start) & (valid < train_cutoff)]

            if len(tr) < 200:
                continue

            X_train = fc.loc[tr]
            y_train = (y_all.loc[tr] > 0).astype(int)

            try:
                sw = compute_sample_weights(X_train.index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_train)
                model = make_logreg()
                model.fit(Xtr, y_train, sample_weight=sw)

                Xte = scaler.transform(X_test)
                yp = model.predict_proba(Xte)[:, 1]
            except Exception:
                continue

            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            sigs = get_signal(yp, num_classes=2)
            sd = discretize_signal(sigs, step_size=0.1)
            ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
            fv = fwd_test.values
            finite = np.isfinite(fv)

            if not finite.any():
                continue

            signals = pd.DataFrame({
                "timestamp": te[finite], "direction": dirs[finite],
                "fwd_return": fv[finite], "confidence": conf[finite],
                "probability": yp[finite], "signal_strength": ss[finite],
            })
            signals["asset"] = asset

            # Confidence filter
            filtered = apply_confidence_filter(
                signals, ConfidenceFilterConfig(mode="absolute", threshold=CONF_THRESH)
            )
            if len(filtered) > 0:
                results_per_lookback[lb].append(filtered.to_dict("list"))

    return asset, results_per_lookback


def main():
    print("=" * 80)
    print(f"LOOKBACK LENGTH EXPERIMENT ({W} workers)")
    print(f"Sweep: {LOOKBACKS}")
    print(f"30 assets, monthly retrain, conf >= 0.08, 1 bps slippage")
    print("=" * 80, flush=True)

    t0 = time.time()

    # Accumulate signals per lookback across all assets
    all_per_lookback = {lb: [] for lb in LOOKBACKS}

    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_process_asset_all_lookbacks, (a, s)): a for a, s in ASSETS.items()}
        done = 0
        for future in as_completed(futures):
            asset, results = future.result()
            done += 1
            total_sigs = sum(
                sum(len(pd.DataFrame(d)) for d in results.get(lb, []))
                for lb in LOOKBACKS
            )
            print(f"  [{done:>2d}/30] {asset}: {total_sigs} total signals across all lookbacks", flush=True)

            for lb in LOOKBACKS:
                all_per_lookback[lb].extend(results.get(lb, []))

    gen_time = time.time() - t0
    print(f"\nGenerated in {gen_time:.0f}s", flush=True)

    # Portfolio sim for each lookback
    cfg = load_config(Path("configs/default.yaml"))
    pcfg = copy.deepcopy(cfg.portfolio)
    pcfg.base_sizing_pct = 0.04
    pcfg.slippage = SLIPPAGE
    pcfg.max_exposure = 0.90
    pcfg.max_per_asset = 0.03

    print(f"\n{'Lookback':<12s} {'Sharpe':>7s} {'Ret%':>8s} {'MaxDD%':>7s} {'Trades':>7s} {'Days':>5s} {'PnL/day':>8s}")
    print("-" * 60)

    for lb in LOOKBACKS:
        sig_dicts = all_per_lookback[lb]
        if not sig_dicts:
            lb_str = str(lb) + "mo" if lb != "expanding" else "expanding"
            print(f"  {lb_str:<12s} NO SIGNALS", flush=True)
            continue

        frames = [pd.DataFrame(d) for d in sig_dicts]
        combined = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)

        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 10:
            lb_str = str(lb) + "mo" if lb != "expanding" else "expanding"
            print(f"  {lb_str:<12s} TOO FEW TRADES", flush=True)
            continue

        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 5:
            continue

        stats = compute_full_stats(dpnl, pcfg.starting_capital,
                                    sim["n_trades"], sim["n_trades"] / len(dpnl))
        pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)

        lb_str = str(lb) + "mo" if lb != "expanding" else "expanding"
        print(
            f"  {lb_str:<12s} {stats['sharpe']:>7.3f} {stats['total_return_pct']:>8.2f} "
            f"{stats['max_dd_pct']:>7.2f} {sim['n_trades']:>7d} {stats['n_days']:>5d} "
            f"${pnl_day:>7.2f}",
            flush=True,
        )

    print(f"\n{'='*60}")
    print("DONE")


if __name__ == "__main__":
    main()
