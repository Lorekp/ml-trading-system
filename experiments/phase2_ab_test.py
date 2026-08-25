"""Phase 2 A/B Test: Isolate each improvement.

Variants (all use LogReg L1, 18 features, lb=8, 30 assets):
A. Baseline: fixed-horizon labels, simple weights
B. CUSUM sampling + fixed-horizon labels (fewer, better events)
C. Return-attributed weights + time decay (better weighting)
D. B + C combined (CUSUM + better weights)
E. Triple-barrier labels + CUSUM + better weights (full AFML)

Each variant is evaluated on FIXED-HORIZON forward returns for fair comparison.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_engine.config import load_config, ConfidenceFilterConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.filters import apply_confidence_filter
from signal_engine.weights import compute_sample_weights
from afml.labeling import get_events, get_bins
from afml.filters import cusum_filter, get_daily_vol
from afml.sampling import (
    get_num_concurrent_events, get_avg_uniqueness,
    get_sample_weights as afml_sample_weights, get_time_decay,
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


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def process_asset(args):
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    from signal_engine.config import load_config, ConfidenceFilterConfig
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights
    from signal_engine.filters import apply_confidence_filter
    from afml.filters import cusum_filter, get_daily_vol
    from afml.labeling import get_events, get_bins
    from afml.sampling import (
        get_num_concurrent_events, get_avg_uniqueness,
        get_sample_weights as afml_sample_weights, get_time_decay,
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
    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in feat.columns:
            feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))
    fc = feat.dropna()
    if len(fc) < 3000:
        return asset, {}

    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
    daily_vol = get_daily_vol(close, span=100)
    fwd_ret = close.pct_change(4).shift(-4)  # fixed-horizon for ALL evaluations
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)

    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    # Results per variant
    results = {v: {8: []} for v in ["A", "B", "C", "D", "E"]}

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

        # Test bars (vol-filtered)
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
        tr_all = valid[(valid >= train_start) & (valid < train_cutoff)]
        if len(tr_all) < 200:
            continue

        # CUSUM events on training period (log prices)
        train_close = close.loc[train_start:train_cutoff]
        train_log = np.log(train_close)
        train_dv = daily_vol.reindex(train_close.index).dropna()
        cusum_events = pd.DatetimeIndex([])
        if len(train_dv) > 50:
            cusum_events = cusum_filter(train_log, train_dv * 1.5)
            cusum_events = cusum_events[cusum_events.isin(valid)]

        # Triple-barrier labels on CUSUM events
        tb_labels = None
        tb_t1 = None
        if len(cusum_events) >= 50:
            target = daily_vol.reindex(cusum_events).dropna()
            if len(target) >= 50:
                events = get_events(close, target.index, [1.0, 1.0], target, num_bars_vertical=16)
                if len(events) >= 50:
                    events_std = events.drop(columns=["side"], errors="ignore")
                    bins = get_bins(events_std, close)
                    non_zero = bins[bins["bin"] != 0].dropna(subset=["bin"])
                    if len(non_zero) >= 50:
                        tb_labels = non_zero
                        tb_t1 = events.loc[non_zero.index, "t1"]

        def train_and_predict(tr_idx, sw, variant_name):
            """Train LogReg on tr_idx with weights sw, predict on te, save signals."""
            if len(tr_idx) < 50:
                return
            tr_y = (y_all.loc[tr_idx] > 0).astype(int)
            try:
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[tr_idx])
                model = make_logreg()
                model.fit(Xtr, tr_y, sample_weight=sw)
                Xte = scaler.transform(fc.loc[te])
                yp = model.predict_proba(Xte)[:, 1]
            except Exception:
                return

            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            sigs = get_signal(yp, num_classes=2)
            sd = discretize_signal(sigs, step_size=0.1)
            ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
            fv = fwd_ret.reindex(te).values
            finite = np.isfinite(fv)
            if not finite.any():
                return

            signals = pd.DataFrame({
                "timestamp": te[finite], "direction": dirs[finite],
                "fwd_return": fv[finite], "confidence": conf[finite],
                "probability": yp[finite], "signal_strength": ss[finite],
            })
            signals["asset"] = asset
            filtered = apply_confidence_filter(
                signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
            if len(filtered) > 0:
                results[variant_name][8].append((month_str, filtered.to_dict("list")))

        # === A: Baseline (all bars, simple weights) ===
        sw_a = compute_sample_weights(fc.loc[tr_all].index.values.astype(np.int64), fwd_bars=4)
        train_and_predict(tr_all, sw_a, "A")

        # === B: CUSUM sampling + fixed-horizon labels ===
        if len(cusum_events) >= 50:
            tr_cusum = cusum_events.intersection(valid)
            tr_cusum = tr_cusum[(tr_cusum >= train_start) & (tr_cusum < train_cutoff)]
            if len(tr_cusum) >= 50:
                sw_b = compute_sample_weights(fc.loc[tr_cusum].index.values.astype(np.int64), fwd_bars=4)
                train_and_predict(tr_cusum, sw_b, "B")

        # === C: All bars + return-attributed weights + time decay ===
        # Use fixed-horizon t1 for weight computation
        t1_fixed = pd.Series(
            close.index[close.index.searchsorted(tr_all) + 4].values,
            index=tr_all,
            dtype=close.index.dtype,
        )
        t1_fixed = t1_fixed[t1_fixed.index != t1_fixed.values]  # drop self-referencing
        if len(t1_fixed) >= 50:
            try:
                num_conc = get_num_concurrent_events(close.index, t1_fixed)
                sw_ret = afml_sample_weights(t1_fixed, num_conc, close)
                avg_u = get_avg_uniqueness(t1_fixed, num_conc)
                sw_decay = get_time_decay(avg_u, clf_last_w=0.5)
                sw_c = (sw_ret * sw_decay.reindex(sw_ret.index).fillna(1.0))
                sw_c = sw_c / sw_c.mean()
                sw_c = sw_c.reindex(tr_all).fillna(1.0).values
                train_and_predict(tr_all, sw_c, "C")
            except Exception:
                pass

        # === D: CUSUM + return-attributed weights ===
        if len(cusum_events) >= 50 and tb_t1 is not None:
            tr_cusum_d = cusum_events.intersection(valid)
            tr_cusum_d = tr_cusum_d[(tr_cusum_d >= train_start) & (tr_cusum_d < train_cutoff)]
            tr_cusum_d = tr_cusum_d.intersection(tb_t1.index)
            if len(tr_cusum_d) >= 50:
                try:
                    t1_d = tb_t1.reindex(tr_cusum_d).dropna()
                    if len(t1_d) >= 50:
                        num_conc_d = get_num_concurrent_events(close.index, t1_d)
                        sw_ret_d = afml_sample_weights(t1_d, num_conc_d, close)
                        avg_u_d = get_avg_uniqueness(t1_d, num_conc_d)
                        sw_decay_d = get_time_decay(avg_u_d, clf_last_w=0.5)
                        sw_d = (sw_ret_d * sw_decay_d.reindex(sw_ret_d.index).fillna(1.0))
                        sw_d = sw_d / sw_d.mean()
                        sw_d = sw_d.reindex(tr_cusum_d).fillna(1.0).values
                        train_and_predict(tr_cusum_d, sw_d, "D")
                except Exception:
                    pass

        # === E: Triple-barrier labels + CUSUM + return-attributed weights ===
        if tb_labels is not None and tb_t1 is not None and len(tb_labels) >= 50:
            tr_tb = tb_labels.index.intersection(fc.index)
            tr_tb = tr_tb[(tr_tb >= train_start) & (tr_tb < train_cutoff)]
            if len(tr_tb) >= 50:
                try:
                    # Labels from triple-barrier, but train to predict DIRECTION
                    y_tb = (tb_labels.loc[tr_tb, "bin"] > 0).astype(int)
                    t1_e = tb_t1.reindex(tr_tb).dropna()
                    if len(t1_e) >= 50:
                        num_conc_e = get_num_concurrent_events(close.index, t1_e)
                        sw_ret_e = afml_sample_weights(t1_e, num_conc_e, close)
                        avg_u_e = get_avg_uniqueness(t1_e, num_conc_e)
                        sw_decay_e = get_time_decay(avg_u_e, clf_last_w=0.5)
                        sw_e = (sw_ret_e * sw_decay_e.reindex(sw_ret_e.index).fillna(1.0))
                        sw_e = sw_e / sw_e.mean()
                        sw_e = sw_e.reindex(tr_tb).fillna(1.0).values

                        scaler = StandardScaler()
                        Xtr = scaler.fit_transform(fc.loc[tr_tb])
                        model = make_logreg()
                        model.fit(Xtr, y_tb, sample_weight=sw_e)
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
                                results["E"][8].append((month_str, filtered.to_dict("list")))
                except Exception:
                    pass

    return asset, results


def flat_bet_stats(all_results, variant, start=None, end=None):
    sigs = []
    for ar in all_results.values():
        bucket = ar.get(variant, {})
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
    selected = os.getenv("ML_SIGNAL_ASSETS", "").strip()
    if selected:
        asset_list = {k: v for k, v in ASSETS.items() if k in selected.split(",")}
    else:
        asset_list = ASSETS

    variants = {
        "A": "Baseline (all bars, simple weights)",
        "B": "CUSUM sampling only",
        "C": "Return-attr weights + time decay",
        "D": "CUSUM + return-attr weights",
        "E": "Triple-barrier + CUSUM + weights",
    }

    print("=" * 80)
    print("PHASE 2 A/B TEST: Isolate each AFML improvement")
    print(f"  {len(asset_list)} assets, {W} workers, lb=8")
    for k, v in variants.items():
        print(f"  {k}: {v}")
    print("=" * 80, flush=True)

    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s)): a for a, s in asset_list.items()}
        done = 0
        for f in as_completed(futures):
            asset, res = f.result()
            done += 1
            all_results[asset] = res
            counts = {v: sum(len(x) for x in res.get(v, {}).values()) for v in variants}
            print(f"  [{done:>2d}/{len(asset_list)}] {asset}: {counts}", flush=True)

    print(f"\n  Done in {time.time()-t0:.0f}s")

    # Report
    for period, start, end in [("SEARCH (2022-2024)", "2022-01", "2024-12"),
                                ("HOLDOUT (2025+)", "2025-01", None),
                                ("FULL", None, None)]:
        print(f"\n{'='*80}")
        print(f"  {period}")
        print(f"{'='*80}")
        print(f"  {'Variant':<40s} {'SR':>7s} {'Win%':>6s} {'bps':>7s} {'Trades':>8s} {'Assets':>6s}")
        print("  " + "-" * 70)
        for v, label in variants.items():
            r = flat_bet_stats(all_results, v, start, end)
            if r:
                print(f"  {v}: {label:<36s} {r['sharpe']:>7.3f} {r['win_rate']:>5.1f}% "
                      f"{r['mean_bps']:>6.2f} {r['n_trades']:>8d} {r['n_assets']:>5d}")
            else:
                print(f"  {v}: {label:<36s}  NO DATA")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
