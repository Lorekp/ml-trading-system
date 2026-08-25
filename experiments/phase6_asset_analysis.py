"""Phase 6: Asset analysis — what makes an asset profitable for this signal?

Can we find ex-ante criteria to predict which assets the model works on,
without using future performance data?

Candidate predictors (all known BEFORE trading):
- Market cap / ADV (liquidity)
- Average spread (from CS spread estimator)
- Volatility level
- Data history length
- Signal count (model confidence on this asset)
- Average confidence when model trades
- Long/short ratio (directional bias)
- Autocorrelation at 15min (signal mechanism)
"""

import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np, pandas as pd, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from signal_engine.config import load_config, ConfidenceFilterConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
LB = 14
TEST_START = "2024-01"
CONF_THR = 0.14


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.01, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.01, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def process_asset(args):
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")
    from signal_engine.config import load_config
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights
    from afml.bet_sizing import get_signal, discretize_signal

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, None

    close = ohlcv_15m["close"]
    n_bars = len(close)

    # === Asset characteristics (ex-ante, from training period) ===
    # Use data up to 2023-12 for characteristics, 2024+ for performance
    train_end = pd.Timestamp("2023-12-31", tz="UTC")
    if close.index.tz is None:
        train_end = train_end.tz_localize(None)
    train_close = close.loc[:train_end]
    if len(train_close) < 5000:
        return asset, None

    # Characteristics computed from training period ONLY
    train_ret = train_close.pct_change().dropna()

    char = {}
    char["asset"] = asset
    char["n_bars_total"] = n_bars
    char["n_bars_train"] = len(train_close)
    char["avg_price"] = float(train_close.mean())
    char["median_price"] = float(train_close.median())

    # Volatility
    char["annual_vol"] = float(train_ret.std() * np.sqrt(365.25 * 24 * 4))
    char["vol_of_vol"] = float(train_ret.rolling(96).std().std() * np.sqrt(365.25 * 24 * 4))

    # Autocorrelation (the mechanism our signal exploits)
    for lag in [1, 2, 4, 8, 16]:
        ac = train_ret.autocorr(lag=lag)
        char[f"autocorr_lag{lag}"] = float(ac) if np.isfinite(ac) else 0.0

    # Volume
    volume = ohlcv_15m["volume"].loc[:train_end]
    char["avg_volume"] = float(volume.mean())
    char["avg_dollar_vol"] = float((volume * train_close).mean())

    # CS spread (from features)
    features = compute_features(ohlcv_15m.loc[:train_end], cfg.features)
    if "cs_spread" in features.columns:
        cs = features["cs_spread"].dropna()
        char["avg_cs_spread"] = float(cs.mean()) if len(cs) > 100 else np.nan
        char["median_cs_spread"] = float(cs.median()) if len(cs) > 100 else np.nan
    else:
        char["avg_cs_spread"] = np.nan
        char["median_cs_spread"] = np.nan

    # Entropy
    if "entropy" in features.columns:
        ent = features["entropy"].dropna()
        char["avg_entropy"] = float(ent.mean()) if len(ent) > 100 else np.nan
    else:
        char["avg_entropy"] = np.nan

    # Parkinson vol
    if "parkinson_vol_20" in features.columns:
        pv = features["parkinson_vol_20"].dropna()
        char["avg_parkinson_vol"] = float(pv.mean()) if len(pv) > 100 else np.nan
    else:
        char["avg_parkinson_vol"] = np.nan

    # Kurtosis and skewness of returns
    char["ret_skew"] = float(train_ret.skew())
    char["ret_kurtosis"] = float(train_ret.kurtosis())

    # Hurst exponent approximation (R/S method, quick)
    def hurst_rs(ts, max_lag=100):
        lags = range(2, min(max_lag, len(ts) // 10))
        rs = []
        for lag in lags:
            chunks = [ts[i:i+lag] for i in range(0, len(ts)-lag, lag)]
            if len(chunks) < 5: continue
            rs_vals = []
            for chunk in chunks:
                if len(chunk) < 2: continue
                m = chunk.mean()
                dev = chunk - m
                cumdev = dev.cumsum()
                R = cumdev.max() - cumdev.min()
                S = chunk.std()
                if S > 0:
                    rs_vals.append(R / S)
            if rs_vals:
                rs.append((np.log(lag), np.log(np.mean(rs_vals))))
        if len(rs) < 5:
            return 0.5
        x, y = zip(*rs)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    char["hurst"] = hurst_rs(train_ret.values)

    # === OOS performance (2024+) ===
    feat_all = features.reindex(ohlcv_15m.index)
    # Recompute features on full data
    features_full = compute_features(ohlcv_15m, cfg.features)
    feat = features_full[cfg.features.feature_list].copy()
    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in feat.columns:
            feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))
    fc = feat.dropna()
    fwd_ret = close.pct_change(4).shift(-4)
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)
    if len(valid) < 3000:
        return asset, char

    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5
    test_start_ts = pd.Timestamp(TEST_START, tz="UTC")
    if fc.index.tz is None:
        test_start_ts = test_start_ts.tz_localize(None)
    test_months = pd.period_range(start=test_start_ts.to_period("M"), end=end_date.to_period("M"), freq="M")

    all_pnls = []
    total_signals = 0
    total_conf = 0.0
    long_count = 0

    for month in test_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None); me = me.tz_localize(None)
        te = valid[(valid >= ms) & (valid <= me)]
        vol_train_start = ms - pd.DateOffset(months=3)
        vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms)].dropna()
        if len(vol_train) < 50: continue
        vt = vol_train.quantile(0.50)
        test_vol = vol.reindex(te)
        hv = (test_vol > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5: continue

        train_start = ms - pd.DateOffset(months=LB)
        if train_start < start_date: continue
        train_cutoff = ms - embargo_td
        tr = valid[(valid >= train_start) & (valid < train_cutoff)]
        if len(tr) < 200: continue

        try:
            sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(fc.loc[tr])
            model = make_logreg()
            model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
            Xte = scaler.transform(fc.loc[te])
            yp = model.predict_proba(Xte)[:, 1]
        except Exception: continue

        conf = np.abs(yp - 0.5)
        dirs = np.where(yp > 0.5, 1.0, -1.0)
        fv = fwd_ret.reindex(te).values
        finite = np.isfinite(fv)
        mask = finite & (conf > CONF_THR)
        if mask.sum() == 0: continue

        pnls = dirs[mask] * fv[mask] - 0.0001
        all_pnls.extend(pnls.tolist())
        total_signals += mask.sum()
        total_conf += conf[mask].sum()
        long_count += (dirs[mask] > 0).sum()

    if all_pnls:
        pnls_arr = np.array(all_pnls)
        ts_fake = pd.date_range("2024-01-01", periods=len(pnls_arr), freq="15min")
        daily = pd.Series(5.0 * pnls_arr, index=ts_fake).resample("1D").sum()
        daily = daily[daily != 0]
        char["oos_sharpe"] = float(daily.mean() / daily.std() * np.sqrt(365.25)) if len(daily) > 5 and daily.std() > 0 else 0
        char["oos_win_rate"] = float((pnls_arr > 0).mean())
        char["oos_bps"] = float(pnls_arr.mean() * 10000)
        char["oos_n_trades"] = len(pnls_arr)
        char["avg_confidence"] = total_conf / total_signals if total_signals > 0 else 0
        char["long_pct"] = long_count / total_signals * 100 if total_signals > 0 else 50
    else:
        char["oos_sharpe"] = 0
        char["oos_win_rate"] = 0
        char["oos_bps"] = 0
        char["oos_n_trades"] = 0
        char["avg_confidence"] = 0
        char["long_pct"] = 50

    return asset, char


def main():
    t0 = time.time()
    print("=" * 90)
    print("PHASE 6: ASSET CHARACTERISTIC ANALYSIS")
    print(f"  What makes an asset profitable for this signal?")
    print(f"  Characteristics from training (<2024), performance from OOS (2024+)")
    print("=" * 90, flush=True)

    all_chars = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s)): a for a, s in orig.ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, char = f.result()
            done += 1
            if char:
                all_chars[asset] = char
                sr = char.get("oos_sharpe", 0)
                print(f"  [{done:>2d}/30] {asset}: OOS SR={sr:.2f}", flush=True)
            else:
                print(f"  [{done:>2d}/30] {asset}: SKIP", flush=True)

    print(f"\n  Done in {time.time() - t0:.0f}s")

    df = pd.DataFrame(all_chars.values()).set_index("asset")
    df = df.sort_values("oos_sharpe", ascending=False)

    # ================================================================
    # ASSET TABLE
    # ================================================================
    print(f"\n{'='*90}")
    print("ASSET TABLE (sorted by OOS Sharpe)")
    print(f"{'='*90}")
    print(f"  {'Asset':<8s} {'OOS_SR':>7s} {'Win%':>5s} {'bps':>6s} {'Trades':>6s} "
          f"{'AnnVol':>7s} {'Hurst':>6s} {'AC1':>6s} {'AC4':>6s} {'Spread':>7s} "
          f"{'Entropy':>7s} {'Kurt':>6s} {'Long%':>6s}")
    print("  " + "-" * 88)
    for asset, row in df.iterrows():
        print(f"  {asset:<8s} {row['oos_sharpe']:>7.2f} {row['oos_win_rate']*100:>4.1f}% "
              f"{row['oos_bps']:>5.1f} {row['oos_n_trades']:>6.0f} "
              f"{row['annual_vol']:>6.1f}% {row['hurst']:>6.3f} "
              f"{row['autocorr_lag1']:>6.4f} {row['autocorr_lag4']:>6.4f} "
              f"{row.get('avg_cs_spread', 0):>7.5f} "
              f"{row.get('avg_entropy', 0):>7.4f} "
              f"{row['ret_kurtosis']:>6.1f} {row['long_pct']:>5.1f}%")

    # ================================================================
    # CORRELATION ANALYSIS
    # ================================================================
    print(f"\n{'='*90}")
    print("CORRELATION WITH OOS SHARPE (Spearman rank)")
    print(f"{'='*90}")

    predictors = [
        "annual_vol", "vol_of_vol", "hurst",
        "autocorr_lag1", "autocorr_lag2", "autocorr_lag4", "autocorr_lag8", "autocorr_lag16",
        "avg_volume", "avg_dollar_vol", "avg_cs_spread", "median_cs_spread",
        "avg_entropy", "avg_parkinson_vol",
        "ret_skew", "ret_kurtosis",
        "n_bars_train", "avg_confidence", "oos_n_trades", "long_pct",
    ]

    target = df["oos_sharpe"]
    results = []
    for pred in predictors:
        if pred not in df.columns: continue
        vals = df[pred].dropna()
        common = target.index.intersection(vals.index)
        if len(common) < 10: continue
        rho, pval = spearmanr(target.loc[common], vals.loc[common])
        results.append((pred, rho, pval, len(common)))

    results.sort(key=lambda x: -abs(x[1]))
    print(f"  {'Predictor':<25s} {'Spearman':>8s} {'p-value':>8s} {'n':>4s} {'Sig':>4s}")
    print("  " + "-" * 52)
    for pred, rho, pval, n in results:
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
        print(f"  {pred:<25s} {rho:>8.4f} {pval:>8.4f} {n:>4d} {sig:>4s}")

    # ================================================================
    # TOP vs BOTTOM COMPARISON
    # ================================================================
    print(f"\n{'='*90}")
    print("TOP 10 vs BOTTOM 10 COMPARISON (ex-ante characteristics)")
    print(f"{'='*90}")

    top10 = df.head(10)
    bot10 = df.tail(10)

    compare_cols = ["annual_vol", "hurst", "autocorr_lag1", "autocorr_lag4",
                    "avg_cs_spread", "avg_entropy", "ret_kurtosis", "ret_skew",
                    "avg_dollar_vol", "n_bars_train"]
    print(f"  {'Characteristic':<25s} {'Top 10 mean':>12s} {'Bot 10 mean':>12s} {'Ratio':>8s}")
    print("  " + "-" * 60)
    for col in compare_cols:
        if col not in df.columns: continue
        t_mean = top10[col].mean()
        b_mean = bot10[col].mean()
        ratio = t_mean / b_mean if b_mean != 0 else float("inf")
        print(f"  {col:<25s} {t_mean:>12.4f} {b_mean:>12.4f} {ratio:>8.2f}x")

    # ================================================================
    # ACTIONABLE FILTER
    # ================================================================
    print(f"\n{'='*90}")
    print("ACTIONABLE EX-ANTE FILTERS")
    print(f"{'='*90}")

    # Find significant predictors and suggest cutoffs
    sig_preds = [(p, r, pv) for p, r, pv, n in results if pv < 0.10]
    if sig_preds:
        print("  Significant predictors (p < 0.10):")
        for pred, rho, pval in sig_preds:
            direction = "higher" if rho > 0 else "lower"
            median_val = df[pred].median()
            # Test: assets above/below median
            above = df[df[pred] >= median_val]["oos_sharpe"].mean()
            below = df[df[pred] < median_val]["oos_sharpe"].mean()
            print(f"    {pred}: {direction} is better (rho={rho:.3f}, p={pval:.3f})")
            print(f"      Above median ({median_val:.4f}): avg OOS SR = {above:.2f}")
            print(f"      Below median: avg OOS SR = {below:.2f}")
    else:
        print("  No significant predictors found at p < 0.10")
        print("  Asset selection based on characteristics is NOT supported by this data.")
        print("  Recommendation: trade ALL assets (no ex-ante filtering).")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
