"""
Stacked Direction Prediction Experiment.

Combines OHLCV features + technical indicators + HL derivatives (funding, OI, premium)
and tests direction prediction at multiple horizons (1, 3, 5, 10, 20 bars).

Run on homeserver where all data is available.
"""

import sys
import os
import warnings
import json

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/ml-data")

import numpy as np
import pandas as pd
import lz4.frame
import csv
import io
from datetime import datetime
from pathlib import Path
from scipy.stats import binom

from technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels
from derivatives import funding_features, oi_features, premium_features, impact_features


def load_btc_15m():
    """Load BTC 1m from Binance CSVs, resample to 15m."""
    btc_dir = Path("/home/user/ml-data/binance-1m/BTCUSDT")
    csvs = sorted(btc_dir.glob("*.csv"))
    dfs = []
    for f in csvs:
        with open(f) as fh:
            first = fh.readline().strip().split(",")[0]
        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "qv", "count", "tbv", "tqv", "ignore"]
        if first.replace(".", "").replace("-", "").isdigit():
            df = pd.read_csv(f, header=None, names=cols)
        else:
            df = pd.read_csv(f)
            df.columns = cols
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.resample("15min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open"])


def load_btc_derivatives(ohlcv_index):
    """Load HL asset_ctxs for BTC, align to OHLCV index."""
    ctx_dir = Path("/home/user/ml-data/hl-asset-ctxs")
    ohlcv_start = ohlcv_index[0].strftime("%Y%m%d")
    ohlcv_end = ohlcv_index[-1].strftime("%Y%m%d")

    ctx_dfs = []
    for fpath in sorted(ctx_dir.glob("*.lz4")):
        date_str = fpath.stem.replace(".csv", "")
        if date_str < ohlcv_start or date_str > ohlcv_end:
            continue
        try:
            raw = lz4.frame.decompress(open(fpath, "rb").read()).decode("utf-8")
            rows = [r for r in csv.DictReader(io.StringIO(raw)) if r.get("coin") == "BTC"]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
            num_cols = ["funding", "open_interest", "premium",
                        "oracle_px", "mark_px", "impact_bid_px", "impact_ask_px"]
            for col in num_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            ctx_dfs.append(df[num_cols])
        except Exception:
            pass

    if not ctx_dfs:
        return pd.DataFrame()

    deriv = pd.concat(ctx_dfs).sort_index()
    deriv = deriv[~deriv.index.duplicated(keep="first")]
    return deriv.resample("15min").last().reindex(ohlcv_index, method="ffill")


def build_features(btc_15m, deriv_15m):
    """Build all features: OHLCV + technical + derivatives."""
    close = btc_15m["close"]
    high = btc_15m["high"]
    low = btc_15m["low"]
    volume = btc_15m["volume"]

    features = pd.DataFrame(index=btc_15m.index)

    # OHLCV returns (winsorized)
    for lb in [1, 3, 7, 14]:
        ret = close.pct_change(lb)
        lo, hi = ret.quantile(0.01), ret.quantile(0.99)
        features[f"ret_{lb}"] = ret.clip(lo, hi)

    features["bar_range"] = (high - low) / close
    features["vol_zscore"] = (
        (volume - volume.rolling(30).mean()) /
        volume.rolling(30).std().replace(0, np.nan)
    )

    # Technical indicators
    features["rsi_14"] = rsi(close, 14)
    features["stoch_rsi"] = stochastic_rsi(close, 14, 14)
    macd_df = macd(close)
    features["macd_hist"] = macd_df["macd_hist"]
    features["bb_pct_b"] = bollinger_pct_b(close, 20)
    features["atr_14"] = atr(high, low, close, 14)

    sw = swing_levels(high, low, lookback=20)
    features["dist_swing_high"] = sw["dist_swing_high"]
    features["dist_swing_low"] = sw["dist_swing_low"]
    features["dist_nearest_level"] = sw["dist_nearest_level"]

    # Derivatives features
    if len(deriv_15m) > 0 and "funding" in deriv_15m.columns:
        fund_f = funding_features(deriv_15m["funding"])
        for col in fund_f.columns:
            features[col] = fund_f[col]

        oi_f = oi_features(deriv_15m["open_interest"], close)
        for col in oi_f.columns:
            features[col] = oi_f[col]

        prem_f = premium_features(deriv_15m["mark_px"], deriv_15m["oracle_px"])
        for col in prem_f.columns:
            features[col] = prem_f[col]

        impact_f = impact_features(
            deriv_15m["impact_bid_px"], deriv_15m["impact_ask_px"],
            deriv_15m["mark_px"],
        )
        for col in impact_f.columns:
            features[col] = impact_f[col]

    return features


def run_experiment():
    print("=== STACKED DIRECTION PREDICTION ===", flush=True)
    print(f"Started: {datetime.now()}", flush=True)

    # Load data
    print("Loading BTC 15m OHLCV...", flush=True)
    btc_15m = load_btc_15m()
    print(f"  {len(btc_15m)} bars ({btc_15m.index[0].date()} to {btc_15m.index[-1].date()})", flush=True)

    print("Loading HL derivatives...", flush=True)
    deriv_15m = load_btc_derivatives(btc_15m.index)
    print(f"  {deriv_15m.notna().sum().min()} aligned rows", flush=True)

    # Build features
    print("Building features...", flush=True)
    features = build_features(btc_15m, deriv_15m)
    features_clean = features.dropna()
    print(f"  {len(features.columns)} features, {len(features_clean)} clean rows", flush=True)
    print(f"  Features: {list(features.columns)}", flush=True)

    # Labels at multiple horizons
    close = btc_15m["close"]
    horizons = [1, 3, 5, 10, 20]

    # Train/test split: train on data before July 2025, test after
    train_mask = features_clean.index < "2025-07-01"
    test_mask = features_clean.index >= "2025-07-01"
    X_train = features_clean[train_mask]
    X_test = features_clean[test_mask]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}", flush=True)

    import xgboost as xgb

    results = []
    print("\n=== RESULTS ===", flush=True)
    print(f"{'Hz':>4} {'Acc':>7} {'p-val':>7} {'n':>6} | {'Acc30%':>7} {'p30%':>7} {'n30%':>6}", flush=True)
    print("-" * 60, flush=True)

    for h in horizons:
        fwd_ret = close.pct_change(h).shift(-h)
        y_all = np.sign(fwd_ret).reindex(features_clean.index).dropna()

        common_train = X_train.index.intersection(y_all.index)
        common_test = X_test.index.intersection(y_all.index)

        if len(common_train) < 500 or len(common_test) < 200:
            print(f"  {h:>2}-bar: insufficient data", flush=True)
            continue

        y_train = (y_all.loc[common_train] > 0).astype(int)
        y_test = (y_all.loc[common_test] > 0).astype(int)

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=42,
        )
        model.fit(X_train.loc[common_train], y_train)

        y_prob = model.predict_proba(X_test.loc[common_test])[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        acc = (y_pred == y_test.values).mean()
        k = int(acc * len(y_test))
        p_val = 1 - binom.cdf(k - 1, len(y_test), 0.5) if k > len(y_test) / 2 else 1.0

        # Confidence filtered (top 30%)
        conf = np.abs(y_prob - 0.5)
        top30 = conf >= np.percentile(conf, 70)
        if top30.sum() > 50:
            acc_f = (y_pred[top30] == y_test.values[top30]).mean()
            n_f = int(top30.sum())
            k_f = int(acc_f * n_f)
            p_f = 1 - binom.cdf(k_f - 1, n_f, 0.5) if k_f > n_f / 2 else 1.0
        else:
            acc_f, n_f, p_f = acc, 0, 1.0

        r = {
            "horizon": h, "accuracy": round(acc, 4), "p_value": round(p_val, 4),
            "n_test": len(y_test), "acc_filtered": round(acc_f, 4),
            "p_filtered": round(p_f, 4), "n_filtered": n_f,
        }
        results.append(r)
        print(f"  {h:>2}  {acc:>7.4f} {p_val:>7.4f} {len(y_test):>6} | {acc_f:>7.4f} {p_f:>7.4f} {n_f:>6}", flush=True)

    # Feature importance for best horizon
    if results:
        best = max(results, key=lambda r: r["accuracy"])
        h = best["horizon"]
        print(f"\nBest horizon: {h}-bar (acc={best['accuracy']})", flush=True)

        fwd_ret = close.pct_change(h).shift(-h)
        y_all = np.sign(fwd_ret).reindex(features_clean.index).dropna()
        common_train = X_train.index.intersection(y_all.index)
        y_train = (y_all.loc[common_train] > 0).astype(int)

        model_best = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42,
        )
        model_best.fit(X_train.loc[common_train], y_train)

        importances = pd.Series(model_best.feature_importances_, index=features_clean.columns)
        importances = importances.sort_values(ascending=False)

        deriv_feats = {"funding", "funding_zscore", "funding_zscore_long", "funding_change",
                       "funding_cumulative", "oi_change_5", "oi_zscore", "oi_price_divergence",
                       "oi_price_interaction", "oi_acceleration", "premium", "premium_zscore",
                       "premium_sign_streak", "impact_spread", "impact_skew"}
        tech_feats = {"rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
                      "dist_swing_high", "dist_swing_low", "dist_nearest_level"}

        print("\nTop 15 features:", flush=True)
        for feat, imp in importances.head(15).items():
            src = "DERIV" if feat in deriv_feats else "TECH" if feat in tech_feats else "OHLCV"
            print(f"  {feat:>25}: {imp:.4f} [{src}]", flush=True)

        # Count by source
        top15_sources = []
        for feat in importances.head(15).index:
            if feat in deriv_feats:
                top15_sources.append("DERIV")
            elif feat in tech_feats:
                top15_sources.append("TECH")
            else:
                top15_sources.append("OHLCV")
        from collections import Counter
        print(f"\nTop 15 by source: {dict(Counter(top15_sources))}", flush=True)

    with open("/home/user/ml-data/direction_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone: {datetime.now()}", flush=True)


if __name__ == "__main__":
    run_experiment()
