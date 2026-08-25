"""
Ablation Study — verify which feature groups actually contribute.
Also: naive mean-reversion baseline, monthly stability, fix non-stationary features.
"""

import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/ml-data")

import numpy as np
import pandas as pd
import xgboost as xgb
import lz4.frame, csv, io
from datetime import datetime
from pathlib import Path
from scipy.stats import binom

from technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels
from derivatives import funding_features, oi_features, premium_features, impact_features


def load_data():
    """Load BTC 15m OHLCV + derivatives (same as direction_stacked.py)."""
    btc_dir = Path("/home/user/ml-data/binance-1m/BTCUSDT")
    dfs = []
    for f in sorted(btc_dir.glob("*.csv")):
        with open(f) as fh:
            first = fh.readline().strip().split(",")[0]
        cols = ["open_time","open","high","low","close","volume","close_time","qv","count","tbv","tqv","ignore"]
        if first.replace(".","").replace("-","").isdigit():
            df = pd.read_csv(f, header=None, names=cols)
        else:
            df = pd.read_csv(f); df.columns = cols
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open","high","low","close","volume"]]
        for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)
    btc_1m = pd.concat(dfs).sort_index()
    btc_1m = btc_1m[~btc_1m.index.duplicated(keep="first")]
    btc_15m = btc_1m.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["open"])

    ctx_dir = Path("/home/user/ml-data/hl-asset-ctxs")
    ohlcv_start = btc_15m.index[0].strftime("%Y%m%d")
    ohlcv_end = btc_15m.index[-1].strftime("%Y%m%d")
    ctx_dfs = []
    for fpath in sorted(ctx_dir.glob("*.lz4")):
        ds = fpath.stem.replace(".csv","")
        if ds < ohlcv_start or ds > ohlcv_end: continue
        try:
            raw = lz4.frame.decompress(open(fpath,"rb").read()).decode("utf-8")
            rows = [r for r in csv.DictReader(io.StringIO(raw)) if r.get("coin")=="BTC"]
            if not rows: continue
            df = pd.DataFrame(rows)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
            for col in ["funding","open_interest","premium","oracle_px","mark_px","impact_bid_px","impact_ask_px"]:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
            ctx_dfs.append(df[["funding","open_interest","premium","mark_px","oracle_px","impact_bid_px","impact_ask_px"]])
        except: pass
    deriv = pd.concat(ctx_dfs).sort_index()
    deriv = deriv[~deriv.index.duplicated(keep="first")]
    deriv_15m = deriv.resample("15min").last().reindex(btc_15m.index, method="ffill")
    return btc_15m, deriv_15m


def build_feature_groups(btc_15m, deriv_15m):
    """Build features in separate groups for ablation."""
    close = btc_15m["close"]
    high = btc_15m["high"]
    low = btc_15m["low"]
    volume = btc_15m["volume"]

    # Group 1: OHLCV
    ohlcv_feats = pd.DataFrame(index=btc_15m.index)
    for lb in [1, 3, 7, 14]:
        ret = close.pct_change(lb)
        lo, hi = ret.iloc[:int(len(ret)*0.6)].quantile(0.01), ret.iloc[:int(len(ret)*0.6)].quantile(0.99)
        ohlcv_feats[f"ret_{lb}"] = ret.clip(lo, hi)
    ohlcv_feats["bar_range"] = (high - low) / close
    ohlcv_feats["vol_zscore"] = (volume - volume.rolling(30).mean()) / volume.rolling(30).std().replace(0, np.nan)

    # Group 2: Technical
    tech_feats = pd.DataFrame(index=btc_15m.index)
    tech_feats["rsi_14"] = rsi(close, 14)
    tech_feats["stoch_rsi"] = stochastic_rsi(close, 14, 14)
    tech_feats["macd_hist"] = macd(close)["macd_hist"]
    tech_feats["bb_pct_b"] = bollinger_pct_b(close, 20)
    tech_feats["atr_14"] = atr(high, low, close, 14)
    sw = swing_levels(high, low, lookback=20)
    tech_feats["dist_swing_high"] = sw["dist_swing_high"]
    tech_feats["dist_swing_low"] = sw["dist_swing_low"]
    tech_feats["dist_nearest_level"] = sw["dist_nearest_level"]

    # Group 3: Derivatives (with non-stationarity fixes)
    deriv_feats = pd.DataFrame(index=btc_15m.index)
    if len(deriv_15m) > 0 and "funding" in deriv_15m.columns:
        fund_f = funding_features(deriv_15m["funding"])
        for col in fund_f.columns:
            if col != "funding_cumulative":  # Drop non-stationary
                deriv_feats[col] = fund_f[col]
        oi_f = oi_features(deriv_15m["open_interest"], close)
        for col in oi_f.columns:
            deriv_feats[col] = oi_f[col]
        prem_f = premium_features(deriv_15m["mark_px"], deriv_15m["oracle_px"])
        for col in prem_f.columns:
            if col != "premium_sign_streak":  # Drop non-stationary
                deriv_feats[col] = prem_f[col]
        impact_f = impact_features(deriv_15m["impact_bid_px"], deriv_15m["impact_ask_px"], deriv_15m["mark_px"])
        for col in impact_f.columns:
            deriv_feats[col] = impact_f[col]

    return ohlcv_feats, tech_feats, deriv_feats


def evaluate(X_train, y_train, X_test, y_test, name):
    """Train XGBoost and evaluate."""
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    acc = (y_pred == y_test.values).mean()
    k = int(acc * len(y_test))
    p = 1 - binom.cdf(k-1, len(y_test), 0.5) if k > len(y_test)/2 else 1.0

    # Top 30% confidence
    conf = np.abs(y_prob - 0.5)
    top30 = conf >= np.percentile(conf, 70)
    acc30 = (y_pred[top30] == y_test.values[top30]).mean() if top30.sum() > 50 else acc

    return {"name": name, "accuracy": acc, "p_value": p, "acc_top30": acc30,
            "n_test": len(y_test), "n_features": X_train.shape[1]}


def main():
    print("=== ABLATION STUDY + AUDIT VERIFICATION ===", flush=True)
    print(f"Started: {datetime.now()}", flush=True)

    btc_15m, deriv_15m = load_data()
    ohlcv_f, tech_f, deriv_f = build_feature_groups(btc_15m, deriv_15m)

    close = btc_15m["close"]
    fwd_ret = close.pct_change(1).shift(-1)
    y_all = (np.sign(fwd_ret) > 0).astype(int)

    # Define feature combos
    combos = {
        "OHLCV only": ohlcv_f,
        "Tech only": tech_f,
        "Deriv only (fixed)": deriv_f,
        "OHLCV + Tech": pd.concat([ohlcv_f, tech_f], axis=1),
        "OHLCV + Deriv": pd.concat([ohlcv_f, deriv_f], axis=1),
        "Tech + Deriv": pd.concat([tech_f, deriv_f], axis=1),
        "ALL (fixed)": pd.concat([ohlcv_f, tech_f, deriv_f], axis=1),
    }

    print("\n=== FEATURE GROUP ABLATION ===", flush=True)
    print(f"{'Model':>25} {'Acc':>7} {'p-val':>9} {'Top30%':>7} {'nFeat':>6}", flush=True)
    print("-" * 58, flush=True)

    for name, feats in combos.items():
        clean = feats.dropna()
        valid = clean.index.intersection(y_all.dropna().index)
        X = clean.loc[valid]
        y = y_all.loc[valid]

        train_mask = X.index < "2025-07-01"
        test_mask = X.index >= "2025-07-01"

        if train_mask.sum() < 500 or test_mask.sum() < 200:
            print(f"  {name}: insufficient data", flush=True)
            continue

        r = evaluate(X[train_mask], y[train_mask], X[test_mask], y[test_mask], name)
        print(f"  {name:>23} {r['accuracy']:>7.4f} {r['p_value']:>9.2e} {r['acc_top30']:>7.4f} {r['n_features']:>6}", flush=True)

    # Naive mean-reversion baseline
    print("\n=== NAIVE BASELINES ===", flush=True)
    test_returns = close.pct_change().dropna()
    test_mask_ret = test_returns.index >= "2025-07-01"
    test_ret = test_returns[test_mask_ret]

    # Mean reversion: predict opposite of last return
    mr_signal = -np.sign(test_ret.shift(1)).dropna()
    mr_actual = np.sign(test_ret.loc[mr_signal.index])
    mr_acc = (mr_signal == mr_actual).mean()
    print(f"  Naive mean-reversion: {mr_acc:.4f} (predict opposite of last bar)", flush=True)

    # Momentum: predict same as last return
    mom_signal = np.sign(test_ret.shift(1)).dropna()
    mom_actual = np.sign(test_ret.loc[mom_signal.index])
    mom_acc = (mom_signal == mom_actual).mean()
    print(f"  Naive momentum: {mom_acc:.4f} (predict same as last bar)", flush=True)

    # Monthly breakdown (using the best combo)
    print("\n=== MONTHLY ACCURACY (OHLCV+Tech) ===", flush=True)
    best_feats = pd.concat([ohlcv_f, tech_f], axis=1).dropna()
    valid = best_feats.index.intersection(y_all.dropna().index)
    X_all = best_feats.loc[valid]
    y_full = y_all.loc[valid]
    train_m = X_all.index < "2025-07-01"
    test_m = X_all.index >= "2025-07-01"

    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        tree_method="hist", random_state=42)
    model.fit(X_all[train_m], y_full[train_m])
    y_prob = model.predict_proba(X_all[test_m])[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    y_true = y_full[test_m]

    monthly = pd.DataFrame({"pred": y_pred, "true": y_true.values, "correct": y_pred == y_true.values},
                           index=X_all[test_m].index)
    for month, group in monthly.groupby(monthly.index.to_period("M")):
        acc = group["correct"].mean()
        n = len(group)
        k = int(acc * n)
        p = 1 - binom.cdf(k-1, n, 0.5) if k > n/2 else 1.0
        sig = "*" if p < 0.05 else " "
        print(f"  {month}: {acc:.4f} (n={n:>5}, p={p:.4f}) {sig}", flush=True)

    print(f"\nDone: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
