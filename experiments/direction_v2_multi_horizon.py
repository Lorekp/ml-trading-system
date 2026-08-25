"""
Direction V2 — Multi-Horizon + Multi-Asset Search.

Tests direction prediction at multiple FORWARD horizons across multiple assets.
Key insight: predicting further into the future means larger expected returns per trade,
which helps overcome the fixed 4 bps cost barrier.

Example: 5m bars predicting 3 bars (15m) forward has ~25 bps expected move
vs 8.7 bps for 1-bar prediction. Breakeven drops from 73% to ~58%.
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/ml-data")

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from scipy.stats import binom
from technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels

DATA_DIR = Path("/home/user/ml-data/binance-1m")
OUTPUT_DIR = Path("/home/user/ml-data/v2-results")
TRAIN_CUTOFF = "2025-07-01"
RT_COST_BPS = 4.0

# Assets to test
SYMBOLS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "XRP": "XRPUSDT", "DOGE": "DOGEUSDT", "BNB": "BNBUSDT",
    "LINK": "LINKUSDT", "AVAX": "AVAXUSDT", "SUI": "SUIUSDT",
    "ADA": "ADAUSDT",
}

# Timeframes and forward horizons to test
# Format: (bar_tf, bar_resample, forward_bars, effective_horizon_label)
CONFIGS = [
    # 5m bars, predicting N bars forward
    ("5m", "5min", 1, "5m→5m"),
    ("5m", "5min", 3, "5m→15m"),
    ("5m", "5min", 6, "5m→30m"),
    ("5m", "5min", 12, "5m→1H"),
    # 15m bars, predicting N bars forward
    ("15m", "15min", 1, "15m→15m"),
    ("15m", "15min", 2, "15m→30m"),
    ("15m", "15min", 4, "15m→1H"),
    ("15m", "15min", 8, "15m→2H"),
    # 30m bars, predicting N bars forward
    ("30m", "30min", 1, "30m→30m"),
    ("30m", "30min", 2, "30m→1H"),
    ("30m", "30min", 4, "30m→2H"),
]

# Best params from grid search per timeframe
BEST_PARAMS = {
    "5min": dict(max_depth=3, learning_rate=0.05, n_estimators=50,
                 min_child_weight=100, subsample=0.7, colsample_bytree=0.6,
                 objective="binary:logistic", eval_metric="logloss",
                 tree_method="hist", random_state=42),
    "15min": dict(max_depth=3, learning_rate=0.05, n_estimators=50,
                  min_child_weight=100, subsample=0.7, colsample_bytree=0.6,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", random_state=42),
    "30min": dict(max_depth=4, learning_rate=0.05, n_estimators=200,
                  min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", random_state=42),
}


def load_ohlcv(symbol, tf_resample):
    dir_name = SYMBOLS.get(symbol)
    if dir_name is None:
        raise ValueError(f"Unknown: {symbol}")
    data_dir = DATA_DIR / dir_name
    if not data_dir.exists():
        raise FileNotFoundError(f"No data: {data_dir}")
    dfs = []
    for f in sorted(data_dir.glob("*.csv")):
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
    return combined.resample(tf_resample).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])


def build_features(ohlcv):
    close, high, low, volume = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]
    train_mask = ohlcv.index < TRAIN_CUTOFF
    features = pd.DataFrame(index=ohlcv.index)
    for lb in [1, 3, 7, 14]:
        ret = close.pct_change(lb)
        lo = ret[train_mask].quantile(0.01)
        hi = ret[train_mask].quantile(0.99)
        features[f"ret_{lb}"] = ret.clip(lo, hi)
    features["bar_range"] = (high - low) / close
    features["vol_zscore"] = (
        (volume - volume.rolling(30).mean()) /
        volume.rolling(30).std().replace(0, np.nan)
    )
    features["rsi_14"] = rsi(close, 14)
    features["stoch_rsi"] = stochastic_rsi(close, 14, 14)
    features["macd_hist"] = macd(close)["macd_hist"]
    features["bb_pct_b"] = bollinger_pct_b(close, 20)
    features["atr_14"] = atr(high, low, close, 14)
    sw = swing_levels(high, low, lookback=20)
    features["dist_swing_high"] = sw["dist_swing_high"]
    features["dist_swing_low"] = sw["dist_swing_low"]
    features["dist_nearest_level"] = sw["dist_nearest_level"]
    return features


def run_config(symbol, bar_tf, bar_resample, fwd_bars, label):
    """Test one (symbol, timeframe, horizon) combination."""
    try:
        ohlcv = load_ohlcv(symbol, bar_resample)
    except (FileNotFoundError, ValueError) as e:
        return None

    features = build_features(ohlcv)
    close = ohlcv["close"]

    # Forward N-bar return as label
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)
    features_clean = features.dropna()
    y_all = fwd_ret.reindex(features_clean.index).dropna()
    valid_idx = features_clean.index.intersection(y_all.index)
    X = features_clean.loc[valid_idx]
    y = (y_all.loc[valid_idx] > 0).astype(int)

    train_mask = X.index < TRAIN_CUTOFF
    test_mask = X.index >= TRAIN_CUTOFF
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(X_train) < 200 or len(X_test) < 100:
        return None

    # Mean abs return for N-bar forward (test period)
    test_fwd = fwd_ret[fwd_ret.index >= TRAIN_CUTOFF].dropna()
    mean_abs_ret_bps = test_fwd.abs().mean() * 10000
    breakeven = 0.5 + RT_COST_BPS / (2 * mean_abs_ret_bps) if mean_abs_ret_bps > 0 else 1.0

    # Use best params for this timeframe
    params = BEST_PARAMS.get(bar_resample, BEST_PARAMS["15min"])
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    acc = float((y_pred == y_test.values).mean())
    k = int((y_pred == y_test.values).sum())
    n_test = len(y_test)
    p_val = 1 - binom.cdf(k - 1, n_test, 0.5) if k > n_test / 2 else 1.0

    # Filtered
    conf = np.abs(y_prob - 0.5)
    filtered = {}
    for fl, pt in [("top30", 70), ("top20", 80), ("top10", 90)]:
        mask = conf >= np.percentile(conf, pt)
        n_f = int(mask.sum())
        if n_f >= 20:
            a = float((y_pred[mask] == y_test.values[mask]).mean())
            edge = (a - 0.5) * 2 * mean_abs_ret_bps
            net = edge - RT_COST_BPS
            filtered[fl] = {"acc": round(a, 4), "n": n_f, "net": round(net, 2)}
        else:
            filtered[fl] = {"acc": None, "n": n_f, "net": None}

    edge_all = (acc - 0.5) * 2 * mean_abs_ret_bps
    net_all = edge_all - RT_COST_BPS

    return {
        "symbol": symbol, "bar_tf": bar_tf, "fwd_bars": fwd_bars, "label": label,
        "n_train": len(X_train), "n_test": n_test,
        "mean_abs_ret_bps": round(mean_abs_ret_bps, 1),
        "breakeven": round(breakeven, 4),
        "acc_all": round(acc, 4), "p_value": float(p_val),
        "net_all": round(net_all, 2),
        "filtered": filtered,
    }


def main():
    print("=" * 110, flush=True)
    print("DIRECTION V2 — MULTI-HORIZON + MULTI-ASSET SEARCH", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Configs: {len(CONFIGS)} (timeframe×horizon), Symbols: {len(SYMBOLS)}", flush=True)
    print("=" * 110, flush=True)

    all_results = []

    # Part 1: Multi-horizon on BTC
    print("\n=== PART 1: BTC MULTI-HORIZON ===", flush=True)
    print(f"{'Config':>12} {'MAbsRet':>8} {'BEAcc':>7} {'AccAll':>7} {'NetAll':>7} | "
          f"{'Acc30%':>7} {'Net30%':>7} {'Acc10%':>7} {'Net10%':>7} | {'p-val':>9}", flush=True)
    print("-" * 110, flush=True)

    for bar_tf, bar_resample, fwd_bars, label in CONFIGS:
        r = run_config("BTC", bar_tf, bar_resample, fwd_bars, label)
        if r is None:
            continue
        all_results.append(r)

        f30 = r["filtered"].get("top30", {})
        f10 = r["filtered"].get("top10", {})
        acc30 = f"{f30['acc']:.4f}" if f30.get("acc") else "    N/A"
        net30 = f"{f30['net']:+.2f}" if f30.get("net") is not None else "    N/A"
        acc10 = f"{f10['acc']:.4f}" if f10.get("acc") else "    N/A"
        net10 = f"{f10['net']:+.2f}" if f10.get("net") is not None else "    N/A"
        p_str = f"{r['p_value']:.2e}" if r['p_value'] < 0.01 else f"{r['p_value']:.4f}"
        print(f"{label:>12} {r['mean_abs_ret_bps']:>7.1f}bp {r['breakeven']*100:>6.2f}% "
              f"{r['acc_all']:>6.4f} {r['net_all']:>+6.2f} | "
              f"{acc30:>7} {net30:>7} {acc10:>7} {net10:>7} | {p_str:>9}", flush=True)

    # Part 2: Best config across all 10 assets
    # Find the most promising BTC config
    btc_results = [r for r in all_results if r["symbol"] == "BTC"]
    best_btc = None
    best_net = -999
    for r in btc_results:
        for fl in ["top30", "top20", "top10"]:
            f = r["filtered"].get(fl, {})
            if f.get("net") is not None and f["net"] > best_net:
                best_net = f["net"]
                best_btc = r

    if best_btc:
        best_config = (best_btc["bar_tf"],
                       {"5m": "5min", "15m": "15min", "30m": "30min"}[best_btc["bar_tf"]],
                       best_btc["fwd_bars"],
                       best_btc["label"])
        print(f"\n=== PART 2: BEST CONFIG ({best_btc['label']}) ACROSS 10 ASSETS ===", flush=True)
    else:
        # Default to 15m→1H
        best_config = ("15m", "15min", 4, "15m→1H")
        print(f"\n=== PART 2: 15m→1H ACROSS 10 ASSETS (default) ===", flush=True)

    bar_tf, bar_resample, fwd_bars, label = best_config

    print(f"{'Symbol':>6} {'MAbsRet':>8} {'BEAcc':>7} {'AccAll':>7} {'NetAll':>7} | "
          f"{'Acc30%':>7} {'Net30%':>7} | {'p-val':>9} {'nTest':>6}", flush=True)
    print("-" * 90, flush=True)

    for symbol in SYMBOLS.keys():
        r = run_config(symbol, bar_tf, bar_resample, fwd_bars, f"{label}-{symbol}")
        if r is None:
            print(f"{symbol:>6} — insufficient data", flush=True)
            continue
        all_results.append(r)

        f30 = r["filtered"].get("top30", {})
        acc30 = f"{f30['acc']:.4f}" if f30.get("acc") else "    N/A"
        net30 = f"{f30['net']:+.2f}" if f30.get("net") is not None else "    N/A"
        p_str = f"{r['p_value']:.2e}" if r['p_value'] < 0.01 else f"{r['p_value']:.4f}"
        prof = " ***" if r["net_all"] > 0 or (f30.get("net") is not None and f30["net"] > 0) else ""
        print(f"{symbol:>6} {r['mean_abs_ret_bps']:>7.1f}bp {r['breakeven']*100:>6.2f}% "
              f"{r['acc_all']:>6.4f} {r['net_all']:>+6.2f} | "
              f"{acc30:>7} {net30:>7} | {p_str:>9} {r['n_test']:>6}{prof}", flush=True)

    # Summary
    print("\n=== PROFITABILITY CHECK ===", flush=True)
    any_prof = False
    for r in all_results:
        for key in ["net_all"]:
            if r.get(key, -999) > 0:
                print(f"  PROFITABLE: {r['symbol']} {r['label']} ALL: +{r[key]:.2f} bps (acc={r['acc_all']:.4f})", flush=True)
                any_prof = True
        for fl in ["top30", "top20", "top10"]:
            f = r["filtered"].get(fl, {})
            if f.get("net") is not None and f["net"] > 0:
                print(f"  PROFITABLE: {r['symbol']} {r['label']} {fl}: +{f['net']:.2f} bps (acc={f['acc']:.4f}, n={f['n']})", flush=True)
                any_prof = True
    if not any_prof:
        print("  No profitable configuration found.", flush=True)

    with open(OUTPUT_DIR / "direction_v2_multi_horizon.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": all_results}, f, indent=2, default=str)
    print(f"\nCompleted: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
