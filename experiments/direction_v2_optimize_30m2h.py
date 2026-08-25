"""
Direction V2 — Optimize the winning config: 30m bars, 4-bar (2H) forward prediction.
Per-asset hyperparameter search. Reports at 4, 3, and 2 bps RT cost.
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
FWD_BARS = 4  # 4 x 30m = 2H

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
           "XRP": "XRPUSDT", "DOGE": "DOGEUSDT", "BNB": "BNBUSDT",
           "LINK": "LINKUSDT", "AVAX": "AVAXUSDT", "SUI": "SUIUSDT",
           "ADA": "ADAUSDT"}

CUSUM_MULTS = [1.5, 2.0, 3.0, 5.0]
N_RANDOM = 40

PARAM_GRID = {
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [50, 100, 200, 300],
    "min_child_weight": [10, 20, 50, 100],
    "subsample": [0.6, 0.7, 0.8],
    "colsample_bytree": [0.6, 0.7, 0.8],
}


def load_ohlcv(symbol):
    d = DATA_DIR / SYMBOLS[symbol]
    dfs = []
    for f in sorted(d.glob("*.csv")):
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
    return combined.resample("30min").agg(
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
    features["vol_zscore"] = (volume - volume.rolling(30).mean()) / volume.rolling(30).std().replace(0, np.nan)
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


def cusum_filter(raw, threshold):
    events, sp, sn = [], 0.0, 0.0
    diff = raw.diff().dropna()
    for t in diff.index:
        h = threshold.get(t, None) if isinstance(threshold, pd.Series) else float(threshold)
        if h is None or np.isnan(h):
            continue
        sp = max(0.0, sp + diff.loc[t])
        sn = min(0.0, sn + diff.loc[t])
        if sn < -h:
            sn = 0.0
            events.append(t)
        elif sp > h:
            sp = 0.0
            events.append(t)
    return pd.DatetimeIndex(events)


def get_daily_vol(close, span=50):
    return close.pct_change().dropna().ewm(span=span).std()


def search_asset(symbol):
    print(f"\n  {symbol}:", flush=True)
    ohlcv = load_ohlcv(symbol)
    features = build_features(ohlcv)
    close = ohlcv["close"]
    vol = get_daily_vol(close, span=50)

    # Mean abs return for 2H forward (test period)
    fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)
    test_fwd = fwd_ret[fwd_ret.index >= TRAIN_CUTOFF].dropna()
    mar = test_fwd.abs().mean() * 10000

    rng = np.random.RandomState(42)
    best_cv = 0.0
    best_state = None

    for cm in CUSUM_MULTS:
        events = cusum_filter(close, vol * close * cm)
        fc = features.dropna()
        valid = events.intersection(fc.index)
        y_all = fwd_ret.reindex(valid).dropna()
        valid = y_all.index
        y_bin = (y_all > 0).astype(int)

        tr_idx = valid[valid < TRAIN_CUTOFF]
        te_idx = valid[valid >= TRAIN_CUTOFF]
        if len(tr_idx) < 100 or len(te_idx) < 50:
            continue

        X_tr = fc.loc[tr_idx]
        y_tr = y_bin.loc[tr_idx]
        X_te = fc.loc[te_idx]
        y_te = y_bin.loc[te_idx]

        for _ in range(N_RANDOM):
            params = {
                "max_depth": int(rng.choice(PARAM_GRID["max_depth"])),
                "learning_rate": float(rng.choice(PARAM_GRID["learning_rate"])),
                "n_estimators": int(rng.choice(PARAM_GRID["n_estimators"])),
                "min_child_weight": int(rng.choice(PARAM_GRID["min_child_weight"])),
                "subsample": float(rng.choice(PARAM_GRID["subsample"])),
                "colsample_bytree": float(rng.choice(PARAM_GRID["colsample_bytree"])),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": 42,
            }
            # 5-fold time-series CV
            n = len(X_tr)
            fold_accs = []
            for i in range(5):
                ts = int(n * i / 5)
                te_end = int(n * (i + 1) / 5)
                ti = list(range(0, ts)) + list(range(te_end, n))
                if len(ti) < 50 or (te_end - ts) < 20:
                    continue
                m = xgb.XGBClassifier(**params)
                m.fit(X_tr.iloc[ti], y_tr.iloc[ti])
                p = m.predict(X_tr.iloc[ts:te_end])
                fold_accs.append((p == y_tr.iloc[ts:te_end].values).mean())

            cv = np.mean(fold_accs) if fold_accs else 0
            if cv > best_cv:
                best_cv = cv
                best_state = {
                    "params": params, "cm": cm,
                    "X_tr": X_tr, "y_tr": y_tr,
                    "X_te": X_te, "y_te": y_te,
                }

    if best_state is None:
        print(f"    No valid config found", flush=True)
        return None

    # Evaluate best on held-out test
    m = xgb.XGBClassifier(**best_state["params"])
    m.fit(best_state["X_tr"], best_state["y_tr"])
    yp = m.predict_proba(best_state["X_te"])[:, 1]
    ypred = (yp > 0.5).astype(int)
    yt = best_state["y_te"].values
    n_test = len(yt)

    acc_all = float((ypred == yt).mean())
    conf = np.abs(yp - 0.5)

    result = {
        "symbol": symbol, "mar": round(mar, 1), "cv": round(best_cv, 4),
        "cm": best_state["cm"], "n_test": n_test,
        "params": {k: (int(v) if isinstance(v, (np.integer,)) else
                       float(v) if isinstance(v, (np.floating,)) else v)
                   for k, v in best_state["params"].items()
                   if k not in ("objective", "eval_metric", "tree_method", "random_state")},
    }

    for label, pt in [("all", 0), ("top30", 70), ("top20", 80), ("top10", 90)]:
        mask = conf >= np.percentile(conf, pt) if pt > 0 else np.ones(len(conf), dtype=bool)
        n = int(mask.sum())
        if n < 10:
            result[label] = {"acc": None, "n": n}
            continue
        a = float((ypred[mask] == yt[mask]).mean())
        k = int((ypred[mask] == yt[mask]).sum())
        p_val = 1 - binom.cdf(k - 1, n, 0.5) if k > n / 2 else 1.0
        result[label] = {
            "acc": round(a, 4), "n": n, "p": float(p_val),
            "net4": round((a - 0.5) * 2 * mar - 4.0, 2),
            "net3": round((a - 0.5) * 2 * mar - 3.0, 2),
            "net2": round((a - 0.5) * 2 * mar - 2.0, 2),
        }

    r30 = result.get("top30", {})
    r10 = result.get("top10", {})
    print(f"    MAR={mar:.0f}bp cv={best_cv:.4f} cm={best_state['cm']} n={n_test}", flush=True)
    print(f"    all:   acc={result['all'].get('acc','N/A')}", flush=True)
    print(f"    top30: acc={r30.get('acc','N/A')} net@3bp={r30.get('net3','N/A')}", flush=True)
    print(f"    top10: acc={r10.get('acc','N/A')} net@3bp={r10.get('net3','N/A')}", flush=True)
    return result


def main():
    print("=" * 100, flush=True)
    print("OPTIMIZE WINNER: 30m bars -> 2H forward, 10 assets, per-asset grid search", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Search: {N_RANDOM} XGB params x {len(CUSUM_MULTS)} CUSUM mults per asset", flush=True)
    print(f"Reporting at 4, 3, and 2 bps RT cost", flush=True)
    print("=" * 100, flush=True)

    results = []
    for sym in SYMBOLS:
        try:
            r = search_asset(sym)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  {sym}: ERROR {e}", flush=True)

    # Summary table
    print("\n" + "=" * 120, flush=True)
    print("OPTIMIZED 30m -> 2H RESULTS", flush=True)
    print("=" * 120, flush=True)

    header = (f"{'Sym':<5} {'MAR':>5} {'CV':>6} {'CM':>3} {'nTe':>5} | "
              f"{'AccAll':>7} {'Acc30':>7} {'Acc10':>7} | "
              f"{'N3_all':>7} {'N3_30':>7} {'N3_10':>7} | "
              f"{'N2_all':>7} {'N2_30':>7} {'N2_10':>7}")
    print(header, flush=True)
    print("-" * 120, flush=True)

    for r in results:
        def g(filt, key):
            v = r.get(filt, {}).get(key)
            if v is None:
                return "    N/A"
            if key == "acc":
                return f"{v:.4f}"
            return f"{v:+.2f}"

        print(f"{r['symbol']:<5} {r['mar']:>5.0f} {r['cv']:>5.4f} {r['cm']:>3.1f} {r['n_test']:>5} | "
              f"{g('all','acc'):>7} {g('top30','acc'):>7} {g('top10','acc'):>7} | "
              f"{g('all','net3'):>7} {g('top30','net3'):>7} {g('top10','net3'):>7} | "
              f"{g('all','net2'):>7} {g('top30','net2'):>7} {g('top10','net2'):>7}", flush=True)

    # Profitable summary
    print("\nPROFITABLE AT 3 bps (Hyperliquid maker):", flush=True)
    count = 0
    for r in results:
        for filt in ["all", "top30", "top20", "top10"]:
            net3 = r.get(filt, {}).get("net3")
            acc = r.get(filt, {}).get("acc")
            n = r.get(filt, {}).get("n", 0)
            if net3 is not None and net3 > 0:
                print(f"  {r['symbol']} {filt}: acc={acc:.4f} net=+{net3:.2f}bps n={n}", flush=True)
                count += 1
    print(f"Total: {count} profitable configs at 3 bps", flush=True)

    print("\nPROFITABLE AT 2 bps (with referral/volume discount):", flush=True)
    count2 = 0
    for r in results:
        for filt in ["all", "top30", "top20", "top10"]:
            net2 = r.get(filt, {}).get("net2")
            acc = r.get(filt, {}).get("acc")
            n = r.get(filt, {}).get("n", 0)
            if net2 is not None and net2 > 0:
                print(f"  {r['symbol']} {filt}: acc={acc:.4f} net=+{net2:.2f}bps n={n}", flush=True)
                count2 += 1
    print(f"Total: {count2} profitable configs at 2 bps", flush=True)

    with open(OUTPUT_DIR / "direction_v2_optimized.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2, default=str)
    print(f"\nCompleted: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
