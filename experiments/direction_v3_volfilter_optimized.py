"""
Direction V3 — Vol Filter + Optimized Params + Multiple Horizons.

Combines the two best improvements:
  - Vol filter (A3): only trade high-vol bars (+3 bps improvement)
  - Per-asset optimized XGBoost params (grid search: +1-2 bps)
  - Test at 30m->2H, 30m->3H, 30m->4H (longer horizons = larger returns)

Walk-forward validation (monthly retrain) for honest results.
Core 3 assets: SUI, XRP, BTC (best from portfolio sim).
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
OUTPUT_DIR = Path("/home/user/ml-data/v3-results")

ASSETS = ["SUI", "XRP", "BTC"]
SYMBOLS = {"BTC": "BTCUSDT", "XRP": "XRPUSDT", "SUI": "SUIUSDT"}

HORIZONS = [
    (4, "2H"),   # 30m x 4 = 2H (proven)
    (6, "3H"),   # 30m x 6 = 3H
    (8, "4H"),   # 30m x 8 = 4H
]

TEST_MONTHS = pd.period_range("2025-01", "2026-02", freq="M")

# Default params (baseline)
DEFAULT_PARAMS = dict(max_depth=3, learning_rate=0.05, n_estimators=100,
                      min_child_weight=50, subsample=0.7, colsample_bytree=0.7,
                      objective="binary:logistic", eval_metric="logloss",
                      tree_method="hist", random_state=42)

# Optimized params per asset (from v2 grid search)
OPTIMIZED_PARAMS = {
    "BTC": dict(max_depth=3, learning_rate=0.05, n_estimators=50,
                min_child_weight=100, subsample=0.7, colsample_bytree=0.6,
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=42),
    "XRP": dict(max_depth=4, learning_rate=0.03, n_estimators=200,
                min_child_weight=50, subsample=0.6, colsample_bytree=0.7,
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=42),
    "SUI": dict(max_depth=3, learning_rate=0.05, n_estimators=100,
                min_child_weight=100, subsample=0.7, colsample_bytree=0.6,
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=42),
}

VOL_PERCENTILE = 50
CONF_PERCENTILE = 70  # top 30%


def load_ohlcv(symbol, tf="30min"):
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
    return combined.resample(tf).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])


def build_features(ohlcv, train_end):
    close, high, low, volume = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]
    train_mask = ohlcv.index < train_end
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


def get_daily_vol(close, span=50):
    return close.pct_change().dropna().ewm(span=span).std()


def walkforward(symbol, ohlcv, fwd_bars, horizon_label, params):
    """Walk-forward with vol filter and given params. Returns monthly results."""
    close = ohlcv["close"]
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)
    vol = get_daily_vol(close, span=50)

    monthly = []
    all_trades = []

    for month in TEST_MONTHS:
        month_start = pd.Timestamp(month.start_time, tz="UTC")
        month_end = pd.Timestamp(month.end_time, tz="UTC")

        features = build_features(ohlcv, train_end=month_start)
        fc = features.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        train_idx = valid[valid < month_start]
        test_idx = valid[(valid >= month_start) & (valid <= month_end)]

        if len(train_idx) < 500 or len(test_idx) < 30:
            continue

        X_train = fc.loc[train_idx]
        y_train = (y_all.loc[train_idx] > 0).astype(int)

        # Vol filter on test set
        train_vol = vol.loc[train_idx].dropna()
        vol_threshold = train_vol.quantile(VOL_PERCENTILE / 100)
        test_vol = vol.reindex(test_idx)
        hv_mask = (test_vol > vol_threshold).fillna(False)
        hv_test = test_idx[hv_mask.values]

        if len(hv_test) < 20:
            hv_test = test_idx  # fallback

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        X_test = fc.loc[hv_test]
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        y_true = (y_all.loc[hv_test] > 0).astype(int).values

        # Confidence filter
        conf = np.abs(y_prob - 0.5)
        if len(conf) > 10:
            conf_thresh = np.percentile(conf, CONF_PERCENTILE)
            trade_mask = conf >= conf_thresh
        else:
            trade_mask = np.ones(len(conf), dtype=bool)

        n_trades = int(trade_mask.sum())
        if n_trades < 5:
            continue

        acc = float((y_pred[trade_mask] == y_true[trade_mask]).mean())
        mar = fwd_ret.loc[hv_test[trade_mask]].abs().mean() * 10000

        # Per-trade returns for Sharpe
        direction = np.where(y_prob[trade_mask] > 0.5, 1.0, -1.0)
        actual = fwd_ret.loc[hv_test[trade_mask]].values
        trade_rets = direction * actual
        trade_rets = trade_rets[np.isfinite(trade_rets)]

        monthly.append({
            "month": str(month), "n_trades": n_trades,
            "acc": round(acc, 4), "mar": round(mar, 1),
            "mean_ret_bps": round(float(np.mean(trade_rets)) * 10000, 2),
        })
        all_trades.extend(trade_rets.tolist())

    return monthly, np.array(all_trades)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100, flush=True)
    print("V3 — VOL FILTER + OPTIMIZED PARAMS + MULTIPLE HORIZONS (Walk-Forward)", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Assets: {ASSETS}, Horizons: {[h[1] for h in HORIZONS]}", flush=True)
    print(f"Vol filter: top {100-VOL_PERCENTILE}% vol, Confidence: top {100-CONF_PERCENTILE}%", flush=True)
    print("=" * 100, flush=True)

    results = []

    for symbol in ASSETS:
        ohlcv = load_ohlcv(symbol, "30min")
        print(f"\n{'='*70}", flush=True)
        print(f"  {symbol} ({len(ohlcv)} bars)", flush=True)
        print(f"{'='*70}", flush=True)

        opt_params = OPTIMIZED_PARAMS.get(symbol, DEFAULT_PARAMS)

        for fwd_bars, hz_label in HORIZONS:
            for param_label, params in [("default", DEFAULT_PARAMS), ("optimized", opt_params)]:
                monthly, trades = walkforward(symbol, ohlcv, fwd_bars, hz_label, params)

                if len(trades) == 0:
                    continue

                mean_ret = np.mean(trades) * 10000
                std_ret = np.std(trades) * 10000 if len(trades) > 1 else 1
                n = len(trades)
                accs = [m["acc"] for m in monthly]
                mean_acc = np.mean(accs) if accs else 0
                months_pos = sum(1 for a in accs if a > 0.5)

                # Sharpe (approximate annualized)
                trading_days = len(monthly) * 30  # rough
                trades_per_day = n / trading_days if trading_days > 0 else 1
                sharpe_0 = (np.mean(trades) / np.std(trades) * np.sqrt(trades_per_day * 365.25)) if np.std(trades) > 0 else 0
                net_3 = mean_ret - 3.0

                tag = f"{symbol} {hz_label} {param_label}"
                print(f"  {tag:<25} acc={mean_acc:.4f} mo>{months_pos}/{len(accs)} "
                      f"ret={mean_ret:+.1f}bp n={n} Sh={sharpe_0:.2f} net@3={net_3:+.1f}bp", flush=True)

                results.append({
                    "symbol": symbol, "horizon": hz_label, "fwd_bars": fwd_bars,
                    "params": param_label, "n_trades": n,
                    "mean_acc": round(mean_acc, 4),
                    "months_above_50": months_pos, "months_total": len(accs),
                    "mean_ret_bps": round(mean_ret, 2),
                    "sharpe": round(sharpe_0, 2),
                    "net_0_bps": round(mean_ret, 2),
                    "net_3_bps": round(net_3, 2),
                })

    # Summary
    print(f"\n{'='*110}", flush=True)
    print("SUMMARY: VOL FILTER + PARAMS + HORIZONS (walk-forward)", flush=True)
    print(f"{'='*110}", flush=True)
    print(f"{'Config':<30} {'Acc':>6} {'Mo>50':>6} {'Trades':>7} {'Ret':>8} {'Sharpe':>7} {'Net@0':>7} {'Net@3':>7}", flush=True)
    print("-" * 110, flush=True)

    for r in sorted(results, key=lambda x: -x["net_0_bps"]):
        tag = f"{r['symbol']} {r['horizon']} {r['params']}"
        print(f"{tag:<30} {r['mean_acc']:>5.4f} {r['months_above_50']:>2}/{r['months_total']:<3} "
              f"{r['n_trades']:>7} {r['mean_ret_bps']:>+7.1f}bp {r['sharpe']:>7.2f} "
              f"{r['net_0_bps']:>+6.1f}bp {r['net_3_bps']:>+6.1f}bp", flush=True)

    with open(OUTPUT_DIR / "direction_v3_volfilter_optimized.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)
    print(f"\nCompleted: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
