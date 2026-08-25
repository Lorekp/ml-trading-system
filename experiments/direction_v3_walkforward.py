"""HISTORICAL EXPERIMENT — its go-live verdict is not current evidence.

Direction V3 walk-forward study retained for provenance.

It evaluates multiple time periods rather than one held-out window, but does
not by itself validate deployment.

Method:
  - Expanding window retraining: train on all data up to month M, test on month M+1
  - Test each month from Jul 2025 to Feb 2026 (8 months)
  - For each month: train fresh model, predict, evaluate
  - Report: per-month accuracy, edge, and aggregate Sharpe
  - Compare: with and without vol filter (A3)
  - Test on top 3 assets: ETH, BNB, XRP (most profitable from v3)

If monthly accuracy is consistently >50% across 6+ months = REAL signal.
If it's noisy (some months 45%, others 55%) = fragile, not production-ready.
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
FWD_BARS = 4  # 30m x 4 = 2H forward

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
           "BNB": "BNBUSDT", "LINK": "LINKUSDT", "SUI": "SUIUSDT"}

# Walk-forward: train on everything before month M, test on month M
# Data range: Mar 2024 - Feb 2026
# Training starts: need at least 6 months of training data
# First test month: Sep 2024 (trained on Mar-Aug 2024)
# Last test month: Feb 2026
TEST_MONTHS = pd.period_range("2024-09", "2026-02", freq="M")

# Conservative XGB params (from grid search)
XGB_PARAMS = dict(max_depth=3, learning_rate=0.05, n_estimators=100,
                  min_child_weight=50, subsample=0.7, colsample_bytree=0.7,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", random_state=42)


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
    """Build features with winsorization on train data only."""
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


def walk_forward_asset(symbol, ohlcv):
    """Run walk-forward for one asset. Returns per-month results."""
    print(f"\n{'='*70}", flush=True)
    print(f"  WALK-FORWARD: {symbol} (30m -> 2H)", flush=True)
    print(f"{'='*70}", flush=True)

    close = ohlcv["close"]
    fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)
    vol = get_daily_vol(close, span=50)

    monthly_results = []
    monthly_results_volfilter = []

    for month in TEST_MONTHS:
        # Month boundaries
        month_start = pd.Timestamp(month.start_time, tz="UTC")
        month_end = pd.Timestamp(month.end_time, tz="UTC")

        # Build features with winsorization on train data (everything before this month)
        features = build_features(ohlcv, train_end=month_start)
        fc = features.dropna()

        # Labels
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        # Train: everything before this month
        train_idx = valid[valid < month_start]
        # Test: this month only
        test_idx = valid[(valid >= month_start) & (valid <= month_end)]

        if len(train_idx) < 500 or len(test_idx) < 50:
            continue

        X_train = fc.loc[train_idx]
        y_train = (y_all.loc[train_idx] > 0).astype(int)
        X_test = fc.loc[test_idx]
        y_test = (y_all.loc[test_idx] > 0).astype(int)

        # Train fresh model for this month
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)

        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        # Accuracy (all predictions)
        acc_all = float((y_pred == y_test.values).mean())
        n_test = len(y_test)

        # Confidence filtered (top 30%)
        conf = np.abs(y_prob - 0.5)
        top30_mask = conf >= np.percentile(conf, 70)
        n_top30 = int(top30_mask.sum())
        acc_top30 = float((y_pred[top30_mask] == y_test.values[top30_mask]).mean()) if n_top30 >= 10 else None

        # Mean abs return this month
        test_fwd = fwd_ret.loc[test_idx].dropna()
        mar = test_fwd.abs().mean() * 10000 if len(test_fwd) > 0 else 0

        # Edge and net at various costs
        edge_all = (acc_all - 0.5) * 2 * mar
        edge_top30 = (acc_top30 - 0.5) * 2 * mar if acc_top30 else None

        # Per-trade return simulation (for Sharpe calc)
        pred_dir = np.where(y_prob > 0.5, 1.0, -1.0)
        actual_fwd = fwd_ret.loc[test_idx].values[:len(pred_dir)]
        trade_returns = pred_dir * actual_fwd

        result = {
            "month": str(month),
            "n_train": len(train_idx),
            "n_test": n_test,
            "acc_all": round(acc_all, 4),
            "acc_top30": round(acc_top30, 4) if acc_top30 else None,
            "n_top30": n_top30,
            "mar": round(mar, 1),
            "edge_all": round(edge_all, 2),
            "edge_top30": round(edge_top30, 2) if edge_top30 else None,
            "mean_trade_return": round(float(np.nanmean(trade_returns)) * 10000, 2),
        }
        monthly_results.append(result)

        # --- VOL FILTER version ---
        # Only predict on bars where realized vol > train median
        train_vol = vol.loc[train_idx]
        vol_threshold = train_vol.quantile(0.50)
        test_vol = vol.reindex(test_idx)
        hv_mask = test_vol > vol_threshold
        hv_test = test_idx[hv_mask.values]

        if len(hv_test) >= 30:
            X_test_hv = fc.loc[hv_test]
            y_test_hv = (y_all.loc[hv_test] > 0).astype(int)
            yp_hv = model.predict_proba(X_test_hv)[:, 1]
            ypred_hv = (yp_hv > 0.5).astype(int)
            acc_hv = float((ypred_hv == y_test_hv.values).mean())

            conf_hv = np.abs(yp_hv - 0.5)
            top30_hv = conf_hv >= np.percentile(conf_hv, 70) if len(conf_hv) > 10 else np.zeros(len(conf_hv), dtype=bool)
            n_top30_hv = int(top30_hv.sum())
            acc_top30_hv = float((ypred_hv[top30_hv] == y_test_hv.values[top30_hv]).mean()) if n_top30_hv >= 10 else None

            mar_hv = fwd_ret.loc[hv_test].abs().mean() * 10000

            pred_dir_hv = np.where(yp_hv > 0.5, 1.0, -1.0)
            actual_hv = fwd_ret.loc[hv_test].values[:len(pred_dir_hv)]
            trade_returns_hv = pred_dir_hv * actual_hv

            vf_result = {
                "month": str(month),
                "n_test": len(hv_test),
                "acc_all": round(acc_hv, 4),
                "acc_top30": round(acc_top30_hv, 4) if acc_top30_hv else None,
                "n_top30": n_top30_hv,
                "mar": round(mar_hv, 1),
                "mean_trade_return": round(float(np.nanmean(trade_returns_hv)) * 10000, 2),
            }
        else:
            vf_result = {"month": str(month), "n_test": len(hv_test), "acc_all": None}
        monthly_results_volfilter.append(vf_result)

        # Print progress
        a30 = f"{acc_top30:.4f}" if acc_top30 else "  N/A"
        vf_acc = f"{vf_result['acc_all']:.4f}" if vf_result.get('acc_all') else "  N/A"
        print(f"  {month}: acc={acc_all:.4f} top30={a30} MAR={mar:.0f}bp n={n_test:>5} "
              f"| vol-filt: acc={vf_acc} n={vf_result['n_test']}", flush=True)

    return monthly_results, monthly_results_volfilter


def compute_aggregate_stats(monthly_results, cost_bps=0):
    """Compute aggregate walk-forward statistics."""
    accs = [r["acc_all"] for r in monthly_results if r.get("acc_all") is not None]
    top30s = [r["acc_top30"] for r in monthly_results if r.get("acc_top30") is not None]
    mars = [r["mar"] for r in monthly_results if r.get("mar")]
    trade_rets = [r["mean_trade_return"] for r in monthly_results if r.get("mean_trade_return") is not None]

    if not accs:
        return {}

    mean_acc = np.mean(accs)
    mean_top30 = np.mean(top30s) if top30s else None
    mean_mar = np.mean(mars) if mars else 0
    months_above_50 = sum(1 for a in accs if a > 0.5)
    months_top30_above_50 = sum(1 for a in top30s if a > 0.5) if top30s else 0

    # Aggregate edge
    mean_edge_all = (mean_acc - 0.5) * 2 * mean_mar - cost_bps
    mean_edge_top30 = (mean_top30 - 0.5) * 2 * mean_mar - cost_bps if mean_top30 else None

    # Sharpe from per-month trade returns
    if trade_rets:
        net_rets = [r - cost_bps for r in trade_rets]
        sharpe_monthly = np.mean(net_rets) / np.std(net_rets) if np.std(net_rets) > 0 else 0
        sharpe_annual = sharpe_monthly * np.sqrt(12)
    else:
        sharpe_annual = 0

    return {
        "mean_acc": round(mean_acc, 4),
        "mean_top30": round(mean_top30, 4) if mean_top30 else None,
        "months_above_50pct": months_above_50,
        "months_total": len(accs),
        "months_top30_above_50": months_top30_above_50,
        "mean_mar": round(mean_mar, 1),
        "mean_edge_bps": round(mean_edge_all, 2),
        "mean_edge_top30_bps": round(mean_edge_top30, 2) if mean_edge_top30 else None,
        "sharpe_annual": round(sharpe_annual, 2),
        "worst_month_acc": round(min(accs), 4),
        "best_month_acc": round(max(accs), 4),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("DIRECTION V3 — WALK-FORWARD VALIDATION", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Config: 30m bars, {FWD_BARS}-bar (2H) forward", flush=True)
    print(f"Test months: {TEST_MONTHS[0]} to {TEST_MONTHS[-1]} ({len(TEST_MONTHS)} months)", flush=True)
    print(f"Method: expanding window, retrain each month", flush=True)
    print("=" * 80, flush=True)

    all_results = {}

    for symbol in SYMBOLS:
        try:
            ohlcv = load_ohlcv(symbol, "30min")
            monthly, monthly_vf = walk_forward_asset(symbol, ohlcv)
            if monthly:
                stats_0 = compute_aggregate_stats(monthly, cost_bps=0)
                stats_3 = compute_aggregate_stats(monthly, cost_bps=3)
                stats_vf_0 = compute_aggregate_stats(monthly_vf, cost_bps=0)
                stats_vf_3 = compute_aggregate_stats(monthly_vf, cost_bps=3)
                all_results[symbol] = {
                    "monthly": monthly,
                    "monthly_volfilter": monthly_vf,
                    "aggregate_0bps": stats_0,
                    "aggregate_3bps": stats_3,
                    "aggregate_vf_0bps": stats_vf_0,
                    "aggregate_vf_3bps": stats_vf_3,
                }
        except Exception as e:
            print(f"  {symbol}: ERROR — {e}", flush=True)

    # Summary
    print("\n" + "=" * 110, flush=True)
    print("WALK-FORWARD SUMMARY", flush=True)
    print("=" * 110, flush=True)

    print(f"\n--- WITHOUT VOL FILTER ---", flush=True)
    print(f"{'Sym':<5} {'MeanAcc':>8} {'Top30':>7} {'Mo>50%':>7} {'MAR':>5} | "
          f"{'Edge@0':>7} {'Edge@3':>7} {'Sharpe':>7} | {'Worst':>7} {'Best':>7}", flush=True)
    print("-" * 85, flush=True)

    for sym, r in all_results.items():
        s = r["aggregate_0bps"]
        s3 = r["aggregate_3bps"]
        t30 = f"{s['mean_top30']:.4f}" if s.get('mean_top30') else "   N/A"
        e30 = f"{s.get('mean_edge_top30_bps', 'N/A')}"
        print(f"{sym:<5} {s['mean_acc']:>7.4f} {t30:>7} {s['months_above_50pct']:>3}/{s['months_total']:<3} "
              f"{s['mean_mar']:>5.0f} | "
              f"{s['mean_edge_bps']:>+6.2f} {s3['mean_edge_bps']:>+6.2f} {s3['sharpe_annual']:>7.2f} | "
              f"{s['worst_month_acc']:>6.4f} {s['best_month_acc']:>6.4f}", flush=True)

    print(f"\n--- WITH VOL FILTER (top 50% vol bars) ---", flush=True)
    print(f"{'Sym':<5} {'MeanAcc':>8} {'Top30':>7} {'Mo>50%':>7} {'MAR':>5} | "
          f"{'Edge@0':>7} {'Edge@3':>7} {'Sharpe':>7} | {'Worst':>7} {'Best':>7}", flush=True)
    print("-" * 85, flush=True)

    for sym, r in all_results.items():
        s = r.get("aggregate_vf_0bps", {})
        s3 = r.get("aggregate_vf_3bps", {})
        if not s or not s.get("mean_acc"):
            print(f"{sym:<5}   insufficient data with vol filter", flush=True)
            continue
        t30 = f"{s['mean_top30']:.4f}" if s.get('mean_top30') else "   N/A"
        print(f"{sym:<5} {s['mean_acc']:>7.4f} {t30:>7} {s['months_above_50pct']:>3}/{s['months_total']:<3} "
              f"{s['mean_mar']:>5.0f} | "
              f"{s['mean_edge_bps']:>+6.2f} {s3['mean_edge_bps']:>+6.2f} {s3['sharpe_annual']:>7.2f} | "
              f"{s['worst_month_acc']:>6.4f} {s['best_month_acc']:>6.4f}", flush=True)

    # Verdict
    print(f"\n--- VERDICT ---", flush=True)
    for sym, r in all_results.items():
        s = r["aggregate_3bps"]
        months_ok = s["months_above_50pct"]
        total = s["months_total"]
        sharpe = s["sharpe_annual"]
        if months_ok >= total * 0.7 and sharpe > 0:
            verdict = "PASS — consistent edge, production-ready"
        elif months_ok >= total * 0.5:
            verdict = "MARGINAL — edge exists but inconsistent"
        else:
            verdict = "FAIL — not reliable enough for production"
        print(f"  {sym}: {months_ok}/{total} months >50%, Sharpe={sharpe:.2f} — {verdict}", flush=True)

    # Save
    with open(OUTPUT_DIR / "direction_v3_walkforward.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": {
            k: {kk: vv for kk, vv in v.items()} for k, v in all_results.items()
        }}, f, indent=2, default=str)

    print(f"\nResults saved. Completed: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
