"""
Direction Prediction V2 — Phase A Experiments.

Four experiments to characterize the v1 edge and find optimal operating parameters:
  A1: Confidence threshold sweep (top 5% through 100%)
  A2: 5-minute timeframe test
  A3: Multi-asset test (ETH, SOL)
  A4: Holding period optimization (1, 2, 3, 5, 8 bars)

V1 baseline: 52.47% accuracy, 55.08% filtered top 30%, BTC 15m, 14 OHLCV+Tech features.

Run on homeserver: scp to /home/user/ml-data/ then `python3 direction_v2_phase_a.py`
"""

import sys
import os
import warnings
import json

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/ml-data")

import numpy as np
import pandas as pd
import xgboost as xgb
import lz4.frame
import csv
import io
from datetime import datetime
from pathlib import Path
from scipy.stats import binom

from technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels

# =============================================================================
# CONSTANTS
# =============================================================================

# Data paths
BINANCE_1M_DIR = Path("/home/user/ml-data/binance-1m")
HL_ASSET_CTXS_DIR = Path("/home/user/ml-data/hl-asset-ctxs")
RESULTS_OUTPUT = Path("/home/user/ml-data/direction_v2_phase_a_results.json")

# Symbols and their Binance directory names
SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}

# Train/test split
TRAIN_CUTOFF = "2025-07-01"

# XGBoost hyperparameters (identical to v1)
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
)

# Confidence threshold percentiles for A1 sweep
# "top X%" means we keep predictions where confidence >= (100-X)th percentile
THRESHOLD_PERCENTILES = [5, 10, 15, 20, 25, 30, 40, 50, 100]

# Holding periods for A4 (in bars)
HOLD_PERIODS = [1, 2, 3, 5, 8]

# Trading costs: round-trip maker cost in decimal (0.04% = 4 bps)
RT_COST_DECIMAL = 0.0004
RT_COST_BPS = 4.0

# Winsorization percentiles
WINSOR_LO = 0.01
WINSOR_HI = 0.99

# Bars per year at various timeframes (24/7 crypto)
BARS_PER_YEAR_15M = 365.25 * 24 * 4     # ~35,064
BARS_PER_YEAR_5M = 365.25 * 24 * 12     # ~105,192


# =============================================================================
# DATA LOADING (identical to v1 — direction_stacked.py)
# =============================================================================

def load_ohlcv(symbol: str, timeframe: str = "15min") -> pd.DataFrame:
    """Load 1m Binance CSVs for a symbol, resample to target timeframe.

    Args:
        symbol: Short name (BTC, ETH, SOL). Mapped to BTCUSDT etc.
        timeframe: Pandas resample string (e.g. "15min", "5min").

    Returns:
        DataFrame with OHLCV columns, UTC datetime index.
    """
    dir_name = SYMBOLS.get(symbol)
    if dir_name is None:
        raise ValueError(f"Unknown symbol: {symbol}. Known: {list(SYMBOLS.keys())}")

    data_dir = BINANCE_1M_DIR / dir_name
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    dfs = []
    for f in csvs:
        with open(f) as fh:
            first = fh.readline().strip().split(",")[0]
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qv", "count", "tbv", "tqv", "ignore",
        ]
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

    resampled = combined.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    return resampled


def build_features(ohlcv: pd.DataFrame, train_end: str = TRAIN_CUTOFF) -> pd.DataFrame:
    """Build the v1 feature set: 14 OHLCV + technical features.

    Winsorization quantiles computed on TRAIN data only (before train_end),
    then applied to the full dataset.

    Args:
        ohlcv: DataFrame with OHLCV columns.
        train_end: Cutoff date string. Quantiles computed on data before this.

    Returns:
        DataFrame of features, same index as ohlcv.
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]

    features = pd.DataFrame(index=ohlcv.index)

    # OHLCV return features (winsorized using TRAIN quantiles only)
    train_mask = ohlcv.index < train_end
    for lb in [1, 3, 7, 14]:
        ret = close.pct_change(lb)
        train_ret = ret[train_mask]
        lo = train_ret.quantile(WINSOR_LO)
        hi = train_ret.quantile(WINSOR_HI)
        features[f"ret_{lb}"] = ret.clip(lo, hi)

    features["bar_range"] = (high - low) / close
    features["vol_zscore"] = (
        (volume - volume.rolling(30).mean())
        / volume.rolling(30).std().replace(0, np.nan)
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

    return features


def train_and_predict(X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame) -> np.ndarray:
    """Train XGBoost with v1 params and return test probabilities.

    Returns:
        Array of P(up) probabilities for each test sample.
    """
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def binomial_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """One-sided binomial test: P(X >= k) under null hypothesis p.

    Returns 1.0 if k <= n*p (not better than chance).
    """
    if k <= n * p:
        return 1.0
    return 1.0 - binom.cdf(k - 1, n, p)


# =============================================================================
# A1: CONFIDENCE THRESHOLD SWEEP
# =============================================================================

def run_a1_threshold_sweep(ohlcv: pd.DataFrame) -> dict:
    """A1: Train v1 model, sweep confidence thresholds on test set.

    Returns dict with results table and best configuration.
    """
    print("\n" + "=" * 70, flush=True)
    print("A1: CONFIDENCE THRESHOLD SWEEP (BTC 15m)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Started: {datetime.now()}", flush=True)

    # Build features and labels
    features = build_features(ohlcv)
    close = ohlcv["close"]
    fwd_ret = close.pct_change(1).shift(-1)  # 1-bar forward return

    # Align features and labels
    features_clean = features.dropna()
    y_all = np.sign(fwd_ret).reindex(features_clean.index).dropna()
    valid_idx = features_clean.index.intersection(y_all.index)

    X = features_clean.loc[valid_idx]
    y = (y_all.loc[valid_idx] > 0).astype(int)
    raw_fwd_ret = fwd_ret.reindex(valid_idx).dropna()

    # Split
    train_mask = X.index < TRAIN_CUTOFF
    test_mask = X.index >= TRAIN_CUTOFF
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}", flush=True)
    print(f"  Features ({len(X.columns)}): {list(X.columns)}", flush=True)

    # Train and get probabilities
    y_prob = train_and_predict(X_train, y_train, X_test)
    y_pred = (y_prob > 0.5).astype(int)

    # Compute mean absolute return on test set for edge calculation
    test_abs_ret = raw_fwd_ret.loc[X_test.index].abs()
    mean_abs_ret = test_abs_ret.mean()
    print(f"  Mean abs 1-bar return (test): {mean_abs_ret * 10000:.2f} bps", flush=True)

    # Confidence = distance from 0.5
    conf = np.abs(y_prob - 0.5)

    # Sweep thresholds
    results_rows = []
    print(f"\n  {'Top%':>5} {'Acc':>8} {'Trades':>7} {'p-value':>10} {'Edge(bps)':>10} {'Net(bps)':>9} {'Profitable':>11}", flush=True)
    print(f"  {'-'*65}", flush=True)

    for pct in THRESHOLD_PERCENTILES:
        if pct == 100:
            mask = np.ones(len(conf), dtype=bool)
        else:
            threshold = np.percentile(conf, 100 - pct)
            mask = conf >= threshold

        n_trades = int(mask.sum())
        if n_trades < 10:
            continue

        correct = (y_pred[mask] == y_test.values[mask])
        acc = correct.mean()
        k = int(correct.sum())
        p_val = binomial_pvalue(k, n_trades)

        edge_bps = (acc - 0.5) * 2 * mean_abs_ret * 10000
        net_edge_bps = edge_bps - RT_COST_BPS
        profitable = net_edge_bps > 0

        row = {
            "top_pct": pct,
            "accuracy": round(float(acc), 4),
            "n_trades": n_trades,
            "p_value": float(p_val),
            "edge_bps": round(float(edge_bps), 2),
            "net_edge_bps": round(float(net_edge_bps), 2),
            "profitable": profitable,
        }
        results_rows.append(row)

        p_str = f"{p_val:.2e}" if p_val < 0.01 else f"{p_val:.4f}"
        prof_str = "YES" if profitable else "no"
        print(f"  {pct:>4}% {acc:>8.4f} {n_trades:>7} {p_str:>10} {edge_bps:>10.2f} {net_edge_bps:>9.2f} {prof_str:>11}", flush=True)

    # Find best net-profitable config
    profitable_rows = [r for r in results_rows if r["profitable"]]
    best = max(profitable_rows, key=lambda r: r["net_edge_bps"]) if profitable_rows else None

    print(f"\n  A1 Done: {datetime.now()}", flush=True)
    if best:
        print(f"  BEST: top {best['top_pct']}% — acc={best['accuracy']:.4f}, "
              f"net_edge={best['net_edge_bps']:.2f} bps, trades={best['n_trades']}", flush=True)
    else:
        print(f"  WARNING: No configuration is net-profitable after {RT_COST_BPS} bps RT cost", flush=True)

    return {"rows": results_rows, "best": best, "mean_abs_ret_bps": round(float(mean_abs_ret * 10000), 2)}


# =============================================================================
# A2: 5-MINUTE TIMEFRAME TEST
# =============================================================================

def run_a2_5m_timeframe(symbol: str = "BTC") -> dict:
    """A2: Test same model on 5-minute bars, compare to 15m baseline.

    Returns dict with 5m results and 15m baseline for comparison.
    """
    print("\n" + "=" * 70, flush=True)
    print("A2: 5-MINUTE TIMEFRAME TEST (BTC)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Started: {datetime.now()}", flush=True)

    results = {}

    for tf_label, tf_resample in [("15m", "15min"), ("5m", "5min")]:
        print(f"\n  Loading {symbol} {tf_label}...", flush=True)
        try:
            ohlcv = load_ohlcv(symbol, tf_resample)
        except (FileNotFoundError, ValueError) as e:
            print(f"  WARNING: {e} — skipping {tf_label}", flush=True)
            continue

        print(f"    {len(ohlcv)} bars ({ohlcv.index[0].date()} to {ohlcv.index[-1].date()})", flush=True)

        features = build_features(ohlcv)
        close = ohlcv["close"]
        fwd_ret = close.pct_change(1).shift(-1)

        features_clean = features.dropna()
        y_all = np.sign(fwd_ret).reindex(features_clean.index).dropna()
        valid_idx = features_clean.index.intersection(y_all.index)

        X = features_clean.loc[valid_idx]
        y = (y_all.loc[valid_idx] > 0).astype(int)

        train_mask = X.index < TRAIN_CUTOFF
        test_mask = X.index >= TRAIN_CUTOFF
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        print(f"    Train: {len(X_train)}, Test: {len(X_test)}", flush=True)

        if len(X_train) < 500 or len(X_test) < 200:
            print(f"    WARNING: insufficient data for {tf_label}", flush=True)
            continue

        y_prob = train_and_predict(X_train, y_train, X_test)
        y_pred = (y_prob > 0.5).astype(int)

        # Full accuracy
        acc = float((y_pred == y_test.values).mean())
        k = int((y_pred == y_test.values).sum())
        p_val = binomial_pvalue(k, len(y_test))

        # Top 30% filtered
        conf = np.abs(y_prob - 0.5)
        top30_mask = conf >= np.percentile(conf, 70)
        n_top30 = int(top30_mask.sum())
        if n_top30 > 50:
            acc_top30 = float((y_pred[top30_mask] == y_test.values[top30_mask]).mean())
            k_top30 = int((y_pred[top30_mask] == y_test.values[top30_mask]).sum())
            p_top30 = binomial_pvalue(k_top30, n_top30)
        else:
            acc_top30, p_top30 = acc, 1.0

        results[tf_label] = {
            "timeframe": tf_label,
            "accuracy": round(acc, 4),
            "p_value": float(p_val),
            "n_test": len(y_test),
            "acc_top30": round(acc_top30, 4),
            "p_top30": float(p_top30),
            "n_top30": n_top30,
        }

    # Print comparison table
    print(f"\n  {'TF':>4} {'Acc':>8} {'p-value':>10} {'n_test':>7} | {'Acc30%':>8} {'p30%':>10} {'n30%':>6}", flush=True)
    print(f"  {'-'*65}", flush=True)
    for tf_label, r in results.items():
        p_str = f"{r['p_value']:.2e}" if r["p_value"] < 0.01 else f"{r['p_value']:.4f}"
        p30_str = f"{r['p_top30']:.2e}" if r["p_top30"] < 0.01 else f"{r['p_top30']:.4f}"
        print(f"  {tf_label:>4} {r['accuracy']:>8.4f} {p_str:>10} {r['n_test']:>7} | "
              f"{r['acc_top30']:>8.4f} {p30_str:>10} {r['n_top30']:>6}", flush=True)

    print(f"\n  A2 Done: {datetime.now()}", flush=True)
    return results


# =============================================================================
# A3: MULTI-ASSET TEST (ETH, SOL)
# =============================================================================

def run_a3_multi_asset() -> dict:
    """A3: Test same model architecture on ETH and SOL (15m).

    Returns dict keyed by symbol with accuracy results.
    """
    print("\n" + "=" * 70, flush=True)
    print("A3: MULTI-ASSET TEST (BTC, ETH, SOL @ 15m)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Started: {datetime.now()}", flush=True)

    results = {}
    test_symbols = ["BTC", "ETH", "SOL"]

    for symbol in test_symbols:
        print(f"\n  Loading {symbol} 15m...", flush=True)
        try:
            ohlcv = load_ohlcv(symbol, "15min")
        except (FileNotFoundError, ValueError) as e:
            print(f"  WARNING: {e} — skipping {symbol}", flush=True)
            continue

        print(f"    {len(ohlcv)} bars ({ohlcv.index[0].date()} to {ohlcv.index[-1].date()})", flush=True)

        features = build_features(ohlcv)
        close = ohlcv["close"]
        fwd_ret = close.pct_change(1).shift(-1)

        features_clean = features.dropna()
        y_all = np.sign(fwd_ret).reindex(features_clean.index).dropna()
        valid_idx = features_clean.index.intersection(y_all.index)

        X = features_clean.loc[valid_idx]
        y = (y_all.loc[valid_idx] > 0).astype(int)

        train_mask = X.index < TRAIN_CUTOFF
        test_mask = X.index >= TRAIN_CUTOFF
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        print(f"    Train: {len(X_train)}, Test: {len(X_test)}", flush=True)

        if len(X_train) < 500 or len(X_test) < 200:
            print(f"    WARNING: insufficient data for {symbol}", flush=True)
            continue

        y_prob = train_and_predict(X_train, y_train, X_test)
        y_pred = (y_prob > 0.5).astype(int)

        # Full accuracy
        acc = float((y_pred == y_test.values).mean())
        k = int((y_pred == y_test.values).sum())
        p_val = binomial_pvalue(k, len(y_test))

        # Top 30% filtered
        conf = np.abs(y_prob - 0.5)
        top30_mask = conf >= np.percentile(conf, 70)
        n_top30 = int(top30_mask.sum())
        if n_top30 > 50:
            acc_top30 = float((y_pred[top30_mask] == y_test.values[top30_mask]).mean())
            k_top30 = int((y_pred[top30_mask] == y_test.values[top30_mask]).sum())
            p_top30 = binomial_pvalue(k_top30, n_top30)
        else:
            acc_top30, p_top30, n_top30 = acc, 1.0, 0

        results[symbol] = {
            "symbol": symbol,
            "accuracy": round(acc, 4),
            "p_value": float(p_val),
            "n_test": len(y_test),
            "acc_top30": round(acc_top30, 4),
            "p_top30": float(p_top30),
            "n_top30": n_top30,
        }

    # Print comparison table
    print(f"\n  {'Symbol':>6} {'Acc':>8} {'p-value':>10} {'n_test':>7} | {'Acc30%':>8} {'p30%':>10} {'n30%':>6}", flush=True)
    print(f"  {'-'*68}", flush=True)
    for symbol, r in results.items():
        p_str = f"{r['p_value']:.2e}" if r["p_value"] < 0.01 else f"{r['p_value']:.4f}"
        p30_str = f"{r['p_top30']:.2e}" if r["p_top30"] < 0.01 else f"{r['p_top30']:.4f}"
        print(f"  {symbol:>6} {r['accuracy']:>8.4f} {p_str:>10} {r['n_test']:>7} | "
              f"{r['acc_top30']:>8.4f} {p30_str:>10} {r['n_top30']:>6}", flush=True)

    print(f"\n  A3 Done: {datetime.now()}", flush=True)
    return results


# =============================================================================
# A4: HOLDING PERIOD OPTIMIZATION
# =============================================================================

def run_a4_holding_period(ohlcv: pd.DataFrame) -> dict:
    """A4: Simulate various hold periods using v1 predictions.

    For each hold period N:
      - Entry when model predicts UP with confidence in top 30%
      - Exit after N bars (fixed hold)
      - Costs: 0.04% RT per trade regardless of hold period
      - Net Sharpe: mean(returns - cost) / std(returns) * sqrt(trades_per_year)

    Returns dict with per-period results.
    """
    print("\n" + "=" * 70, flush=True)
    print("A4: HOLDING PERIOD OPTIMIZATION (BTC 15m)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Started: {datetime.now()}", flush=True)

    # Build features and train model (1-bar prediction, same as v1)
    features = build_features(ohlcv)
    close = ohlcv["close"]
    fwd_ret_1bar = close.pct_change(1).shift(-1)

    features_clean = features.dropna()
    y_all = np.sign(fwd_ret_1bar).reindex(features_clean.index).dropna()
    valid_idx = features_clean.index.intersection(y_all.index)

    X = features_clean.loc[valid_idx]
    y = (y_all.loc[valid_idx] > 0).astype(int)

    train_mask = X.index < TRAIN_CUTOFF
    test_mask = X.index >= TRAIN_CUTOFF
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}", flush=True)

    # Get predictions
    y_prob = train_and_predict(X_train, y_train, X_test)

    # Filter to top 30% confidence predictions that predict UP
    conf = np.abs(y_prob - 0.5)
    conf_threshold = np.percentile(conf, 70)
    pred_up = y_prob > 0.5
    signal_mask = pred_up & (conf >= conf_threshold)

    signal_indices = X_test.index[signal_mask]
    print(f"  Signal entries (top 30% conf, predict UP): {len(signal_indices)}", flush=True)

    # For each hold period, compute forward N-bar returns
    results = {}

    print(f"\n  {'Hold':>5} {'AvgRet':>10} {'WinRate':>8} {'Trades':>7} {'Sharpe':>8} {'NetSharpe':>10} {'TotRet%':>8}", flush=True)
    print(f"  {'-'*65}", flush=True)

    for n_bars in HOLD_PERIODS:
        # Forward return over N bars from each signal bar
        # Entry at close of signal bar (bar t), exit at close of bar t+n_bars
        fwd_ret_n = close.pct_change(n_bars).shift(-n_bars)

        # Get returns for signal entries
        valid_signals = signal_indices.intersection(fwd_ret_n.dropna().index)
        trade_returns = fwd_ret_n.loc[valid_signals].values

        if len(trade_returns) < 10:
            print(f"  {n_bars:>4}b   insufficient trades ({len(trade_returns)})", flush=True)
            continue

        # Remove overlapping trades: only enter if not already in a position
        # Simple approach: skip entries within n_bars of previous entry
        non_overlapping_idx = []
        last_entry = None
        for idx in valid_signals:
            if last_entry is None:
                non_overlapping_idx.append(idx)
                last_entry = idx
            else:
                # Compute bars between entries
                bar_diff = (idx - last_entry).total_seconds() / (15 * 60)  # 15m bars
                if bar_diff >= n_bars:
                    non_overlapping_idx.append(idx)
                    last_entry = idx

        trade_returns_no = fwd_ret_n.loc[non_overlapping_idx].values

        n_trades = len(trade_returns_no)
        if n_trades < 10:
            print(f"  {n_bars:>4}b   insufficient non-overlapping trades ({n_trades})", flush=True)
            continue

        avg_ret = float(np.mean(trade_returns_no))
        win_rate = float((trade_returns_no > 0).mean())
        total_ret = float(np.sum(trade_returns_no - RT_COST_DECIMAL))

        # Net returns after costs
        net_returns = trade_returns_no - RT_COST_DECIMAL

        # Estimate trades per year: test period duration -> annualize
        test_duration_days = (valid_signals[-1] - valid_signals[0]).total_seconds() / 86400
        if test_duration_days > 0:
            trades_per_year = n_trades / test_duration_days * 365.25
        else:
            trades_per_year = 0

        # Sharpe ratio (annualized)
        if np.std(trade_returns_no) > 0 and trades_per_year > 0:
            gross_sharpe = np.mean(trade_returns_no) / np.std(trade_returns_no) * np.sqrt(trades_per_year)
            net_sharpe = np.mean(net_returns) / np.std(net_returns) * np.sqrt(trades_per_year)
        else:
            gross_sharpe, net_sharpe = 0.0, 0.0

        results[n_bars] = {
            "hold_bars": n_bars,
            "avg_return_bps": round(avg_ret * 10000, 2),
            "win_rate": round(win_rate, 4),
            "n_trades": n_trades,
            "gross_sharpe": round(float(gross_sharpe), 2),
            "net_sharpe": round(float(net_sharpe), 2),
            "total_return_pct": round(total_ret * 100, 2),
            "trades_per_year": round(trades_per_year, 0),
        }

        print(f"  {n_bars:>4}b {avg_ret*10000:>9.2f}bp {win_rate:>7.4f} {n_trades:>7} "
              f"{gross_sharpe:>8.2f} {net_sharpe:>10.2f} {total_ret*100:>7.2f}%", flush=True)

    # Key insight summary
    if results:
        best_sharpe = max(results.values(), key=lambda r: r["net_sharpe"])
        print(f"\n  KEY INSIGHT: Costs are FIXED at {RT_COST_BPS:.0f} bps per trade.", flush=True)
        print(f"  Longer holds amortize costs over larger returns.", flush=True)
        print(f"  Best net Sharpe: {best_sharpe['hold_bars']}-bar hold "
              f"(Sharpe={best_sharpe['net_sharpe']:.2f}, "
              f"avg_ret={best_sharpe['avg_return_bps']:.1f}bps)", flush=True)

    print(f"\n  A4 Done: {datetime.now()}", flush=True)
    return results


# =============================================================================
# PHASE A SUMMARY
# =============================================================================

def print_summary(a1_results: dict, a2_results: dict, a3_results: dict, a4_results: dict):
    """Print consolidated Phase A summary with recommendations."""
    print("\n" + "=" * 70, flush=True)
    print("PHASE A SUMMARY", flush=True)
    print("=" * 70, flush=True)

    # A1: Best threshold
    print("\n--- A1: Confidence Threshold ---", flush=True)
    if a1_results.get("best"):
        b = a1_results["best"]
        print(f"  Best config: top {b['top_pct']}%", flush=True)
        print(f"    Accuracy:  {b['accuracy']:.4f}", flush=True)
        print(f"    Trades:    {b['n_trades']}", flush=True)
        print(f"    Gross edge: {b['edge_bps']:.2f} bps", flush=True)
        print(f"    Net edge:   {b['net_edge_bps']:.2f} bps (after {RT_COST_BPS:.0f} bps RT cost)", flush=True)
    else:
        print("  NO net-profitable threshold found.", flush=True)

    any_profitable = any(r["profitable"] for r in a1_results.get("rows", []))
    profitable_rows = [r for r in a1_results.get("rows", []) if r["profitable"]]
    print(f"  Profitable configs: {len(profitable_rows)} / {len(a1_results.get('rows', []))}", flush=True)

    # A2: Timeframe comparison
    print("\n--- A2: Timeframe Comparison ---", flush=True)
    best_tf = None
    best_tf_acc = 0
    for tf, r in a2_results.items():
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        print(f"  {tf}: acc={r['accuracy']:.4f} (p={r['p_value']:.4f}) {sig}  "
              f"top30%={r['acc_top30']:.4f}", flush=True)
        if r["accuracy"] > best_tf_acc:
            best_tf_acc = r["accuracy"]
            best_tf = tf
    if best_tf:
        print(f"  Best timeframe: {best_tf}", flush=True)

    # A3: Multi-asset
    print("\n--- A3: Multi-Asset ---", flush=True)
    generalizes = True
    for symbol, r in a3_results.items():
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        print(f"  {symbol}: acc={r['accuracy']:.4f} (p={r['p_value']:.4f}) {sig}  "
              f"top30%={r['acc_top30']:.4f}", flush=True)
        if r["p_value"] > 0.05:
            generalizes = False
    print(f"  Edge generalizes across assets: {'YES' if generalizes else 'PARTIAL/NO'}", flush=True)

    # A4: Holding period
    print("\n--- A4: Holding Period ---", flush=True)
    best_hold = None
    for n_bars, r in a4_results.items():
        prof = "PROFITABLE" if r["net_sharpe"] > 0 else "unprofitable"
        print(f"  {n_bars}-bar: avg={r['avg_return_bps']:.1f}bps, "
              f"winrate={r['win_rate']:.4f}, "
              f"netSharpe={r['net_sharpe']:.2f} ({prof})", flush=True)
        if best_hold is None or r["net_sharpe"] > a4_results.get(best_hold, {}).get("net_sharpe", -999):
            best_hold = n_bars
    if best_hold:
        print(f"  Best hold: {best_hold} bars (net Sharpe={a4_results[best_hold]['net_sharpe']:.2f})", flush=True)

    # Overall recommendation
    print("\n--- RECOMMENDATION ---", flush=True)

    conditions_met = 0
    total_conditions = 4

    if any_profitable:
        print(f"  [PASS] At least one threshold config is net-profitable", flush=True)
        conditions_met += 1
    else:
        print(f"  [FAIL] No threshold config is net-profitable after costs", flush=True)

    if a2_results and any(r["p_value"] < 0.05 for r in a2_results.values()):
        print(f"  [PASS] At least one timeframe shows significant edge", flush=True)
        conditions_met += 1
    else:
        print(f"  [FAIL] No timeframe shows significant edge", flush=True)

    if a3_results and sum(1 for r in a3_results.values() if r["p_value"] < 0.10) >= 2:
        print(f"  [PASS] Edge present in 2+ assets", flush=True)
        conditions_met += 1
    else:
        print(f"  [WARN] Edge does not generalize well across assets", flush=True)

    if a4_results and any(r["net_sharpe"] > 1.0 for r in a4_results.values()):
        print(f"  [PASS] At least one hold period has net Sharpe > 1.0", flush=True)
        conditions_met += 1
    else:
        print(f"  [WARN] No hold period achieves net Sharpe > 1.0", flush=True)

    print(f"\n  Conditions met: {conditions_met}/{total_conditions}", flush=True)
    if conditions_met >= 3:
        print(f"  >>> PROCEED TO PHASE B: feature engineering and model improvements", flush=True)
    elif conditions_met >= 2:
        print(f"  >>> PROCEED TO PHASE B with CAUTION: edge is marginal, focus on cost reduction", flush=True)
    else:
        print(f"  >>> DO NOT PROCEED: insufficient edge. Re-examine v1 assumptions.", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70, flush=True)
    print("DIRECTION PREDICTION V2 — PHASE A EXPERIMENTS", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"V1 baseline: 52.47% acc, 55.08% filtered top 30% (BTC 15m)", flush=True)
    print("=" * 70, flush=True)

    # Load BTC 15m once — reused by A1 and A4
    print("\nLoading BTC 15m OHLCV...", flush=True)
    btc_15m = load_ohlcv("BTC", "15min")
    print(f"  {len(btc_15m)} bars ({btc_15m.index[0].date()} to {btc_15m.index[-1].date()})", flush=True)

    # --- Run all four experiments ---

    a1_results = run_a1_threshold_sweep(btc_15m)
    a2_results = run_a2_5m_timeframe()
    a3_results = run_a3_multi_asset()
    a4_results = run_a4_holding_period(btc_15m)

    # --- Summary ---

    print_summary(a1_results, a2_results, a3_results, a4_results)

    # Save all results to JSON
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "a1_threshold_sweep": a1_results,
        "a2_timeframe": a2_results,
        "a3_multi_asset": a3_results,
        "a4_holding_period": {str(k): v for k, v in a4_results.items()},
    }

    with open(RESULTS_OUTPUT, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_OUTPUT}", flush=True)

    print(f"\nAll Phase A experiments complete: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
