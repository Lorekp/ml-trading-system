"""
Direction Prediction V2 — Multi-Timeframe Analysis.

Key question: at what timeframe does the mean absolute return per bar
become large enough to overcome 4 bps round-trip trading costs?

Phase A showed 15m has 15.16 bps mean abs 1-bar return — too small for 4 bps costs.
Phase B showed CUSUM + simple labels is the best approach (54.67% accuracy).
Now we test [5m, 15m, 30m, 1H, 4H] to find the optimal timeframe.

For each timeframe:
  1. Load BTC 1m data, resample to target timeframe
  2. Build same 14 features (OHLCV + Tech)
  3. CUSUM event sampling with dynamic threshold (mult=2.0)
  4. Labels: sign of 1-bar forward return at CUSUM events (Pipeline 1)
  5. Sample weights with uniqueness + time decay
  6. Train XGBoost (same params as v1)
  7. Evaluate: accuracy, confidence-filtered accuracy, mean abs return,
     edge in bps, net edge after 4 bps cost

Run on homeserver: python3 direction_v2_timeframes.py
Data at /home/user/ml-data/
"""

from __future__ import annotations

import sys
import os
import warnings
import json

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/ml-data")

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import binom
from typing import Optional


# ===========================================================================
# CONSTANTS
# ===========================================================================

DATA_DIR = Path("/home/user/ml-data")
BTC_DIR = DATA_DIR / "binance-1m" / "BTCUSDT"
OUTPUT_JSON = DATA_DIR / "direction_v2_timeframes_results.json"

TRAIN_CUTOFF = "2025-07-01"

# Timeframes to test: (pandas resample string, display label, minutes per bar)
TIMEFRAMES = [
    ("5min",  "5m",  5),
    ("15min", "15m", 15),
    ("30min", "30m", 30),
    ("1h",    "1H",  60),
    ("4h",    "4H",  240),
]

# CUSUM parameters (Phase B Pipeline 1 approach)
CUSUM_MULT = 2.0                       # Fixed multiplier (Phase B best)
CUSUM_VOL_SPAN = 50                    # EWM span for get_daily_vol
TIME_DECAY_CLF_LAST_W = 0.5            # Oldest sample = 50% weight

# XGBoost hyperparameters (identical to v1)
XGBOOST_PARAMS = dict(
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

# Trading costs
RT_COST_BPS = 4.0        # Round-trip maker cost: 0.04% = 4 bps

# Winsorization
WINSOR_LO = 0.01
WINSOR_HI = 0.99


# ===========================================================================
# INLINED TECHNICAL INDICATORS (from Phase B — no external dependency)
# ===========================================================================

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")


def stochastic_rsi(close: pd.Series, rsi_period: int = 14,
                   stoch_period: int = 14) -> pd.Series:
    """Stochastic RSI."""
    rsi_val = rsi(close, rsi_period)
    rsi_min = rsi_val.rolling(stoch_period).min()
    rsi_max = rsi_val.rolling(stoch_period).max()
    rsi_range = rsi_max - rsi_min
    stoch = ((rsi_val - rsi_min) / rsi_range.replace(0, np.nan) * 100)
    return stoch.rename("stoch_rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD histogram, normalized by price."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    result = pd.DataFrame(index=close.index)
    result["macd_hist"] = histogram / close * 100
    return result


def bollinger_pct_b(close: pd.Series, window: int = 20,
                    n_std: float = 2.0) -> pd.Series:
    """Bollinger %B — position within Bollinger Bands."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    band_width = upper - lower
    pct_b = (close - lower) / band_width.replace(0, np.nan)
    return pct_b.rename("bb_pct_b")


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average True Range — volatility from H/L/C including gaps."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return (tr.ewm(span=period, adjust=False).mean() / close).rename(f"atr_{period}")


def swing_levels(high: pd.Series, low: pd.Series,
                 lookback: int = 20) -> pd.DataFrame:
    """Swing high/low detection — backward-looking only."""
    result = pd.DataFrame(index=high.index)
    rolling_high = high.rolling(lookback).max()
    rolling_low = low.rolling(lookback).min()
    is_swing_high = (high == rolling_high)
    is_swing_low = (low == rolling_low)
    swing_highs = high.where(is_swing_high).ffill()
    swing_lows = low.where(is_swing_low).ffill()
    mid = (high + low) / 2
    result["dist_swing_high"] = (swing_highs - mid) / mid
    result["dist_swing_low"] = (mid - swing_lows) / mid
    result["dist_nearest_level"] = pd.concat(
        [result["dist_swing_high"].abs(), result["dist_swing_low"].abs()],
        axis=1
    ).min(axis=1)
    return result


# ===========================================================================
# AFML FUNCTIONS (inlined from Phase B)
# ===========================================================================

def get_daily_vol(close: pd.Series, span: int = 50) -> pd.Series:
    """AFML Ch 2: Exponentially weighted moving std of returns."""
    returns = close.pct_change().dropna()
    vol = returns.ewm(span=span).std()
    return vol


def cusum_filter(raw_series: pd.Series,
                 threshold: float | pd.Series) -> pd.DatetimeIndex:
    """AFML Ch 2, Snippet 2.4: CUSUM filter for event-based sampling."""
    events: list = []
    s_pos = 0.0
    s_neg = 0.0
    diff = raw_series.diff().dropna()

    for t in diff.index:
        if isinstance(threshold, pd.Series):
            h = threshold.get(t, None)
            if h is None or np.isnan(h):
                continue
        else:
            h = float(threshold)

        s_pos = max(0.0, s_pos + diff.loc[t])
        s_neg = min(0.0, s_neg + diff.loc[t])

        if s_neg < -h:
            s_neg = 0.0
            events.append(t)
        elif s_pos > h:
            s_pos = 0.0
            events.append(t)

    return pd.DatetimeIndex(events)


# ---------------------------------------------------------------------------
# Sample Weights (AFML Ch 4, Snippets 4.1-4.5)
# ---------------------------------------------------------------------------

def get_num_concurrent_events(close_idx: pd.DatetimeIndex,
                              t1: pd.Series) -> pd.Series:
    """AFML Ch 4, Snippet 4.1: Count concurrent labels at each bar."""
    t1 = t1.fillna(close_idx[-1])
    iloc_range = close_idx.searchsorted([t1.index.min(), t1.max()])
    count = pd.Series(0, index=close_idx[iloc_range[0]:iloc_range[1] + 1],
                      dtype=float)
    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1.0
    return count


def get_avg_uniqueness(t1: pd.Series,
                       num_concurrent: pd.Series) -> pd.Series:
    """AFML Ch 4, Snippet 4.2: Average uniqueness per label."""
    weights = pd.Series(index=t1.index, dtype=float)
    for t_in, t_out in t1.items():
        if pd.isna(t_out):
            continue
        concurrent_slice = num_concurrent.loc[t_in:t_out]
        if len(concurrent_slice) == 0:
            weights.loc[t_in] = 0.0
            continue
        weights.loc[t_in] = (1.0 / concurrent_slice).mean()
    return weights


def get_time_decay(avg_uniqueness: pd.Series,
                   clf_last_w: float = 1.0) -> pd.Series:
    """AFML Ch 4, Snippet 4.5: Time decay on cumulative uniqueness."""
    cum_w = avg_uniqueness.sort_index().cumsum()
    if cum_w.iloc[-1] == 0:
        return avg_uniqueness
    if clf_last_w >= 0:
        slope = (1.0 - clf_last_w) / cum_w.iloc[-1]
    else:
        slope = 1.0 / ((clf_last_w + 1) * cum_w.iloc[-1])
    const = 1.0 - slope * cum_w.iloc[-1]
    result = const + slope * cum_w
    result[result < 0] = 0.0
    return result


def compute_sample_weights(t1: pd.Series, close: pd.Series,
                           clf_last_w: float = TIME_DECAY_CLF_LAST_W
                           ) -> pd.Series:
    """Compute full AFML sample weights: uniqueness + time decay.

    Normalized to mean=1 for XGBoost sample_weight.
    """
    if len(t1) == 0:
        return pd.Series(dtype=float)

    num_conc = get_num_concurrent_events(close.index, t1)
    avg_uniq = get_avg_uniqueness(t1, num_conc)

    if avg_uniq.sum() == 0:
        return pd.Series(1.0, index=t1.index)

    decayed = get_time_decay(avg_uniq, clf_last_w=clf_last_w)

    if decayed.mean() > 0:
        decayed = decayed / decayed.mean()
    else:
        decayed = pd.Series(1.0, index=t1.index)

    return decayed


# ===========================================================================
# DATA LOADING (from Phase A — load once, resample per timeframe)
# ===========================================================================

def load_btc_1m() -> pd.DataFrame:
    """Load raw BTC 1m data from Binance CSVs."""
    csvs = sorted(BTC_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {BTC_DIR}")

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
    return combined


def resample_ohlcv(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1m OHLCV to target timeframe."""
    return df_1m.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])


# ===========================================================================
# FEATURE ENGINEERING (14 features — same as v1/Phase B)
# ===========================================================================

def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build 14 features: 6 OHLCV + 8 technical.

    Winsorization quantiles computed on TRAIN data only (before TRAIN_CUTOFF).
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]

    features = pd.DataFrame(index=ohlcv.index)
    train_mask = ohlcv.index < TRAIN_CUTOFF

    # OHLCV return features (winsorized using TRAIN quantiles only)
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


# ===========================================================================
# SINGLE TIMEFRAME EVALUATION (Pipeline 1: CUSUM + simple labels)
# ===========================================================================

def evaluate_timeframe(
    ohlcv: pd.DataFrame,
    tf_label: str,
    tf_minutes: int,
) -> dict | None:
    """Run Pipeline 1 (CUSUM + simple labels) on a single timeframe.

    Steps:
      1. Compute mean abs return per bar on test set
      2. Build 14 features
      3. CUSUM event sampling (mult=2.0)
      4. Labels: sign of 1-bar forward return at events
      5. Sample weights with uniqueness + time decay
      6. Train XGBoost, evaluate on test events
      7. Compute accuracy at all / top 30% / top 20% / top 10%
      8. Compute edge and net edge after costs

    Returns dict of results or None on failure.
    """
    print(f"\n{'='*70}", flush=True)
    print(f"  TIMEFRAME: {tf_label} ({len(ohlcv)} bars)", flush=True)
    print(f"  Date range: {ohlcv.index[0].date()} to {ohlcv.index[-1].date()}", flush=True)
    print(f"{'='*70}", flush=True)

    close = ohlcv["close"]

    # ----- Mean absolute return per bar (test period, ALL bars) -----
    all_ret = close.pct_change(1)
    test_ret = all_ret[all_ret.index >= TRAIN_CUTOFF].dropna()
    mean_abs_ret_bps = float(test_ret.abs().mean() * 10000)
    print(f"  Mean abs 1-bar return (all test bars): {mean_abs_ret_bps:.2f} bps", flush=True)

    # Breakeven accuracy
    if mean_abs_ret_bps > 0:
        breakeven_acc = 0.5 + RT_COST_BPS / (2 * mean_abs_ret_bps)
    else:
        breakeven_acc = 1.0
    print(f"  Breakeven accuracy: {breakeven_acc:.4f} ({breakeven_acc*100:.2f}%)", flush=True)

    # ----- Build features -----
    print(f"  Building 14 features...", flush=True)
    features = build_features(ohlcv)
    features_clean = features.dropna()
    print(f"  {len(features_clean)} clean feature rows "
          f"({len(features.columns)} features)", flush=True)

    # ----- CUSUM event sampling -----
    print(f"  CUSUM sampling (mult={CUSUM_MULT}, vol_span={CUSUM_VOL_SPAN})...",
          flush=True)
    vol = get_daily_vol(close, span=CUSUM_VOL_SPAN)
    threshold = vol * close * CUSUM_MULT
    cusum_events = cusum_filter(close, threshold)
    n_bars = len(close)
    event_rate = len(cusum_events) / n_bars if n_bars > 0 else 0
    print(f"  CUSUM events: {len(cusum_events)}, "
          f"rate={event_rate:.4f} ({event_rate*100:.1f}%)", flush=True)

    if len(cusum_events) < 100:
        print(f"  WARNING: Too few CUSUM events ({len(cusum_events)}). "
              f"Skipping.", flush=True)
        return None

    # ----- Labels: sign of 1-bar forward return at CUSUM events -----
    valid_events = cusum_events.intersection(features_clean.index)
    fwd_ret = close.pct_change(1).shift(-1)
    y_all = fwd_ret.reindex(valid_events).dropna()
    valid_events = y_all.index
    y_binary = (y_all > 0).astype(int)

    # t1 for sample weights: event_time + 1 bar
    t1 = pd.Series(
        valid_events + pd.Timedelta(minutes=tf_minutes),
        index=valid_events,
    )
    t1 = t1.clip(upper=close.index[-1])

    # ----- Train/test split on events -----
    train_mask = valid_events < TRAIN_CUTOFF
    test_mask = valid_events >= TRAIN_CUTOFF
    events_train = valid_events[train_mask]
    events_test = valid_events[test_mask]

    n_train = len(events_train)
    n_test = len(events_test)
    print(f"  Valid events: {len(valid_events)} "
          f"(train={n_train}, test={n_test})", flush=True)

    if n_train < 200 or n_test < 50:
        print(f"  WARNING: Insufficient events. Skipping.", flush=True)
        return None

    # Label balance
    train_pos = y_binary.loc[events_train].mean() * 100
    test_pos = y_binary.loc[events_test].mean() * 100
    print(f"  Label balance: train {train_pos:.1f}% pos, "
          f"test {test_pos:.1f}% pos", flush=True)

    # ----- Sample weights (AFML Ch 4) -----
    print(f"  Computing sample weights (uniqueness + time decay)...",
          flush=True)
    t1_train = t1.loc[events_train]
    weights_train = compute_sample_weights(t1_train, close)

    # ----- Train XGBoost -----
    print(f"  Training XGBoost (n_estimators={XGBOOST_PARAMS['n_estimators']}, "
          f"max_depth={XGBOOST_PARAMS['max_depth']})...", flush=True)
    X_train = features_clean.loc[events_train]
    y_train = y_binary.loc[events_train]

    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, sample_weight=weights_train.values)

    # ----- Evaluate on test events -----
    X_test = features_clean.loc[events_test]
    y_test = y_binary.loc[events_test].values
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    # Overall accuracy
    acc_all = float((y_pred == y_test).mean())
    k_all = int((y_pred == y_test).sum())
    p_val = float(1 - binom.cdf(k_all - 1, n_test, 0.5)
                  if k_all > n_test / 2 else 1.0)

    # Confidence = distance from 0.5
    conf = np.abs(y_prob - 0.5)

    # ----- Confidence-filtered accuracy -----
    filtered = {}
    for pct_label, pct_threshold in [("top30", 70), ("top20", 80), ("top10", 90)]:
        thresh = np.percentile(conf, pct_threshold)
        mask = conf >= thresh
        n_filt = int(mask.sum())
        if n_filt >= 10:
            acc_filt = float((y_pred[mask] == y_test[mask]).mean())
            k_filt = int((y_pred[mask] == y_test[mask]).sum())
            p_filt = float(1 - binom.cdf(k_filt - 1, n_filt, 0.5)
                           if k_filt > n_filt / 2 else 1.0)
        else:
            acc_filt, p_filt, n_filt = float("nan"), 1.0, 0
        filtered[pct_label] = {
            "accuracy": acc_filt, "p_value": p_filt, "n": n_filt
        }

    # ----- Edge computations -----
    # Edge = (accuracy - 0.5) * 2 * mean_abs_return_bps
    edge_all = (acc_all - 0.5) * 2 * mean_abs_ret_bps
    net_all = edge_all - RT_COST_BPS

    def safe_edge(acc):
        if acc is None or np.isnan(acc):
            return None, None
        e = (acc - 0.5) * 2 * mean_abs_ret_bps
        return round(e, 2), round(e - RT_COST_BPS, 2)

    acc_30 = filtered["top30"]["accuracy"]
    acc_20 = filtered["top20"]["accuracy"]
    acc_10 = filtered["top10"]["accuracy"]

    edge_30, net_30 = safe_edge(acc_30)
    edge_20, net_20 = safe_edge(acc_20)
    edge_10, net_10 = safe_edge(acc_10)

    # ----- Build result dict -----
    result = {
        "timeframe": tf_label,
        "tf_minutes": tf_minutes,
        "n_bars_total": len(ohlcv),
        "n_bars_test": len(test_ret),
        "n_cusum_events": len(cusum_events),
        "cusum_event_rate": round(event_rate, 4),
        "n_train": n_train,
        "n_test": n_test,
        "mean_abs_ret_bps": round(mean_abs_ret_bps, 2),
        "breakeven_acc": round(breakeven_acc, 4),
        "acc_all": round(acc_all, 4),
        "acc_top30": round(acc_30, 4) if not np.isnan(acc_30) else None,
        "acc_top20": round(acc_20, 4) if not np.isnan(acc_20) else None,
        "acc_top10": round(acc_10, 4) if not np.isnan(acc_10) else None,
        "n_top30": filtered["top30"]["n"],
        "n_top20": filtered["top20"]["n"],
        "n_top10": filtered["top10"]["n"],
        "p_value": p_val,
        "p_top30": filtered["top30"]["p_value"],
        "edge_all_bps": round(edge_all, 2),
        "net_edge_all_bps": round(net_all, 2),
        "edge_top30_bps": edge_30,
        "net_edge_top30_bps": net_30,
        "edge_top20_bps": edge_20,
        "net_edge_top20_bps": net_20,
        "edge_top10_bps": edge_10,
        "net_edge_top10_bps": net_10,
    }

    # ----- Print per-timeframe summary -----
    print(f"\n  --- {tf_label} Results ---", flush=True)
    print(f"  Mean abs return/bar:  {mean_abs_ret_bps:.2f} bps", flush=True)
    print(f"  Breakeven accuracy:   {breakeven_acc:.4f} "
          f"({breakeven_acc*100:.2f}%)", flush=True)
    print(f"  Accuracy (all):       {acc_all:.4f}  "
          f"(p={p_val:.4e})", flush=True)
    if not np.isnan(acc_30):
        print(f"  Accuracy (top 30%):   {acc_30:.4f}  "
              f"(n={filtered['top30']['n']})", flush=True)
    if not np.isnan(acc_20):
        print(f"  Accuracy (top 20%):   {acc_20:.4f}  "
              f"(n={filtered['top20']['n']})", flush=True)
    if not np.isnan(acc_10):
        print(f"  Accuracy (top 10%):   {acc_10:.4f}  "
              f"(n={filtered['top10']['n']})", flush=True)
    print(f"  Edge (all):           {edge_all:.2f} bps", flush=True)
    prof_str = "PROFITABLE" if net_all > 0 else "unprofitable"
    print(f"  Net edge (all):       {net_all:+.2f} bps  [{prof_str}]",
          flush=True)
    if net_30 is not None:
        prof30 = "PROFITABLE" if net_30 > 0 else "unprofitable"
        print(f"  Net edge (top 30%):   {net_30:+.2f} bps  [{prof30}]",
              flush=True)

    return result


# ===========================================================================
# OUTPUT TABLES
# ===========================================================================

def print_main_table(results: list[dict]) -> None:
    """Print the main comparison table across all timeframes."""
    print("\n" + "=" * 140, flush=True)
    print("MULTI-TIMEFRAME COMPARISON TABLE", flush=True)
    print("=" * 140, flush=True)

    header = (
        f"{'TF':>4}  {'Bars':>7}  {'Events':>7}  "
        f"{'MeanAbsRet':>11}  "
        f"{'Acc(all)':>9}  {'Acc(30%)':>9}  {'Acc(10%)':>9}  "
        f"{'Edge':>7}  {'NetEdge':>8}  "
        f"{'Edge30%':>8}  {'NetEdge30%':>11}"
    )
    print(header, flush=True)
    print("-" * 140, flush=True)

    for r in results:
        tf = r["timeframe"]
        bars = r["n_bars_test"]
        events = r["n_test"]
        mar = f"{r['mean_abs_ret_bps']:.2f}"

        acc = f"{r['acc_all']:.4f}"
        acc30 = (f"{r['acc_top30']:.4f}"
                 if r["acc_top30"] is not None else "  ---  ")
        acc10 = (f"{r['acc_top10']:.4f}"
                 if r["acc_top10"] is not None else "  ---  ")

        edge = f"{r['edge_all_bps']:.2f}"
        net = f"{r['net_edge_all_bps']:+.2f}"
        e30 = (f"{r['edge_top30_bps']:.2f}"
               if r["edge_top30_bps"] is not None else "  ---  ")
        ne30 = (f"{r['net_edge_top30_bps']:+.2f}"
                if r["net_edge_top30_bps"] is not None else "    ---    ")

        print(
            f"{tf:>4}  {bars:>7}  {events:>7}  "
            f"{mar:>11}  "
            f"{acc:>9}  {acc30:>9}  {acc10:>9}  "
            f"{edge:>7}  {net:>8}  "
            f"{e30:>8}  {ne30:>11}",
            flush=True,
        )

    print("=" * 140, flush=True)
    print(f"  All edges in bps. RT cost = {RT_COST_BPS:.1f} bps.", flush=True)
    print(f"  Edge = (acc - 0.5) * 2 * MeanAbsRet.", flush=True)
    print(f"  NetEdge = Edge - {RT_COST_BPS:.1f} bps.", flush=True)


def print_breakeven_table(results: list[dict]) -> None:
    """Print the critical insight: breakeven accuracy at each timeframe."""
    print("\n" + "=" * 90, flush=True)
    print("CRITICAL INSIGHT: Minimum Accuracy to Break Even at Each Timeframe",
          flush=True)
    print("=" * 90, flush=True)
    print(f"  Formula: min_acc = 0.5 + {RT_COST_BPS:.1f} / "
          f"(2 * MeanAbsRetBps)", flush=True)
    print(f"  At what timeframe does the mean move become large enough "
          f"to overcome costs?", flush=True)
    print()

    header = (f"{'TF':>4}  {'MeanAbsRet(bps)':>16}  "
              f"{'MinAcc':>8}  {'ActualAcc(all)':>15}  "
              f"{'ActualAcc(30%)':>15}  {'Margin(all)':>12}  "
              f"{'Verdict':>10}")
    print(header, flush=True)
    print("-" * 90, flush=True)

    for r in results:
        tf = r["timeframe"]
        mar = r["mean_abs_ret_bps"]
        min_acc = r["breakeven_acc"]
        actual_all = r["acc_all"]
        actual_30 = r["acc_top30"]
        margin = actual_all - min_acc

        if margin > 0.02:
            verdict = "STRONG"
        elif margin > 0:
            verdict = "MARGINAL"
        else:
            verdict = "BELOW"

        acc30_s = (f"{actual_30:.4f}" if actual_30 is not None
                   else "     ---      ")

        print(
            f"{tf:>4}  {mar:>16.2f}  "
            f"{min_acc:>8.4f}  {actual_all:>15.4f}  "
            f"{acc30_s:>15}  {margin:>+12.4f}  "
            f"{verdict:>10}",
            flush=True,
        )

    print()
    print("  STRONG   = accuracy > breakeven + 2%  "
          "(confident edge after costs)", flush=True)
    print("  MARGINAL = accuracy > breakeven but < +2%  "
          "(edge may not survive slippage)", flush=True)
    print("  BELOW    = accuracy < breakeven  "
          "(no edge after costs)", flush=True)


def print_filtered_edge_table(results: list[dict]) -> None:
    """Print confidence-filtered net edges across timeframes."""
    print("\n" + "=" * 110, flush=True)
    print(f"CONFIDENCE-FILTERED NET EDGES "
          f"(bps after {RT_COST_BPS:.0f} bps RT cost)", flush=True)
    print("=" * 110, flush=True)

    header = (
        f"{'TF':>4}  {'NetEdge(all)':>13}  {'NetEdge(30%)':>13}  "
        f"{'NetEdge(20%)':>13}  {'NetEdge(10%)':>13}  "
        f"{'BestFilter':>14}"
    )
    print(header, flush=True)
    print("-" * 110, flush=True)

    for r in results:
        tf = r["timeframe"]
        ne_all = r["net_edge_all_bps"]
        ne_30 = r["net_edge_top30_bps"]
        ne_20 = r["net_edge_top20_bps"]
        ne_10 = r["net_edge_top10_bps"]

        ne_all_s = f"{ne_all:>+.2f}"
        ne_30_s = (f"{ne_30:>+.2f}" if ne_30 is not None else "      ---    ")
        ne_20_s = (f"{ne_20:>+.2f}" if ne_20 is not None else "      ---    ")
        ne_10_s = (f"{ne_10:>+.2f}" if ne_10 is not None else "      ---    ")

        # Find best filter
        candidates = [("all", ne_all)]
        if ne_30 is not None:
            candidates.append(("top30%", ne_30))
        if ne_20 is not None:
            candidates.append(("top20%", ne_20))
        if ne_10 is not None:
            candidates.append(("top10%", ne_10))
        best_label, best_val = max(
            candidates,
            key=lambda x: x[1] if x[1] is not None else -999
        )
        best_s = f"{best_label}={best_val:+.1f}"

        print(
            f"{tf:>4}  {ne_all_s:>13}  {ne_30_s:>13}  "
            f"{ne_20_s:>13}  {ne_10_s:>13}  "
            f"{best_s:>14}",
            flush=True,
        )

    print("=" * 110, flush=True)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    t_start = datetime.now()
    print("=" * 70, flush=True)
    print("DIRECTION V2 — MULTI-TIMEFRAME ANALYSIS", flush=True)
    print(f"Started: {t_start}", flush=True)
    print(f"Train cutoff: {TRAIN_CUTOFF}", flush=True)
    print(f"CUSUM: mult={CUSUM_MULT}, vol_span={CUSUM_VOL_SPAN}", flush=True)
    print(f"Sample weights: uniqueness + time decay "
          f"(clf_last_w={TIME_DECAY_CLF_LAST_W})", flush=True)
    print(f"XGBoost: n_est={XGBOOST_PARAMS['n_estimators']}, "
          f"depth={XGBOOST_PARAMS['max_depth']}, "
          f"lr={XGBOOST_PARAMS['learning_rate']}", flush=True)
    print(f"RT cost: {RT_COST_BPS} bps", flush=True)
    print(f"Timeframes: {[tf[1] for tf in TIMEFRAMES]}", flush=True)
    print("=" * 70, flush=True)

    # Load 1m data once, resample per timeframe
    print("\n[1] Loading BTC 1m data...", flush=True)
    btc_1m = load_btc_1m()
    print(f"  {len(btc_1m)} 1m bars "
          f"({btc_1m.index[0].date()} to {btc_1m.index[-1].date()})",
          flush=True)

    results = []

    for i, (tf_resample, tf_label, tf_mins) in enumerate(TIMEFRAMES):
        print(f"\n[{i+2}] Resampling to {tf_label}...", flush=True)
        ohlcv = resample_ohlcv(btc_1m, tf_resample)
        print(f"  {len(ohlcv)} {tf_label} bars", flush=True)

        t_tf_start = datetime.now()
        result = evaluate_timeframe(ohlcv, tf_label, tf_mins)
        t_tf_elapsed = datetime.now() - t_tf_start
        print(f"  {tf_label} completed in {t_tf_elapsed}", flush=True)

        if result is not None:
            results.append(result)

    # ===== Output tables =====
    if not results:
        print("\nFATAL: No timeframe produced results.", flush=True)
        return

    print_main_table(results)
    print_breakeven_table(results)
    print_filtered_edge_table(results)

    # ===== Final strategic summary =====
    print("\n" + "=" * 70, flush=True)
    print("STRATEGIC SUMMARY", flush=True)
    print("=" * 70, flush=True)

    # Returns scaling with timeframe
    print(f"\n  RETURNS SCALING (MeanAbsRet vs {RT_COST_BPS:.0f} bps cost):",
          flush=True)
    for r in results:
        ratio = r["mean_abs_ret_bps"] / RT_COST_BPS
        print(f"    {r['timeframe']:>4}: {r['mean_abs_ret_bps']:>7.2f} bps = "
              f"{ratio:.1f}x trading cost", flush=True)

    # Best net edge (all predictions)
    best_all = max(results, key=lambda r: r["net_edge_all_bps"])
    print(f"\n  Best net edge (all predictions):", flush=True)
    print(f"    {best_all['timeframe']}: "
          f"{best_all['net_edge_all_bps']:+.2f} bps "
          f"(acc={best_all['acc_all']:.4f}, "
          f"MeanAbsRet={best_all['mean_abs_ret_bps']:.2f} bps)", flush=True)

    # Best net edge (top 30%)
    candidates_30 = [r for r in results
                     if r["net_edge_top30_bps"] is not None]
    if candidates_30:
        best_30 = max(candidates_30,
                      key=lambda r: r["net_edge_top30_bps"])
        print(f"\n  Best net edge (top 30% confidence):", flush=True)
        print(f"    {best_30['timeframe']}: "
              f"{best_30['net_edge_top30_bps']:+.2f} bps "
              f"(acc_30%={best_30['acc_top30']:.4f})", flush=True)

    # Profitable timeframes
    profitable_all = [r for r in results if r["net_edge_all_bps"] > 0]
    profitable_30 = [r for r in results
                     if r.get("net_edge_top30_bps") is not None
                     and r["net_edge_top30_bps"] > 0]

    print(f"\n  Profitable timeframes (all):   "
          f"{[r['timeframe'] for r in profitable_all] if profitable_all else 'NONE'}",
          flush=True)
    print(f"  Profitable timeframes (top30): "
          f"{[r['timeframe'] for r in profitable_30] if profitable_30 else 'NONE'}",
          flush=True)

    # Key theoretical insight
    print(f"\n  KEY INSIGHT:", flush=True)
    print(f"  At 55% accuracy: edge = 0.10 * MeanAbsRet", flush=True)
    print(f"  Need 0.10 * MeanAbsRet > {RT_COST_BPS} bps", flush=True)
    print(f"  => MeanAbsRet > {RT_COST_BPS / 0.10:.0f} bps "
          f"for 55% accuracy to be profitable", flush=True)
    print(f"  At 54% accuracy: need MeanAbsRet > "
          f"{RT_COST_BPS / 0.08:.0f} bps", flush=True)
    print(f"  At 53% accuracy: need MeanAbsRet > "
          f"{RT_COST_BPS / 0.06:.0f} bps", flush=True)

    # ===== Save results =====
    def jsonify(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        "experiment": "direction_v2_timeframes",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "train_cutoff": TRAIN_CUTOFF,
            "cusum_mult": CUSUM_MULT,
            "cusum_vol_span": CUSUM_VOL_SPAN,
            "time_decay_clf_last_w": TIME_DECAY_CLF_LAST_W,
            "xgboost_params": XGBOOST_PARAMS,
            "rt_cost_bps": RT_COST_BPS,
            "timeframes": [tf[1] for tf in TIMEFRAMES],
        },
        "results": [{k: jsonify(v) for k, v in r.items()} for r in results],
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}", flush=True)

    t_elapsed = datetime.now() - t_start
    print(f"\nTotal runtime: {t_elapsed}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
