"""
B5: Synthetic Data Test — Run the full pipeline on random noise.

If the system finds Sharpe > 0.3 on synthetic data, our methodology has
an overfitting problem. If Sharpe < 0.3, the pipeline correctly finds
no signal in noise, confirming the real-data results are genuine.

Tests 3 random seeds for robustness.
"""

import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

SEEDS = [42, 123, 999]
N_BARS = 70_000  # ~same as 15m data for one asset over 24 months
PASS_THRESHOLD = 0.3   # Sharpe must be below this on ALL seeds
FAIL_THRESHOLD = 0.5   # If ANY seed exceeds this, methodology is broken

# Model parameters (same as real pipeline)
MODEL_C = 0.1
MODEL_PENALTY = "l1"
MODEL_SOLVER = "saga"
MODEL_MAX_ITER = 1000
MODEL_SEED = 42

# Portfolio parameters (same as real pipeline)
STARTING_CAPITAL = 10000.0
MAX_DD = 0.15
DELEVER_START = 0.075
SLIPPAGE = 0.0001  # 1 bps
MAX_EXPOSURE = 0.60
MAX_PER_ASSET = 0.20
BASE_SIZING_PCT = 0.02
MAX_SIGNAL_MULT = 2.0

# Filters
VOL_FILTER_QUANTILE = 0.50

# Winsorization
WINSOR_LO = 0.01
WINSOR_HI = 0.99

# Signal discretization
SIGNAL_STEP_SIZE = 0.1

FEATURES_14 = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low", "dist_nearest_level",
]

RESULTS_DIR = Path("C:/Claude/ml-trading-system/v4-results")


# =============================================================================
# INLINE TECHNICAL INDICATORS (copied from multi_approach_comparison.py)
# =============================================================================

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")


def stochastic_rsi(close: pd.Series, rsi_period: int = 14,
                   stoch_period: int = 14) -> pd.Series:
    rsi_val = rsi(close, rsi_period)
    rsi_min = rsi_val.rolling(stoch_period).min()
    rsi_max = rsi_val.rolling(stoch_period).max()
    rsi_range = rsi_max - rsi_min
    stoch = (rsi_val - rsi_min) / rsi_range.replace(0, np.nan) * 100
    return stoch.rename("stoch_rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
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
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    band_width = upper - lower
    pct_b = (close - lower) / band_width.replace(0, np.nan)
    return pct_b.rename("bb_pct_b")


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return (tr.ewm(span=period, adjust=False).mean() / close).rename(f"atr_{period}")


def swing_levels(high: pd.Series, low: pd.Series,
                 lookback: int = 20) -> pd.DataFrame:
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
        axis=1,
    ).min(axis=1)
    return result


# =============================================================================
# SIGNAL / SIZING HELPERS (copied from multi_approach_comparison.py)
# =============================================================================

def get_signal(probability, num_classes=2):
    p = np.asarray(probability, dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = (p - 1.0 / num_classes) / np.sqrt(p * (1.0 - p))
    signal = 2.0 * norm.cdf(z) - 1.0
    return float(signal) if np.ndim(probability) == 0 else signal


def discretize_signal(signal, step_size=0.1):
    return np.round(np.asarray(signal) / step_size) * step_size


def compute_sample_weights(timestamps: np.ndarray, fwd_bars: int) -> np.ndarray:
    n = len(timestamps)
    if n == 0:
        return np.array([])
    c = min(fwd_bars + 1, n)
    weights = np.full(n, 1.0 / c, dtype=np.float64)
    for i in range(min(fwd_bars, n)):
        weights[i] = 1.0 / (i + 1)
        if n - 1 - i != i:
            weights[n - 1 - i] = 1.0 / (i + 1)
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    return weights


# =============================================================================
# FEATURE COMPUTATION (same as real pipeline)
# =============================================================================

def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]

    f = pd.DataFrame(index=ohlcv.index)

    for lb in [1, 3, 7, 14]:
        f[f"ret_{lb}"] = close.pct_change(lb)

    f["bar_range"] = (high - low) / close.replace(0, np.nan)
    vol_mean = volume.rolling(30, min_periods=10).mean()
    vol_std = volume.rolling(30, min_periods=10).std()
    if vol_std.max() > 0:
        f["vol_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
    else:
        f["vol_zscore"] = 0.0

    f["rsi_14"] = rsi(close, 14)
    f["stoch_rsi"] = stochastic_rsi(close, 14, 14)
    f["macd_hist"] = macd(close)["macd_hist"]
    f["bb_pct_b"] = bollinger_pct_b(close, 20)
    f["atr_14"] = atr(high, low, close, 14)

    sw = swing_levels(high, low, lookback=20)
    f["dist_swing_high"] = sw["dist_swing_high"]
    f["dist_swing_low"] = sw["dist_swing_low"]
    f["dist_nearest_level"] = sw["dist_nearest_level"]

    for col in f.columns:
        f[col] = f[col].astype(np.float32)

    return f


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_ohlcv(seed, n_bars=N_BARS):
    """Generate random walk OHLCV with NO directional signal."""
    np.random.seed(seed)

    # Random walk returns (mean=0, no drift, no signal)
    returns = np.random.normal(0, 0.001, n_bars)
    prices = 100 * np.exp(np.cumsum(returns))

    # Synthetic OHLCV
    noise_o = np.random.normal(0, 0.0003, n_bars)
    noise_h = np.abs(np.random.normal(0, 0.0008, n_bars))
    noise_l = np.abs(np.random.normal(0, 0.0008, n_bars))

    ohlcv = pd.DataFrame({
        "open": prices * (1 + noise_o),
        "high": prices * (1 + noise_h),
        "low": prices * (1 - noise_l),
        "close": prices,
        "volume": np.random.lognormal(10, 1, n_bars),
    }, index=pd.date_range("2024-03-01", periods=n_bars, freq="15min", tz="UTC"))

    # Ensure high >= max(open, close) and low <= min(open, close)
    ohlcv["high"] = ohlcv[["open", "high", "close"]].max(axis=1)
    ohlcv["low"] = ohlcv[["open", "low", "close"]].min(axis=1)

    return ohlcv


# =============================================================================
# SIGNAL GENERATION (walk-forward, same as real pipeline)
# =============================================================================

def generate_raw_signals(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    fwd_bars: int,
    asset_name: str,
    feature_cols: list,
) -> pd.DataFrame:
    """Walk-forward monthly retrain with expanding window. No confidence filter."""
    close = ohlcv["close"]
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)

    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    start_date = ohlcv.index.min()
    end_date = ohlcv.index.max()

    if (end_date - start_date).days < 180:
        return pd.DataFrame()

    first_test_month = start_date + pd.DateOffset(months=6)
    all_months = pd.period_range(
        start=first_test_month.to_period("M"),
        end=end_date.to_period("M"),
        freq="M",
    )

    if len(all_months) == 0:
        return pd.DataFrame()

    retrain_every = 1
    if len(ohlcv) > 500_000:
        retrain_every = 3
    elif len(ohlcv) > 200_000:
        retrain_every = 2

    if len(ohlcv) > 1:
        median_gap = ohlcv.index.to_series().diff().dropna().median()
        embargo_td = median_gap * fwd_bars
    else:
        embargo_td = pd.Timedelta(hours=2)

    rows = []
    cached_model = None
    cached_scaler = None
    last_retrain_idx = -999

    for month_idx, month in enumerate(all_months):
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")

        if features.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

        embargo_cutoff = ms - embargo_td
        train_cutoff = embargo_cutoff

        feat_cols = [c for c in feature_cols if c in features.columns]
        if len(feat_cols) < 8:
            continue

        feat = features[feat_cols].copy()

        # Per-window winsorization
        train_mask = feat.index < train_cutoff
        for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
            if col in feat.columns:
                train_vals = feat.loc[train_mask, col]
                if len(train_vals.dropna()) > 10:
                    lo = train_vals.quantile(WINSOR_LO)
                    hi = train_vals.quantile(WINSOR_HI)
                    feat[col] = feat[col].clip(lo, hi)

        fc = feat.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        tr = valid[valid < train_cutoff]
        te = valid[(valid >= ms) & (valid <= me)]

        if len(tr) < 200 or len(te) < 10:
            continue

        # Vol filter
        if vol is not None and len(vol) > 0:
            tv = vol.loc[vol.index.intersection(tr)].dropna()
            if len(tv) < 20:
                continue
            vt = tv.quantile(VOL_FILTER_QUANTILE)
            vol_aligned = vol.reindex(te)
            hv = (vol_aligned > vt).fillna(False)
            te = te[hv.reindex(te).fillna(False).values]

        if len(te) < 5:
            continue

        X_train = fc.loc[tr]
        y_train = (y_all.loc[tr] > 0).astype(int)
        X_test = fc.loc[te]

        need_retrain = (month_idx - last_retrain_idx) >= retrain_every or cached_model is None

        if need_retrain:
            train_timestamps = X_train.index.values.astype(np.int64)
            sample_w = compute_sample_weights(train_timestamps, fwd_bars=fwd_bars)

        try:
            if need_retrain:
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_train)
                model = LogisticRegression(
                    C=MODEL_C, penalty=MODEL_PENALTY, solver=MODEL_SOLVER,
                    max_iter=MODEL_MAX_ITER, random_state=MODEL_SEED,
                )
                model.fit(X_tr_scaled, y_train, sample_weight=sample_w)
                cached_model = model
                cached_scaler = scaler
                last_retrain_idx = month_idx
            else:
                scaler = cached_scaler
                model = cached_model

            X_te_scaled = scaler.transform(X_test)
            yp = model.predict_proba(X_te_scaled)[:, 1]
        except Exception:
            continue

        conf = np.abs(yp - 0.5)
        fwd_ret_vals = fwd_ret.reindex(te)

        directions = np.where(yp > 0.5, 1.0, -1.0)
        signals = get_signal(yp, num_classes=2)
        signals_disc = discretize_signal(signals, step_size=SIGNAL_STEP_SIZE)

        fr_vals = fwd_ret_vals.values if hasattr(fwd_ret_vals, "values") else np.array(fwd_ret_vals)
        finite_mask = np.isfinite(fr_vals)
        if finite_mask.any():
            sig_str = signals_disc if np.ndim(signals_disc) > 0 else np.full(len(yp), float(signals_disc))
            batch = pd.DataFrame({
                "timestamp": te[finite_mask],
                "direction": directions[finite_mask],
                "fwd_return": fr_vals[finite_mask],
                "confidence": conf[finite_mask],
                "probability": yp[finite_mask],
                "signal_strength": np.abs(sig_str[finite_mask]),
            })
            rows.append(batch)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["asset"] = asset_name
    return df


# =============================================================================
# CONFIDENCE FILTER (percentile-70, same as Approach A)
# =============================================================================

def apply_confidence_filter(signals_df: pd.DataFrame, percentile: float = 70.0) -> pd.DataFrame:
    if signals_df is None or len(signals_df) == 0:
        return pd.DataFrame()
    df = signals_df.copy()
    conf = df["confidence"].values
    if len(conf) > 5:
        conf_threshold = np.percentile(conf, percentile)
        mask = conf >= conf_threshold
    else:
        mask = np.ones(len(conf), dtype=bool)
    return df[mask].reset_index(drop=True)


# =============================================================================
# PORTFOLIO SIMULATION (same as real pipeline)
# =============================================================================

def simulate_portfolio(signals_df: pd.DataFrame) -> Optional[Dict]:
    if signals_df is None or len(signals_df) < 10:
        return None

    sub = signals_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sub)

    timestamps = sub["timestamp"].values
    assets_arr = sub["asset"].values if "asset" in sub.columns else np.full(n, "SINGLE")
    directions = sub["direction"].values.astype(np.float64)
    fwd_returns = sub["fwd_return"].values.astype(np.float64)
    signal_strengths = (
        sub["signal_strength"].values.astype(np.float64)
        if "signal_strength" in sub.columns else None
    )

    equity_arr = np.empty(n)
    pnl_arr = np.zeros(n)
    dd_arr = np.empty(n)
    size_arr = np.zeros(n)
    skipped = np.ones(n, dtype=bool)

    equity = STARTING_CAPITAL
    peak_equity = equity
    halted = False

    current_exposure = 0.0
    asset_exposure: dict = {}
    use_prob = signal_strengths is not None

    for i in range(n):
        if halted:
            equity_arr[i] = equity
            dd_arr[i] = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            continue

        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

        if dd >= MAX_DD:
            halted = True
            equity_arr[i] = equity
            dd_arr[i] = dd
            continue

        if dd > DELEVER_START:
            dd_ratio = (dd - DELEVER_START) / (MAX_DD - DELEVER_START)
            size_mult = max(1.0 - dd_ratio, 0.0)
        else:
            size_mult = 1.0

        if use_prob:
            sig_abs = abs(signal_strengths[i])
            raw_size = equity * BASE_SIZING_PCT * sig_abs * size_mult
            raw_size = min(raw_size, equity * BASE_SIZING_PCT * MAX_SIGNAL_MULT * size_mult)
        else:
            raw_size = equity * BASE_SIZING_PCT * size_mult

        asset = assets_arr[i]
        asset_exp = asset_exposure.get(asset, 0.0)

        if asset_exp + raw_size > equity * MAX_PER_ASSET:
            raw_size = max(equity * MAX_PER_ASSET - asset_exp, 0.0)

        if current_exposure + raw_size > equity * MAX_EXPOSURE:
            raw_size = max(equity * MAX_EXPOSURE - current_exposure, 0.0)

        if raw_size < 1.0:
            equity_arr[i] = equity
            dd_arr[i] = dd
            continue

        trade_return = directions[i] * fwd_returns[i] - SLIPPAGE
        pnl = raw_size * trade_return

        equity += pnl
        if equity > peak_equity:
            peak_equity = equity

        current_exposure = min(
            current_exposure + raw_size * 0.1, equity * MAX_EXPOSURE
        )
        if i > 0 and timestamps[i] != timestamps[i - 1]:
            current_exposure *= 0.9
        asset_exposure[asset] = asset_exposure.get(asset, 0.0) * 0.9

        equity_arr[i] = equity
        pnl_arr[i] = pnl
        dd_arr[i] = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        size_arr[i] = raw_size
        skipped[i] = False

    return {
        "equity": equity_arr,
        "pnl": pnl_arr,
        "dd": dd_arr,
        "sizes": size_arr,
        "skipped": skipped,
        "timestamps": timestamps,
        "halted": halted,
        "final_equity": equity,
        "n_trades": int((~skipped).sum()),
    }


# =============================================================================
# STATS COMPUTATION
# =============================================================================

def compute_stats(sim: Dict) -> Dict[str, Any]:
    """Compute Sharpe, return%, and trade count from simulation."""
    if sim is None or sim["n_trades"] < 10:
        return {"sharpe": 0.0, "total_return_pct": 0.0, "n_trades": 0, "error": "insufficient_trades"}

    active = ~sim["skipped"]
    pnls = sim["pnl"][active]
    timestamps = sim["timestamps"][active]

    ts_series = pd.Series(pnls, index=pd.to_datetime(timestamps))
    daily = ts_series.resample("1D").sum()
    t0 = pd.Timestamp(timestamps[0])
    daily = daily[daily.index >= t0]
    daily_nonzero = daily[daily != 0]

    if len(daily_nonzero) < 5:
        return {"sharpe": 0.0, "total_return_pct": 0.0, "n_trades": sim["n_trades"], "error": "insufficient_days"}

    daily_ret = daily_nonzero / STARTING_CAPITAL
    mean_r = daily_ret.mean()
    std_r = daily_ret.std()
    sharpe = mean_r / std_r * np.sqrt(365.25) if std_r > 0 else 0.0

    total_ret = (sim["final_equity"] - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    return {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_ret, 2),
        "n_trades": sim["n_trades"],
        "final_equity": round(sim["final_equity"], 2),
        "halted": sim["halted"],
        "n_active_days": len(daily_nonzero),
    }


# =============================================================================
# MAIN — Run synthetic data test
# =============================================================================

def main():
    print("=" * 56)
    print("SYNTHETIC DATA TEST (B5)")
    print("=" * 56)
    print(f"Seeds: {SEEDS}")
    print(f"N bars per seed: {N_BARS:,}")
    print(f"Pass threshold: Sharpe < {PASS_THRESHOLD}")
    print(f"Fail threshold: Sharpe > {FAIL_THRESHOLD}")
    print()

    results = []
    t0 = time.time()

    for seed in SEEDS:
        print(f"--- Seed {seed} ---")
        t_seed = time.time()

        # 1. Generate synthetic OHLCV
        print(f"  Generating {N_BARS:,} bars of random-walk OHLCV...")
        ohlcv = generate_synthetic_ohlcv(seed, N_BARS)
        print(f"  Date range: {ohlcv.index.min().date()} to {ohlcv.index.max().date()}")

        # 2. Compute features
        print("  Computing 14 features...")
        features = compute_features(ohlcv)
        print(f"  Features shape: {features.shape}")

        # 3. Generate raw signals (walk-forward, fwd_bars=4)
        print("  Running walk-forward signal generation (fwd_bars=4)...")
        raw_signals = generate_raw_signals(
            ohlcv, features, fwd_bars=4,
            asset_name=f"SYNTHETIC_{seed}",
            feature_cols=FEATURES_14,
        )
        print(f"  Raw signals: {len(raw_signals):,}")

        if len(raw_signals) == 0:
            print("  WARNING: No signals generated!")
            results.append({
                "seed": seed, "sharpe": 0.0, "total_return_pct": 0.0,
                "n_trades": 0, "status": "NO_SIGNALS",
            })
            continue

        # 4. Apply percentile-70 confidence filter
        filtered = apply_confidence_filter(raw_signals, percentile=70.0)
        print(f"  After p70 filter: {len(filtered):,} signals")

        # 5. Simulate portfolio
        print("  Simulating portfolio...")
        sim = simulate_portfolio(filtered)
        stats = compute_stats(sim)

        sharpe = stats["sharpe"]
        ret_pct = stats["total_return_pct"]
        n_trades = stats["n_trades"]

        if abs(sharpe) > FAIL_THRESHOLD:
            status = "FAIL"
        elif abs(sharpe) > PASS_THRESHOLD:
            status = "WARN"
        else:
            status = "PASS"

        elapsed = time.time() - t_seed
        print(f"  Sharpe={sharpe:.3f}  Return={ret_pct:.2f}%  Trades={n_trades}  Status={status}  ({elapsed:.1f}s)")

        results.append({
            "seed": seed,
            "sharpe": sharpe,
            "total_return_pct": ret_pct,
            "n_trades": n_trades,
            "n_raw_signals": len(raw_signals),
            "n_filtered_signals": len(filtered),
            "status": status,
            "halted": stats.get("halted", False),
            "final_equity": stats.get("final_equity", STARTING_CAPITAL),
            "n_active_days": stats.get("n_active_days", 0),
            "elapsed_s": round(elapsed, 1),
        })
        print()

    total_elapsed = time.time() - t0

    # Summary
    sharpes = [r["sharpe"] for r in results]
    mean_sharpe = np.mean(sharpes) if sharpes else 0.0
    max_sharpe = max(abs(s) for s in sharpes) if sharpes else 0.0

    any_fail = any(abs(r["sharpe"]) > FAIL_THRESHOLD for r in results)
    all_pass = all(abs(r["sharpe"]) < PASS_THRESHOLD for r in results)
    overall = "PASS" if all_pass else ("FAIL" if any_fail else "WARN")

    print()
    print("SYNTHETIC DATA TEST (B5)")
    print("=" * 56)
    print(f"{'Seed':<6} {'Sharpe':>8} {'Return%':>9} {'Trades':>8} {'Status':>8}")
    print("-" * 56)
    for r in results:
        print(f"{r['seed']:<6} {r['sharpe']:>8.3f} {r['total_return_pct']:>8.2f}% {r['n_trades']:>8} {r['status']:>8}")
    print("-" * 56)
    print(f"Mean Sharpe: {mean_sharpe:.3f}")
    print(f"Max |Sharpe|: {max_sharpe:.3f}")
    print(f"OVERALL: {overall}")
    print(f"Total time: {total_elapsed:.1f}s")

    # Save results
    output = {
        "test": "B5_synthetic_data",
        "description": "Full pipeline on random-walk noise — tests for overfitting",
        "pass_threshold": PASS_THRESHOLD,
        "fail_threshold": FAIL_THRESHOLD,
        "n_bars": N_BARS,
        "seeds": SEEDS,
        "results": results,
        "summary": {
            "mean_sharpe": round(mean_sharpe, 3),
            "max_abs_sharpe": round(max_sharpe, 3),
            "overall": overall,
            "total_elapsed_s": round(total_elapsed, 1),
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    out_path = RESULTS_DIR / "synthetic_data_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
