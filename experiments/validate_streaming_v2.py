"""Paranoid validation of Sharpe ~20 signal.

12 diagnostic tests to determine whether the extraordinary performance
of the streaming backtest v2 is genuine or an artifact.

Tests ranked by diagnostic value:
  1. Causal confidence filter (replaces monthly batch lookahead)
  2. Component isolation (regime, retrain, winsor, confidence)
  3. Direction shuffle (is model prediction necessary?)
  4. Random features (do features matter?)
  5. Non-overlapping trades (overlap inflation?)
  6. Sub-period stability (monthly/quarterly Sharpe)
  7. Cost sensitivity (breakeven slippage)
  8. Time-reversed backtest (is signal causal?)
  9. Cross-asset (ETH, SOL, BNB)
 10. Bootstrap CI (statistical confidence)
 11. DSR (multiple-testing correction)
 12. Direction bias (long-only proxy?)
"""

import sys
sys.path.insert(0, "src")

import logging
import os
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from afml.bet_sizing import discretize_signal, get_signal
from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.streaming_backtest import BacktestResult, Signal, StreamingBacktester, Trade
from signal_engine.strategies import AlwaysLongStrategy, RandomStrategy
from signal_engine.weights import compute_sample_weights

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)

LOOKBACK_MONTHS = 8
FWD_BARS = 4
WINSOR_COLS = ["ret_1", "ret_3", "ret_7", "ret_14"]
N_JOBS = max(1, os.cpu_count() or 1)


# ============================================================
# Strategy variants
# ============================================================

class ReportAlgorithmStrategy:
    """Exact replica of old pipeline: regime features, vol filter, winsorization."""

    def __init__(self, vol_filter_quantile=0.50, shuffle_direction=False,
                 use_random_features=False, rng_seed=42):
        self.model = None
        self._winsor_bounds = {}
        self.vol_threshold = 0.0
        self.vol_filter_quantile = vol_filter_quantile
        self.shuffle_direction = shuffle_direction
        self.use_random_features = use_random_features
        self._rng = np.random.RandomState(rng_seed)

    def on_train(self, bar_idx, features, labels, vol_series):
        n = len(features)
        if n < 200:
            return

        if len(vol_series) > 50:
            self.vol_threshold = float(vol_series.quantile(self.vol_filter_quantile))

        feat_train = features.copy()
        for col in WINSOR_COLS:
            if col in feat_train.columns:
                arr = feat_train[col].to_numpy(dtype=np.float32)
                lo, hi = np.nanquantile(arr, [0.01, 0.99])
                feat_train[col] = np.clip(arr, lo, hi)
                self._winsor_bounds[col] = (float(lo), float(hi))

        y = labels.to_numpy(dtype=np.int8)

        ts_idx = features.index
        months = pd.DatetimeIndex(ts_idx).to_period("M") if not hasattr(ts_idx, "to_period") else ts_idx.to_period("M")
        unique_months = months.unique()
        fit_mask = np.ones(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        if len(unique_months) >= 2:
            val_month = unique_months.max()
            val_mask = np.asarray(months == val_month)
            fit_mask = ~val_mask
            if fit_mask.sum() < 200 or val_mask.sum() < 50:
                fit_mask[:] = True
                val_mask[:] = False

        X_train = feat_train.to_numpy(dtype=np.float32)
        if self.use_random_features:
            X_train = self._rng.randn(*X_train.shape).astype(np.float32)

        fit_X = X_train[fit_mask]
        fit_y = y[fit_mask]
        fit_sw = compute_sample_weights(features.index[fit_mask].values.astype(np.int64), fwd_bars=FWD_BARS)

        self.model = xgb.XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            reg_alpha=0.1, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=42, n_jobs=N_JOBS, verbosity=0,
            early_stopping_rounds=25 if val_mask.any() else None,
        )
        kwargs = {"sample_weight": fit_sw, "verbose": False}
        if val_mask.any():
            val_X = X_train[val_mask] if not self.use_random_features else self._rng.randn(val_mask.sum(), X_train.shape[1]).astype(np.float32)
            kwargs["eval_set"] = [(val_X, y[val_mask])]
        self.model.fit(fit_X, fit_y, **kwargs)

    def on_bar(self, bar_idx, feature_row, vol):
        if self.model is None:
            return None
        if vol < self.vol_threshold:
            return None

        row = feature_row.copy()
        for col, (lo, hi) in self._winsor_bounds.items():
            if col in row.index:
                row[col] = np.clip(row[col], lo, hi)

        X = row.to_numpy(dtype=np.float32).reshape(1, -1)
        if self.use_random_features:
            X = self._rng.randn(*X.shape).astype(np.float32)

        if not np.isfinite(X).all():
            return None

        prob = float(self.model.predict_proba(X)[0, 1])
        conf = abs(prob - 0.5)
        direction = 1.0 if prob > 0.5 else -1.0

        if self.shuffle_direction:
            direction = 1.0 if self._rng.rand() > 0.5 else -1.0

        sig = get_signal(np.array([prob]), num_classes=2)
        sig_disc = discretize_signal(sig, step_size=0.1)
        strength = float(abs(sig_disc[0])) if np.ndim(sig_disc) > 0 else float(abs(sig_disc))
        return Signal(direction=direction, probability=prob, confidence=conf, signal_strength=strength)


class NoWinsorStrategy(ReportAlgorithmStrategy):
    """Same as ReportAlgorithm but no winsorization."""
    def on_train(self, bar_idx, features, labels, vol_series):
        n = len(features)
        if n < 200:
            return
        if len(vol_series) > 50:
            self.vol_threshold = float(vol_series.quantile(self.vol_filter_quantile))
        self._winsor_bounds = {}
        y = labels.to_numpy(dtype=np.int8)
        ts_idx = features.index
        months = pd.DatetimeIndex(ts_idx).to_period("M") if not hasattr(ts_idx, "to_period") else ts_idx.to_period("M")
        unique_months = months.unique()
        fit_mask = np.ones(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        if len(unique_months) >= 2:
            val_month = unique_months.max()
            val_mask = np.asarray(months == val_month)
            fit_mask = ~val_mask
            if fit_mask.sum() < 200 or val_mask.sum() < 50:
                fit_mask[:] = True
                val_mask[:] = False
        fit_X = features.to_numpy(dtype=np.float32)[fit_mask]
        fit_y = y[fit_mask]
        fit_sw = compute_sample_weights(features.index[fit_mask].values.astype(np.int64), fwd_bars=FWD_BARS)
        self.model = xgb.XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            reg_alpha=0.1, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=42, n_jobs=N_JOBS, verbosity=0,
            early_stopping_rounds=25 if val_mask.any() else None,
        )
        kwargs = {"sample_weight": fit_sw, "verbose": False}
        if val_mask.any():
            kwargs["eval_set"] = [(features.to_numpy(dtype=np.float32)[val_mask], y[val_mask])]
        self.model.fit(fit_X, fit_y, **kwargs)


# ============================================================
# Helpers
# ============================================================

def compute_sharpe(result):
    return result.sharpe

def compute_sub_period_sharpes(result):
    """Compute Sharpe by year, quarter, month."""
    if not result.trades:
        return {}, {}, {}
    pnls = np.array([t.pnl for t in result.trades])
    entry_ts = pd.DatetimeIndex([t.entry_ts for t in result.trades])
    daily = pd.Series(5.0 * pnls, index=entry_ts).resample("1D").sum()
    daily = daily[daily != 0]

    def sharpe_of(s):
        if len(s) < 5:
            return 0.0
        std = s.std()
        return float(s.mean() / std * np.sqrt(365.25)) if std > 0 else 0.0

    yearly = {}
    for yr, grp in daily.groupby(daily.index.year):
        yearly[yr] = sharpe_of(grp)

    quarterly = {}
    for q, grp in daily.groupby(daily.index.to_period("Q")):
        quarterly[str(q)] = sharpe_of(grp)

    monthly = {}
    for m, grp in daily.groupby(daily.index.to_period("M")):
        monthly[str(m)] = sharpe_of(grp)

    return yearly, quarterly, monthly


def direction_stats(result):
    """What fraction of trades are long?"""
    if not result.trades:
        return 0.0, 0.0
    dirs = np.array([t.direction for t in result.trades])
    long_frac = float((dirs > 0).mean())
    return long_frac, 1.0 - long_frac


def bootstrap_sharpe_ci(result, n_boot=1000, block_size=5, seed=42):
    """Block bootstrap CI for Sharpe."""
    if result.n_days < 20:
        return 0.0, 0.0, 0.0
    daily = result.daily_pnl.values
    n = len(daily)
    rng = np.random.RandomState(seed)
    sharpes = []
    for _ in range(n_boot):
        # Block bootstrap
        n_blocks = n // block_size + 1
        starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        boot = np.concatenate([daily[s:s+block_size] for s in starts])[:n]
        s = boot.std()
        if s > 0 and len(boot) > 5:
            sr = float(boot.mean() / s * np.sqrt(365.25))
            sharpes.append(sr)
    if not sharpes:
        return 0.0, 0.0, 0.0
    sharpes = np.array(sharpes)
    return float(np.percentile(sharpes, 2.5)), float(np.median(sharpes)), float(np.percentile(sharpes, 97.5))


def deflated_sharpe_ratio(observed_sr, n_days, n_trials, skew=0.0, kurt=3.0):
    """DSR from ML for Asset Managers."""
    from scipy.stats import norm
    emc = 0.5772156649015328606065120900824024310421593359
    # Expected max SR under null
    e_max_sr = (1 - emc) * norm.ppf(1 - 1.0 / n_trials) + emc * norm.ppf(1 - 1.0 / (n_trials * np.e))
    # SR variance (non-normal returns)
    sr_var = (1 - skew * observed_sr + ((kurt - 1) / 4) * observed_sr ** 2) / (n_days - 1)
    if sr_var <= 0:
        return 1.0
    z = (observed_sr - e_max_sr) / np.sqrt(sr_var)
    return float(norm.cdf(z))


def run_test(name, engine, strategy, all_features, close, vol, timestamps, start_bar):
    """Run a single test and return result dict."""
    t0 = time.time()
    result = engine.run(strategy, all_features, close, vol, timestamps,
                        start_bar=start_bar, progress_every=0)
    elapsed = time.time() - t0
    long_frac, short_frac = direction_stats(result)
    return {
        "name": name,
        "sharpe": result.sharpe,
        "win_rate": result.win_rate,
        "mean_bps": result.mean_return_bps,
        "n_trades": result.n_trades,
        "n_days": result.n_days,
        "long_frac": long_frac,
        "elapsed": elapsed,
        "result": result,
    }


def fmt(r):
    return (f"  {r['name']:<45s} SR={r['sharpe']:>7.2f}  Win={r['win_rate']*100:>5.1f}%  "
            f"bps={r['mean_bps']:>6.1f}  n={r['n_trades']:>6d}  Long={r['long_frac']*100:.0f}%  "
            f"({r['elapsed']:.0f}s)")


# ============================================================
# Data loading (shared across all tests)
# ============================================================

def load_data():
    cfg = load_config(Path("configs/default.yaml"))
    data_dirs = [Path(p) for p in cfg.data.data_dirs] or [Path("data/binance-1m")]

    print("Loading BTC 1m data...")
    ohlcv_1m = load_binance_1m("BTCUSDT", data_dirs)
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")

    print("Computing features (base + regime)...")
    base_features = compute_features(ohlcv_15m, cfg.features)

    regime_parts = []
    for rtf in cfg.close_model.regime_timeframes:
        rtf_ohlcv = resample_ohlcv(ohlcv_1m, rtf)
        rtf_feat = compute_features(rtf_ohlcv, cfg.features)
        aligned = rtf_feat.reindex(ohlcv_15m.index, method="ffill")
        aligned = aligned.add_prefix(f"signal_regime_{rtf}_")
        regime_parts.append(aligned)

    if regime_parts:
        all_features = pd.concat([base_features] + regime_parts, axis=1)
    else:
        all_features = base_features

    close = ohlcv_15m["close"].to_numpy(dtype=np.float64)
    vol = ohlcv_15m["close"].pct_change().ewm(span=50, min_periods=20).std().to_numpy(dtype=np.float64)
    timestamps = ohlcv_15m.index

    start_ts = pd.Timestamp("2022-01-01", tz="UTC")
    if timestamps.tz is None:
        start_ts = start_ts.tz_localize(None)
    start_bar = int(timestamps.searchsorted(start_ts))

    print(f"  {all_features.shape[1]} features, {len(ohlcv_15m)} bars, start_bar={start_bar}")
    return cfg, all_features, base_features, close, vol, timestamps, start_bar, ohlcv_1m


# ============================================================
# Main
# ============================================================

def main():
    t_global = time.time()
    cfg, all_features, base_features, close, vol, timestamps, start_bar, ohlcv_1m = load_data()

    results = []

    # -------------------------------------------------------
    # TEST 0: BASELINE (monthly batch — the original Sharpe ~20)
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print("TEST 0: BASELINE (monthly_batch confidence — original config)")
    print(f"{'='*70}")
    engine_batch = StreamingBacktester(
        fwd_bars=FWD_BARS, embargo_bars=FWD_BARS + 1, slippage=0.0001,
        retrain_mode="monthly", lookback_months=LOOKBACK_MONTHS,
        confidence_mode="monthly_batch", confidence_percentile=70.0,
    )
    r = run_test("T0: Baseline (monthly_batch)", engine_batch,
                 ReportAlgorithmStrategy(), all_features, close, vol, timestamps, start_bar)
    results.append(r)
    print(fmt(r))

    # -------------------------------------------------------
    # TEST 1a: CAUSAL CONFIDENCE — previous month's threshold
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print("TEST 1a: CAUSAL CONFIDENCE (previous month's threshold)")
    print(f"{'='*70}")
    engine_causal = StreamingBacktester(
        fwd_bars=FWD_BARS, embargo_bars=FWD_BARS + 1, slippage=0.0001,
        retrain_mode="monthly", lookback_months=LOOKBACK_MONTHS,
        confidence_mode="previous_month", confidence_percentile=70.0,
    )
    r = run_test("T1a: Causal confidence (prev month)", engine_causal,
                 ReportAlgorithmStrategy(), all_features, close, vol, timestamps, start_bar)
    results.append(r)
    print(fmt(r))

    # -------------------------------------------------------
    # TEST 1c: NO CONFIDENCE FILTER
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print("TEST 1c: NO CONFIDENCE FILTER (all vol-filtered signals)")
    print(f"{'='*70}")
    engine_noconf = StreamingBacktester(
        fwd_bars=FWD_BARS, embargo_bars=FWD_BARS + 1, slippage=0.0001,
        retrain_mode="monthly", lookback_months=LOOKBACK_MONTHS,
        confidence_mode="none", confidence_percentile=0.0,
    )
    r = run_test("T1c: No confidence filter", engine_noconf,
                 ReportAlgorithmStrategy(), all_features, close, vol, timestamps, start_bar)
    results.append(r)
    print(fmt(r))

    # -------------------------------------------------------
    # TEST 3: DIRECTION SHUFFLE
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print("TEST 3: DIRECTION SHUFFLE (random direction, same filtering)")
    print(f"{'='*70}")
    r = run_test("T3: Direction shuffle (batch)", engine_batch,
                 ReportAlgorithmStrategy(shuffle_direction=True),
                 all_features, close, vol, timestamps, start_bar)
    results.append(r)
    print(fmt(r))

    # Also with causal confidence
    r = run_test("T3b: Direction shuffle (causal)", engine_causal,
                 ReportAlgorithmStrategy(shuffle_direction=True),
                 all_features, close, vol, timestamps, start_bar)
    results.append(r)
    print(fmt(r))

    # -------------------------------------------------------
    # TEST 4: RANDOM FEATURES
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print("TEST 4: RANDOM FEATURES (noise input, same model/filtering)")
    print(f"{'='*70}")
    r = run_test("T4: Random features (batch)", engine_batch,
                 ReportAlgorithmStrategy(use_random_features=True),
                 all_features, close, vol, timestamps, start_bar)
    results.append(r)
    print(fmt(r))

    r = run_test("T4b: Random features (causal)", engine_causal,
                 ReportAlgorithmStrategy(use_random_features=True),
                 all_features, close, vol, timestamps, start_bar)
    results.append(r)
    print(fmt(r))

    # -------------------------------------------------------
    # TEST 2: COMPONENT ISOLATION (base features only, no regime)
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print("TEST 2: COMPONENT ISOLATION")
    print(f"{'='*70}")

    # 2b: base features + monthly retrain + batch confidence (no regime)
    r = run_test("T2b: Base feat + batch conf", engine_batch,
                 ReportAlgorithmStrategy(), base_features, close, vol, timestamps, start_bar)
    results.append(r)
    print(fmt(r))

    # 2c: all features + monthly retrain + causal confidence
    r = run_test("T2c: All feat + causal conf", engine_causal,
                 ReportAlgorithmStrategy(), all_features, close, vol, timestamps, start_bar)
    results.append(r)
    print(fmt(r))

    # 2d: all features + monthly retrain + no confidence
    r = run_test("T2d: All feat + no conf", engine_noconf,
                 ReportAlgorithmStrategy(), all_features, close, vol, timestamps, start_bar)
    results.append(r)
    print(fmt(r))

    # 2e: all features + no winsorization + batch confidence
    r = run_test("T2e: No winsor + batch conf", engine_batch,
                 NoWinsorStrategy(), all_features, close, vol, timestamps, start_bar)
    results.append(r)
    print(fmt(r))

    # -------------------------------------------------------
    # TEST 5: NON-OVERLAPPING TRADES
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print("TEST 5: NON-OVERLAPPING TRADES (one at a time)")
    print(f"{'='*70}")
    engine_nooverlap = StreamingBacktester(
        fwd_bars=FWD_BARS, embargo_bars=FWD_BARS + 1, slippage=0.0001,
        retrain_mode="monthly", lookback_months=LOOKBACK_MONTHS,
        confidence_mode="none", confidence_percentile=0.0,
    )
    # We simulate non-overlapping by using a wrapper strategy
    class NonOverlappingWrapper:
        def __init__(self, inner):
            self.inner = inner
            self.last_entry_bar = -999
        def on_train(self, *a, **kw):
            return self.inner.on_train(*a, **kw)
        def on_bar(self, bar_idx, feature_row, vol):
            if bar_idx - self.last_entry_bar < FWD_BARS:
                return None
            sig = self.inner.on_bar(bar_idx, feature_row, vol)
            if sig is not None:
                self.last_entry_bar = bar_idx
            return sig

    r = run_test("T5: Non-overlapping trades", engine_nooverlap,
                 NonOverlappingWrapper(ReportAlgorithmStrategy()),
                 all_features, close, vol, timestamps, start_bar)
    results.append(r)
    print(fmt(r))

    # -------------------------------------------------------
    # TEST 7: COST SENSITIVITY
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print("TEST 7: COST SENSITIVITY")
    print(f"{'='*70}")
    for slip_bps in [0, 1, 5, 10, 25, 50, 100]:
        engine_cost = StreamingBacktester(
            fwd_bars=FWD_BARS, embargo_bars=FWD_BARS + 1,
            slippage=slip_bps / 10000.0,
            retrain_mode="monthly", lookback_months=LOOKBACK_MONTHS,
            confidence_mode="none", confidence_percentile=0.0,
        )
        r = run_test(f"T7: {slip_bps}bps slippage (no conf)", engine_cost,
                     ReportAlgorithmStrategy(), all_features, close, vol, timestamps, start_bar)
        results.append(r)
        print(fmt(r))

    # -------------------------------------------------------
    # TEST 8: TIME-REVERSED
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print("TEST 8: TIME-REVERSED BACKTEST")
    print(f"{'='*70}")
    close_rev = close[::-1].copy()
    vol_rev = vol[::-1].copy()
    feat_rev = all_features.iloc[::-1].reset_index(drop=True)
    feat_rev.index = all_features.index  # keep same index for month detection
    r = run_test("T8: Time-reversed (causal conf)", engine_causal,
                 ReportAlgorithmStrategy(), feat_rev, close_rev, vol_rev, timestamps, start_bar)
    results.append(r)
    print(fmt(r))

    # -------------------------------------------------------
    # TEST 12: DIRECTION BIAS + ALWAYS-LONG
    # -------------------------------------------------------
    print(f"\n{'='*70}")
    print("TEST 12: BASELINES (Always Long, Random)")
    print(f"{'='*70}")
    engine_simple = StreamingBacktester(fwd_bars=FWD_BARS, embargo_bars=FWD_BARS+1, slippage=0.0001)
    for name, strat in [("Always Long", AlwaysLongStrategy()), ("Random", RandomStrategy())]:
        r = run_test(f"T12: {name}", engine_simple, strat,
                     all_features, close, vol, timestamps, start_bar)
        results.append(r)
        print(fmt(r))

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print(f"\n{'='*90}")
    print("COMPREHENSIVE RESULTS")
    print(f"{'='*90}")
    print(f"  {'Test':<45s} {'Sharpe':>8s} {'Win%':>7s} {'bps':>7s} {'Trades':>8s} {'Long%':>6s}")
    print("  " + "-" * 83)
    for r in results:
        print(f"  {r['name']:<45s} {r['sharpe']:>8.2f} {r['win_rate']*100:>6.1f}% "
              f"{r['mean_bps']:>6.1f} {r['n_trades']:>8d} {r['long_frac']*100:>5.0f}%")

    # ============================================================
    # TEST 6: SUB-PERIOD STABILITY (on baseline and causal)
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 6: SUB-PERIOD STABILITY")
    print(f"{'='*70}")
    for r in results:
        if r["name"] in ["T0: Baseline (monthly_batch)", "T1a: Causal confidence (prev month)",
                          "T1c: No confidence filter"]:
            yearly, quarterly, monthly = compute_sub_period_sharpes(r["result"])
            print(f"\n  {r['name']}:")
            print(f"    Yearly:  {', '.join(f'{y}={s:.1f}' for y, s in sorted(yearly.items()))}")
            neg_months = sum(1 for s in monthly.values() if s < 0)
            print(f"    Monthly: {neg_months}/{len(monthly)} negative months")
            if quarterly:
                worst_q = min(quarterly, key=quarterly.get)
                best_q = max(quarterly, key=quarterly.get)
                print(f"    Worst quarter: {worst_q} (SR={quarterly[worst_q]:.1f})")
                print(f"    Best quarter:  {best_q} (SR={quarterly[best_q]:.1f})")

    # ============================================================
    # TEST 10: BOOTSTRAP CI (on baseline and causal)
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 10: BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"{'='*70}")
    for r in results:
        if r["name"] in ["T0: Baseline (monthly_batch)", "T1a: Causal confidence (prev month)",
                          "T1c: No confidence filter"]:
            lo, med, hi = bootstrap_sharpe_ci(r["result"])
            print(f"  {r['name']}: 95% CI [{lo:.2f}, {hi:.2f}], median={med:.2f}")

    # ============================================================
    # TEST 11: DSR
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 11: DEFLATED SHARPE RATIO")
    print(f"{'='*70}")
    K_trials = 50  # conservative estimate of variants tested
    for r in results:
        if r["name"] in ["T0: Baseline (monthly_batch)", "T1a: Causal confidence (prev month)",
                          "T1c: No confidence filter"]:
            daily_pnl = r["result"].daily_pnl
            if len(daily_pnl) > 10:
                skew = float(daily_pnl.skew())
                kurt = float(daily_pnl.kurtosis()) + 3  # excess -> raw
                dsr = deflated_sharpe_ratio(r["sharpe"], r["n_days"], K_trials, skew, kurt)
                print(f"  {r['name']}: DSR={dsr:.4f} (K={K_trials}, skew={skew:.2f}, kurt={kurt:.2f})")

    # ============================================================
    # DIAGNOSIS
    # ============================================================
    print(f"\n{'='*90}")
    print("DIAGNOSIS")
    print(f"{'='*90}")
    baseline = next((r for r in results if "Baseline" in r["name"]), None)
    causal = next((r for r in results if "T1a" in r["name"]), None)
    noconf = next((r for r in results if "T1c" in r["name"]), None)
    shuffle_batch = next((r for r in results if "T3:" in r["name"]), None)
    shuffle_causal = next((r for r in results if "T3b" in r["name"]), None)
    rand_batch = next((r for r in results if "T4:" in r["name"]), None)
    rand_causal = next((r for r in results if "T4b" in r["name"]), None)

    if baseline and causal:
        delta = baseline["sharpe"] - causal["sharpe"]
        print(f"  Batch lookahead contribution:   {delta:+.2f} Sharpe ({baseline['sharpe']:.2f} -> {causal['sharpe']:.2f})")
    if baseline and noconf:
        print(f"  Confidence filter contribution: {baseline['sharpe'] - noconf['sharpe']:+.2f} Sharpe")
    if shuffle_batch:
        print(f"  Direction shuffle (batch):      SR={shuffle_batch['sharpe']:.2f} (should be ~0 if model predicts)")
    if shuffle_causal:
        print(f"  Direction shuffle (causal):     SR={shuffle_causal['sharpe']:.2f}")
    if rand_batch:
        print(f"  Random features (batch):        SR={rand_batch['sharpe']:.2f} (should be ~0 if features matter)")
    if rand_causal:
        print(f"  Random features (causal):       SR={rand_causal['sharpe']:.2f}")

    if baseline and causal:
        if causal["sharpe"] < 3.0 and baseline["sharpe"] > 15.0:
            print(f"\n  VERDICT: Monthly batch lookahead is the PRIMARY driver of Sharpe ~{baseline['sharpe']:.0f}.")
            print(f"           True signal (causal) has Sharpe ~{causal['sharpe']:.1f}.")
        elif causal["sharpe"] > 10.0:
            print(f"\n  VERDICT: Signal is GENUINE. Causal Sharpe = {causal['sharpe']:.1f}.")
            print(f"           Batch filter adds {delta:.1f} but signal is strong regardless.")
        else:
            print(f"\n  VERDICT: Mixed. Causal Sharpe = {causal['sharpe']:.1f}. Further investigation needed.")

    print(f"\n  Total runtime: {time.time()-t_global:.0f}s")


if __name__ == "__main__":
    main()
