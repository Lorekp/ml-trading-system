"""Binary search ablation: find exactly which features/components drive Sharpe ~20.

Round 1: Which regime timeframe (1h vs 4h)?
Round 2: Which feature group within the dominant timeframe?
Round 3: Narrow to individual features
Round 4: Pipeline component ablation
Round 5: Label horizon investigation
"""

import sys
sys.path.insert(0, "src")

import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from afml.bet_sizing import discretize_signal, get_signal
from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.streaming_backtest import BacktestResult, Signal, StreamingBacktester
from signal_engine.weights import compute_sample_weights

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)

MAX_WORKERS = min(4, max(1, (os.cpu_count() or 4) // 2))

LOOKBACK_MONTHS = 8
FWD_BARS = 4
WINSOR_COLS = ["ret_1", "ret_3", "ret_7", "ret_14"]
N_JOBS = max(1, (os.cpu_count() or 4) // MAX_WORKERS)  # per-model threads (reduced for parallelism)

# Feature groups
HALF_A = ["ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
           "atr_14", "parkinson_vol_20", "cs_spread"]
HALF_B = ["rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b",
           "dist_swing_high", "dist_swing_low", "entropy", "ffd_close", "ffd_ret_1"]

# Quarter splits of Half A
QA1 = ["ret_1", "ret_3", "ret_7", "ret_14"]
QA2 = ["bar_range", "vol_zscore", "atr_14", "parkinson_vol_20", "cs_spread"]

# Quarter splits of Half B
QB1 = ["rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b"]
QB2 = ["dist_swing_high", "dist_swing_low", "entropy", "ffd_close", "ffd_ret_1"]


class AblationStrategy:
    """Strategy with configurable feature selection and pipeline toggles."""

    def __init__(self, feature_cols=None, vol_filter_quantile=0.50,
                 use_winsor=True, use_sample_weights=True,
                 use_val_split=True, model_kwargs=None, fwd_bars=4):
        self.model = None
        self._winsor_bounds = {}
        self.vol_threshold = 0.0
        self.vol_filter_quantile = vol_filter_quantile
        self.feature_cols = feature_cols  # None = use all
        self.use_winsor = use_winsor
        self.use_sample_weights = use_sample_weights
        self.use_val_split = use_val_split
        self.fwd_bars = fwd_bars
        self.model_kwargs = model_kwargs or {
            "n_estimators": 250, "max_depth": 4, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        }

    def _select_cols(self, df):
        if self.feature_cols is not None:
            cols = [c for c in self.feature_cols if c in df.columns]
            return df[cols]
        return df

    def on_train(self, bar_idx, features, labels, vol_series):
        features = self._select_cols(features)
        n = len(features)
        if n < 200:
            return

        if len(vol_series) > 50:
            self.vol_threshold = float(vol_series.quantile(self.vol_filter_quantile))

        feat_train = features.copy()
        if self.use_winsor:
            for col in WINSOR_COLS:
                if col in feat_train.columns:
                    arr = feat_train[col].to_numpy(dtype=np.float32)
                    lo, hi = np.nanquantile(arr, [0.01, 0.99])
                    feat_train[col] = np.clip(arr, lo, hi)
                    self._winsor_bounds[col] = (float(lo), float(hi))

        y = labels.to_numpy(dtype=np.int8)

        fit_mask = np.ones(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        if self.use_val_split:
            ts_idx = features.index
            months = pd.DatetimeIndex(ts_idx).to_period("M") if not hasattr(ts_idx, "to_period") else ts_idx.to_period("M")
            unique_months = months.unique()
            if len(unique_months) >= 2:
                val_month = unique_months.max()
                val_mask = np.asarray(months == val_month)
                fit_mask = ~val_mask
                if fit_mask.sum() < 200 or val_mask.sum() < 50:
                    fit_mask[:] = True
                    val_mask[:] = False

        fit_X = feat_train.to_numpy(dtype=np.float32)[fit_mask]
        fit_y = y[fit_mask]

        if self.use_sample_weights:
            fit_sw = compute_sample_weights(features.index[fit_mask].values.astype(np.int64), fwd_bars=self.fwd_bars)
        else:
            fit_sw = np.ones(fit_mask.sum(), dtype=np.float64)

        self.model = xgb.XGBClassifier(
            **self.model_kwargs,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=42, n_jobs=N_JOBS, verbosity=0,
            early_stopping_rounds=25 if (val_mask.any() and self.use_val_split) else None,
        )
        kwargs = {"sample_weight": fit_sw, "verbose": False}
        if val_mask.any() and self.use_val_split:
            kwargs["eval_set"] = [(feat_train.to_numpy(dtype=np.float32)[val_mask], y[val_mask])]
        self.model.fit(fit_X, fit_y, **kwargs)

    def on_bar(self, bar_idx, feature_row, vol):
        if self.model is None:
            return None
        if self.vol_filter_quantile > 0 and vol < self.vol_threshold:
            return None

        row = feature_row.copy()
        if self.feature_cols is not None:
            row = row[[c for c in self.feature_cols if c in row.index]]

        if self.use_winsor:
            for col, (lo, hi) in self._winsor_bounds.items():
                if col in row.index:
                    row[col] = np.clip(row[col], lo, hi)

        X = row.to_numpy(dtype=np.float32).reshape(1, -1)
        if not np.isfinite(X).all():
            return None

        prob = float(self.model.predict_proba(X)[0, 1])
        conf = abs(prob - 0.5)
        direction = 1.0 if prob > 0.5 else -1.0
        sig = get_signal(np.array([prob]), num_classes=2)
        sig_disc = discretize_signal(sig, step_size=0.1)
        strength = float(abs(sig_disc[0])) if np.ndim(sig_disc) > 0 else float(abs(sig_disc))
        return Signal(direction=direction, probability=prob, confidence=conf, signal_strength=strength)


def run_ablation(name, engine, strategy, features, close, vol, timestamps, start_bar):
    t0 = time.time()
    result = engine.run(strategy, features, close, vol, timestamps,
                        start_bar=start_bar, progress_every=0)
    elapsed = time.time() - t0
    dirs = np.array([t.direction for t in result.trades]) if result.trades else np.array([])
    long_frac = float((dirs > 0).mean()) if len(dirs) > 0 else 0.0
    ncols = len(strategy.feature_cols) if strategy.feature_cols else features.shape[1]
    return {
        "name": name, "sharpe": result.sharpe, "win_rate": result.win_rate,
        "mean_bps": result.mean_return_bps, "n_trades": result.n_trades,
        "long_frac": long_frac, "elapsed": elapsed, "n_features": ncols,
    }


def fmt(r):
    return (f"  {r['name']:<50s} SR={r['sharpe']:>7.2f}  Win={r['win_rate']*100:>5.1f}%  "
            f"bps={r['mean_bps']:>6.1f}  n={r['n_trades']:>6d}  feat={r['n_features']:>3d}  "
            f"({r['elapsed']:.0f}s)")


def prefixed(names, prefix):
    return [f"{prefix}{n}" for n in names]


def main():
    t_global = time.time()
    cfg = load_config(Path("configs/default.yaml"))
    data_dirs = [Path(p) for p in cfg.data.data_dirs] or [Path("data/binance-1m")]

    print("Loading BTC 1m data...")
    ohlcv_1m = load_binance_1m("BTCUSDT", data_dirs)
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")

    print("Computing features...")
    base_features = compute_features(ohlcv_15m, cfg.features)
    base_cols = list(base_features.columns)

    regime_parts = {}
    for rtf in cfg.close_model.regime_timeframes:
        rtf_ohlcv = resample_ohlcv(ohlcv_1m, rtf)
        rtf_feat = compute_features(rtf_ohlcv, cfg.features)
        aligned = rtf_feat.reindex(ohlcv_15m.index, method="ffill")
        aligned = aligned.add_prefix(f"signal_regime_{rtf}_")
        regime_parts[rtf] = aligned

    all_features = pd.concat([base_features] + list(regime_parts.values()), axis=1)
    regime_1h_cols = list(regime_parts["1h"].columns)
    regime_4h_cols = list(regime_parts["4h"].columns)

    close = ohlcv_15m["close"].to_numpy(dtype=np.float64)
    vol = ohlcv_15m["close"].pct_change().ewm(span=50, min_periods=20).std().to_numpy(dtype=np.float64)
    timestamps = ohlcv_15m.index

    start_ts = pd.Timestamp("2022-01-01", tz="UTC")
    if timestamps.tz is None:
        start_ts = start_ts.tz_localize(None)
    start_bar = int(timestamps.searchsorted(start_ts))

    print(f"  {all_features.shape[1]} total features: {len(base_cols)} base, "
          f"{len(regime_1h_cols)} 1h, {len(regime_4h_cols)} 4h")
    print(f"  Base: {base_cols}")

    def engine(fwd=FWD_BARS):
        return StreamingBacktester(
            fwd_bars=fwd, embargo_bars=fwd + 1, slippage=0.0001,
            retrain_mode="monthly", lookback_months=LOOKBACK_MONTHS,
            confidence_mode="none",
        )

    results = []

    def run_batch(round_name, jobs):
        """Run a batch of ablation jobs in parallel. Each job = (name, engine, strategy)."""
        print(f"\n{'='*70}")
        print(f"{round_name} ({len(jobs)} runs, {MAX_WORKERS} workers)")
        print(f"{'='*70}")
        batch_results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {}
            for name, eng, strat in jobs:
                f = pool.submit(run_ablation, name, eng, strat,
                                all_features, close, vol, timestamps, start_bar)
                futures[f] = name
            for f in as_completed(futures):
                r = f.result()
                batch_results.append(r)
                print(fmt(r), flush=True)
        # Sort by original submission order
        order = {name: i for i, (name, _, _) in enumerate(jobs)}
        batch_results.sort(key=lambda r: order.get(r["name"], 999))
        return batch_results

    # ============================================================
    # ROUND 1: Which regime timeframe?
    # ============================================================
    jobs_r1 = []
    for name, cols in [
        ("R1-ref: ALL features", None),
        ("R1-ref: Base only (no regime)", base_cols),
        ("R1a: Base + 1h regime", base_cols + regime_1h_cols),
        ("R1b: Base + 4h regime", base_cols + regime_4h_cols),
        ("R1c: 1h + 4h regime (no base)", regime_1h_cols + regime_4h_cols),
        ("R1d: 1h regime only", regime_1h_cols),
        ("R1e: 4h regime only", regime_4h_cols),
    ]:
        jobs_r1.append((name, engine(), AblationStrategy(feature_cols=cols)))
    results.extend(run_batch("ROUND 1: REGIME TIMEFRAME ISOLATION", jobs_r1))

    # ============================================================
    # ROUND 2: Feature groups within regime timeframes
    # ============================================================
    jobs_r2 = []
    for tf, tf_prefix in [("1h", "signal_regime_1h_"), ("4h", "signal_regime_4h_")]:
        half_a_cols = prefixed(HALF_A, tf_prefix)
        half_b_cols = prefixed(HALF_B, tf_prefix)
        jobs_r2.append((f"R2a: Base + {tf} Half-A (ret/vol)", engine(),
                         AblationStrategy(feature_cols=base_cols + half_a_cols)))
        jobs_r2.append((f"R2b: Base + {tf} Half-B (ind/struct)", engine(),
                         AblationStrategy(feature_cols=base_cols + half_b_cols)))

    both_half_a = prefixed(HALF_A, "signal_regime_1h_") + prefixed(HALF_A, "signal_regime_4h_")
    both_half_b = prefixed(HALF_B, "signal_regime_1h_") + prefixed(HALF_B, "signal_regime_4h_")
    jobs_r2.append(("R2c: Base + BOTH TF Half-A", engine(),
                     AblationStrategy(feature_cols=base_cols + both_half_a)))
    jobs_r2.append(("R2d: Base + BOTH TF Half-B", engine(),
                     AblationStrategy(feature_cols=base_cols + both_half_b)))
    results.extend(run_batch("ROUND 2: FEATURE GROUPS WITHIN REGIME", jobs_r2))

    # ============================================================
    # ROUND 3: Quarter-level within regime features
    # ============================================================
    jobs_r3 = []
    for tf, tf_prefix in [("1h", "signal_regime_1h_"), ("4h", "signal_regime_4h_")]:
        for qname, qfeats in [
            ("returns", QA1), ("vol/spread", QA2),
            ("momentum", QB1), ("structure", QB2),
        ]:
            q_cols = prefixed(qfeats, tf_prefix)
            jobs_r3.append((
                f"R3: Base + {tf} {qname} ({len(qfeats)}f)",
                engine(), AblationStrategy(feature_cols=base_cols + q_cols)))
    results.extend(run_batch("ROUND 3: QUARTER-LEVEL FEATURE SPLITS", jobs_r3))

    # ============================================================
    # ROUND 4: Pipeline component ablation (full features)
    # ============================================================
    jobs_r4 = []
    for name, kwargs in [
        ("R4a: No vol filter", {"vol_filter_quantile": 0.0}),
        ("R4b: No sample weights", {"use_sample_weights": False}),
        ("R4c: No val split / no ES", {"use_val_split": False}),
        ("R4d: Simple model (d=1, n=50)", {"model_kwargs": {
            "n_estimators": 50, "max_depth": 1, "learning_rate": 0.1,
            "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1,
            "reg_alpha": 0, "reg_lambda": 1.0,
        }}),
        ("R4e: Logistic (d=1, n=1)", {"model_kwargs": {
            "n_estimators": 1, "max_depth": 1, "learning_rate": 1.0,
            "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1,
            "reg_alpha": 0, "reg_lambda": 0,
        }}),
    ]:
        jobs_r4.append((name, engine(), AblationStrategy(**kwargs)))
    results.extend(run_batch("ROUND 4: PIPELINE COMPONENT ABLATION", jobs_r4))

    # ============================================================
    # ROUND 5: Label horizon investigation
    # ============================================================
    jobs_r5 = []
    for fwd in [1, 2, 4, 8, 16]:
        jobs_r5.append((
            f"R5: fwd_bars={fwd} ({fwd*15}min)",
            engine(fwd=fwd), AblationStrategy(fwd_bars=fwd)))
    results.extend(run_batch("ROUND 5: LABEL HORIZON (fwd_bars)", jobs_r5))

    # ============================================================
    # COMPREHENSIVE TABLE
    # ============================================================
    print(f"\n{'='*90}")
    print("COMPREHENSIVE ABLATION RESULTS")
    print(f"{'='*90}")
    print(f"  {'Test':<50s} {'SR':>7s} {'Win%':>6s} {'bps':>6s} {'Trades':>7s} {'Feat':>4s}")
    print("  " + "-" * 82)
    for r in results:
        print(f"  {r['name']:<50s} {r['sharpe']:>7.2f} {r['win_rate']*100:>5.1f}% "
              f"{r['mean_bps']:>5.1f} {r['n_trades']:>7d} {r['n_features']:>4d}")

    # Find minimal feature set that still gives SR > 15
    print(f"\n  Top 5 by Sharpe:")
    for r in sorted(results, key=lambda x: -x["sharpe"])[:5]:
        print(f"    {r['name']:<50s} SR={r['sharpe']:.2f}")

    print(f"\n  Bottom 5 by Sharpe:")
    for r in sorted(results, key=lambda x: x["sharpe"])[:5]:
        print(f"    {r['name']:<50s} SR={r['sharpe']:.2f}")

    print(f"\n  Total runtime: {time.time()-t_global:.0f}s")


if __name__ == "__main__":
    main()
