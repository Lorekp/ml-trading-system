"""Streaming backtest v2: exact match to old pipeline config.

Monthly walk-forward, regime features, batch confidence filter, lb=8.
"""

import sys
sys.path.insert(0, "src")

import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from afml.bet_sizing import discretize_signal, get_signal
from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.streaming_backtest import BacktestResult, Signal, StreamingBacktester
from signal_engine.strategies import AlwaysLongStrategy, RandomStrategy
from signal_engine.weights import compute_sample_weights

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

LOOKBACK_MONTHS = 8
FWD_BARS = 4
WINSOR_COLS = ["ret_1", "ret_3", "ret_7", "ret_14"]
N_JOBS = max(1, os.cpu_count() or 1)


class ReportAlgorithmStrategy:
    """Exact replica of old pipeline: regime features, vol filter, winsorization."""

    def __init__(self, vol_filter_quantile=0.50):
        self.model = None
        self._winsor_bounds = {}
        self.vol_threshold = 0.0
        self.vol_filter_quantile = vol_filter_quantile

    def on_train(self, bar_idx, features, labels, vol_series):
        n = len(features)
        if n < 200:
            return

        # Vol threshold from training vol (3-month window approximation)
        if len(vol_series) > 50:
            self.vol_threshold = float(vol_series.quantile(self.vol_filter_quantile))

        # Winsorize
        feat_train = features.copy()
        for col in WINSOR_COLS:
            if col in feat_train.columns:
                arr = feat_train[col].to_numpy(dtype=np.float32)
                lo, hi = np.nanquantile(arr, [0.01, 0.99])
                feat_train[col] = np.clip(arr, lo, hi)
                self._winsor_bounds[col] = (float(lo), float(hi))

        y = labels.to_numpy(dtype=np.int8)

        # Val split: last month
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

        fit_X = feat_train.to_numpy(dtype=np.float32)[fit_mask]
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
            kwargs["eval_set"] = [(feat_train.to_numpy(dtype=np.float32)[val_mask], y[val_mask])]
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
        if not np.isfinite(X).all():
            return None

        prob = float(self.model.predict_proba(X)[0, 1])
        conf = abs(prob - 0.5)
        direction = 1.0 if prob > 0.5 else -1.0
        sig = get_signal(np.array([prob]), num_classes=2)
        sig_disc = discretize_signal(sig, step_size=0.1)
        strength = float(abs(sig_disc[0])) if np.ndim(sig_disc) > 0 else float(abs(sig_disc))
        return Signal(direction=direction, probability=prob, confidence=conf, signal_strength=strength)


def flat_bet_sharpe(df):
    if df is None or len(df) < 10:
        return 0.0, 0.0, 0.0, 0, 0
    pnls = df["direction"].values * df["fwd_return"].values
    daily = pd.Series(5.0 * pnls, index=pd.to_datetime(df["timestamp"], utc=True)).resample("1D").sum()
    daily = daily[daily != 0]
    s = daily.std()
    sr = float(daily.mean() / s * np.sqrt(365.25)) if s > 0 and len(daily) > 5 else 0
    wr = float((pnls > 0).mean()) * 100
    bps = float(pnls.mean() * 10000)
    return sr, wr, bps, len(df), len(daily)


def main():
    t0 = time.time()
    cfg = load_config(Path("configs/default.yaml"))
    data_dirs = [Path(p) for p in cfg.data.data_dirs] or [Path("data/binance-1m")]

    print("Loading BTC data...")
    ohlcv_1m = load_binance_1m("BTCUSDT", data_dirs)
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")

    print("Pre-computing features (base + regime)...")
    base_features = compute_features(ohlcv_15m, cfg.features)

    # Regime features: 1h and 4h, forward-filled to 15min index
    regime_parts = []
    for rtf in cfg.close_model.regime_timeframes:
        rtf_ohlcv = resample_ohlcv(ohlcv_1m, rtf)
        rtf_feat = compute_features(rtf_ohlcv, cfg.features)
        # CRITICAL: shift by 1 period so features only use COMPLETED bars (no future leakage)
        aligned = rtf_feat.shift(1).reindex(ohlcv_15m.index, method="ffill")
        aligned = aligned.add_prefix(f"signal_regime_{rtf}_")
        regime_parts.append(aligned)

    if regime_parts:
        all_features = pd.concat([base_features] + regime_parts, axis=1)
    else:
        all_features = base_features
    print(f"  {all_features.shape[1]} total features ({base_features.shape[1]} base + {all_features.shape[1]-base_features.shape[1]} regime)")

    close = ohlcv_15m["close"].to_numpy(dtype=np.float64)
    vol = ohlcv_15m["close"].pct_change().ewm(span=50, min_periods=20).std().to_numpy(dtype=np.float64)
    timestamps = ohlcv_15m.index

    start_ts = pd.Timestamp("2022-01-01", tz="UTC")
    if timestamps.tz is None:
        start_ts = start_ts.tz_localize(None)
    start_bar = int(timestamps.searchsorted(start_ts))

    # === OLD PIPELINE ===
    print(f"\n{'='*65}")
    print("1. OLD PIPELINE (generate_asset_signals)")
    print(f"{'='*65}")
    st = time.time()
    from signal_engine.research import close_walkforward as _mod
    _mod.LOOKBACKS = [LOOKBACK_MONTHS]
    os.environ["ML_CLOSE_SKIP_CV"] = "1"
    features_15m = compute_features(ohlcv_15m, cfg.features)
    regime_feats = {}
    for rtf in cfg.close_model.regime_timeframes:
        rtf_ohlcv = resample_ohlcv(ohlcv_1m, rtf)
        regime_feats[rtf] = compute_features(rtf_ohlcv, cfg.features)
    signals = _mod.generate_asset_signals("BTC", ohlcv_15m, features_15m, regime_feats, cfg)
    old_df = signals.get(LOOKBACK_MONTHS, pd.DataFrame())
    if len(old_df) > 0:
        months = pd.to_datetime(old_df["timestamp"], utc=True).dt.strftime("%Y-%m")
        old_df = old_df.loc[months >= "2022-01"].reset_index(drop=True)
    sr, wr, bps, n, d = flat_bet_sharpe(old_df)
    print(f"  Sharpe={sr:.3f}  Win={wr:.1f}%  Mean={bps:.1f}bps  Trades={n}  Days={d}  ({time.time()-st:.0f}s)")

    # === STREAMING: MONTHLY + REGIME + BATCH CONFIDENCE ===
    print(f"\n{'='*65}")
    print("2. STREAMING (monthly retrain, regime features, batch confidence)")
    print(f"{'='*65}")
    engine = StreamingBacktester(
        fwd_bars=FWD_BARS,
        embargo_bars=FWD_BARS + 1,
        slippage=0.0001,
        retrain_mode="monthly",
        lookback_months=LOOKBACK_MONTHS,
        confidence_mode="monthly_batch",
        confidence_percentile=70.0,
    )
    strategy = ReportAlgorithmStrategy(vol_filter_quantile=0.50)
    result = engine.run(strategy, all_features, close, vol, timestamps,
                        start_bar=start_bar, progress_every=20000)
    print(f"  Sharpe={result.sharpe:.3f}  Win={result.win_rate*100:.1f}%  "
          f"Mean={result.mean_return_bps:.1f}bps  Trades={result.n_trades}  Days={result.n_days}")

    # === BASELINES ===
    engine_simple = StreamingBacktester(fwd_bars=FWD_BARS, embargo_bars=FWD_BARS+1, slippage=0.0001)
    print(f"\n{'='*65}")
    print("3. BASELINES")
    print(f"{'='*65}")
    for name, strat in [("Always Long", AlwaysLongStrategy()), ("Random", RandomStrategy())]:
        r = engine_simple.run(strat, all_features, close, vol, timestamps,
                              start_bar=start_bar, progress_every=0)
        print(f"  {name:<15s} Sharpe={r.sharpe:.3f}  Win={r.win_rate*100:.1f}%  Mean={r.mean_return_bps:.1f}bps")

    # === SUMMARY ===
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  Old Pipeline:  Sharpe={sr:.3f}  Win={wr:.1f}%  Mean={bps:.1f}bps  n={n}")
    print(f"  Streaming:     Sharpe={result.sharpe:.3f}  Win={result.win_rate*100:.1f}%  Mean={result.mean_return_bps:.1f}bps  n={result.n_trades}")
    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
