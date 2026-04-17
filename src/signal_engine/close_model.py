"""Walk-forward ML close-model utilities.

Builds close-decision examples from realized signal trades and applies a
monthly retrained binary model that decides whether to close a position
before the fixed 60-minute expiry.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

try:
    import joblib as _joblib
except ImportError:
    _joblib = None

from afml.bet_sizing import discretize_signal, get_signal
from signal_engine.config import CloseModelConfig

_wf_logger = logging.getLogger(__name__)
_WF_CACHE_VERSION = 2


REGIME_PREFIX = "regime_"
PLACEHOLDER_STATE_FEATURE_COLUMNS = [
    "signal_prob_change_since_entry",
    "signal_strength_change_since_entry",
]
STATE_FEATURE_COLUMNS = [
    "entry_direction",
    "entry_probability",
    "entry_confidence",
    "entry_signal_strength",
    "bars_since_entry",
    "minutes_since_entry",
    "time_to_expiry_minutes",
    "remaining_decision_steps",
    "current_unrealized_return",
    "max_favorable_return_since_entry",
    "max_adverse_return_since_entry",
    "realized_vol_since_entry",
    "return_since_last_decision",
    "rolling_2bar",
    "rolling_3bar",
    "distance_from_entry_mfe",
    "distance_from_entry_mae",
    "mfe_minus_mae",
    "entry_to_now_trend_slope",
    *PLACEHOLDER_STATE_FEATURE_COLUMNS,
]
MODEL_STATE_FEATURE_COLUMNS = [
    col for col in STATE_FEATURE_COLUMNS if col not in PLACEHOLDER_STATE_FEATURE_COLUMNS
]
ENTRY_AUX_COLUMNS = ["regime_exposure_mult"]
EXTRA_REGIME_ONLY_FEATURE_COLUMNS = ["sadf"]


def normalize_timeframe(timeframe: str) -> str:
    """Normalize a minute-based timeframe label to pandas offset alias."""
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return f"{int(tf[:-1])}min"
    return tf


def timeframe_minutes(timeframe: str) -> int:
    """Convert a timeframe like ``5min`` or ``15m`` to integer minutes."""
    tf = normalize_timeframe(timeframe)
    if not tf.endswith("min"):
        raise ValueError(f"Only minute-based timeframes are supported, got {timeframe}")
    return int(tf[:-3])


def regime_feature_columns(
    base_feature_cols: Iterable[str],
    regime_timeframes: Iterable[str] | None = None,
) -> list[str]:
    timeframes = list(regime_timeframes or ["1h"])
    base = list(dict.fromkeys(list(base_feature_cols) + EXTRA_REGIME_ONLY_FEATURE_COLUMNS))
    cols: list[str] = []
    for timeframe in timeframes:
        cols.extend([f"{REGIME_PREFIX}{timeframe}_{col}" for col in base])
    return cols


def close_feature_columns(
    base_feature_cols: Iterable[str],
    regime_timeframes: Iterable[str] | None = None,
) -> list[str]:
    """Market + trade-state features used by the close model."""
    base = list(base_feature_cols)
    return (
        base
        + regime_feature_columns(base, regime_timeframes=regime_timeframes)
        + MODEL_STATE_FEATURE_COLUMNS
    )


def make_close_model(warm_start: bool = False) -> LogisticRegression:
    """Create the default close-model classifier."""
    try:
        import sklearn

        if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8):
            return LogisticRegression(
                C=0.1,
                l1_ratio=1.0,
                solver="saga",
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
                warm_start=warm_start,
            )
    except Exception:
        pass
    return LogisticRegression(
        C=0.1,
        penalty="l1",
        solver="saga",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        warm_start=warm_start,
    )


def make_xgb_close_model(
    scale_pos_weight: float = 1.0,
    early_stopping_rounds: int | None = None,
    n_jobs: int | None = None,
) -> xgb.XGBClassifier:
    """Create the AFML-style XGBoost close-model classifier."""
    return xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=max(scale_pos_weight, 1e-6),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=os.environ.get("XGB_DEVICE", "cpu"),
        random_state=42,
        n_jobs=int(n_jobs) if n_jobs else -1,
        early_stopping_rounds=early_stopping_rounds,
    )


def make_xgb_close_model_with_params(
    scale_pos_weight: float,
    params: dict[str, float | int],
    early_stopping_rounds: int | None = None,
    n_jobs: int | None = None,
) -> xgb.XGBClassifier:
    try:
        model = make_xgb_close_model(
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=n_jobs,
        )
    except TypeError:
        try:
            model = make_xgb_close_model(
                scale_pos_weight=scale_pos_weight,
                n_jobs=n_jobs,
            )
        except TypeError:
            model = make_xgb_close_model(scale_pos_weight=scale_pos_weight)
    if hasattr(model, "set_params"):
        model.set_params(**params)
    return model


def close_model_thread_budget() -> int:
    override = os.getenv("ML_CLOSE_MODEL_THREADS")
    if override:
        return max(1, int(override))
    cpu = max(int(os.cpu_count() or 1), 1)
    eval_workers = max(1, int(os.getenv("ML_CLOSE_EVAL_WORKERS", "1")))
    budget = max(1, cpu // eval_workers)
    if cpu >= 16 and eval_workers >= 6:
        budget = max(budget, 2)
    return budget


def xgb_grid_candidates(close_cfg: CloseModelConfig) -> list[dict[str, float | int]]:
    depths = list(close_cfg.xgb_max_depth_grid or [4])
    child_weights = list(close_cfg.xgb_min_child_weight_grid or [20])
    learning_rates = list(close_cfg.xgb_learning_rate_grid or [0.05])
    sample_pairs = list(
        zip(
            close_cfg.xgb_subsample_grid or [0.8],
            close_cfg.xgb_colsample_grid or [0.8],
            strict=False,
        )
    )
    if not sample_pairs:
        sample_pairs = [(0.8, 0.8)]

    candidates: list[dict[str, float | int]] = []
    for max_depth in depths:
        for min_child_weight in child_weights:
            for learning_rate in learning_rates:
                for subsample, colsample in sample_pairs:
                    candidates.append(
                        {
                            "max_depth": int(max_depth),
                            "min_child_weight": int(min_child_weight),
                            "learning_rate": float(learning_rate),
                            "subsample": float(subsample),
                            "colsample_bytree": float(colsample),
                        }
                    )
    if not candidates:
        return [{}]
    return candidates[: max(int(close_cfg.xgb_max_grid_trials), 1)]


def compute_afml_close_sample_weights_arrays(
    decision_timestamp: np.ndarray,
    label_end_timestamp: np.ndarray,
    label_return_abs: np.ndarray,
    clf_last_w: float = 0.5,
) -> np.ndarray:
    """Approximate AFML weights from already-sorted numpy arrays."""
    if len(decision_timestamp) == 0:
        return np.empty(0, dtype=np.float64)

    decision_timestamp = np.asarray(decision_timestamp, dtype="datetime64[ns]")
    label_end_timestamp = np.asarray(label_end_timestamp, dtype="datetime64[ns]")
    label_return_abs = np.asarray(label_return_abs, dtype=np.float64)

    order = np.lexsort((label_end_timestamp, decision_timestamp))
    decision_timestamp = decision_timestamp[order]
    label_end_timestamp = label_end_timestamp[order]
    label_return_abs = label_return_abs[order]

    timeline = np.unique(np.concatenate([decision_timestamp, label_end_timestamp]))
    if len(timeline) == 0:
        return np.ones(len(decision_timestamp), dtype=np.float64)

    start_pos = np.searchsorted(timeline, decision_timestamp, side="left")
    end_pos = np.searchsorted(timeline, label_end_timestamp, side="left")
    diff = np.zeros(len(timeline) + 1, dtype=np.float64)
    np.add.at(diff, start_pos, 1.0)
    np.add.at(diff, end_pos + 1, -1.0)
    concurrent = np.cumsum(diff[:-1])
    concurrent = np.maximum(concurrent, 1.0)

    inv_concurrent = 1.0 / concurrent
    prefix = np.cumsum(inv_concurrent)
    prefix_before = np.concatenate(([0.0], prefix[:-1]))
    span_sum = prefix[end_pos] - prefix_before[start_pos]
    span_len = np.maximum(end_pos - start_pos + 1, 1)
    uniqueness = span_sum / span_len

    cum_u = np.cumsum(uniqueness)
    if cum_u[-1] > 0:
        if clf_last_w >= 0:
            slope = (1.0 - clf_last_w) / cum_u[-1]
        else:
            slope = 1.0 / ((clf_last_w + 1.0) * cum_u[-1])
        const = 1.0 - slope * cum_u[-1]
        time_decay = np.asarray(const + slope * cum_u, dtype=np.float64)
        time_decay = np.maximum(time_decay, 0.0)
    else:
        time_decay = np.ones(len(decision_timestamp), dtype=np.float64)

    ret_attr = np.maximum(label_return_abs, 1e-6)
    weights = uniqueness * ret_attr * time_decay
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)

    out = np.empty(len(weights), dtype=np.float64)
    out[order] = weights
    out = np.where(np.isfinite(out) & (out > 0), out, 1.0)
    return out


def compute_afml_close_sample_weights(
    examples: pd.DataFrame,
    clf_last_w: float = 0.5,
) -> pd.Series:
    """Approximate AFML uniqueness + return-attribution + time-decay weights."""
    if examples is None or len(examples) == 0:
        return pd.Series(dtype=np.float64)

    weights = compute_afml_close_sample_weights_arrays(
        pd.to_datetime(examples["decision_timestamp"], utc=True).to_numpy(
            dtype="datetime64[ns]"
        ),
        pd.to_datetime(examples["label_end_timestamp"], utc=True).to_numpy(
            dtype="datetime64[ns]"
        ),
        examples["label_return_abs"].to_numpy(dtype=np.float64),
        clf_last_w=clf_last_w,
    )
    return pd.Series(weights, index=examples.index, dtype=np.float64)


def _masked_cumstd(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    mask = valid.astype(np.float64)
    masked = np.where(valid, values, 0.0)
    count = np.cumsum(mask, axis=1)
    running_sum = np.cumsum(masked, axis=1)
    running_sumsq = np.cumsum(masked * masked, axis=1)
    mean = np.divide(
        running_sum,
        count,
        out=np.zeros_like(running_sum, dtype=np.float64),
        where=count > 0,
    )
    mean_sq = np.divide(
        running_sumsq,
        count,
        out=np.zeros_like(running_sumsq, dtype=np.float64),
        where=count > 0,
    )
    var = np.maximum(mean_sq - mean * mean, 0.0)
    out = np.sqrt(var)
    out[~valid] = np.nan
    return out


def _best_remaining_labels(
    exit_returns: np.ndarray,
    candidate_ts: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_remaining = np.maximum.accumulate(exit_returns[:, ::-1], axis=1)[:, ::-1]
    best_abs_remaining = np.maximum.accumulate(
        np.abs(exit_returns[:, ::-1]),
        axis=1,
    )[:, ::-1]
    labels = valid & np.isfinite(exit_returns) & (
        exit_returns >= (best_remaining - 1e-12)
    )

    valid_counts = valid.sum(axis=1)
    last_valid_pos = np.maximum(valid_counts - 1, 0)
    label_end = np.repeat(
        candidate_ts[np.arange(len(valid_counts)), last_valid_pos][:, None],
        valid.shape[1],
        axis=1,
    )
    return labels, label_end, best_abs_remaining


def _residual_triple_barrier_labels(
    exit_returns: np.ndarray,
    candidate_ts: np.ndarray,
    valid: np.ndarray,
    target_matrix: np.ndarray,
    pt_mult: float,
    sl_mult: float,
    min_improvement_ret: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_trades, max_steps = exit_returns.shape
    labels = np.zeros_like(valid, dtype=bool)
    label_end = np.empty_like(candidate_ts)
    label_return_abs = np.zeros_like(exit_returns, dtype=np.float64)

    valid_counts = valid.sum(axis=1)
    last_valid_pos = np.maximum(valid_counts - 1, 0)
    label_end[:] = np.repeat(
        candidate_ts[np.arange(n_trades), last_valid_pos][:, None],
        max_steps,
        axis=1,
    )

    for step in range(max_steps):
        step_valid = valid[:, step]
        if not step_valid.any():
            continue

        remaining_valid = valid[:, step:]
        current_ret = exit_returns[:, [step]]
        future_delta = exit_returns[:, step:] - current_ret
        future_delta[~remaining_valid] = np.nan

        pt_level = np.maximum(pt_mult * target_matrix[:, step], 1e-6)[:, None]
        sl_level = np.maximum(sl_mult * target_matrix[:, step], 1e-6)[:, None]
        down_hit = future_delta <= -sl_level
        up_hit = future_delta >= pt_level

        has_down = np.any(down_hit, axis=1)
        has_up = np.any(up_hit, axis=1)
        first_down = np.where(has_down, np.argmax(down_hit, axis=1), max_steps)
        first_up = np.where(has_up, np.argmax(up_hit, axis=1), max_steps)
        down_first = has_down & ((~has_up) | (first_down <= first_up))
        up_first = has_up & ((~has_down) | (first_up < first_down))

        expiry_delta = (
            exit_returns[np.arange(n_trades), last_valid_pos] - exit_returns[:, step]
        )
        expiry_close = (~down_first & ~up_first) & (expiry_delta <= -min_improvement_ret)
        labels[:, step] = step_valid & (down_first | expiry_close)

        expiry_offset = np.maximum(last_valid_pos - step, 0)
        resolution_offset = np.where(
            down_first,
            first_down,
            np.where(up_first, first_up, expiry_offset),
        )
        resolution_pos = np.clip(step + resolution_offset, 0, max_steps - 1)
        label_end[:, step] = candidate_ts[np.arange(n_trades), resolution_pos]
        resolution_delta = (
            exit_returns[np.arange(n_trades), resolution_pos] - exit_returns[:, step]
        )
        label_return_abs[:, step] = np.abs(np.nan_to_num(resolution_delta, nan=0.0))

    return labels, label_end, label_return_abs


def _lookup_entry_prices(
    signals_df: pd.DataFrame,
    entry_ohlcv: pd.DataFrame,
) -> np.ndarray:
    if "entry_price" in signals_df.columns:
        return signals_df["entry_price"].to_numpy(dtype=np.float64)
    aligned = entry_ohlcv["close"].reindex(pd.to_datetime(signals_df["timestamp"]))
    return aligned.to_numpy(dtype=np.float64)


def build_close_decision_examples(
    signals_df: pd.DataFrame,
    entry_ohlcv: pd.DataFrame,
    decision_ohlcv: pd.DataFrame,
    decision_features: pd.DataFrame,
    feature_cols: list[str],
    decision_timeframe: str,
    max_hold_minutes: int,
    slippage: float = 0.0,
    label_mode: str = "best_remaining",
    min_hold_minutes: int = 0,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    min_improvement_bps: float = 2.0,
    regime_features: pd.DataFrame | None = None,
    regime_feature_sets: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Vectorize close-decision example generation for one asset.

    Each signal trade expands into one row per eligible close decision.
    """
    if signals_df is None or len(signals_df) == 0:
        return pd.DataFrame()

    sub = signals_df.sort_values("timestamp").reset_index(drop=True).copy()
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], utc=True)

    bar_minutes = timeframe_minutes(decision_timeframe)
    max_steps = max(int(np.ceil(max_hold_minutes / bar_minutes)), 1)

    entry_price = _lookup_entry_prices(sub, entry_ohlcv)
    valid_entry = np.isfinite(entry_price) & (entry_price > 0)
    if not valid_entry.any():
        return pd.DataFrame()
    sub = sub.loc[valid_entry].reset_index(drop=True)
    entry_dt = pd.DatetimeIndex(sub["timestamp"])
    entry_ts = entry_dt.to_numpy(dtype="datetime64[ns]")
    entry_price = entry_price[valid_entry]
    direction = sub["direction"].to_numpy(dtype=np.float64)

    decision_idx = pd.DatetimeIndex(pd.to_datetime(decision_ohlcv.index, utc=True))
    decision_times = decision_idx.to_numpy(dtype="datetime64[ns]")
    if len(decision_times) == 0:
        return pd.DataFrame()

    first_idx = decision_idx.searchsorted(entry_dt, side="right")
    offsets = np.arange(max_steps, dtype=np.int32)
    candidate_idx = first_idx[:, None] + offsets[None, :]
    clipped_idx = np.clip(candidate_idx, 0, len(decision_times) - 1)

    candidate_ts = decision_times[clipped_idx]
    elapsed_minutes = (
        (candidate_ts - entry_ts[:, None]) / np.timedelta64(1, "m")
    ).astype(np.int32)
    valid = candidate_idx < len(decision_times)
    valid &= elapsed_minutes > 0
    valid &= elapsed_minutes <= max_hold_minutes
    if min_hold_minutes > 0:
        valid &= elapsed_minutes >= min_hold_minutes

    close_prices = decision_ohlcv["close"].to_numpy(dtype=np.float64)
    candidate_close = close_prices[clipped_idx]
    exit_returns = direction[:, None] * (
        (candidate_close - entry_price[:, None]) / entry_price[:, None]
    )
    exit_returns -= slippage
    exit_returns[~valid] = -np.inf

    cummax = np.maximum.accumulate(np.where(valid, exit_returns, -np.inf), axis=1)
    cummin = np.minimum.accumulate(np.where(valid, exit_returns, np.inf), axis=1)
    cummax[~valid] = np.nan
    cummin[~valid] = np.nan

    feat = decision_features.reindex(decision_idx)[feature_cols]
    feat_vals = feat.to_numpy(dtype=np.float32)
    feature_tensor = feat_vals[clipped_idx]
    regime_sets = dict(regime_feature_sets or {})
    if regime_features is not None and "1h" not in regime_sets:
        regime_sets["1h"] = regime_features
    if not regime_sets:
        regime_sets["1h"] = pd.DataFrame(
            0.0,
            index=decision_idx,
            columns=feature_cols,
        )
    regime_tensors: dict[str, np.ndarray] = {}
    for regime_tf, frame in regime_sets.items():
        if frame is not None and len(frame) > 0:
            cols = list(dict.fromkeys(list(feature_cols) + EXTRA_REGIME_ONLY_FEATURE_COLUMNS))
            # Shift by 1 so we only see the COMPLETED previous bar.
            # Without shift, the bar labelled 10:00 contains close at
            # 10:59 — future data relative to any 15-min decision bar
            # inside the 10:00-10:59 window.
            regime_frame = frame.shift(1).reindex(decision_idx, method="ffill")
            for col in cols:
                if col not in regime_frame.columns:
                    regime_frame[col] = 0.0
            regime_frame = regime_frame[cols]
            regime_vals = regime_frame.to_numpy(dtype=np.float32)
            regime_tensors[regime_tf] = regime_vals[clipped_idx]
        else:
            zeros = np.zeros(
                (len(clipped_idx), len(feature_cols) + len(EXTRA_REGIME_ONLY_FEATURE_COLUMNS)),
                dtype=np.float32,
            )
            regime_tensors[regime_tf] = zeros

    n_trades = len(sub)
    flat_valid = valid.ravel()
    if not flat_valid.any():
        return pd.DataFrame()

    decision_ret = decision_ohlcv["close"].pct_change()
    vol_span = max(12, int(np.ceil(240 / bar_minutes)))
    vol_series = decision_ret.ewm(span=vol_span, min_periods=3).std()
    vol_fill = float(np.nanmedian(np.abs(decision_ret.to_numpy(dtype=np.float64))))
    if not np.isfinite(vol_fill) or vol_fill <= 0:
        vol_fill = 1e-4
    vol_series = vol_series.bfill().fillna(vol_fill)
    target_matrix = np.maximum(
        vol_series.to_numpy(dtype=np.float64)[clipped_idx],
        1e-6,
    )
    valid_counts = valid.sum(axis=1)
    last_valid_pos = np.maximum(valid_counts - 1, 0)
    expiry_ts = candidate_ts[np.arange(n_trades), last_valid_pos]
    gross_expiry_return = exit_returns[np.arange(n_trades), last_valid_pos]
    prev_close = np.concatenate([entry_price[:, None], candidate_close[:, :-1]], axis=1)
    step_return = direction[:, None] * np.divide(
        candidate_close - prev_close,
        prev_close,
        out=np.zeros_like(candidate_close, dtype=np.float64),
        where=prev_close > 0,
    )
    step_return[~valid] = np.nan
    rolling_2bar = decision_ohlcv["close"].pct_change(2).fillna(0.0).to_numpy(dtype=np.float64)[
        clipped_idx
    ]
    rolling_3bar = decision_ohlcv["close"].pct_change(3).fillna(0.0).to_numpy(dtype=np.float64)[
        clipped_idx
    ]
    rolling_2bar[~valid] = np.nan
    rolling_3bar[~valid] = np.nan
    realized_vol = _masked_cumstd(step_return, valid)
    time_to_expiry_minutes = np.maximum(max_hold_minutes - elapsed_minutes, 0)
    remaining_steps = np.maximum(valid_counts[:, None] - (offsets[None, :] + 1), 0)
    distance_from_mfe = exit_returns - cummax
    distance_from_mae = exit_returns - cummin
    mfe_minus_mae = cummax - cummin
    trend_slope = np.divide(
        exit_returns,
        np.maximum(offsets[None, :] + 1, 1),
        out=np.zeros_like(exit_returns, dtype=np.float64),
    )

    min_improvement_ret = max(min_improvement_bps, 0.0) / 10000.0
    if label_mode == "residual_triple_barrier":
        labels, label_end_ts, label_return_abs = _residual_triple_barrier_labels(
            exit_returns=exit_returns,
            candidate_ts=candidate_ts,
            valid=valid,
            target_matrix=target_matrix,
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            min_improvement_ret=min_improvement_ret,
        )
    else:
        labels, label_end_ts, label_return_abs = _best_remaining_labels(
            exit_returns=exit_returns,
            candidate_ts=candidate_ts,
            valid=valid,
        )

    repeated = lambda arr: np.repeat(arr, max_steps)
    tiled_steps = np.tile(offsets + 1, n_trades)
    regime_cols = list(dict.fromkeys(list(feature_cols) + EXTRA_REGIME_ONLY_FEATURE_COLUMNS))
    out = pd.DataFrame(
        {
            "trade_id": repeated(np.arange(n_trades, dtype=np.int64))[flat_valid],
            "asset": repeated(sub["asset"].to_numpy())[flat_valid],
            "timestamp": repeated(sub["timestamp"].to_numpy())[flat_valid],
            "entry_month": repeated(
                sub["timestamp"].dt.strftime("%Y-%m").to_numpy()
            )[flat_valid],
            "direction": repeated(direction)[flat_valid],
            "entry_direction": repeated(direction)[flat_valid],
            "entry_price": repeated(entry_price)[flat_valid],
            "entry_probability": repeated(
                sub["probability"].to_numpy(dtype=np.float64)
            )[flat_valid],
            "entry_confidence": repeated(
                sub["confidence"].to_numpy(dtype=np.float64)
            )[flat_valid],
            "entry_signal_strength": repeated(
                sub["signal_strength"].to_numpy(dtype=np.float64)
            )[flat_valid],
            "regime_exposure_mult": repeated(
                sub.get(
                    "regime_exposure_mult",
                    pd.Series(1.0, index=sub.index),
                ).to_numpy(dtype=np.float64)
            )[flat_valid],
            "decision_timestamp": candidate_ts.ravel()[flat_valid],
            "decision_month": pd.to_datetime(
                candidate_ts.ravel()[flat_valid], utc=True
            ).strftime("%Y-%m"),
            "expiry_timestamp": np.repeat(expiry_ts, max_steps)[flat_valid],
            "label_end_timestamp": label_end_ts.ravel()[flat_valid],
            "decision_step": tiled_steps[flat_valid],
            "bars_since_entry": tiled_steps[flat_valid],
            "minutes_since_entry": elapsed_minutes.ravel()[flat_valid],
            "time_to_expiry_minutes": time_to_expiry_minutes.ravel()[flat_valid],
            "remaining_decision_steps": remaining_steps.ravel()[flat_valid],
            "current_unrealized_return": exit_returns.ravel()[flat_valid],
            "max_favorable_return_since_entry": cummax.ravel()[flat_valid],
            "max_adverse_return_since_entry": cummin.ravel()[flat_valid],
            "realized_vol_since_entry": realized_vol.ravel()[flat_valid],
            "return_since_last_decision": step_return.ravel()[flat_valid],
            "rolling_2bar": rolling_2bar.ravel()[flat_valid],
            "rolling_3bar": rolling_3bar.ravel()[flat_valid],
            "distance_from_entry_mfe": distance_from_mfe.ravel()[flat_valid],
            "distance_from_entry_mae": distance_from_mae.ravel()[flat_valid],
            "mfe_minus_mae": mfe_minus_mae.ravel()[flat_valid],
            "entry_to_now_trend_slope": trend_slope.ravel()[flat_valid],
            "signal_prob_change_since_entry": np.zeros(flat_valid.sum(), dtype=np.float64),
            "signal_strength_change_since_entry": np.zeros(flat_valid.sum(), dtype=np.float64),
            "close_label": labels.astype(np.int8).ravel()[flat_valid],
            "exit_return": exit_returns.ravel()[flat_valid],
            "gross_exit_return": (exit_returns + slippage).ravel()[flat_valid],
            "expiry_return": np.repeat(gross_expiry_return, max_steps)[flat_valid],
            "label_return_abs": label_return_abs.ravel()[flat_valid],
        }
    )

    flat_feat = feature_tensor.reshape(n_trades * max_steps, len(feature_cols))[flat_valid]
    for idx, col in enumerate(feature_cols):
        out[col] = flat_feat[:, idx]
    for regime_tf, regime_tensor in regime_tensors.items():
        flat_regime = regime_tensor.reshape(n_trades * max_steps, len(regime_cols))[flat_valid]
        for idx, col in enumerate(regime_cols):
            out[f"{REGIME_PREFIX}{regime_tf}_{col}"] = flat_regime[:, idx]

    ordered_cols = [
        "trade_id",
        "asset",
        "timestamp",
        "entry_month",
        "direction",
        "entry_direction",
        "entry_price",
        "entry_probability",
        "entry_confidence",
        "entry_signal_strength",
        "regime_exposure_mult",
        "decision_timestamp",
        "decision_month",
        "expiry_timestamp",
        "label_end_timestamp",
        "decision_step",
        "bars_since_entry",
        "minutes_since_entry",
        "time_to_expiry_minutes",
        "remaining_decision_steps",
        "current_unrealized_return",
        "max_favorable_return_since_entry",
        "max_adverse_return_since_entry",
        "realized_vol_since_entry",
        "return_since_last_decision",
        "rolling_2bar",
        "rolling_3bar",
        "distance_from_entry_mfe",
        "distance_from_entry_mae",
        "mfe_minus_mae",
        "entry_to_now_trend_slope",
        "signal_prob_change_since_entry",
        "signal_strength_change_since_entry",
        "close_label",
        "exit_return",
        "gross_exit_return",
        "expiry_return",
        "label_return_abs",
        *feature_cols,
        *regime_feature_columns(regime_cols, regime_sets.keys()),
    ]
    out = out[ordered_cols]

    finite_feature_mask = np.isfinite(
        out[close_feature_columns(feature_cols, regime_sets.keys())].to_numpy(dtype=np.float32)
    ).all(axis=1)
    return out.loc[finite_feature_mask].reset_index(drop=True)


def _fit_isotonic_calibrator(
    raw_prob: np.ndarray,
    y_true: np.ndarray,
    close_cfg: CloseModelConfig,
) -> IsotonicRegression | None:
    if close_cfg.calibration != "isotonic":
        return None
    if len(raw_prob) < 50 or np.unique(y_true).size < 2:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_prob.astype(np.float64), y_true.astype(np.float64))
    return calibrator


def _apply_calibrator(
    raw_prob: np.ndarray,
    calibrator: IsotonicRegression | None,
) -> np.ndarray:
    if calibrator is None:
        return raw_prob.astype(np.float64, copy=False)
    return np.asarray(calibrator.predict(raw_prob.astype(np.float64)), dtype=np.float64)


def _build_selected_rows(
    rows: pd.DataFrame,
    threshold: float,
    signal_bar_minutes: int,
    close_cfg: CloseModelConfig | None = None,
) -> pd.DataFrame:
    close_cfg = close_cfg or CloseModelConfig()
    if len(rows) == 0:
        return pd.DataFrame(columns=_selection_frame_columns(True))
    payload = _build_selection_payload(rows)
    chosen_idx, model_hit = _select_trade_row_indices(payload, threshold)
    return _materialize_selected_rows_from_payload(
        payload,
        chosen_idx,
        model_hit,
        threshold,
        signal_bar_minutes,
        close_cfg,
        full=True,
    )


def _score_selected_rows(
    selected: pd.DataFrame,
    portfolio_cfg=None,
) -> tuple[float, float]:
    if len(selected) == 0:
        return -np.inf, np.inf
    if portfolio_cfg is None or len(selected) < 10:
        trade_returns = selected["fwd_return"].to_numpy(dtype=np.float64)
        if len(trade_returns) < 2:
            return -np.inf, np.inf
        std = trade_returns.std()
        sharpe = (trade_returns.mean() / std * np.sqrt(365)) if std > 0 else -np.inf
        eq = np.cumprod(1.0 + trade_returns)
        peak = np.maximum.accumulate(eq)
        dd = np.max(1.0 - np.divide(eq, peak, out=np.ones_like(eq), where=peak > 0))
        return float(sharpe), float(dd * 100.0)

    from signal_engine.portfolio import compute_daily_pnl_series
    from signal_engine.portfolio_path import simulate_portfolio_precomputed_exits
    from signal_engine.stats import compute_full_stats

    sim = simulate_portfolio_precomputed_exits(selected, portfolio_cfg)
    if sim is None or sim["n_trades"] < 10:
        return -np.inf, np.inf
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 3:
        return -np.inf, np.inf
    stats = compute_full_stats(
        dpnl,
        portfolio_cfg.starting_capital,
        sim["n_trades"],
        sim["n_trades"] / max(len(dpnl), 1),
        final_equity=sim.get("final_equity"),
    )
    return float(stats["sharpe"]), float(stats["max_dd_pct"])


def _score_from_payload_arrays(
    payload: dict[str, np.ndarray],
    chosen_idx: np.ndarray,
    model_hit: np.ndarray,
    threshold: float,
    signal_bar_minutes: int,
    close_cfg: CloseModelConfig,
    portfolio_cfg,
) -> tuple[float, float]:
    """Score a threshold selection by simulating directly from payload arrays.

    Bypasses DataFrame construction — computes the same arrays as
    ``_materialize_selected_rows_from_payload`` but feeds them straight to
    ``simulate_portfolio_from_arrays``.
    """
    if len(chosen_idx) == 0:
        return -np.inf, np.inf

    # --- replicate the array logic from _materialize (lines 1099-1145) ---
    probs = payload["close_probability"][chosen_idx]
    minutes_since_entry = payload["minutes_since_entry"][chosen_idx]
    thresholds_ladder = np.asarray(close_cfg.partial_exit_ladder, dtype=np.float64)
    remaining_ladder = np.asarray(close_cfg.partial_exit_remaining, dtype=np.float64)
    if len(remaining_ladder) != len(thresholds_ladder) + 1:
        remaining_ladder = np.array([1.0, 0.5, 0.25, 0.0], dtype=np.float64)
        thresholds_ladder = np.array([threshold, 0.75, 0.90], dtype=np.float64)

    entry_ts = payload["timestamp"][chosen_idx]
    directions = payload["direction"][chosen_idx].astype(np.float64, copy=False)
    entry_prob = payload["entry_probability"][chosen_idx].astype(np.float64, copy=False)
    entry_conf = payload["entry_confidence"][chosen_idx].astype(np.float64, copy=False)
    entry_sig = payload["entry_signal_strength"][chosen_idx].astype(np.float64, copy=False)
    regime_mult = payload["regime_exposure_mult"][chosen_idx].astype(np.float64, copy=False)
    assets = payload["asset"][chosen_idx]
    bars = np.maximum(np.ceil(minutes_since_entry / signal_bar_minutes), 1).astype(np.int32)

    if close_cfg.de_risk_mode == "partial_ladder":
        bins = np.digitize(probs, thresholds_ladder, right=False)
        remaining_fraction = remaining_ladder[np.clip(bins, 0, len(remaining_ladder) - 1)]
        remaining_fraction = np.where(model_hit, remaining_fraction, 1.0)
        close_fraction = 1.0 - remaining_fraction
        de_risk_ret = payload["exit_return"][chosen_idx].astype(np.float64, copy=False)
        expiry_ret = payload["expiry_return"][chosen_idx].astype(np.float64, copy=False)
        fwd_ret = close_fraction * de_risk_ret + remaining_fraction * expiry_ret
        de_risk_ts = np.where(
            model_hit,
            payload["decision_timestamp"][chosen_idx],
            np.datetime64("NaT"),
        )
        exit_ts = np.where(
            remaining_fraction <= 0.0,
            payload["decision_timestamp"][chosen_idx],
            payload["expiry_timestamp"][chosen_idx],
        )
    else:
        remaining_fraction = np.zeros(len(chosen_idx), dtype=np.float64)
        de_risk_ret = payload["exit_return"][chosen_idx].astype(np.float64, copy=False)
        expiry_ret = payload["expiry_return"][chosen_idx].astype(np.float64, copy=False)
        fwd_ret = payload["gross_exit_return"][chosen_idx].astype(np.float64, copy=False)
        de_risk_ts = np.where(
            model_hit,
            payload["decision_timestamp"][chosen_idx],
            np.datetime64("NaT"),
        )
        exit_ts = payload["decision_timestamp"][chosen_idx]

    return_signed = np.ones(len(chosen_idx), dtype=bool)

    # sort by entry timestamp (simulate_portfolio_from_arrays expects sorted input)
    order = np.argsort(entry_ts, kind="mergesort")
    entry_ts = entry_ts[order]
    exit_ts_arr = exit_ts[order].astype("datetime64[ns]")
    assets_sorted = assets[order]
    directions = directions[order]
    fwd_ret = fwd_ret[order]
    de_risk_ret_sorted = de_risk_ret[order]
    expiry_ret_sorted = expiry_ret[order]
    bars = bars[order]
    entry_sig = entry_sig[order]
    entry_prob = entry_prob[order]
    regime_mult = regime_mult[order]
    de_risk_ts_sorted = de_risk_ts[order].astype("datetime64[ns]")
    remaining_sorted = remaining_fraction[order].astype(np.float64)
    return_signed = return_signed[order]

    if portfolio_cfg is None or len(chosen_idx) < 10:
        # fast path: no full simulation
        trade_returns = fwd_ret
        if len(trade_returns) < 2:
            return -np.inf, np.inf
        std = trade_returns.std()
        sharpe = (trade_returns.mean() / std * np.sqrt(365)) if std > 0 else -np.inf
        eq = np.cumprod(1.0 + trade_returns)
        peak = np.maximum.accumulate(eq)
        dd = np.max(1.0 - np.divide(eq, peak, out=np.ones_like(eq), where=peak > 0))
        return float(sharpe), float(dd * 100.0)

    from signal_engine.portfolio import compute_daily_pnl_series
    from signal_engine.portfolio_path import simulate_portfolio_from_arrays
    from signal_engine.stats import compute_full_stats

    sim = simulate_portfolio_from_arrays(
        entry_ts.astype("datetime64[ns]"),
        exit_ts_arr,
        assets_sorted,
        directions,
        fwd_ret,
        de_risk_ret_sorted,
        expiry_ret_sorted,
        bars,
        entry_sig,
        entry_prob,
        regime_mult,
        de_risk_ts_sorted,
        remaining_sorted,
        return_signed,
        portfolio_cfg,
    )
    if sim is None or sim["n_trades"] < 10:
        return -np.inf, np.inf
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 3:
        return -np.inf, np.inf
    stats = compute_full_stats(
        dpnl,
        portfolio_cfg.starting_capital,
        sim["n_trades"],
        sim["n_trades"] / max(len(dpnl), 1),
        final_equity=sim.get("final_equity"),
    )
    return float(stats["sharpe"]), float(stats["max_dd_pct"])


def _choose_threshold_from_validation(
    validation_rows: pd.DataFrame | dict[str, np.ndarray],
    close_cfg: CloseModelConfig,
    signal_bar_minutes: int,
    portfolio_cfg=None,
) -> dict:
    if isinstance(validation_rows, dict):
        payload = validation_rows
        n_rows = int(payload.get("n_rows", 0))
    else:
        payload = _build_selection_payload(validation_rows)
        n_rows = len(validation_rows)

    if n_rows == 0:
        return {
            "threshold": float(close_cfg.threshold),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "positive_rate": 0.0,
            "validation_sharpe": float("-inf"),
            "validation_max_dd_pct": float("inf"),
        }

    thresholds = sorted(
        {round(float(x), 2) for x in np.linspace(0.30, 0.85, 12)}
        | {float(close_cfg.threshold)}
    )
    y_true = payload["close_label"].astype(np.int8, copy=False)
    raw = payload["close_probability"].astype(np.float64, copy=False)
    min_precision = float(close_cfg.threshold_min_precision)

    # Stage 1: cheap metrics only — prune thresholds that can't qualify
    stage1 = []
    any_qualifies = False
    for threshold in thresholds:
        y_pred = (raw >= threshold).astype(np.int8)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        positive_rate = float(y_pred.mean()) if len(y_pred) else 0.0
        chosen_idx, model_hit = _select_trade_row_indices(payload, threshold)
        qualifies = precision >= min_precision
        any_qualifies = any_qualifies or qualifies
        stage1.append({
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "positive_rate": positive_rate,
            "chosen_idx": chosen_idx,
            "model_hit": model_hit,
            "qualifies": qualifies,
        })

    # Stage 2: expensive simulation only for thresholds that could win
    candidates = []
    for entry in stage1:
        needs_sim = (
            entry["qualifies"]
            or not any_qualifies  # fallback: simulate all if none qualifies
        )
        skip = (
            entry["positive_rate"] == 0.0
            or len(entry["chosen_idx"]) < 10
        )
        if needs_sim and not skip:
            sharpe, max_dd = _score_from_payload_arrays(
                payload,
                entry["chosen_idx"],
                entry["model_hit"],
                entry["threshold"],
                signal_bar_minutes,
                close_cfg,
                portfolio_cfg,
            )
        else:
            sharpe, max_dd = float("-inf"), float("inf")
        candidates.append({
            "threshold": entry["threshold"],
            "precision": entry["precision"],
            "recall": entry["recall"],
            "f1": entry["f1"],
            "positive_rate": entry["positive_rate"],
            "validation_sharpe": sharpe,
            "validation_max_dd_pct": max_dd,
        })

    qualified = [
        row for row in candidates
        if row["precision"] >= min_precision
    ]
    pool = qualified or candidates
    pool = sorted(
        pool,
        key=lambda row: (
            row["validation_sharpe"],
            -row["validation_max_dd_pct"],
            row["precision"],
        ),
        reverse=True,
    )
    return pool[0]


def _extract_feature_importance(
    model,
    feature_names: list[str],
) -> dict[str, float]:
    if hasattr(model, "get_booster"):
        raw = model.get_booster().get_score(importance_type="gain")
        out = {}
        for name, value in raw.items():
            if name.startswith("f"):
                idx = int(name[1:])
                if 0 <= idx < len(feature_names):
                    out[feature_names[idx]] = float(value)
        return out
    if hasattr(model, "coef_"):
        coef = np.abs(np.asarray(model.coef_).ravel())
        return {
            feature_names[idx]: float(coef[idx])
            for idx in range(min(len(feature_names), len(coef)))
        }
    return {}


def _aggregate_feature_importances(
    feature_importances: list[tuple[dict[str, float], int]],
    top_n: int = 10,
) -> list[dict]:
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}
    for importance_map, weight in feature_importances:
        for feature, score in importance_map.items():
            totals[feature] = totals.get(feature, 0.0) + score * weight
            weights[feature] = weights.get(feature, 0.0) + weight
    ranked = []
    for feature, total in totals.items():
        ranked.append(
            {
                "feature": feature,
                "importance": float(total / max(weights.get(feature, 1.0), 1.0)),
            }
        )
    ranked.sort(key=lambda row: row["importance"], reverse=True)
    return ranked[:top_n]


def _build_theory_sanity(selected: pd.DataFrame) -> list[dict]:
    if len(selected) == 0 or "close_reason" not in selected.columns:
        return []
    model_closes = selected[selected["close_reason"] == "model"]
    if len(model_closes) == 0:
        return []
    keys = [
        "current_unrealized_return",
        "time_to_expiry_minutes",
        "realized_vol_since_entry",
        "mfe_minus_mae",
        "distance_from_entry_mfe",
    ]
    out = []
    for key in keys:
        if key in model_closes.columns:
            out.append({"feature": key, "mean": float(model_closes[key].mean())})
    return out


def _prediction_output_columns() -> list[str]:
    return [
        "trade_id",
        "timestamp",
        "decision_timestamp",
        "expiry_timestamp",
        "direction",
        "entry_probability",
        "entry_confidence",
        "entry_signal_strength",
        "regime_exposure_mult",
        "asset",
        "exit_return",
        "gross_exit_return",
        "expiry_return",
        "minutes_since_entry",
        "time_to_expiry_minutes",
        "current_unrealized_return",
        "realized_vol_since_entry",
        "mfe_minus_mae",
        "distance_from_entry_mfe",
        "close_label",
    ]


def _selection_frame_columns(full: bool) -> list[str]:
    base = [
        "timestamp",
        "de_risk_timestamp",
        "exit_timestamp",
        "direction",
        "probability",
        "confidence",
        "signal_strength",
        "regime_exposure_mult",
        "asset",
        "fwd_return",
        "de_risk_return",
        "expiry_return",
        "remaining_fraction",
        "size_mult_from_close_model",
        "de_risk_reason",
        "return_already_signed",
        "bars_held",
        "minutes_since_entry",
    ]
    if not full:
        return base
    return base + [
        "close_probability",
        "close_threshold",
        "close_reason",
        "close_model_type",
        "time_to_expiry_minutes",
        "current_unrealized_return",
        "realized_vol_since_entry",
        "mfe_minus_mae",
        "distance_from_entry_mfe",
    ]


def _build_selection_payload(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    if rows is None or len(rows) == 0:
        return {
            "n_rows": np.int32(0),
            "n_trades": np.int32(0),
            "trade_start_idx": np.empty(0, dtype=np.int32),
            "trade_last_idx": np.empty(0, dtype=np.int32),
            "row_group_id": np.empty(0, dtype=np.int32),
        }

    ordered = rows.sort_values(["trade_id", "decision_timestamp"]).reset_index(drop=True)
    trade_ids = ordered["trade_id"].to_numpy()
    n_rows = len(ordered)
    trade_start_idx = np.flatnonzero(
        np.r_[True, trade_ids[1:] != trade_ids[:-1]]
    ).astype(np.int32)
    trade_last_idx = np.r_[trade_start_idx[1:] - 1, n_rows - 1].astype(np.int32)
    row_group_id = np.repeat(
        np.arange(len(trade_start_idx), dtype=np.int32),
        trade_last_idx - trade_start_idx + 1,
    )

    def dt64(col: str, fallback_col: str | None = None) -> np.ndarray:
        target = col if col in ordered.columns else fallback_col
        if target is None or target not in ordered.columns:
            return np.full(n_rows, np.datetime64("NaT"), dtype="datetime64[ns]")
        return pd.to_datetime(ordered[target], utc=True).to_numpy(dtype="datetime64[ns]")

    def f64(col: str, default: float = 0.0) -> np.ndarray:
        if col not in ordered.columns:
            return np.full(n_rows, default, dtype=np.float64)
        return ordered[col].to_numpy(dtype=np.float64, copy=False)

    payload: dict[str, np.ndarray] = {
        "trade_start_idx": trade_start_idx,
        "trade_last_idx": trade_last_idx,
        "row_group_id": row_group_id,
        "close_probability": f64("close_probability"),
        "close_label": ordered["close_label"].to_numpy(dtype=np.int8, copy=False),
        "timestamp": dt64("timestamp", "decision_timestamp"),
        "decision_timestamp": dt64("decision_timestamp"),
        "expiry_timestamp": dt64("expiry_timestamp", "decision_timestamp"),
        "direction": f64("direction"),
        "entry_probability": f64("entry_probability"),
        "entry_confidence": f64("entry_confidence"),
        "entry_signal_strength": f64("entry_signal_strength"),
        "regime_exposure_mult": f64("regime_exposure_mult", 1.0),
        "exit_return": f64("exit_return"),
        "gross_exit_return": f64("gross_exit_return"),
        "expiry_return": f64("expiry_return"),
        "minutes_since_entry": f64("minutes_since_entry"),
        "time_to_expiry_minutes": f64("time_to_expiry_minutes"),
        "current_unrealized_return": f64("current_unrealized_return"),
        "realized_vol_since_entry": f64("realized_vol_since_entry"),
        "mfe_minus_mae": f64("mfe_minus_mae"),
        "distance_from_entry_mfe": f64("distance_from_entry_mfe"),
        "asset": ordered["asset"].astype(str).to_numpy(copy=False),
    }
    payload["n_rows"] = np.int32(n_rows)
    payload["n_trades"] = np.int32(len(trade_start_idx))
    return payload


def _select_trade_row_indices(
    payload: dict[str, np.ndarray],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_trades = int(payload["n_trades"])
    if n_trades == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=bool)

    probs = payload["close_probability"]
    trade_last_idx = payload["trade_last_idx"]
    hit_idx = np.flatnonzero(probs >= threshold)
    first_hit_idx = np.full(n_trades, -1, dtype=np.int32)
    if len(hit_idx) > 0:
        group_ids = payload["row_group_id"][hit_idx]
        unique_groups, first_positions = np.unique(group_ids, return_index=True)
        first_hit_idx[unique_groups] = hit_idx[first_positions].astype(np.int32)
    model_hit = first_hit_idx >= 0
    chosen_idx = np.where(model_hit, first_hit_idx, trade_last_idx).astype(np.int32)
    return chosen_idx, model_hit


def _materialize_selected_rows_from_payload(
    payload: dict[str, np.ndarray],
    chosen_idx: np.ndarray,
    model_hit: np.ndarray,
    threshold: float,
    signal_bar_minutes: int,
    close_cfg: CloseModelConfig,
    full: bool,
) -> pd.DataFrame:
    if len(chosen_idx) == 0:
        return pd.DataFrame(columns=_selection_frame_columns(full))

    probs = payload["close_probability"][chosen_idx]
    minutes_since_entry = payload["minutes_since_entry"][chosen_idx]
    thresholds = np.asarray(close_cfg.partial_exit_ladder, dtype=np.float64)
    remaining = np.asarray(close_cfg.partial_exit_remaining, dtype=np.float64)
    if len(remaining) != len(thresholds) + 1:
        remaining = np.array([1.0, 0.5, 0.25, 0.0], dtype=np.float64)
        thresholds = np.array([threshold, 0.75, 0.90], dtype=np.float64)

    if close_cfg.de_risk_mode == "partial_ladder":
        bins = np.digitize(probs, thresholds, right=False)
        remaining_fraction = remaining[np.clip(bins, 0, len(remaining) - 1)]
        remaining_fraction = np.where(model_hit, remaining_fraction, 1.0)
        close_fraction = 1.0 - remaining_fraction
        de_risk_return = payload["exit_return"][chosen_idx]
        expiry_return = payload["expiry_return"][chosen_idx]
        fwd_return = close_fraction * de_risk_return + remaining_fraction * expiry_return
        de_risk_ts = np.where(
            model_hit,
            payload["decision_timestamp"][chosen_idx],
            np.datetime64("NaT"),
        )
        exit_ts = np.where(
            remaining_fraction <= 0.0,
            payload["decision_timestamp"][chosen_idx],
            payload["expiry_timestamp"][chosen_idx],
        )
        de_risk_reason = np.where(
            ~model_hit,
            "hold_to_expiry",
            np.where(remaining_fraction <= 0.0, "full_exit", "partial_exit"),
        )
        close_reason = np.where(model_hit, "model", "expiry")
        size_mult = remaining_fraction
    else:
        remaining_fraction = np.zeros(len(chosen_idx), dtype=np.float64)
        de_risk_return = payload["exit_return"][chosen_idx]
        expiry_return = payload["expiry_return"][chosen_idx]
        fwd_return = payload["gross_exit_return"][chosen_idx]
        de_risk_ts = np.where(
            model_hit,
            payload["decision_timestamp"][chosen_idx],
            np.datetime64("NaT"),
        )
        exit_ts = payload["decision_timestamp"][chosen_idx]
        de_risk_reason = np.where(model_hit, "full_exit", "expiry")
        close_reason = np.where(model_hit, "model", "expiry")
        size_mult = np.zeros(len(chosen_idx), dtype=np.float64)

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(payload["timestamp"][chosen_idx], utc=True),
            "de_risk_timestamp": pd.to_datetime(de_risk_ts, utc=True),
            "exit_timestamp": pd.to_datetime(exit_ts, utc=True),
            "direction": payload["direction"][chosen_idx].astype(np.float32, copy=False),
            "probability": payload["entry_probability"][chosen_idx].astype(np.float32, copy=False),
            "confidence": payload["entry_confidence"][chosen_idx].astype(np.float32, copy=False),
            "signal_strength": payload["entry_signal_strength"][chosen_idx].astype(np.float32, copy=False),
            "regime_exposure_mult": payload["regime_exposure_mult"][chosen_idx].astype(np.float32, copy=False),
            "asset": pd.Series(payload["asset"][chosen_idx], copy=False, dtype="category"),
            "fwd_return": fwd_return.astype(np.float32, copy=False),
            "de_risk_return": de_risk_return.astype(np.float32, copy=False),
            "expiry_return": expiry_return.astype(np.float32, copy=False),
            "remaining_fraction": remaining_fraction.astype(np.float32, copy=False),
            "size_mult_from_close_model": size_mult.astype(np.float32, copy=False),
            "de_risk_reason": pd.Series(de_risk_reason, copy=False).astype("category"),
            "return_already_signed": np.ones(len(chosen_idx), dtype=bool),
            "bars_held": np.maximum(
                np.ceil(minutes_since_entry / signal_bar_minutes),
                1,
            ).astype(np.int16),
            "minutes_since_entry": minutes_since_entry.astype(np.float32, copy=False),
        }
    )
    if not full:
        return frame

    frame["close_probability"] = probs.astype(np.float32, copy=False)
    frame["close_threshold"] = np.full(len(chosen_idx), threshold, dtype=np.float32)
    frame["close_reason"] = pd.Series(close_reason, copy=False).astype("category")
    frame["close_model_type"] = close_cfg.model_type
    frame["time_to_expiry_minutes"] = payload["time_to_expiry_minutes"][chosen_idx].astype(np.float32, copy=False)
    frame["current_unrealized_return"] = payload["current_unrealized_return"][chosen_idx].astype(np.float32, copy=False)
    frame["realized_vol_since_entry"] = payload["realized_vol_since_entry"][chosen_idx].astype(np.float32, copy=False)
    frame["mfe_minus_mae"] = payload["mfe_minus_mae"][chosen_idx].astype(np.float32, copy=False)
    frame["distance_from_entry_mfe"] = payload["distance_from_entry_mfe"][chosen_idx].astype(np.float32, copy=False)
    return frame


def _wf_config_hash(close_cfg: CloseModelConfig, portfolio_cfg) -> str:
    cfg_d = close_cfg.model_dump()
    port_d = portfolio_cfg.model_dump() if portfolio_cfg is not None else {}
    blob = json.dumps({"c": cfg_d, "p": port_d}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _wf_feature_hash(feature_cols: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(feature_cols)).encode()).hexdigest()[:12]


def _wf_data_fingerprint(timestamps: np.ndarray, labels: np.ndarray) -> str:
    n = len(timestamps)
    if n == 0:
        return "empty"
    parts = [
        str(n),
        str(timestamps[0]),
        str(timestamps[-1]),
        str(int(labels.sum())),
    ]
    if n > 4:
        parts.append(str(timestamps[n // 4]))
        parts.append(str(timestamps[3 * n // 4]))
    if n > 2:
        parts.append(str(timestamps[n // 2]))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _wf_serialize(obj) -> bytes | None:
    if obj is None or _joblib is None:
        return None
    buf = io.BytesIO()
    _joblib.dump(obj, buf, compress=3)
    return buf.getvalue()


def _wf_deserialize(data: bytes | None):
    if data is None or _joblib is None:
        return None
    return _joblib.load(io.BytesIO(data))


def walk_forward_close_prediction_cache(
    decision_examples: pd.DataFrame,
    feature_cols: list[str],
    lookback_months: int,
    close_cfg: CloseModelConfig,
    test_start: str = "2022-01",
    signal_bar_minutes: int = 15,
    min_train_rows: int = 200,
    portfolio_cfg=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run monthly walk-forward close-model inference and return raw probability artifacts."""
    if decision_examples is None or len(decision_examples) == 0:
        empty_diag = {
            "model_type": close_cfg.model_type,
            "label_mode": close_cfg.label_mode,
            "months": [],
            "summary": {},
        }
        return pd.DataFrame(), pd.DataFrame(), empty_diag

    examples = decision_examples.sort_values(
        ["timestamp", "decision_timestamp"]
    ).reset_index(drop=True)
    examples["timestamp"] = pd.to_datetime(examples["timestamp"], utc=True)
    examples["decision_timestamp"] = pd.to_datetime(
        examples["decision_timestamp"], utc=True
    )
    if "decision_month" not in examples.columns:
        examples["decision_month"] = examples["decision_timestamp"].dt.strftime("%Y-%m")
    if "label_end_timestamp" not in examples.columns:
        examples["label_end_timestamp"] = examples.groupby("trade_id", sort=False)[
            "decision_timestamp"
        ].transform("max")
    if "expiry_timestamp" not in examples.columns:
        examples["expiry_timestamp"] = examples.groupby("trade_id", sort=False)[
            "decision_timestamp"
        ].transform("max")
    examples["label_end_timestamp"] = pd.to_datetime(
        examples["label_end_timestamp"], utc=True
    )
    examples["expiry_timestamp"] = pd.to_datetime(
        examples["expiry_timestamp"], utc=True
    )
    if "label_return_abs" not in examples.columns:
        reversed_abs = examples["exit_return"].abs().iloc[::-1]
        reversed_trade = examples["trade_id"].iloc[::-1]
        examples["label_return_abs"] = (
            reversed_abs.groupby(reversed_trade, sort=False).cummax().iloc[::-1].to_numpy()
        )
    if "gross_exit_return" not in examples.columns:
        examples["gross_exit_return"] = examples["exit_return"]
    if "expiry_return" not in examples.columns:
        examples["expiry_return"] = examples.groupby("trade_id", sort=False)[
            "gross_exit_return"
        ].transform("last")

    regime_timeframes = list(
        dict.fromkeys(close_cfg.regime_timeframes or ["1h"])
    )
    all_feature_cols = close_feature_columns(
        feature_cols,
        regime_timeframes=regime_timeframes,
    )
    for col in all_feature_cols:
        if col not in examples.columns:
            examples[col] = 0.0
    finite_feature_mask = np.isfinite(
        examples[all_feature_cols].to_numpy(dtype=np.float32)
    ).all(axis=1)
    examples = examples.loc[finite_feature_mask].reset_index(drop=True)
    if len(examples) == 0:
        empty_diag = {
            "model_type": close_cfg.model_type,
            "label_mode": close_cfg.label_mode,
            "months": [],
            "summary": {},
        }
        return pd.DataFrame(), pd.DataFrame(), empty_diag

    entry_month_ord = pd.PeriodIndex(
        examples["entry_month"].astype(str), freq="M"
    ).asi8.astype(np.int32)
    decision_month_ord = pd.PeriodIndex(
        examples["decision_month"].astype(str), freq="M"
    ).asi8.astype(np.int32)
    examples["entry_month_ord"] = entry_month_ord
    examples["decision_month_ord"] = decision_month_ord
    if "regime_exposure_mult" not in examples.columns:
        examples["regime_exposure_mult"] = 1.0
    feature_matrix = examples[all_feature_cols].to_numpy(dtype=np.float32, copy=True)
    label_array = examples["close_label"].to_numpy(dtype=np.int8)
    entry_month_ord_arr = examples["entry_month_ord"].to_numpy(dtype=np.int32)
    decision_month_ord_arr = examples["decision_month_ord"].to_numpy(dtype=np.int32)
    decision_ts_arr = examples["decision_timestamp"].to_numpy(dtype="datetime64[ns]")
    label_end_arr = examples["label_end_timestamp"].to_numpy(dtype="datetime64[ns]")
    label_return_abs_arr = examples["label_return_abs"].to_numpy(dtype=np.float64)
    output_cols = _prediction_output_columns()
    months = sorted(
        pd.Period(m)
        for m in examples["entry_month"].dropna().astype(str).unique()
        if m >= test_start
    )
    month_table = [
        {
            "period": month,
            "month_ord": int(month.ordinal),
            "month_start": pd.Timestamp(month.start_time, tz="UTC"),
            "test_idx": np.flatnonzero(entry_month_ord_arr == month.ordinal),
        }
        for month in months
    ]
    prediction_rows: list[pd.DataFrame] = []
    validation_rows: list[pd.DataFrame] = []
    diagnostics = {
        "model_type": close_cfg.model_type,
        "label_mode": close_cfg.label_mode,
        "months": [],
    }
    feature_importances: list[tuple[dict[str, float], int]] = []
    last_train_month: pd.Period | None = None
    scaler: StandardScaler | None = None
    model = None
    calibrator: IsotonicRegression | None = None

    # Per-month walk-forward cache setup
    _wf_cache_dir: Path | None = None
    if _joblib is not None and os.getenv("ML_CLOSE_WF_CACHE_DISABLE") != "1":
        _ch = _wf_config_hash(close_cfg, portfolio_cfg)
        _fh = _wf_feature_hash(feature_cols)
        _wf_cache_dir = (
            Path(os.getenv("ML_CLOSE_WF_CACHE_DIR", "data/cache/close_model_wf"))
            / f"{_ch}_{_fh}"
            / f"lb{lookback_months}_sbm{signal_bar_minutes}"
        )
        _wf_cache_dir.mkdir(parents=True, exist_ok=True)

    import time as _time_mod
    _wf_t0 = _time_mod.perf_counter()
    _wf_cache_hits = 0
    for month_info in month_table:
        month = month_info["period"]
        month_ord = month_info["month_ord"]
        train_start_ord = (month - lookback_months + 1).ordinal
        month_start = month_info["month_start"]
        train_mask = (
            (decision_month_ord_arr >= train_start_ord)
            & (decision_month_ord_arr < month_ord)
            & (decision_ts_arr < month_start.to_datetime64())
            & (label_end_arr < month_start.to_datetime64())
        )
        test_idx = month_info["test_idx"]
        if len(test_idx) == 0:
            continue

        # --- Per-month cache check ---
        _month_cache_path: Path | None = None
        _train_fp = ""
        _test_fp = ""
        if _wf_cache_dir is not None:
            _month_cache_path = _wf_cache_dir / f"month_{month_ord}.pkl"
            _train_fp = _wf_data_fingerprint(
                decision_ts_arr[train_mask], label_array[train_mask],
            )
            _test_fp = _wf_data_fingerprint(
                decision_ts_arr[test_idx], label_array[test_idx],
            )
            if _month_cache_path.exists():
                try:
                    _cached = _joblib.load(_month_cache_path)
                    if (
                        isinstance(_cached, dict)
                        and _cached.get("version") == _WF_CACHE_VERSION
                        and _cached.get("fp_train") == _train_fp
                        and _cached.get("fp_test") == _test_fp
                    ):
                        prediction_rows.append(_cached["pred"])
                        if _cached.get("val") is not None:
                            validation_rows.append(_cached["val"])
                        diagnostics["months"].append(_cached["diag"])
                        if _cached.get("fi") is not None:
                            feature_importances.append(_cached["fi"])
                        if _cached.get("model") is not None:
                            model = _wf_deserialize(_cached["model"])
                            scaler = _wf_deserialize(_cached.get("scaler"))
                            calibrator = _wf_deserialize(_cached.get("cal"))
                        if _cached.get("ltm") is not None:
                            last_train_month = pd.Period(
                                ordinal=_cached["ltm"], freq="M",
                            )
                        _wf_cache_hits += 1
                        continue
                except Exception as _exc:
                    _wf_logger.debug("stale/corrupt cache month %s: %s", month, _exc)

        month_diag = {
            "month": str(month),
            "n_train": 0,
            "n_validation": 0,
        }

        need_retrain = (
            model is None
            or last_train_month is None
            or (month.ordinal - last_train_month.ordinal)
            >= max(close_cfg.retrain_frequency_months, 1)
        )

        if need_retrain:
            train_idx = np.flatnonzero(train_mask)
            if close_cfg.pct_embargo > 0.0 and len(train_idx) > 1:
                embargo_rows = int(len(train_idx) * close_cfg.pct_embargo)
                if embargo_rows > 0:
                    train_idx = train_idx[:-embargo_rows]
            month_diag["n_train"] = int(len(train_idx))
            scaler = None
            model = None
            calibrator = None
            if (
                len(train_idx) >= min_train_rows
                and np.unique(label_array[train_idx]).size > 1
            ):
                fit_idx = train_idx
                val_idx = np.array([], dtype=np.int64)

                if close_cfg.model_type == "xgboost":
                    unique_train_months = np.unique(decision_month_ord_arr[train_idx])
                    if len(unique_train_months) >= 2:
                        val_month_ord = int(unique_train_months.max())
                        val_start = pd.Period(ordinal=val_month_ord, freq="M").start_time
                        val_start = pd.Timestamp(val_start, tz="UTC")
                        val_idx = train_idx[
                            decision_month_ord_arr[train_idx] == val_month_ord
                        ]
                        fit_idx = train_idx[
                            (decision_month_ord_arr[train_idx] < val_month_ord)
                            & (label_end_arr[train_idx] < val_start.to_datetime64())
                        ]
                        if len(fit_idx) < min_train_rows or len(val_idx) < 100:
                            fit_idx = train_idx
                            val_idx = np.array([], dtype=np.int64)

                fit_weights = compute_afml_close_sample_weights_arrays(
                    decision_ts_arr[fit_idx],
                    label_end_arr[fit_idx],
                    label_return_abs_arr[fit_idx],
                    clf_last_w=close_cfg.time_decay_last_weight,
                )

                if close_cfg.model_type == "xgboost":
                    positives = int(label_array[fit_idx].sum())
                    negatives = int(len(fit_idx) - positives)
                    scale_pos_weight = negatives / max(positives, 1) if positives > 0 else 1.0
                    best_params: dict[str, float | int] = {}
                    best_rounds = None
                    train_model = None
                    best_val_rows: pd.DataFrame | None = None
                    if len(val_idx) > 0 and np.unique(label_array[val_idx]).size > 1:
                        val_base = examples.iloc[val_idx][output_cols].copy()
                        val_base["target_month_ord"] = np.int32(month_ord)
                        best_score = (-np.inf, -np.inf, -np.inf)

                        # Shared read-only arrays for all grid candidates
                        _fit_X = feature_matrix[fit_idx]
                        _fit_y = label_array[fit_idx]
                        _val_X = feature_matrix[val_idx]
                        _val_y = label_array[val_idx]
                        # Build validation payload ONCE — reuse across all candidates
                        _val_payload_base = _build_selection_payload(val_base)

                        # --- PurgedKFold CV for grid search (AFML Ch 7) ---
                        # Use purged CV log_loss to select hyperparams, then
                        # threshold search only on the best candidate.
                        from afml.cross_val import PurgedKFold as _PurgedKFold
                        from sklearn.metrics import log_loss as _log_loss

                        _t1_train = pd.Series(
                            label_end_arr[train_idx],
                            index=pd.DatetimeIndex(decision_ts_arr[train_idx]),
                        )
                        _cv_usable = len(train_idx) >= 500 and os.getenv("ML_CLOSE_SKIP_CV") != "1"
                        if _cv_usable:
                            try:
                                _purged_cv = _PurgedKFold(
                                    n_splits=3,
                                    t1=_t1_train,
                                    pct_embargo=close_cfg.pct_embargo,
                                )
                                _cv_X = feature_matrix[train_idx]
                                _cv_y = label_array[train_idx]
                                _cv_w = compute_afml_close_sample_weights_arrays(
                                    decision_ts_arr[train_idx],
                                    label_end_arr[train_idx],
                                    label_return_abs_arr[train_idx],
                                    clf_last_w=close_cfg.time_decay_last_weight,
                                )
                            except Exception:
                                _cv_usable = False

                        def _eval_grid_candidate(params):
                            # Phase 1: PurgedKFold CV score (fast, no threshold search)
                            cv_score = -np.inf
                            if _cv_usable:
                                fold_losses = []
                                try:
                                    for _tr, _te in _purged_cv.split(_cv_X, _cv_y):
                                        _m = make_xgb_close_model_with_params(
                                            scale_pos_weight=scale_pos_weight,
                                            params=params,
                                            early_stopping_rounds=close_cfg.xgb_early_stopping_rounds,
                                            n_jobs=1,
                                        )
                                        _m.fit(
                                            _cv_X[_tr], _cv_y[_tr],
                                            sample_weight=_cv_w[_tr],
                                            eval_set=[(_cv_X[_te], _cv_y[_te])],
                                            verbose=False,
                                        )
                                        _p = _m.predict_proba(_cv_X[_te])[:, 1]
                                        fold_losses.append(_log_loss(_cv_y[_te], _p))
                                    cv_score = -float(np.mean(fold_losses))  # negative log_loss (higher=better)
                                except Exception:
                                    pass

                            # Phase 2: Train on fit_idx, validate on val_idx for threshold
                            cand = make_xgb_close_model_with_params(
                                scale_pos_weight=scale_pos_weight,
                                params=params,
                                early_stopping_rounds=close_cfg.xgb_early_stopping_rounds,
                                n_jobs=1,
                            )
                            cand.fit(
                                _fit_X, _fit_y,
                                sample_weight=fit_weights,
                                eval_set=[(_val_X, _val_y)],
                                verbose=False,
                            )
                            rv = cand.predict_proba(_val_X)[:, 1]
                            cal = _fit_isotonic_calibrator(rv, _val_y, close_cfg)
                            calibrated = _apply_calibrator(rv, cal)
                            val_payload = {**_val_payload_base, "close_probability": calibrated}
                            td = _choose_threshold_from_validation(
                                val_payload, close_cfg=close_cfg,
                                signal_bar_minutes=signal_bar_minutes,
                                portfolio_cfg=portfolio_cfg,
                            )
                            val_sharpe = float(td["validation_sharpe"])
                            # Combined score: CV log_loss (AFML-proper) + validation Sharpe
                            if _cv_usable and np.isfinite(cv_score):
                                score = (cv_score, val_sharpe, float(td["precision"]))
                            else:
                                score = (val_sharpe, -float(td["validation_max_dd_pct"]), float(td["precision"]))
                            vr = val_base.copy()
                            vr["close_probability"] = calibrated
                            return cand, cal, dict(params), td, rv, vr, score

                        grid_candidates = xgb_grid_candidates(close_cfg)
                        _grid_workers = min(
                            len(grid_candidates),
                            max(1, os.cpu_count() or 1),
                        )
                        with ThreadPoolExecutor(max_workers=_grid_workers) as _gpool:
                            grid_results = list(_gpool.map(
                                _eval_grid_candidate, grid_candidates,
                            ))

                        for cand, cal, params, td, rv, vr, score in grid_results:
                            if score > best_score:
                                best_score = score
                                best_params = params
                                best_rounds = getattr(cand, "best_iteration", None)
                                train_model = cand
                                calibrator = cal
                                month_diag.update(td)
                                month_diag["n_validation"] = int(len(val_idx))
                                best_val_rows = vr.copy()
                    if train_model is None:
                        train_model = make_xgb_close_model_with_params(
                            scale_pos_weight=scale_pos_weight,
                            params=best_params,
                            n_jobs=1,
                        )
                        train_model.fit(
                            feature_matrix[fit_idx],
                            label_array[fit_idx],
                            sample_weight=fit_weights,
                        )
                    if len(val_idx) > 0 and np.unique(label_array[val_idx]).size > 1:
                        raw_val = train_model.predict_proba(feature_matrix[val_idx])[:, 1]
                        calibrator = _fit_isotonic_calibrator(
                            raw_val,
                            label_array[val_idx],
                            close_cfg,
                        )
                        if best_val_rows is None:
                            best_val_rows = examples.iloc[val_idx][output_cols].copy()
                            best_val_rows["target_month_ord"] = np.int32(month_ord)
                            best_val_rows["close_probability"] = _apply_calibrator(
                                raw_val,
                                calibrator,
                            )
                    if best_val_rows is not None:
                        validation_rows.append(best_val_rows)

                    if len(val_idx) == 0:
                        full_weights = fit_weights
                    else:
                        full_weights = compute_afml_close_sample_weights_arrays(
                            decision_ts_arr[train_idx],
                            label_end_arr[train_idx],
                            label_return_abs_arr[train_idx],
                            clf_last_w=close_cfg.time_decay_last_weight,
                        )
                    positives = int(label_array[train_idx].sum())
                    negatives = int(len(train_idx) - positives)
                    scale_pos_weight = negatives / max(positives, 1) if positives > 0 else 1.0
                    if best_rounds is not None and best_rounds >= 1:
                        best_params = {
                            **best_params,
                            "n_estimators": int(best_rounds) + 1,
                        }
                    model = make_xgb_close_model_with_params(
                        scale_pos_weight=scale_pos_weight,
                        params=best_params,
                        n_jobs=1,
                    )
                    model.fit(
                        feature_matrix[train_idx],
                        label_array[train_idx],
                        sample_weight=full_weights,
                    )
                    feature_importances.append(
                        (_extract_feature_importance(model, all_feature_cols), len(train_idx))
                    )
                else:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(feature_matrix[train_idx]).astype(
                        np.float32,
                        copy=False,
                    )
                    model = make_close_model(warm_start=True)
                    model.fit(
                        X_train,
                        label_array[train_idx],
                        sample_weight=fit_weights if close_cfg.use_afml_optimization else None,
                    )
                    feature_importances.append(
                        (_extract_feature_importance(model, all_feature_cols), len(train_idx))
                    )
            last_train_month = month

        month_examples = examples.iloc[test_idx][output_cols].copy()
        month_examples["target_month_ord"] = np.int32(month_ord)
        if model is not None:
            if close_cfg.model_type == "xgboost":
                raw_test = model.predict_proba(feature_matrix[test_idx])[:, 1]
            else:
                X_test = scaler.transform(feature_matrix[test_idx]).astype(
                    np.float32,
                    copy=False,
                )
                raw_test = model.predict_proba(X_test)[:, 1]
            month_examples["close_probability"] = _apply_calibrator(raw_test, calibrator)
        else:
            month_examples["close_probability"] = 0.0

        # Rank IC: cross-sectional signal quality (AFML Ch 14)
        if model is not None and "close_label" in month_examples.columns:
            try:
                from afml.evaluation import rank_ic as _rank_ic
                _pred = pd.Series(
                    month_examples["close_probability"].to_numpy(dtype=np.float64),
                )
                _actual = pd.Series(
                    month_examples["close_label"].to_numpy(dtype=np.float64),
                )
                _ric = _rank_ic(_pred, _actual)
                if np.isfinite(_ric):
                    month_diag["rank_ic"] = round(float(_ric), 4)
            except Exception:
                pass

        prediction_rows.append(month_examples)
        diagnostics["months"].append(month_diag)

        # --- Per-month cache write ---
        if _month_cache_path is not None:
            try:
                _last_val = validation_rows[-1] if validation_rows else None
                _last_fi = feature_importances[-1] if feature_importances else None
                _entry = {
                    "version": _WF_CACHE_VERSION,
                    "fp_train": _train_fp,
                    "fp_test": _test_fp,
                    "pred": month_examples,
                    "val": _last_val,
                    "diag": month_diag,
                    "fi": _last_fi,
                    "model": _wf_serialize(model),
                    "scaler": _wf_serialize(scaler),
                    "cal": _wf_serialize(calibrator),
                    "ltm": last_train_month.ordinal if last_train_month else None,
                }
                _tmp = _month_cache_path.with_suffix(".tmp")
                _joblib.dump(_entry, _tmp, compress=3)
                os.replace(_tmp, _month_cache_path)
            except Exception as _exc:
                _wf_logger.warning("cache write failed month %s: %s", month, _exc)

    _wf_logger.info(
        "walk_forward %d months in %.1fs (%d cache hits, %d computed)",
        len(month_table),
        _time_mod.perf_counter() - _wf_t0,
        _wf_cache_hits,
        len(month_table) - _wf_cache_hits,
    )

    if not prediction_rows:
        diagnostics["summary"] = {}
        return pd.DataFrame(), pd.DataFrame(), diagnostics
    out = pd.concat(prediction_rows, ignore_index=True).sort_values(
        ["timestamp", "decision_timestamp"]
    )
    if len(out) > 0:
        numeric_cols = [
            "direction",
            "entry_probability",
            "entry_confidence",
            "entry_signal_strength",
            "regime_exposure_mult",
            "exit_return",
            "gross_exit_return",
            "expiry_return",
            "minutes_since_entry",
            "time_to_expiry_minutes",
            "current_unrealized_return",
            "realized_vol_since_entry",
            "mfe_minus_mae",
            "distance_from_entry_mfe",
            "close_probability",
        ]
        for col in numeric_cols:
            if col in out.columns:
                out[col] = out[col].astype(np.float32, copy=False)
        if "target_month_ord" in out.columns:
            out["target_month_ord"] = out["target_month_ord"].astype(np.int32, copy=False)
        if "asset" in out.columns:
            out["asset"] = out["asset"].astype("category")
    validation_out = (
        pd.concat(validation_rows, ignore_index=True).sort_values(
            ["target_month_ord", "decision_timestamp"]
        )
        if validation_rows
        else pd.DataFrame()
    )
    if len(validation_out) > 0:
        numeric_cols = [
            "direction",
            "entry_probability",
            "entry_confidence",
            "entry_signal_strength",
            "regime_exposure_mult",
            "exit_return",
            "gross_exit_return",
            "expiry_return",
            "minutes_since_entry",
            "time_to_expiry_minutes",
            "current_unrealized_return",
            "realized_vol_since_entry",
            "mfe_minus_mae",
            "distance_from_entry_mfe",
            "close_probability",
        ]
        for col in numeric_cols:
            if col in validation_out.columns:
                validation_out[col] = validation_out[col].astype(np.float32, copy=False)
        if "target_month_ord" in validation_out.columns:
            validation_out["target_month_ord"] = validation_out["target_month_ord"].astype(np.int32, copy=False)
        if "asset" in validation_out.columns:
            validation_out["asset"] = validation_out["asset"].astype("category")
    diagnostics["summary"] = {
        "calibration": close_cfg.calibration,
        "calibration_note": (
            "probability calibration disabled in this path"
            if close_cfg.calibration == "none"
            else "probability calibration enabled"
        ),
        "top_features": _aggregate_feature_importances(feature_importances),
    }
    return out.reset_index(drop=True), validation_out.reset_index(drop=True), diagnostics


def select_close_predictions(
    prediction_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    close_cfg: CloseModelConfig,
    signal_bar_minutes: int = 15,
    portfolio_cfg=None,
    prediction_diagnostics: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    if prediction_rows is None or len(prediction_rows) == 0:
        empty_diag = prediction_diagnostics or {
            "model_type": close_cfg.model_type,
            "label_mode": close_cfg.label_mode,
            "months": [],
            "summary": {},
        }
        return pd.DataFrame(), empty_diag

    prediction_rows = prediction_rows.copy()
    if "target_month_ord" not in prediction_rows.columns:
        source = (
            prediction_rows["target_month"].astype(str)
            if "target_month" in prediction_rows.columns
            else pd.to_datetime(prediction_rows["timestamp"], utc=True).dt.to_period("M").astype(str)
        )
        prediction_rows["target_month_ord"] = pd.PeriodIndex(source, freq="M").asi8.astype(np.int32)
    if validation_rows is not None and len(validation_rows) > 0:
        validation_rows = validation_rows.copy()
        if "target_month_ord" not in validation_rows.columns:
            source = (
                validation_rows["target_month"].astype(str)
                if "target_month" in validation_rows.columns
                else pd.to_datetime(validation_rows["timestamp"], utc=True).dt.to_period("M").astype(str)
            )
            validation_rows["target_month_ord"] = pd.PeriodIndex(source, freq="M").asi8.astype(np.int32)

    rows = prediction_rows.sort_values(
        ["target_month_ord", "timestamp", "decision_timestamp"]
    ).reset_index(drop=True)
    val_rows_all = (
        validation_rows.sort_values(["target_month_ord", "decision_timestamp"]).reset_index(drop=True)
        if validation_rows is not None and len(validation_rows) > 0
        else pd.DataFrame()
    )
    prediction_diagnostics = prediction_diagnostics or {
        "model_type": close_cfg.model_type,
        "label_mode": close_cfg.label_mode,
        "months": [],
        "summary": {},
    }
    month_meta = {
        int(pd.Period(str(item.get("month")), freq="M").ordinal): dict(item)
        for item in prediction_diagnostics.get("months", [])
        if item.get("month") is not None
    }
    selected_rows: list[pd.DataFrame] = []
    diagnostics = {
        "model_type": close_cfg.model_type,
        "label_mode": close_cfg.label_mode,
        "months": [],
    }

    month_ord_arr = rows["target_month_ord"].to_numpy(dtype=np.int32, copy=False)
    unique_months, month_start_pos = np.unique(month_ord_arr, return_index=True)
    month_end_pos = np.r_[month_start_pos[1:], len(rows)]
    val_payloads: dict[int, dict[str, np.ndarray]] = {}
    if len(val_rows_all) > 0:
        val_month_ord_arr = val_rows_all["target_month_ord"].to_numpy(dtype=np.int32, copy=False)
        val_unique, val_starts = np.unique(val_month_ord_arr, return_index=True)
        val_ends = np.r_[val_starts[1:], len(val_rows_all)]
        for month_ord, start, end in zip(val_unique, val_starts, val_ends, strict=False):
            val_payloads[int(month_ord)] = _build_selection_payload(val_rows_all.iloc[start:end])

    for month_ord, start, end in zip(unique_months, month_start_pos, month_end_pos, strict=False):
        month_rows = rows.iloc[start:end]
        month_payload = _build_selection_payload(month_rows)
        cached_meta = month_meta.get(int(month_ord), {})
        if "threshold" in cached_meta and "validation_sharpe" in cached_meta:
            # Reuse threshold from walk_forward — skip expensive re-search
            threshold_diag = cached_meta
        else:
            threshold_diag = _choose_threshold_from_validation(
                val_payloads.get(int(month_ord), {"n_rows": np.int32(0)}),
                close_cfg=close_cfg,
                signal_bar_minutes=signal_bar_minutes,
                portfolio_cfg=portfolio_cfg,
            )
        month_label = str(pd.Period(ordinal=int(month_ord), freq="M"))
        month_diag = {**cached_meta, **threshold_diag, "month": month_label}
        if "n_validation" not in month_diag:
            month_diag["n_validation"] = int(val_payloads.get(int(month_ord), {}).get("n_rows", 0))
        threshold = float(month_diag.get("threshold", close_cfg.threshold))
        chosen_idx, model_hit = _select_trade_row_indices(month_payload, threshold)
        chosen = _materialize_selected_rows_from_payload(
            month_payload,
            chosen_idx,
            model_hit,
            threshold,
            signal_bar_minutes,
            close_cfg,
            full=True,
        )
        selected_rows.append(
            chosen[_selection_frame_columns(True)]
        )
        diagnostics["months"].append(month_diag)

    if not selected_rows:
        diagnostics["summary"] = prediction_diagnostics.get("summary", {})
        return pd.DataFrame(), diagnostics
    out = pd.concat(selected_rows, ignore_index=True).sort_values("timestamp")
    month_weights = np.array(
        [max(int(row.get("n_validation", 0)), 1) for row in diagnostics["months"]],
        dtype=np.float64,
    )
    base_summary = prediction_diagnostics.get("summary", {})
    diagnostics["summary"] = {
        "calibration": base_summary.get("calibration", close_cfg.calibration),
        "calibration_note": base_summary.get(
            "calibration_note",
            "probability calibration disabled in this path",
        ),
        "avg_threshold": float(
            np.average(
                [row.get("threshold", close_cfg.threshold) for row in diagnostics["months"]],
                weights=month_weights,
            )
        ),
        "precision": float(
            np.average([row.get("precision", 0.0) for row in diagnostics["months"]], weights=month_weights)
        ),
        "recall": float(
            np.average([row.get("recall", 0.0) for row in diagnostics["months"]], weights=month_weights)
        ),
        "f1": float(
            np.average([row.get("f1", 0.0) for row in diagnostics["months"]], weights=month_weights)
        ),
        "positive_rate": float(
            np.average([row.get("positive_rate", 0.0) for row in diagnostics["months"]], weights=month_weights)
        ),
        "top_features": base_summary.get("top_features", []),
        "theory_sanity": _build_theory_sanity(out),
    }
    return out.reset_index(drop=True), diagnostics


def walk_forward_close_predictions(
    decision_examples: pd.DataFrame,
    feature_cols: list[str],
    lookback_months: int,
    close_cfg: CloseModelConfig,
    test_start: str = "2022-01",
    signal_bar_minutes: int = 15,
    min_train_rows: int = 200,
    portfolio_cfg=None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    prediction_rows, validation_rows, prediction_diag = walk_forward_close_prediction_cache(
        decision_examples,
        feature_cols,
        lookback_months,
        close_cfg,
        test_start=test_start,
        signal_bar_minutes=signal_bar_minutes,
        min_train_rows=min_train_rows,
        portfolio_cfg=portfolio_cfg,
    )
    return select_close_predictions(
        prediction_rows,
        validation_rows,
        close_cfg,
        signal_bar_minutes=signal_bar_minutes,
        portfolio_cfg=portfolio_cfg,
        prediction_diagnostics=prediction_diag,
    )
