"""Parallel walk-forward close model — 12x speedup over sequential.

Reimplements the core WF training loop from close_model.py with:
- ProcessPoolExecutor across months (12 workers = 12 cores)
- No grid search (uses validated defaults)
- Numpy memmap for shared data (no copies)
- n_jobs=1 per worker (avoids thread contention)

Usage:
    from signal_engine.parallel_close_wf import parallel_walk_forward_close
    predictions = parallel_walk_forward_close(examples, feature_cols, ...)
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _train_predict_month(
    month_ord: int,
    test_idx: np.ndarray,
    train_start_ord: int,
    feature_path: str,
    label_path: str,
    decision_month_ord_path: str,
    decision_ts_path: str,
    label_end_path: str,
    shape: tuple[int, int],
    n_total: int,
    scale_pos_weight: float,
    use_gpu: bool = False,
) -> dict | None:
    """Process one month: train XGBoost on past data, predict test set."""
    import warnings
    warnings.filterwarnings("ignore")

    # Load shared data via memmap (no copy — OS page cache)
    feat = np.load(feature_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    month_ords = np.load(decision_month_ord_path, mmap_mode="r")
    decision_ts = np.load(decision_ts_path, mmap_mode="r")
    label_end = np.load(label_end_path, mmap_mode="r")

    # Build train mask: months in [train_start_ord, month_ord)
    month_start_ns = pd.Period(ordinal=month_ord, freq="M").start_time.value
    train_mask = (
        (month_ords >= train_start_ord) &
        (month_ords < month_ord) &
        (decision_ts < month_start_ns) &
        (label_end < month_start_ns)
    )
    train_idx = np.flatnonzero(train_mask)

    if len(train_idx) < 200 or len(test_idx) == 0:
        return None

    X_train = np.array(feat[train_idx])  # copy from memmap
    y_train = np.array(labels[train_idx])
    X_test = np.array(feat[test_idx])

    if len(np.unique(y_train)) < 2:
        return None

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train XGBoost — n_jobs=1 (one core per worker)
    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        scale_pos_weight=max(scale_pos_weight, 1e-6),
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        random_state=42, n_jobs=1, verbosity=0,
    )
    model.fit(X_train_s, y_train)

    # Predict
    proba = model.predict_proba(X_test_s)[:, 1]

    return {
        "month_ord": month_ord,
        "test_idx": test_idx,
        "close_probability": proba,
        "n_train": len(train_idx),
    }


def parallel_walk_forward_close(
    examples: pd.DataFrame,
    feature_cols: list[str],
    lookback_months: int = 10,
    test_start: str = "2021-01",
    max_workers: int | None = None,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """Parallel walk-forward close model prediction.

    Returns examples DataFrame with added 'close_probability' column.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    t0 = time.time()

    # Ensure timestamps are UTC
    examples["timestamp"] = pd.to_datetime(examples["timestamp"], utc=True)
    examples["decision_timestamp"] = pd.to_datetime(examples["decision_timestamp"], utc=True)

    # Extract numpy arrays
    avail_cols = [c for c in feature_cols if c in examples.columns]
    feature_matrix = examples[avail_cols].to_numpy(dtype=np.float32)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
    label_array = examples["close_label"].to_numpy(dtype=np.int8) if "close_label" in examples.columns else np.zeros(len(examples), dtype=np.int8)
    decision_ts = examples["decision_timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    decision_month_ord = examples["decision_timestamp"].dt.to_period("M").astype("int64").to_numpy(dtype=np.int64)
    label_end = (examples["label_end_timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
                 if "label_end_timestamp" in examples.columns
                 else examples.groupby("trade_id")["decision_timestamp"].transform("max").to_numpy(dtype="datetime64[ns]").astype(np.int64))

    # Compute positive class weight
    pos = int(label_array.sum())
    neg = int(len(label_array) - pos)
    scale_pos_weight = neg / max(pos, 1)

    # Build month -> test indices map
    test_start_period = pd.Period(test_start, freq="M")
    test_start_ord = test_start_period.ordinal
    unique_months = sorted(set(decision_month_ord))
    month_map = {}
    for m_ord in unique_months:
        if m_ord >= test_start_ord:
            test_mask = decision_month_ord == m_ord
            test_idx = np.flatnonzero(test_mask)
            if len(test_idx) > 0:
                train_start_ord = m_ord - lookback_months * 1  # approximate
                month_map[int(m_ord)] = {
                    "test_idx": test_idx,
                    "train_start_ord": int(max(unique_months[0], m_ord - lookback_months)),
                }

    n_months = len(month_map)
    print(f"    Dispatching {n_months} months to {max_workers} workers...")
    sys.stdout.flush()

    # Save shared arrays to temp files (memmap — no copy in workers)
    tmpdir = tempfile.mkdtemp(prefix="close_wf_")
    feat_path = os.path.join(tmpdir, "features.npy")
    label_path = os.path.join(tmpdir, "labels.npy")
    month_ord_path = os.path.join(tmpdir, "month_ords.npy")
    decision_ts_path = os.path.join(tmpdir, "decision_ts.npy")
    label_end_path = os.path.join(tmpdir, "label_end.npy")

    np.save(feat_path, feature_matrix)
    np.save(label_path, label_array)
    np.save(month_ord_path, decision_month_ord)
    np.save(decision_ts_path, decision_ts)
    np.save(label_end_path, label_end)

    # Dispatch months in parallel
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for m_ord, info in month_map.items():
            f = pool.submit(
                _train_predict_month,
                m_ord, info["test_idx"], info["train_start_ord"],
                feat_path, label_path, month_ord_path,
                decision_ts_path, label_end_path,
                feature_matrix.shape, len(examples),
                scale_pos_weight, use_gpu,
            )
            futures[f] = m_ord

        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result is not None:
                results.append(result)
            if completed % 10 == 0 or completed == n_months:
                print(f"    {completed}/{n_months} months done ({time.time()-t0:.0f}s)")
                sys.stdout.flush()

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Merge predictions back into examples
    examples["close_probability"] = 0.5  # default: no opinion

    for r in results:
        examples.iloc[r["test_idx"], examples.columns.get_loc("close_probability")] = r["close_probability"]

    elapsed = time.time() - t0
    total_preds = sum(len(r["test_idx"]) for r in results)
    print(f"    WF done: {total_preds:,} predictions from {len(results)} months in {elapsed:.0f}s")

    return examples


# Need sys for flush
import sys
