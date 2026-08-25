"""Detailed timing breakdown of walk_forward_close_prediction_cache."""
import sys
sys.path.insert(0, "src")

import time, os, numpy as np, pandas as pd, logging, glob, io
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

from signal_engine.close_model import (
    _choose_threshold_from_validation,
    _build_selection_payload,
    compute_afml_close_sample_weights_arrays,
    make_xgb_close_model_with_params,
    xgb_grid_candidates,
    _fit_isotonic_calibrator,
    _apply_calibrator,
    close_feature_columns,
    _prediction_output_columns,
    CloseModelConfig,
)
from signal_engine.config import PortfolioConfig, load_config
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

cfg = CloseModelConfig()
pcfg = PortfolioConfig()
real_cfg = load_config(Path("configs/default.yaml"))
feat_cols = real_cfg.features.feature_list

# Load real combined examples
examples_path = "experiments/cache/ml_close_walkforward/close_examples_combined"
scope_dirs = [d for d in os.listdir(examples_path) if os.path.isdir(os.path.join(examples_path, d))]
scope = scope_dirs[0]
files = sorted(glob.glob(f"{examples_path}/{scope}/lb*_residtb_15m_*.parquet"))
if not files:
    print("No 15min parquet found")
    sys.exit(1)

f = files[0]
print(f"Loading: {f}")
t0 = time.perf_counter()
examples = pd.read_parquet(f)
print(f"Loaded {len(examples)} rows in {time.perf_counter()-t0:.2f}s")

all_feature_cols = close_feature_columns(feat_cols, regime_timeframes=cfg.regime_timeframes)
print(f"Feature cols: {len(feat_cols)}, all (with regime+state): {len(all_feature_cols)}")

months = sorted(examples["entry_month"].dropna().astype(str).unique())
test_months = [m for m in months if m >= "2022-01"]
print(f"Total months: {len(months)}, test months: {len(test_months)}")

print()
print("=" * 70)
print("SECTION 1: DATA PREPARATION")
print("=" * 70)

t0 = time.perf_counter()
examples = examples.sort_values(["timestamp", "decision_timestamp"]).reset_index(drop=True)
print(f"  Sort: {time.perf_counter()-t0:.3f}s")

t0 = time.perf_counter()
for col in ["decision_timestamp", "label_end_timestamp", "expiry_timestamp"]:
    if col in examples.columns:
        examples[col] = pd.to_datetime(examples[col], utc=True, errors="coerce")
print(f"  Datetime conversions: {time.perf_counter()-t0:.3f}s")

t0 = time.perf_counter()
for col in all_feature_cols:
    if col not in examples.columns:
        examples[col] = 0.0
print(f"  Fill missing cols: {time.perf_counter()-t0:.3f}s")

t0 = time.perf_counter()
present = [c for c in all_feature_cols if c in examples.columns]
feature_matrix = examples[present].to_numpy(dtype=np.float32, copy=True)
print(f"  Feature matrix {feature_matrix.shape}: {time.perf_counter()-t0:.3f}s")

t0 = time.perf_counter()
label_array = examples["close_label"].to_numpy(dtype=np.int8)
decision_ts_arr = examples["decision_timestamp"].to_numpy(dtype="datetime64[ns]")
label_end_arr = examples["label_end_timestamp"].to_numpy(dtype="datetime64[ns]")
label_return_abs_arr = examples["label_return_abs"].to_numpy(dtype=np.float64)
print(f"  Array extractions: {time.perf_counter()-t0:.3f}s")

t0 = time.perf_counter()
entry_month_str = examples["entry_month"].astype(str)
entry_periods = pd.PeriodIndex(entry_month_str, freq="M")
entry_month_ord_arr = entry_periods.asi8.astype(np.int32)
dm_periods = pd.to_datetime(decision_ts_arr).to_period("M")
decision_month_ord_arr = pd.PeriodIndex(dm_periods, freq="M").asi8.astype(np.int32)
print(f"  Month ordinals: {time.perf_counter()-t0:.3f}s")

output_cols = _prediction_output_columns()

print()
print("=" * 70)
print("SECTION 2: PER-MONTH BREAKDOWN (first test month)")
print("=" * 70)
first_month = pd.Period(test_months[0])
month_ord = int(first_month.ordinal)
month_start = pd.Timestamp(first_month.start_time, tz="UTC")
train_start_ord = (first_month - 4 + 1).ordinal

t0 = time.perf_counter()
train_mask = (
    (decision_month_ord_arr >= train_start_ord)
    & (decision_month_ord_arr < month_ord)
    & (decision_ts_arr < month_start.to_datetime64())
    & (label_end_arr < month_start.to_datetime64())
)
train_idx = np.flatnonzero(train_mask)
test_idx = np.flatnonzero(entry_month_ord_arr == month_ord)
print(f"  Train/test mask: {time.perf_counter()-t0:.3f}s  (train={len(train_idx)}, test={len(test_idx)})")

unique_train_months = np.unique(decision_month_ord_arr[train_idx])
val_month_ord = int(unique_train_months.max())
val_start_ts = pd.Timestamp(pd.Period(ordinal=val_month_ord, freq="M").start_time, tz="UTC")
val_idx = train_idx[decision_month_ord_arr[train_idx] == val_month_ord]
fit_idx = train_idx[
    (decision_month_ord_arr[train_idx] < val_month_ord)
    & (label_end_arr[train_idx] < val_start_ts.to_datetime64())
]
print(f"  fit_idx={len(fit_idx)}, val_idx={len(val_idx)}")

t0 = time.perf_counter()
fit_weights = compute_afml_close_sample_weights_arrays(
    decision_ts_arr[fit_idx], label_end_arr[fit_idx],
    label_return_abs_arr[fit_idx], clf_last_w=cfg.time_decay_last_weight,
)
print(f"  Sample weights: {time.perf_counter()-t0:.3f}s")

t0 = time.perf_counter()
model = make_xgb_close_model_with_params(
    scale_pos_weight=1.0, params={},
    early_stopping_rounds=cfg.xgb_early_stopping_rounds, n_jobs=1,
)
model.fit(
    feature_matrix[fit_idx], label_array[fit_idx],
    sample_weight=fit_weights,
    eval_set=[(feature_matrix[val_idx], label_array[val_idx])],
    verbose=False,
)
t_fit = time.perf_counter() - t0
best_iter = getattr(model, "best_iteration", "?")
print(f"  Single XGB fit (n_jobs=1): {t_fit:.3f}s  (best_iteration={best_iter})")

t0 = time.perf_counter()
raw_val = model.predict_proba(feature_matrix[val_idx])[:, 1]
print(f"  predict_proba on val: {time.perf_counter()-t0:.3f}s")

t0 = time.perf_counter()
cal = _fit_isotonic_calibrator(raw_val, label_array[val_idx], cfg)
calibrated = _apply_calibrator(raw_val, cal)
print(f"  Isotonic calibration: {time.perf_counter()-t0:.3f}s")

t0 = time.perf_counter()
val_base = examples.iloc[val_idx][output_cols].copy()
val_base["target_month_ord"] = np.int32(month_ord)
val_base["close_probability"] = calibrated
print(f"  Build val_base DataFrame: {time.perf_counter()-t0:.3f}s")

print()
print("=" * 70)
print("SECTION 3: THRESHOLD SEARCH (the suspected bottleneck)")
print("=" * 70)

t0 = time.perf_counter()
td = _choose_threshold_from_validation(
    val_base, close_cfg=cfg, signal_bar_minutes=15, portfolio_cfg=pcfg,
)
print(f"  _choose_threshold_from_validation: {time.perf_counter()-t0:.3f}s")
print(f"    threshold={td['threshold']}, sharpe={td['validation_sharpe']:.3f}")

# Break down threshold search internals
t0 = time.perf_counter()
payload = _build_selection_payload(val_base)
print(f"  _build_selection_payload: {time.perf_counter()-t0:.3f}s")

print()
print("=" * 70)
print("SECTION 4: GRID SEARCH")
print("=" * 70)

grid_candidates = xgb_grid_candidates(cfg)
print(f"  Grid candidates: {len(grid_candidates)}")

# Per-candidate breakdown
params0 = grid_candidates[0]
t0 = time.perf_counter()
cand = make_xgb_close_model_with_params(
    scale_pos_weight=1.0, params=params0,
    early_stopping_rounds=cfg.xgb_early_stopping_rounds, n_jobs=1,
)
cand.fit(feature_matrix[fit_idx], label_array[fit_idx],
         sample_weight=fit_weights,
         eval_set=[(feature_matrix[val_idx], label_array[val_idx])],
         verbose=False)
t_cand_fit = time.perf_counter() - t0

t0 = time.perf_counter()
rv = cand.predict_proba(feature_matrix[val_idx])[:, 1]
t_cand_pred = time.perf_counter() - t0

t0 = time.perf_counter()
c = _fit_isotonic_calibrator(rv, label_array[val_idx], cfg)
vr = val_base.copy()
vr["close_probability"] = _apply_calibrator(rv, c)
t_cand_cal = time.perf_counter() - t0

t0 = time.perf_counter()
td2 = _choose_threshold_from_validation(vr, close_cfg=cfg, signal_bar_minutes=15, portfolio_cfg=pcfg)
t_cand_thresh = time.perf_counter() - t0

print(f"  Per candidate breakdown:")
print(f"    XGB fit:         {t_cand_fit:.3f}s")
print(f"    predict_proba:   {t_cand_pred:.3f}s")
print(f"    calibration:     {t_cand_cal:.3f}s")
print(f"    threshold search:{t_cand_thresh:.3f}s")
print(f"    TOTAL per cand:  {t_cand_fit+t_cand_pred+t_cand_cal+t_cand_thresh:.3f}s")
print(f"    x{len(grid_candidates)} sequential: {(t_cand_fit+t_cand_pred+t_cand_cal+t_cand_thresh)*len(grid_candidates):.1f}s")

# Full sequential
t0 = time.perf_counter()
for params in grid_candidates:
    cand = make_xgb_close_model_with_params(
        scale_pos_weight=1.0, params=params,
        early_stopping_rounds=cfg.xgb_early_stopping_rounds, n_jobs=1,
    )
    cand.fit(feature_matrix[fit_idx], label_array[fit_idx],
             sample_weight=fit_weights,
             eval_set=[(feature_matrix[val_idx], label_array[val_idx])],
             verbose=False)
    rv = cand.predict_proba(feature_matrix[val_idx])[:, 1]
    c = _fit_isotonic_calibrator(rv, label_array[val_idx], cfg)
    vr = val_base.copy()
    vr["close_probability"] = _apply_calibrator(rv, c)
    td2 = _choose_threshold_from_validation(vr, close_cfg=cfg, signal_bar_minutes=15, portfolio_cfg=pcfg)
t_seq = time.perf_counter() - t0
print(f"  Sequential grid: {t_seq:.3f}s ({t_seq/len(grid_candidates):.3f}s/cand)")

# Parallel
workers = min(len(grid_candidates), os.cpu_count() or 1)
def _eval(params):
    cand = make_xgb_close_model_with_params(
        scale_pos_weight=1.0, params=params,
        early_stopping_rounds=cfg.xgb_early_stopping_rounds, n_jobs=1,
    )
    cand.fit(feature_matrix[fit_idx], label_array[fit_idx],
             sample_weight=fit_weights,
             eval_set=[(feature_matrix[val_idx], label_array[val_idx])],
             verbose=False)
    rv = cand.predict_proba(feature_matrix[val_idx])[:, 1]
    c = _fit_isotonic_calibrator(rv, label_array[val_idx], cfg)
    vr = val_base.copy()
    vr["close_probability"] = _apply_calibrator(rv, c)
    td2 = _choose_threshold_from_validation(vr, close_cfg=cfg, signal_bar_minutes=15, portfolio_cfg=pcfg)
    return td2
t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=workers) as pool:
    list(pool.map(_eval, grid_candidates))
t_par = time.perf_counter() - t0
print(f"  Parallel grid ({workers}w): {t_par:.3f}s  (speedup: {t_seq/t_par:.1f}x)")

# Final model + weights
print()
print("=" * 70)
print("SECTION 5: FINAL MODEL + CACHE WRITE")
print("=" * 70)

t0 = time.perf_counter()
full_weights = compute_afml_close_sample_weights_arrays(
    decision_ts_arr[train_idx], label_end_arr[train_idx],
    label_return_abs_arr[train_idx], clf_last_w=cfg.time_decay_last_weight,
)
print(f"  Full weights: {time.perf_counter()-t0:.3f}s")

t0 = time.perf_counter()
final = make_xgb_close_model_with_params(scale_pos_weight=1.0, params={}, n_jobs=os.cpu_count())
final.fit(feature_matrix[train_idx], label_array[train_idx], sample_weight=full_weights)
print(f"  Final model fit (n_jobs={os.cpu_count()}): {time.perf_counter()-t0:.3f}s")

import joblib
t0 = time.perf_counter()
buf = io.BytesIO()
joblib.dump(final, buf, compress=3)
model_bytes = buf.getvalue()
print(f"  Model serialize: {time.perf_counter()-t0:.3f}s ({len(model_bytes)//1024}KB)")

print()
print("=" * 70)
print("SUMMARY: ESTIMATED PER-MONTH COST")
print("=" * 70)
per_month = t_par + 0.5 + 0.3 + 0.2  # parallel grid + weights + final + cache
print(f"  Parallel grid search: {t_par:.1f}s")
print(f"  Weights + final fit:  ~0.8s")
print(f"  Cache write:          ~0.2s")
print(f"  TOTAL per month:      ~{per_month:.1f}s")
print(f"  x{len(test_months)} months:          ~{len(test_months)*per_month:.0f}s = {len(test_months)*per_month/60:.1f}min")
