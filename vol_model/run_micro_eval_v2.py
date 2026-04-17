#!/usr/bin/env python
"""Aggressive pruning + Optuna eval for microstructure features."""

import json, logging, time
import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from scipy.stats import spearmanr
from pathlib import Path

from vol_model.config import VolModelConfig
from vol_model.model import qlike_objective, qlike_loss, qlike_metric
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward import OptunaPatience, _make_objective

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval")

config = VolModelConfig()
X, y, t1 = build_dataset(config)
print(f"Dataset: {len(X)} rows, {len(X.columns)} features")

# ── Aggressive pruning: drop zero-gain AND bottom 10% each round ──
prune_cutoff = pd.Timestamp("2023-10-01", tz="UTC")
mask = X.index < prune_cutoff
X_tr_full = X[mask]
y_tr_full = y[mask]

lo = X_tr_full.quantile(0.01)
hi = X_tr_full.quantile(0.99)
X_w = X_tr_full.clip(lo, hi, axis=1)

val_split = int(len(X_w) * 0.85)
y_tr = y_tr_full.iloc[:val_split].values
y_val = y_tr_full.iloc[val_split:].values

params = {
    "objective": qlike_objective,
    "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
    "min_data_in_leaf": 50, "colsample_bytree": 0.8, "subsample": 0.8,
    "device": "cpu", "num_threads": -1, "verbosity": -1, "seed": 42,
}

cols = list(X_w.columns)
no_change = 0

while no_change < 3 and len(cols) > 10:
    Xtr = X_w[cols].iloc[:val_split].values
    Xv = X_w[cols].iloc[val_split:].values

    dtrain = lgb.Dataset(Xtr, label=y_tr, feature_name=cols, free_raw_data=False)
    dval = lgb.Dataset(Xv, label=y_val, reference=dtrain, free_raw_data=False)

    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dval], feval=qlike_metric,
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])

    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(cols, importance), key=lambda x: -x[1])

    # Drop zero-gain
    nonzero = [(f, g) for f, g in feat_imp if g > 0]

    # Also drop bottom 10%
    if len(nonzero) > 10:
        cutoff_idx = max(int(len(nonzero) * 0.9), 10)
        surviving = [f for f, g in nonzero[:cutoff_idx]]
    else:
        surviving = [f for f, g in nonzero]

    dropped = len(cols) - len(surviving)
    logger.info("  Prune round: %d -> %d features (dropped %d)", len(cols), len(surviving), dropped)

    if dropped == 0:
        no_change += 1
    else:
        no_change = 0
        cols = surviving

pruned = cols
logger.info("Final pruned set: %d features", len(pruned))
for i, f in enumerate(pruned):
    marker = ""
    if "ofi" in f or "depth" in f:
        marker = " <-- OFI/DEPTH"
    elif "vpin" in f:
        marker = " <-- VPIN"
    print(f"  {i+1:3d}. {f}{marker}")

MODEL_DIR = Path("vol_model/output/models/expanding_microstructure")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / "pruned_features_v2.json").write_text(json.dumps(pruned))

# ── Train 3 months with Optuna ──
TEST_MONTHS = ["2023-10", "2024-06", "2025-06"]
EXISTING_DIR = Path("vol_model/output/models/expanding")

results = []
for month_str in TEST_MONTHS:
    M = pd.Timestamp(month_str + "-01", tz="UTC")
    train_mask = X.index < M
    test_end = M + pd.DateOffset(months=1)
    test_mask = (X.index >= M) & (X.index < test_end)

    X_used = X[pruned]
    X_train = X_used[train_mask]; y_train = y[train_mask]
    X_test = X_used[test_mask]; y_test = y[test_mask]

    lo = X_train.quantile(0.01); hi = X_train.quantile(0.99)
    X_train_w = X_train.clip(lo, hi, axis=1)
    X_test_w = X_test.clip(lo, hi, axis=1)

    val_split = int(len(X_train_w) * 0.85)
    X_tr = X_train_w.iloc[:val_split].values; y_tr = y_train.iloc[:val_split].values
    X_val = X_train_w.iloc[val_split:].values; y_val = y_train.iloc[val_split:].values

    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False, params={"feature_pre_filter": False})
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
    dtrain.construct(); dval.construct()

    logger.info("%s: Optuna (500 trials, patience 100)...", month_str)
    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42),
                                pruner=MedianPruner(n_startup_trials=5))
    obj = _make_objective(dtrain, dval, X_val, y_val, "cpu")
    study.optimize(obj, n_trials=500, callbacks=[OptunaPatience(100)], show_progress_bar=False)
    logger.info("%s: %d trials in %.0fs", month_str, len(study.trials), time.time() - t0)

    best = study.best_params.copy()
    best["objective"] = qlike_objective
    best["subsample"] = 0.8
    best["device"] = "cpu"
    best["num_threads"] = -1
    best["verbosity"] = -1
    best["seed"] = 42

    dtrain_full = lgb.Dataset(X_train_w.values, label=y_train.values, free_raw_data=False)
    model = lgb.train(best, dtrain_full, num_boost_round=2000,
                      valid_sets=[dval], feval=qlike_metric,
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

    preds = model.predict(X_test_w.values)
    y_arr = y_test.values
    q = qlike_loss(y_arr, preds)
    rmse = float(np.sqrt(np.mean((y_arr - preds) ** 2)))
    ic = float(spearmanr(y_arr, preds).statistic)
    model.save_model(str(MODEL_DIR / f"{month_str}_optuna.txt"))

    # Baseline
    old_path = EXISTING_DIR / f"{month_str}.txt"
    old_q, old_rmse, old_ic = None, None, None
    if old_path.exists():
        old_feats = pd.read_csv("vol_model/output/pruned_features.csv", header=None)[0].tolist()
        X_old = X[old_feats]
        old_model = lgb.Booster(model_file=str(old_path))
        lo2 = X_old[train_mask].quantile(0.01)
        hi2 = X_old[train_mask].quantile(0.99)
        old_preds = old_model.predict(X_old[test_mask].clip(lo2, hi2, axis=1).values)
        old_q = qlike_loss(y_arr, old_preds)
        old_rmse = float(np.sqrt(np.mean((y_arr - old_preds) ** 2)))
        old_ic = float(spearmanr(y_arr, old_preds).statistic)

    logger.info("%s: NEW q=%.4f ic=%.3f | OLD q=%.4f ic=%.3f",
                month_str, q, ic, old_q or 0, old_ic or 0)
    results.append((month_str, q, rmse, ic, old_q, old_rmse, old_ic, len(study.trials)))

# ── Comparison ──
print(f"\n{'=' * 70}")
print(f"  COMPARISON (Optuna-tuned, aggressive pruning)")
print(f"{'=' * 70}")
header = f"  {'Month':>8s}  {'Model':>16s}  {'QLIKE':>8s}  {'RMSE':>8s}  {'RankIC':>8s}  {'Feats':>6s}"
print(header)
print("-" * 70)

for ms, q, rmse, ic, oq, ormse, oic, nt in results:
    print(f"  {ms:>8s}  {'+ microstructure':>16s}  {q:>8.4f}  {rmse:>8.4f}  {ic:>8.3f}  {len(pruned):>6d}")
    if oq is not None:
        print(f"  {'':>8s}  {'baseline':>16s}  {oq:>8.4f}  {ormse:>8.4f}  {oic:>8.3f}  {35:>6d}")
        dq = q - oq
        better = "BETTER" if dq < 0 else "WORSE"
        print(f"  {'':>8s}  {'delta':>16s}  {dq:>+8.4f}  {'':>8s}  {ic - oic:>+8.3f}  {better}")
    print()

avg_new = np.mean([r[1] for r in results])
avg_old = np.mean([r[4] for r in results if r[4] is not None])
avg_new_ic = np.mean([r[3] for r in results])
avg_old_ic = np.mean([r[6] for r in results if r[6] is not None])
print(f"  Average QLIKE:  new={avg_new:.4f}  old={avg_old:.4f}  delta={avg_new - avg_old:+.4f}")
print(f"  Average RankIC: new={avg_new_ic:.3f}  old={avg_old_ic:.3f}  delta={avg_new_ic - avg_old_ic:+.3f}")
