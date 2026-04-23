#!/usr/bin/env python
"""Walk-forward rolling retrain with Optuna + trading performance comparison.

Optimizations:
- Feature pruning (--pruned): 116 -> 34 features, ~3x faster per trial
- Smart Optuna stopping (--patience): stop after N no-improve trials
- Dataset caching: LGB datasets built once per month, reused across trials
- Dual CPU+GPU workers (--parallel): concurrent Optuna on both devices

Usage:
    PYTHONPATH=. python vol_model/run_walkforward.py --lookbacks exp --save-best
    PYTHONPATH=. python vol_model/run_walkforward.py --lookbacks exp --pruned --patience 50 --parallel --save-best
    PYTHONPATH=. python vol_model/run_walkforward.py --lookbacks 10,12,exp --no-optuna
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from scipy.stats import spearmanr

from vol_model.config import VolModelConfig
from vol_model.model import qlike_objective, qlike_loss, qlike_metric
from vol_model.pipeline import build_dataset

logger = logging.getLogger("walkforward")


def _setup_logging():
    """Configure logging. Called once from main(), not on import."""
    _log_path = Path("vol_model/output/walkforward.log")
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(_log_path), mode="w"),
        ],
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("Log file: %s", _log_path)


# ── Optuna early stopping callback ───────────────────────────

class OptunaPatience:
    """Stop Optuna study after `patience` trials with no improvement."""

    def __init__(self, patience: int = 50):
        self.patience = patience
        self.best = float("inf")
        self.no_improve = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value < self.best:
            self.best = trial.value
            self.no_improve = 0
        else:
            self.no_improve += 1
        if self.no_improve >= self.patience:
            study.stop()


# ── Per-month feature pruning ───────────────────────────────

def prune_features_for_month(
    X_train: pd.DataFrame, y_train: pd.Series, patience: int = 2,
) -> list[str]:
    """Iteratively train and drop zero-importance features until stable.

    Each round: train a fast LightGBM, drop features with zero gain,
    retrain on the smaller set. Stops after `patience` consecutive
    rounds with no features removed (typically converges in 2-3 rounds).
    """
    lo = X_train.quantile(0.01)
    hi = X_train.quantile(0.99)
    X_w = X_train.clip(lo, hi, axis=1)

    val_split = int(len(X_w) * 0.85)
    y_tr = y_train.iloc[:val_split].values
    y_val = y_train.iloc[val_split:].values

    params = {
        "objective": qlike_objective,
        "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
        "min_data_in_leaf": 50, "colsample_bytree": 0.8, "subsample": 0.8,
        "device": "cpu", "num_threads": -1, "verbosity": -1, "seed": 42,
    }

    cols = list(X_w.columns)
    no_change = 0
    round_num = 0

    while no_change < patience:
        round_num += 1
        X_tr = X_w[cols].iloc[:val_split].values
        X_v = X_w[cols].iloc[val_split:].values

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=cols, free_raw_data=False)
        dval = lgb.Dataset(X_v, label=y_val, reference=dtrain, free_raw_data=False)

        model = lgb.train(
            params, dtrain, num_boost_round=200,
            valid_sets=[dval], feval=qlike_metric,
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )

        importance = model.feature_importance(importance_type="gain")
        surviving = [c for c, imp in zip(cols, importance) if imp > 0]

        if len(surviving) == len(cols):
            no_change += 1
        else:
            no_change = 0
            cols = surviving

        if len(cols) == 0:
            break

    return cols


def _save_month_features(model_dir: Path, month_str: str, features: list[str]) -> None:
    """Save per-month feature list alongside model file."""
    (model_dir / f"{month_str}_features.json").write_text(json.dumps(features))


def _load_month_features(model_dir: Path, month_str: str) -> list[str] | None:
    """Load per-month feature list. Returns None if not found."""
    path = model_dir / f"{month_str}_features.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ── Optuna objective with cached datasets ────────────────────

def _make_objective(dtrain: lgb.Dataset, dval: lgb.Dataset,
                    X_val: np.ndarray, y_val: np.ndarray, device: str,
                    *, horizon_minutes: int = 15):
    """Create an Optuna objective closure over pre-built LGB datasets.

    ``horizon_minutes`` scales the ``min_data_in_leaf`` search range so the
    leaf-size bounds in WALL-CLOCK terms stay roughly constant across
    bar frequencies. At h=15 (legacy default) the range is the original
    [50, 500] — byte-identical. At h=60 it becomes [200, 2000]
    (~50–500 hours of leaf data); at h=5 [16, 166] (~1.4–13.8 hours).
    Bar-count granularity is what LGBM actually optimises over, but
    leaf-time governs how generalisable each leaf is — keeping the latter
    constant prevents the search from collapsing to either degenerate
    over-fitting (h=5 with [50, 500] = trivial leaves) or
    under-resolution (h=60 with [50, 500] = monstrously coarse leaves).
    See plan starry-rolling-rabin.md Phase 1e and
    feedback_gauntlet_budget.md (TPE convergence is bar-count-agnostic;
    don't crank trials).
    """
    scale = horizon_minutes / 15.0
    leaf_lo = max(10, int(round(50 * scale)))
    leaf_hi = max(leaf_lo + 1, int(round(500 * scale)))

    def objective(trial):
        params = {
            "objective": qlike_objective,
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", leaf_lo, leaf_hi),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "subsample": 0.8,
            "device": device,
            "num_threads": -1 if device == "cpu" else 1,
            "verbosity": -1,
            "seed": 42,
        }

        model = lgb.train(
            params, dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            feval=qlike_metric,
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        preds = model.predict(X_val)
        loss = qlike_loss(y_val, preds)

        # Explicit release of the per-trial LightGBM booster. Without
        # this, CPython's reference counter holds the booster alive across
        # trials long enough for a 400K+ row training set at h=5m to
        # exhaust system memory around trial ~9-15. Seen in Phase 2h5.3:
        # 0.4 GB booster × 10 trials = 4 GB leaked before MemoryError at
        # a later 3 MiB allocation inside qlike_objective. gc.collect()
        # is cheap (~5 ms) and one call per trial is safe.
        del model, preds
        import gc
        gc.collect()
        return loss

    return objective


# ── Core: optimize + retrain for one month ───────────────────

def optimize_and_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    max_trials: int,
    patience: int,
    device: str = "cpu",
) -> tuple[np.ndarray, dict, object]:
    """Optuna-optimize hyperparams on one device, retrain with best, predict.

    Each worker calls this independently for its own month.
    device='cpu' uses all cores; device='gpu' uses 1 CPU thread + OpenCL GPU.
    Returns (predictions, best_params, model).
    """
    # Winsorize
    lo = X_train.quantile(0.01)
    hi = X_train.quantile(0.99)
    X_train_w = X_train.clip(lo, hi, axis=1)
    X_test_w = X_test.clip(lo, hi, axis=1)

    # 85/15 train/val split
    val_split = int(len(X_train_w) * 0.85)
    X_tr = X_train_w.iloc[:val_split].values
    y_tr = y_train.iloc[:val_split].values
    X_val = X_train_w.iloc[val_split:].values
    y_val = y_train.iloc[val_split:].values

    # Cache LGB datasets (built once, reused across all trials)
    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False,
                         params={"feature_pre_filter": False})
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
    dtrain.construct()
    dval.construct()

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    obj = _make_objective(dtrain, dval, X_val, y_val, device)
    study.optimize(obj, n_trials=max_trials, callbacks=[OptunaPatience(patience)],
                   show_progress_bar=False)

    # Retrain final model on full train set with best params (always CPU, all cores)
    best = study.best_params.copy()
    best["objective"] = qlike_objective
    best["subsample"] = 0.8
    best["device"] = "cpu"
    best["num_threads"] = -1
    best["verbosity"] = -1
    best["seed"] = 42

    dtrain_full = lgb.Dataset(X_train_w.values, label=y_train.values, free_raw_data=False)
    model = lgb.train(
        best, dtrain_full,
        num_boost_round=2000,
        valid_sets=[dval],
        feval=qlike_metric,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    preds = model.predict(X_test_w.values)
    best["_n_trials"] = len(study.trials)
    return preds, best, model


# ── Default params (no Optuna) ───────────────────────────────

def _train_default_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    config: VolModelConfig,
) -> tuple[np.ndarray, object]:
    """Train with default hyperparams. Returns (predictions, model)."""
    from vol_model.model import build_lgbm_params

    lo = X_train.quantile(config.winsorize_lo)
    hi = X_train.quantile(config.winsorize_hi)
    X_train_w = X_train.clip(lo, hi, axis=1)
    X_test_w = X_test.clip(lo, hi, axis=1)

    val_split = int(len(X_train_w) * 0.85)
    dtrain = lgb.Dataset(X_train_w.iloc[:val_split], label=y_train.iloc[:val_split].values, free_raw_data=False)
    dval = lgb.Dataset(X_train_w.iloc[val_split:], label=y_train.iloc[val_split:].values, reference=dtrain, free_raw_data=False)

    params = build_lgbm_params(config)
    model = lgb.train(
        params, dtrain,
        num_boost_round=config.n_estimators,
        valid_sets=[dval],
        feval=qlike_metric,
        callbacks=[lgb.early_stopping(config.early_stopping_rounds), lgb.log_evaluation(0)],
    )
    return model.predict(X_test_w), model


# ── Walk-forward loop ────────────────────────────────────────

def _process_month(
    month_idx: int,
    M: pd.Timestamp,
    X: pd.DataFrame,
    y: pd.Series,
    lookback_months: int,
    max_trials: int,
    patience: int,
    config: VolModelConfig,
    use_optuna: bool,
    device: str,
    save_models: bool,
    model_dir: Path | None,
    total_months: int,
    per_month_prune: bool = False,
) -> dict | None:
    """Process a single month. Called by each worker independently."""
    data_start = X.index.min()
    expanding = (lookback_months == -1)

    if expanding:
        train_start = data_start
    else:
        train_start = M - pd.DateOffset(months=lookback_months)
    test_end = M + pd.DateOffset(months=1)

    train_mask = (X.index >= train_start) & (X.index < M)
    test_mask = (X.index >= M) & (X.index < test_end)

    n_train = train_mask.sum()
    n_test = test_mask.sum()

    if n_train < 100 or n_test == 0:
        return None

    month_str = M.strftime("%Y-%m")
    model_path = model_dir / f"{month_str}.txt" if model_dir else None
    full_n_features = len(X.columns)
    n_features_used = full_n_features
    features_pruned = 0

    # Resume: skip retraining if cached model exists
    if model_path and model_path.exists():
        cached_features = _load_month_features(model_dir, month_str) if model_dir else None
        if cached_features is None and per_month_prune:
            # Old cache without feature metadata — incompatible with per-month pruning
            logger.info("  [%d/%d] %s: re-training (no feature metadata in cache)",
                        month_idx + 1, total_months, month_str)
        else:
            # Use cached model
            if cached_features is not None:
                X_month = X[cached_features]
            else:
                X_month = X
            model = lgb.Booster(model_file=str(model_path))
            lo = X_month[train_mask].quantile(0.01)
            hi = X_month[train_mask].quantile(0.99)
            preds = model.predict(X_month[test_mask].clip(lo, hi, axis=1).values)
            n_features_used = len(X_month.columns)

            test_idx = np.where(test_mask)[0]
            y_test_arr = y.values[test_idx]
            fold_qlike = qlike_loss(y_test_arr, preds)
            fold_rmse = float(np.sqrt(np.mean((y_test_arr - preds) ** 2)))
            rank_ic = float(spearmanr(y_test_arr, preds).statistic)
            logger.info(
                "  [%d/%d] %s: QLIKE=%.4f RankIC=%.3f (train=%d, cached, feats=%d)",
                month_idx + 1, total_months, month_str,
                fold_qlike, rank_ic, n_train, n_features_used,
            )
            return {
                "month": month_str, "M": M, "n_train": n_train, "n_test": n_test,
                "qlike": fold_qlike, "rmse": fold_rmse, "rank_ic": rank_ic,
                "n_trials": -1, "test_idx": test_idx, "preds": preds,
                "n_features": n_features_used, "features_pruned": 0,
            }

    # Per-month feature pruning
    if per_month_prune:
        try:
            pruned_cols = prune_features_for_month(X[train_mask], y[train_mask])
            features_pruned = full_n_features - len(pruned_cols)
            if len(pruned_cols) == 0:
                logger.warning("  [%d/%d] %s: all features pruned, using full set",
                               month_idx + 1, total_months, month_str)
                X_month = X
            else:
                X_month = X[pruned_cols]
        except Exception as e:
            logger.warning("  [%d/%d] %s: pruning failed (%s), using full set",
                           month_idx + 1, total_months, month_str, e)
            X_month = X
            features_pruned = 0
    else:
        X_month = X

    n_features_used = len(X_month.columns)

    # Train
    if use_optuna:
        preds, best_params, model = optimize_and_train(
            X_month[train_mask], y[train_mask], X_month[test_mask],
            max_trials, patience, device,
        )
        n_trials_run = best_params.get("_n_trials", 0)
    else:
        preds, model = _train_default_params(
            X_month[train_mask], y[train_mask], X_month[test_mask], config,
        )
        n_trials_run = 0

    if save_models and model_dir:
        model.save_model(str(model_path))
        _save_month_features(model_dir, month_str, list(X_month.columns))

    test_idx = np.where(test_mask)[0]
    y_test_arr = y.values[test_idx]
    fold_qlike = qlike_loss(y_test_arr, preds)
    fold_rmse = float(np.sqrt(np.mean((y_test_arr - preds) ** 2)))
    rank_ic = float(spearmanr(y_test_arr, preds).statistic)

    status = f"trials={n_trials_run}, {device}"
    if per_month_prune:
        status += f", feats={n_features_used}/{full_n_features}"
    logger.info(
        "  [%d/%d] %s: QLIKE=%.4f RankIC=%.3f (train=%d, %s)",
        month_idx + 1, total_months, month_str,
        fold_qlike, rank_ic, n_train, status,
    )

    return {
        "month": month_str, "M": M, "n_train": n_train, "n_test": n_test,
        "qlike": fold_qlike, "rmse": fold_rmse, "rank_ic": rank_ic,
        "n_trials": n_trials_run, "test_idx": test_idx, "preds": preds,
        "n_features": n_features_used, "features_pruned": features_pruned,
    }


def train_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    lookback_months: int,
    max_trials: int,
    patience: int,
    config: VolModelConfig,
    test_start: pd.Timestamp | None = None,
    use_optuna: bool = True,
    gpu_workers: int = 0,
    save_models: bool = False,
    model_dir: Path | None = None,
    per_month_prune: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Monthly walk-forward. 1 CPU worker + N GPU workers, each on its own month."""
    oof_preds = np.full(len(y), np.nan)

    if save_models and model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)

    data_start = X.index.min()
    data_end = X.index.max()

    earliest = (data_start + pd.DateOffset(months=lookback_months)).replace(day=1)
    if test_start is not None and test_start > earliest:
        earliest = test_start

    test_months = pd.date_range(start=earliest, end=data_end, freq="MS")
    total = len(test_months)

    # Shared args for each month
    common = dict(
        X=X, y=y, lookback_months=lookback_months,
        max_trials=max_trials, patience=patience, config=config,
        use_optuna=use_optuna, save_models=save_models,
        model_dir=model_dir, total_months=total,
        per_month_prune=per_month_prune,
    )

    n_total_workers = 1 + gpu_workers
    if n_total_workers == 1:
        # Sequential: CPU only
        results = []
        for i, M in enumerate(test_months):
            r = _process_month(i, M, device="cpu", **common)
            if r:
                results.append(r)
    else:
        # Parallel: CPU works front-to-back (small months first),
        # GPU workers work back-to-front (large months first, where GPU has parity).
        from collections import deque
        import threading

        cpu_queue = deque()  # front-to-back
        gpu_queue = deque()  # back-to-front

        months_list = list(enumerate(test_months))
        mid = len(months_list) - gpu_workers  # GPU gets the tail
        if mid < 1:
            mid = 1
        for item in months_list[:mid]:
            cpu_queue.append(item)
        for item in reversed(months_list[mid:]):
            gpu_queue.append(item)

        results = []
        results_lock = threading.Lock()
        # When one side finishes, steal from the other
        steal_lock = threading.Lock()

        def _get_work(primary: deque, secondary: deque):
            with steal_lock:
                if primary:
                    return primary.popleft()
                if secondary:
                    return secondary.popleft()
            return None

        def _worker(device: str, primary: deque, secondary: deque):
            while True:
                item = _get_work(primary, secondary)
                if item is None:
                    break
                i, M = item
                r = _process_month(i, M, device=device, **common)
                if r:
                    with results_lock:
                        results.append(r)

        logger.info("Queue split: CPU=%d months (front), GPU=%d months (back)",
                    len(cpu_queue), len(gpu_queue))

        with ThreadPoolExecutor(max_workers=n_total_workers) as pool:
            futures = [pool.submit(_worker, "cpu", cpu_queue, gpu_queue)]
            futures += [pool.submit(_worker, "gpu", gpu_queue, cpu_queue)
                        for _ in range(gpu_workers)]
            for f in futures:
                f.result()

    # Assemble results in chronological order
    results.sort(key=lambda r: r["M"])
    fold_metrics = []
    for r in results:
        oof_preds[r["test_idx"]] = r["preds"]
        fold_metrics.append({k: v for k, v in r.items() if k not in ("test_idx", "preds", "M")})

    return oof_preds, fold_metrics


# ── Trading performance ──────────────────────────────────────

def run_backtest_for_predictions(predictions_path: str, threshold: float = 40.0) -> dict | None:
    """Run vol-arb backtester on a predictions file."""
    try:
        from vol_arb.backtester import BacktestConfig, VolArbBacktester
    except ImportError:
        return None

    if not Path("vol_arb/output/BTC_vol_arb.parquet").exists():
        return None

    cfg = BacktestConfig(predictions_path=predictions_path, entry_threshold=threshold)
    bt = VolArbBacktester(cfg)

    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    try:
        metrics, df = bt.run_backtest()
    except Exception as e:
        logging.disable(logging.NOTSET)
        return {"error": str(e)}
    logging.disable(logging.NOTSET)

    active = df[df["signal"] != 0]
    if len(active) > 0:
        monthly_pnl = active["net_pnl"].resample("ME").sum()
        metrics["monthly_pnl"] = {d.strftime("%Y-%m"): float(v) for d, v in monthly_pnl.items()}
    return metrics


# ── Summarize ────────────────────────────────────────────────

def summarize(oof_preds, y, fold_metrics, lookback):
    valid = ~np.isnan(oof_preds)
    n_valid = valid.sum()
    if n_valid == 0:
        return {"lookback": lookback, "n_months": 0}

    y_v = y.values[valid]
    p_v = oof_preds[valid]
    monthly_qlikes = [m["qlike"] for m in fold_metrics]
    avg_trials = np.mean([m.get("n_trials", 0) for m in fold_metrics])

    feat_counts = [m["n_features"] for m in fold_metrics if "n_features" in m]
    return {
        "lookback": lookback,
        "n_months": len(fold_metrics),
        "n_rows": int(n_valid),
        "qlike": qlike_loss(y_v, p_v),
        "rmse": float(np.sqrt(np.mean((y_v - p_v) ** 2))),
        "rank_ic": float(spearmanr(y_v, p_v).statistic),
        "qlike_std": float(np.std(monthly_qlikes)),
        "avg_trials": float(avg_trials),
        "feat_min": int(min(feat_counts)) if feat_counts else 0,
        "feat_max": int(max(feat_counts)) if feat_counts else 0,
        "feat_mean": float(np.mean(feat_counts)) if feat_counts else 0,
    }


# ── Main ─────────────────────────────────────────────────────

def main():
    _setup_logging()
    parser = argparse.ArgumentParser(description="Walk-forward with Optuna + trading")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--tf", type=int, default=900)
    parser.add_argument("--lookbacks", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--no-optuna", action="store_true")
    parser.add_argument("--max-trials", type=int, default=500,
                        help="Max Optuna trials ceiling (default 500)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Stop Optuna after N no-improve trials (default 50)")
    parser.add_argument("--full-features", action="store_true",
                        help="Use all 116 features instead of default pruned 34")
    parser.add_argument("--per-month-prune", action="store_true",
                        help="Re-prune from full features each month (adaptive)")
    parser.add_argument("--gpu-workers", type=int, default=0,
                        help="Number of GPU Optuna workers (default 0, CPU-only)")
    parser.add_argument("--threshold", type=float, default=40.0)
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    config = VolModelConfig(asset=args.asset, freq_s=args.tf)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_lbs = args.lookbacks.split(",")
    lookbacks = []
    for lb in raw_lbs:
        lb = lb.strip()
        if lb.lower() in ("exp", "expanding"):
            lookbacks.append(-1)
        else:
            lookbacks.append(int(lb))
    positive_lbs = [lb for lb in lookbacks if lb > 0]
    max_lb = max(positive_lbs) if positive_lbs else 12

    optuna_desc = "OFF" if args.no_optuna else f"max={args.max_trials}, patience={args.patience}"
    if args.per_month_prune:
        feat_desc = "per-month prune (adaptive)"
    elif args.full_features:
        feat_desc = "full (all)"
    else:
        feat_desc = "pruned (static 34)"
    print("=" * 70)
    print(f"  WALK-FORWARD -- {args.asset}")
    print(f"  Lookbacks: {[('exp' if lb==-1 else lb) for lb in lookbacks]}")
    print(f"  Optuna: {optuna_desc}")
    print(f"  Features: {feat_desc}")
    print(f"  Workers: 1 CPU + {args.gpu_workers} GPU")
    print("=" * 70)

    t0 = time.time()
    print("\n[1] Loading dataset...")
    X, y, t1 = build_dataset(config)

    # Feature pruning
    if args.per_month_prune:
        # Pass full features to walk-forward; each month prunes its own subset
        print(f"  Per-month pruning: enabled (from {len(X.columns)} features)")
    elif not args.full_features:
        pruned_path = Path(config.output_dir) / "pruned_features.csv"
        if pruned_path.exists():
            pruned = pd.read_csv(pruned_path, header=None)[0].tolist()
            pruned_cols = [c for c in pruned if c in X.columns]
            original_n = len(X.columns)
            X = X[pruned_cols]
            print(f"  Pruned: {original_n} -> {len(pruned_cols)} features")
        else:
            print(f"  WARNING: {pruned_path} not found, using full features")

    print(f"  {len(X):,} rows, {len(X.columns)} features")
    print(f"  {X.index.min().strftime('%Y-%m-%d')} to {X.index.max().strftime('%Y-%m-%d')}")

    test_start = (X.index.min() + pd.DateOffset(months=max_lb)).replace(day=1)
    test_end = X.index.max()
    n_test_months = len(pd.date_range(start=test_start, end=test_end, freq="MS"))
    print(f"  Test period: {test_start.strftime('%Y-%m')} to "
          f"{test_end.strftime('%Y-%m')} ({n_test_months} months)")

    # ── Sweep ──
    results = []
    all_preds = {}

    for lb in lookbacks:
        lb_label = "expanding" if lb == -1 else f"{lb}m"
        print(f"\n{'=' * 70}")
        print(f"  LOOKBACK = {lb_label}")
        print(f"{'=' * 70}")

        t_lb = time.time()
        tf_suffix = f"_{args.tf // 60}m" if args.tf != 900 else ""
        model_dir = out / "models" / f"{lb_label}{tf_suffix}"
        oof, fold_metrics = train_walkforward(
            X, y, lb, args.max_trials, args.patience, config,
            test_start=test_start,
            use_optuna=not args.no_optuna,
            gpu_workers=args.gpu_workers,
            save_models=True, model_dir=model_dir,
            per_month_prune=args.per_month_prune,
        )
        elapsed = time.time() - t_lb

        summary = summarize(oof, y, fold_metrics, lb)
        summary["runtime_s"] = elapsed
        results.append(summary)
        all_preds[lb] = oof

        oof_df = pd.DataFrame({"y_true": y.values, "y_pred": oof}, index=y.index)
        oof_path = out / f"{args.asset}_wf_{lb_label}_predictions.parquet"
        oof_df.to_parquet(oof_path)

        if summary["n_months"] > 0:
            trials_str = f"  avg_trials={summary['avg_trials']:.0f}" if not args.no_optuna else ""
            feat_str = (f"  feats={summary['feat_min']}-{summary['feat_max']}"
                        if args.per_month_prune and summary.get("feat_min") else "")
            print(f"  QLIKE={summary['qlike']:.4f}  RMSE={summary['rmse']:.4f}  "
                  f"RankIC={summary['rank_ic']:.3f}{trials_str}{feat_str}  ({elapsed:.0f}s)")

    # ── Model comparison table ──
    print(f"\n{'=' * 70}")
    print(f"  MODEL QUALITY COMPARISON")
    print(f"{'=' * 70}")
    hdr = f"{'LB':>4} {'Months':>7} {'QLIKE':>8} {'RMSE':>8} {'RankIC':>8} {'Q_std':>8}"
    if not args.no_optuna:
        hdr += f" {'Trials':>7}"
    hdr += f" {'Time':>7}"
    print(hdr)
    print("-" * len(hdr))

    best_lb = None
    best_qlike = float("inf")
    for r in results:
        if r["n_months"] == 0:
            continue
        marker = ""
        if r["qlike"] < best_qlike:
            best_qlike = r["qlike"]
            best_lb = r["lookback"]
            marker = " *"
        lb_str = "exp" if r["lookback"] == -1 else f"{r['lookback']}"
        row = (f"{lb_str:>4} {r['n_months']:>7} "
               f"{r['qlike']:>8.4f} {r['rmse']:>8.4f} {r['rank_ic']:>8.3f} "
               f"{r['qlike_std']:>8.4f}")
        if not args.no_optuna:
            row += f" {r['avg_trials']:>7.0f}"
        row += f" {r['runtime_s']:>6.0f}s{marker}"
        print(row)

    # ── Trading comparison ──
    print(f"\n{'=' * 70}")
    print(f"  TRADING PERFORMANCE (threshold={args.threshold})")
    print(f"{'=' * 70}")
    print(f"{'LB':>4} {'Trades':>7} {'PnL':>12} {'Return':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'PF':>6} {'WinRate':>8}")
    print("-" * 70)

    trading_results = []
    for lb in lookbacks:
        lb_label = "expanding" if lb == -1 else f"{lb}m"
        pred_path = str(out / f"{args.asset}_wf_{lb_label}_predictions.parquet")
        metrics = run_backtest_for_predictions(pred_path, args.threshold)
        lb_str = "exp" if lb == -1 else f"{lb}"
        if metrics is None or "error" in metrics or metrics.get("n_trades", 0) == 0:
            print(f"{lb_str:>4}  -- no trades or backtest unavailable --")
            trading_results.append({"lookback": lb})
            continue

        trading_results.append({"lookback": lb, **metrics})
        print(f"{lb_str:>4} {metrics['n_trades']:>7,} "
              f"${metrics['total_pnl']:>10,.0f} "
              f"{metrics['total_return_pct']:>7.1f}% "
              f"{metrics['sharpe_ratio']:>8.2f} "
              f"{metrics['max_drawdown_pct']:>7.1f}% "
              f"{metrics['profit_factor']:>6.2f} "
              f"{metrics['win_rate_pct']:>7.1f}%")

    # ── Monthly PnL comparison ──
    monthly_data = {}
    for tr in trading_results:
        lb = tr["lookback"]
        mp = tr.get("monthly_pnl", {})
        if mp:
            monthly_data[lb] = mp

    if len(monthly_data) >= 2:
        all_months_set = sorted(set().union(*monthly_data.values()))
        if all_months_set:
            print(f"\n{'=' * 70}")
            print(f"  MONTHLY PNL BY LOOKBACK")
            print(f"{'=' * 70}")
            header = f"{'Month':>8}"
            for lb in sorted(monthly_data.keys()):
                lbl = "  exp" if lb == -1 else f"{lb:>5}m"
                header += f" {lbl:>6}"
            print(header)
            print("-" * (8 + 7 * len(monthly_data)))
            for m in all_months_set:
                row = f"{m:>8}"
                for lb in sorted(monthly_data.keys()):
                    val = monthly_data[lb].get(m, 0)
                    row += f" {val:>+6.0f}"
                print(row)

    # ── Save best ──
    if best_lb is not None:
        best_lbl = "expanding" if best_lb == -1 else f"{best_lb} months"
        print(f"\n  Best model (QLIKE): {best_lbl}")
    if args.save_best and best_lb is not None:
        best_fn = "expanding" if best_lb == -1 else f"{best_lb}m"
        src = out / f"{args.asset}_wf_{best_fn}_predictions.parquet"
        dst = out / f"{args.asset}_oof_predictions.parquet"
        shutil.copy2(src, dst)
        print(f"  Saved {src.name} -> {dst.name}")

    total = time.time() - t0
    print(f"\n  Total runtime: {total:.0f}s ({total/60:.1f}min)")


if __name__ == "__main__":
    main()
