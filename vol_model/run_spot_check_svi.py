#!/usr/bin/env python
"""Spot-check: train 3 hybrid+SVI models and compare to hybrid baseline.

Reuses the existing walk-forward hybrid pipeline but adds SVI surface features
(svi_atm_vol, svi_25d_skew, svi_term_slope) to the LightGBM feature set.

Models saved to vol_model/output/models/expanding_hybrid_svi/ (never overwrites).

Usage:
    PYTHONPATH=. python vol_model/run_spot_check_svi.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from scipy.stats import spearmanr

from vol_model.config import VolModelConfig
from vol_model.model import qlike_objective, qlike_loss, qlike_metric
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward import OptunaPatience, _make_objective
from vol_model.transvol.config import TransVolConfig
from vol_model.transvol.model import TransVolEncoder
from vol_model.transvol.dataset import TransVolDataset, FeatureNormalizer, TIME_COLS

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("spot_check_svi")

HYBRID_DIR = Path("vol_model/output/models/expanding_hybrid")
SVI_DIR = Path("vol_model/output/models/expanding_hybrid_svi")
SVI_FEATURES_PATH = Path("vol_arb/output/svi_features_15m.parquet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MONTHS = ["2026-01", "2026-02", "2026-03"]
MAX_TRIALS = 200
PATIENCE = 50


def _extract_compact_embeddings(
    tv_path: Path, X_tv: pd.DataFrame, labels: pd.DataFrame, end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Extract 16-dim TransVol embeddings (same as run_walkforward_hybrid.py)."""
    from vol_model.labels import load_labels
    from torch.utils.data import DataLoader

    common = X_tv.index.intersection(labels.index)
    feats = X_tv.loc[common].iloc[:-1]
    cont_var = labels["continuous_var"].loc[common].iloc[1:]
    jump_var = labels["jump_var"].loc[common].iloc[1:]
    feats.index = cont_var.index

    feats = feats.replace([np.inf, -np.inf], np.nan)
    valid = ~feats.isna().any(axis=1) & ~cont_var.isna() & ~jump_var.isna()
    feats = feats[valid]

    # Limit to end_ts
    feats = feats[feats.index < end_ts]

    feat_vals = feats.values.astype(np.float32)
    time_vals = feats[TIME_COLS].values.astype(np.float32)

    # Load checkpoint
    ckpt = torch.load(str(tv_path), map_location="cpu", weights_only=False)
    config = TransVolConfig(n_features=feat_vals.shape[1])
    model = TransVolEncoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(DEVICE)

    # Normalize using training stats from checkpoint
    normalizer = FeatureNormalizer()
    normalizer.mean = ckpt.get("feat_mean", feat_vals.mean(axis=0))
    normalizer.std = ckpt.get("feat_std", feat_vals.std(axis=0).clip(min=1e-8))
    feat_norm = normalizer.transform(feat_vals)

    time_normalizer = FeatureNormalizer()
    time_normalizer.mean = ckpt.get("time_mean", time_vals.mean(axis=0))
    time_normalizer.std = ckpt.get("time_std", time_vals.std(axis=0).clip(min=1e-8))
    time_norm = time_normalizer.transform(time_vals)

    seq_len = config.seq_len
    ds = TransVolDataset(feat_norm, time_norm,
                         np.zeros(len(feat_norm), dtype=np.float32),
                         np.zeros(len(feat_norm), dtype=np.float32),
                         seq_len)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(DEVICE)
            t = batch["time_features"].to(DEVICE)
            out = model(x, t)
            embeddings.append(out["embedding_compact"].cpu().numpy())

    emb_arr = np.concatenate(embeddings, axis=0)
    # Align: dataset drops first seq_len-1 rows
    emb_index = feats.index[seq_len - 1:]
    emb_df = pd.DataFrame(
        emb_arr, index=emb_index[:len(emb_arr)],
        columns=[f"tv_{i}" for i in range(emb_arr.shape[1])],
    )
    return emb_df


def main():
    SVI_DIR.mkdir(parents=True, exist_ok=True)

    # Load SVI features
    if not SVI_FEATURES_PATH.exists():
        logger.error("SVI features not found at %s — run batch SVI calibration first", SVI_FEATURES_PATH)
        return
    svi_feats = pd.read_parquet(SVI_FEATURES_PATH)
    # Use constant-maturity features (comparable across time, not affected by nearest expiry)
    svi_cols = [
        "svi_atm_1d", "svi_atm_1w", "svi_atm_2w", "svi_atm_1m",
        "svi_25d_skew_1w", "svi_25d_skew_1m",
        "svi_term_slope_1w_2w", "svi_term_slope_1w_1m",
    ]
    available_cols = [c for c in svi_cols if c in svi_feats.columns]
    svi_feats = svi_feats[available_cols]

    # LEAKAGE FIX: X.index is the LABEL timestamp (T+1), but X's data is from time T.
    # SVI features indexed at T+1 would be contemporaneous with the label (leakage).
    # Shift SVI index forward by 1 period (15min) so SVI[T] aligns with X[T+1].
    svi_feats.index = svi_feats.index + pd.Timedelta(minutes=15)

    coverage = svi_feats[available_cols[0]].notna().mean() if available_cols else 0
    logger.info("Loaded SVI features: %d rows, %d cols %s, %.1f%% coverage (+15m aligned)",
                len(svi_feats), len(svi_feats.columns), available_cols, 100 * coverage)

    # Load base data (same as walk-forward hybrid)
    vol_config = VolModelConfig()
    X, y, t1 = build_dataset(vol_config)

    pruned = pd.read_csv("vol_model/output/pruned_features.csv", header=None)[0].tolist()
    pruned_cols = [c for c in pruned if c in X.columns]
    X_pruned = X[pruned_cols]

    drop_prefixes = ("ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration")
    tv_cols = [c for c in X.columns if not c.startswith(drop_prefixes)]
    X_tv = X[tv_cols]

    from vol_model.labels import load_labels
    labels = load_labels(vol_config)

    data_start = X.index.min()

    results = []

    for month_str in MONTHS:
        M = pd.Timestamp(month_str + "-01", tz="UTC")
        test_end = M + pd.DateOffset(months=1)
        V = max(M - pd.DateOffset(months=4), data_start + pd.DateOffset(months=12))

        logger.info("=" * 60)
        logger.info("%s: train=[%s, %s), val=[%s, %s), test=%s",
                    month_str, data_start.strftime("%Y-%m"), V.strftime("%Y-%m"),
                    V.strftime("%Y-%m"), M.strftime("%Y-%m"), month_str)

        t0 = time.time()

        # ── Step 1: Load existing TransVol checkpoint ──
        tv_path = HYBRID_DIR / f"{month_str}_transvol.pt"
        if not tv_path.exists():
            logger.warning("%s: no TransVol checkpoint, skipping", month_str)
            continue

        # ── Step 2: Extract embeddings ──
        emb_df = _extract_compact_embeddings(tv_path, X_tv, labels, test_end)

        # ── Step 3: Build hybrid + SVI features ──
        common = X_pruned.index.intersection(emb_df.index).intersection(svi_feats.index)
        X_month = pd.concat([
            emb_df.loc[common],
            X_pruned.loc[common],
            svi_feats.loc[common],
        ], axis=1)

        # Align y
        common_y = X_month.index.intersection(y.index)
        X_month = X_month.loc[common_y]
        y_month = y.loc[common_y]

        # Masks
        train_mask = X_month.index < V
        val_mask = (X_month.index >= V) & (X_month.index < M)
        test_mask = (X_month.index >= M) & (X_month.index < test_end)

        n_train, n_val, n_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
        logger.info("  Data: train=%d, val=%d, test=%d, features=%d",
                    n_train, n_val, n_test, X_month.shape[1])

        if n_train < 100 or n_test == 0:
            logger.warning("%s: insufficient data", month_str)
            continue

        X_tr, y_tr = X_month[train_mask], y_month[train_mask]
        X_val, y_val = X_month[val_mask], y_month[val_mask]
        X_test, y_test = X_month[test_mask], y_month[test_mask]

        # Winsorize
        lo, hi = X_tr.quantile(0.01), X_tr.quantile(0.99)
        X_tr_w = X_tr.clip(lo, hi, axis=1)
        X_val_w = X_val.clip(lo, hi, axis=1)
        X_test_w = X_test.clip(lo, hi, axis=1)

        # ── Step 4: Optuna ──
        dtrain = lgb.Dataset(X_tr_w.values, label=y_tr.values, free_raw_data=False,
                             params={"feature_pre_filter": False})
        dval = lgb.Dataset(X_val_w.values, label=y_val.values, reference=dtrain, free_raw_data=False)
        dtrain.construct()
        dval.construct()

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42),
                                    pruner=MedianPruner(n_startup_trials=5))
        obj = _make_objective(dtrain, dval, X_val_w.values, y_val.values, "cpu")
        study.optimize(obj, n_trials=MAX_TRIALS,
                       callbacks=[OptunaPatience(PATIENCE)], show_progress_bar=False)

        n_trials = len(study.trials)

        # ── Step 5: Final retrain on [start, M) ──
        full_mask = X_month.index < M
        X_full = X_month[full_mask].clip(lo, hi, axis=1)
        y_full = y_month[full_mask]

        best = study.best_params.copy()
        best["objective"] = qlike_objective
        best["subsample"] = 0.8
        best["device"] = "cpu"
        best["num_threads"] = -1
        best["verbosity"] = -1
        best["seed"] = 42

        dtrain_full = lgb.Dataset(X_full.values, label=y_full.values, free_raw_data=False)
        model = lgb.train(best, dtrain_full, num_boost_round=2000,
                          valid_sets=[dval], feval=qlike_metric,
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

        # ── Step 6: Predict + evaluate ──
        preds = model.predict(X_test_w.values)
        svi_qlike = qlike_loss(y_test.values, preds)
        svi_rmse = float(np.sqrt(np.mean((y_test.values - preds) ** 2)))
        svi_rank_ic = float(spearmanr(y_test.values, preds).statistic)

        # Save model + features
        model.save_model(str(SVI_DIR / f"{month_str}.txt"))
        (SVI_DIR / f"{month_str}_features.json").write_text(json.dumps(list(X_month.columns)))

        # ── Step 7: Load existing hybrid baseline for comparison ──
        baseline_path = HYBRID_DIR / f"{month_str}.txt"
        baseline_qlike = None
        if baseline_path.exists():
            base_feat_path = HYBRID_DIR / f"{month_str}_features.json"
            base_cols = json.loads(base_feat_path.read_text()) if base_feat_path.exists() else pruned_cols
            bmodel = lgb.Booster(model_file=str(baseline_path))

            # Build baseline test features (no SVI)
            common_base = X_pruned.index.intersection(emb_df.index)
            X_base = pd.concat([emb_df.loc[common_base], X_pruned.loc[common_base]], axis=1)
            X_base = X_base.loc[X_base.index.intersection(y.index)]
            X_base_test = X_base[(X_base.index >= M) & (X_base.index < test_end)]
            y_base_test = y.loc[X_base_test.index]

            X_base_test_w = pd.DataFrame(index=X_base_test.index)
            for c in base_cols:
                X_base_test_w[c] = X_base_test[c].values if c in X_base_test.columns else 0.0
            X_base_test_w = X_base_test_w.clip(lo, hi, axis=1)

            base_preds = bmodel.predict(X_base_test_w.values)
            baseline_qlike = qlike_loss(y_base_test.values, base_preds)

        elapsed = time.time() - t0
        delta_pct = ((svi_qlike / baseline_qlike - 1) * 100) if baseline_qlike else float("nan")

        logger.info("  %s: SVI_QLIKE=%.4f  BASELINE_QLIKE=%.4f  delta=%.1f%%  trials=%d  (%.0fs)",
                    month_str, svi_qlike, baseline_qlike or 0, delta_pct, n_trials, elapsed)

        results.append({
            "month": month_str,
            "hybrid_qlike": baseline_qlike,
            "hybrid_svi_qlike": svi_qlike,
            "delta_pct": delta_pct,
            "svi_rank_ic": svi_rank_ic,
            "n_features": X_month.shape[1],
            "n_trials": n_trials,
            "time_s": elapsed,
        })

    # Summary
    if results:
        print(f"\n{'=' * 70}")
        print(f"  SPOT-CHECK: Hybrid vs Hybrid+SVI ({len(results)} months)")
        print(f"{'=' * 70}")
        print(f"  {'Month':>8s}  {'Hybrid':>10s}  {'Hybrid+SVI':>10s}  {'Delta':>8s}  {'RankIC':>8s}  {'Feats':>6s}")
        print("-" * 60)
        for r in results:
            print(f"  {r['month']:>8s}  {r['hybrid_qlike']:>10.4f}  {r['hybrid_svi_qlike']:>10.4f}"
                  f"  {r['delta_pct']:>+7.1f}%  {r['svi_rank_ic']:>8.3f}  {r['n_features']:>6d}")

        avg_base = np.mean([r["hybrid_qlike"] for r in results if r["hybrid_qlike"]])
        avg_svi = np.mean([r["hybrid_svi_qlike"] for r in results])
        avg_delta = (avg_svi / avg_base - 1) * 100 if avg_base else 0
        print("-" * 60)
        print(f"  {'AVG':>8s}  {avg_base:>10.4f}  {avg_svi:>10.4f}  {avg_delta:>+7.1f}%")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
