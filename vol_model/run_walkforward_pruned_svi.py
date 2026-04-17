#!/usr/bin/env python
"""Walk-forward retrain: T2.1 SHAP prune + svi_atm_1d (pruned hybrid+SVI).

Runs the full walk-forward LightGBM training across all months with:

1. **Pruned base features**: drops 8 dead-weight SHAP features from the
   standard pruned_features.csv list — ``ffd_rv``, ``ffd_rv_lag3``,
   ``ffd_rv_ema10``, ``ffd_rv_ema50`` (combined <1% SHAP), and
   ``noise_level``, ``noise_level_lag1``, ``noise_level_ema10``,
   ``noise_level_ema50`` (combined ~1% SHAP). These carry near-zero
   predictive content per Stage 15.9.

2. **Adds ``svi_atm_1d``**: the top SVI constant-maturity feature at
   8.91% average SHAP importance — the 4th-ranked feature overall
   per Stage 15.7. Loaded from ``vol_arb/output/svi_features_15m.parquet``
   with the mandatory +15-minute leakage shift (Stage 15.1).

3. **Reuses existing TransVol checkpoints** from
   ``vol_model/output/models/expanding_hybrid/`` so we only retrain the
   fast LightGBM stage, not the expensive TransVol encoder stage. This
   is correct because TransVol operates on the full 101-feature base
   set which is unchanged by the LightGBM-level SHAP prune.

4. **Saves models** to a NEW directory
   ``vol_model/output/models/expanding_pruned_svi_v1/`` so the production
   ``expanding_hybrid/`` path is never touched. Comparison tearsheets can
   run both and diff the trade logs.

Target: 50 months × ~10-15 min LightGBM-only per month ≈ 8-12h wall clock,
fits an overnight run.

Usage:
    PYTHONPATH=. python vol_model/run_walkforward_pruned_svi.py
    PYTHONPATH=. python vol_model/run_walkforward_pruned_svi.py \\
        --max-trials 100 --patience 30
"""

from __future__ import annotations

import argparse
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

from vol_arb.paths import (
    drop_features_path, pruned_features_path, pruned_svi_model_dir,
    svi_features_path, transvol_dir, CURRENCY_FOR_ASSET,
)
from vol_model.config import VolModelConfig
from vol_model.model import qlike_loss, qlike_metric, qlike_objective
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward import OptunaPatience, _make_objective
# Reuse the production embedding extractor so this script stays in sync
# with run_walkforward_hybrid.py if the TransVol API evolves.
from vol_model.run_walkforward_hybrid import _extract_compact_embeddings

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wf_pruned_svi")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dead-weight features to drop per Stage 15.9 SHAP analysis. The BTC
# list is kept as a fallback constant for code that imports it, but the
# per-asset DROP list now lives at
# ``vol_model/output/{asset}_drop_features.csv`` (produced by
# run_shap_analysis.py --drop-bottom-n). Loaded lazily in main() so each
# asset gets its own data-driven Stage 15.9 tail — no hand-curation.
DROP_FEATURES = [
    "ffd_rv", "ffd_rv_lag3", "ffd_rv_ema10", "ffd_rv_ema50",
    "noise_level", "noise_level_lag1", "noise_level_ema10", "noise_level_ema50",
]


def _load_drop_features(asset: str) -> list[str]:
    """Read the per-asset SHAP-tail drop list. Falls back to the BTC
    hand-curated constant if the asset hasn't been calibrated yet — so a
    fresh repo still runs BTC byte-identical."""
    path = drop_features_path(asset)
    if path.exists():
        with open(path) as f:
            drops = [line.strip() for line in f if line.strip()]
        logger.info("Loaded %d drop features for %s from %s", len(drops), asset, path)
        return drops
    logger.warning("No drop-features csv at %s — falling back to BTC DROP_FEATURES", path)
    return list(DROP_FEATURES)

# SVI features to add (only svi_atm_1d per Stage 15.10 recommendation)
ADD_SVI_FEATURES = ["svi_atm_1d"]


def _load_svi_atm_1d(currency: str) -> pd.DataFrame:
    """Load svi_atm_1d from the per-currency SVI features parquet with leakage shift."""
    svi_path = svi_features_path(currency)
    if not svi_path.exists():
        raise FileNotFoundError(
            f"SVI features not found at {svi_path} — "
            f"rebuild via `python -m vol_arb.run_pipeline --currency {currency} --svi`"
        )
    svi = pd.read_parquet(svi_path)
    # Ensure DatetimeIndex (ETH parquet may have timestamp as a column, not the index)
    if "timestamp" in svi.columns and not isinstance(svi.index, pd.DatetimeIndex):
        svi = svi.set_index("timestamp")
    if not isinstance(svi.index, pd.DatetimeIndex):
        svi.index = pd.to_datetime(svi.index, utc=True)
    missing = [c for c in ADD_SVI_FEATURES if c not in svi.columns]
    if missing:
        raise KeyError(
            f"SVI features parquet is missing required columns: {missing}. "
            f"Available SVI columns: {[c for c in svi.columns if 'svi' in c.lower()]}",
        )
    svi_add = svi[ADD_SVI_FEATURES].copy()

    # LEAKAGE FIX (Stage 15.1): X.index is the LABEL timestamp (T+1) but X's
    # data comes from the feature time T.  SVI features indexed at T+1 would
    # be contemporaneous with the label.  Shift SVI index forward by 15 min
    # so SVI[T] aligns with X[T+1] — i.e., SVI from the PREVIOUS bar is
    # associated with the CURRENT bar's label.
    svi_add.index = svi_add.index + pd.Timedelta(minutes=15)

    coverage = svi_add[ADD_SVI_FEATURES[0]].notna().mean()
    logger.info(
        "Loaded SVI features (+15m aligned): %d rows, cols=%s, %.1f%% coverage",
        len(svi_add), ADD_SVI_FEATURES, 100 * coverage,
    )
    return svi_add


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward pruned + svi_atm_1d training (T2.1)",
    )
    parser.add_argument("--asset", default="BTCUSDT",
                        help="Training asset (BTCUSDT or ETHUSDT). Resolves "
                             "TransVol checkpoint dir, SVI features, pruned "
                             "feature list, and output model dir per asset.")
    parser.add_argument("--start", default="2022-02",
                        help="First month to retrain (inclusive), YYYY-MM")
    parser.add_argument("--end", default="2026-03",
                        help="Last month to retrain (inclusive), YYYY-MM")
    parser.add_argument("--max-trials", type=int, default=100,
                        help="Optuna trial budget per month (default 100; "
                             "production hybrid uses 200)")
    parser.add_argument("--patience", type=int, default=30,
                        help="Optuna patience (default 30; production uses 50)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Override save dir (else per-asset via resolver)")
    parser.add_argument("--tf", type=int, default=900,
                        help="Timeframe in seconds (default 900 = 15m)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix tag (appended after asset-aware default)")
    parser.add_argument("--skip-drop", action="store_true",
                        help="Train WITHOUT the Stage-15.9 SHAP-tail DROP list — "
                             "keeps all 35 pruned features (no drop, pure top-N) "
                             "so the A/B comparison full-35 vs drop-27 is clean.")
    parser.add_argument("--save-shap", action="store_true",
                        help="After each month's LGBM trains, compute TreeExplainer "
                             "SHAP on the test window and save per-month mean |SHAP| "
                             "to {output_dir}/shap_{month}.csv. Enables the "
                             "aggregate_walkforward_shap.py rollup — data-driven "
                             "drop list from walk-forward (not single pre-cutoff).")
    parser.add_argument("--prune-source-asset", default="",
                        help="Override the asset used to resolve pruned_features.csv "
                             "and drop_features.csv (but NOT the training data asset). "
                             "Pass BTCUSDT to train ETH with BTC's 27-feature pruned "
                             "list — useful for diagnosing ETH-specific outlier months "
                             "where the 12 ETH-only features may destabilise LGBM.")
    parser.add_argument("--predict-excess-logvol", action="store_true",
                        help="Reparameterize the target as y = log(RV_future) - "
                             "log(IV_at_decision_time). Auto-applies tag 'excess' "
                             "to the output model directory so existing predictions "
                             "are not overwritten. See VolModelConfig.predict_excess_logvol "
                             "docstring for rationale.")
    args = parser.parse_args()
    # Auto-tag excess runs so we never overwrite the default-target predictions.
    if args.predict_excess_logvol and not args.tag and not args.output_dir:
        args.tag = "excess"

    currency = CURRENCY_FOR_ASSET[args.asset]
    hybrid_dir = transvol_dir(args.asset)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.tag:
        base = pruned_svi_model_dir(args.asset)
        output_dir = base.with_name(f"{base.name}_{args.tag}")
    else:
        output_dir = pruned_svi_model_dir(args.asset)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    tf_label = f"{args.tf // 60}m" if args.tf < 86400 else f"{args.tf // 3600}h"
    print(f"  WALK-FORWARD PRUNED + svi_atm_1d (T2.1) @ {tf_label} — {args.asset}")
    print("=" * 70)
    print(f"  Drop features: {DROP_FEATURES}")
    print(f"  Add SVI: {ADD_SVI_FEATURES}")
    print(f"  Reuse TransVol from: {hybrid_dir}")
    print(f"  Save models to: {output_dir}")
    print(f"  Optuna: {args.max_trials} trials, patience {args.patience}")
    print("=" * 70)

    # Load all data once
    logger.info("Loading base features + labels...")
    vol_config = VolModelConfig(
        asset=args.asset,
        predict_excess_logvol=args.predict_excess_logvol,
    )
    if args.tf != 900:
        vol_config.freq_s = args.tf
        logger.info("Timeframe override: freq_s = %d (%dm bars)", args.tf, args.tf // 60)
    if args.predict_excess_logvol:
        logger.info("Target reparameterization ACTIVE: y = log(RV) - log(IV_at_T)")
    X, y, t1 = build_dataset(vol_config)

    # Per-asset SHAP-tail DROP list: bottom-N of the top-N pruned (Stage
    # 15.9 methodology applied per asset). For BTC this reproduces the
    # original 8-feature drop exactly; for ETH/other assets it's the
    # data-driven bottom-N of the ETH SHAP ranking with the same N so
    # pruning depth stays consistent (never deeper than BTC).
    #
    # --prune-source-asset overrides the feature-list source (for diagnosing
    # outlier months by retraining ETH with BTC's 27-feature list).
    prune_asset = args.prune_source_asset or args.asset
    pruned_path = pruned_features_path(prune_asset)
    pruned = pd.read_csv(pruned_path, header=None)[0].tolist()
    pruned_cols_v1 = [c for c in pruned if c in X.columns]
    missing_pruned = [c for c in pruned if c not in X.columns]
    if missing_pruned:
        logger.warning(
            "%d features from %s's pruned list not in %s features: %s",
            len(missing_pruned), prune_asset, args.asset, missing_pruned,
        )
    if args.skip_drop:
        drop_set: set[str] = set()
        pruned_cols_v2 = list(pruned_cols_v1)
        logger.info(
            "Pruned base (%s, --skip-drop): %d features (no Stage-15.9 drop applied)",
            args.asset, len(pruned_cols_v2),
        )
    else:
        drop_set = set(_load_drop_features(prune_asset))
        pruned_cols_v2 = [c for c in pruned_cols_v1 if c not in drop_set]
        dropped = [c for c in pruned_cols_v1 if c in drop_set]
        logger.info(
            "Pruned base (%s): %d -> %d features (dropped %d: %s)",
            args.asset, len(pruned_cols_v1), len(pruned_cols_v2), len(dropped), dropped,
        )
    X_pruned = X[pruned_cols_v2]

    # TransVol feature set (unchanged; TransVol uses the full 101 features
    # regardless of the LightGBM-level prune)
    drop_prefixes = ("ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration")
    tv_cols = [c for c in X.columns if not c.startswith(drop_prefixes)]
    X_tv = X[tv_cols]

    from vol_model.labels import load_labels
    labels = load_labels(vol_config)

    # Load the SVI feature(s)
    svi_add = _load_svi_atm_1d(currency)

    data_start = X.index.min()
    start = pd.Timestamp(args.start + "-01", tz="UTC")
    end = pd.Timestamp(args.end + "-01", tz="UTC")
    months = pd.date_range(start, end, freq="MS")

    results = []
    t_start = time.time()

    for i, M in enumerate(months):
        month_str = M.strftime("%Y-%m")
        lgb_path = output_dir / f"{month_str}.txt"

        if lgb_path.exists():
            logger.info("[%d/%d] %s: cached, skipping", i + 1, len(months), month_str)
            continue

        tv_path = hybrid_dir / f"{month_str}_transvol.pt"
        if not tv_path.exists():
            logger.warning(
                "[%d/%d] %s: no TransVol checkpoint at %s — skipping",
                i + 1, len(months), month_str, tv_path,
            )
            continue

        test_end = M + pd.DateOffset(months=1)
        V = max(M - pd.DateOffset(months=4), data_start + pd.DateOffset(months=12))

        logger.info("=" * 60)
        logger.info(
            "[%d/%d] %s: train=[%s, %s), val=[%s, %s), test=%s",
            i + 1, len(months), month_str,
            data_start.strftime("%Y-%m"), V.strftime("%Y-%m"),
            V.strftime("%Y-%m"), M.strftime("%Y-%m"), month_str,
        )
        t0 = time.time()

        # Extract TransVol embeddings from the frozen checkpoint
        try:
            emb_df = _extract_compact_embeddings(tv_path, X_tv, labels, test_end)
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s: TransVol embedding extraction failed: %s", month_str, exc)
            continue

        # Build the training matrix: TransVol embeddings + pruned base + svi_atm_1d
        common = (
            X_pruned.index.intersection(emb_df.index).intersection(svi_add.index)
        )
        X_month = pd.concat(
            [emb_df.loc[common], X_pruned.loc[common], svi_add.loc[common]],
            axis=1,
        )

        common_y = X_month.index.intersection(y.index)
        X_month = X_month.loc[common_y]
        y_month = y.loc[common_y]

        # Drop any rows with NaN in svi_atm_1d (5% coverage gap)
        before = len(X_month)
        X_month = X_month.dropna(subset=ADD_SVI_FEATURES)
        y_month = y_month.loc[X_month.index]
        dropped_nan = before - len(X_month)
        if dropped_nan > 0:
            logger.info("  Dropped %d rows with NaN svi_atm_1d (%.1f%%)",
                        dropped_nan, 100 * dropped_nan / max(before, 1))

        # Temporal masks
        train_mask = X_month.index < V
        val_mask = (X_month.index >= V) & (X_month.index < M)
        test_mask = (X_month.index >= M) & (X_month.index < test_end)

        n_train, n_val, n_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
        logger.info(
            "  Data: train=%d, val=%d, test=%d, features=%d",
            n_train, n_val, n_test, X_month.shape[1],
        )
        if n_train < 100 or n_val == 0 or n_test == 0:
            logger.warning(
                "%s: insufficient data (train=%d, val=%d, test=%d) — "
                "walk-forward needs all three windows populated; typically "
                "happens when M < data_start + 12 months (not enough history "
                "for the 12-month training lead-in).",
                month_str, n_train, n_val, n_test,
            )
            continue

        X_tr, y_tr = X_month[train_mask], y_month[train_mask]
        X_val, y_val = X_month[val_mask], y_month[val_mask]
        X_test, y_test = X_month[test_mask], y_month[test_mask]

        # Winsorize on train only
        lo, hi = X_tr.quantile(0.01), X_tr.quantile(0.99)
        X_tr_w = X_tr.clip(lo, hi, axis=1)
        X_val_w = X_val.clip(lo, hi, axis=1)
        X_test_w = X_test.clip(lo, hi, axis=1)

        # Optuna search
        dtrain = lgb.Dataset(
            X_tr_w.values, label=y_tr.values, free_raw_data=False,
            params={"feature_pre_filter": False},
        )
        dval = lgb.Dataset(
            X_val_w.values, label=y_val.values, reference=dtrain, free_raw_data=False,
        )
        dtrain.construct()
        dval.construct()

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5),
        )
        obj = _make_objective(dtrain, dval, X_val_w.values, y_val.values, "cpu")
        study.optimize(
            obj,
            n_trials=args.max_trials,
            callbacks=[OptunaPatience(args.patience)],
            show_progress_bar=False,
        )
        n_trials = len(study.trials)

        # Final retrain on [start, M)
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

        dtrain_full = lgb.Dataset(
            X_full.values, label=y_full.values, free_raw_data=False,
        )
        model = lgb.train(
            best, dtrain_full, num_boost_round=2000,
            valid_sets=[dval], feval=qlike_metric,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        # Predict + evaluate on test month
        preds = model.predict(X_test_w.values)
        test_qlike = qlike_loss(y_test.values, preds)
        test_rmse = float(np.sqrt(np.mean((y_test.values - preds) ** 2)))
        test_rank_ic = float(spearmanr(y_test.values, preds).statistic)

        # Save the model + feature manifest
        model.save_model(str(lgb_path))
        (output_dir / f"{month_str}_features.json").write_text(
            json.dumps(list(X_month.columns)),
        )

        # Per-month SHAP on the test window — for walk-forward Stage 15.9
        # tail analysis. Saving is opt-in (heavy); aggregate later with
        # vol_model/aggregate_walkforward_shap.py.
        if args.save_shap:
            try:
                import shap as _shap  # noqa: WPS433 — lazy import, not every run needs it
                explainer = _shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_test_w.values)
                mean_abs = np.abs(shap_vals).mean(axis=0)
                shap_df = pd.DataFrame({
                    "feature": list(X_month.columns),
                    "mean_abs_shap": mean_abs,
                })
                shap_df.to_csv(output_dir / f"shap_{month_str}.csv", index=False)
            except Exception as shap_err:  # noqa: BLE001
                logger.warning("SHAP failed for %s: %s", month_str, shap_err)

        # Compare to the cached baseline from expanding_hybrid/ for this month
        baseline_path = hybrid_dir / f"{month_str}.txt"
        baseline_qlike = None
        if baseline_path.exists():
            base_feat_path = hybrid_dir / f"{month_str}_features.json"
            if base_feat_path.exists():
                base_cols = json.loads(base_feat_path.read_text())
                bmodel = lgb.Booster(model_file=str(baseline_path))

                # Build baseline test features (no SVI, full pruned_v1)
                X_pruned_v1 = X[pruned_cols_v1]
                common_base = X_pruned_v1.index.intersection(emb_df.index)
                X_base = pd.concat(
                    [emb_df.loc[common_base], X_pruned_v1.loc[common_base]],
                    axis=1,
                )
                X_base = X_base.loc[X_base.index.intersection(y.index)]
                X_base_test = X_base[
                    (X_base.index >= M) & (X_base.index < test_end)
                ]
                y_base_test = y.loc[X_base_test.index]
                X_base_test_w = pd.DataFrame(index=X_base_test.index)
                for c in base_cols:
                    X_base_test_w[c] = (
                        X_base_test[c].values if c in X_base_test.columns else 0.0
                    )
                X_base_test_w = X_base_test_w.clip(lo, hi, axis=1)
                base_preds = bmodel.predict(X_base_test_w.values)
                baseline_qlike = qlike_loss(y_base_test.values, base_preds)

        elapsed = time.time() - t0
        delta_pct = (
            ((test_qlike / baseline_qlike - 1) * 100)
            if baseline_qlike else float("nan")
        )

        logger.info(
            "  [%d/%d] %s: qlike=%.4f  base=%.4f  delta=%+.1f%%  "
            "rank_ic=%.3f  trials=%d  (%.0fs)",
            i + 1, len(months), month_str, test_qlike,
            baseline_qlike or 0, delta_pct, test_rank_ic, n_trials, elapsed,
        )

        results.append({
            "month": month_str,
            "baseline_qlike": baseline_qlike,
            "pruned_svi_qlike": test_qlike,
            "delta_pct": delta_pct,
            "rank_ic": test_rank_ic,
            "n_features": X_month.shape[1],
            "n_trials": n_trials,
            "time_s": elapsed,
        })

    # Summary
    total_elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print(f"  PRUNED+SVI WALK-FORWARD — {len(results)} months retrained")
    print(f"  Total elapsed: {total_elapsed / 3600:.2f}h")
    print("=" * 70)
    if results:
        print(
            f"  {'Month':>8s}  {'Baseline':>10s}  {'Pruned+SVI':>10s}  "
            f"{'Delta':>8s}  {'RankIC':>8s}  {'Feats':>6s}  {'Trials':>7s}"
        )
        print("-" * 70)
        for r in results:
            base = r["baseline_qlike"] or 0
            print(
                f"  {r['month']:>8s}  {base:>10.4f}  {r['pruned_svi_qlike']:>10.4f}  "
                f"{r['delta_pct']:>+7.1f}%  {r['rank_ic']:>8.3f}  "
                f"{r['n_features']:>6d}  {r['n_trials']:>7d}"
            )
        valid = [r for r in results if r["baseline_qlike"]]
        if valid:
            avg_base = float(np.mean([r["baseline_qlike"] for r in valid]))
            avg_new = float(np.mean([r["pruned_svi_qlike"] for r in valid]))
            avg_delta = (avg_new / avg_base - 1) * 100
            print("-" * 70)
            print(
                f"  {'AVG':>8s}  {avg_base:>10.4f}  {avg_new:>10.4f}  "
                f"{avg_delta:>+7.1f}%"
            )
        print("=" * 70)

        # Persist the summary
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps({
            "total_hours": total_elapsed / 3600,
            "n_months": len(results),
            "avg_delta_pct": avg_delta if valid else None,
            "per_month": results,
        }, indent=2, default=str))
        logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
