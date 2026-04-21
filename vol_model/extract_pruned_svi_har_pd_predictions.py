#!/usr/bin/env python
"""Extract predictions from the pruned_svi_har_pd_v1 walk-forward models.

After ``run_walkforward_pruned_svi_har_pd.py`` finishes training 50 monthly
models, this script loads each one, rebuilds its exact test-window feature
matrix (TransVol embeddings + pruned base + svi_atm_1d + 6 HAR-PD features,
winsorized on the train slice), predicts, and concatenates all 50 months of
OOS predictions into a single parquet for the backtester to consume.

The output file is compatible with ``BacktestConfig.predictions_path`` —
same ``y_pred`` column name, same DatetimeIndex — so comparing pruned+SVI+HAR-PD
vs pruned+SVI is just two ``run_backtest`` invocations with different
``--predictions-path`` overrides, or (preferred for certification) a QLIKE/DM
comparison via ``vol_model/compare_pruned_svi_har_pd_vs_baseline.py``.

Usage:
    PYTHONPATH=. python vol_model/extract_pruned_svi_har_pd_predictions.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from vol_arb.paths import (
    drop_features_path, har_pd_predictions_path, pruned_features_path,
    pruned_svi_model_dir, pruned_svi_predictions_path, transvol_dir,
    CURRENCY_FOR_ASSET,
)
from vol_model.config import VolModelConfig
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward_hybrid import _extract_compact_embeddings
from vol_model.run_walkforward_pruned_svi import (
    ADD_SVI_FEATURES,
    DROP_FEATURES,
    _load_svi_atm_1d,
)
from vol_model.run_walkforward_pruned_svi_har_pd import (
    ADD_HAR_PD_FEATURES,
    _load_har_pd,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_pruned_svi_har_pd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pruned+SVI predictions")
    parser.add_argument("--asset", default="BTCUSDT",
                        help="Training asset (BTCUSDT or ETHUSDT). Resolves "
                             "model dir, SVI features, pruned feature list, "
                             "TransVol checkpoints, and output path per asset.")
    parser.add_argument("--start", default="2022-02")
    parser.add_argument("--end", default="2026-03")
    parser.add_argument("--model-dir", type=str, default="",
                        help="Override model dir (else per-asset resolver)")
    parser.add_argument("--output", type=str, default="",
                        help="Override output path (else per-asset resolver)")
    parser.add_argument("--tf", type=int, default=900,
                        help="Timeframe in seconds (default 900 = 15m)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional model dir suffix tag")
    parser.add_argument("--predict-excess-logvol", action="store_true",
                        help="Match the excess-logvol training target: build "
                             "the same reparam'd label so test-time QLIKE/RankIC "
                             "are computed on the same target the model saw. "
                             "Auto-applies tag 'excess' when unset, and suffixes "
                             "the output predictions parquet with '_excess'.")
    args = parser.parse_args()
    # Auto-tag & output-suffix so excess predictions never clobber defaults.
    if args.predict_excess_logvol and not args.tag and not args.model_dir:
        args.tag = "excess"

    currency = CURRENCY_FOR_ASSET[args.asset]
    # TransVol checkpoint dir must match the active horizon (see
    # foundation-fix invariant: model horizon == feature horizon ==
    # TransVol horizon). Legacy h=15 returns the unsuffixed BTC path.
    hybrid_dir = transvol_dir(args.asset, horizon_minutes=args.tf // 60)

    # Resolve model dir — default mirrors the HAR-PD walk-forward's output dir.
    if args.model_dir:
        model_dir = Path(args.model_dir)
    elif args.tag:
        base = pruned_svi_model_dir(args.asset)
        model_dir = base.with_name(f"{base.name}_har_pd_v1_{args.tag}")
    else:
        base = pruned_svi_model_dir(args.asset)
        model_dir = base.with_name(f"{base.name.replace('_v1', '')}_har_pd_v1")

    tf_min = args.tf // 60
    if args.output:
        output_path = Path(args.output)
    elif args.tf != 900:
        # Horizon-suffixed output via the Phase 1c path helper so this
        # matches what backtester presets + sidecar generators expect.
        # har_pd_predictions_path returns e.g. vol_model/output/
        # BTCUSDT_pruned_svi_har_pd_predictions_h60m.parquet at h=60.
        output_path = har_pd_predictions_path(
            args.asset, tf_min, aligned_bias=False,
        )
    elif args.predict_excess_logvol:
        base_path = pruned_svi_predictions_path(args.asset)
        output_path = base_path.with_name(
            base_path.stem.replace("_pruned_svi", "_pruned_svi_har_pd") + "_excess" + base_path.suffix
        )
    else:
        # h=15 default — legacy unsuffixed name via the path helper.
        output_path = har_pd_predictions_path(
            args.asset, tf_min, aligned_bias=False,
        )

    print("=" * 70)
    print(f"  EXTRACT PRUNED+SVI+HAR-PD PREDICTIONS — {args.asset}")
    print("=" * 70)
    print(f"  Models from:  {model_dir}")
    print(f"  TransVol:     {hybrid_dir}")
    print(f"  Output:       {output_path}")
    print("=" * 70)

    logger.info("Loading base data...")
    vol_config = VolModelConfig(
        asset=args.asset,
        predict_excess_logvol=args.predict_excess_logvol,
    )
    if args.tf != 900:
        vol_config.freq_s = args.tf
    X, y, _ = build_dataset(vol_config)

    # Per-asset SHAP-tail DROP list from vol_model/output/{asset}_drop_features.csv
    # (written by run_shap_analysis.py --drop-bottom-n). Falls back to the
    # BTC-hardcoded DROP_FEATURES constant if the file is missing.
    drop_path = drop_features_path(args.asset)
    if drop_path.exists() and drop_path.stat().st_size > 0:
        drop_set = set(pd.read_csv(drop_path, header=None)[0].tolist())
    elif drop_path.exists():
        drop_set = set()  # empty file = no drops (spot-check found none safe)
    else:
        drop_set = set(DROP_FEATURES)
    pruned = pd.read_csv(pruned_features_path(args.asset), header=None)[0].tolist()
    pruned_cols_v2 = [c for c in pruned if c in X.columns and c not in drop_set]
    X_pruned = X[pruned_cols_v2]

    drop_prefixes = (
        "ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration",
    )
    tv_cols = [c for c in X.columns if not c.startswith(drop_prefixes)]
    X_tv = X[tv_cols]

    from vol_model.labels import load_labels
    labels = load_labels(vol_config)

    svi_add = _load_svi_atm_1d(currency)

    # HAR-PD sidecar with -15m alignment shift (matches training).
    har_pd_add = _load_har_pd(args.asset)

    data_start = X.index.min()
    start = pd.Timestamp(args.start + "-01", tz="UTC")
    end = pd.Timestamp(args.end + "-01", tz="UTC")
    months = pd.date_range(start, end, freq="MS")

    all_preds: list[pd.Series] = []
    all_true: list[pd.Series] = []
    n_ok = 0
    n_skip = 0

    for i, M in enumerate(months):
        month_str = M.strftime("%Y-%m")
        lgb_path = model_dir / f"{month_str}.txt"
        feat_manifest = model_dir / f"{month_str}_features.json"

        if not lgb_path.exists() or not feat_manifest.exists():
            logger.warning("[%d/%d] %s: model/manifest missing, skipping",
                           i + 1, len(months), month_str)
            n_skip += 1
            continue

        tv_path = hybrid_dir / f"{month_str}_transvol.pt"
        if not tv_path.exists():
            logger.warning("[%d/%d] %s: TransVol checkpoint missing", i + 1, len(months), month_str)
            n_skip += 1
            continue

        test_end = M + pd.DateOffset(months=1)
        V = max(M - pd.DateOffset(months=4), data_start + pd.DateOffset(months=12))

        # Extract TransVol embeddings (same as training)
        try:
            emb_df = _extract_compact_embeddings(tv_path, X_tv, labels, test_end)
        except Exception:  # noqa: BLE001
            logger.exception("[%d/%d] %s: TransVol extraction failed", i + 1, len(months), month_str)
            n_skip += 1
            continue

        # Build X_month — exactly the same concat as training
        common = (
            X_pruned.index
            .intersection(emb_df.index)
            .intersection(svi_add.index)
            .intersection(har_pd_add.index)
        )
        X_month = pd.concat(
            [
                emb_df.loc[common],
                X_pruned.loc[common],
                svi_add.loc[common],
                har_pd_add.loc[common],
            ],
            axis=1,
        )
        common_y = X_month.index.intersection(y.index)
        X_month = X_month.loc[common_y]
        y_month = y.loc[common_y]

        # Drop NaN rows in svi_atm_1d or HAR-PD (same as training)
        X_month = X_month.dropna(subset=ADD_SVI_FEATURES + ADD_HAR_PD_FEATURES)
        y_month = y_month.loc[X_month.index]

        # Temporal masks
        train_mask = X_month.index < V
        test_mask = (X_month.index >= M) & (X_month.index < test_end)
        if not test_mask.any():
            logger.warning("[%d/%d] %s: no test rows", i + 1, len(months), month_str)
            n_skip += 1
            continue

        # Winsorize on train (same as training loop) — this is the crucial
        # step to reproduce the exact same feature values the model saw
        X_tr = X_month[train_mask]
        lo, hi = X_tr.quantile(0.01), X_tr.quantile(0.99)
        X_test = X_month[test_mask].clip(lo, hi, axis=1)
        y_test = y_month[test_mask]

        # Load model + feature order
        feat_cols = json.loads(feat_manifest.read_text())
        # Reorder X_test to match the model's training feature order
        missing_cols = [c for c in feat_cols if c not in X_test.columns]
        if missing_cols:
            logger.warning(
                "[%d/%d] %s: %d feature(s) missing from X_test: %s",
                i + 1, len(months), month_str, len(missing_cols), missing_cols[:5],
            )
            for c in missing_cols:
                X_test[c] = 0.0
        X_test_ordered = X_test[feat_cols]

        model = lgb.Booster(model_file=str(lgb_path))
        preds = model.predict(X_test_ordered.values)

        preds_series = pd.Series(preds, index=X_test_ordered.index, name="y_pred")
        true_series = pd.Series(y_test.values, index=X_test_ordered.index, name="y_true")
        all_preds.append(preds_series)
        all_true.append(true_series)
        n_ok += 1
        logger.info(
            "[%d/%d] %s: %d predictions",
            i + 1, len(months), month_str, len(preds_series),
        )

    if not all_preds:
        logger.error("No predictions extracted — nothing to save")
        return

    # Concatenate and sort by timestamp
    y_pred_full = pd.concat(all_preds).sort_index()
    y_true_full = pd.concat(all_true).sort_index()

    # Drop any duplicate indices (shouldn't happen but defensive)
    if not y_pred_full.index.is_unique:
        dup_count = y_pred_full.index.duplicated().sum()
        logger.warning("Found %d duplicate indices, keeping first", dup_count)
        y_pred_full = y_pred_full[~y_pred_full.index.duplicated(keep="first")]
        y_true_full = y_true_full[~y_true_full.index.duplicated(keep="first")]

    out_df = pd.DataFrame({
        "y_pred": y_pred_full.values,
        "y_true": y_true_full.values,
    }, index=y_pred_full.index)
    out_df.index.name = "timestamp"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path)

    print()
    print("=" * 70)
    print(f"  EXTRACTED {n_ok}/{len(months)} months ({n_skip} skipped)")
    print(f"  Total predictions: {len(out_df):,}")
    print(f"  Date range: {out_df.index.min()} to {out_df.index.max()}")
    print(f"  y_pred stats: mean={out_df['y_pred'].mean():.4f}, std={out_df['y_pred'].std():.4f}")
    print(f"  y_true stats: mean={out_df['y_true'].mean():.4f}, std={out_df['y_true'].std():.4f}")
    print(f"  Saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
