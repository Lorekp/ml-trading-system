#!/usr/bin/env python
"""Extract OOS predictions from the W4.1 directional classifier head.

After ``run_walkforward_classifier_head.py`` finishes training the monthly
classifiers, this script loads each booster, rebuilds its exact test-window
feature matrix (matching the training winsorization), predicts, and
concatenates all OOS predictions into a single parquet for the backtester
to consume.

Output schema:

    Columns:
        direction_prob : float  — P(y_true > y_pred_base) from the classifier
        direction_true : int8   — realized label (from the base preds parquet)
    Index:
        DatetimeIndex named "timestamp" (UTC)

The output path (``BTCUSDT_classifier_head_predictions.parquet``) is what
``BacktestConfig.classifier_path`` consumes through ``--classifier-path``.

Usage:
    PYTHONPATH=. python vol_model/extract_classifier_predictions.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from vol_model.config import VolModelConfig
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward_classifier_head import (
    BASE_PREDS_PATH,
    MIN_TRAIN_MONTHS,
    OUTPUT_ROOT,
    TARGET_CHOICES,
    _load_direction_target,
)
from vol_model.run_walkforward_hybrid import _extract_compact_embeddings
from vol_model.run_walkforward_pruned_svi import (
    ADD_SVI_FEATURES,
    DROP_FEATURES,
    HYBRID_DIR,
    _load_svi_atm_1d,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_classifier_head")

PRUNED_PATH = Path("vol_model/output/pruned_features.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract classifier head OOS predictions",
    )
    parser.add_argument("--start", default="2023-06")
    parser.add_argument("--end", default="2026-03")
    parser.add_argument(
        "--target", choices=TARGET_CHOICES, default="edge",
        help=(
            "Must match the --target used during training. Default 'edge' "
            "picks classifier_head_edge_v1/ and the matching output parquet."
        ),
    )
    parser.add_argument(
        "--model-dir", type=str, default="",
        help=(
            "Where the per-month classifier models live. Default: "
            "vol_model/output/models/classifier_head_{target}_v1/"
        ),
    )
    parser.add_argument(
        "--output", type=str, default="",
        help=(
            "Where to write the concatenated predictions parquet. Default: "
            "vol_model/output/BTCUSDT_classifier_head_{target}_predictions.parquet"
        ),
    )
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = OUTPUT_ROOT / f"classifier_head_{args.target}_v1"
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(
            f"vol_model/output/BTCUSDT_classifier_head_{args.target}_predictions.parquet"
        )

    print("=" * 70)
    print("  EXTRACT CLASSIFIER HEAD PREDICTIONS (W4.1)")
    print("=" * 70)
    print(f"  Target variant:  {args.target}")
    print(f"  Models from:     {model_dir}")
    print(f"  Base preds:      {BASE_PREDS_PATH}")
    print(f"  Output:          {output_path}")
    print("=" * 70)

    logger.info("Loading base features...")
    vol_config = VolModelConfig()
    X, _, _ = build_dataset(vol_config)

    pruned = pd.read_csv(PRUNED_PATH, header=None)[0].tolist()
    pruned_cols_v2 = [
        c for c in pruned if c in X.columns and c not in DROP_FEATURES
    ]
    X_pruned = X[pruned_cols_v2]

    drop_prefixes = (
        "ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration",
    )
    tv_cols = [c for c in X.columns if not c.startswith(drop_prefixes)]
    X_tv = X[tv_cols]

    from vol_model.labels import load_labels
    labels = load_labels(vol_config)

    svi_add = _load_svi_atm_1d()
    direction_target = _load_direction_target(args.target)

    target_start = direction_target.index.min()
    min_V = target_start + pd.DateOffset(months=MIN_TRAIN_MONTHS)

    start = pd.Timestamp(args.start + "-01", tz="UTC")
    end = pd.Timestamp(args.end + "-01", tz="UTC")
    months = pd.date_range(start, end, freq="MS")

    all_probs: list[pd.Series] = []
    all_true: list[pd.Series] = []
    n_ok = 0
    n_skip = 0

    for i, M in enumerate(months):
        month_str = M.strftime("%Y-%m")
        lgb_path = model_dir / f"{month_str}.txt"
        feat_manifest = model_dir / f"{month_str}_features.json"

        if not lgb_path.exists() or not feat_manifest.exists():
            logger.warning(
                "[%d/%d] %s: model/manifest missing, skipping",
                i + 1, len(months), month_str,
            )
            n_skip += 1
            continue

        tv_path = HYBRID_DIR / f"{month_str}_transvol.pt"
        if not tv_path.exists():
            logger.warning(
                "[%d/%d] %s: TransVol checkpoint missing",
                i + 1, len(months), month_str,
            )
            n_skip += 1
            continue

        test_end = M + pd.DateOffset(months=1)
        V = max(M - pd.DateOffset(months=4), min_V)

        try:
            emb_df = _extract_compact_embeddings(
                tv_path, X_tv, labels, test_end,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[%d/%d] %s: TransVol extraction failed",
                i + 1, len(months), month_str,
            )
            n_skip += 1
            continue

        # Rebuild the exact same 44-feature matrix as training
        common = (
            X_pruned.index.intersection(emb_df.index)
            .intersection(svi_add.index)
            .intersection(direction_target.index)
        )
        X_month = pd.concat(
            [emb_df.loc[common], X_pruned.loc[common], svi_add.loc[common]],
            axis=1,
        )
        y_month = direction_target.loc[X_month.index]

        X_month = X_month.dropna(subset=ADD_SVI_FEATURES)
        y_month = y_month.loc[X_month.index]

        train_mask = (X_month.index >= target_start) & (X_month.index < V)
        test_mask = (X_month.index >= M) & (X_month.index < test_end)
        if not test_mask.any():
            logger.warning(
                "[%d/%d] %s: no test rows", i + 1, len(months), month_str,
            )
            n_skip += 1
            continue

        # Reconstruct the training-slice winsorization bounds so the test
        # feature matrix exactly matches what the model saw at training.
        X_tr = X_month[train_mask]
        lo, hi = X_tr.quantile(0.01), X_tr.quantile(0.99)
        X_test = X_month[test_mask].clip(lo, hi, axis=1)
        y_test = y_month[test_mask]

        # Load the classifier + feature order
        feat_cols = json.loads(feat_manifest.read_text())
        missing_cols = [c for c in feat_cols if c not in X_test.columns]
        if missing_cols:
            logger.warning(
                "[%d/%d] %s: %d feature(s) missing from X_test: %s",
                i + 1, len(months), month_str,
                len(missing_cols), missing_cols[:5],
            )
            for c in missing_cols:
                X_test[c] = 0.0
        X_test_ordered = X_test[feat_cols]

        model = lgb.Booster(model_file=str(lgb_path))
        probs = model.predict(X_test_ordered.values)

        # Defensive: length must match
        if len(probs) != len(X_test_ordered):
            raise RuntimeError(
                f"{month_str}: prediction length mismatch "
                f"({len(probs)} vs {len(X_test_ordered)})",
            )

        probs_series = pd.Series(
            probs, index=X_test_ordered.index, name="direction_prob",
            dtype=np.float64,
        )
        true_series = pd.Series(
            y_test.values, index=X_test_ordered.index, name="direction_true",
            dtype=np.int8,
        )
        all_probs.append(probs_series)
        all_true.append(true_series)
        n_ok += 1
        logger.info(
            "[%d/%d] %s: %d predictions, mean prob %.3f",
            i + 1, len(months), month_str,
            len(probs_series), float(probs_series.mean()),
        )

    if not all_probs:
        logger.error("No predictions extracted — nothing to save")
        return

    probs_full = pd.concat(all_probs).sort_index()
    true_full = pd.concat(all_true).sort_index()

    if not probs_full.index.is_unique:
        dup_count = int(probs_full.index.duplicated().sum())
        logger.warning(
            "Found %d duplicate indices, keeping first", dup_count,
        )
        probs_full = probs_full[~probs_full.index.duplicated(keep="first")]
        true_full = true_full[~true_full.index.duplicated(keep="first")]

    out_df = pd.DataFrame({
        "direction_prob": probs_full.values,
        "direction_true": true_full.values,
    }, index=probs_full.index)
    out_df.index.name = "timestamp"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path)

    print()
    print("=" * 70)
    print(f"  EXTRACTED {n_ok}/{len(months)} months ({n_skip} skipped)")
    print(f"  Total predictions: {len(out_df):,}")
    print(f"  Date range: {out_df.index.min()} to {out_df.index.max()}")
    print(
        f"  direction_prob: mean={out_df['direction_prob'].mean():.4f}, "
        f"std={out_df['direction_prob'].std():.4f}",
    )
    print(
        f"  direction_true: positive rate "
        f"{out_df['direction_true'].mean():.4f}",
    )
    print(f"  Saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
