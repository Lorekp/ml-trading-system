#!/usr/bin/env python
"""Hybrid LightGBM — TransVol embeddings + raw features.

Concatenates TransVol embeddings (128-dim) with the original 126 features,
then trains LightGBM with QLIKE loss on the augmented 254-feature set.

Usage:
    PYTHONPATH=. python vol_model/hybrid.py
    PYTHONPATH=. python vol_model/hybrid.py --embedding-path vol_model/output/transvol/BTCUSDT_transvol_embeddings.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vol_model.config import VolModelConfig
from vol_model.model import train_purged_cv, qlike_loss
from vol_model.pipeline import build_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("hybrid")


def build_hybrid_dataset(
    vol_config: VolModelConfig,
    embedding_path: str | Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build augmented feature matrix: TransVol embeddings + raw features.

    Returns (X_hybrid, y, t1) compatible with train_purged_cv() and walk-forward.
    """
    # Load base dataset
    X_base, y, t1 = build_dataset(vol_config)
    logger.info("Base: %d rows, %d features", len(X_base), len(X_base.columns))

    # Load TransVol embeddings
    emb = pd.read_parquet(str(embedding_path))
    logger.info("Embeddings: %d rows, %d dims", len(emb), len(emb.columns))

    # Align by index if embeddings have timestamps, otherwise by position
    if hasattr(emb.index, "tz") or emb.index.dtype == "datetime64[ns, UTC]":
        common = X_base.index.intersection(emb.index)
        X_base = X_base.loc[common]
        emb = emb.loc[common]
        y = y.loc[common]
        t1 = t1.loc[common]
    else:
        # Positional alignment: embeddings may be shorter due to seq_len warmup
        n = min(len(X_base), len(emb))
        # Embeddings correspond to the LAST n timestamps
        offset = len(X_base) - n
        X_base = X_base.iloc[offset:]
        emb = emb.iloc[:n]
        emb.index = X_base.index
        y = y.iloc[offset:]
        t1 = t1.iloc[offset:]

    # Concatenate
    X_hybrid = pd.concat([emb, X_base], axis=1)

    # Fill any NaN embeddings (from warmup) with 0
    emb_cols = [c for c in X_hybrid.columns if c.startswith("transvol_emb_")]
    n_nan = X_hybrid[emb_cols].isna().any(axis=1).sum()
    if n_nan > 0:
        X_hybrid[emb_cols] = X_hybrid[emb_cols].fillna(0.0)
        logger.info("Filled %d rows of NaN embeddings with 0", n_nan)

    logger.info("Hybrid: %d rows, %d features", len(X_hybrid), len(X_hybrid.columns))
    return X_hybrid, y, t1


def main():
    parser = argparse.ArgumentParser(description="Train hybrid LightGBM")
    parser.add_argument("--embedding-path",
                        default="vol_model/output/transvol/BTCUSDT_transvol_embeddings.parquet")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    vol_config = VolModelConfig(n_splits=args.n_splits)

    # Build hybrid dataset
    X, y, t1 = build_hybrid_dataset(vol_config, args.embedding_path)

    # Train with existing infrastructure
    logger.info("Training hybrid LightGBM (PurgedKFold, %d splits)...", vol_config.n_splits)
    results = train_purged_cv(X, y, t1, vol_config)

    # Summary
    oof = results["oof_preds"]
    valid = np.isfinite(oof)
    y_valid = y.values[valid]
    p_valid = oof[valid]

    overall_qlike = qlike_loss(y_valid, p_valid)
    overall_rmse = float(np.sqrt(np.mean((y_valid - p_valid) ** 2)))
    overall_ic = float(spearmanr(y_valid, p_valid).statistic)

    print(f"\n{'=' * 60}")
    print(f"  HYBRID LIGHTGBM RESULTS")
    print(f"{'=' * 60}")
    print(f"  Features: {len(X.columns)} ({len([c for c in X.columns if c.startswith('transvol')])} embedding + {len(X.columns) - len([c for c in X.columns if c.startswith('transvol')])} raw)")
    print(f"  QLIKE:  {overall_qlike:.4f}")
    print(f"  RMSE:   {overall_rmse:.4f}")
    print(f"  RankIC: {overall_ic:.3f}")

    for fm in results["fold_metrics"]:
        print(f"  Fold {fm['fold']}: QLIKE={fm['qlike']:.4f} RMSE={fm['rmse']:.4f}")


if __name__ == "__main__":
    main()
