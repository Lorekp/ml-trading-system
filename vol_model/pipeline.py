"""Feature assembly pipeline — merge all feature groups, apply windowing, align labels.

Pipeline order (prevents data leakage):
1. Compute raw features per completed window (strict [start, end) boundaries)
2. Apply windowing (lags, EMAs) — backward-looking by construction
3. One-step-ahead alignment: features[:-1] predict labels[1:]
4. Drop NaN warmup rows
5. Winsorize features (per-fold in training, globally for caching)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_model.config import VolModelConfig
from vol_model.labels import load_labels, align_features_labels
from vol_model.features.windowing import apply_windowing

logger = logging.getLogger(__name__)


def load_cached_features(config: VolModelConfig) -> dict[str, pd.DataFrame]:
    """Load all cached feature parquets."""
    out = Path(config.output_dir)
    groups = {}

    for name in ["memory", "toxicity", "liquidity", "derivatives"]:
        path = out / f"{config.asset}_features_{name}.parquet"
        if path.exists():
            groups[name] = pd.read_parquet(path)
            logger.info("  Loaded %s: %d rows, %d cols", name, len(groups[name]), len(groups[name].columns))
        else:
            logger.warning("  %s features not found at %s", name, path)

    return groups


def assemble_features(config: VolModelConfig) -> pd.DataFrame:
    """Load features — prefers dollar-bar-engineered, falls back to legacy groups.

    Priority:
        1. {asset}_features_engineered.parquet (dollar-bar-native, from FeatureEngineer)
        2. Legacy per-group parquets + windowing (deprecated)
    """
    out = Path(config.output_dir)

    # Prefer dollar-bar-engineered features
    tf_label = f"{config.freq_s // 60}m" if config.freq_s < 3600 else f"{config.freq_s // 3600}h"
    eng_path = out / f"{config.asset}_features_{tf_label}.parquet"
    if eng_path.exists():
        features = pd.read_parquet(eng_path)
        logger.info("  Loaded engineered features: %d rows, %d cols", len(features), len(features.columns))
        return features

    # Fallback: legacy per-group assembly
    logger.warning("  Engineered features not found — using legacy groups")
    groups = load_cached_features(config)

    if not groups:
        raise ValueError("No cached features found. Run run_engineer.py or run_features.py first.")

    all_features = None
    for name, df in groups.items():
        if all_features is None:
            all_features = df
        else:
            all_features = all_features.join(df, how="inner")

    logger.info("  Merged features: %d rows, %d raw columns", len(all_features), len(all_features.columns))

    windowed = apply_windowing(all_features, lags=config.lags, ema_spans=config.ema_spans)
    logger.info("  After windowing: %d columns", len(windowed.columns))

    return windowed


def build_dataset(config: VolModelConfig) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the full training dataset.

    Returns:
        X: feature matrix (windowed, NaN-free)
        y: ln(vol_ann) labels (one-step-ahead)
        t1: label end times for PurgedKFold
    """
    features = assemble_features(config)
    labels = load_labels(config)

    X, y, t1 = align_features_labels(features, labels)

    # Fill OI features with 0 where missing (no OI data before May 2023)
    # This lets the model use OI when available without losing early rows
    oi_cols = [c for c in X.columns if c.startswith("oi_")]
    if oi_cols:
        n_missing = X[oi_cols].isna().any(axis=1).sum()
        X[oi_cols] = X[oi_cols].fillna(0.0)
        if n_missing > 0:
            logger.info("  Filled %d rows of OI features with 0 (no data before May 2023)",
                        n_missing)

    # Fill OFI/depth features with 0 where missing (L2 coverage is partial)
    ofi_cols = [c for c in X.columns if c.startswith(("ofi_", "depth_mid"))]
    if ofi_cols:
        n_missing = X[ofi_cols].isna().any(axis=1).sum()
        X[ofi_cols] = X[ofi_cols].fillna(0.0)
        if n_missing > 0:
            logger.info("  Filled %d rows of OFI/depth features with 0 (partial L2 coverage)",
                        n_missing)

    # Drop rows with any NaN (warmup period from rolling features)
    valid_mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    t1 = t1[valid_mask]

    logger.info("  Dataset: %d rows, %d features (dropped %d NaN warmup rows)",
                len(X), len(X.columns), (~valid_mask).sum())

    return X, y, t1


def winsorize(X: pd.DataFrame, lo: float = 0.01, hi: float = 0.99) -> pd.DataFrame:
    """Winsorize features at given quantiles. Fit on the data provided."""
    lo_vals = X.quantile(lo)
    hi_vals = X.quantile(hi)
    return X.clip(lo_vals, hi_vals, axis=1)
