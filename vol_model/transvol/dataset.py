"""TransVol dataset — sliding window sequences from 15m feature grid."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from vol_model.transvol.config import TransVolConfig

logger = logging.getLogger(__name__)

# Time-related columns extracted from features for Time2Vec input
TIME_COLS = ["duration_s", "n_bars_in_window"]


class FeatureNormalizer:
    """Z-normalization fitted on training data only."""

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).clip(min=1e-8).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def save(self, path: str | Path) -> None:
        np.savez(str(path), mean=self.mean, std=self.std)

    def load(self, path: str | Path) -> "FeatureNormalizer":
        data = np.load(str(path))
        self.mean = data["mean"]
        self.std = data["std"]
        return self


class TransVolDataset(Dataset):
    """Sliding window dataset for TransVol encoder.

    Each sample is a sequence of `seq_len` consecutive 15m windows.
    The prediction target corresponds to the LAST window in the sequence
    (one-step-ahead alignment is handled upstream by labels.py).
    """

    def __init__(
        self,
        features: np.ndarray,          # (N, F) float32, z-normalized
        time_features: np.ndarray,     # (N, 2) float32
        continuous_var: np.ndarray,    # (N,) float32 — raw variance
        jump_var: np.ndarray,          # (N,) float32 — raw variance
        seq_len: int = 32,
    ):
        self.features = features
        self.time_features = time_features
        self.continuous_var = continuous_var
        self.jump_var = jump_var
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end = idx + self.seq_len
        return {
            "features": torch.from_numpy(self.features[idx:end]),
            "time_features": torch.from_numpy(self.time_features[idx:end]),
            "continuous_var": torch.tensor(
                self.continuous_var[end - 1], dtype=torch.float32
            ),
            "jump_var": torch.tensor(
                self.jump_var[end - 1], dtype=torch.float32
            ),
        }


def build_transvol_datasets(
    config: TransVolConfig,
) -> tuple[TransVolDataset, TransVolDataset, TransVolDataset, FeatureNormalizer, pd.DatetimeIndex]:
    """Build train/val/test datasets from existing feature + label parquets.

    Returns:
        train_ds, val_ds, test_ds, normalizer, full_index
    """
    from vol_model.config import VolModelConfig
    from vol_model.pipeline import assemble_features
    from vol_model.labels import load_labels

    vol_config = VolModelConfig()

    # Load features and labels
    logger.info("Loading features...")
    features = assemble_features(vol_config)

    logger.info("Loading labels...")
    labels = load_labels(vol_config)

    # Align (one-step-ahead)
    common = features.index.intersection(labels.index)
    features = features.loc[common]
    labels = labels.loc[common]

    # One-step-ahead: features[:-1] predict labels[1:]
    X = features.iloc[:-1].copy()
    cont_var = labels["continuous_var"].iloc[1:].copy()
    jump_var = labels["jump_var"].iloc[1:].copy()
    X.index = cont_var.index

    # Fill NaN (OI, OFI) with 0 — same as pipeline.py
    oi_cols = [c for c in X.columns if c.startswith("oi_")]
    ofi_cols = [c for c in X.columns if c.startswith(("ofi_", "depth_mid"))]
    X[oi_cols + ofi_cols] = X[oi_cols + ofi_cols].fillna(0.0)

    # Replace inf with NaN then drop all invalid rows
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop rows with any remaining NaN (warmup from rolling features)
    valid = ~X.isna().any(axis=1) & ~cont_var.isna() & ~jump_var.isna()
    X = X[valid]
    cont_var = cont_var[valid]
    jump_var = jump_var[valid]

    # Optionally filter to data_start
    if hasattr(config, "data_start") and config.data_start:
        data_start = pd.Timestamp(config.data_start, tz="UTC")
        mask = X.index >= data_start
        X = X[mask]
        cont_var = cont_var[mask]
        jump_var = jump_var[mask]
        logger.info("Filtered to >= %s: %d rows", config.data_start, len(X))

    # Drop OFI/depth/OI features (partial coverage, near-zero SHAP)
    drop_prefixes = ("ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration")
    drop_cols = [c for c in X.columns if c.startswith(drop_prefixes)]
    if drop_cols:
        X = X.drop(columns=drop_cols)
        logger.info("Dropped %d partial-coverage features: %s", len(drop_cols), drop_cols)

    logger.info("Dataset: %d rows, %d features", len(X), len(X.columns))

    # Extract time features before converting to arrays
    time_feat_vals = X[TIME_COLS].values.astype(np.float32)

    # Convert to float32 arrays
    feat_vals = X.values.astype(np.float32)
    cont_var_vals = cont_var.values.astype(np.float32)
    jump_var_vals = jump_var.values.astype(np.float32)
    full_index = X.index

    # Temporal splits on cleaned data (contiguous after dropna)
    train_end = pd.Timestamp(config.train_end, tz="UTC")
    val_end = pd.Timestamp(config.val_end, tz="UTC")

    train_end_pos = X.index.searchsorted(train_end)
    val_end_pos = X.index.searchsorted(val_end)

    logger.info("Split: train=%d, val=%d, test=%d",
                train_end_pos, val_end_pos - train_end_pos, len(X) - val_end_pos)

    # Z-normalize features (fit on train only)
    normalizer = FeatureNormalizer()
    normalizer.fit(feat_vals[:train_end_pos])
    feat_norm = normalizer.transform(feat_vals)

    # Also normalize time features
    time_normalizer = FeatureNormalizer()
    time_normalizer.fit(time_feat_vals[:train_end_pos])
    time_norm = time_normalizer.transform(time_feat_vals)

    # Normalize jump_var to O(1) so Huber loss has meaningful gradients
    # (raw jump_var ~ 9e-7 mean, Huber delta needs to be on same scale as targets)
    jump_mean = float(jump_var_vals[:train_end_pos].mean())
    jump_mean = max(jump_mean, 1e-15)
    jump_var_vals = jump_var_vals / jump_mean
    logger.info("Jump normalization: mean=%.2e, after norm mean=%.2f", jump_mean, jump_var_vals[:train_end_pos].mean())

    # Build datasets from contiguous slices
    train_ds = TransVolDataset(
        feat_norm[:train_end_pos], time_norm[:train_end_pos],
        cont_var_vals[:train_end_pos], jump_var_vals[:train_end_pos],
        config.seq_len,
    )
    val_ds = TransVolDataset(
        feat_norm[train_end_pos:val_end_pos], time_norm[train_end_pos:val_end_pos],
        cont_var_vals[train_end_pos:val_end_pos], jump_var_vals[train_end_pos:val_end_pos],
        config.seq_len,
    )
    test_ds = TransVolDataset(
        feat_norm[val_end_pos:], time_norm[val_end_pos:],
        cont_var_vals[val_end_pos:], jump_var_vals[val_end_pos:],
        config.seq_len,
    )

    return train_ds, val_ds, test_ds, normalizer, full_index
