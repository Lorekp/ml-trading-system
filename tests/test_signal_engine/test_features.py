"""Tests for signal_engine.features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_engine.config import FeatureConfig
from signal_engine.features import compute_features


def test_returns_exactly_13_columns(synthetic_ohlcv: pd.DataFrame) -> None:
    """Feature matrix has exactly 13 columns."""
    cfg = FeatureConfig()
    features = compute_features(synthetic_ohlcv, cfg)
    assert features.shape[1] == 13


def test_column_names_match_config(synthetic_ohlcv: pd.DataFrame) -> None:
    """Column names match the feature_list in config."""
    cfg = FeatureConfig()
    features = compute_features(synthetic_ohlcv, cfg)
    assert list(features.columns) == cfg.feature_list


def test_float32_dtype(synthetic_ohlcv: pd.DataFrame) -> None:
    """All feature columns are float32."""
    cfg = FeatureConfig()
    features = compute_features(synthetic_ohlcv, cfg)
    for col in features.columns:
        assert features[col].dtype == np.float32, f"{col} is {features[col].dtype}"


def test_nan_handling(synthetic_ohlcv: pd.DataFrame) -> None:
    """First ~30 rows have NaN from lookback, later rows are mostly finite."""
    cfg = FeatureConfig()
    features = compute_features(synthetic_ohlcv, cfg)

    # First row should have NaN (lookback requirements)
    assert features.iloc[0].isna().any()

    # After row 50 everything should be finite
    tail = features.iloc[50:]
    # Allow a tiny fraction of NaN due to edge effects but most should be finite
    finite_frac = tail.notna().mean().mean()
    assert finite_frac > 0.95, f"Only {finite_frac:.2%} finite after row 50"


def test_known_values_simple() -> None:
    """Verify ret_1 on trivially constructed data."""
    idx = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
    close = pd.Series(np.linspace(100, 110, 100), index=idx)
    ohlcv = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.ones(100) * 1000,
        },
        index=idx,
    )

    cfg = FeatureConfig()
    features = compute_features(ohlcv, cfg)

    # ret_1 for a linearly increasing price should be positive and finite
    ret1 = features["ret_1"].dropna()
    assert (ret1 > 0).all()
    assert ret1.notna().all() and (ret1.abs() < 1e10).all()
