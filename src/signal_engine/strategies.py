"""Strategy implementations for the streaming backtester.

Each strategy only receives one feature row at a time (no ohlcv access).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from signal_engine.streaming_backtest import Signal
from signal_engine.weights import compute_sample_weights

logger = logging.getLogger(__name__)


class WalkForwardXGBStrategy:
    """XGBoost walk-forward — trains on historical features, predicts one bar."""

    def __init__(
        self,
        fwd_bars: int = 4,
        vol_filter_quantile: float = 0.50,
        confidence_percentile: float = 70.0,
        use_features: bool = True,
    ):
        self.fwd_bars = fwd_bars
        self.vol_filter_quantile = vol_filter_quantile
        self.confidence_percentile = confidence_percentile
        self.use_features = use_features
        self.model = None
        self.scaler = None
        self.vol_threshold = 0.0
        self.conf_threshold = 0.0
        self._n_features = None

    def on_train(self, bar_idx: int, features: pd.DataFrame, labels: pd.Series,
                 vol_series: pd.Series) -> None:
        if len(features) < 200:
            return
        self._n_features = features.shape[1]

        # Vol threshold from training data
        if self.vol_filter_quantile > 0 and len(vol_series) > 50:
            self.vol_threshold = float(vol_series.quantile(self.vol_filter_quantile))

        X = features.to_numpy(dtype=np.float32)
        y = labels.to_numpy(dtype=np.int8)

        if not self.use_features:
            X = np.zeros_like(X)

        # Sample weights
        sw = compute_sample_weights(
            features.index.values.astype(np.int64),
            fwd_bars=self.fwd_bars,
        )

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            tree_method="hist", random_state=42, n_jobs=1, verbosity=0,
        )
        self.model.fit(X_scaled, y, sample_weight=sw)

        # Confidence threshold from training predictions
        probs = self.model.predict_proba(X_scaled)[:, 1]
        self.conf_threshold = float(np.percentile(np.abs(probs - 0.5), self.confidence_percentile))

    def on_bar(self, bar_idx: int, feature_row: pd.Series, vol: float) -> Signal | None:
        if self.model is None or self.scaler is None:
            return None

        if self.vol_threshold > 0 and vol < self.vol_threshold:
            return None

        X = feature_row.to_numpy(dtype=np.float32).reshape(1, -1)
        if not self.use_features:
            X = np.zeros_like(X)

        X_scaled = self.scaler.transform(X)
        prob = float(self.model.predict_proba(X_scaled)[0, 1])
        conf = abs(prob - 0.5)

        if conf < self.conf_threshold:
            return None

        return Signal(
            direction=1.0 if prob > 0.5 else -1.0,
            probability=prob,
            confidence=conf,
            signal_strength=conf * 2.0,
        )


class AlwaysLongStrategy:
    def on_train(self, bar_idx, features, labels, vol_series):
        pass

    def on_bar(self, bar_idx, feature_row, vol):
        return Signal(direction=1.0, probability=0.6, confidence=0.1)


class RandomStrategy:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def on_train(self, bar_idx, features, labels, vol_series):
        pass

    def on_bar(self, bar_idx, feature_row, vol):
        d = 1.0 if self.rng.rand() > 0.5 else -1.0
        return Signal(direction=d, probability=0.5, confidence=0.0)
