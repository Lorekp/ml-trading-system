"""Strategy 1: Vol Regime Momentum — low vol=trend-follow, high vol=mean-revert."""

from __future__ import annotations
import numpy as np
import pandas as pd
from strategies.base import BaseVolStrategy
from strategies.common import vol_regime, momentum_signal


class VolRegimeMomentum(BaseVolStrategy):
    """
    Use vol prediction to classify regime, then apply appropriate strategy:
    - Low vol regime → momentum (trend-following)
    - High vol regime → contrarian (mean-reversion)
    - Neutral → no trade
    """

    name = "vol_regime_momentum"

    def __init__(self, momentum_lookback: int = 14, **kwargs):
        super().__init__(**kwargs)
        self.momentum_lookback = momentum_lookback

    def generate_signals(self, ohlcv: pd.DataFrame, predicted_vol: pd.Series) -> pd.DataFrame:
        close = ohlcv["close"]
        pred_aligned = predicted_vol.reindex(close.index, method="ffill")

        # Classify regime
        regime = vol_regime(pred_aligned, window=100, low_pct=0.30, high_pct=0.70)

        # Momentum signal (already lagged 1 bar internally)
        mom = momentum_signal(close, lookback=self.momentum_lookback)

        # Combine: low vol → follow momentum, high vol → fade momentum
        direction = pd.Series(0.0, index=close.index)

        low_vol = regime == -1.0
        high_vol = regime == 1.0

        direction[low_vol] = mom[low_vol]        # trend-follow
        direction[high_vol] = -mom[high_vol]      # mean-revert

        # Size: inversely proportional to vol
        vol_median = pred_aligned.rolling(100).median()
        size_mult = (vol_median / pred_aligned).clip(0.3, 2.0).fillna(1.0)

        signals = pd.DataFrame(index=ohlcv.index)
        signals["direction"] = direction
        signals["size"] = direction.abs() * size_mult.reindex(signals.index, fill_value=1.0)
        signals["confidence"] = (regime.abs() * 0.5).clip(0, 1)
        return signals
