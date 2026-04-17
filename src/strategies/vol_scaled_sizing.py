"""Strategy 5: Vol-Scaled Existing Signals. Highest priority — proven concept."""

from __future__ import annotations
import numpy as np
import pandas as pd
from strategies.base import BaseVolStrategy


class VolScaledSizing(BaseVolStrategy):
    """
    Size existing crash reversal signals using vol prediction.
    High predicted vol → smaller position. Low predicted vol → larger position.
    Direction comes from the crash signal, not from ML.
    """

    name = "vol_scaled_sizing"

    def __init__(self, crash_threshold: float = -0.02, **kwargs):
        super().__init__(**kwargs)
        self.crash_threshold = crash_threshold

    def generate_signals(self, ohlcv: pd.DataFrame, predicted_vol: pd.Series) -> pd.DataFrame:
        close = ohlcv["close"]
        returns = close.pct_change()

        # Primary signal: crash reversal (go LONG after big drop)
        # Lagged 1 bar: crash at t → signal at t+1
        crash_detected = (returns < self.crash_threshold).astype(float).shift(1).fillna(0)

        # Vol-based sizing: inverse vol (high vol → smaller)
        vol_median = predicted_vol.rolling(100).median()
        vol_ratio = (vol_median / predicted_vol).clip(0.3, 2.0).fillna(1.0)

        signals = pd.DataFrame(index=ohlcv.index)
        signals["direction"] = crash_detected  # +1 on crash bars, 0 otherwise
        signals["size"] = crash_detected * vol_ratio.reindex(signals.index, fill_value=1.0)
        signals["confidence"] = vol_ratio.reindex(signals.index, fill_value=0.5)
        return signals
