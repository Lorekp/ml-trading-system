"""Strategy 3: Vol Compression → Expansion (Bollinger Squeeze)."""

from __future__ import annotations
import numpy as np
import pandas as pd
from strategies.base import BaseVolStrategy
from strategies.common import bb_squeeze


class VolSqueeze(BaseVolStrategy):
    """
    Detect BB squeeze + vol model predicts expansion → enter in breakout direction.
    ML tells us WHEN the squeeze breaks. Direction from the first breakout bar.
    """

    name = "vol_squeeze"

    def __init__(self, expansion_ratio: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.expansion_ratio = expansion_ratio

    def generate_signals(self, ohlcv: pd.DataFrame, predicted_vol: pd.Series) -> pd.DataFrame:
        close = ohlcv["close"]
        from afml.features import parkinson_vol
        realized_vol = parkinson_vol(ohlcv["high"], ohlcv["low"], window=20)

        # Bollinger squeeze detection
        bb = bb_squeeze(close, window=20, n_std=2.0, squeeze_pct=0.10, lookback=50)

        # Vol model predicts expansion: predicted > realized * expansion_ratio
        pred_aligned = predicted_vol.reindex(close.index, method="ffill")
        expansion_expected = pred_aligned > (realized_vol * self.expansion_ratio)

        # Armed: squeeze AND expansion expected
        armed = bb["is_squeeze"] & expansion_expected

        # Entry: breakout bar AFTER being armed (1-bar lag built into breakout detection)
        # breakout_up/down already require recent_squeeze
        direction = pd.Series(0.0, index=close.index)
        direction[bb["breakout_up"]] = 1.0
        direction[bb["breakout_down"]] = -1.0

        # Only take breakouts where we were armed recently (last 5 bars)
        armed_recent = armed.rolling(5).max().fillna(0) > 0
        direction = direction * armed_recent.astype(float)

        # Lag 1 bar for execution
        direction = direction.shift(1).fillna(0)

        # Size: inversely proportional to predicted vol
        vol_median = predicted_vol.rolling(100).median()
        size_mult = (vol_median / pred_aligned).clip(0.3, 2.0).fillna(1.0)

        signals = pd.DataFrame(index=ohlcv.index)
        signals["direction"] = direction
        signals["size"] = direction.abs() * size_mult.reindex(signals.index, fill_value=1.0)
        signals["confidence"] = armed_recent.astype(float) * 0.7
        return signals
