"""Strategy 2: Vol Breakout — predicted vol spike → follow first significant bar."""

from __future__ import annotations
import numpy as np
import pandas as pd
from strategies.base import BaseVolStrategy
from strategies.common import vol_zscore_signal


class VolBreakout(BaseVolStrategy):
    """
    When vol model predicts unusually high vol (z>2), a big move is imminent.
    Enter in the direction of the first significant bar after the spike signal.
    """

    name = "vol_breakout"

    def __init__(self, zscore_threshold: float = 2.0, move_threshold_mult: float = 0.5, lookforward: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.zscore_threshold = zscore_threshold
        self.move_threshold_mult = move_threshold_mult
        self.lookforward = lookforward

    def generate_signals(self, ohlcv: pd.DataFrame, predicted_vol: pd.Series) -> pd.DataFrame:
        close = ohlcv["close"]
        returns = close.pct_change()

        pred_aligned = predicted_vol.reindex(close.index, method="ffill")
        vol_z = vol_zscore_signal(pred_aligned, window=50)

        # Signal: vol z-score > threshold
        spike_signal = vol_z > self.zscore_threshold

        # After spike: find first bar with significant move
        direction = pd.Series(0.0, index=close.index)
        size = pd.Series(0.0, index=close.index)

        active = False
        bars_since = 0

        for i in range(1, len(close)):
            if spike_signal.iloc[i] and not active:
                active = True
                bars_since = 0

            if active:
                bars_since += 1
                if bars_since > self.lookforward:
                    active = False
                    continue

                move_threshold = pred_aligned.iloc[i] * self.move_threshold_mult
                if abs(returns.iloc[i]) > move_threshold and move_threshold > 0:
                    # Found significant bar — trade in its direction
                    # Lag 1: we see this bar close, trade next bar
                    if i + 1 < len(close):
                        direction.iloc[i + 1] = np.sign(returns.iloc[i])
                        # Size inversely to vol (high vol = smaller)
                        vol_med = pred_aligned.iloc[max(0, i-100):i].median()
                        if pred_aligned.iloc[i] > 0:
                            size.iloc[i + 1] = min(vol_med / pred_aligned.iloc[i], 2.0)
                        else:
                            size.iloc[i + 1] = 1.0
                    active = False

        signals = pd.DataFrame(index=ohlcv.index)
        signals["direction"] = direction
        signals["size"] = size.clip(0.2, 2.0)
        signals["confidence"] = (vol_z > self.zscore_threshold).astype(float) * 0.6
        return signals
