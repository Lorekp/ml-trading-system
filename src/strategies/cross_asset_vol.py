"""Strategy 4: Cross-Asset Vol Contagion — BTC vol leads alt vol."""

from __future__ import annotations
import numpy as np
import pandas as pd
from strategies.base import BaseVolStrategy


class CrossAssetVol(BaseVolStrategy):
    """
    When BTC vol spikes but alt vol hasn't followed yet,
    trade alts in BTC's direction (anticipate contagion).
    """

    name = "cross_asset_vol"

    def __init__(self, btc_spike_pct: float = 0.90, **kwargs):
        super().__init__(**kwargs)
        self.btc_spike_pct = btc_spike_pct

    def generate_signals(
        self,
        ohlcv: pd.DataFrame,
        predicted_vol: pd.Series,
        btc_ohlcv: pd.DataFrame | None = None,
        btc_predicted_vol: pd.Series | None = None,
    ) -> pd.DataFrame:
        """For cross-asset, also takes BTC data as reference."""
        close = ohlcv["close"]

        signals = pd.DataFrame(index=ohlcv.index)
        signals["direction"] = 0.0
        signals["size"] = 0.0
        signals["confidence"] = 0.0

        if btc_ohlcv is None or btc_predicted_vol is None:
            return signals

        btc_close = btc_ohlcv["close"].reindex(close.index, method="ffill")
        btc_vol = btc_predicted_vol.reindex(close.index, method="ffill")
        alt_vol = predicted_vol.reindex(close.index, method="ffill")

        # BTC vol spike: predicted vol > 90th percentile
        btc_vol_threshold = btc_vol.rolling(100).quantile(self.btc_spike_pct)
        btc_spiking = btc_vol > btc_vol_threshold

        # Alt vol NOT spiking: alt vol < 70th percentile (lag exists)
        alt_vol_threshold = alt_vol.rolling(100).quantile(0.70)
        alt_calm = alt_vol < alt_vol_threshold

        # Contagion window: BTC spiking AND alt still calm
        contagion = btc_spiking & alt_calm

        # BTC direction (3-bar trailing return, lagged 1 bar)
        btc_dir = np.sign(btc_close.pct_change(3)).shift(1).fillna(0)

        # Trade alts in BTC direction during contagion
        direction = contagion.astype(float) * btc_dir

        # Size: smaller (we're anticipating, not reacting)
        vol_median = alt_vol.rolling(100).median()
        size_mult = (vol_median / alt_vol).clip(0.2, 1.5).fillna(1.0)

        signals["direction"] = direction.shift(1).fillna(0)  # Extra lag for safety
        signals["size"] = signals["direction"].abs() * size_mult * 0.5  # Half-size for anticipatory trades
        signals["confidence"] = contagion.astype(float) * 0.4
        return signals
