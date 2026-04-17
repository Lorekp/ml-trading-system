"""Continuous market impact model calibrated from Deribit ATM option trades.

Replaces the legacy 4x3 slippage grid with a continuous sqrt model:

    slip_usd_per_btc = alpha(t) + beta(t) * sqrt(N)

where alpha and beta are daily time series calibrated by
``scripts/calibrate_impact.py`` from real trade data.

Volume conditioning scales impact inversely with trailing liquidity:
when the market is half as liquid as normal, slippage doubles.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ImpactModel:
    """Continuous sqrt slippage model with volume conditioning."""

    def __init__(self, params_path: str | Path):
        p = pd.read_parquet(params_path)
        self._alpha = p["alpha"].values.astype(np.float64)
        self._beta = p["beta"].values.astype(np.float64)
        self._volumes = p["trailing_volume_1d"].values.astype(np.float64)
        # Convert daily volume to hourly for matching with trailing_1h_volume
        self._vol_median_hourly = float(np.nanmedian(self._volumes)) / 24.0
        self._timestamps = p["timestamp"].values
        self._n = len(p)
        logger.info(
            "ImpactModel loaded: %d days, alpha=[%.1f, %.1f], "
            "beta=[%.2f, %.2f], vol_median=%.0f BTC/hr",
            self._n,
            float(np.nanmin(self._alpha)), float(np.nanmax(self._alpha)),
            float(np.nanmin(self._beta)), float(np.nanmax(self._beta)),
            self._vol_median_hourly,
        )

    def slippage_usd_per_btc(
        self,
        size_btc: float,
        date_idx: int,
        trailing_volume: float | None = None,
    ) -> float:
        """Per-BTC slippage for a given order size and date.

        Parameters
        ----------
        size_btc : float
            Order size per leg in BTC.
        date_idx : int
            Index into the calibration array (aligned to the backtester's
            bar array via ``align_to_bars``).
        trailing_volume : float, optional
            Trailing daily ATM option volume in BTC. If provided, scales
            impact inversely: thin markets cost more.
        """
        idx = min(max(date_idx, 0), self._n - 1)
        alpha = self._alpha[idx]
        beta = self._beta[idx]
        base = alpha + beta * np.sqrt(max(size_btc, 0.0))

        if trailing_volume is not None and trailing_volume > 0:
            vol_ratio = self._vol_median_hourly / trailing_volume
            base *= np.clip(vol_ratio, 0.5, 3.0)

        return max(base, 0.0)

    def straddle_slippage(
        self,
        n_btc: float,
        date_idx: int,
        trailing_volume: float | None = None,
        jump_fraction: float = 0.0,
        spread_jump_k: float = 3.0,
    ) -> float:
        """Total USD slippage for an N-BTC straddle (2 legs).

        Includes the jump-shock multiplier: during liquidation cascades
        (high jump_fraction), spreads widen exponentially.
        """
        per_btc = self.slippage_usd_per_btc(n_btc, date_idx, trailing_volume)
        total = 2.0 * per_btc * n_btc
        if jump_fraction > 0:
            total *= np.exp(spread_jump_k * jump_fraction)
        return total

    def align_to_bars(self, bar_timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Map bar timestamps to calibration date indices.

        Returns an integer array where idx[i] is the calibration day
        index for bar i (backward lookup — uses the most recent
        calibration at or before each bar's date).
        """
        bar_dates = bar_timestamps.normalize().values
        cal_dates = self._timestamps
        indices = np.searchsorted(cal_dates, bar_dates, side="right") - 1
        return np.clip(indices, 0, self._n - 1)
