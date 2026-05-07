"""Live TSRV computation on the perp trade buffer.

Thin adapter over ``TSRV.tsrv_series.compute_tsrv_window``.  At each
bar close, extracts the prices and dollar volumes from the perp trade
DataFrame for the bar's window and computes the noise-corrected
realized variance plus jump decomposition.

Output fields match the batch ``tsrv_series`` output so the live row
can slot directly into the backtester's bar schema:

- ``tsrv`` — two-scale noise-corrected realized variance
- ``rv_fast`` / ``rv_slow`` — raw scale estimates
- ``bpv`` / ``medrv`` — jump-robust variance proxies
- ``continuous_var`` / ``jump_var`` — decomposition
- ``jump_fraction`` — share of variance from jumps
- ``rs_down`` / ``rs_up`` — signed semivariances
- ``down_jump_fraction`` / ``up_jump_fraction`` — signed jump shares
- ``n_ticks``, ``n_sub_bars``, ``n_dash`` — diagnostic counts
- ``vol_ann`` — annualised volatility = ``sqrt(continuous_var / dt_year)``
  where dt_year is the bar length in years (standard vol-arb convention)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from TSRV.tsrv_series import compute_tsrv_window

logger = logging.getLogger(__name__)


# Default dollar-bar threshold: 50k USD.  Matches the batch TSRV config
# default for BTC.  Can be overridden per-call.
DEFAULT_DOLLAR_THRESHOLD_USD: float = 50_000.0

# Seconds in a year — Zhang-style annualisation.  The batch pipeline
# uses the same value (365.25 * 24 * 3600).
_SECONDS_PER_YEAR: float = 365.25 * 24 * 3600


@dataclass
class BarTSRV:
    """All TSRV-derived fields for a single bar.

    Matches the dict returned by ``compute_tsrv_window`` plus the
    derived ``vol_ann`` and ``vol_ann_continuous`` fields that the
    backtester uses downstream.  Convertible to a dict with ``asdict``
    for attaching to a feature row.
    """
    tsrv: float
    rv_fast: float
    rv_slow: float
    bpv: float
    medrv: float
    continuous_var: float
    jump_var: float
    jump_fraction: float
    rs_down: float
    rs_up: float
    down_jump_fraction: float
    up_jump_fraction: float
    n_ticks: int
    n_sub_bars: int
    n_dash: float
    vol_ann: float              # annualised √tsrv
    vol_ann_continuous: float   # annualised √continuous_var (used by backtester)

    @property
    def rv_ann(self) -> float:
        """Annualised realised VARIANCE = ``vol_ann ** 2``.

        Round-4 follow-up (2026-05-08): the offline HAR-PD pipeline
        (vol_model/features/har_pd.py) reads a column called ``rv_ann``
        from the TSRV parquet. The live HAR-PD live-state buffer
        (``feature_assembler.py:_har_pd_state.append``) needs the same
        scalar — but the BarTSRV dataclass only carried ``vol_ann``
        (the square root). Pre-round-4 the live append ran inside a
        try/except that swallowed the AttributeError every bar with
        a warning ("HAR-PD append failed"); the buffer never advanced
        from its prewarm seed in production. Adding ``rv_ann`` as a
        derived property fixes the live HAR-PD circular buffer
        without changing the dataclass storage layout.
        """
        return float(self.vol_ann) ** 2

    def to_dict(self) -> dict:
        return asdict(self)


def compute_bar_tsrv(
    perp_trades: pd.DataFrame,
    *,
    bar_seconds: float = 900.0,
    dollar_threshold_usd: float = DEFAULT_DOLLAR_THRESHOLD_USD,
) -> BarTSRV:
    """Compute the TSRV block for a single bar's worth of perp trades.

    Parameters
    ----------
    perp_trades : pd.DataFrame
        Trades in the bar's window.  Must have columns ``price`` and
        ``amount`` (BTC).  Rows outside the window must already be
        filtered out by the caller (the aggregator does this via
        :meth:`deribit_paper.market_data.buffer.TradeBuffer.slice_window`).
    bar_seconds : float, default 900.0
        Bar duration in seconds.  15 minutes = 900 s.  Used to
        annualise the variance → ``vol_ann``.
    dollar_threshold_usd : float
        Sub-bar dollar threshold passed through to ``compute_tsrv_window``.
        Controls the K-subgrid averaging granularity.

    Returns
    -------
    BarTSRV
        All TSRV fields plus ``vol_ann`` and ``vol_ann_continuous``.
        Returns a zero-filled dataclass when the input is empty or
        has fewer than 2 ticks.
    """
    if perp_trades is None or perp_trades.empty:
        return _zero_bar_tsrv()

    if len(perp_trades) < 2:
        return _zero_bar_tsrv(n_ticks=len(perp_trades))

    # Sort by timestamp just in case the caller passed unsorted data.
    # Round-4 F6/F8 (parity-cycle 2026-05-07): stable mergesort so trades
    # tied on a millisecond timestamp keep their feed-order. Critical for
    # byte-identity vs the offline pipeline, which reads numpy arrays
    # straight from parquet (i.e. no resort, just feed order). Default
    # quicksort would reshuffle ties non-deterministically.
    if "timestamp" in perp_trades.columns:
        perp_trades = perp_trades.sort_values("timestamp", kind="mergesort")

    prices = perp_trades["price"].to_numpy(dtype=np.float64)
    amounts = perp_trades["amount"].to_numpy(dtype=np.float64)
    dollar_vol = prices * amounts

    raw = compute_tsrv_window(prices, dollar_vol, dollar_threshold_usd)

    # Annualisation: the bar's tsrv is realized VARIANCE over ``bar_seconds``.
    # To convert to annualised vol, multiply variance by (seconds_per_year / bar_seconds)
    # then take sqrt.  ``vol_ann`` uses the TSRV estimate, ``vol_ann_continuous``
    # uses the jump-free component (what the backtester's ``vol_ann`` column
    # contains — see ``vol_arb/pipeline.py`` + ``TSRV/tsrv_series.py``).
    ann_scale = _SECONDS_PER_YEAR / max(bar_seconds, 1e-9)
    vol_ann = float(np.sqrt(max(raw["tsrv"], 0.0) * ann_scale))
    vol_ann_continuous = float(np.sqrt(max(raw["continuous_var"], 0.0) * ann_scale))

    return BarTSRV(
        tsrv=float(raw["tsrv"]),
        rv_fast=float(raw["rv_fast"]),
        rv_slow=float(raw["rv_slow"]),
        bpv=float(raw["bpv"]),
        medrv=float(raw["medrv"]),
        continuous_var=float(raw["continuous_var"]),
        jump_var=float(raw["jump_var"]),
        jump_fraction=float(raw["jump_fraction"]),
        rs_down=float(raw["rs_down"]),
        rs_up=float(raw["rs_up"]),
        down_jump_fraction=float(raw["down_jump_fraction"]),
        up_jump_fraction=float(raw["up_jump_fraction"]),
        n_ticks=int(raw["n_ticks"]),
        n_sub_bars=int(raw["n_sub_bars"]),
        n_dash=float(raw["n_dash"]),
        vol_ann=vol_ann,
        vol_ann_continuous=vol_ann_continuous,
    )


def _zero_bar_tsrv(n_ticks: int = 0) -> BarTSRV:
    """Return a BarTSRV with all variance fields zeroed (empty or thin bar)."""
    return BarTSRV(
        tsrv=0.0, rv_fast=0.0, rv_slow=0.0, bpv=0.0, medrv=0.0,
        continuous_var=0.0, jump_var=0.0, jump_fraction=0.0,
        rs_down=0.0, rs_up=0.0, down_jump_fraction=0.0, up_jump_fraction=0.0,
        n_ticks=n_ticks, n_sub_bars=0, n_dash=0.0,
        vol_ann=0.0, vol_ann_continuous=0.0,
    )
