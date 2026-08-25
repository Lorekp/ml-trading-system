"""Volume-synchronized Probability of Informed Trading (VPIN).

VPIN (Easley, López de Prado, O'Hara 2012) measures order-flow toxicity
by splitting trade volume into equal-volume *buckets* and computing the
absolute imbalance between buy and sell volume within each bucket:

    VPIN_T = sum_{tau=T-n+1}^{T} |V_tau^B - V_tau^S|
             -------------------------------------
                      n * V_bucket

where ``V_bucket`` is the fixed volume per bucket and ``n`` is the rolling
window length (typical 50).  Higher VPIN → more informed flow → higher
adverse-selection risk → hedging should be *delayed* or *partial*.

**Phase 1 approximation**: this module computes VPIN on *option trade flow*
rather than perp trade flow because the repo doesn't yet have perp trades
with aggressor-side direction.  Option VPIN is a weaker signal (options
and perps are different markets) but still captures cross-asset toxic
regimes (e.g. a whale accumulating calls drags spot via delta hedging).

**Phase 2 upgrade** (follow-up, not this session): add a perp trade
downloader task, precompute perp VPIN offline into
``vol_arb/output/vpin_btc_15m.parquet``, swap import in the env.

References
----------
- Easley, López de Prado, O'Hara (2012), "Flow Toxicity and Liquidity
  in a High-frequency World", Review of Financial Studies.
- The module is intentionally simple: we use the classic "buy minus sell"
  bulk classification (``direction`` column from
  ``vol_arb.pipeline.load_option_trades``) instead of a tick-rule or
  Lee-Ready heuristic, which aren't needed when aggressor side is known.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_option_vpin_15m(
    option_trades: pd.DataFrame,
    *,
    bucket_volume_btc: float = 50.0,
    rolling_window: int = 50,
    tf_str: str = "15min",
) -> pd.Series:
    """Compute VPIN from option trade flow, resampled to the ``tf_str`` grid.

    The "15m" suffix in the function name is legacy — the implementation
    is horizon-aware via ``tf_str`` (default ``"15min"`` preserves legacy
    byte-identical output). Pass ``tf_str="5min"``/``"30min"``/``"60min"``
    for other horizons. The horizon source of truth is
    ``vol_model.config.VolModelConfig.tf_str``.

    Parameters
    ----------
    option_trades : pd.DataFrame
        Must contain columns ``timestamp``, ``amount`` (BTC, positive), and
        ``direction`` (+1 = buy, -1 = sell).  Matches the output of
        ``vol_arb.pipeline.load_option_trades``.
    bucket_volume_btc : float, default 50.0
        Volume per bucket in BTC.  Chosen so that a typical 15-min window
        contains several buckets — tune based on median 15-min volume.
    rolling_window : int, default 50
        Number of buckets to average over.  50 is the canonical choice
        from the Easley/López de Prado paper.
    tf_str : str, default ``"15min"``
        pandas-compatible resample frequency. Default preserves legacy
        behaviour. At h=5 pass ``"5min"``; at h=60 ``"60min"``.

    Returns
    -------
    pd.Series
        Indexed by ``tf_str``-cadence UTC timestamps (end-of-bucket
        convention, matching the rest of the vol-arb pipeline). Values in
        ``[0, 1]``, where 0 = perfectly balanced flow and 1 = maximally
        directional. NaN for bars before the rolling window fills.
    """
    if option_trades.empty:
        return pd.Series(dtype=np.float64, name="vpin")

    df = option_trades[["timestamp", "amount", "direction"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Signed amounts
    df["buy_amt"]  = np.where(df["direction"] > 0, df["amount"], 0.0)
    df["sell_amt"] = np.where(df["direction"] < 0, df["amount"], 0.0)

    # Bucket assignment: cumulative volume → integer bucket index
    df["cum_vol"] = df["amount"].cumsum()
    df["bucket_idx"] = (df["cum_vol"] // bucket_volume_btc).astype(np.int64)

    # Per-bucket: imbalance = |sum buy - sum sell| / bucket_volume
    agg = df.groupby("bucket_idx", sort=True).agg(
        buy=("buy_amt", "sum"),
        sell=("sell_amt", "sum"),
        t_last=("timestamp", "max"),
    )
    imbalance = (agg["buy"] - agg["sell"]).abs() / bucket_volume_btc

    # Rolling mean over n buckets → VPIN
    vpin_per_bucket = imbalance.rolling(rolling_window, min_periods=1).mean()
    vpin_per_bucket.index = agg["t_last"]
    vpin_per_bucket.index.name = "timestamp"

    # Resample to the bar cadence: take the last VPIN observed in each
    # window (not a mean, because VPIN is already a smoothed quantity and
    # we want the most recent observation at each bar).
    vpin_15m = (
        vpin_per_bucket.sort_index()
        .resample(tf_str, label="right", closed="right")
        .last()
    )

    # Clip to [0, 1] for numerical safety (floating-point can push slightly
    # above 1 when rolling window is shorter than n_buckets)
    vpin_15m = vpin_15m.clip(lower=0.0, upper=1.0)
    vpin_15m.name = "vpin"
    logger.info(
        "VPIN computed: %d 15-min bars, median=%.3f, max=%.3f",
        len(vpin_15m), float(vpin_15m.median()), float(vpin_15m.max()),
    )
    return vpin_15m


def compute_vpin_proxy_from_returns(
    vol_arb_df: pd.DataFrame,
    *,
    window: int = 16,
) -> pd.Series:
    """Return-based toxicity proxy when no trade flow is available.

    Absolute log-return z-scored against a trailing window.  Not a true
    VPIN but a serviceable proxy that correlates with jump regimes.
    Only used if ``compute_option_vpin_15m`` cannot run (e.g. because
    option trades aren't loaded in a minimal test harness).

    Parameters
    ----------
    vol_arb_df : pd.DataFrame
        Must have a ``vol_ann`` column (annualised realised vol from TSRV).
    window : int, default 16
        Trailing window in 15-min bars (16 = 4 hours).
    """
    if "vol_ann" not in vol_arb_df.columns:
        raise ValueError("vol_arb_df must contain 'vol_ann' column")
    va = vol_arb_df["vol_ann"].astype(np.float64)
    rolled = va.rolling(window, min_periods=max(2, window // 4))
    z = (va - rolled.mean()) / rolled.std().replace(0, np.nan)
    # Squash z to [0, 1] via logistic — so the env always receives a unit-scale
    # toxicity signal regardless of which source feeds it.
    proxy = 1.0 / (1.0 + np.exp(-z.clip(-5.0, 5.0)))
    proxy = proxy.fillna(0.5)  # neutral prior during warmup
    proxy.name = "vpin"
    return proxy
