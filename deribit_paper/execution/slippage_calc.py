"""Live taker-slippage calculator for Deribit IOC entries.

Wraps the per-bar empirical estimate from ``vol_arb/sidecars.py``
(historical mainnet tape via ``bar_execution_quality.parquet``) with a
live-spread override clause: if the observed Deribit bid-ask spread is
materially wider than the sidecar's implied tolerance, fall back to the
observed spread instead of posting a limit that won't fill.

Result is a ``max_slippage_pct`` value sized for ``OrderManager.place_ioc_taker``.

C4 in plan ``C:\\Users\\laure\\.claude\\plans\\live-engine-h15-alignment-2026-04-26\\``.
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# When observed_spread_usd > _OVERRIDE_THRESHOLD x sidecar slip, prefer
# observed (testnet liquidity tends to be 5-50x mainnet; threshold 1.5x
# catches "materially wider" without overreacting to small drift).
_OVERRIDE_THRESHOLD: float = 1.5

# Hard floor: never request a tighter limit than this even if the
# sidecar says spreads are vanishingly tight. Below ~1bp, the cost of
# missing the fill outweighs the saved slippage.
_MIN_SLIPPAGE_PCT: float = 0.01

# Hard ceiling: never request more headroom than this even on testnet's
# wide-spread cases. Caller can override; default 1.0% is generous on
# Deribit options whose typical spread is 0.05-0.20%.
_MAX_SLIPPAGE_PCT: float = 1.0


def live_slippage_pct(
    *,
    slip_usd_per_btc: float,
    spot: float,
    observed_spread_usd: float | None = None,
    override_threshold: float = _OVERRIDE_THRESHOLD,
    fallback_pct: float = 0.5,
    floor_pct: float = _MIN_SLIPPAGE_PCT,
    cap_pct: float = _MAX_SLIPPAGE_PCT,
) -> float:
    """Convert sidecar slip-per-BTC and live spread into a ``max_slippage_pct``.

    Algorithm:
    1. Compute the sidecar-implied pct: ``100 * slip_usd_per_btc / spot``.
    2. If ``observed_spread_usd`` is supplied and exceeds ``override_threshold *
       sidecar_pct``, use ``100 * observed_spread_usd / spot`` instead.
    3. Clip the result to ``[floor_pct, cap_pct]``. The floor protects
       against stale / zero sidecar values; the cap protects against
       runaway widening when the override fires.

    Arguments:
        slip_usd_per_btc: Sidecar-derived empirical median slip per leg.
        spot: BTC index price (used for the USD→pct conversion).
        observed_spread_usd: Live Deribit bid-ask spread in USD, or
            None when not available. When None, the sidecar is the only
            source.
        override_threshold: Multiplier on the sidecar pct above which the
            observed spread takes over.
        fallback_pct: Returned when both ``slip_usd_per_btc`` is
            non-positive and ``observed_spread_usd`` is None — preserves
            the legacy live default of 0.5%.
        floor_pct, cap_pct: Final clip range.
    """
    sidecar_valid = (
        slip_usd_per_btc is not None
        and slip_usd_per_btc > 0.0
        and math.isfinite(slip_usd_per_btc)
    )
    observed_valid = (
        observed_spread_usd is not None
        and observed_spread_usd > 0.0
        and math.isfinite(observed_spread_usd)
    )

    if not sidecar_valid and not observed_valid:
        return fallback_pct

    sidecar_pct = (100.0 * slip_usd_per_btc / spot) if sidecar_valid else 0.0

    if observed_valid:
        observed_pct = 100.0 * observed_spread_usd / spot
        if observed_pct > override_threshold * sidecar_pct:
            logger.info(
                "Live slip override: observed=%.4f%% > %.1fx sidecar=%.4f%% — using observed",
                observed_pct, override_threshold, sidecar_pct,
            )
            return float(min(max(observed_pct, floor_pct), cap_pct))

    chosen_pct = sidecar_pct if sidecar_valid else fallback_pct
    return float(min(max(chosen_pct, floor_pct), cap_pct))


__all__ = ["live_slippage_pct"]
