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
    n_btc: float | None = None,
    jump_fraction: float | None = None,
    spread_jump_k: float = 3.0,
    ioc_cap_multiplier: float = 2.0,
) -> float:
    """Convert sidecar slip-per-BTC and live spread into a ``max_slippage_pct``.

    Algorithm:
    1. **D7 (parity 2026-05-07) — size + jump-stress impact**: when both
       ``n_btc`` and ``jump_fraction`` are supplied (the post-sizing
       call site has them), apply the V2 backtester impact formula
       (``vol_arb/backtester.py:1968-1973``):

           full_slip_usd = min(
               2 * slip_usd_per_btc * sqrt(max(n_btc, 0.1)) *
                   exp(spread_jump_k * jump_fraction),
               ioc_cap_multiplier * slip_usd_per_btc,
           )
           sidecar_pct = 100 * full_slip_usd / spot

       This makes the live IOC cap reflect both trade size (sqrt impact)
       and market stress (exp jump-fraction widening). The ``ioc_cap_multiplier``
       upper bound prevents runaway slippage requests on tail-event bars.
    2. Without n_btc/jump_fraction (the legacy / pre-sizing path), use
       the bare sidecar pct: ``100 * slip_usd_per_btc / spot``.
    3. If ``observed_spread_usd`` is supplied and exceeds ``override_threshold *
       sidecar_pct``, use ``100 * observed_spread_usd / spot`` instead.
    4. Clip the result to ``[floor_pct, cap_pct]``. The floor protects
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
        n_btc: Trade size in BTC. When supplied alongside
            ``jump_fraction`` the V2 size-impact formula applies.
        jump_fraction: Jump fraction in [0, 1]. When supplied alongside
            ``n_btc`` the V2 stress-impact formula applies.
        spread_jump_k, ioc_cap_multiplier: V2 impact-formula constants.
    """
    # Defensive: degenerate ticker (spot=0 or non-finite) → fallback.
    # Pre-E.1.9 the helper raised ZeroDivisionError on this path.
    if spot is None or not math.isfinite(spot) or spot <= 0.0:
        return fallback_pct

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

    # D7 (parity 2026-05-07): apply V2 impact formula when both
    # ``n_btc`` and ``jump_fraction`` are supplied. Mirrors V2 backtester
    # at backtester.py:1968-1973. ``max(n, 0.1)`` matches V2's lower
    # bound on size for the sqrt term (otherwise sqrt(0)=0 would zero
    # the impact estimate on dry/no-trade probes).
    if (
        sidecar_valid
        and n_btc is not None
        and jump_fraction is not None
        and math.isfinite(n_btc)
        and math.isfinite(jump_fraction)
    ):
        size_mult = math.sqrt(max(float(n_btc), 0.1))
        jf_clipped = max(0.0, min(1.0, float(jump_fraction)))
        full_slip_usd = min(
            2.0 * slip_usd_per_btc * size_mult * math.exp(spread_jump_k * jf_clipped),
            ioc_cap_multiplier * slip_usd_per_btc,
        )
        sidecar_pct = 100.0 * full_slip_usd / spot
    else:
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
