"""BS-decomposition of the realized straddle gross PnL (E4 — parity-cycle round 2).

Decomposes a closed straddle trade's GROSS (pre-fee, pre-slippage) PnL
into the V2 backtester's ``vega + theta + delta`` shape so the live
``profit_scale`` empirical-Bayes EMA sees the same per-trade input the
backtester does.

Pre-E4 the engine fed ``option_pnl_usd + hedge_pnl_usd`` (the actual
fill-difference) to ``ProfitScaleEstimator.update_realized``. That's
the ACTUAL realized number — but it implicitly contains taker
slippage and adverse-selection costs that ``ProfitScaleEstimator``
ALREADY subtracts separately via ``est_costs`` (full_slip + hedge +
option_fee). Net effect: those costs were double-subtracted, biasing
the live EMA below the V2 reference and pushing ``profit_scale``
toward its floor over time.

This module provides ``decompose_gross_pnl`` which mirrors V2's:

    vega_pnl_btc  = signal * vega_s     * (iv_exit - iv_entry)   # BS Greek × dIV
    theta_pnl_btc = signal * theta_s    * dt_days                # BS Greek × dt
    delta_pnl_btc = simulate_discrete_hedging(...)               # realized hedge

(see ``vol_arb/backtester.py:1555-1559, 2303-2305``)

For the live engine the V+T legs are computed directly from BS Greeks
at entry IV, and the delta leg uses the realized perp-hedge MTM the
``PerpHedger`` reports — that's the closest analog to the simulated
``delta_pnl_btc`` V2 uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from deribit_paper.execution.profit_scale import bsm_vega_per_vol_pt


_INV_SQRT_2PI: float = 1.0 / math.sqrt(2.0 * math.pi)


def bsm_theta_per_day(
    *,
    spot: float,
    strike: float,
    iv_decimal: float,
    tau_years: float,
) -> float:
    """Black-Scholes theta per **calendar day** for ONE option (call OR put).

    Returns the per-leg theta in USD per day. Negative for a long
    option (decays). For an ATM straddle the caller doubles this.

    Convention: r=q=0 (BS-on-forward-with-zero-drift, matches the rest
    of the engine which never books a discount factor). Under r=q=0
    the magnitudes are equal for ATM call vs put; the sign convention
    is "theta of long option" = ``-S * pdf(d1) * sigma / (2 * sqrt(T))``
    per year, divided by 365.25 for the per-day rate.

    Returns 0.0 on degenerate inputs (matches ``bsm_vega_per_vol_pt``).
    """
    if (
        spot <= 0.0
        or strike <= 0.0
        or iv_decimal <= 0.0
        or tau_years <= 0.0
    ):
        return 0.0
    sqrt_t = math.sqrt(tau_years)
    d1 = (
        math.log(spot / strike) + 0.5 * iv_decimal * iv_decimal * tau_years
    ) / (iv_decimal * sqrt_t)
    pdf_d1 = math.exp(-0.5 * d1 * d1) * _INV_SQRT_2PI
    theta_per_year = -spot * pdf_d1 * iv_decimal / (2.0 * sqrt_t)
    return float(theta_per_year / 365.25)


@dataclass
class GrossPnLDecomp:
    """Per-trade gross-PnL decomposition (USD totals, pre-fees).

    Total gross = vega + theta + delta. Mirrors V2's
    ``vega_pnl_btc + theta_pnl_btc + delta_pnl_btc`` shape, just in
    USD-per-trade rather than per-BTC.
    """
    vega_pnl_usd: float
    theta_pnl_usd: float
    delta_pnl_usd: float

    @property
    def total_usd(self) -> float:
        return self.vega_pnl_usd + self.theta_pnl_usd + self.delta_pnl_usd


def decompose_gross_pnl(
    *,
    signal_direction: int,
    entry_index_price: float,
    strike: float,
    entry_iv_decimal: float,
    exit_iv_decimal: float,
    entry_tau_years: float,
    holding_dt_days: float,
    n_btc: float,
    realized_hedge_pnl_usd: float,
) -> GrossPnLDecomp:
    """Decompose a closed straddle trade's gross PnL into vega+theta+delta.

    Mirrors ``vol_arb/backtester.py:1555-1559``:

        vega_pnl_btc  = signal * vega_s     * dIV
        theta_pnl_btc = signal * theta_s    * dt_days
        delta_pnl_btc = simulate(...)

    with USD scaling appropriate for the live engine's USD bookkeeping.

    Parameters
    ----------
    signal_direction : int
        ``+1`` for long straddle, ``-1`` for short straddle. ``0`` (or
        any other value) returns a zero decomp.
    entry_index_price : float
        Index spot at trade entry (USD).
    strike : float
        Straddle strike (USD).
    entry_iv_decimal : float
        ATM IV at entry, decimal (e.g. 0.75 for 75%).
    exit_iv_decimal : float
        ATM IV at exit, decimal. ``NaN`` is treated as the entry IV
        (zero vega contribution) — caller is responsible for fetching
        a live exit IV; this fallback prevents a degraded SVI calibrator
        from poisoning ``profit_scale``.
    entry_tau_years : float
        Time-to-expiry at entry, years. Greeks are computed at entry.
    holding_dt_days : float
        Holding period in days (calendar). Drives the theta leg.
    n_btc : float
        Trade size in BTC. ``0`` returns a zero decomp.
    realized_hedge_pnl_usd : float
        Realized perp-hedge MTM over the holding window (USD). This
        is V2's ``delta_pnl_btc`` analog scaled by ``n_btc * spot``.

    Returns
    -------
    GrossPnLDecomp
        ``(vega_pnl_usd, theta_pnl_usd, delta_pnl_usd)`` USD totals.
    """
    if signal_direction not in (1, -1) or n_btc <= 0.0:
        return GrossPnLDecomp(0.0, 0.0, 0.0)

    # Per-leg vega in USD per VOL POINT (1.0 == 1%)
    vega_per_pt_per_btc = bsm_vega_per_vol_pt(
        spot=float(entry_index_price),
        strike=float(strike),
        iv_decimal=float(entry_iv_decimal),
        tau_years=float(entry_tau_years),
    )
    # Per-leg theta in USD per DAY
    theta_per_day_per_btc = bsm_theta_per_day(
        spot=float(entry_index_price),
        strike=float(strike),
        iv_decimal=float(entry_iv_decimal),
        tau_years=float(entry_tau_years),
    )

    # Vega: signal × straddle_vega × dIV × n_btc
    # Straddle = call + put → 2 × per-leg vega.
    if math.isfinite(exit_iv_decimal) and exit_iv_decimal > 0.0:
        d_iv_pts = (
            float(exit_iv_decimal) - float(entry_iv_decimal)
        ) * 100.0  # decimal → vol pts (because vega_per_pt is per-pt)
    else:
        d_iv_pts = 0.0
    vega_pnl_usd = (
        float(signal_direction) * 2.0 * vega_per_pt_per_btc * d_iv_pts * n_btc
    )

    # Theta: signal × straddle_theta × dt_days × n_btc
    theta_pnl_usd = (
        float(signal_direction)
        * 2.0
        * theta_per_day_per_btc
        * float(holding_dt_days)
        * n_btc
    )

    # Delta: realized perp-hedge MTM (USD). This is the live analog of
    # V2's ``delta_pnl_btc`` scaled to USD per trade.
    delta_pnl_usd = float(realized_hedge_pnl_usd)
    if not math.isfinite(delta_pnl_usd):
        delta_pnl_usd = 0.0

    return GrossPnLDecomp(
        vega_pnl_usd=vega_pnl_usd,
        theta_pnl_usd=theta_pnl_usd,
        delta_pnl_usd=delta_pnl_usd,
    )


__all__ = [
    "GrossPnLDecomp",
    "bsm_theta_per_day",
    "decompose_gross_pnl",
]
