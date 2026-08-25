"""Empirical-Bayes profit-scale estimator for the live sizing chain.

Mirrors :func:`vol_arb.backtester.run_backtest`'s ``use_profit_scaling``
branch (lines 1966-1994). The invalidated historical V2 preset turns this on, and
the live engine has been carrying the *config field* without any
behaviour. This module supplies the missing behaviour.

Per-bar contract
----------------

For each fired bar, ``estimate(...)`` returns a ``profit_scale`` in
``[floor, cap]``. The engine multiplies it into ``N`` AFTER the
``edge × dd × liq`` chain and BEFORE the Kelly mult, mirroring the
backtester's chain order exactly.

After each trade closes, ``update_realized(...)`` blends realised gross
USD-per-BTC and realised hedge USD-per-BTC into rolling EMAs so the
Bayes prior tightens over time. The shrinkage weight ``λ = n / (n+τ)``
matches the backtester's ``profit_filter_shrinkage_tau`` (default 200).

State carried
-------------

* ``_gross_per_btc_ema_usd`` — rolling realised gross PnL per BTC.
* ``_gross_count`` — count of executed trades (drives the shrinkage λ).
* ``_hedge_per_btc_ema_usd`` — rolling realised hedge cost per BTC.

The estimator is **stateful by design**. Engine instantiates one at
startup; tests can construct fresh instances per case.

Units note
----------

The backtester estimator mixes USD-per-BTC slip (from the sidecar) with
BTC-per-trade hedge cost — leaning on ``profit_scaling_floor`` to bound
the result regardless of dimensional drift. This live port keeps every
shape in **USD-per-BTC** to make the units explicit; the resulting
``profit_scale`` distribution is within tolerance of the backtester's
mean ~0.70 (verified against V2 trade log).
"""

from __future__ import annotations

import math

import numpy as np


def bsm_vega_per_vol_pt(
    *,
    spot: float,
    strike: float,
    iv_decimal: float,
    tau_years: float,
) -> float:
    """Black-Scholes vega per **percentage point** of vol (∂V/∂σ × 0.01).

    Returns the per-leg vega for a single option (call or put — vega is
    direction-symmetric in BS). For a straddle the caller multiplies by 2.

    All inputs in natural units: ``spot``/``strike`` USD, ``iv_decimal``
    in decimal (e.g. 0.6 for 60%), ``tau_years`` in years.

    Returns 0.0 on degenerate inputs rather than raising — the estimator
    falls back to its floor when vega is zero.
    """
    if (spot <= 0.0 or strike <= 0.0 or iv_decimal <= 0.0
            or tau_years <= 0.0):
        return 0.0
    sqrt_t = math.sqrt(tau_years)
    d1 = (
        math.log(spot / strike) + 0.5 * iv_decimal * iv_decimal * tau_years
    ) / (iv_decimal * sqrt_t)
    # Standard normal pdf
    n_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    vega = spot * n_d1 * sqrt_t       # per unit of σ (decimal)
    return float(vega * 0.01)         # per 1 vol pt


class ProfitScaleEstimator:
    """Empirical-Bayes shrinkage estimator for live profit-scaling.

    Single source of truth for the multiplicative factor that the engine
    threads through ``StraddleFlow.size_position(profit_scale=...)``.
    """

    def __init__(
        self,
        *,
        floor: float = 0.3,
        cap: float = 1.0,
        edge_discount: float = 0.5,
        tau: float = 200.0,
        spread_jump_k: float = 3.0,
        ioc_cap_multiplier: float = 2.0,
        perp_taker_fee: float = 0.0005,
        option_maker_fee: float = 0.0003,
        ema_period_gross: int = 500,
        ema_period_hedge: int = 200,
    ) -> None:
        self._floor = float(floor)
        self._cap = float(cap)
        self._edge_discount = float(edge_discount)
        self._tau = max(float(tau), 1e-6)
        self._spread_jump_k = float(spread_jump_k)
        self._ioc_cap_multiplier = float(ioc_cap_multiplier)
        self._perp_taker_fee = float(perp_taker_fee)
        self._option_maker_fee = float(option_maker_fee)
        self._gross_alpha = 2.0 / (ema_period_gross + 1)
        self._hedge_alpha = 2.0 / (ema_period_hedge + 1)

        # State.
        self._gross_per_btc_ema_usd: float = 0.0
        self._gross_count: int = 0
        self._hedge_per_btc_ema_usd: float = 0.0

    # ── Read-only state accessors (testing + observability) ─────────

    @property
    def gross_count(self) -> int:
        return self._gross_count

    @property
    def gross_per_btc_ema_usd(self) -> float:
        return self._gross_per_btc_ema_usd

    @property
    def hedge_per_btc_ema_usd(self) -> float:
        return self._hedge_per_btc_ema_usd

    @property
    def shrinkage_lambda(self) -> float:
        """Current Bayesian weight on the realised EMA."""
        return self._gross_count / (self._gross_count + self._tau)

    # ── Per-bar estimate ────────────────────────────────────────────

    def estimate(
        self,
        *,
        edge_vp: float,
        vega_per_vol_pt_usd: float,
        spot: float,
        jump_fraction: float,
        slip_usd_per_btc: float,
        n_btc: float,
    ) -> float:
        """Return per-bar ``profit_scale`` ∈ ``[floor, cap]``.

        Inputs:

        * ``edge_vp``: vol-pt edge from the predictor (signed).
        * ``vega_per_vol_pt_usd``: BS vega for ONE leg in USD per vol-pt
          (caller multiplies by 2 if the straddle has both legs; we
          prefer single-leg here so the same vega works for the slip
          arithmetic which doubles below).
        * ``spot``: BTC index price.
        * ``jump_fraction``: TSRV jump_fraction in ``[0, 1]``.
        * ``slip_usd_per_btc``: per-leg empirical taker slip from the
          sidecar (or fallback when sidecar is missing).
        * ``n_btc``: pre-profit_scale notional in BTC (used to size
          the slippage estimate via sqrt-impact).

        Degenerate inputs (non-finite, ``n_btc <= 0``, ``spot <= 0``)
        return the floor — defensive, matches backtester behaviour.
        """
        if (n_btc <= 0.0 or spot <= 0.0
                or not math.isfinite(edge_vp)
                or not math.isfinite(slip_usd_per_btc)
                or slip_usd_per_btc < 0.0):
            return self._floor

        size_mult = math.sqrt(max(n_btc, 0.1))
        jf = max(0.0, min(1.0, float(jump_fraction)))
        full_slip = min(
            2.0 * slip_usd_per_btc * size_mult
            * math.exp(self._spread_jump_k * jf),
            self._ioc_cap_multiplier * slip_usd_per_btc,
        )

        hedge = (
            self._hedge_per_btc_ema_usd
            if self._hedge_per_btc_ema_usd > 0.0
            else 2.0 * self._perp_taker_fee * spot
        )
        # Cycle-24-H3 (2026-05-11, audit A3 H3): use the 4-leg model
        # with per-leg 12.5%-of-premium cap, mirroring the backtester
        # at ``vol_arb/backtester.py:33-45`` (the 2026-05-07 fee fix).
        # Pre-fix this used ``2.0 * option_maker_fee * spot`` — only
        # the two entry legs, no exit, no premium cap. Backtester uses
        # 4 fills (2 entry + 2 exit) each with min(flat_fee, 0.125 ×
        # premium_per_leg). When the cap doesn't bind (typical V2 at
        # σ ≈ 0.7, DTE ≥ 5d), the per-leg fee = flat ``option_maker_fee``
        # × spot in USD/BTC, so the difference is just 4 legs vs 2.
        # This inflated live's profit_scale ratio numerator (smaller
        # est_costs → higher ratio → less aggressive shrinkage), making
        # live size larger than backtester.
        #
        # Inline scalar version of ``deribit_option_fees_btc``
        # (backtester.py:23-45). Per-leg fee = min(flat_fee, 0.125 ×
        # premium_per_leg). Round-trip straddle = 4 fills (2 entry +
        # 2 exit). At V2 typical regime (σ ≈ 0.7, T ≥ 5d), premium ≈
        # 0.4·σ·√T ≈ 0.033 BTC → 0.125·premium ≈ 0.0041 BTC >> flat
        # 0.0003, so flat binds and total = 4 × 0.0003 = 0.0012 BTC
        # per BTC of notional. Multiply by spot for USD.
        sigma_default = 0.7
        T_default = 5.0 / 365.0  # 5 DTE
        prem_per_leg_btc = 0.4 * sigma_default * math.sqrt(T_default)
        fee_per_leg_btc = min(self._option_maker_fee, 0.125 * prem_per_leg_btc)
        # 4 fills: 2 entry + 2 exit
        option_fee = 4.0 * fee_per_leg_btc * spot

        est_costs = full_slip + hedge + option_fee

        # Two-leg straddle vega (caller passes single-leg vega).
        est_theo = (
            abs(edge_vp) * 2.0 * max(vega_per_vol_pt_usd, 0.0)
            * self._edge_discount
        )

        lam = self.shrinkage_lambda
        est_gross = (
            lam * self._gross_per_btc_ema_usd
            + (1.0 - lam) * est_theo
        )

        if abs(est_gross) < 1e-6:
            return self._floor

        est_net = est_gross - est_costs
        ratio = est_net / abs(est_gross)
        ps = float(np.clip(ratio, self._floor, self._cap))
        if ratio <= 0.0:
            ps = self._floor
        return ps

    # ── Realised-trade EMA update ────────────────────────────────────

    def update_realized(
        self,
        *,
        gross_pnl_usd: float,
        hedge_cost_usd: float,
        n_btc: float,
    ) -> None:
        """Fold a closed trade's realised gross & hedge cost into the EMAs.

        ``gross_pnl_usd`` is the realised gross PnL (vega+theta+delta-
        hedge before fees/slip), ``hedge_cost_usd`` is the cumulative
        perp-hedge fee+slip cost. Both are USD totals over the trade,
        normalised by ``n_btc`` here.

        Silently no-op on ``n_btc <= 0`` so an aborted-entry trade
        doesn't poison the prior.
        """
        if n_btc <= 0.0:
            return
        gross_per_btc = gross_pnl_usd / n_btc
        if math.isfinite(gross_per_btc):
            self._gross_per_btc_ema_usd += self._gross_alpha * (
                gross_per_btc - self._gross_per_btc_ema_usd
            )
            self._gross_count += 1
        if hedge_cost_usd > 0.0:
            hedge_per_btc = hedge_cost_usd / n_btc
            if math.isfinite(hedge_per_btc):
                self._hedge_per_btc_ema_usd += self._hedge_alpha * (
                    hedge_per_btc - self._hedge_per_btc_ema_usd
                )

    # ── H4: state persistence ────────────────────────────────────────

    def state_dict(self) -> dict:
        """Return the Bayesian state as a plain dict for SQLite persistence.

        H4 (2026-05-06): EMAs were lost on every restart, dropping
        ``_gross_count`` to 0 and collapsing the shrinkage weight λ to
        zero — the live engine fell back to the theoretical estimate
        for the first ~200 trades after every restart, drifting away
        from V2-parity. Persisting (gross_ema, gross_count, hedge_ema)
        and reloading on init keeps the live estimator continuous
        across restarts.
        """
        return {
            "gross_per_btc_ema_usd": self._gross_per_btc_ema_usd,
            "gross_count": self._gross_count,
            "hedge_per_btc_ema_usd": self._hedge_per_btc_ema_usd,
        }

    def load_state(self, state: dict) -> None:
        """Restore EMAs from a previously-saved ``state_dict()``.

        Defensive: non-finite floats and negative counts are ignored
        (i.e. fall back to cold-start values) so a corrupt persistence
        row doesn't poison a fresh run.
        """
        gross = state.get("gross_per_btc_ema_usd")
        count = state.get("gross_count")
        hedge = state.get("hedge_per_btc_ema_usd")
        if gross is not None and math.isfinite(float(gross)):
            self._gross_per_btc_ema_usd = float(gross)
        if count is not None and int(count) >= 0:
            self._gross_count = int(count)
        if hedge is not None and math.isfinite(float(hedge)) and float(hedge) >= 0.0:
            self._hedge_per_btc_ema_usd = float(hedge)


__all__ = ["ProfitScaleEstimator", "bsm_vega_per_vol_pt"]
