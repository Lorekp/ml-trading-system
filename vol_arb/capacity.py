"""Empirical capacity analysis for the BTC vol-arb V2 strategy.

Produces (a) an analytic vega-absorption bound (Method 2) and (b) an
iso-Sharpe scaling backtest (Method 1) under a calibrated square-root
impact model. Both follow Donier-Bouchaud-Mastromatteo et al. (2015)
adapted to options vol-arb (arxiv:1602.03043), with circularity-break
calibration on a held-out time window.

Used by:
- ``scripts/capacity_phase1_vega_bound.py`` (Method 2, fast, ~1 day)
- ``scripts/capacity_phase2_iso_sharpe.py`` (Method 1, ~2-4 days)

Key formulas
------------
Square-root law for IV impact (per-trade, signed AGAINST the trade):
    ΔIV(K) = Y_vol · σ_σ · √(K · η)
where
    Y_vol   ≈ 0.5   (universal impact constant; sensitivity stress 0.25-1.0)
    σ_σ            = vol-of-vol (std of daily log-DVOL changes)
    η = Q_ν / V_ν  = our trade vega / market daily vega traded
    K              = scale multiplier vs current sizing

Vega-absorption bound (Method 2):
    K_max = (edge_per_trade / (2 · Y_vol · σ_σ))² / η_current
Solved per-trade then aggregated as a regime-stratified median.

Impact-adjusted PnL (Method 1):
    iv_exit_K[i] = iv_exit[i] + signal[i] · ΔIV(K)[i]
    pnl_K[i]     = signal[i] · vega[i] · (iv_exit_K[i] - iv_used[i])
                   + theta_pnl_nc[i] + delta_hedge_pnl_nc[i]
                   - option_fees_nc[i] - slippage_nc[i] - hedge_costs_nc[i]

For the 15-min holding window, impact is largely transient; with the
permanent-fraction parameter ``perm_frac`` (default 0.3 per Bouchaud
et al. 2023, arxiv:2311.18283) only that fraction of the IV displacement
persists to the exit fill, scaled by the bar-rate decay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ── Calibration ────────────────────────────────────────────────


def compute_sigma_sigma(
    dvol_close: pd.Series,
    window_days: int = 30,
    min_periods: int = 10,
) -> pd.Series:
    """Rolling vol-of-vol, defined as the std of daily log-DVOL increments.

    DVOL is the Deribit IV index, percentage points (e.g. 60.0 = 60%).
    Daily log-returns are computed by aligning to UTC midnight closes,
    then a ``window_days`` rolling std is returned (trailing window,
    no lookahead).
    """
    if dvol_close.index.tz is None:
        raise ValueError("dvol_close must have a tz-aware DatetimeIndex")
    daily = dvol_close.resample("1D").last().dropna()
    log_changes = np.log(daily).diff()
    sigma_sigma = log_changes.rolling(window_days, min_periods=min_periods).std()
    return sigma_sigma.dropna()


def compute_market_vega_proxy(
    trade_log: pd.DataFrame,
    bar_minutes: int = 15,
) -> pd.Series:
    """Daily market vega traded ($/vp) at the ATM, derived from the
    ``trailing_1h_volume`` column (BTC) × per-BTC vega ($/vp).

    This is a proxy — true per-strike Deribit volume requires Laevitas /
    Amberdata. For the V2 pipeline ``trailing_1h_volume`` is a 1-hour
    rolling sum of ATM-band option volume in BTC contracts, so:

        market_vega_per_day ≈ (trailing_1h_volume / 1h) · 24h · vega_per_btc

    That is, trailing_1h_volume × 24 × vega.

    The proxy treats the market's traded vega as homogeneously ATM,
    which OVERSTATES capacity (deeper strikes have less depth) but
    UNDERSTATES it for very tight ATM trading. We flag this in the
    report; the result must be interpreted as an order-of-magnitude
    bound.
    """
    if "trailing_1h_volume" not in trade_log.columns:
        raise KeyError("trade_log missing 'trailing_1h_volume'")
    if "vega" not in trade_log.columns:
        raise KeyError("trade_log missing 'vega'")
    return trade_log["trailing_1h_volume"] * 24.0 * trade_log["vega"]


def align_sigma_sigma_to_trades(
    sigma_sigma: pd.Series,
    trade_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Forward-fill sigma_sigma onto the bar index; backfill any leading
    NaNs with the median (trades before the rolling window finishes are
    rare; when they exist we need a defensible fallback)."""
    aligned = sigma_sigma.reindex(trade_index, method="ffill")
    if aligned.isna().any():
        aligned = aligned.fillna(sigma_sigma.median())
    return aligned.values


# ── Method 2: Vega-absorption bound ────────────────────────────


@dataclass
class CapacityBound:
    """Per-trade capacity result. K_max is the multiplier above current
    sizing at which the impact would consume ``edge_fraction`` of the
    trade's predicted edge."""
    K_max: np.ndarray              # shape (n_exec_trades,)
    eta_current: np.ndarray        # shape (n_exec_trades,)
    delta_iv_at_K1: np.ndarray     # vol points displacement at K=1


def compute_vega_capacity(
    trade_log: pd.DataFrame,
    sigma_sigma_per_trade: np.ndarray,
    Y_vol: float = 0.5,
    edge_fraction: float = 0.5,
    bar_minutes: int = 15,
    use_deployed_vega: bool = True,
) -> CapacityBound:
    """Method 2: per-trade square-root capacity, regime-aggregable.

    For each executed trade i, compute:
        η_i        = Q_ν(i) / V_ν(i)              [unitless]
        ΔIV_i(K)   = Y · σ_σ(i) · √(K · η_i)       [vol points]
        K_max(i)   = (|edge_i| · edge_fraction / (Y · σ_σ(i)))² / η_i

    If ``use_deployed_vega=True`` (the class default), Q_ν uses
    N_contracts_nc × vega — i.e. the deployed dollar-vega at canonical
    sizing. This makes K=1 correspond to "today's deployed scale", not
    "1 BTC of straddle". K=10 then means "10× the canonical per-trade
    deployed vega". The older `use_deployed_vega=False` path (Q_ν =
    unit vega per BTC) is kept for backward compatibility.

    Trades where V_ν proxy is zero, NaN, or |edge| ≤ 0 are masked
    (K_max = +inf for those — they don't constrain the aggregate
    median). Edge is in vol points (matches `df["edge"]`).

    Returns ``CapacityBound`` with per-trade arrays for downstream
    regime stratification.
    """
    exe_mask = trade_log["N_contracts_nc"] > 0
    df = trade_log.loc[exe_mask].copy()

    if use_deployed_vega:
        Q_nu = (df["N_contracts_nc"] * df["vega"]).values     # $/vp deployed
    else:
        Q_nu = df["vega"].values                              # $/vp per BTC
    V_nu = compute_market_vega_proxy(df).values               # $/vp per day, market-wide
    edge = np.abs(df["edge"].values)                          # vol points

    s = sigma_sigma_per_trade[exe_mask.values]
    if len(s) != len(df):
        raise ValueError(
            f"sigma_sigma length {len(s)} != exe trades {len(df)} after mask"
        )

    # Convert σ_σ from "log-DVOL daily std" to "vol points per √day"
    # by multiplying by current DVOL level (locally). DVOL is at iv_used*100.
    # Approximate: σ_σ × DVOL_level gives "absolute vol-pt change per day".
    # ATM IV stand-in: iv_used (vol-points). Use it as DVOL level proxy
    # since DVOL is the 30-day model-free IV — within ~5-10% of ATM 1m IV.
    dvol_level = df["iv_used"].values
    sigma_sigma_vp = s * dvol_level                            # vol-pt/day

    eta_current = np.where(V_nu > 0, Q_nu / V_nu, np.nan)
    delta_iv_at_K1 = Y_vol * sigma_sigma_vp * np.sqrt(eta_current)

    # Solve for K such that ΔIV(K) = edge_fraction · |edge|:
    #   K = (edge_fraction · edge / (Y · σ_σ_vp))² / η
    target = edge_fraction * edge / np.maximum(Y_vol * sigma_sigma_vp, 1e-9)
    K_max = (target ** 2) / np.maximum(eta_current, 1e-12)

    # Mask invalid trades (no edge, no market depth) to +inf so they
    # don't drag down the regime aggregate.
    invalid = (edge <= 0) | (V_nu <= 0) | ~np.isfinite(eta_current)
    K_max = np.where(invalid, np.inf, K_max)
    K_max = np.where(np.isfinite(K_max), K_max, np.inf)

    return CapacityBound(
        K_max=K_max,
        eta_current=eta_current,
        delta_iv_at_K1=delta_iv_at_K1,
    )


# ── Method 1: Impact-adjusted PnL replay ───────────────────────


def compute_impact_adjusted_pnl_nc(
    trade_log: pd.DataFrame,
    K: float,
    sigma_sigma_per_trade: np.ndarray,
    Y_vol: float = 0.5,
    perm_frac: float = 0.30,
    decay_seconds: float = 600.0,
    bar_minutes: int = 15,
    use_deployed_vega: bool = True,
) -> pd.Series:
    """Replay NC PnL at scale multiplier K with square-root impact.

    The exit IV is shifted AGAINST the trade by the residual portion of
    ΔIV_K after exponential decay over the bar holding window:

        ΔIV_residual = ΔIV_K · (perm_frac + (1 - perm_frac) · exp(-Δt/τ))
        iv_exit_adj  = iv_exit + signal · ΔIV_residual

    Then the vega PnL is recomputed; theta, delta-hedge, fees, slippage,
    and hedge costs are taken from the original trade log unchanged
    (they are not first-order functions of size scaling at this stage).

    If ``use_deployed_vega=True`` (the class default), η uses
    N_contracts_nc × vega, so K=1 = today's canonical deployed vega
    and K=10 = 10× the deployed sizing. Cycle-11 K1+slope is already
    baked into N_contracts_nc, so the K=1 baseline reflects the
    long-amplified profile.

    K=0 returns the original NC PnL (identity by construction —
    no-op test in the smoke suite). Y_vol=0 also returns identity.

    Returns the per-bar net_pnl_nc series; sum over executed bars to
    get the impact-adjusted NC PnL aggregate.
    """
    if K < 0:
        raise ValueError(f"K must be non-negative, got {K}")
    if not 0.0 <= perm_frac <= 1.0:
        raise ValueError(f"perm_frac must be in [0, 1], got {perm_frac}")

    df = trade_log
    exe_mask = (df["N_contracts_nc"] > 0).values
    signal = df["signal"].values
    vega = df["vega"].values
    N_nc = df["N_contracts_nc"].values
    iv_used = df["iv_used"].values
    iv_exit = df["iv_exit"].values
    if use_deployed_vega:
        Q_nu = N_nc * vega
    else:
        Q_nu = vega
    V_nu = compute_market_vega_proxy(df).values
    eta = np.where(V_nu > 0, Q_nu / V_nu, 0.0)

    sigma_sigma_vp = sigma_sigma_per_trade * iv_used
    delta_iv = Y_vol * sigma_sigma_vp * np.sqrt(np.maximum(K * eta, 0.0))

    # Holding-window decay: residual share at bar end
    holding_seconds = bar_minutes * 60.0
    decay_factor = perm_frac + (1.0 - perm_frac) * np.exp(
        -holding_seconds / max(decay_seconds, 1e-9)
    )
    delta_iv_residual = delta_iv * decay_factor

    # IV moves AGAINST the trade ⇒ exit IV adjusted by signed -signal:
    # signal=+1 (long): we paid up at entry, market reverts ⇒ exit_IV ↓
    # signal=-1 (short): we sold low, market reverts ⇒ exit_IV ↑
    iv_exit_adj = iv_exit - signal * delta_iv_residual

    # Recompute vega_pnl_nc at the original sizing — the IV displacement
    # is per-trade, baked into ΔIV_K. PnL = N_contracts × signal × vega
    # × (iv_exit_K − iv_used). Sizing scaling enters through K via the
    # impact magnitude only.
    dIV_adj = iv_exit_adj - iv_used
    vega_pnl_nc_adj = N_nc * signal * vega * dIV_adj

    # Reuse non-vega components from the trade log
    theta_nc = df["theta_pnl_nc"].values
    delta_nc = df["delta_hedge_pnl_nc"].values
    fees_nc = df["option_fees_nc"].values
    slip_nc = df["slippage_nc"].values
    hedge_nc = df["hedge_costs_nc"].values

    pnl = (
        vega_pnl_nc_adj
        + theta_nc + delta_nc
        - fees_nc - slip_nc - hedge_nc
    )
    pnl = np.where(exe_mask, pnl, 0.0)
    return pd.Series(pnl, index=df.index, name=f"net_pnl_nc_K{K:g}")


# ── Bootstrap CI ───────────────────────────────────────────────


def stationary_block_bootstrap_sharpe(
    trade_pnl_series: np.ndarray,
    n_years: float,
    block_size: int = 30,
    n_replications: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Stationary block bootstrap (Politis & Romano 1994) of NC trade
    Sharpe = mean(per-trade-pnl) / std × √(n_trades / years).

    The annualizer uses CALENDAR years (passed in by the caller from
    the trade-log index span), matching ``vol_arb.metrics.trade_sharpe``.

    ``block_size`` is the mean geometric block length in TRADE units
    (not bars). For 8000 trades over ~4 years a 30-trade block ≈ 0.5%
    of the sample — preserves short-run autocorrelation while giving
    enough independent draws for the CI.

    Returns (point_estimate, lcb_2.5, ucb_97.5).
    """
    rng = np.random.default_rng(seed)
    pnl = np.asarray(trade_pnl_series, dtype=float)
    pnl = pnl[pnl != 0]
    n = len(pnl)
    if n < 30 or n_years <= 0:
        return float("nan"), float("nan"), float("nan")

    p_geom = 1.0 / max(1, block_size)

    def _trade_sharpe(x):
        s = x.std(ddof=1)
        if s <= 1e-12:
            return 0.0
        return x.mean() / s * np.sqrt(len(x) / n_years)

    point = _trade_sharpe(pnl)
    boot = np.empty(n_replications, dtype=float)
    for r in range(n_replications):
        out = np.empty(n, dtype=float)
        i = 0
        while i < n:
            start = rng.integers(0, n)
            block_len = rng.geometric(p_geom)
            for j in range(block_len):
                if i >= n:
                    break
                out[i] = pnl[(start + j) % n]
                i += 1
        boot[r] = _trade_sharpe(out)

    lcb = float(np.percentile(boot, 2.5))
    ucb = float(np.percentile(boot, 97.5))
    return float(point), lcb, ucb


# ── Regime stratification ──────────────────────────────────────


def dvol_regime(dvol_close: pd.Series) -> pd.Series:
    """Tercile classification {low, mid, high} on DVOL close. Used to
    stratify capacity / Sharpe(K) curves so the headline number isn't
    dominated by a single regime."""
    q33, q67 = dvol_close.quantile([1 / 3, 2 / 3])
    out = pd.Series("mid", index=dvol_close.index, dtype=object)
    out[dvol_close < q33] = "low"
    out[dvol_close > q67] = "high"
    return out
