"""Position sizing strategies for the vol-arb backtester.

Nine strategies, all sharing the same base sizing formula and then
applying a multiplicative scale factor:

    N_base = equity * dynamic_margin / (im * spot)
    if short: N_base *= short_penalty
    N = N_base * strategy_scale
    N = min(N, participation_cap)

Strategies:
    baseline      — scale = 1.0 (current production)
    edge          — scale proportional to |edge| / rolling_median_edge
    vega_risk     — scale = target_risk / (vega_dollar * vol_of_iv)
    kelly         — scale from quarter-Kelly formula
    dd            — scale = f(drawdown from peak)
    composite     — edge_scale * dd_scale
    liq           — scale inversely with current execution cost
    cost_edge     — scale by net edge after execution cost
    composite_v2  — edge_scale * dd_scale * liq_scale

All rolling statistics use only past data (no lookahead). Parameters
are fixed ex-ante — no in-sample optimization.
"""

from __future__ import annotations

import numpy as np


def _base_size(
    equity: float,
    spot: float,
    signal: int,
    dynamic_margin: float,
    dynamic_btc_cap: float,
    im_fraction: float,
    short_penalty: float,
    im_override: float = 0.0,
) -> float:
    """Compute the base position size before strategy scaling.

    ``dynamic_margin`` is pre-computed by the backtester (DVOL-scaled,
    clipped). We just apply it here.

    If ``im_override > 0``, it replaces ``im_fraction`` — used for
    direction-dependent IM (long IM = premium, short IM = Deribit formula).
    """
    if equity <= 0 or spot <= 0:
        return 0.0
    im = im_override if im_override > 0 else im_fraction
    n = equity * dynamic_margin / (im * spot)
    if signal == -1:
        n *= short_penalty
    return min(n, dynamic_btc_cap)


def size_baseline(
    equity: float, spot: float, signal: int,
    dynamic_margin: float, dynamic_btc_cap: float, cfg, **kwargs,
) -> float:
    """S0: Current production sizing — flat margin fraction."""
    return _base_size(
        equity, spot, signal, dynamic_margin, dynamic_btc_cap,
        cfg.im_fraction, cfg.short_sizing_penalty,
        im_override=kwargs.get("im_override", 0.0),
    )


def size_edge_proportional(
    equity: float, spot: float, signal: int,
    dynamic_margin: float, dynamic_btc_cap: float, cfg,
    edge_vp: float = 0.0, rolling_median_edge: float = 1.0, **kwargs,
) -> float:
    """S1: Scale linearly with edge magnitude.

    edge_ratio = |edge| / rolling_median_edge, clipped to [floor, cap].
    Higher-conviction trades get more capital.
    """
    base = _base_size(
        equity, spot, signal, dynamic_margin, dynamic_btc_cap,
        cfg.im_fraction, cfg.short_sizing_penalty,
        im_override=kwargs.get("im_override", 0.0),
    )
    if rolling_median_edge <= 0:
        return base
    edge_ratio = edge_vp / rolling_median_edge
    scale = np.clip(edge_ratio, 0.25, cfg.sizing_edge_cap)
    return min(base * scale, dynamic_btc_cap)


def size_vega_risk_targeted(
    equity: float, spot: float, signal: int, dynamic_margin: float,
    dynamic_btc_cap: float, cfg,
    vega: float = 0.0, rolling_iv_std: float = 1.0, **kwargs,
) -> float:
    """S2: Target constant dollar risk per trade.

    N = target_risk / (vega_dollar * expected_iv_move).
    Auto-sizes down for high-gamma (short DTE) or volatile IV regimes.
    """
    target_risk = equity * cfg.sizing_risk_fraction
    vega_dollar = abs(vega) * spot * 100.0  # $ per 1 vol point
    if vega_dollar <= 0 or rolling_iv_std <= 0:
        return size_baseline(equity, spot, signal, dynamic_margin, dynamic_btc_cap, cfg)
    n = target_risk / (vega_dollar * rolling_iv_std)
    if signal == -1:
        n *= cfg.short_sizing_penalty
    return min(max(n, 0.0), dynamic_btc_cap)


def size_kelly(
    equity: float, spot: float, signal: int, dynamic_margin: float,
    dynamic_btc_cap: float, cfg,
    edge_vp: float = 0.0, rolling_win_rate: float = 0.5,
    rolling_payoff_ratio: float = 1.0, **kwargs,
) -> float:
    """S3: Quarter-Kelly sizing.

    kelly_f = win_prob - (1 - win_prob) / payoff_ratio
    N = base * kelly_fraction * kelly_f
    """
    base = _base_size(
        equity, spot, signal, dynamic_margin, dynamic_btc_cap,
        cfg.im_fraction, cfg.short_sizing_penalty,
        im_override=kwargs.get("im_override", 0.0),
    )
    w = rolling_win_rate
    r = max(rolling_payoff_ratio, 0.01)
    kelly_f = w - (1.0 - w) / r
    scale = cfg.sizing_kelly_fraction * kelly_f
    scale = np.clip(scale, 0.1, 2.0)
    return min(base * scale, dynamic_btc_cap)


def size_drawdown_conditioned(
    equity: float, spot: float, signal: int, dynamic_margin: float,
    dynamic_btc_cap: float, cfg,
    peak_equity: float = 0.0, **kwargs,
) -> float:
    """S4: Anti-martingale — reduce size during drawdowns.

    dd_scale = clip(1 + 2*dd, floor, 1.0) where dd <= 0.
    At -37.5% drawdown, scale = 0.25 (floor).
    """
    base = _base_size(
        equity, spot, signal, dynamic_margin, dynamic_btc_cap,
        cfg.im_fraction, cfg.short_sizing_penalty,
        im_override=kwargs.get("im_override", 0.0),
    )
    if peak_equity <= 0:
        return base
    dd = (equity - peak_equity) / peak_equity  # negative or zero
    dd_scale = np.clip(1.0 + 2.0 * dd, cfg.sizing_dd_floor, 1.0)
    return min(base * dd_scale, dynamic_btc_cap)


def size_composite(
    equity: float, spot: float, signal: int, dynamic_margin: float,
    dynamic_btc_cap: float, cfg,
    edge_vp: float = 0.0, rolling_median_edge: float = 1.0,
    peak_equity: float = 0.0, **kwargs,
) -> float:
    """S5: Edge-proportional * drawdown-conditioned.

    Multiplicative combination: overweight high-edge AND de-risk in DD.
    """
    base = _base_size(
        equity, spot, signal, dynamic_margin, dynamic_btc_cap,
        cfg.im_fraction, cfg.short_sizing_penalty,
        im_override=kwargs.get("im_override", 0.0),
    )
    # Edge scale
    if rolling_median_edge > 0:
        edge_ratio = edge_vp / rolling_median_edge
        edge_scale = np.clip(edge_ratio, 0.25, cfg.sizing_edge_cap)
    else:
        edge_scale = 1.0
    # DD scale
    if peak_equity > 0:
        dd = (equity - peak_equity) / peak_equity
        dd_scale = np.clip(1.0 + 2.0 * dd, cfg.sizing_dd_floor, 1.0)
    else:
        dd_scale = 1.0
    return min(base * edge_scale * dd_scale, dynamic_btc_cap)


def size_liquidity_conditioned(
    equity: float, spot: float, signal: int,
    dynamic_margin: float, dynamic_btc_cap: float, cfg,
    bar_taker_slip: float = 12.0,
    rolling_slip_ema: float = 12.0, **kwargs,
) -> float:
    """S6: Scale inversely with current execution cost.

    liq_scale = clip(rolling_slip_ema / bar_taker_slip, 0.25, cap)
    When slip is 2x median → trade 50% less. When 0.5x median → 2x more.
    Every signal still trades (floor=0.25).
    """
    base = _base_size(
        equity, spot, signal, dynamic_margin, dynamic_btc_cap,
        cfg.im_fraction, cfg.short_sizing_penalty,
        im_override=kwargs.get("im_override", 0.0),
    )
    if bar_taker_slip <= 0 or rolling_slip_ema <= 0:
        return base
    liq_scale = np.clip(rolling_slip_ema / bar_taker_slip, 0.25, cfg.sizing_liq_cap)
    return min(base * liq_scale, dynamic_btc_cap)


def size_cost_edge(
    equity: float, spot: float, signal: int,
    dynamic_margin: float, dynamic_btc_cap: float, cfg,
    edge_vp: float = 0.0, rolling_median_edge: float = 1.0,
    bar_taker_slip: float = 12.0,
    vega_per_pt: float = 0.0, **kwargs,
) -> float:
    """S7: Size proportional to net edge after execution cost.

    Converts per-bar execution cost to vol-point equivalent, subtracts
    from gross edge. Cartea-Jaimungal: optimal size ∝ net-of-cost signal.
    """
    base = _base_size(
        equity, spot, signal, dynamic_margin, dynamic_btc_cap,
        cfg.im_fraction, cfg.short_sizing_penalty,
        im_override=kwargs.get("im_override", 0.0),
    )
    if vega_per_pt <= 0 or spot <= 0 or rolling_median_edge <= 0:
        return base
    # Convert $/BTC taker cost to vol-point equivalent (2 legs)
    cost_vol_pts = bar_taker_slip * 2.0 / (vega_per_pt * spot)
    net_edge = max(edge_vp - cost_vol_pts, 0.01)
    scale = np.clip(net_edge / rolling_median_edge, 0.25, cfg.sizing_edge_cap)
    return min(base * scale, dynamic_btc_cap)


def size_composite_v2(
    equity: float, spot: float, signal: int,
    dynamic_margin: float, dynamic_btc_cap: float, cfg,
    edge_vp: float = 0.0, rolling_median_edge: float = 1.0,
    peak_equity: float = 0.0,
    bar_taker_slip: float = 12.0,
    rolling_slip_ema: float = 12.0, **kwargs,
) -> float:
    """S8: Edge × Drawdown × Liquidity.

    Three multiplicative dampeners. Each has floor=0.25, so worst case
    is ~1.5% of base (very conservative, but trade still fires).
    """
    base = _base_size(
        equity, spot, signal, dynamic_margin, dynamic_btc_cap,
        cfg.im_fraction, cfg.short_sizing_penalty,
        im_override=kwargs.get("im_override", 0.0),
    )
    # Edge scale (same as composite)
    if rolling_median_edge > 0:
        edge_scale = np.clip(edge_vp / rolling_median_edge, 0.25, cfg.sizing_edge_cap)
    else:
        edge_scale = 1.0
    # DD scale (same as composite)
    if peak_equity > 0:
        dd = (equity - peak_equity) / peak_equity
        dd_scale = np.clip(1.0 + 2.0 * dd, cfg.sizing_dd_floor, 1.0)
    else:
        dd_scale = 1.0
    # Liquidity scale (new)
    if bar_taker_slip > 0 and rolling_slip_ema > 0:
        liq_scale = np.clip(rolling_slip_ema / bar_taker_slip, 0.25, cfg.sizing_liq_cap)
    else:
        liq_scale = 1.0
    return min(base * edge_scale * dd_scale * liq_scale, dynamic_btc_cap)


# ── Dispatch ──────────────────────────────────────────────────────

_STRATEGIES = {
    "baseline": size_baseline,
    "edge": size_edge_proportional,
    "vega_risk": size_vega_risk_targeted,
    "kelly": size_kelly,
    "dd": size_drawdown_conditioned,
    "composite": size_composite,
    "liq": size_liquidity_conditioned,
    "cost_edge": size_cost_edge,
    "composite_v2": size_composite_v2,
}


def compute_position_size(strategy: str, **kwargs) -> float:
    """Dispatch to the named sizing strategy."""
    fn = _STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(
            f"Unknown sizing strategy '{strategy}'. "
            f"Available: {sorted(_STRATEGIES)}"
        )
    return fn(**kwargs)
