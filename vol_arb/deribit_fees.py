"""Deribit / Binance fee primitives — pure, stateless, fully-vectorized.

NO pandas, NO IO. Every function takes np.ndarray (or scalars broadcastable
to them) and returns np.ndarray. These are the realistic-eval cost stack.

CONSTANTS — coin-margined BTC, corroborated against:
  * vol_arb/honest_pnl.py:54-55  (flat option fee 0.0003 BTC, 12.5% premium cap)
  * the realistic-eval map's web-research block (Deribit Insights + 3+ 2026
    third-party reviews; the support fee page 403s direct WebFetch).
  * deribit_paper/config.py live fee defaults.
TODO at deploy: confirm against the live API `public/get_instrument`
  {taker_commission, maker_commission, block_trade_commission} per instrument.

Sign / units conventions are documented per-function. All percentage fees are
fractions of NOTIONAL (underlying value), except where the 12.5%-of-premium cap
binds. For BTC contracts 1 option contract == 1 BTC underlying, so 0.03% of
underlying == 0.0003 BTC.
"""

from __future__ import annotations

import numpy as np

# ── Deribit fee constants (coin-margined BTC; June 2026). ──
OPT_TRADE_FEE_BTC = 0.0003      # 0.03% of underlying per option fill; MAKER == TAKER (no option rebate)
OPT_FEE_PREMIUM_CAP = 0.125     # per-fill fee <= 12.5% of THAT fill's premium
OPT_DELIVERY_FEE_BTC = 0.00015  # 0.015% of underlying on ITM expiry settle; same 12.5% cap; 0DTE exempt
PERP_TAKER_FEE = 0.0005         # 0.05% coin-margined BTC-PERPETUAL taker (NOT the stale 0.0004 in backtester.py:1007)
PERP_MAKER_FEE = 0.0            # 0.00% maker (no rebate, standard contract)
PERP_SLIPPAGE_BPS = 3.0         # MkrTkr60 perp ~3.9bps RT -> ~2-3bps one-way taker slippage
FUNDING_EXCHANGE_CUT = 0.0      # funding is peer-to-peer; exchange takes nothing


def bs_straddle_delta(S, K, T_years, sigma) -> np.ndarray:
    """Vectorized LONG straddle delta per BTC = 2*Phi(d1) - 1.

    d1 = (log(S/K) + 0.5*sigma^2*T) / (sigma*sqrt(T)), r = q = 0 (coin-margined,
    index-forward). T and sigma are floored to keep d1 finite; at/after expiry
    (T_years <= 0) the delta collapses to the intrinsic sign(S - K) (a long
    straddle held to expiry is +1 if the call is ITM, -1 if the put is ITM).

    S, K, T_years, sigma are broadcastable arrays (shape (n,) or (n, H)). Returns
    a value in [-1, 1] with the same broadcast shape.
    """
    T = np.maximum(np.asarray(T_years, float), 1.0 / (365.25 * 24 * 60))
    sig = np.maximum(np.asarray(sigma, float), 1e-6)
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    d1 = (np.log(S / K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
    from scipy.stats import norm
    out = 2.0 * norm.cdf(d1) - 1.0
    return np.where(np.asarray(T_years, float) <= 0.0, np.sign(S - K), out)


def option_fill_fee_btc(premium_btc: np.ndarray) -> np.ndarray:
    """Per-fill option fee in BTC = min(OPT_TRADE_FEE_BTC, OPT_FEE_PREMIUM_CAP*premium_btc).

    Caps against the REAL marked premium (NOT Brenner-Subrahmanyam 0.4*sigma*sqrt(T)
    as backtester.deribit_option_fees_btc:86 does). Vectorized; 1 element per fill.
    """
    prem = np.asarray(premium_btc, dtype=np.float64)
    return np.minimum(OPT_TRADE_FEE_BTC, OPT_FEE_PREMIUM_CAP * prem)


def straddle_roundtrip_fee_usd(
    prem_in_c: np.ndarray,
    prem_in_p: np.ndarray,
    prem_out_c: np.ndarray,
    prem_out_p: np.ndarray,
    index_price: np.ndarray,
    N_btc: np.ndarray,
) -> np.ndarray:
    """4-fill RT option fee in USD = (fee(ic)+fee(ip)+fee(xc)+fee(xp))*N*index.

    Each leg capped against its OWN real fill premium (BTC). Returns USD per
    trade (n,). Mirrors honest_pnl.py:205-207 exactly so the two agree.
    """
    fee = (
        option_fill_fee_btc(prem_in_c)
        + option_fill_fee_btc(prem_in_p)
        + option_fill_fee_btc(prem_out_c)
        + option_fill_fee_btc(prem_out_p)
    )
    return fee * np.asarray(N_btc, float) * np.asarray(index_price, float)


def option_delivery_fee_usd(
    premium_btc: np.ndarray,
    index_price: np.ndarray,
    N_btc: np.ndarray,
    is_0dte: np.ndarray,
    itm_mask: np.ndarray,
) -> np.ndarray:
    """min(OPT_DELIVERY_FEE_BTC, 0.125*premium)*N*index on legs held to expiry & ITM.

    0 where is_0dte | ~itm_mask. Hold-to-expiry VRP variant only; identically 0
    for 15-min holds (closed before expiry). Per-leg fee; caller passes the leg's
    premium. Returns USD per trade (n,).
    """
    prem = np.asarray(premium_btc, dtype=np.float64)
    fee_btc = np.minimum(OPT_DELIVERY_FEE_BTC, OPT_FEE_PREMIUM_CAP * prem)
    charged = (~np.asarray(is_0dte, bool)) & np.asarray(itm_mask, bool)
    fee_btc = np.where(charged, fee_btc, 0.0)
    return fee_btc * np.asarray(N_btc, float) * np.asarray(index_price, float)


def perp_rebalance_fee_usd(
    trade_notional_usd: np.ndarray,
    *,
    is_taker: bool = True,
    slippage_bps: float = PERP_SLIPPAGE_BPS,
) -> np.ndarray:
    """|notional|*((taker|maker fee) + slippage_bps/1e4). Vectorized.

    Charged on the initial hedge and on every band-breach rebalance. perp books
    are deep and the hedge is small, so slippage is a flat one-way bps.
    """
    fee_rate = (PERP_TAKER_FEE if is_taker else PERP_MAKER_FEE) + slippage_bps / 1e4
    return np.abs(np.asarray(trade_notional_usd, float)) * fee_rate


def perp_funding_usd(
    hedge_notional_usd: np.ndarray,
    funding_rate: np.ndarray,
) -> np.ndarray:
    """Funding carry per (trade, stamp) = hedge_notional_usd * rate.

    Sign: +rate => LONGS PAY SHORTS. ``hedge_notional_usd`` is SIGNED (positive
    for a net-LONG perp hedge). So net-LONG + rate>0 => POSITIVE number = a COST
    to be SUBTRACTED from PnL; net-SHORT + rate>0 => negative = a CREDIT.

    This is the per-element product; the caller does the searchsorted
    segmentation and the segmented reduce (np.add.reduceat) to sum over the
    funding stamps inside each trade's [t0, t1]. Exchange cut = 0.
    """
    return (
        np.asarray(hedge_notional_usd, float)
        * np.asarray(funding_rate, float)
        * (1.0 - FUNDING_EXCHANGE_CUT)
    )
