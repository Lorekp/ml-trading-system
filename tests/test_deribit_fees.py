"""Unit tests for vol_arb/fees.py — Deribit fee primitives."""
import numpy as np
import pytest

from vol_arb import deribit_fees as fees


def test_option_fill_fee_cap():
    # ATM premium (0.03 BTC): flat 0.0003 dominates (0.125*0.03 = 0.00375 >> 0.0003)
    prem_atm = np.array([0.03, 0.05, 0.10])
    f = fees.option_fill_fee_btc(prem_atm)
    assert np.allclose(f, fees.OPT_TRADE_FEE_BTC)
    # Tiny premium: cap binds (0.125*prem < 0.0003)
    prem_tiny = np.array([0.001, 0.0005])
    f2 = fees.option_fill_fee_btc(prem_tiny)
    assert np.allclose(f2, fees.OPT_FEE_PREMIUM_CAP * prem_tiny)
    # boundary: premium where 0.125*prem == 0.0003 -> prem = 0.0024
    assert np.isclose(fees.option_fill_fee_btc(np.array([0.0024]))[0], 0.0003)


def test_straddle_roundtrip_fee_matches_honest_pnl():
    # Mirror honest_pnl.py:205-207: (fee(ec)+fee(ep)+fee(xc)+fee(xp))*N*index.
    ec = np.array([0.03]); ep = np.array([0.025])
    xc = np.array([0.02]); xp = np.array([0.018])
    index = np.array([40000.0]); N = np.array([1.5])
    got = fees.straddle_roundtrip_fee_usd(ec, ep, xc, xp, index, N)

    flat = fees.OPT_TRADE_FEE_BTC; cap = fees.OPT_FEE_PREMIUM_CAP
    def fee(p): return np.minimum(flat, cap * p)
    expect = (fee(ec) + fee(ep) + fee(xc) + fee(xp)) * N * index
    assert np.allclose(got, expect)


def test_perp_rebalance_fee():
    notional = np.array([10000.0, -5000.0])
    f = fees.perp_rebalance_fee_usd(notional, is_taker=True, slippage_bps=3.0)
    rate = fees.PERP_TAKER_FEE + 3.0 / 1e4
    assert np.allclose(f, np.abs(notional) * rate)
    # maker path = slippage only (maker fee 0)
    fm = fees.perp_rebalance_fee_usd(notional, is_taker=False, slippage_bps=3.0)
    assert np.allclose(fm, np.abs(notional) * (fees.PERP_MAKER_FEE + 3.0 / 1e4))


def test_delivery_fee_zero_for_0dte_and_15min():
    prem = np.array([0.03, 0.03]); index = np.array([40000.0, 40000.0]); N = np.array([1.0, 1.0])
    # 0DTE exempt
    f0 = fees.option_delivery_fee_usd(prem, index, N, is_0dte=np.array([True, True]),
                                      itm_mask=np.array([True, True]))
    assert np.allclose(f0, 0.0)
    # not ITM -> 0
    f1 = fees.option_delivery_fee_usd(prem, index, N, is_0dte=np.array([False, False]),
                                      itm_mask=np.array([False, False]))
    assert np.allclose(f1, 0.0)
    # ITM at expiry, not 0DTE -> min(0.00015, 0.125*prem)*N*index
    f2 = fees.option_delivery_fee_usd(prem, index, N, is_0dte=np.array([False, False]),
                                      itm_mask=np.array([True, True]))
    expect = np.minimum(fees.OPT_DELIVERY_FEE_BTC, fees.OPT_FEE_PREMIUM_CAP * prem) * N * index
    assert np.allclose(f2, expect)


def test_funding_sign():
    # +rate, net-LONG hedge (positive notional) => POSITIVE cost (longs pay shorts)
    long_notional = np.array([10000.0])
    rate = np.array([0.0001])
    assert fees.perp_funding_usd(long_notional, rate)[0] > 0
    # net-SHORT hedge (negative notional) + rate>0 => CREDIT (negative cost)
    short_notional = np.array([-10000.0])
    assert fees.perp_funding_usd(short_notional, rate)[0] < 0
    # exchange cut 0 -> exact product
    assert np.isclose(fees.perp_funding_usd(long_notional, rate)[0], 10000.0 * 0.0001)


def test_constants_match_schedule():
    assert fees.OPT_TRADE_FEE_BTC == 0.0003
    assert fees.OPT_FEE_PREMIUM_CAP == 0.125
    assert fees.OPT_DELIVERY_FEE_BTC == 0.00015
    assert fees.PERP_TAKER_FEE == 0.0005   # NOT the stale 0.0004
    assert fees.PERP_MAKER_FEE == 0.0
    assert fees.FUNDING_EXCHANGE_CUT == 0.0
