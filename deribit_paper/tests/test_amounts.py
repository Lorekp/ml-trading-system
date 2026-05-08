"""Tests for Deribit amount convention helpers."""

from __future__ import annotations

import pytest

from deribit_paper.execution.amounts import (
    OPTION_MIN_CONTRACT_BTC,
    PERP_MIN_USD,
    option_amount_btc,
    perp_amount_usd,
    round_to_increment,
    side_from_signed,
)


# ─── round_to_increment ──────────────────────────────────────────────

def test_round_to_increment_basic():
    assert round_to_increment(0.12, 0.1) == pytest.approx(0.1)
    assert round_to_increment(0.17, 0.1) == pytest.approx(0.2)
    assert round_to_increment(0.25, 0.1) == pytest.approx(0.2)  # banker's or half-up


def test_round_to_increment_preserves_sign():
    assert round_to_increment(-0.27, 0.1) == pytest.approx(-0.3)


def test_round_to_increment_zero():
    assert round_to_increment(0.0, 0.1) == 0.0


def test_round_to_increment_invalid_increment():
    assert round_to_increment(1.23, 0.0) == 1.23  # no-op when increment invalid


# ─── option_amount_btc ───────────────────────────────────────────────

def test_option_amount_below_min_returns_zero():
    assert option_amount_btc(0.05) == 0.0
    assert option_amount_btc(0.099) == 0.0


def test_option_amount_at_min_rounds_to_min():
    assert option_amount_btc(0.1) == pytest.approx(0.1)


def test_option_amount_rounds_to_nearest_increment():
    assert option_amount_btc(0.13) == pytest.approx(0.1)
    assert option_amount_btc(0.16) == pytest.approx(0.2)
    assert option_amount_btc(0.55) == pytest.approx(0.6)


def test_option_amount_always_positive():
    assert option_amount_btc(-0.5) == pytest.approx(0.5)  # sign stripped — it's the amount


def test_option_amount_large_values():
    assert option_amount_btc(5.23) == pytest.approx(5.2)
    assert option_amount_btc(100.0) == pytest.approx(100.0)


# ─── perp_amount_usd ─────────────────────────────────────────────────

def test_perp_amount_converts_btc_to_usd():
    # 0.5 BTC × $50k = $25,000
    result = perp_amount_usd(0.5, 50_000.0)
    assert result == pytest.approx(25_000.0)


def test_perp_amount_rounds_to_10_usd():
    # 0.00015 BTC × $50k = $7.50 → below min
    assert perp_amount_usd(0.00015, 50_000.0) == 0.0
    # 0.001 BTC × $50k = $50 → rounded to $50
    assert perp_amount_usd(0.001, 50_000.0) == pytest.approx(50.0)
    # 0.00023 BTC × $50k = $11.50 → rounded to $10
    assert perp_amount_usd(0.00023, 50_000.0) == pytest.approx(10.0)


def test_perp_amount_below_min_returns_zero():
    # 0.0001 BTC × $50k = $5 < $10 min
    assert perp_amount_usd(0.0001, 50_000.0) == 0.0


def test_perp_amount_sign_stripped():
    assert perp_amount_usd(-0.5, 50_000.0) == pytest.approx(25_000.0)


def test_perp_amount_zero_spot_returns_zero():
    assert perp_amount_usd(0.5, 0.0) == 0.0


def test_perp_amount_negative_spot_returns_zero():
    assert perp_amount_usd(0.5, -50_000.0) == 0.0


# ─── side_from_signed ───────────────────────────────────────────────

def test_side_from_signed_positive():
    assert side_from_signed(0.5) == "buy"


def test_side_from_signed_negative():
    assert side_from_signed(-0.3) == "sell"


def test_side_from_signed_zero_raises():
    with pytest.raises(ValueError, match="zero amount"):
        side_from_signed(0.0)


# ─── O-7 (parity-r3 cycle 8 — 2026-05-08): bias characterisation ───
# Round-3 audit O-7 flagged "perp $10-lot rounding accumulates one-
# sided drift across hedge ticks". Empirically the rounding logic
# (``round`` = banker's rounding) is unbiased on its own — the
# drift comes from the ``min_usd=10`` floor: notionals < 10 USD
# truncate to 0 (loss) but there's no symmetric gain because
# negative notionals are stripped of sign. Tracked inventory_btc
# in PerpHedger carries that residual to the next tick, so the
# drift is bounded — these tests lock the bound.


def test_perp_amount_increment_symmetric_around_midpoint():
    """An exact half-increment (5 USD past a multiple of 10) should
    round to the EVEN multiple per banker's rounding. Asserts ties
    don't always round one direction."""
    # 0.15 BTC × $100 = $15.0 → ties between 10 and 20; banker's → 20
    # 0.25 BTC × $100 = $25.0 → ties between 20 and 30; banker's → 20
    assert perp_amount_usd(0.15, 100.0) == pytest.approx(20.0)
    assert perp_amount_usd(0.25, 100.0) == pytest.approx(20.0)
    # The two values both round to 20 → no monotonic up/down bias
    # at the tie midpoint.


def test_perp_amount_rounding_bias_bounded_by_min_floor():
    """Above the ``min_usd=10`` floor, rounding to the 10-USD
    increment introduces sub-USD per-tick bias. The cumulative bias
    over N ticks is bounded by ~min_usd × N / (input_range / min_usd)
    in the worst case (uniform input on [10, 35]). This test locks
    the bound at 5000 USD over 10k ticks (empirical: ~3700 USD).
    Larger drifts indicate a regression in the rounding behaviour."""
    import numpy as np
    rng = np.random.default_rng(2026_05_08)
    spot = 70_000.0
    n_ticks = 10_000
    bias_usd = 0.0
    for _ in range(n_ticks):
        btc_delta = rng.uniform(-0.0005, 0.0005)
        exact_usd = abs(btc_delta) * spot
        if exact_usd < 10.0:
            continue
        rounded_usd = perp_amount_usd(btc_delta, spot)
        bias_usd += rounded_usd - exact_usd
    # Lock the empirical bound at 5000 USD; PerpHedger.run carries
    # the residual via tracked_inventory_btc (actual-fill-quantity
    # accounting), so this drift does NOT compound across trades.
    assert abs(bias_usd) < 5000.0, (
        f"Per-tick rounding bias {bias_usd:.2f} USD over {n_ticks} "
        f"ticks exceeds the documented bound (5000 USD) — regression."
    )
