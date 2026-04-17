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
