"""Tests for live TSRV wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.tsrv_live import BarTSRV, compute_bar_tsrv


def _make_trades(n: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic perp trade DataFrame for testing."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-04-10 12:00:00", periods=n, freq="1s", tz="UTC")
    # Log-normal walk around $50k with ~1% vol
    log_p = np.cumsum(rng.normal(0, 0.001, n))
    prices = 50_000.0 * np.exp(log_p)
    amounts = rng.uniform(0.01, 2.0, n)
    return pd.DataFrame({
        "timestamp": ts,
        "price": prices,
        "amount": amounts,
    })


def test_empty_dataframe_returns_zero():
    result = compute_bar_tsrv(pd.DataFrame())
    assert isinstance(result, BarTSRV)
    assert result.tsrv == 0.0
    assert result.vol_ann == 0.0
    assert result.n_ticks == 0


def test_single_trade_returns_zero_variance():
    df = _make_trades(1)
    result = compute_bar_tsrv(df)
    assert result.tsrv == 0.0
    assert result.n_ticks == 1


def test_realistic_bar_produces_finite_tsrv():
    df = _make_trades(500)
    result = compute_bar_tsrv(df, bar_seconds=900.0)
    assert result.n_ticks == 500
    assert np.isfinite(result.tsrv)
    assert result.tsrv >= 0.0
    assert np.isfinite(result.vol_ann)
    assert result.vol_ann > 0.0  # non-zero for 1% random walk
    # Annualised vol should be in a plausible range (not 0, not 10x reality)
    assert 0.01 < result.vol_ann < 50.0


def test_vol_ann_scaling_by_bar_seconds():
    """Doubling the bar_seconds should halve the annualisation multiplier → 1/sqrt(2)*vol."""
    df = _make_trades(1000)
    r_900 = compute_bar_tsrv(df, bar_seconds=900.0)
    r_1800 = compute_bar_tsrv(df, bar_seconds=1800.0)
    # Same variance, double the horizon → vol_ann scales with 1/sqrt(2)
    if r_900.vol_ann > 0:
        ratio = r_1800.vol_ann / r_900.vol_ann
        assert abs(ratio - (1.0 / np.sqrt(2))) < 0.01


def test_jump_decomposition_sums():
    """continuous_var + jump_var ≈ rv_slow (within floating-point tolerance)."""
    df = _make_trades(800, seed=7)
    result = compute_bar_tsrv(df)
    total = result.continuous_var + result.jump_var
    assert abs(total - result.rv_slow) < 1e-10 * max(1.0, abs(result.rv_slow))


def test_jump_fraction_in_unit_interval():
    df = _make_trades(600, seed=11)
    result = compute_bar_tsrv(df)
    assert 0.0 <= result.jump_fraction <= 1.0


def test_vol_ann_and_continuous_both_finite():
    """Both vol_ann and vol_ann_continuous should be finite + non-negative.

    Note: we can NOT assert vol_ann_continuous <= vol_ann in general
    because TSRV subtracts a noise-bias term (n_bar_avg / n) * rv_fast
    that continuous_var doesn't have, so continuous_var > tsrv is
    possible when the noise correction pulls TSRV below BPV.
    """
    df = _make_trades(1000, seed=13)
    result = compute_bar_tsrv(df)
    assert np.isfinite(result.vol_ann)
    assert np.isfinite(result.vol_ann_continuous)
    assert result.vol_ann >= 0.0
    assert result.vol_ann_continuous >= 0.0


def test_to_dict_round_trip():
    df = _make_trades(200)
    result = compute_bar_tsrv(df)
    as_dict = result.to_dict()
    assert "tsrv" in as_dict
    assert "vol_ann" in as_dict
    assert "n_ticks" in as_dict
    assert as_dict["n_ticks"] == 200


def test_unsorted_input_is_sorted():
    """The wrapper should sort by timestamp before computing."""
    df = _make_trades(100, seed=99).sample(frac=1.0, random_state=0)
    # Shuffled order → should produce the same result as sorted
    sorted_df = df.sort_values("timestamp").reset_index(drop=True)
    assert compute_bar_tsrv(df).tsrv == pytest.approx(
        compute_bar_tsrv(sorted_df).tsrv,
        rel=1e-12,
    )
