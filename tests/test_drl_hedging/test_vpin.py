"""Unit tests for the option-flow VPIN proxy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_arb.drl_hedging.vpin import (
    compute_option_vpin_15m,
    compute_vpin_proxy_from_returns,
)


def _make_flow(n: int, direction_fn, base_ts="2024-01-01") -> pd.DataFrame:
    """Synthetic option trade flow with a caller-provided direction function."""
    rng = np.random.default_rng(0)
    ts = pd.date_range(base_ts, periods=n, freq="30s", tz="UTC")
    amt = rng.uniform(0.5, 3.0, n)
    dirn = np.array([direction_fn(i) for i in range(n)], dtype=np.int8)
    return pd.DataFrame({"timestamp": ts, "amount": amt, "direction": dirn})


def test_balanced_flow_vpin_much_lower_than_directional():
    """Perfectly alternating buy/sell must be *much* less toxic than all-buy.

    With bucket_volume=10 BTC and trade sizes ~U(0.5, 3) BTC, each bucket
    holds ~5-6 trades, so the imbalance on alternating flow has floor
    ~1/5 = 0.2 due to partial-bucket effects.  The signal we actually
    care about is balanced << directional, not balanced ≈ 0.
    """
    df_bal = _make_flow(2000, lambda i: 1 if i % 2 == 0 else -1)
    df_dir = _make_flow(2000, lambda i: 1)
    vpin_bal = compute_option_vpin_15m(df_bal, bucket_volume_btc=10.0, rolling_window=5).dropna()
    vpin_dir = compute_option_vpin_15m(df_dir, bucket_volume_btc=10.0, rolling_window=5).dropna()
    assert len(vpin_bal) > 0 and len(vpin_dir) > 0
    # Balanced should be < 0.30 AND at least 3x lower than directional
    assert vpin_bal.median() < 0.30, f"balanced VPIN too high: {vpin_bal.median():.3f}"
    assert vpin_dir.median() > 3 * vpin_bal.median(), (
        f"directional ({vpin_dir.median():.3f}) not separated from balanced ({vpin_bal.median():.3f})"
    )


def test_directional_flow_vpin_near_one():
    """All-buy flow should give VPIN ≈ 1."""
    df = _make_flow(2000, lambda i: 1)
    vpin = compute_option_vpin_15m(df, bucket_volume_btc=10.0, rolling_window=5)
    vpin_stable = vpin.dropna()
    assert vpin_stable.median() > 0.95, f"all-buy flow should be high VPIN, got median={vpin_stable.median():.3f}"


def test_empty_trades_returns_empty_series():
    df = pd.DataFrame(columns=["timestamp", "amount", "direction"])
    vpin = compute_option_vpin_15m(df)
    assert isinstance(vpin, pd.Series)
    assert len(vpin) == 0


def test_vpin_clipped_to_unit_interval():
    df = _make_flow(500, lambda i: 1 if (i // 20) % 2 == 0 else -1)
    vpin = compute_option_vpin_15m(df, bucket_volume_btc=5.0, rolling_window=3)
    assert (vpin.dropna() >= 0.0).all()
    assert (vpin.dropna() <= 1.0).all()


def test_returns_proxy_shape_and_bounds():
    idx = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"vol_ann": np.abs(rng.normal(0.8, 0.2, 100))}, index=idx)
    proxy = compute_vpin_proxy_from_returns(df, window=8)
    assert len(proxy) == 100
    assert (proxy >= 0.0).all() and (proxy <= 1.0).all()


def test_returns_proxy_rejects_missing_column():
    df = pd.DataFrame({"foo": [1.0, 2.0]})
    with pytest.raises(ValueError, match="vol_ann"):
        compute_vpin_proxy_from_returns(df)
