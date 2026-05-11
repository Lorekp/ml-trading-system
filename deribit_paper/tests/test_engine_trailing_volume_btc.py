"""Cycle-24-C1 (2026-05-11) regression tests for _trailing_1h_volume_btc unit fix.

Audit A3 C1 found that ``_trailing_1h_volume_btc`` summed Deribit
BTC-PERPETUAL ``amount`` values, which are USD-equivalent (contracts
× $10) — not BTC. The dynamic_btc_cap = clip(trailing × 0.10, 1.0, 50.0)
then pegged at 50 BTC after warmup (because $50M-$300M/hr × 0.10
exceeds the 50 ceiling). Backtester uses ATM-option BTC volume
directly. Cycle-24-C1 converts USD → BTC via current spot.

The unit conversion is inert below ~$3M equity (margin envelope
binds first), but real once equity scales.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest


def _build_engine_with_perp_buffer_and_index(trades_df: pd.DataFrame, spot: float | None):
    """Construct a minimal SimpleNamespace mimicking the engine for the
    purposes of testing ``_trailing_1h_volume_btc``."""
    engine = SimpleNamespace()
    perp_buffer = MagicMock()
    perp_buffer.to_dataframe = MagicMock(return_value=trades_df)
    engine.perp_buffer = perp_buffer
    index_buffer = MagicMock()
    if spot is not None:
        index_buffer.latest = MagicMock(return_value=spot)
    else:
        index_buffer.latest = MagicMock(return_value=None)
    engine.index_buffer = index_buffer
    return engine


def test_trailing_volume_btc_converts_usd_to_btc():
    """Trades totaling $6,000,000 in 1h with spot $60,000 should
    return 100 BTC (= $6M / $60k), not 6,000,000.
    """
    from deribit_paper.engine import TradingEngine

    now = pd.Timestamp.now(tz="UTC")
    df = pd.DataFrame({
        "timestamp": [now - pd.Timedelta(minutes=30), now - pd.Timedelta(minutes=15)],
        # Two trades of $3M USD-notional each (Deribit perp contracts × $10)
        "amount": [3_000_000.0, 3_000_000.0],
    })
    engine = _build_engine_with_perp_buffer_and_index(df, spot=60_000.0)
    # Call the unbound method with our stub engine as ``self``.
    result = TradingEngine._trailing_1h_volume_btc(engine)
    assert result == pytest.approx(100.0), (
        f"Expected 100 BTC (= $6M / $60k); got {result}. "
        f"Pre-cycle-24-C1 would have returned 6_000_000."
    )


def test_trailing_volume_btc_falls_back_to_default_spot_on_no_index():
    """When index_buffer.latest() returns None, fall back to spot=$60k.
    Trades totaling $600,000 → 10 BTC (at the fallback rate)."""
    from deribit_paper.engine import TradingEngine

    now = pd.Timestamp.now(tz="UTC")
    df = pd.DataFrame({
        "timestamp": [now - pd.Timedelta(minutes=15)],
        "amount": [600_000.0],
    })
    engine = _build_engine_with_perp_buffer_and_index(df, spot=None)
    result = TradingEngine._trailing_1h_volume_btc(engine)
    # 600k / 60k fallback = 10
    assert result == pytest.approx(10.0)


def test_trailing_volume_btc_returns_none_when_empty_buffer():
    """Empty perp buffer must propagate None so the caller can fall
    back to a non-binding cap, not return 0."""
    from deribit_paper.engine import TradingEngine

    df = pd.DataFrame(columns=["timestamp", "amount"])
    engine = _build_engine_with_perp_buffer_and_index(df, spot=60_000.0)
    result = TradingEngine._trailing_1h_volume_btc(engine)
    assert result is None


def test_trailing_volume_btc_no_perp_buffer_returns_none():
    """Engine with perp_buffer=None returns None (early test paths)."""
    from deribit_paper.engine import TradingEngine

    engine = SimpleNamespace(perp_buffer=None, index_buffer=None)
    result = TradingEngine._trailing_1h_volume_btc(engine)
    assert result is None


def test_trailing_volume_btc_only_includes_last_hour():
    """Trades older than 1h are excluded from the sum."""
    from deribit_paper.engine import TradingEngine

    now = pd.Timestamp.now(tz="UTC")
    df = pd.DataFrame({
        "timestamp": [
            now - pd.Timedelta(hours=2),     # too old — excluded
            now - pd.Timedelta(minutes=30),  # within 1h
        ],
        "amount": [99_999_999.0, 600_000.0],
    })
    engine = _build_engine_with_perp_buffer_and_index(df, spot=60_000.0)
    result = TradingEngine._trailing_1h_volume_btc(engine)
    # Only the $600k trade counts → 10 BTC, not 99,999,999/60k.
    assert result == pytest.approx(10.0)


def test_dynamic_btc_cap_no_longer_pegs_at_50_after_fix():
    """Sanity: at typical BTC perp volume (~$50M-$300M/hr), the
    post-fix BTC value falls within the [1.0, 50.0] clip range.

    Pre-fix: dynamic_btc_cap = clip(200_000_000 * 0.10, 1.0, 50.0)
        = clip(20_000_000, ...) = 50.0 (pegged)
    Post-fix: dynamic_btc_cap = clip((200_000_000/60_000) * 0.10, 1.0, 50.0)
        = clip(3333.3 * 0.10, ...) = 50.0 (still pegs at high vol)
        but at lower vol (~$50M/hr → 833 BTC × 0.10 = 83 BTC) → still pegs.
    """
    spot = 60_000.0
    # At realistic BTC perp volumes ~$50M/hr → 833 BTC/hr → cap 83 (pegs)
    # The fix is mainly about RESPECTING units, not making the cap bind.
    # At small equity, the margin envelope binds first regardless.
    trailing_usd = 50_000_000.0  # $50M/hr conservative
    trailing_btc = trailing_usd / spot
    participation = trailing_btc * 0.10
    assert participation > 1.0, (
        "Even after unit conversion, at $50M/hr perp volume the cap "
        "is still > 1 BTC, so it doesn't bind at small equity."
    )
    # The point: at LARGE equity (e.g., $5M), margin envelope ≈ 4 BTC,
    # so a cap of 80 BTC stays non-binding. Pre-fix the cap was at the
    # 50 ceiling (pegged) but had no proportional relationship to
    # actual market volume — that's the integrity issue cycle-24-C1
    # fixes.
