"""Byte-identity tests for the round-4 F6/F8 live spot-TSRV path.

The live engine must compute :func:`compute_bar_tsrv` byte-for-byte
the same way the offline TSRV pipeline does when fed the same Binance
Spot trade window. This is the venue-parity gate that closes F6/F8:

* **Live path** — Binance Spot aggTrades flow through the
  ``DollarBarSpotStream`` fan-out hook into a :class:`TradeBuffer`,
  then :class:`BarAggregator` slices a 15-min window into
  ``Bar.tsrv_trades``, then ``Bar.tsrv_dataframe()`` produces the
  ``[price, amount]`` DataFrame consumed by :func:`compute_bar_tsrv`.
* **Offline path** — :func:`TSRV.tsrv_series.compute_tsrv_window`
  consumes the same ``prices`` + ``dollar_volume`` arrays from the
  Binance Vision parquet directly, with the same threshold.

Both paths converge on :func:`compute_tsrv_window`. The byte-identity
gate is that the live path's :class:`TradeBuffer` -> :meth:`Bar.tsrv_dataframe`
-> :func:`compute_bar_tsrv` plumbing introduces no drift vs reading
the numpy arrays straight from parquet.

Tolerances: 1e-12 relative on every float field (float64 ULP from the
``prices * amounts`` reconstruction in :func:`compute_bar_tsrv` vs the
parquet's stored ``dollar_volume``). Integer fields exact-equal.

The test SKIPS automatically when the parquet isn't present (deploy
contexts without the data dir).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.tsrv_live import compute_bar_tsrv
from deribit_paper.market_data.aggregator import Bar, BarAggregator
from deribit_paper.market_data.buffer import (
    IndexPriceBuffer,
    Trade,
    TradeBuffer,
)


_PARQUET_PATH = Path(
    "data/binance-aggTrades-parquet/BTCUSDT/2026-04.parquet"
)
_DOLLAR_THRESHOLD = 50_000.0  # matches compute_bar_tsrv default
_BAR_SECONDS = 900.0


# ── Fixture loaders ─────────────────────────────────────────────────


def _load_chunk(n_trades: int = 200_000) -> pd.DataFrame:
    """Read the first ``n_trades`` rows of the test month's parquet.

    Returns columns: ``[transact_time(ms), price, quantity,
    dollar_volume, is_buyer_maker, agg_trade_id]``. Same shape the
    offline ``_build_bars_sequential`` uses.
    """
    if not _PARQUET_PATH.exists():
        pytest.skip(
            f"Binance aggTrades fixture not present at {_PARQUET_PATH}; "
            "this test only runs in dev environments with the data dir."
        )
    raw = pd.read_parquet(_PARQUET_PATH).head(n_trades)
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _select_window(df: pd.DataFrame, n_trades_in_window: int) -> pd.DataFrame:
    """Pick ``n_trades_in_window`` consecutive trades that fit in 15 min."""
    # Walk forward until we find a 15-min window with >= n trades
    bar_seconds = pd.Timedelta(seconds=_BAR_SECONDS)
    for start_idx in range(0, len(df) - n_trades_in_window):
        end_idx = start_idx + n_trades_in_window
        ts0 = df.loc[start_idx, "timestamp"]
        ts1 = df.loc[end_idx - 1, "timestamp"]
        if ts1 - ts0 <= bar_seconds:
            return df.iloc[start_idx:end_idx].reset_index(drop=True)
    # Fall back: take a smaller chunk
    return df.iloc[:1000].reset_index(drop=True)


# ── Path B (offline reference) ──────────────────────────────────────


def _offline_tsrv(window: pd.DataFrame) -> dict:
    """Call compute_tsrv_window on numpy arrays straight from parquet."""
    from TSRV.tsrv_series import compute_tsrv_window

    prices = window["price"].astype(np.float64).to_numpy()
    dollar_vol = window["dollar_volume"].astype(np.float64).to_numpy()
    return compute_tsrv_window(prices, dollar_vol, _DOLLAR_THRESHOLD)


# ── Path A (live) ────────────────────────────────────────────────────


def _live_via_buffer(window: pd.DataFrame) -> dict:
    """Push trades into TradeBuffer, slice window, run compute_bar_tsrv.

    Mirrors the round-4 F6/F8 flow exactly: each Binance Spot aggTrade
    becomes a :class:`Trade`, lands in :class:`TradeBuffer`, then
    :class:`BarAggregator` calls ``slice_window`` to populate
    ``Bar.tsrv_trades`` -> ``Bar.tsrv_dataframe()`` -> ``compute_bar_tsrv``.
    """
    buf = TradeBuffer(name="spot", window_seconds=3600.0)
    for i, row in window.iterrows():
        trade = Trade(
            trade_id=f"binance:{int(row['agg_trade_id'])}"
            if "agg_trade_id" in row else f"binance:{i}",
            timestamp=row["timestamp"],
            instrument_name="BTCUSDT",
            price=float(row["price"]),
            amount=float(row["quantity"]),
            direction=-1 if bool(row["is_buyer_maker"]) else 1,
        )
        buf.append(trade)

    # Build a Bar that the engine would see — pull all trades into the slice.
    ts_end = window["timestamp"].iloc[-1] + pd.Timedelta(milliseconds=1)
    ts_start = window["timestamp"].iloc[0] - pd.Timedelta(milliseconds=1)
    sliced = buf.slice_window(ts_start, ts_end)
    assert len(sliced) == len(window), (
        f"slice lost trades: {len(sliced)} vs {len(window)}"
    )

    bar = Bar(
        timestamp=ts_end,
        bar_minutes=15,
        perp_trades=[],   # not used by tsrv_dataframe when tsrv_trades is set
        option_trades=[],
        index_price=None,
        tsrv_trades=sliced,
    )
    bar_tsrv = compute_bar_tsrv(
        bar.tsrv_dataframe(),
        bar_seconds=_BAR_SECONDS,
        dollar_threshold_usd=_DOLLAR_THRESHOLD,
    )
    return bar_tsrv.to_dict()


# ── Test ────────────────────────────────────────────────────────────


def test_spot_tsrv_byte_identity_500_trades():
    """Live TradeBuffer -> compute_bar_tsrv == offline compute_tsrv_window
    on a 500-trade window from real Binance Vision aggTrades.
    """
    df = _load_chunk(n_trades=200_000)
    window = _select_window(df, n_trades_in_window=500)

    offline = _offline_tsrv(window)
    live = _live_via_buffer(window)

    # Integer fields: exact match.
    for k in ("n_ticks", "n_sub_bars"):
        assert int(live[k]) == int(offline[k]), (
            f"{k} mismatch: live={live[k]} offline={offline[k]}"
        )

    # Float fields: 1e-12 rel tol (float64 ULP from the
    # ``prices * amounts`` reconstruction). Note: ``vol_ann`` and
    # ``vol_ann_continuous`` are added by compute_bar_tsrv but not by
    # the bare compute_tsrv_window; skip them here — the variance fields
    # they're derived from are tested directly.
    LIVE_ONLY = {"vol_ann", "vol_ann_continuous"}
    for k in (
        "tsrv", "rv_fast", "rv_slow", "bpv", "medrv",
        "continuous_var", "jump_var", "jump_fraction",
        "rs_down", "rs_up", "down_jump_fraction", "up_jump_fraction",
        "n_dash",
    ):
        live_v = float(live[k])
        off_v = float(offline[k])
        if abs(off_v) < 1e-30:
            assert abs(live_v) < 1e-12, (
                f"{k}: live={live_v} offline={off_v} both should be ~0"
            )
        else:
            np.testing.assert_allclose(
                live_v, off_v, rtol=1e-12, atol=1e-15,
                err_msg=f"{k} not within float64 ULP tol",
            )

    # Live-only derived quantities should be self-consistent
    # (sqrt of variance * scale).
    seconds_per_year = 365.25 * 24 * 3600.0
    expected_vol_ann = float(np.sqrt(
        max(live["tsrv"], 0.0) * (seconds_per_year / _BAR_SECONDS)
    ))
    np.testing.assert_allclose(
        live["vol_ann"], expected_vol_ann,
        rtol=1e-12, atol=1e-15, err_msg="vol_ann self-consistency",
    )


def test_spot_tsrv_byte_identity_5000_trades():
    """Same byte-identity gate on a wider window (5000 trades)."""
    df = _load_chunk(n_trades=200_000)
    window = _select_window(df, n_trades_in_window=5000)

    offline = _offline_tsrv(window)
    live = _live_via_buffer(window)

    for k in (
        "tsrv", "rv_fast", "rv_slow", "bpv", "medrv",
        "continuous_var", "jump_var",
    ):
        live_v = float(live[k])
        off_v = float(offline[k])
        if abs(off_v) < 1e-30:
            assert abs(live_v) < 1e-12
        else:
            np.testing.assert_allclose(
                live_v, off_v, rtol=1e-12, atol=1e-15,
                err_msg=f"{k} not within tol on wider window",
            )


# ── Path-integration test (BarAggregator end-to-end) ────────────────


@pytest.mark.asyncio
async def test_bar_aggregator_routes_to_tsrv_buffer():
    """End-to-end: trades go into spot_trade_buffer, BarAggregator slices
    them into Bar.tsrv_trades, Bar.tsrv_dataframe drives compute_bar_tsrv.
    """
    df = _load_chunk(n_trades=200_000)
    window = _select_window(df, n_trades_in_window=300)

    perp_buf = TradeBuffer(name="perp", window_seconds=3600.0)
    option_buf = TradeBuffer(name="option", window_seconds=3600.0)
    index_buf = IndexPriceBuffer(window_seconds=3600.0)
    spot_buf = TradeBuffer(name="spot", window_seconds=3600.0)

    # Populate spot_buf only — perp_buf intentionally empty so the test
    # confirms the new tsrv_buffer path is being read, not perp.
    for i, row in window.iterrows():
        spot_buf.append(
            Trade(
                trade_id=f"binance:{int(row['agg_trade_id'])}",
                timestamp=row["timestamp"],
                instrument_name="BTCUSDT",
                price=float(row["price"]),
                amount=float(row["quantity"]),
                direction=-1 if bool(row["is_buyer_maker"]) else 1,
            )
        )

    agg = BarAggregator(
        bar_minutes=15,
        perp_buffer=perp_buf,
        option_buffer=option_buf,
        index_buffer=index_buf,
        tsrv_buffer=spot_buf,
    )

    captured: list[Bar] = []

    async def _capture(bar: Bar) -> None:
        captured.append(bar)

    agg.subscribe(_capture)

    # Pick a bar boundary just after the last trade timestamp.
    bar_ts = window["timestamp"].iloc[-1] + pd.Timedelta(seconds=1)
    await agg.on_bar_close(bar_ts)

    assert len(captured) == 1, "expected one bar"
    bar = captured[0]
    assert bar.tsrv_trades is not None, "tsrv_trades should be populated"
    assert len(bar.tsrv_trades) == len(window), (
        f"sliced {len(bar.tsrv_trades)} but expected {len(window)}"
    )
    # tsrv_dataframe should produce non-empty output
    df_tsrv = bar.tsrv_dataframe()
    assert not df_tsrv.empty
    assert set(df_tsrv.columns) >= {"timestamp", "price", "amount", "direction"}

    # And the TSRV output should match the offline byte-identity gate.
    offline = _offline_tsrv(window)
    bar_tsrv = compute_bar_tsrv(
        df_tsrv, bar_seconds=_BAR_SECONDS,
        dollar_threshold_usd=_DOLLAR_THRESHOLD,
    )
    np.testing.assert_allclose(
        bar_tsrv.tsrv, offline["tsrv"],
        rtol=1e-12, atol=1e-15,
        err_msg="bar.tsrv_dataframe()-driven tsrv != offline",
    )
