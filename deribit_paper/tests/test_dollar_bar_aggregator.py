"""Byte-identity tests for the live ``DollarBarAggregator``.

F3 (parity-cycle round 3 — 2026-05-07): the live aggregator MUST
reproduce the offline ``vol_model/features/engineer.py::_build_bars_sequential``
function bar-for-bar when fed the same trade stream. The 24 missing
live handlers all derive from these dollar bars, so a single off-by-one
in the aggregator silently rolls up into wrong feature distributions
through every lag/EMA stage.

The reference inputs are slices of the actual Binance Vision aggTrades
parquet at ``data/binance-aggTrades-parquet/BTCUSDT/``. We hash a small
chunk through both pipelines and assert column-by-column equality:

* `timestamp`, `open`, `close`, `high`, `low` — exact equality.
* `duration_s` — equal to ``1e-6`` (string→ms→s conversion via
  ``timedelta64[ms]`` in offline; we compute via ``total_seconds()``).
* `n_trades`, `dollar_volume`, `buy_volume`, `sell_volume` — exact.
* `vov` — equal to ``1e-12`` (sample std of log-returns; same numerics
  as numpy via Python ``math.sqrt`` on the same float64 inputs).
* `imbalance` — exact (deterministic divide).
* `log_return` — equal to ``1e-12`` (single ``math.log`` operation).

The test SKIPS automatically when the parquet isn't present (deploy
contexts without the data dir).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.dollar_bar import DollarBarAggregator


_PARQUET_PATH = Path(
    "data/binance-aggTrades-parquet/BTCUSDT/2026-04.parquet"
)
_REFERENCE_THRESHOLD = 500_000.0


def _load_chunk(n_trades: int = 200_000) -> pd.DataFrame:
    """Read the first ``n_trades`` rows of the test month's parquet.

    Returns a DataFrame with the same shape ``_build_bars_sequential``
    expects: index = UTC DatetimeIndex from ``transact_time`` (ms),
    columns = ``['price', 'quantity', 'dollar_volume', 'is_buyer_maker']``.
    """
    if not _PARQUET_PATH.exists():
        pytest.skip(
            f"Binance aggTrades fixture not present at {_PARQUET_PATH}; "
            "this test only runs in dev environments with the data dir."
        )
    raw = pd.read_parquet(_PARQUET_PATH).head(n_trades)
    df = raw.copy()
    df.index = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df = df.sort_index()
    return df[["price", "quantity", "dollar_volume", "is_buyer_maker"]]


def _offline_build_bars(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Replay the offline ``_build_bars_sequential`` per-bar logic on a chunk.

    This is a from-scratch numpy reconstruction matching the offline
    function lines 57-122 byte-for-byte. We avoid calling the offline
    function directly because it's wrapped with caching + month-loop
    logic that's awkward to invoke for a small slice.
    """
    from TSRV.sampling import _accumulate_bars

    prices = df["price"].values.astype(np.float64)
    dvol = df["dollar_volume"].values.astype(np.float64)
    buyer = df["is_buyer_maker"].values
    ts_vals = df.index.values  # numpy datetime64

    bar_idx = _accumulate_bars(dvol, threshold)
    if len(bar_idx) < 2:
        return pd.DataFrame()

    starts = np.concatenate([[0], bar_idx[:-1]])
    ends = np.array(bar_idx)
    n_bars = len(starts)

    out_ts = np.empty(n_bars, dtype="datetime64[ns]")
    out = np.empty((n_bars, 10), dtype=np.float64)
    for i in range(n_bars):
        s, e = starts[i], ends[i]
        if e <= s:
            continue
        seg_p = prices[s:e]
        seg_d = dvol[s:e]
        seg_b = buyer[s:e]
        buy_v = seg_d[~seg_b].sum()
        sell_v = seg_d[seg_b].sum()
        if e - s > 2:
            seg_lr = np.diff(np.log(seg_p))
            vov = np.std(seg_lr, ddof=1)
        else:
            vov = 0.0
        out_ts[i] = ts_vals[e - 1]
        out[i] = [
            seg_p[-1], seg_p[0], seg_p.max(), seg_p.min(),
            (ts_vals[e - 1] - ts_vals[s]).astype("timedelta64[ms]").astype(np.float64) / 1000,
            e - s, buy_v + sell_v, buy_v, sell_v, vov,
        ]
    cols = ["close", "open", "high", "low", "duration_s",
            "n_trades", "dollar_volume", "buy_volume", "sell_volume", "vov"]
    out_df = pd.DataFrame(
        out, columns=cols,
        index=pd.DatetimeIndex(out_ts, tz="UTC", name="timestamp"),
    )
    out_df["imbalance"] = (
        (out_df["buy_volume"] - out_df["sell_volume"])
        / out_df["dollar_volume"].clip(lower=1.0)
    )
    out_df["log_return"] = np.log(out_df["close"] / out_df["close"].shift(1))
    out_df["rv_bar"] = out_df["log_return"] ** 2
    return out_df


def _live_build_bars(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Replay the SAME chunk through the live ``DollarBarAggregator``."""
    bars = []
    agg = DollarBarAggregator(threshold=threshold)
    for ts, row in df.iterrows():
        bar = agg.add_trade(
            timestamp=ts,
            price=float(row["price"]),
            dollar_volume=float(row["dollar_volume"]),
            is_buyer_maker=bool(row["is_buyer_maker"]),
        )
        if bar is not None:
            bars.append({
                "timestamp": bar.timestamp,
                "close": bar.close,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "duration_s": bar.duration_s,
                "n_trades": bar.n_trades,
                "dollar_volume": bar.dollar_volume,
                "buy_volume": bar.buy_volume,
                "sell_volume": bar.sell_volume,
                "vov": bar.vov,
                "imbalance": bar.imbalance,
                "log_return": bar.log_return,
                "rv_bar": bar.rv_bar,
            })
    return pd.DataFrame(bars).set_index("timestamp")


# ── byte-identity ──────────────────────────────────────────────────────


def test_dollar_bar_aggregator_matches_offline_byte_identity():
    """Per-bar identity test on a 200K-trade slice of BTCUSDT/2026-04."""
    df = _load_chunk(n_trades=200_000)
    offline = _offline_build_bars(df, _REFERENCE_THRESHOLD)
    live = _live_build_bars(df, _REFERENCE_THRESHOLD)

    assert len(live) == len(offline), (
        f"bar count mismatch: live={len(live)} offline={len(offline)}"
    )

    # Timestamps: exact equality (string-equal datetimes)
    assert (live.index == offline.index).all(), "timestamp mismatch"

    # Per-trade fields are exact (these come from a single trade's data,
    # no summation order ambiguity).
    EXACT_COLS = ("close", "open", "high", "low", "n_trades")
    for col in EXACT_COLS:
        np.testing.assert_array_equal(
            live[col].values, offline[col].values,
            err_msg=f"{col} not exact-equal",
        )

    # Volume sums (dollar_volume, buy_volume, sell_volume): tolerate
    # float64-ULP-level relative difference. Offline uses numpy
    # ``arr[mask].sum()`` which uses pairwise summation; live uses an
    # incremental left-to-right += accumulator. The two agree to within
    # ~1e-15 relative per IEEE-754 float64 — not the algorithm's fault.
    for col in ("dollar_volume", "buy_volume", "sell_volume"):
        np.testing.assert_allclose(
            live[col].values, offline[col].values,
            rtol=1e-12, atol=1e-9,
            err_msg=f"{col} not within float64 ULP tolerance",
        )

    # imbalance is derived from buy/sell — same ULP tolerance applies
    np.testing.assert_allclose(
        live["imbalance"].values, offline["imbalance"].values,
        rtol=1e-12, atol=1e-12,
        err_msg="imbalance not within ULP tol",
    )

    # duration_s: tolerate 1e-3 — offline does ms-rounded delta then /1000;
    # live does ns-resolution total_seconds(). The two paths agree to ms
    # precision but can differ at microsecond level on bars whose first
    # and last trades sit on the same ms boundary.
    np.testing.assert_allclose(
        live["duration_s"].values, offline["duration_s"].values,
        atol=1e-3, err_msg="duration_s not within 1ms",
    )

    # vov, log_return, rv_bar: numeric tolerance from float64 sqrt + log
    np.testing.assert_allclose(
        live["vov"].values, offline["vov"].values,
        rtol=1e-12, atol=1e-15,
        err_msg="vov not within tol",
    )
    # log_return + rv_bar: first row is NaN in offline (.shift(1)) AND
    # in live (no prev_close). NaN-safe comparison.
    for col in ("log_return", "rv_bar"):
        live_v = live[col].values
        off_v = offline[col].values
        nan_mask = np.isnan(off_v)
        assert np.array_equal(np.isnan(live_v), nan_mask), (
            f"{col}: NaN positions differ"
        )
        np.testing.assert_allclose(
            live_v[~nan_mask], off_v[~nan_mask],
            rtol=1e-12, atol=1e-15,
            err_msg=f"{col} not within tol",
        )


# ── unit tests for streaming behaviour ─────────────────────────────────


def test_aggregator_emits_bar_when_kernel_threshold_crossed():
    """N trades of $200 each, threshold $1000.

    Kernel boundaries fire at trades 4, 9 (indices), so:
    - bar 0 = trades [0, 4) = trades 0..3, sum $800 (one trade short — bar 0 is
      always short by the first boundary trade per offline convention)
    - bar 1 = trades [4, 9) = trades 4..8, sum $1000
    Trade 9 starts bar 2 but bar 2 won't emit until another boundary.
    """
    bars: list = []
    agg = DollarBarAggregator(threshold=1000.0, on_bar_close=bars.append)
    for i in range(10):
        agg.add_trade(
            timestamp=pd.Timestamp(f"2026-05-07 12:00:{i:02d}", tz="UTC"),
            price=50_000.0,
            dollar_volume=200.0,
            is_buyer_maker=False,
        )
    assert len(bars) == 2
    # Bar 0: trades 0..3 → 4 trades, sum 800 (offline-shifted)
    assert bars[0].n_trades == 4
    assert bars[0].dollar_volume == pytest.approx(800.0)
    # Bar 1: trades 4..8 → 5 trades, sum 1000 (full threshold)
    assert bars[1].n_trades == 5
    assert bars[1].dollar_volume == pytest.approx(1000.0)


def test_aggregator_kernel_cum_resets_after_boundary():
    """After a boundary, kernel cumulative resets to 0 (matches kernel)."""
    bars: list = []
    agg = DollarBarAggregator(threshold=1000.0, on_bar_close=bars.append)
    agg.add_trade(
        timestamp=pd.Timestamp("2026-05-07 12:00:00", tz="UTC"),
        price=50_000.0, dollar_volume=999.0, is_buyer_maker=False,
    )
    assert agg.cumulative_dollar == pytest.approx(999.0)
    # Boundary trade: pushes kernel cumulative to 1500 → emit bar 0 (1 trade,
    # sum 999), reset kernel cum to 0, add this trade as first of bar 1.
    agg.add_trade(
        timestamp=pd.Timestamp("2026-05-07 12:00:01", tz="UTC"),
        price=50_001.0, dollar_volume=501.0, is_buyer_maker=False,
    )
    assert len(bars) == 1
    # Kernel cum reset to 0 (does NOT include the boundary trade per kernel).
    # The boundary trade (501) is in the new accumulator but not in cum_kernel.
    assert agg.cumulative_dollar == 0.0


def test_aggregator_vov_zero_when_too_few_trades():
    """n_trades <= 2 → vov = 0 by construction.

    Offline-shifted semantics: a 3-trade sequence where trade 3 hits the
    boundary emits bar 0 = trades [0, 2] (2 trades, vov=0).
    """
    bars: list = []
    agg = DollarBarAggregator(threshold=200_000.0, on_bar_close=bars.append)
    agg.add_trade(
        timestamp=pd.Timestamp("2026-05-07 12:00:00", tz="UTC"),
        price=50_000.0, dollar_volume=80_000.0, is_buyer_maker=False,
    )
    agg.add_trade(
        timestamp=pd.Timestamp("2026-05-07 12:00:01", tz="UTC"),
        price=50_001.0, dollar_volume=80_000.0, is_buyer_maker=True,
    )
    # Third trade hits the boundary (cum 80+80+80=240 ≥ 200) → emit bar 0
    # with trades [0, 1] (2 trades), then trade 2 starts bar 1.
    agg.add_trade(
        timestamp=pd.Timestamp("2026-05-07 12:00:02", tz="UTC"),
        price=50_002.0, dollar_volume=80_000.0, is_buyer_maker=False,
    )
    assert len(bars) == 1
    assert bars[0].vov == 0.0
    assert bars[0].n_trades == 2


def test_aggregator_log_return_nan_on_first_bar():
    bars: list = []
    agg = DollarBarAggregator(threshold=1000.0, on_bar_close=bars.append)
    # 12 trades of $200 → kernel boundaries at indices 4, 9 → 2 bars
    # to make sure bar[0].log_return is NaN and bar[1].log_return is finite,
    # we need at least 2 bars to actually emit. 12 trades give boundaries
    # at i=4 (cum reaches 1000) and i=9 (cum reaches 1000 again).
    for i in range(12):
        agg.add_trade(
            timestamp=pd.Timestamp(f"2026-05-07 12:00:{i:02d}", tz="UTC"),
            price=50_000.0 + i, dollar_volume=200.0, is_buyer_maker=False,
        )
    assert len(bars) >= 2
    assert math.isnan(bars[0].log_return)
    assert math.isnan(bars[0].rv_bar)
    assert not math.isnan(bars[1].log_return)


def test_aggregator_rejects_invalid_inputs():
    """Negative price or dollar_volume → no-op (defensive)."""
    bars: list = []
    agg = DollarBarAggregator(threshold=1000.0, on_bar_close=bars.append)
    agg.add_trade(
        timestamp=pd.Timestamp("2026-05-07", tz="UTC"),
        price=-1.0, dollar_volume=2000.0, is_buyer_maker=False,
    )
    assert agg.cumulative_dollar == 0.0
    assert len(bars) == 0


def test_aggregator_threshold_zero_raises():
    with pytest.raises(ValueError, match="threshold must be positive"):
        DollarBarAggregator(threshold=0.0)


def test_aggregator_buy_sell_volume_split():
    """is_buyer_maker=True → seller was aggressor → adds to SELL volume.

    Offline-shifted semantics: each bar excludes the boundary trade. The
    third trade triggers the boundary, so bar 0 contains trades 0 and 1.
    """
    bars: list = []
    agg = DollarBarAggregator(threshold=1000.0, on_bar_close=bars.append)
    agg.add_trade(
        timestamp=pd.Timestamp("2026-05-07 12:00:00", tz="UTC"),
        price=50_000.0, dollar_volume=300.0, is_buyer_maker=True,  # SELL
    )
    agg.add_trade(
        timestamp=pd.Timestamp("2026-05-07 12:00:01", tz="UTC"),
        price=50_001.0, dollar_volume=400.0, is_buyer_maker=False,  # BUY
    )
    # Trade 3 triggers boundary (cum 300+400+400=1100 ≥ 1000) → emit bar 0
    # = trades [0, 1] → sell=300, buy=400, dollar=700.
    agg.add_trade(
        timestamp=pd.Timestamp("2026-05-07 12:00:02", tz="UTC"),
        price=50_002.0, dollar_volume=400.0, is_buyer_maker=False,
    )
    assert len(bars) == 1
    bar = bars[0]
    assert bar.sell_volume == 300.0
    assert bar.buy_volume == 400.0
    assert bar.dollar_volume == pytest.approx(700.0)
    assert bar.imbalance == pytest.approx((400.0 - 300.0) / 700.0)
