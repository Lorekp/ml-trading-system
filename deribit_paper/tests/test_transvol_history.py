"""S-1 (parity-r3 cycle 4 — 2026-05-08) regression tests for the live
TransVol sequence buffer.

The buffer accumulates 32 z-normalised feature rows + 2 time-feature
rows per bar so the encoder can run causal inference per bar. Pre-S-1
the live engine had no such buffer and passed
``transvol_embedding=None`` to the predictor, which silently filled
``tv_0..tv_15`` from a stale baseline row.

These tests exercise the buffer in isolation — the encoder inference
wrapper has its own test file (``test_transvol_inference.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.transvol_history import (
    TransVolHistoryBuffer,
    TransVolHistoryRow,
)


# ── Construction guards ──────────────────────────────────────────────


def test_constructor_rejects_zero_n_features():
    with pytest.raises(ValueError):
        TransVolHistoryBuffer(n_features=0)


def test_constructor_rejects_zero_seq_len():
    with pytest.raises(ValueError):
        TransVolHistoryBuffer(n_features=101, seq_len=0)


def test_constructor_rejects_max_size_below_seq_len():
    """A max_size smaller than seq_len would never let the buffer
    become inference-ready — the constructor catches this design bug."""
    with pytest.raises(ValueError):
        TransVolHistoryBuffer(n_features=101, seq_len=32, max_size=16)


def test_constructor_defaults_max_size_to_seq_len():
    buf = TransVolHistoryBuffer(n_features=101, seq_len=32)
    assert buf.max_size == 32
    assert buf.seq_len == 32
    assert buf.n_features == 101


# ── append / shape validation ────────────────────────────────────────


def _make_normalised_row(n_features: int, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(n_features).astype(np.float32),
        rng.standard_normal(2).astype(np.float32),
    )


def test_append_rejects_naive_timestamp():
    buf = TransVolHistoryBuffer(n_features=101)
    feat, time_feat = _make_normalised_row(101)
    with pytest.raises(TypeError):
        buf.append(pd.Timestamp("2026-05-08 12:00:00"), feat, time_feat)


def test_append_rejects_wrong_features_shape():
    buf = TransVolHistoryBuffer(n_features=101)
    feat = np.zeros(50, dtype=np.float32)
    time_feat = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError):
        buf.append(pd.Timestamp("2026-05-08", tz="UTC"), feat, time_feat)


def test_append_rejects_wrong_time_features_shape():
    buf = TransVolHistoryBuffer(n_features=101)
    feat = np.zeros(101, dtype=np.float32)
    time_feat = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError):
        buf.append(pd.Timestamp("2026-05-08", tz="UTC"), feat, time_feat)


def test_append_grows_buffer_until_max_size_then_evicts():
    buf = TransVolHistoryBuffer(n_features=101, seq_len=4, max_size=4)
    base = pd.Timestamp("2026-05-08 12:00:00", tz="UTC")
    for i in range(8):
        feat, time_feat = _make_normalised_row(101, seed=i)
        buf.append(base + pd.Timedelta(minutes=15 * i), feat, time_feat)
    assert len(buf) == 4
    # Last 4 timestamps preserved (eviction is FIFO).
    rows = list(buf)
    assert rows[0].timestamp == base + pd.Timedelta(minutes=15 * 4)
    assert rows[-1].timestamp == base + pd.Timedelta(minutes=15 * 7)


# ── inference-ready state ───────────────────────────────────────────


def test_buffer_not_inference_ready_when_underfull():
    buf = TransVolHistoryBuffer(n_features=101, seq_len=32)
    base = pd.Timestamp("2026-05-08 12:00:00", tz="UTC")
    for i in range(16):  # half-full
        feat, time_feat = _make_normalised_row(101, seed=i)
        buf.append(base + pd.Timedelta(minutes=15 * i), feat, time_feat)
    assert buf.is_inference_ready is False
    with pytest.raises(RuntimeError):
        buf.latest_window()


def test_buffer_becomes_inference_ready_at_seq_len():
    buf = TransVolHistoryBuffer(n_features=101, seq_len=32)
    base = pd.Timestamp("2026-05-08 12:00:00", tz="UTC")
    for i in range(31):
        feat, time_feat = _make_normalised_row(101, seed=i)
        buf.append(base + pd.Timedelta(minutes=15 * i), feat, time_feat)
    assert buf.is_inference_ready is False
    feat, time_feat = _make_normalised_row(101, seed=31)
    buf.append(base + pd.Timedelta(minutes=15 * 31), feat, time_feat)
    assert buf.is_inference_ready is True


# ── latest_window: shape + ordering + values ────────────────────────


def test_latest_window_returns_correct_shape():
    buf = TransVolHistoryBuffer(n_features=101, seq_len=32)
    base = pd.Timestamp("2026-05-08 12:00:00", tz="UTC")
    for i in range(40):  # overflow past seq_len
        feat, time_feat = _make_normalised_row(101, seed=i)
        buf.append(base + pd.Timedelta(minutes=15 * i), feat, time_feat)
    features, time_features = buf.latest_window()
    assert features.shape == (32, 101)
    assert time_features.shape == (32, 2)
    assert features.dtype == np.float32
    assert time_features.dtype == np.float32


def test_latest_window_returns_most_recent_seq_len_rows():
    """When the buffer holds more than seq_len rows, latest_window
    returns the LAST seq_len — preserves the causal-attention contract
    where the encoder's last token is the prediction bar."""
    buf = TransVolHistoryBuffer(n_features=4, seq_len=3, max_size=10)
    base = pd.Timestamp("2026-05-08 12:00:00", tz="UTC")
    for i in range(5):
        feat = np.array([i, i, i, i], dtype=np.float32)
        time_feat = np.array([i, i], dtype=np.float32)
        buf.append(base + pd.Timedelta(minutes=15 * i), feat, time_feat)
    features, time_features = buf.latest_window()
    # Most recent 3 rows have features [2,2,2,2], [3,3,3,3], [4,4,4,4]
    np.testing.assert_array_equal(features[0], [2, 2, 2, 2])
    np.testing.assert_array_equal(features[-1], [4, 4, 4, 4])
    np.testing.assert_array_equal(time_features[0], [2, 2])


def test_latest_timestamp_returns_most_recent_or_none():
    buf = TransVolHistoryBuffer(n_features=101)
    assert buf.latest_timestamp() is None
    base = pd.Timestamp("2026-05-08 12:00:00", tz="UTC")
    feat, time_feat = _make_normalised_row(101)
    buf.append(base, feat, time_feat)
    assert buf.latest_timestamp() == base
    feat2, time_feat2 = _make_normalised_row(101, seed=1)
    later = base + pd.Timedelta(minutes=15)
    buf.append(later, feat2, time_feat2)
    assert buf.latest_timestamp() == later


# ── prewarm_from_parquet ────────────────────────────────────────────


def test_prewarm_loads_n_rows_from_offline_parquet(tmp_path):
    """Prewarm reads the offline features parquet, normalises with the
    bundle's stats, and populates the buffer in chronological order."""
    n_features = 4
    seq_len = 3
    cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    time_cols = ("duration_s", "n_bars_in_window")
    base = pd.Timestamp("2026-05-08 12:00:00", tz="UTC")
    df = pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_c": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_d": [1.0, 2.0, 3.0, 4.0, 5.0],
            "duration_s": [10.0, 20.0, 30.0, 40.0, 50.0],
            "n_bars_in_window": [1, 2, 3, 4, 5],
        },
        index=pd.DatetimeIndex(
            [base + pd.Timedelta(minutes=15 * i) for i in range(5)],
            tz="UTC",
        ),
    )
    parquet_path = tmp_path / "feats.parquet"
    df.to_parquet(parquet_path)

    buf = TransVolHistoryBuffer(
        n_features=n_features, seq_len=seq_len, max_size=seq_len,
    )
    feat_mean = np.zeros(n_features, dtype=np.float32)
    feat_std = np.ones(n_features, dtype=np.float32)
    time_mean = np.array([0.0, 0.0], dtype=np.float32)
    time_std = np.array([1.0, 1.0], dtype=np.float32)
    n_loaded = buf.prewarm_from_parquet(
        features_parquet=parquet_path,
        feature_columns=cols,
        time_columns=time_cols,
        feat_mean=feat_mean, feat_std=feat_std,
        time_mean=time_mean, time_std=time_std,
    )
    assert n_loaded == seq_len
    assert buf.is_inference_ready
    features, time_features = buf.latest_window()
    # Last 3 rows have feat values [3,4,5]
    np.testing.assert_array_equal(features[:, 0], [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(time_features[:, 0], [30.0, 40.0, 50.0])


def test_prewarm_normalises_with_bundle_stats(tmp_path):
    """Verify normalisation: (x - mean) / std applied row-wise."""
    n_features = 2
    seq_len = 2
    base = pd.Timestamp("2026-05-08", tz="UTC")
    df = pd.DataFrame(
        {
            "feat_a": [10.0, 20.0],
            "feat_b": [4.0, 8.0],
            "duration_s": [100.0, 200.0],
            "n_bars_in_window": [1, 2],
        },
        index=pd.DatetimeIndex([base, base + pd.Timedelta(minutes=15)], tz="UTC"),
    )
    parquet_path = tmp_path / "feats.parquet"
    df.to_parquet(parquet_path)

    buf = TransVolHistoryBuffer(
        n_features=n_features, seq_len=seq_len, max_size=seq_len,
    )
    # Mean = 15, std = 5 → norm(10) = -1, norm(20) = +1
    feat_mean = np.array([15.0, 6.0], dtype=np.float32)
    feat_std = np.array([5.0, 2.0], dtype=np.float32)
    time_mean = np.array([150.0, 1.5], dtype=np.float32)
    time_std = np.array([50.0, 0.5], dtype=np.float32)
    buf.prewarm_from_parquet(
        features_parquet=parquet_path,
        feature_columns=["feat_a", "feat_b"],
        time_columns=("duration_s", "n_bars_in_window"),
        feat_mean=feat_mean, feat_std=feat_std,
        time_mean=time_mean, time_std=time_std,
    )
    features, time_features = buf.latest_window()
    np.testing.assert_array_almost_equal(
        features, [[-1.0, -1.0], [1.0, 1.0]],
    )
    np.testing.assert_array_almost_equal(
        time_features, [[-1.0, -1.0], [1.0, 1.0]],
    )


def test_prewarm_respects_as_of_cutoff(tmp_path):
    """as_of=T0 only loads rows with index <= T0."""
    base = pd.Timestamp("2026-05-08", tz="UTC")
    df = pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "duration_s": [10.0, 20.0, 30.0, 40.0, 50.0],
            "n_bars_in_window": [1, 2, 3, 4, 5],
        },
        index=pd.DatetimeIndex(
            [base + pd.Timedelta(minutes=15 * i) for i in range(5)],
            tz="UTC",
        ),
    )
    parquet_path = tmp_path / "feats.parquet"
    df.to_parquet(parquet_path)

    buf = TransVolHistoryBuffer(n_features=1, seq_len=2, max_size=4)
    cutoff = base + pd.Timedelta(minutes=15 * 2)  # only first 3 rows
    n_loaded = buf.prewarm_from_parquet(
        features_parquet=parquet_path,
        feature_columns=["feat_a"],
        time_columns=("duration_s", "n_bars_in_window"),
        feat_mean=np.zeros(1, dtype=np.float32),
        feat_std=np.ones(1, dtype=np.float32),
        time_mean=np.zeros(2, dtype=np.float32),
        time_std=np.ones(2, dtype=np.float32),
        as_of=cutoff,
    )
    # 3 rows ≤ cutoff, but max_size=4 so all 3 fit.
    assert n_loaded == 3
    assert buf.is_inference_ready  # 3 ≥ seq_len=2
    features, _ = buf.latest_window()
    # Last 2 (the seq_len window) are rows 1 and 2 (values 2 and 3).
    np.testing.assert_array_almost_equal(features[:, 0], [2.0, 3.0])


def test_prewarm_raises_on_missing_columns(tmp_path):
    base = pd.Timestamp("2026-05-08", tz="UTC")
    df = pd.DataFrame(
        {"feat_a": [1.0], "duration_s": [10.0], "n_bars_in_window": [1]},
        index=pd.DatetimeIndex([base], tz="UTC"),
    )
    parquet_path = tmp_path / "feats.parquet"
    df.to_parquet(parquet_path)
    buf = TransVolHistoryBuffer(n_features=2, seq_len=1, max_size=1)
    with pytest.raises(KeyError):
        buf.prewarm_from_parquet(
            features_parquet=parquet_path,
            feature_columns=["feat_a", "feat_missing"],
            time_columns=("duration_s", "n_bars_in_window"),
            feat_mean=np.zeros(2, dtype=np.float32),
            feat_std=np.ones(2, dtype=np.float32),
            time_mean=np.zeros(2, dtype=np.float32),
            time_std=np.ones(2, dtype=np.float32),
        )


def test_prewarm_returns_zero_when_parquet_empty_at_cutoff(tmp_path):
    """If as_of is BEFORE the parquet's start, prewarm loads zero rows
    and the buffer stays warmup-only."""
    base = pd.Timestamp("2026-05-08", tz="UTC")
    df = pd.DataFrame(
        {"feat_a": [1.0], "duration_s": [10.0], "n_bars_in_window": [1]},
        index=pd.DatetimeIndex([base], tz="UTC"),
    )
    parquet_path = tmp_path / "feats.parquet"
    df.to_parquet(parquet_path)
    buf = TransVolHistoryBuffer(n_features=1, seq_len=1, max_size=1)
    n = buf.prewarm_from_parquet(
        features_parquet=parquet_path,
        feature_columns=["feat_a"],
        time_columns=("duration_s", "n_bars_in_window"),
        feat_mean=np.zeros(1, dtype=np.float32),
        feat_std=np.ones(1, dtype=np.float32),
        time_mean=np.zeros(2, dtype=np.float32),
        time_std=np.ones(2, dtype=np.float32),
        as_of=base - pd.Timedelta(days=1),
    )
    assert n == 0
    assert not buf.is_inference_ready
