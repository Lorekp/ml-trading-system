"""S-1 (parity-r3 cycle 4 — 2026-05-08) regression tests for the
TransVol inference wrapper.

The wrapper bridges :class:`TransVolHistoryBuffer` and a loaded
:class:`TransVolEncoder` checkpoint:

  ``compute_embedding() -> np.ndarray (16,)``  during normal inference,
  ``compute_embedding() -> None``               during warmup.

These tests exercise the wrapper's contract via a stub encoder + a
real-checkpoint smoke test (the latter only runs when the production
checkpoint is available on disk).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.transvol_history import TransVolHistoryBuffer
from deribit_paper.signal.transvol_inference import TransVolInference


# ── Stub encoder + buffer fixtures ──────────────────────────────────


class _StubEncoder:
    """Minimal stand-in for TransVolEncoder.

    Returns a deterministic embedding_compact = mean of last-row
    features (so the test can verify shape/finite/determinism without
    depending on torch's random init).
    """

    def __init__(self, n_features: int, seq_len: int):
        self.config = MagicMock(n_features=n_features, seq_len=seq_len)

    def __call__(self, x, t):
        # x: (B, T, F) torch tensor. Take last row of batch 0,
        # tile/slice into 16-dim. Fully deterministic.
        import torch
        last = x[0, -1, :]                 # (F,)
        # Take mean of consecutive groups so output is 16-dim regardless
        # of F. For F < 16, repeat-tile.
        F = last.shape[0]
        if F >= 16:
            chunks = torch.tensor_split(last, 16)
            emb_compact = torch.stack([c.mean() for c in chunks])
        else:
            emb_compact = last.repeat((16 + F - 1) // F)[:16]
        return {
            "embedding": last.unsqueeze(0),
            "embedding_compact": emb_compact.unsqueeze(0),
        }


def _make_filled_buffer(
    n_features: int = 101, seq_len: int = 32, seed: int = 0,
) -> TransVolHistoryBuffer:
    buf = TransVolHistoryBuffer(n_features=n_features, seq_len=seq_len)
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-05-08 12:00:00", tz="UTC")
    for i in range(seq_len):
        feat = rng.standard_normal(n_features).astype(np.float32)
        time_feat = rng.standard_normal(2).astype(np.float32)
        buf.append(base + pd.Timedelta(minutes=15 * i), feat, time_feat)
    return buf


# ── Construction ────────────────────────────────────────────────────


def test_constructor_holds_encoder_and_buffer():
    encoder = _StubEncoder(n_features=101, seq_len=32)
    buf = TransVolHistoryBuffer(n_features=101, seq_len=32)
    inf = TransVolInference(encoder=encoder, buffer=buf)
    assert inf.buffer is buf


def test_constructor_validates_n_features_match():
    """Encoder expects n_features=101 but buffer holds 50 → reject."""
    encoder = _StubEncoder(n_features=101, seq_len=32)
    buf = TransVolHistoryBuffer(n_features=50, seq_len=32)
    with pytest.raises(ValueError):
        TransVolInference(encoder=encoder, buffer=buf)


def test_constructor_warns_on_seq_len_mismatch(caplog):
    """Encoder seq_len ≠ buffer seq_len logs a warning but doesn't error
    — the buffer's seq_len drives the window slice."""
    encoder = _StubEncoder(n_features=101, seq_len=64)
    buf = TransVolHistoryBuffer(n_features=101, seq_len=32)
    with caplog.at_level("WARNING"):
        inf = TransVolInference(encoder=encoder, buffer=buf)
    assert any("seq_len mismatch" in r.message for r in caplog.records)
    assert inf.buffer.seq_len == 32


# ── Warmup behaviour ───────────────────────────────────────────────


def test_compute_embedding_returns_none_during_warmup():
    encoder = _StubEncoder(n_features=101, seq_len=32)
    buf = TransVolHistoryBuffer(n_features=101, seq_len=32)
    # Buffer is empty
    inf = TransVolInference(encoder=encoder, buffer=buf)
    assert inf.is_inference_ready is False
    assert inf.compute_embedding() is None


def test_compute_embedding_returns_none_when_partially_filled():
    encoder = _StubEncoder(n_features=101, seq_len=32)
    buf = TransVolHistoryBuffer(n_features=101, seq_len=32)
    # Fill only half
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2026-05-08 12:00:00", tz="UTC")
    for i in range(16):
        feat = rng.standard_normal(101).astype(np.float32)
        time_feat = rng.standard_normal(2).astype(np.float32)
        buf.append(base + pd.Timedelta(minutes=15 * i), feat, time_feat)
    inf = TransVolInference(encoder=encoder, buffer=buf)
    assert inf.compute_embedding() is None


# ── Inference path ──────────────────────────────────────────────────


def test_compute_embedding_returns_16dim_finite_array():
    encoder = _StubEncoder(n_features=101, seq_len=32)
    buf = _make_filled_buffer()
    inf = TransVolInference(encoder=encoder, buffer=buf)
    emb = inf.compute_embedding()
    assert emb is not None
    assert emb.shape == (16,)
    assert emb.dtype == np.float32
    assert np.all(np.isfinite(emb))


def test_compute_embedding_is_deterministic_for_same_buffer():
    """Same buffer state → same embedding every call. Critical
    invariant for live trading reproducibility."""
    encoder = _StubEncoder(n_features=101, seq_len=32)
    buf = _make_filled_buffer(seed=42)
    inf = TransVolInference(encoder=encoder, buffer=buf)
    emb1 = inf.compute_embedding()
    emb2 = inf.compute_embedding()
    np.testing.assert_array_equal(emb1, emb2)


def test_compute_embedding_changes_when_buffer_advances():
    """A new bar's worth of features changes the encoder's input
    sequence; the embedding must reflect that, not stay frozen."""
    encoder = _StubEncoder(n_features=101, seq_len=32)
    buf = _make_filled_buffer(seed=0)
    inf = TransVolInference(encoder=encoder, buffer=buf)
    emb_before = inf.compute_embedding()
    # Append one more bar; buffer evicts the oldest, latest_window
    # shifts forward.
    rng = np.random.default_rng(999)  # different seed → different row
    feat = rng.standard_normal(101).astype(np.float32)
    time_feat = rng.standard_normal(2).astype(np.float32)
    buf.append(
        pd.Timestamp("2026-05-08 20:00:00", tz="UTC"),
        feat, time_feat,
    )
    emb_after = inf.compute_embedding()
    assert emb_before is not None
    assert emb_after is not None
    # Stub encoder = mean of last-row chunks; since the last row
    # changed, the output must differ.
    assert not np.allclose(emb_before, emb_after)


def test_compute_embedding_handles_non_finite_output():
    """If the encoder somehow emits NaN/inf, return None so the caller
    falls back to baseline rather than silently passing corrupt
    embeddings to LightGBM."""
    class _BadEncoder:
        config = MagicMock(n_features=101, seq_len=32)

        def __call__(self, x, t):
            import torch
            return {
                "embedding": torch.zeros(1, 101),
                "embedding_compact": torch.full((1, 16), float("nan")),
            }

    buf = _make_filled_buffer()
    inf = TransVolInference(encoder=_BadEncoder(), buffer=buf)
    assert inf.compute_embedding() is None


def test_compute_embedding_handles_missing_compact_key():
    """Old-format encoders without ``embedding_compact`` → return None
    + log a warning, never raise."""
    class _LegacyEncoder:
        config = MagicMock(n_features=101, seq_len=32)

        def __call__(self, x, t):
            import torch
            return {"embedding": torch.zeros(1, 128)}  # no compact head

    buf = _make_filled_buffer()
    inf = TransVolInference(encoder=_LegacyEncoder(), buffer=buf)
    assert inf.compute_embedding() is None


def test_compute_embedding_handles_unexpected_compact_shape():
    """Encoder emits compact with wrong dim (not 16) → return None."""
    class _WrongShapeEncoder:
        config = MagicMock(n_features=101, seq_len=32)

        def __call__(self, x, t):
            import torch
            return {
                "embedding": torch.zeros(1, 128),
                "embedding_compact": torch.zeros(1, 8),  # wrong
            }

    buf = _make_filled_buffer()
    inf = TransVolInference(encoder=_WrongShapeEncoder(), buffer=buf)
    assert inf.compute_embedding() is None


# ── Real-checkpoint smoke test (skip when not available) ───────────


_REAL_CHECKPOINT_PATH = Path(
    "vol_model/output/models/expanding_hybrid/2026-04_transvol.pt"
)


@pytest.mark.skipif(
    not _REAL_CHECKPOINT_PATH.exists(),
    reason="Production TransVol checkpoint not present in this environment",
)
def test_real_checkpoint_inference_smoke():
    """End-to-end smoke: load the latest production checkpoint, fill a
    buffer with random normalised inputs, run inference, assert the
    output is a valid 16-dim embedding."""
    import torch

    from vol_model.transvol.config import TransVolConfig
    from vol_model.transvol.model import TransVolEncoder

    ckpt = torch.load(
        str(_REAL_CHECKPOINT_PATH), map_location="cpu", weights_only=False,
    )
    config: TransVolConfig = ckpt["config"]
    encoder = TransVolEncoder(config)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()

    n_features = config.n_features
    seq_len = config.seq_len
    buf = TransVolHistoryBuffer(n_features=n_features, seq_len=seq_len)
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2026-05-08 12:00:00", tz="UTC")
    for i in range(seq_len):
        feat = rng.standard_normal(n_features).astype(np.float32)
        time_feat = rng.standard_normal(2).astype(np.float32)
        buf.append(base + pd.Timedelta(minutes=15 * i), feat, time_feat)

    inf = TransVolInference(encoder=encoder, buffer=buf)
    emb = inf.compute_embedding()
    assert emb is not None
    assert emb.shape == (16,)
    assert emb.dtype == np.float32
    assert np.all(np.isfinite(emb))
    # Compact embedding from Tanh head: bounded in [-1, 1].
    assert np.all(emb >= -1.001) and np.all(emb <= 1.001)
