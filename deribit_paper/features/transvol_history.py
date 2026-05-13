"""Live TransVol sequence buffer (S-1 — parity-r3 cycle 4 — 2026-05-08).

The TransVol encoder consumes a 32-bar sequence of 101 z-normalised
features + 2 z-normalised time features (``duration_s``,
``n_bars_in_window``). It outputs a 16-dim ``embedding_compact`` that
LightGBM consumes as ``tv_0..tv_15``.

This buffer mirrors the role of :class:`TSRVHistoryBuffer` for the
TransVol pipeline: it accumulates the per-bar feature row + time-feature
row, and exposes a ``latest_window`` method that returns the most
recent ``seq_len`` rows ready for encoder inference.

Schema:

* ``timestamp`` (UTC, 15-min cadence)
* ``features`` (``n_features``-dim float32; default 101 — the production
  TransVol encoder's input dim. Z-normalised using the bundle's
  ``transvol_feat_mean`` / ``transvol_feat_std`` BEFORE being appended.
  The buffer stores the normalised values so inference is a zero-copy
  ``np.stack`` of the deque.)
* ``time_features`` (2-dim float32; ``duration_s`` + ``n_bars_in_window``,
  z-normalised using ``transvol_time_mean`` / ``transvol_time_std``)

Pre-warm via :meth:`prewarm_from_parquet` reads the offline 15-min
features parquet and applies the bundle's normalisation, so the live
buffer's first ``seq_len`` rows match offline byte-for-byte at startup.

The buffer's ``max_size`` defaults to ``seq_len`` (32) — strictly bounded
so memory stays O(seq_len). A future-looking knob: increase to a
multiple of seq_len if downstream consumers want to inspect drift.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransVolHistoryRow:
    """One 15-min-bar's normalised TransVol input row."""
    timestamp: pd.Timestamp
    features: np.ndarray           # (n_features,) float32, z-normalised
    time_features: np.ndarray      # (2,) float32, z-normalised


class TransVolHistoryBuffer:
    """Bounded deque of recent normalised TransVol input rows.

    Default ``max_size=32`` matches the encoder's ``seq_len`` so the
    buffer is pre-warmed exactly to the encoder's input shape. Increase
    if downstream code wants to inspect a longer trailing window.

    The expected lifecycle:

    1. ``prewarm_from_parquet(...)`` at engine startup loads 32 historical
       bars (or however many fit in ``max_size``) so live inference is
       valid from the first live bar.
    2. ``append(...)`` per live bar (after the bar's features are
       normalised by the FeatureAssembler).
    3. ``latest_window()`` returns the most recent ``seq_len`` rows as
       a tuple of stacked arrays ``(features, time_features)`` ready
       to hand to ``TransVolEncoder.forward``.
    """

    def __init__(
        self,
        *,
        n_features: int,
        seq_len: int = 32,
        max_size: int | None = None,
    ) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        capped = max_size if max_size is not None else seq_len
        if capped < seq_len:
            raise ValueError(
                f"max_size ({capped}) cannot be smaller than seq_len "
                f"({seq_len}) — the buffer would never be inference-ready."
            )
        self._n_features = int(n_features)
        self._seq_len = int(seq_len)
        self._max_size = int(capped)
        self._rows: deque[TransVolHistoryRow] = deque(maxlen=self._max_size)

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def max_size(self) -> int:
        return self._max_size

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterable[TransVolHistoryRow]:
        return iter(self._rows)

    @property
    def is_inference_ready(self) -> bool:
        """True once the buffer has at least ``seq_len`` rows so the
        encoder can run a full causal sequence."""
        return len(self._rows) >= self._seq_len

    def append(
        self,
        timestamp: pd.Timestamp,
        features_normalised: np.ndarray,
        time_features_normalised: np.ndarray,
    ) -> None:
        """Append a single bar's normalised feature row + time-feature row.

        Both arrays must already be z-normalised by the caller (using
        the bundle's TransVol normaliser stats). This buffer does NOT
        normalise — keeping it a dumb deque means the same buffer
        works for any encoder checkpoint without coupling to the
        bundle's stats.

        Raises ``ValueError`` if shapes mismatch the configured
        ``n_features`` (or 2 for time features); raises ``TypeError``
        if the timestamp lacks tz-awareness.
        """
        if not isinstance(timestamp, pd.Timestamp) or timestamp.tzinfo is None:
            raise TypeError(
                f"timestamp must be a tz-aware pd.Timestamp; got {timestamp!r}"
            )
        feat = np.asarray(features_normalised, dtype=np.float32)
        if feat.shape != (self._n_features,):
            raise ValueError(
                f"features shape mismatch: expected ({self._n_features},), "
                f"got {feat.shape}"
            )
        time_feat = np.asarray(time_features_normalised, dtype=np.float32)
        if time_feat.shape != (2,):
            raise ValueError(
                f"time_features shape mismatch: expected (2,), "
                f"got {time_feat.shape}"
            )
        self._rows.append(
            TransVolHistoryRow(
                timestamp=timestamp,
                features=feat,
                time_features=time_feat,
            )
        )

    def latest_window(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the most recent ``seq_len`` rows as stacked arrays.

        Returns ``(features, time_features)`` shaped
        ``(seq_len, n_features)`` and ``(seq_len, 2)`` respectively —
        the inputs to ``TransVolEncoder.forward`` (which adds a leading
        batch dim of 1).

        Raises ``RuntimeError`` if the buffer is not yet
        inference-ready; the caller should check
        :attr:`is_inference_ready` before calling.
        """
        if not self.is_inference_ready:
            raise RuntimeError(
                f"buffer not inference-ready: have {len(self._rows)} rows, "
                f"need {self._seq_len}"
            )
        # Take the last seq_len rows (deque slicing isn't supported;
        # convert to list once).
        rows = list(self._rows)[-self._seq_len:]
        features = np.stack([r.features for r in rows], axis=0)
        time_features = np.stack([r.time_features for r in rows], axis=0)
        return features, time_features

    def latest_timestamp(self) -> pd.Timestamp | None:
        """The timestamp of the most-recent appended row (None if empty)."""
        if not self._rows:
            return None
        return self._rows[-1].timestamp

    def prewarm_from_parquet(
        self,
        *,
        features_parquet: str | Path,
        feature_columns: list[str],
        time_columns: tuple[str, str],
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        time_mean: np.ndarray,
        time_std: np.ndarray,
        as_of: pd.Timestamp | None = None,
        n_rows: int | None = None,
    ) -> int:
        """Load the most recent ``n_rows`` (default ``max_size``) rows
        from ``features_parquet``, normalise via the supplied stats,
        and populate the buffer.

        ``feature_columns`` is the ordered list of 101 column names the
        encoder expects (matches the order used at training time).
        ``time_columns`` is the 2-tuple for time features (typically
        ``("duration_s", "n_bars_in_window")``).

        Returns the number of rows loaded.
        """
        path = Path(features_parquet)
        if not path.exists():
            raise FileNotFoundError(f"features parquet not found: {path}")
        n_target = n_rows if n_rows is not None else self._max_size
        if n_target <= 0:
            raise ValueError(f"n_rows must be positive, got {n_target}")

        df = pd.read_parquet(path)
        if as_of is not None:
            df = df[df.index <= pd.Timestamp(as_of)]
        if len(df) == 0:
            logger.warning(
                "TransVol prewarm: parquet has no rows at/before %s", as_of,
            )
            return 0

        missing_feats = [c for c in feature_columns if c not in df.columns]
        if missing_feats:
            raise KeyError(
                f"features parquet missing columns: {missing_feats[:5]}"
                f"{'...' if len(missing_feats) > 5 else ''}"
            )
        missing_time = [c for c in time_columns if c not in df.columns]
        if missing_time:
            raise KeyError(
                f"features parquet missing time columns: {missing_time}"
            )

        tail = df.tail(n_target)
        X = tail[feature_columns].values.astype(np.float32)
        T = tail[list(time_columns)].values.astype(np.float32)

        # Forward-fill any NaN (offline pipeline drops them, but a
        # partial last bar might still carry one).
        if np.isnan(X).any():
            logger.warning(
                "TransVol prewarm: %d NaN cells in feature window; "
                "filling with zeros for safety",
                int(np.isnan(X).sum()),
            )
            X = np.nan_to_num(X, nan=0.0)
        if np.isnan(T).any():
            T = np.nan_to_num(T, nan=0.0)

        # Z-normalise using the bundle's stats.
        feat_mean = np.asarray(feat_mean, dtype=np.float32)
        feat_std = np.asarray(feat_std, dtype=np.float32).clip(min=1e-8)
        time_mean = np.asarray(time_mean, dtype=np.float32)
        time_std = np.asarray(time_std, dtype=np.float32)
        if feat_mean.shape != (self._n_features,):
            raise ValueError(
                f"feat_mean shape mismatch: ({self._n_features},) "
                f"expected, got {feat_mean.shape}"
            )

        # Cycle-29 (2026-05-13) legacy fallback for pre-cycle-25 checkpoints
        # that never persisted ``time_normalizer_{mean,std}``: compute the
        # normalizer in-place from the prewarm parquet's time columns. The
        # ``as_of`` filter above caps to past-only rows, so this is
        # leakage-safe in the LIVE context (unlike offline extraction
        # where the leak motivated cycle-25). Mirrors the legacy fallback
        # in vol_model/run_walkforward_hybrid.py::_resolve_time_normalizer
        # which slices ``[:max(len-1000, 1)]`` to exclude the latest
        # ``test-window`` slice; we replicate that slice here for byte
        # parity with the offline legacy path.
        if time_mean.shape == (0,) or time_std.shape == (0,):
            import warnings
            warnings.warn(
                f"TransVol checkpoint missing time_normalizer_{{mean,std}} keys; "
                f"computing legacy fallback from prewarm parquet ({path}). "
                "Retrain a cycle-25+ checkpoint to close.",
                DeprecationWarning,
                stacklevel=2,
            )
            slice_end = max(len(T) - 1000, 1)
            time_mean = T[:slice_end].mean(axis=0).astype(np.float32)
            time_std = T[:slice_end].std(axis=0).astype(np.float32)

        time_std = np.asarray(time_std, dtype=np.float32).clip(min=1e-8)
        Xn = (X - feat_mean) / feat_std
        Tn = (T - time_mean) / time_std

        for i in range(len(tail)):
            ts = tail.index[i]
            if not isinstance(ts, pd.Timestamp):
                ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            self.append(ts, Xn[i], Tn[i])
        logger.info(
            "TransVol buffer pre-warmed with %d rows from %s "
            "(latest=%s, seq_len=%d, ready=%s)",
            len(tail), path, self.latest_timestamp(),
            self._seq_len, self.is_inference_ready,
        )
        return len(tail)
