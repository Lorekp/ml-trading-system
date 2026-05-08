"""TransVol live inference wrapper (S-1 — parity-r3 cycle 4 — 2026-05-08).

Wraps the loaded :class:`TransVolEncoder` checkpoint with a per-bar
inference contract:

  ``compute_embedding(buffer) -> np.ndarray``  (16-dim, the
  ``embedding_compact`` head consumed by LightGBM as ``tv_0..tv_15``)

The wrapper is intentionally thin — all heavy lifting lives in
:class:`TransVolHistoryBuffer` (data) and
``vol_model.transvol.model.TransVolEncoder`` (model). This module's job
is the one-liner-per-bar plumbing: assemble the (1, T, F) tensor,
forward through the encoder under ``no_grad``, return the 16-dim
numpy slice.

Returns ``None`` when the buffer is not yet inference-ready (warmup
state); the caller falls back to baseline embeddings (the existing
contract — preserves the pre-S-1 behaviour for the warmup window).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from deribit_paper.features.transvol_history import TransVolHistoryBuffer

logger = logging.getLogger(__name__)


class TransVolInference:
    """Production-side inference wrapper around a loaded TransVol encoder.

    The encoder + buffer + this wrapper form a clean trio:

    - ``buffer`` holds normalised feature rows; the bar loop appends
      to it per bar.
    - ``encoder`` is the loaded ``TransVolEncoder`` from the bundle
      (``ModelBundle.transvol_model``); already in ``eval()`` mode.
    - ``inference`` (this class) calls the encoder once per bar IF the
      buffer is ready, else returns None (warmup window).

    The encoder runs on CPU (no GPU dependency in production); inference
    cost is negligible at 15-min cadence (~ms per call).
    """

    def __init__(
        self,
        *,
        encoder: Any,                  # TransVolEncoder (avoid hard import)
        buffer: TransVolHistoryBuffer,
        device: str = "cpu",
    ) -> None:
        self._encoder = encoder
        self._buffer = buffer
        self._device = device
        # Validate that the buffer's n_features matches the encoder.
        cfg = getattr(encoder, "config", None)
        if cfg is not None:
            n_enc = int(getattr(cfg, "n_features", 0))
            if n_enc and n_enc != buffer.n_features:
                raise ValueError(
                    f"TransVol n_features mismatch: encoder expects "
                    f"{n_enc}, buffer holds {buffer.n_features}"
                )
            seq_enc = int(getattr(cfg, "seq_len", 0))
            if seq_enc and seq_enc != buffer.seq_len:
                logger.warning(
                    "TransVol seq_len mismatch (encoder %d vs buffer %d) "
                    "— buffer will use its own seq_len for the window slice",
                    seq_enc, buffer.seq_len,
                )

    @property
    def buffer(self) -> TransVolHistoryBuffer:
        return self._buffer

    @property
    def is_inference_ready(self) -> bool:
        return self._buffer.is_inference_ready

    def compute_embedding(self) -> np.ndarray | None:
        """Run encoder forward over the latest seq_len rows; return the
        16-dim ``embedding_compact`` slice as a float32 numpy array.

        Returns ``None`` during warmup (buffer < seq_len rows) — caller
        keeps the existing baseline-fallback behaviour intact, so a cold
        engine doesn't emit signal until enough history is accumulated.
        """
        if not self._buffer.is_inference_ready:
            return None
        try:
            import torch
        except ImportError as exc:
            logger.error(
                "torch not importable; TransVol inference disabled (%s)", exc,
            )
            return None

        features, time_features = self._buffer.latest_window()
        x = torch.from_numpy(features).unsqueeze(0).to(self._device)        # (1, T, F)
        t = torch.from_numpy(time_features).unsqueeze(0).to(self._device)   # (1, T, 2)

        with torch.no_grad():
            outputs = self._encoder(x, t)
        emb_compact = outputs.get("embedding_compact")
        if emb_compact is None:
            logger.warning(
                "TransVol encoder returned no 'embedding_compact' key; "
                "older checkpoint without bottleneck head?"
            )
            return None

        arr = emb_compact.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if arr.shape[0] != 16:
            logger.warning(
                "TransVol embedding_compact has unexpected shape %s; "
                "expected (16,)", arr.shape,
            )
            return None
        if not np.all(np.isfinite(arr)):
            logger.warning(
                "TransVol embedding contained non-finite values; "
                "treating as warmup (returning None) so baseline-fallback "
                "kicks in"
            )
            return None
        return arr
