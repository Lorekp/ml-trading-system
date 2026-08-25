"""F3 Tier 5 TSRV-grid feature wrapper.

Wraps a :class:`TSRVHistoryBuffer` and exposes the 6 TSRV-derived
manifest features that were baseline-frozen pre-Tier 5:

* ``tsrv_mean_1d`` — already lived in ``feature_assembler._h_tsrv_mean_1d``
  as a baseline-fallback handler. Tier 5 upgrades it to a live rolling
  mean of ``log(vol_ann)`` over the last 96 15-min bars.
* ``tsrv_mean_1d_lag3`` — lag-3 of the rolling-1d series.
* ``tsrv_mean_1d_ema50`` — EMA50 of the rolling-1d series.
* ``tsrv_mean_1w`` — rolling mean over the last 672 15-min bars (1 week).
* ``tsrv_ffd`` — fractional differencing of ``log(vol_ann)`` with d=0.4.
* ``up_jump_frac_ema50`` — EMA50 of the ``up_jump_fraction`` series.
* ``bpv_tsrv_ratio_ema50`` — EMA50 of ``bpv / rv_slow``.

**Deviation from offline byte-identity (documented):**

The offline pipeline (``vol_model/features/engineer.py:243-353``) ffills
TSRV onto the dollar-bar grid then rolls/EMAs on that grid before
aggregating to 15-min wall-clock via ``groupby(window_labels).last()``.
The dollar-bar grid clusters during high-volume periods, so offline
weights bursty periods more heavily than calm ones.

Tier 5 ships a 15-min-grid implementation: rolling/EMA windows are
scaled to the 15-min cadence (96=1d, 672=1w). This deviates from
offline byte-identity when volume is bursty; it remains substantially
better than baseline-frozen and is mathematically equivalent under
constant-rate trading. Exact offline parity would require maintaining a
1750-bar dollar-bar buffer and forward-filling TSRV onto it per handler call.

Caches the log-transformed vol_ann series + the rolling-1d series with
dirty-bit invalidation tied to buffer length, so repeated handler calls
within a single bar don't recompute.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from deribit_paper.features.tsrv_history import TSRVHistoryBuffer

logger = logging.getLogger(__name__)


# Window sizes on the 15-min grid (4 bars/hour × 24h = 96 bars/day).
_BARS_PER_DAY: int = 96
_BARS_PER_WEEK: int = 672

# Fractional-differencing parameters (mirrors vol_model/config.py:128-129)
_FRACDIFF_D: float = 0.4
_FRACDIFF_THRESHOLD: float = 1e-5


class TSRVFeatures:
    """Cached view over a ``TSRVHistoryBuffer`` for live feature computation."""

    def __init__(self, buffer: TSRVHistoryBuffer):
        self._buffer = buffer
        self._cached_len: int = -1
        self._cached_log_vol_ann: pd.Series | None = None
        self._cached_tsrv_mean_1d: pd.Series | None = None
        # S-1c (parity-r3 cycle 6 — 2026-05-08): cached series for
        # the 1w / FFD / noise_level / bpv_tsrv_ratio / down_jump_frac /
        # up_jump_frac variants. Each entry is a pandas Series aligned
        # with the buffer's bar grid; invalidated on length change.
        self._cached_derived: dict[str, pd.Series] = {}

    # ── cache management ─────────────────────────────────────────────

    def _refresh(self) -> None:
        n = len(self._buffer)
        if n == self._cached_len and self._cached_log_vol_ann is not None:
            return
        self._cached_len = n
        if n == 0:
            self._cached_log_vol_ann = None
            self._cached_tsrv_mean_1d = None
            self._cached_derived.clear()
            return
        # ln(vol_ann) — match offline ``np.log(tsrv_df["vol_ann"].clip(lower=1e-10))``
        vol_ann = self._buffer.as_series("vol_ann")
        self._cached_log_vol_ann = np.log(vol_ann.clip(lower=1e-10))
        # Invalidate dependent caches
        self._cached_tsrv_mean_1d = None
        self._cached_derived.clear()

    def _log_vol_ann(self) -> pd.Series | None:
        self._refresh()
        return self._cached_log_vol_ann

    def _tsrv_mean_1d_series(self) -> pd.Series | None:
        """Rolling mean of log(vol_ann) over the last 96 bars."""
        self._refresh()
        if self._cached_tsrv_mean_1d is not None:
            return self._cached_tsrv_mean_1d
        ln = self._cached_log_vol_ann
        if ln is None or len(ln) == 0:
            return None
        # min_periods matches offline ``mp = max(w//4, 10)`` ≈ max(24, 10) = 24.
        self._cached_tsrv_mean_1d = ln.rolling(
            _BARS_PER_DAY, min_periods=max(_BARS_PER_DAY // 4, 10),
        ).mean()
        return self._cached_tsrv_mean_1d

    # ── public API ──────────────────────────────────────────────────

    def tsrv_mean_1d(self) -> float | None:
        ser = self._tsrv_mean_1d_series()
        if ser is None or ser.empty:
            return None
        v = ser.iloc[-1]
        return None if not np.isfinite(v) else float(v)

    def tsrv_mean_1d_lag3(self) -> float | None:
        return self.tsrv_mean_1d_lag(3)

    def tsrv_mean_1d_ema50(self, span: int = 50) -> float | None:
        return self.tsrv_mean_1d_ema(span)

    def tsrv_mean_1w(self) -> float | None:
        ser = self._tsrv_mean_1w_series()
        if ser is None or ser.empty:
            return None
        v = ser.iloc[-1]
        return None if not np.isfinite(v) else float(v)

    def tsrv_ffd(self) -> float | None:
        """Fractional differencing of log(vol_ann), d=0.4 (mirrors offline)."""
        ser = self._tsrv_ffd_series()
        if ser is None or ser.empty:
            return None
        v = ser.iloc[-1]
        return None if not np.isfinite(v) else float(v)

    def up_jump_frac(self) -> float | None:
        ser = self._buffer.as_series("up_jump_fraction")
        if ser.empty:
            return None
        v = ser.iloc[-1]
        return None if not np.isfinite(v) else float(v)

    def up_jump_frac_ema50(self, span: int = 50) -> float | None:
        return self.up_jump_frac_ema(span)

    def bpv_tsrv_ratio_ema50(self, span: int = 50) -> float | None:
        return self.bpv_tsrv_ratio_ema(span)

    # ── S-1c (parity-r3 cycle 6 — 2026-05-08): generalised lag/EMA ──
    # Variants for tsrv_mean_1d, tsrv_mean_1w, tsrv_ffd, up_jump_frac,
    # bpv_tsrv_ratio, down_jump_frac, noise_level. Pattern: cache the
    # underlying series (per-bar pandas Series), then ``.shift(n)`` /
    # ``.ewm(span)`` at the latest bar.

    def _tsrv_mean_1w_series(self) -> pd.Series | None:
        cached = self._cached_derived.get("tsrv_mean_1w")
        if cached is not None:
            return cached
        ln = self._log_vol_ann()
        if ln is None or len(ln) == 0:
            return None
        ser = ln.rolling(
            _BARS_PER_WEEK, min_periods=max(_BARS_PER_WEEK // 4, 10),
        ).mean()
        self._cached_derived["tsrv_mean_1w"] = ser
        return ser

    def _tsrv_ffd_series(self) -> pd.Series | None:
        cached = self._cached_derived.get("tsrv_ffd")
        if cached is not None:
            return cached
        ln = self._log_vol_ann()
        if ln is None or len(ln) < 30:
            return None
        try:
            from src.afml.fracdiff import frac_diff_ffd
        except ImportError:
            logger.warning(
                "TSRVFeatures._tsrv_ffd_series: src.afml.fracdiff not "
                "importable; returning None (handler falls back to baseline)",
            )
            return None
        df = ln.to_frame(name="tsrv")
        try:
            ffd = frac_diff_ffd(df, d=_FRACDIFF_D, threshold=_FRACDIFF_THRESHOLD)
        except Exception:  # noqa: BLE001
            logger.debug("frac_diff_ffd failed", exc_info=True)
            return None
        if ffd is None or "tsrv" not in ffd.columns or ffd["tsrv"].empty:
            return None
        # Reindex to the full buffer index so lag/EMA align positionally
        # with the bar grid (offline does the same when ffilling the FFD
        # output back onto the dollar-bar grid before lag/EMA aggregation).
        ser = ffd["tsrv"].reindex(ln.index)
        self._cached_derived["tsrv_ffd"] = ser
        return ser

    def _bpv_tsrv_ratio_series(self) -> pd.Series | None:
        cached = self._cached_derived.get("bpv_tsrv_ratio")
        if cached is not None:
            return cached
        bpv = self._buffer.as_series("bpv")
        rv_slow = self._buffer.as_series("rv_slow")
        if bpv.empty or rv_slow.empty:
            return None
        ratio = bpv / rv_slow.clip(lower=1e-20)
        self._cached_derived["bpv_tsrv_ratio"] = ratio
        return ratio

    def _noise_level_series(self) -> pd.Series | None:
        """``rv_fast / (2 × max(n_ticks, 1))`` per bar.

        Mirrors offline ``vol_model/features/engineer.py:350``::

            feats["noise_level"] = _ffill_to_bars(
                tsrv_df["rv_fast"] / (2 * tsrv_df["n_ticks"].clip(lower=1)))
        """
        cached = self._cached_derived.get("noise_level")
        if cached is not None:
            return cached
        rv_fast = self._buffer.as_series("rv_fast")
        n_ticks = self._buffer.as_series("n_ticks")
        if rv_fast.empty or n_ticks.empty:
            return None
        ser = rv_fast / (2.0 * n_ticks.clip(lower=1.0))
        self._cached_derived["noise_level"] = ser
        return ser

    def _down_jump_frac_series(self) -> pd.Series | None:
        cached = self._cached_derived.get("down_jump_frac")
        if cached is not None:
            return cached
        ser = self._buffer.as_series("down_jump_fraction")
        if ser.empty:
            return None
        self._cached_derived["down_jump_frac"] = ser
        return ser

    def _up_jump_frac_series(self) -> pd.Series | None:
        cached = self._cached_derived.get("up_jump_frac")
        if cached is not None:
            return cached
        ser = self._buffer.as_series("up_jump_fraction")
        if ser.empty:
            return None
        self._cached_derived["up_jump_frac"] = ser
        return ser

    @staticmethod
    def _series_lag(ser: pd.Series | None, n: int) -> float | None:
        if ser is None or len(ser) <= n:
            return None
        v = ser.iloc[-1 - n]
        return None if not np.isfinite(v) else float(v)

    @staticmethod
    def _series_ema_at_last(ser: pd.Series | None, span: int) -> float | None:
        if ser is None or ser.empty:
            return None
        v = ser.ewm(span=span, adjust=True, min_periods=1).mean().iloc[-1]
        return None if not np.isfinite(v) else float(v)

    # ── tsrv_mean_1d generalised lag/EMA ────────────────────────────

    def tsrv_mean_1d_lag(self, n: int) -> float | None:
        return self._series_lag(self._tsrv_mean_1d_series(), n)

    def tsrv_mean_1d_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._tsrv_mean_1d_series(), span)

    # ── tsrv_mean_1w generalised lag/EMA ────────────────────────────

    def tsrv_mean_1w_lag(self, n: int) -> float | None:
        return self._series_lag(self._tsrv_mean_1w_series(), n)

    def tsrv_mean_1w_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._tsrv_mean_1w_series(), span)

    # ── tsrv_ffd generalised lag/EMA ────────────────────────────────

    def tsrv_ffd_lag(self, n: int) -> float | None:
        return self._series_lag(self._tsrv_ffd_series(), n)

    def tsrv_ffd_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._tsrv_ffd_series(), span)

    # ── up_jump_frac generalised lag/EMA ────────────────────────────

    def up_jump_frac_lag(self, n: int) -> float | None:
        return self._series_lag(self._up_jump_frac_series(), n)

    def up_jump_frac_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._up_jump_frac_series(), span)

    # ── down_jump_frac (raw + lag/EMA) ─────────────────────────────

    def down_jump_frac(self) -> float | None:
        ser = self._down_jump_frac_series()
        if ser is None or ser.empty:
            return None
        v = ser.iloc[-1]
        return None if not np.isfinite(v) else float(v)

    def down_jump_frac_lag(self, n: int) -> float | None:
        return self._series_lag(self._down_jump_frac_series(), n)

    def down_jump_frac_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._down_jump_frac_series(), span)

    # ── bpv_tsrv_ratio (raw + lag/EMA) ─────────────────────────────

    def bpv_tsrv_ratio(self) -> float | None:
        ser = self._bpv_tsrv_ratio_series()
        if ser is None or ser.empty:
            return None
        v = ser.iloc[-1]
        return None if not np.isfinite(v) else float(v)

    def bpv_tsrv_ratio_lag(self, n: int) -> float | None:
        return self._series_lag(self._bpv_tsrv_ratio_series(), n)

    def bpv_tsrv_ratio_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._bpv_tsrv_ratio_series(), span)

    # ── noise_level (raw + lag/EMA) ────────────────────────────────

    def noise_level(self) -> float | None:
        ser = self._noise_level_series()
        if ser is None or ser.empty:
            return None
        v = ser.iloc[-1]
        return None if not np.isfinite(v) else float(v)

    def noise_level_lag(self, n: int) -> float | None:
        return self._series_lag(self._noise_level_series(), n)

    def noise_level_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._noise_level_series(), span)


__all__ = ["TSRVFeatures"]
