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
constant-rate trading. For full byte-identity, future work would need
to maintain a 1750-bar dollar-bar buffer (memory cost ~500 KB) and
ffill TSRV onto it per handler call. Documented in the
``vol_arb/output/PARITY_AUDIT_ROUND2.md`` synthesis.

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

    # ── cache management ─────────────────────────────────────────────

    def _refresh(self) -> None:
        n = len(self._buffer)
        if n == self._cached_len and self._cached_log_vol_ann is not None:
            return
        self._cached_len = n
        if n == 0:
            self._cached_log_vol_ann = None
            self._cached_tsrv_mean_1d = None
            return
        # ln(vol_ann) — match offline ``np.log(tsrv_df["vol_ann"].clip(lower=1e-10))``
        vol_ann = self._buffer.as_series("vol_ann")
        self._cached_log_vol_ann = np.log(vol_ann.clip(lower=1e-10))
        # Invalidate dependent caches
        self._cached_tsrv_mean_1d = None

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
        ser = self._tsrv_mean_1d_series()
        if ser is None or len(ser) < 4:
            return None
        v = ser.iloc[-4]
        return None if not np.isfinite(v) else float(v)

    def tsrv_mean_1d_ema50(self, span: int = 50) -> float | None:
        ser = self._tsrv_mean_1d_series()
        if ser is None or ser.empty:
            return None
        v = ser.ewm(span=span, adjust=True, min_periods=1).mean().iloc[-1]
        return None if not np.isfinite(v) else float(v)

    def tsrv_mean_1w(self) -> float | None:
        ln = self._log_vol_ann()
        if ln is None or len(ln) == 0:
            return None
        v = ln.rolling(
            _BARS_PER_WEEK, min_periods=max(_BARS_PER_WEEK // 4, 10),
        ).mean().iloc[-1]
        return None if not np.isfinite(v) else float(v)

    def tsrv_ffd(self) -> float | None:
        """Fractional differencing of log(vol_ann), d=0.4 (mirrors offline)."""
        ln = self._log_vol_ann()
        if ln is None or len(ln) < 30:
            return None
        try:
            from src.afml.fracdiff import frac_diff_ffd
        except ImportError:
            logger.warning(
                "TSRVFeatures.tsrv_ffd: src.afml.fracdiff not importable; "
                "returning None (handler falls back to baseline)",
            )
            return None
        # frac_diff_ffd expects a DataFrame; wrap then unwrap.
        df = ln.to_frame(name="tsrv")
        try:
            ffd = frac_diff_ffd(df, d=_FRACDIFF_D, threshold=_FRACDIFF_THRESHOLD)
        except Exception:  # noqa: BLE001
            logger.debug("frac_diff_ffd failed", exc_info=True)
            return None
        if ffd is None or "tsrv" not in ffd.columns or ffd["tsrv"].empty:
            return None
        v = ffd["tsrv"].iloc[-1]
        return None if not np.isfinite(v) else float(v)

    def up_jump_frac_ema50(self, span: int = 50) -> float | None:
        ser = self._buffer.as_series("up_jump_fraction")
        if ser.empty:
            return None
        v = ser.ewm(span=span, adjust=True, min_periods=1).mean().iloc[-1]
        return None if not np.isfinite(v) else float(v)

    def bpv_tsrv_ratio_ema50(self, span: int = 50) -> float | None:
        """EMA50 of ``bpv / rv_slow`` per the offline definition."""
        bpv = self._buffer.as_series("bpv")
        rv_slow = self._buffer.as_series("rv_slow")
        if bpv.empty or rv_slow.empty:
            return None
        # Mirror offline ``tsrv_df["bpv"] / tsrv_df["rv_slow"].clip(lower=1e-20)``
        ratio = bpv / rv_slow.clip(lower=1e-20)
        v = ratio.ewm(span=span, adjust=True, min_periods=1).mean().iloc[-1]
        return None if not np.isfinite(v) else float(v)


__all__ = ["TSRVFeatures"]
