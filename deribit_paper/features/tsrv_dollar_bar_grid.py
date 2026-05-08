"""TSRV-on-dollar-bar-grid features (S-1d — parity-r3 cycle 7 — 2026-05-08).

The offline pipeline (``vol_model/features/engineer.py``) computes the
TSRV-derived TransVol input columns by:

1. Forward-filling each TSRV scalar (vol_ann, up_jump_fraction,
   down_jump_fraction, rv_fast/(2*max(n_ticks,1)), bpv/rv_slow.clip)
   onto the DOLLAR-BAR grid via ``shift(1).reindex(union).ffill()``.
2. Rolling / EMA / lag on the dollar-bar grid via
   ``_window_and_aggregate``.
3. Aggregating to the 15-min grid via
   ``series.groupby(window_labels).last()`` — i.e. each 15-min row
   carries the LAST DOLLAR BAR's value within that window.

Cycle-5 + cycle-6 shipped a pragmatic 15-min-grid implementation
(``TSRVFeatures``) that deviates 13–27 % from offline byte-equality
on bursty days — accepted by design as round-2 audit C1.

This module closes C1 strictly: replays the offline computation on
the live ``DollarBarBuffer + TSRVHistoryBuffer`` pair so every
Tier-B column matches offline at ``rtol=1e-5, atol=1e-7``.

The 6 base feature groups handled (all 30 columns including
variants):

- ``tsrv_mean_1d`` (rolling mean of log(vol_ann) over bpd × 1)
- ``tsrv_mean_1w`` (rolling mean over min(bpd × 7, 50_000))
- ``tsrv_ffd`` (frac_diff_ffd of log(vol_ann), d=0.4)
- ``up_jump_frac`` (ffilled directly)
- ``down_jump_frac`` (ffilled directly)
- ``noise_level`` (ffilled rv_fast/(2*max(n_ticks,1)))
- ``bpv_tsrv_ratio`` (ffilled bpv/rv_slow.clip(lower=1e-20))

Each base supports raw + lag(n) + ema(span) accessors, computed
on the dollar-bar grid then read at the latest position (which
equals offline's ``groupby(window_labels).last().iloc[-1]``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Mirrors offline `vol_model/config.py:128-129`.
_FRACDIFF_D: float = 0.4
_FRACDIFF_THRESHOLD: float = 1e-5


def _last_finite(values) -> float | None:
    """Return the last element if finite; ``None`` otherwise.

    Used to extract the "latest" value from a per-dollar-bar series
    (mirrors offline's ``groupby(window_labels).last().iloc[-1]`` for
    the current 15-min window — the latest dollar bar IS the last in
    the most recent window). ``None`` input -> ``None`` (handler
    fall-through).
    """
    if values is None:
        return None
    if isinstance(values, pd.Series):
        if values.empty:
            return None
        v = values.iloc[-1]
    else:
        if len(values) == 0:
            return None
        v = values[-1]
    return None if not np.isfinite(v) else float(v)


class TSRVDollarBarFeatures:
    """Cached compute over a (TSRVHistoryBuffer, DollarBarBuffer) pair.

    Exposes byte-equivalent versions of the 30 Tier-B TransVol input
    columns via the offline ``_ffill_to_bars + _window_and_aggregate``
    pipeline — runs the live equivalent on the cached buffer pair.

    All public methods return ``float | None``. ``None`` is emitted
    when the underlying series is too short for the requested
    operation; the caller (handler) maps that to a baseline-fallback.

    Cache contract: invalidated when EITHER buffer's length changes.
    Repeated calls within one bar share a cached ffill + rolling
    series so per-bar compute cost stays O(buffer_length) once
    rather than per-feature.
    """

    def __init__(self, tsrv_buffer: Any, db_buffer: Any) -> None:
        """``tsrv_buffer``: TSRVHistoryBuffer; ``db_buffer``:
        DollarBarBuffer. Late-bound types to avoid circular imports.
        """
        self._tsrv = tsrv_buffer
        self._db = db_buffer
        self._cached_tsrv_len: int = -1
        self._cached_db_len: int = -1
        self._cached_db_idx: pd.DatetimeIndex | None = None
        # Cache keyed by feature-name (or (name, params)) for the
        # per-dollar-bar series. Cleared on buffer-length change.
        self._cached_series: dict[Any, pd.Series] = {}

    # ── cache management ────────────────────────────────────────────

    def _refresh(self) -> None:
        tsrv_len = len(self._tsrv)
        db_len = len(self._db)
        if (
            tsrv_len == self._cached_tsrv_len
            and db_len == self._cached_db_len
            and self._cached_db_idx is not None
        ):
            return
        self._cached_tsrv_len = tsrv_len
        self._cached_db_len = db_len
        self._cached_db_idx = self._db.as_index() if db_len > 0 else None
        self._cached_series.clear()

    # ── ffill primitive (offline _ffill_to_bars equivalent) ─────────

    def _ffill_to_dollar_bars(
        self, tsrv_column: str,
    ) -> pd.Series | None:
        """Mirror of offline ``_ffill_to_bars(tsrv_df[col])``::

            lagged = series.shift(1)
            ci = lagged.dropna().index.union(bars.index).sort_values()
            return lagged.reindex(ci).ffill().reindex(bars.index)

        ``tsrv_column`` is the field name on TSRVHistoryRow (e.g.
        ``"up_jump_fraction"``, ``"vol_ann"``). Returns the
        per-dollar-bar series indexed by the buffer's bar timestamps.
        """
        self._refresh()
        if self._cached_db_idx is None or self._cached_tsrv_len == 0:
            return None
        cache_key = ("_ffill", tsrv_column)
        cached = self._cached_series.get(cache_key)
        if cached is not None:
            return cached
        tsrv_series = self._tsrv.as_series(tsrv_column)
        if tsrv_series.empty:
            return None
        lagged = tsrv_series.shift(1)
        union_idx = lagged.dropna().index.union(self._cached_db_idx).sort_values()
        ffilled = lagged.reindex(union_idx).ffill().reindex(self._cached_db_idx)
        self._cached_series[cache_key] = ffilled
        return ffilled

    def _ffill_compound_to_dollar_bars(
        self, name: str, compute_fn,
    ) -> pd.Series | None:
        """ffill a COMPUTED TSRV scalar onto the dollar-bar grid.

        Used for ``noise_level`` (rv_fast / (2 × max(n_ticks, 1)))
        and ``bpv_tsrv_ratio`` (bpv / rv_slow.clip(lower=1e-20)) —
        the offline pipeline computes these values FIRST on the TSRV
        cadence, then ffills onto bars (engineer.py:350-353).

        ``compute_fn`` accepts no args; returns the per-TSRV-bar
        scalar series (same index as ``self._tsrv.as_series``).
        """
        self._refresh()
        if self._cached_db_idx is None or self._cached_tsrv_len == 0:
            return None
        cache_key = ("_ffill_compound", name)
        cached = self._cached_series.get(cache_key)
        if cached is not None:
            return cached
        tsrv_scalar = compute_fn()
        if tsrv_scalar is None or tsrv_scalar.empty:
            return None
        lagged = tsrv_scalar.shift(1)
        union_idx = lagged.dropna().index.union(self._cached_db_idx).sort_values()
        ffilled = lagged.reindex(union_idx).ffill().reindex(self._cached_db_idx)
        self._cached_series[cache_key] = ffilled
        return ffilled

    # ── per-base raw series (cached) ────────────────────────────────

    def _tsrv_log_vol_at_db(self) -> pd.Series | None:
        """ffill log(vol_ann.clip(lower=1e-10)) onto dollar bar grid.

        Mirrors offline ``ln_vol_lagged.reindex(ci).ffill()`` at
        engineer.py:251-253. Used by tsrv_mean_1d / tsrv_mean_1w /
        tsrv_ffd.
        """
        cache_key = ("_log_vol_at_db",)
        cached = self._cached_series.get(cache_key)
        if cached is not None:
            return cached
        self._refresh()
        if self._cached_db_idx is None or self._cached_tsrv_len == 0:
            return None
        vol_ann = self._tsrv.as_series("vol_ann")
        if vol_ann.empty:
            return None
        ln = np.log(vol_ann.clip(lower=1e-10))
        lagged = ln.shift(1)
        union_idx = lagged.dropna().index.union(self._cached_db_idx).sort_values()
        out = lagged.reindex(union_idx).ffill().reindex(self._cached_db_idx)
        self._cached_series[cache_key] = out
        return out

    def _down_jump_frac_at_db(self) -> pd.Series | None:
        return self._ffill_to_dollar_bars("down_jump_fraction")

    def _up_jump_frac_at_db(self) -> pd.Series | None:
        return self._ffill_to_dollar_bars("up_jump_fraction")

    def _noise_level_at_db(self) -> pd.Series | None:
        def _compute():
            rv_fast = self._tsrv.as_series("rv_fast")
            n_ticks = self._tsrv.as_series("n_ticks")
            if rv_fast.empty or n_ticks.empty:
                return None
            return rv_fast / (2.0 * n_ticks.clip(lower=1.0))
        return self._ffill_compound_to_dollar_bars("noise_level", _compute)

    def _bpv_tsrv_ratio_at_db(self) -> pd.Series | None:
        def _compute():
            bpv = self._tsrv.as_series("bpv")
            rv_slow = self._tsrv.as_series("rv_slow")
            if bpv.empty or rv_slow.empty:
                return None
            return bpv / rv_slow.clip(lower=1e-20)
        return self._ffill_compound_to_dollar_bars("bpv_tsrv_ratio", _compute)

    # ── tsrv_mean_1d / 1w (rolling on dollar-bar grid) ──────────────

    def _tsrv_mean_at_db(
        self, mult: int, cap: int = 50_000,
    ) -> pd.Series | None:
        """Rolling-mean of log(vol_ann) ffilled-to-db with window
        ``min(max(bpd × mult, 10), cap)`` and ``min_periods = w//4``.

        Mirrors offline engineer.py:264-267::

            for mult, label, cap in [(1, "1d", 50_000), (7, "1w", 50_000)]:
                w = min(max(bpd * mult, 10), cap)
                mp = max(w // 4, 10)
                feats[f"tsrv_mean_{label}"] = tsrv_at_bar.rolling(
                    w, min_periods=mp).mean()
        """
        cache_key = ("_tsrv_mean_at_db", int(mult), int(cap))
        cached = self._cached_series.get(cache_key)
        if cached is not None:
            return cached
        ln_at_db = self._tsrv_log_vol_at_db()
        if ln_at_db is None:
            return None
        bpd = self._bpd()
        w = min(max(bpd * mult, 10), cap)
        mp = max(w // 4, 10)
        ser = ln_at_db.rolling(w, min_periods=mp).mean()
        self._cached_series[cache_key] = ser
        return ser

    # ── tsrv_ffd (frac_diff on dollar-bar grid) ─────────────────────

    def _tsrv_ffd_at_db(self) -> pd.Series | None:
        cache_key = ("_tsrv_ffd_at_db",)
        cached = self._cached_series.get(cache_key)
        if cached is not None:
            return cached
        ln_at_db = self._tsrv_log_vol_at_db()
        if ln_at_db is None or len(ln_at_db) < 30:
            return None
        try:
            from src.afml.fracdiff import frac_diff_ffd
        except ImportError:
            logger.warning(
                "TSRVDollarBarFeatures: src.afml.fracdiff missing; "
                "falling back to None (handler uses baseline)"
            )
            return None
        try:
            ffd = frac_diff_ffd(
                ln_at_db.to_frame(name="tsrv"),
                d=_FRACDIFF_D, threshold=_FRACDIFF_THRESHOLD,
            )
        except Exception:  # noqa: BLE001
            logger.debug("frac_diff_ffd failed", exc_info=True)
            return None
        if ffd is None or "tsrv" not in ffd.columns or ffd["tsrv"].empty:
            return None
        ser = ffd["tsrv"].reindex(ln_at_db.index)
        self._cached_series[cache_key] = ser
        return ser

    # ── bpd (matches offline avg_bars_per_day) ──────────────────────

    def _bpd(self, *, trailing_days: int = 30) -> int:
        """Same formula as offline ``avg_bars_per_day`` and
        ``DollarBarFeatures.bars_per_day``. Floor at 10."""
        self._refresh()
        if self._cached_db_idx is None or len(self._cached_db_idx) < 2:
            return 250  # cold-start fallback
        idx = self._cached_db_idx
        cutoff = idx[-1] - pd.Timedelta(days=trailing_days)
        recent = idx[idx >= cutoff]
        if len(recent) < 2:
            return 250
        n_days = max((recent[-1] - recent[0]).days, 1)
        return max(int(len(recent) / n_days), 10)

    # ── public API: per-feature raw + lag + ema ─────────────────────

    @staticmethod
    def _series_lag(ser: pd.Series | None, n: int) -> float | None:
        if ser is None or len(ser) <= n:
            return None
        v = ser.iloc[-1 - n]
        return None if not np.isfinite(v) else float(v)

    @staticmethod
    def _series_ema_at_last(
        ser: pd.Series | None, span: int,
    ) -> float | None:
        if ser is None or ser.empty:
            return None
        ema = ser.ewm(span=span, adjust=True, min_periods=1).mean()
        v = ema.iloc[-1]
        return None if not np.isfinite(v) else float(v)

    # ── tsrv_mean_1d ────────────────────────────────────────────────

    def tsrv_mean_1d(self) -> float | None:
        return _last_finite(self._tsrv_mean_at_db(1))

    def tsrv_mean_1d_lag(self, n: int) -> float | None:
        return self._series_lag(self._tsrv_mean_at_db(1), n)

    def tsrv_mean_1d_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._tsrv_mean_at_db(1), span)

    # ── tsrv_mean_1w ────────────────────────────────────────────────

    def tsrv_mean_1w(self) -> float | None:
        return _last_finite(self._tsrv_mean_at_db(7))

    def tsrv_mean_1w_lag(self, n: int) -> float | None:
        return self._series_lag(self._tsrv_mean_at_db(7), n)

    def tsrv_mean_1w_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._tsrv_mean_at_db(7), span)

    # ── tsrv_ffd ────────────────────────────────────────────────────

    def tsrv_ffd(self) -> float | None:
        return _last_finite(self._tsrv_ffd_at_db())

    def tsrv_ffd_lag(self, n: int) -> float | None:
        return self._series_lag(self._tsrv_ffd_at_db(), n)

    def tsrv_ffd_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._tsrv_ffd_at_db(), span)

    # ── down_jump_frac ──────────────────────────────────────────────

    def down_jump_frac(self) -> float | None:
        return _last_finite(self._down_jump_frac_at_db())

    def down_jump_frac_lag(self, n: int) -> float | None:
        return self._series_lag(self._down_jump_frac_at_db(), n)

    def down_jump_frac_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._down_jump_frac_at_db(), span)

    # ── up_jump_frac ────────────────────────────────────────────────

    def up_jump_frac(self) -> float | None:
        return _last_finite(self._up_jump_frac_at_db())

    def up_jump_frac_lag(self, n: int) -> float | None:
        return self._series_lag(self._up_jump_frac_at_db(), n)

    def up_jump_frac_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._up_jump_frac_at_db(), span)

    # ── noise_level ─────────────────────────────────────────────────

    def noise_level(self) -> float | None:
        return _last_finite(self._noise_level_at_db())

    def noise_level_lag(self, n: int) -> float | None:
        return self._series_lag(self._noise_level_at_db(), n)

    def noise_level_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._noise_level_at_db(), span)

    # ── bpv_tsrv_ratio ──────────────────────────────────────────────

    def bpv_tsrv_ratio(self) -> float | None:
        return _last_finite(self._bpv_tsrv_ratio_at_db())

    def bpv_tsrv_ratio_lag(self, n: int) -> float | None:
        return self._series_lag(self._bpv_tsrv_ratio_at_db(), n)

    def bpv_tsrv_ratio_ema(self, span: int) -> float | None:
        return self._series_ema_at_last(self._bpv_tsrv_ratio_at_db(), span)


__all__ = ["TSRVDollarBarFeatures"]
