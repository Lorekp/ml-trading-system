"""Compute lag/EMA/parkinson features over a ``DollarBarBuffer``.

F3 Tier 3 (parity-cycle round 3 — 2026-05-07): wraps a
:class:`DollarBarBuffer` and exposes the offline lag/EMA/parkinson
semantics the live handlers in
``deribit_paper/signal/feature_assembler.py`` need.

Caches the buffer's DataFrame view, the VPIN series (one VPIN per
dollar bar via :func:`vol_model.features.microstructure._vpin_rolling`),
and the per-bar parkinson series — all of which would otherwise be
recomputed for every handler call within a single bar.

Byte-identity contract: every method must reproduce the offline
``vol_model/features/engineer.py::_window_and_aggregate`` logic on the
buffer's ``as_dataframe()`` view to 1e-12 relative tolerance.

Offline reference (``vol_model/features/engineer.py:426-470``)::

    for col in raw_feats.columns:
        series = raw_feats[col]
        result_cols[col] = series.groupby(window_labels).last()
        for lag in self.config.lags:           # [1, 3]
            shifted = series.shift(lag)
            result_cols[f"{col}_lag{lag}"] = shifted.groupby(window_labels).last()
        for span in self.config.ema_spans:     # [10, 50]
            ema = series.ewm(span=span, min_periods=1).mean()
            result_cols[f"{col}_ema{span}"] = ema.groupby(window_labels).last()

The ``.groupby(window_labels).last()`` step is implicit in live: when
the engine fires ``process_bar`` at the 15-min wall-clock boundary, it
queries the dollar-bar buffer's CURRENT state — which is the value of
the last dollar bar emitted before the boundary, exactly what the
offline groupby-last produces.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer
from vol_model.features.microstructure import _vpin_rolling

logger = logging.getLogger(__name__)


_RAW_NAMES: tuple[str, ...] = ("vov", "vpin", "imbalance", "duration_s", "n_trades")


class DollarBarFeatures:
    """Cached view over a DollarBarBuffer for live feature computation.

    Construct one per buffer; the cache is invalidated whenever the
    buffer length changes (new bar appended). Idempotent — repeat calls
    inside a single bar boundary cost ~1µs after the first.

    All methods return ``None`` when the buffer is too short for the
    requested operation. The caller (handler) maps that to a baseline
    fallback (matches the HAR-PD warmup pattern in feature_assembler).
    """

    def __init__(self, buffer: DollarBarBuffer):
        self._buffer = buffer
        self._cached_len: int = -1
        self._cached_df: pd.DataFrame | None = None
        self._cached_vpin: np.ndarray | None = None
        self._cached_parkinson_bar: np.ndarray | None = None
        # F-2 fix (round-3 parity audit, 2026-05-08): rolling mean of the
        # per-bar parkinson series — distinct from the raw per-bar series.
        # The offline pipeline stores ``feats["parkinson_1d"]`` as a
        # rolling mean (``parkinson_bar.rolling(w_park).mean()``); the
        # subsequent ``parkinson_1d_ema50`` is then ``EMA(rolling_mean)``,
        # NOT ``EMA(raw_bar_series)``. Pre-fix live EMA'd the raw series
        # — same shape but wildly different reactivity. We cache by
        # window so different callers (handler with window=250 today,
        # F-3's bpd-driven future) don't recompute.
        self._cached_parkinson_rolling: dict[int, np.ndarray] = {}

    # ── cache management ─────────────────────────────────────────────

    def _refresh(self) -> None:
        n = len(self._buffer)
        if n == self._cached_len and self._cached_df is not None:
            return
        self._cached_len = n
        self._cached_df = self._buffer.as_dataframe()
        # Invalidate dependent caches when length changes
        self._cached_vpin = None
        self._cached_parkinson_bar = None
        self._cached_parkinson_rolling.clear()

    def _df(self) -> pd.DataFrame:
        self._refresh()
        return self._cached_df

    def _vpin_series(self) -> np.ndarray | None:
        """Per-dollar-bar VPIN series via the offline ``_vpin_rolling``.

        First 49 entries are NaN (rolling 50-bucket window). Returns
        ``None`` when the buffer has fewer than 50 bars (no defined
        VPIN at all).
        """
        self._refresh()
        if self._cached_vpin is not None:
            return self._cached_vpin
        df = self._cached_df
        if df is None or len(df) < 50:
            return None
        buy = df["buy_volume"].to_numpy(dtype=np.float64)
        sell = df["sell_volume"].to_numpy(dtype=np.float64)
        self._cached_vpin = _vpin_rolling(buy, sell, 50)
        return self._cached_vpin

    def _parkinson_bar_series(self) -> np.ndarray | None:
        """Per-dollar-bar parkinson values: ``log(high/low)^2 / (4*log(2))``.

        Mirror of ``vol_model/features/engineer.py:359-360``::

            hl_ratio = (bars["high"] / bars["low"]).clip(lower=1.0 + 1e-10)
            parkinson_bar = np.log(hl_ratio) ** 2 / (4.0 * np.log(2.0))
        """
        self._refresh()
        if self._cached_parkinson_bar is not None:
            return self._cached_parkinson_bar
        df = self._cached_df
        if df is None or len(df) == 0:
            return None
        hl = (df["high"] / df["low"]).clip(lower=1.0 + 1e-10)
        self._cached_parkinson_bar = (
            np.log(hl) ** 2 / (4.0 * np.log(2.0))
        ).to_numpy(dtype=np.float64)
        return self._cached_parkinson_bar

    def _parkinson_1d_rolling_series(
        self, window: int,
    ) -> np.ndarray | None:
        """Rolling mean of the per-bar parkinson values.

        F-2 fix (round-3 parity audit, 2026-05-08): mirrors offline's
        ``feats["parkinson_1d"] = parkinson_bar.rolling(w_park,
        min_periods=max(w_park//4, 10)).mean()`` (``engineer.py:362``).
        Pandas' rolling fills the head with NaN until ``min_periods``
        bars are observed; we preserve that NaN prefix so downstream
        EMA matches offline byte-for-byte (pandas EMA ignores leading
        NaNs).

        Cached per-window because F-3 (round-3 audit, P1) will plumb
        the dynamic ``bpd`` width — different callers may want different
        window sizes within one bar.
        """
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        # IMPORTANT: refresh BEFORE checking the cache so a buffer append
        # since the last call invalidates this dict. ``_refresh`` clears
        # ``_cached_parkinson_rolling`` when buffer length changes.
        self._refresh()
        cached = self._cached_parkinson_rolling.get(window)
        if cached is not None:
            return cached
        bar_arr = self._parkinson_bar_series()
        if bar_arr is None or len(bar_arr) == 0:
            return None
        min_periods = max(window // 4, 10)
        # Use pandas rolling for byte-identity to offline (engineer.py:362-363).
        rolling = pd.Series(bar_arr).rolling(
            window=window, min_periods=min_periods,
        ).mean().to_numpy(dtype=np.float64)
        self._cached_parkinson_rolling[window] = rolling
        return rolling

    # ── public API ──────────────────────────────────────────────────

    def latest_raw(self, name: str) -> float | None:
        """Return the most-recent dollar bar's raw value for ``name``.

        Special-case: ``"vpin"`` calls ``buffer.compute_vpin(50)`` which
        returns the rolling-50 VPIN at the latest bar (or None if
        underfull). Other names read directly from ``buffer.latest_bar()``.
        """
        if name == "vpin":
            return self._buffer.compute_vpin(50)
        bar = self._buffer.latest_bar()
        if bar is None:
            return None
        if name == "vov":
            return float(bar.vov)
        if name == "imbalance":
            return float(bar.imbalance)
        if name == "duration_s":
            return float(bar.duration_s)
        if name == "n_trades":
            return float(bar.n_trades)
        raise KeyError(f"Unknown raw feature name: {name!r}")

    def ema(self, name: str, span: int) -> float | None:
        """Return the EWM(span) value at the latest bar.

        Mirrors offline ``series.ewm(span=span, adjust=True, min_periods=1).mean()``
        evaluated at the last index. For ``"vpin"`` the underlying series is
        the per-bar VPIN computed via :func:`_vpin_rolling` (with its 49-NaN
        prefix preserved — pandas ``ewm`` ignores NaN positions, matching
        how the offline pipeline composes VPIN+EMA).

        Returns ``None`` when the buffer is too short for the underlying
        series to exist (vpin needs >=50 bars).
        """
        if span <= 0:
            raise ValueError(f"span must be positive, got {span}")
        if name == "vpin":
            arr = self._vpin_series()
            if arr is None:
                return None
            ser = pd.Series(arr)
            ema_val = ser.ewm(
                span=span, adjust=True, min_periods=1,
            ).mean().iloc[-1]
            return None if not np.isfinite(ema_val) else float(ema_val)
        df = self._df()
        if df is None or len(df) == 0:
            return None
        col = df[name].astype(np.float64)
        return float(col.ewm(span=span, adjust=True, min_periods=1).mean().iloc[-1])

    def lag(self, name: str, n: int) -> float | None:
        """Return the value of ``name`` at ``n`` bars before the latest.

        ``n=1`` means the previous bar (one before latest). ``n=3`` means
        four bars from the end (matches offline ``series.shift(3).iloc[-1]``
        which is ``series.iloc[-4]``).

        Returns ``None`` when the buffer doesn't have enough history.
        """
        if n < 1:
            raise ValueError(f"lag n must be >= 1, got {n}")
        if name == "vpin":
            arr = self._vpin_series()
            if arr is None or len(arr) <= n:
                return None
            v = arr[-1 - n]
            return None if not np.isfinite(v) else float(v)
        df = self._df()
        if df is None or len(df) <= n:
            return None
        v = df[name].iloc[-1 - n]
        return None if (v is None or not np.isfinite(v)) else float(v)

    def parkinson_1d(self, window: int = 250) -> float | None:
        """Latest value of the rolling-mean parkinson series.

        Mirror of offline ``feats["parkinson_1d"]`` evaluated at the
        last index. Returns ``None`` when the rolling-mean series
        hasn't reached its ``min_periods`` warmup floor at the latest
        bar (matches offline's NaN propagation through the warmup
        head).
        """
        arr = self._parkinson_1d_rolling_series(window)
        if arr is None or len(arr) == 0:
            return None
        v = arr[-1]
        return None if not np.isfinite(v) else float(v)

    def parkinson_1d_ema(
        self, span: int = 50, *, rolling_window: int = 250,
    ) -> float | None:
        """EWM(span) over the parkinson_1d ROLLING-MEAN series.

        F-2 fix (round-3 parity audit, 2026-05-08): pre-fix this EMAed
        the raw per-bar parkinson series, which is wildly more reactive
        than offline. Offline `_window_and_aggregate` (engineer.py:460)
        computes `series.ewm(span=span).mean()` over the column
        `feats["parkinson_1d"]` — which is itself the rolling-mean of
        the per-bar parkinson (engineer.py:362-363). Live now matches
        that two-stage transformation byte-for-byte.

        ``rolling_window`` is the inner rolling-mean window before
        EMAing. Default 250 mirrors the production handler binding
        (``_LIVE_HANDLERS["parkinson_1d_ema50"]`` in
        ``feature_assembler.py``); F-3 (round-3 P1) will plumb the
        dynamic ``bpd`` value here.
        """
        if span <= 0:
            raise ValueError(f"span must be positive, got {span}")
        arr = self._parkinson_1d_rolling_series(rolling_window)
        if arr is None or len(arr) == 0:
            return None
        # pandas ewm with adjust=True + min_periods=1 ignores leading
        # NaNs (the rolling-mean head before min_periods is satisfied).
        ema_val = pd.Series(arr).ewm(
            span=span, adjust=True, min_periods=1,
        ).mean().iloc[-1]
        return None if not np.isfinite(ema_val) else float(ema_val)


__all__ = ["DollarBarFeatures"]
