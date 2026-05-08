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


# ── S-1b (parity-r3 cycle 5 — 2026-05-08): FFD weights cache ────────
# Fractional differentiation weights are deterministic in (d, threshold)
# so we cache per (d, threshold) pair across the process lifetime.
# Mirrors offline ``src/afml/fracdiff.get_weights_ffd`` byte-for-byte
# (same iterative recurrence + same threshold cutoff).
_FFD_WEIGHTS_CACHE: dict[tuple[float, float], np.ndarray] = {}


def _ffd_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Cache-backed FFD weight vector. Returns weights chronological
    (oldest first), shape (n_weights,) — flattened from the offline
    column-vector convention so a per-bar dot-product is a single
    ``np.dot`` call.

    Mirrors `src/afml/fracdiff.get_weights_ffd` byte-for-byte.
    """
    key = (float(d), float(threshold))
    cached = _FFD_WEIGHTS_CACHE.get(key)
    if cached is not None:
        return cached
    # Iterative recurrence: w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k.
    # Truncate at |w_k| < threshold.
    weights: list[float] = [1.0]
    k = 1
    while True:
        wk = -weights[-1] * (d - k + 1) / k
        if abs(wk) < threshold:
            break
        weights.append(wk)
        k += 1
        if k > 10000:  # defensive cap
            break
    arr = np.asarray(weights, dtype=np.float64)
    # Convolve with reversed weights elsewhere; we cache chronological.
    _FFD_WEIGHTS_CACHE[key] = arr
    return arr


def _hurst_subsampled_window(returns: np.ndarray, window: int = 100) -> float | None:
    """Compute a single Hurst exponent via R/S over the last ``window``
    bars of ``returns``.

    Live-equivalent of the per-position computation inside offline's
    ``vol_model/features/engineer.py::_hurst_subsampled``. The offline
    version computes Hurst at every ``step`` bars over the full series
    and linearly interpolates between steps; live only needs the
    LATEST value, so we evaluate once on the trailing-``window`` slice.
    Returns None when the window has too few finite returns or when the
    R/S regression would be degenerate.
    """
    if len(returns) < window:
        return None
    chunk = returns[-window:]
    valid = chunk[~np.isnan(chunk)]
    if len(valid) < 20:
        return None
    sub_sizes = [s for s in (8, 16, 32, 64) if s <= window // 2]
    if len(sub_sizes) < 2:
        return None
    log_rs: list[float] = []
    log_n: list[float] = []
    for size in sub_sizes:
        nc = len(valid) // size
        if nc < 1:
            continue
        rs_list: list[float] = []
        for j in range(nc):
            seg = valid[j * size:(j + 1) * size]
            std = seg.std(ddof=1)
            if std < 1e-15:
                continue
            cd = np.cumsum(seg - seg.mean())
            rs_list.append((cd.max() - cd.min()) / std)
        if rs_list:
            log_rs.append(np.log(np.mean(rs_list)))
            log_n.append(np.log(size))
    if len(log_rs) < 2:
        return None
    slope = np.polyfit(log_n, log_rs, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


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
        # S-1c (parity-r3 cycle 6 — 2026-05-08): cache for derived
        # bar-grid SERIES used by lag/EMA variants of hurst_100, ffd_rv,
        # and rv_zscore_{1d,1w,1m}. Keyed on ``(name, params...)``;
        # invalidated together with ``_cached_df`` on every buffer
        # length change. Each entry is a numpy array aligned 1:1 with
        # the buffer's bars (NaN-padded prefix for warmup positions).
        self._cached_derived: dict[tuple, np.ndarray] = {}

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
        self._cached_derived.clear()

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

    def bars_per_day(self, *, trailing_days: int = 30) -> int:
        """Estimate the dollar-bar density (bars per UTC day).

        F-3 (parity-r3 cycle 2 — 2026-05-08): mirrors offline's
        ``vol_model/features/engineer.py::avg_bars_per_day`` byte-for-byte.
        Replaces the hardcoded ``window=250`` that the parkinson_1d /
        parkinson_1d_ema50 handlers used pre-fix — offline derives ``bpd``
        from the trailing 30 days of bar timestamps per training run, so
        the rolling-mean window matches per-asset / per-regime activity.

        Floor at 10 (matches offline) so a near-empty buffer can't push
        the rolling window below the offline ``min_periods`` floor.
        """
        self._refresh()
        df = self._cached_df
        if df is None or len(df) < 2:
            return 250  # F-3 cold-start fallback (matches pre-fix behaviour)
        idx = df.index
        cutoff = idx[-1] - pd.Timedelta(days=trailing_days)
        recent = idx[idx >= cutoff]
        if len(recent) < 2:
            return 250
        n_days = max((recent[-1] - recent[0]).days, 1)
        return max(int(len(recent) / n_days), 10)

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

    # ── S-1b (parity-r3 cycle 5): hurst_100 / ffd_rv / rv_zscore_* ──
    # All three feed the TransVol encoder's 101-feature input row.
    # Mirrors offline `vol_model/features/engineer.py:222-240` byte-for-byte
    # except for evaluation site: live only needs the LATEST bar's value
    # (offline computes the full series for training).

    def hurst_100(self, *, window: int = 100) -> float | None:
        """Hurst exponent on the trailing ``window`` log-returns.

        Mirrors offline ``_hurst_subsampled(returns, window=100)`` evaluated
        at the latest bar. The offline version interpolates between
        ``step=50`` Hurst evaluations across the full series; live takes
        the trailing window's instantaneous Hurst, which equals offline's
        value at every position in {window, window+step, ...}.
        """
        df = self._df()
        if df is None or "log_return" not in df.columns:
            return None
        log_ret = df["log_return"].to_numpy(dtype=np.float64)
        return _hurst_subsampled_window(log_ret, window=window)

    def ffd_rv(
        self, *, d: float = 0.4, threshold: float = 1e-5,
    ) -> float | None:
        """Latest fractionally-differenced log-close value.

        Mirrors offline ``frac_diff_ffd(np.log(bars["close"]), d, threshold)``
        evaluated at the last index. The dot-product over the cached
        weight vector matches ``np.convolve`` exactly when we walk the
        weights in the conventional chronological-against-most-recent
        order.
        """
        df = self._df()
        if df is None or "close" not in df.columns:
            return None
        weights = _ffd_weights(d=d, threshold=threshold)
        n_weights = len(weights)
        if len(df) < n_weights:
            return None
        log_close = np.log(
            df["close"].to_numpy(dtype=np.float64)[-n_weights:]
        )
        # Offline: np.convolve(values, weights[::-1], mode="valid"). For
        # the LAST output (the only one we want) the convolution sum is
        # ``sum(values[-1-k] * weights[k] for k in range(n_weights))``,
        # which equals ``np.dot(values[-n_weights:][::-1], weights)``.
        result = float(np.dot(log_close[::-1], weights))
        return None if not np.isfinite(result) else result

    def _rv_bar_series(self) -> np.ndarray | None:
        """Cached numpy view of the buffer's per-bar realised variance."""
        df = self._df()
        if df is None or "rv_bar" not in df.columns:
            return None
        return df["rv_bar"].to_numpy(dtype=np.float64)

    def rv_zscore(
        self, horizon: str, *, bpd: int | None = None,
    ) -> float | None:
        """Rolling z-score of realised variance over a horizon.

        Mirrors offline ``vol_model/features/engineer.py:233-240``::

            for mult, label, cap in [(1, "1d", 50_000), (7, "1w", 50_000),
                                     (30, "1m", 200_000)]:
                w = min(max(bpd * mult, 10), cap)
                mp = max(w // 4, 10)
                rv_roll = rv_bar.rolling(w, min_periods=mp).sum()
                mu = rv_roll.rolling(w, min_periods=mp).mean()
                sigma = rv_roll.rolling(w, min_periods=mp).std()
                feats[f"rv_zscore_{label}"] = (rv_roll - mu) / sigma.clip(lower=1e-15)

        ``horizon`` ∈ {"1d", "1w", "1m"} → multipliers {1, 7, 30} and
        caps {50000, 50000, 200000}.
        """
        if horizon not in ("1d", "1w", "1m"):
            raise ValueError(f"horizon must be 1d/1w/1m, got {horizon!r}")
        rv_bar = self._rv_bar_series()
        if rv_bar is None or len(rv_bar) < 20:
            return None
        if bpd is None:
            bpd = self.bars_per_day(trailing_days=30)
        mult, cap = {"1d": (1, 50_000), "1w": (7, 50_000), "1m": (30, 200_000)}[horizon]
        w = min(max(bpd * mult, 10), cap)
        mp = max(w // 4, 10)
        ser = pd.Series(rv_bar)
        rv_roll = ser.rolling(w, min_periods=mp).sum()
        mu = rv_roll.rolling(w, min_periods=mp).mean()
        sigma = rv_roll.rolling(w, min_periods=mp).std()
        denom = sigma.clip(lower=1e-15)
        z = (rv_roll - mu) / denom
        v = z.iloc[-1]
        return None if not np.isfinite(v) else float(v)

    # ── S-1c (parity-r3 cycle 6): derived-series + lag/EMA variants ──
    # The 4 windowed variants (lag1, lag3, ema10, ema50) of hurst_100,
    # ffd_rv, and rv_zscore_{1d,1w,1m} need to be computed against the
    # FULL bar-grid series (offline ``_window_and_aggregate`` evaluates
    # ``shift(lag).iloc[-1]`` and ``ewm(span).mean().iloc[-1]`` over the
    # complete pre-aggregation column). We compute the trailing series
    # once per bar advance, cache, and have lag/ema methods read from it.

    def _hurst_series(self, *, window: int = 100) -> np.ndarray | None:
        """Per-bar Hurst series aligned 1:1 with ``self._df()``.

        Each entry is the Hurst computed via ``_hurst_subsampled_window``
        on ``log_return.iloc[i-window:i]`` for that bar's position; NaN
        for positions where the trailing slice is too short / degenerate.
        """
        key = ("hurst_series", int(window))
        cached = self._cached_derived.get(key)
        if cached is not None:
            return cached
        df = self._df()
        if df is None or "log_return" not in df.columns:
            return None
        log_ret = df["log_return"].to_numpy(dtype=np.float64)
        n = len(log_ret)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(window, n + 1):
            v = _hurst_subsampled_window(log_ret[:i], window=window)
            if v is not None:
                out[i - 1] = v
        self._cached_derived[key] = out
        return out

    def hurst_100_lag(self, n: int, *, window: int = 100) -> float | None:
        """``hurst_100.shift(n).iloc[-1]`` — value n bars before latest."""
        if n < 1:
            raise ValueError(f"lag n must be >= 1, got {n}")
        arr = self._hurst_series(window=window)
        if arr is None or len(arr) <= n:
            return None
        v = arr[-1 - n]
        return None if not np.isfinite(v) else float(v)

    def hurst_100_ema(self, span: int, *, window: int = 100) -> float | None:
        """``hurst_100.ewm(span=span, adjust=True, min_periods=1).mean().iloc[-1]``."""
        if span <= 0:
            raise ValueError(f"span must be positive, got {span}")
        arr = self._hurst_series(window=window)
        if arr is None or len(arr) == 0:
            return None
        ema_val = pd.Series(arr).ewm(
            span=span, adjust=True, min_periods=1,
        ).mean().iloc[-1]
        return None if not np.isfinite(ema_val) else float(ema_val)

    def _ffd_rv_series(
        self, *, d: float = 0.4, threshold: float = 1e-5,
    ) -> np.ndarray | None:
        """Per-bar FFD series aligned 1:1 with ``self._df()``.

        Offline does ``np.convolve(values, get_weights_ffd[::-1])`` where
        ``get_weights_ffd`` returns weights in REVERSED chronological
        order (oldest weight at the END), so ``get_weights_ffd[::-1]``
        is the chronological-ordered vector. Live's ``_ffd_weights``
        already stores chronological — so the convolution argument is
        ``weights`` directly, NOT ``weights[::-1]``.
        """
        key = ("ffd_rv_series", float(d), float(threshold))
        cached = self._cached_derived.get(key)
        if cached is not None:
            return cached
        df = self._df()
        if df is None or "close" not in df.columns:
            return None
        weights = _ffd_weights(d=d, threshold=threshold)  # chronological
        n_w = len(weights)
        log_close = np.log(df["close"].to_numpy(dtype=np.float64))
        n = len(log_close)
        out = np.full(n, np.nan, dtype=np.float64)
        if n >= n_w:
            # Equivalent to offline's np.convolve(values,
            # get_weights_ffd[::-1]) since live's weights are already
            # chronological. Aligns to indices [n_w-1 .. n-1].
            conv = np.convolve(log_close, weights, mode="valid")
            out[n_w - 1:] = conv
        self._cached_derived[key] = out
        return out

    def ffd_rv_lag(
        self, n: int, *, d: float = 0.4, threshold: float = 1e-5,
    ) -> float | None:
        if n < 1:
            raise ValueError(f"lag n must be >= 1, got {n}")
        arr = self._ffd_rv_series(d=d, threshold=threshold)
        if arr is None or len(arr) <= n:
            return None
        v = arr[-1 - n]
        return None if not np.isfinite(v) else float(v)

    def ffd_rv_ema(
        self, span: int, *, d: float = 0.4, threshold: float = 1e-5,
    ) -> float | None:
        if span <= 0:
            raise ValueError(f"span must be positive, got {span}")
        arr = self._ffd_rv_series(d=d, threshold=threshold)
        if arr is None or len(arr) == 0:
            return None
        ema_val = pd.Series(arr).ewm(
            span=span, adjust=True, min_periods=1,
        ).mean().iloc[-1]
        return None if not np.isfinite(ema_val) else float(ema_val)

    def _rv_zscore_series(
        self, horizon: str, *, bpd: int | None = None,
    ) -> np.ndarray | None:
        """Per-bar rv_zscore series aligned 1:1 with ``self._df()``.

        Vectorised pandas rolling chain — one shot for the full series.
        """
        if horizon not in ("1d", "1w", "1m"):
            raise ValueError(f"horizon must be 1d/1w/1m, got {horizon!r}")
        if bpd is None:
            bpd = self.bars_per_day(trailing_days=30)
        key = ("rv_zscore_series", horizon, int(bpd))
        cached = self._cached_derived.get(key)
        if cached is not None:
            return cached
        rv_bar = self._rv_bar_series()
        if rv_bar is None or len(rv_bar) == 0:
            return None
        mult, cap = {"1d": (1, 50_000), "1w": (7, 50_000), "1m": (30, 200_000)}[horizon]
        w = min(max(bpd * mult, 10), cap)
        mp = max(w // 4, 10)
        ser = pd.Series(rv_bar)
        rv_roll = ser.rolling(w, min_periods=mp).sum()
        mu = rv_roll.rolling(w, min_periods=mp).mean()
        sigma = rv_roll.rolling(w, min_periods=mp).std()
        denom = sigma.clip(lower=1e-15)
        z = (rv_roll - mu) / denom
        out = z.to_numpy(dtype=np.float64)
        self._cached_derived[key] = out
        return out

    def rv_zscore_lag(
        self, horizon: str, n: int, *, bpd: int | None = None,
    ) -> float | None:
        if n < 1:
            raise ValueError(f"lag n must be >= 1, got {n}")
        arr = self._rv_zscore_series(horizon, bpd=bpd)
        if arr is None or len(arr) <= n:
            return None
        v = arr[-1 - n]
        return None if not np.isfinite(v) else float(v)

    def rv_zscore_ema(
        self, horizon: str, span: int, *, bpd: int | None = None,
    ) -> float | None:
        if span <= 0:
            raise ValueError(f"span must be positive, got {span}")
        arr = self._rv_zscore_series(horizon, bpd=bpd)
        if arr is None or len(arr) == 0:
            return None
        ema_val = pd.Series(arr).ewm(
            span=span, adjust=True, min_periods=1,
        ).mean().iloc[-1]
        return None if not np.isfinite(ema_val) else float(ema_val)

    def parkinson_1d_lag(
        self, n: int, *, rolling_window: int = 250,
    ) -> float | None:
        """``parkinson_1d.shift(n).iloc[-1]`` over the rolling-mean series."""
        if n < 1:
            raise ValueError(f"lag n must be >= 1, got {n}")
        arr = self._parkinson_1d_rolling_series(rolling_window)
        if arr is None or len(arr) <= n:
            return None
        v = arr[-1 - n]
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
