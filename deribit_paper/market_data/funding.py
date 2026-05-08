"""Live funding-rate provider — S-1c (parity-r3 cycle 6 — 2026-05-08).

The TransVol encoder consumes 5 ``funding_basis_*`` columns. Offline
(``vol_model/features/engineer.py:407-415``) defines them as::

    fr = pd.read_parquet(funding_dir / f"{asset}_funding.parquet")["fundingRate"]
    feats["funding_basis"] = fr.reindex(ci).ffill().reindex(bars.index)

— i.e. the most recent Binance perp fundingRate at or before each
15-min bar timestamp, forward-filled across the 32 bars between
funding refreshes (Binance funds every 8h on the 00:00/08:00/16:00 UTC
boundaries).

This module supplies the live equivalent. Two paths, in order:

1. **Path A (preferred)** — ``BinanceFundingRESTClient`` polls
   ``GET /fapi/v1/premiumIndex?symbol=BTCUSDT`` (or the equivalent for
   other symbols). The endpoint returns the *current* mark price +
   ``lastFundingRate`` value; we cache it with a TTL slightly under the
   8h funding cadence (default 1h) and refetch when stale. This matches
   the offline parquet's source (Binance perp), so the model sees the
   same numerical distribution it trained on.

2. **Path B (fallback)** — read the historical parquet at
   ``data/funding-rates/{asset}_funding.parquet`` once at startup;
   maintain the series in memory. On every bar advance the provider
   "ffills" — exposes the most recent fundingRate whose ``fundingTime
   <= bar.timestamp``. Falls through to ``0.0`` only when both A and
   B are unavailable (matches offline's ``feats["funding_basis"] = 0.0``
   default at ``engineer.py:415``).

The provider is **bar-grid sampled**: ``advance(bar.timestamp)`` is
called once per 15-min bar by the engine; the provider records the
ffilled value at that bar so the lag/EMA variants compute over the
same 15-min cadence as offline. The internal deque holds the last
``history_bars`` 15-min readings (default 200, enough for ema50 to
converge to <1e-10 contribution from the oldest entry).

Behaviour matches offline byte-for-byte when the live engine has
the historical parquet; the REST path adds *forward* freshness when
the parquet hasn't been refreshed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default REST cadence (Binance funds every 8h; refetch hourly to stay
# fresh without spamming the endpoint).
_DEFAULT_REST_TTL_SECONDS: float = 3600.0
# Default trailing window for the lag/EMA variants on the 15-min grid.
_DEFAULT_HISTORY_BARS: int = 200


@dataclass(frozen=True)
class FundingObservation:
    """One ``fundingRate`` reading from any source."""
    funding_time: pd.Timestamp     # UTC, the rate's official effective time
    funding_rate: float


def _binance_premium_index_fetcher(symbol: str) -> Callable[[], FundingObservation | None]:
    """Build a closure that fetches Binance's ``/fapi/v1/premiumIndex``.

    Returns ``None`` on any HTTP/network error so the caller can fall
    through to the parquet path. Constructed lazily so importing this
    module never triggers a network call.
    """
    def _fetch() -> FundingObservation | None:
        try:
            import requests  # type: ignore
        except ImportError:
            logger.warning(
                "FundingRateProvider: 'requests' not installed; REST path "
                "disabled. Falling back to parquet-only mode.",
            )
            return None
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        try:
            resp = requests.get(url, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Binance premiumIndex fetch failed: %s", exc)
            return None
        try:
            funding_time = pd.Timestamp(int(data["nextFundingTime"]), unit="ms", tz="UTC")
            # nextFundingTime is the *next* refresh; subtract 8h to get
            # the boundary the current value applies from.
            funding_time = funding_time - pd.Timedelta(hours=8)
            funding_rate = float(data["lastFundingRate"])
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Binance premiumIndex parse failed: %s (data=%r)", exc, data)
            return None
        return FundingObservation(funding_time=funding_time, funding_rate=funding_rate)

    return _fetch


class FundingRateProvider:
    """Per-bar funding-rate accessor with REST + parquet fallback paths.

    Lifecycle:

    1. ``__init__`` — load parquet (Path B) eagerly; set up REST closure.
    2. ``advance(bar_ts)`` — called once per 15-min bar by the engine.
       Refreshes REST if TTL expired; appends current ffill value to
       the per-bar history deque.
    3. ``latest()`` / ``latest_lag(n)`` / ``latest_ema(span)`` — read
       from the bar-grid history deque.

    Thread-safe (engine bar loop is single-threaded but tests may
    concurrently advance from a fixture).
    """

    def __init__(
        self,
        *,
        symbol: str = "BTCUSDT",
        parquet_path: str | Path | None = None,
        rest_fetcher: Callable[[], FundingObservation | None] | None = None,
        rest_ttl_seconds: float = _DEFAULT_REST_TTL_SECONDS,
        history_bars: int = _DEFAULT_HISTORY_BARS,
        default_value: float = 0.0,
    ) -> None:
        self._symbol = symbol
        self._rest_ttl = float(rest_ttl_seconds)
        self._default_value = float(default_value)
        self._lock = threading.Lock()

        # Path A: REST closure (None disables the path).
        if rest_fetcher is None:
            self._rest_fetcher = _binance_premium_index_fetcher(symbol)
        else:
            self._rest_fetcher = rest_fetcher
        self._rest_last_fetch_monotonic: float | None = None
        self._rest_last_observation: FundingObservation | None = None

        # Path B: parquet-loaded series (sorted by funding_time ascending).
        self._parquet_observations: list[FundingObservation] = []
        if parquet_path is not None:
            try:
                self._parquet_observations = self._load_parquet(parquet_path)
                logger.info(
                    "FundingRateProvider: loaded %d Path-B observations "
                    "from %s (latest=%s)",
                    len(self._parquet_observations), parquet_path,
                    (self._parquet_observations[-1].funding_time
                     if self._parquet_observations else "n/a"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FundingRateProvider: parquet load failed (%s); "
                    "Path B disabled, falling back to REST + default",
                    exc,
                )
                self._parquet_observations = []

        # Bar-grid history (one entry per advance() call).
        self._history: deque[tuple[pd.Timestamp, float]] = deque(
            maxlen=int(history_bars),
        )

    # ── Path B parquet load ─────────────────────────────────────

    @staticmethod
    def _load_parquet(path: str | Path) -> list[FundingObservation]:
        """Load the offline funding parquet into a sorted list of
        ``FundingObservation`` records."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"funding parquet not found: {p}")
        df = pd.read_parquet(p)
        if "fundingRate" not in df.columns:
            raise KeyError(f"funding parquet {p} has no 'fundingRate' column")
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError(f"funding parquet {p} must be DatetimeIndex")
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        out: list[FundingObservation] = []
        for ts, rate in zip(idx, df["fundingRate"].astype(float).to_numpy()):
            if not np.isfinite(rate):
                continue
            out.append(FundingObservation(
                funding_time=pd.Timestamp(ts), funding_rate=float(rate),
            ))
        out.sort(key=lambda o: o.funding_time)
        return out

    # ── Path A REST refresh (lazy, TTL-gated) ───────────────────

    def _maybe_refresh_rest(self) -> None:
        if self._rest_fetcher is None:
            return
        now_monotonic = time.monotonic()
        if (
            self._rest_last_fetch_monotonic is not None
            and (now_monotonic - self._rest_last_fetch_monotonic) < self._rest_ttl
        ):
            return
        observation = self._rest_fetcher()
        self._rest_last_fetch_monotonic = now_monotonic
        if observation is not None:
            self._rest_last_observation = observation

    # ── Per-bar public API ──────────────────────────────────────

    def advance(self, bar_ts: pd.Timestamp) -> float:
        """Sample the ffilled funding rate at ``bar_ts`` and append to
        the bar-grid history.

        Called once per 15-min bar by the engine BEFORE the assembler's
        handler loop so the ``funding_basis_*`` handlers see the freshest
        value. Returns the value for callers that want to log it.
        """
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")
        with self._lock:
            self._maybe_refresh_rest()
            value = self._ffill_value_at(bar_ts)
            self._history.append((bar_ts, value))
        return value

    def latest(self) -> float | None:
        with self._lock:
            if not self._history:
                return None
            return float(self._history[-1][1])

    def latest_lag(self, n: int) -> float | None:
        if n < 0:
            raise ValueError(f"lag n must be >= 0, got {n}")
        with self._lock:
            if len(self._history) <= n:
                return None
            return float(self._history[-1 - n][1])

    def latest_ema(self, span: int) -> float | None:
        if span <= 0:
            raise ValueError(f"span must be positive, got {span}")
        with self._lock:
            if not self._history:
                return None
            ser = pd.Series([v for _, v in self._history], dtype=np.float64)
        ema_val = ser.ewm(span=span, adjust=True, min_periods=1).mean().iloc[-1]
        return None if not np.isfinite(ema_val) else float(ema_val)

    # ── Internals ───────────────────────────────────────────────

    def _ffill_value_at(self, ts: pd.Timestamp) -> float:
        """ffill semantic over Path A (latest REST obs) ∪ Path B (parquet).

        Both sources are merged on the fly: take the most recent
        observation across the two whose ``funding_time <= ts``. If
        neither has anything ≤ ts, return ``default_value`` (0.0).
        """
        candidates: list[FundingObservation] = []
        if self._rest_last_observation is not None and self._rest_last_observation.funding_time <= ts:
            candidates.append(self._rest_last_observation)
        # Bisect into the parquet list (sorted ascending).
        if self._parquet_observations:
            from bisect import bisect_right

            keys = [o.funding_time for o in self._parquet_observations]
            i = bisect_right(keys, ts)
            if i > 0:
                candidates.append(self._parquet_observations[i - 1])
        if not candidates:
            return self._default_value
        # Pick whichever observation is most recent.
        latest = max(candidates, key=lambda o: o.funding_time)
        return float(latest.funding_rate)

    # ── Test/debug helpers ──────────────────────────────────────

    @property
    def history_size(self) -> int:
        with self._lock:
            return len(self._history)

    def history_snapshot(self) -> list[tuple[pd.Timestamp, float]]:
        with self._lock:
            return list(self._history)

    def _force_rest_observation(
        self, observation: FundingObservation | None,
    ) -> None:
        """Test hook: bypass TTL gate to inject a REST observation."""
        with self._lock:
            self._rest_last_fetch_monotonic = time.monotonic()
            self._rest_last_observation = observation


__all__ = ["FundingRateProvider", "FundingObservation"]
