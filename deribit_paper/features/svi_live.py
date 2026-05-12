"""Live SVI surface calibration with hourly caching.

Wraps :func:`vol_arb.svi.calibrate_surface` for use in the live
trading loop.  The batch calibrator takes 5-15 seconds per bucket so
we can't afford to run it at every 15-min bar close — instead we:

1. Cache the most recent params_df in memory.
2. Persist it to ``deribit_paper/state/svi_cache.parquet`` so a
   restart doesn't lose calibration state.
3. Expose :meth:`needs_refresh` so the main engine can decide to kick
   off a background recalibration at the top of each hour.
4. Provide :meth:`evaluate` that runs
   :func:`vol_arb.svi.evaluate_surface_batch` against the cached
   params for a query DataFrame — this is cheap (vectorised
   ``merge_asof``) and can run on every bar close without issue.

The recalibration is designed to run in a background ``asyncio.Task``
so the main bar loop never blocks on SVI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vol_arb.svi import (
    SVIConfig,
    calibrate_surface,
    evaluate_surface_batch,
    surface_from_params_group,
)

logger = logging.getLogger(__name__)


@dataclass
class SVICacheMeta:
    """Bookkeeping for the latest cached calibration.

    Persisted to disk alongside the params parquet.
    """
    last_recalibrated_at: pd.Timestamp
    n_buckets: int
    n_params_rows: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp


class LiveSVICalibrator:
    """Rolling SVI calibration with in-memory + disk cache.

    Usage::

        calibrator = LiveSVICalibrator(
            cache_path="deribit_paper/state/svi_cache.parquet",
            recalibrate_hours=1,
        )

        # At each bar close:
        if calibrator.needs_refresh(bar_close_time):
            # Kick off in background, do NOT block the bar loop
            asyncio.create_task(calibrator.recalibrate_async(option_trades_df))

        # Use the cached params for fast lookups:
        iv_series = calibrator.evaluate(query_df)

    Thread-safety: asyncio-single-threaded; no locks needed.  The
    background recalibrate task runs in a worker thread via
    ``asyncio.to_thread`` (calibrate_surface is CPU-bound, not async).
    """

    def __init__(
        self,
        *,
        cache_path: str | Path = "deribit_paper/state/svi_cache.parquet",
        recalibrate_hours: float = 1.0,
        svi_config: SVIConfig | None = None,
        started_at: pd.Timestamp | None = None,
    ):
        self.cache_path = Path(cache_path)
        self.recalibrate_hours = recalibrate_hours
        self.svi_config = svi_config or SVIConfig(
            bucket_freq="1h",
            min_strikes=5,
            fallback_min_strikes=3,
        )
        self._params_df: pd.DataFrame | None = None
        self._features_df: pd.DataFrame | None = None
        self._meta: SVICacheMeta | None = None
        self._is_recalibrating = False
        # Bug fix (2026-05-07): track construction time so ``is_stale``
        # can transition from "warmup" to "stale" after a reasonable
        # grace period when ``_meta`` stays None forever (e.g.,
        # option_buffer never fills, all recalibrate attempts raise).
        # Without this, the calibrator reports "fresh" indefinitely and
        # the engine's stale-IV check (engine.py around line 1549)
        # never fires — same family as the just-fixed VPIN bug.
        self._started_at: pd.Timestamp = (
            started_at if started_at is not None
            else pd.Timestamp.now(tz="UTC")
        )

    @property
    def has_params(self) -> bool:
        return self._params_df is not None and not self._params_df.empty

    @property
    def is_recalibrating(self) -> bool:
        return self._is_recalibrating

    @property
    def last_recalibrated_at(self) -> pd.Timestamp | None:
        return self._meta.last_recalibrated_at if self._meta else None

    def needs_refresh(self, now: pd.Timestamp) -> bool:
        """Return True if a fresh calibration is due.

        Cycle-27 (2026-05-13): trigger on bucket-boundary crossing
        rather than elapsed-time. Pre-cycle-27 the predicate was
        ``now - last_recalibrated_at >= recalibrate_hours``; with
        ``recalibrate_hours=1h``, a recal that landed at HH:45 anchored
        the NEXT recal at HH+1:45 → every recal saw the most recent
        bucket truncated to ~45 min of trades. Offline calibrates each
        ``[t0, t0+bucket_freq)`` bucket from trades fully within its
        window — the 11:00 bucket at decision time 12:00 was fitted
        from 83 trades offline vs ~75 live (cycle-25 audit). The
        resulting ~1.4vp IV drift propagates into the LGBM signal.

        Post-cycle-27: fire on the first bar where
        ``now.floor(bucket_freq) > anchor.floor(bucket_freq)``. The
        just-closed bucket is then fitted from a full
        ``[t0, t0+bucket_freq)`` window, byte-identical to offline.

        Bootstrap (``_meta is None``): the engine may restart
        mid-bucket. The first FULLY-OBSERVED bucket is the one that
        STARTS after ``_started_at``. We wait for that bucket to close
        before firing the first recal — otherwise we'd fit a bucket
        whose prefix trades are missing from ``option_buffer`` (we
        only started capturing them at ``_started_at``). Until then,
        the SVI cache prewarm (``svi_cache.parquet``) carries the
        engine through. ``is_stale`` (max_stale_hours=4h, see L126)
        is the safety belt if prewarm is too old.
        """
        bucket_freq_td = pd.Timedelta(self.svi_config.bucket_freq)
        cur_bucket_start = now.floor(self.svi_config.bucket_freq)
        if self._meta is None:
            first_clean_bucket_start = self._started_at.ceil(
                self.svi_config.bucket_freq,
            )
            return cur_bucket_start > first_clean_bucket_start
        last_bucket_start = self._meta.last_recalibrated_at.floor(
            self.svi_config.bucket_freq,
        )
        return cur_bucket_start > last_bucket_start

    def is_stale(self, now: pd.Timestamp, max_stale_hours: float = 4.0) -> bool:
        """Return True if cached SVI params are too old to trust.

        Two failure modes produce stale params:

        1. The calibrator HAS calibrated at least once but the most
           recent calibration is older than ``max_stale_hours``.
        2. The calibrator has NEVER calibrated AND has been alive
           longer than ``max_stale_hours``. Pre-fix this case
           returned False forever (treating "never calibrated" as
           "warmup"). Same silent-failure family as the VPIN bug
           — a recalibrate task that always fails (option buffer
           empty due to broken WS, SVI fitter raising on every
           bucket) leaves ``_meta`` as None indefinitely while
           ``is_stale`` reports "fresh".

        Used by the engine to refuse to trade on stale IV.  The threshold
        is separate from ``recalibrate_hours`` (which is 1h): a failed
        recalibration is tolerable for a few hours, but after
        ``max_stale_hours`` the IV surface is unreliable.
        """
        if self._meta is None:
            elapsed = now - self._started_at
            return elapsed >= pd.Timedelta(hours=max_stale_hours)
        elapsed = now - self._meta.last_recalibrated_at
        return elapsed >= pd.Timedelta(hours=max_stale_hours)

    def stale_age_hours(self, now: pd.Timestamp) -> float:
        """Hours since last successful recalibration.

        When the calibrator has never produced ``_meta`` (no successful
        recalibration yet), returns hours since construction so the
        operator-facing heartbeat reports an honest "we've been waiting
        N hours" rather than the misleading 0.0 the pre-fix code emitted.
        """
        if self._meta is None:
            elapsed = now - self._started_at
            return elapsed.total_seconds() / 3600.0
        elapsed = now - self._meta.last_recalibrated_at
        return elapsed.total_seconds() / 3600.0

    def slice_params_for_expiry(
        self,
        expiry_dt: pd.Timestamp,
    ) -> dict[str, float] | None:
        """Return the SVI slice params for a specific expiry, or ``None``.

        Looks up the most-recent-bucket slice for the given expiry and
        returns a dict ``{a, b, rho, m, sigma, T, F}`` matching the
        signature consumed by
        :class:`deribit_paper.signal.drl_hedger_live._TradeSliceSVI`. The
        DRL hedger uses these to evaluate IV at the trade's strike on
        every intra-bar tick (see ``LiveDRLHedger.step`` lines 299-303).

        Pre-D1 (parity-cycle round 2 — 2026-05-07): the live engine
        constructed ``LiveDRLHedger`` with the slice-evaluation
        machinery wired up but never passed the slice in at
        ``begin_trade``, so IV stayed constant for all 10 hedge ticks.
        V2 ``DRLHedger.simulate`` updates IV per-tick from the trade's
        SVI slice; this method closes that asymmetry.

        Returns ``None`` when:
        - the calibrator is uncalibrated (warmup), OR
        - the latest bucket has no slice for the requested expiry.

        Expiry matching is done on ``YYYY-MM-DD`` (Deribit's expiry
        granularity) — same convention as ``evaluate_surface_batch``.
        """
        if self._params_df is None or self._params_df.empty:
            return None
        if "expiry_dt" not in self._params_df.columns:
            # Older cached parquets pre-date the per-expiry split; can't match.
            return None
        latest_ts = self._params_df["timestamp"].max()
        bucket = self._params_df[self._params_df["timestamp"] == latest_ts]
        if bucket.empty:
            return None
        target_key = pd.to_datetime(expiry_dt, utc=True).strftime("%Y-%m-%d")
        bucket_keys = pd.to_datetime(bucket["expiry_dt"], utc=True).dt.strftime(
            "%Y-%m-%d",
        )
        matches = bucket[bucket_keys.values == target_key]
        if matches.empty:
            return None
        row = matches.iloc[0]
        try:
            return {
                "a": float(row["a"]),
                "b": float(row["b"]),
                "rho": float(row["rho"]),
                "m": float(row["m"]),
                "sigma": float(row["sigma"]),
                "T": float(row["T"]),
                "F": float(row["F"]),
            }
        except (KeyError, TypeError, ValueError):
            return None

    def atm_vol_at_dte(
        self,
        dte_days: float,
        *,
        now: pd.Timestamp | None = None,
    ) -> float:
        """Return the constant-maturity ATM implied vol at ``dte_days`` DTE.

        Matches the offline ``_constant_maturity_features`` output in
        ``vol_arb/svi.py`` — returns IV in PERCENT (e.g., 55.0 for 55%)
        so the scale matches the ``svi_atm_{1d,1w,2w,1m}`` columns in
        ``vol_arb/output/svi_features_15m.parquet`` that the Stage 18
        models were trained on.

        The calibrator reconstructs the full ``ArbitrageFreeSVI``
        surface from its most recent bucket of per-expiry SVI slices
        and interpolates in total variance between the bracketing
        slices (see ``ArbitrageFreeSVI.total_variance`` for the math).
        If the calibrator has no cached params yet (warmup), returns
        ``float('nan')``.

        F7 (parity-cycle round 2 — 2026-05-07): when the latest bucket
        has fewer than 2 expiries, the ``ArbitrageFreeSVI`` surface
        degenerates — ``atm_vol_pct(T)`` returns the single slice's
        ATM regardless of the requested ``T``. That's a silent
        approximation error on the warmup path (and on intra-day
        windows when only 1 expiry has enough strikes to calibrate).
        Return ``NaN`` instead so the F9 baseline-fallback path takes
        over rather than emitting a misleading value.

        F-4 (parity-cycle round 3 — 2026-05-08): ``params_df["timestamp"]``
        is the bucket START. The offline pipeline shifts each
        ``bucket_ts`` forward by ``svi_config.bucket_freq`` (default 1h)
        before reindex onto the 15-min decision grid (see
        ``vol_arb/rebuild_svi_features_with_spot.py:117``) so a bar at
        wall-clock T uses the most recent bucket whose
        ``bucket_ts + bucket_freq <= T``. Pre-fix, live always picked
        ``bucket_ts.max()`` regardless of decision-time T — making
        live IV up to ~1 hour fresher than offline. When the caller
        supplies ``now`` (typically ``ctx.bar.timestamp``), filter
        ``params_df`` to buckets that respect the same shift contract.
        ``now=None`` preserves the legacy behaviour for back-compat
        with paths that don't have a decision-time anchor (e.g.,
        tests, telemetry).
        """
        if self._params_df is None or self._params_df.empty:
            return float("nan")
        # F-4: enforce offline's bucket_ts + bucket_freq <= now contract.
        # ``params_df["timestamp"]`` is the bucket START; offline reindex
        # treats a bucket as observable only after its FULL window has
        # elapsed (bucket_freq later). Live must do the same.
        params = self._params_df
        if now is not None:
            try:
                bucket_freq_td = pd.Timedelta(self.svi_config.bucket_freq)
            except (TypeError, ValueError) as exc:
                # Unparseable bucket_freq is a config error, not a
                # silent-bias path. Fail loudly so the operator catches
                # it instead of degrading to the pre-fix behaviour.
                raise ValueError(
                    f"svi_config.bucket_freq={self.svi_config.bucket_freq!r} "
                    f"could not be parsed as a Timedelta: {exc}"
                ) from exc
            ts_observable_at = params["timestamp"] + bucket_freq_td
            params = params[ts_observable_at <= pd.Timestamp(now)]
            if params.empty:
                return float("nan")
        # Pick the most recent bucket timestamp — surface_from_params_group
        # expects rows for exactly one bucket.
        latest_ts = params["timestamp"].max()
        bucket = params[params["timestamp"] == latest_ts]
        if bucket.empty:
            return float("nan")
        # F7: refuse to interpolate when only one expiry is present.
        # A single slice cannot bracket the requested DTE, so the
        # surface evaluation collapses to that one expiry's ATM.
        if "expiry_dt" in bucket.columns:
            n_expiries = bucket["expiry_dt"].nunique()
            if n_expiries < 2:
                return float("nan")
        try:
            surface = surface_from_params_group(bucket)
            T = float(dte_days) / 365.25
            return float(surface.atm_vol_pct(T=T))
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.warning("atm_vol_at_dte(%s) failed: %s", dte_days, exc)
            return float("nan")

    def skew_25d_at_dte(
        self,
        dte_days: float,
        *,
        now: pd.Timestamp | None = None,
    ) -> float:
        """Return the constant-maturity 25-delta skew at ``dte_days`` DTE.

        Matches the offline ``svi_25d_skew_1w`` computation in
        ``vol_arb/svi.py::_constant_maturity_features`` line 1112-1123 byte-equally:

            k_25d  = atm_iv_dec * sqrt(T) * 0.6745
            iv_put = sqrt(total_variance(-k_25d, T) / T) * 100
            iv_call = sqrt(total_variance(+k_25d, T) / T) * 100
            skew_25d = iv_put - iv_call

        Positive output = put-bid relative to call-bid (downside-priced regime).
        Cycle-10 (2026-05-10): used as a short-side filter — drop SHORT
        signals when this exceeds the V2 preset's
        ``skip_short_if_skew_1w_gt`` threshold (default 25.0).

        Returns ``NaN`` during warmup / single-expiry buckets, mirroring
        ``atm_vol_at_dte`` so the live predictor can pass through (don't
        suppress signals during warmup).
        """
        import numpy as np
        if self._params_df is None or self._params_df.empty:
            return float("nan")
        params = self._params_df
        if now is not None:
            try:
                bucket_freq_td = pd.Timedelta(self.svi_config.bucket_freq)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"svi_config.bucket_freq={self.svi_config.bucket_freq!r} "
                    f"could not be parsed as a Timedelta: {exc}"
                ) from exc
            ts_observable_at = params["timestamp"] + bucket_freq_td
            params = params[ts_observable_at <= pd.Timestamp(now)]
            if params.empty:
                return float("nan")
        latest_ts = params["timestamp"].max()
        bucket = params[params["timestamp"] == latest_ts]
        if bucket.empty:
            return float("nan")
        if "expiry_dt" in bucket.columns:
            n_expiries = bucket["expiry_dt"].nunique()
            if n_expiries < 2:
                return float("nan")
        try:
            surface = surface_from_params_group(bucket)
            T = float(dte_days) / 365.25
            atm_pct = float(surface.atm_vol_pct(T=T))
            atm_dec = atm_pct / 100.0
            if atm_dec <= 0 or T <= 0:
                return float("nan")
            k_25d = atm_dec * np.sqrt(T) * 0.6745
            tv_put = float(surface.total_variance(np.array([-k_25d]), T)[0])
            tv_call = float(surface.total_variance(np.array([k_25d]), T)[0])
            iv_put = float(np.sqrt(max(tv_put, 0.0) / T) * 100.0)
            iv_call = float(np.sqrt(max(tv_call, 0.0) / T) * 100.0)
            return iv_put - iv_call
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.warning("skew_25d_at_dte(%s) failed: %s", dte_days, exc)
            return float("nan")

    def load_cache(self) -> bool:
        """Attempt to load params from disk.  Returns True on success."""
        if not self.cache_path.exists():
            return False
        try:
            self._params_df = pd.read_parquet(self.cache_path)
        except Exception as exc:  # noqa: BLE001 - log + move on
            logger.warning("Failed to load SVI cache from %s: %s", self.cache_path, exc)
            return False
        meta_path = self.cache_path.with_suffix(".meta.parquet")
        if meta_path.exists():
            try:
                meta_row = pd.read_parquet(meta_path).iloc[0].to_dict()
                self._meta = SVICacheMeta(
                    last_recalibrated_at=pd.Timestamp(meta_row["last_recalibrated_at"]),
                    n_buckets=int(meta_row["n_buckets"]),
                    n_params_rows=int(meta_row["n_params_rows"]),
                    window_start=pd.Timestamp(meta_row["window_start"]),
                    window_end=pd.Timestamp(meta_row["window_end"]),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load SVI cache meta: %s", exc)
        logger.info(
            "Loaded SVI cache: %d rows from %s",
            len(self._params_df), self.cache_path,
        )
        return True

    def recalibrate(
        self,
        option_trades: pd.DataFrame,
        *,
        now: pd.Timestamp | None = None,
    ) -> SVICacheMeta:
        """Run a fresh calibration on ``option_trades`` and update the cache.

        Synchronous / blocking.  For the async wrapper that doesn't
        block the bar loop, use :meth:`recalibrate_async`.
        """
        if self._is_recalibrating:
            raise RuntimeError("LiveSVICalibrator.recalibrate is already running")
        self._is_recalibrating = True
        try:
            if option_trades is None or option_trades.empty:
                raise ValueError("recalibrate called with empty option_trades")

            # The batch calibrator needs T_years and dte_days — derive them
            # if the caller hasn't provided them already.
            df = option_trades.copy()
            if "T_years" not in df.columns or df["T_years"].isna().all():
                df["T_years"] = (
                    (df["expiry_dt"] - df["timestamp"]).dt.total_seconds()
                    / (365.25 * 24 * 3600)
                )
            if "dte_days" not in df.columns or df["dte_days"].isna().all():
                df["dte_days"] = (
                    (df["expiry_dt"] - df["timestamp"]).dt.total_seconds() / 86_400.0
                )

            features_df, params_df = calibrate_surface(df, self.svi_config)

            if params_df.empty:
                logger.warning(
                    "SVI recalibrate produced empty params "
                    "(%d input trades, %d expiries)",
                    len(df), df["expiry_dt"].nunique(),
                )
                return self._meta or _empty_meta(now or pd.Timestamp.utcnow())

            self._features_df = features_df
            self._params_df = params_df
            window_start = pd.Timestamp(df["timestamp"].min())
            window_end = pd.Timestamp(df["timestamp"].max())
            recalibrated_at = now or pd.Timestamp.now(tz="UTC")
            self._meta = SVICacheMeta(
                last_recalibrated_at=recalibrated_at,
                n_buckets=int(features_df.shape[0]),
                n_params_rows=int(params_df.shape[0]),
                window_start=window_start,
                window_end=window_end,
            )
            self._persist_cache()
            logger.info(
                "SVI recalibrated: %d buckets, %d param rows, window %s → %s",
                self._meta.n_buckets, self._meta.n_params_rows,
                window_start, window_end,
            )
            return self._meta
        finally:
            self._is_recalibrating = False

    async def recalibrate_async(
        self,
        option_trades: pd.DataFrame,
        *,
        now: pd.Timestamp | None = None,
    ) -> SVICacheMeta:
        """Run :meth:`recalibrate` in a worker thread via ``asyncio.to_thread``.

        The batch calibrator is CPU-bound, so running it in a thread
        keeps the main event loop unblocked.  The worker thread shares
        memory with the event loop, so the cache update is visible
        immediately after the call returns.
        """
        import asyncio
        return await asyncio.to_thread(
            self.recalibrate, option_trades, now=now,
        )

    def evaluate(self, query_df: pd.DataFrame) -> pd.Series:
        """Evaluate the cached SVI surface at arbitrary (timestamp, strike, expiry) points.

        Thin wrapper around ``vol_arb.svi.evaluate_surface_batch``.
        Returns IV in percent (Deribit convention) or NaN for rows
        with no matching slice.  Raises if no cache is present yet.
        """
        if not self.has_params:
            raise RuntimeError(
                "LiveSVICalibrator.evaluate called before any successful recalibration",
            )
        return evaluate_surface_batch(self._params_df, query_df)

    def _persist_cache(self) -> None:
        """Write the current params_df + meta to disk atomically.

        S7 (parity-cycle round 2 — 2026-05-07): pre-S7 the params and
        meta parquets were written sequentially. A crash between the
        two writes left stale-meta-with-fresh-params (or vice versa),
        which the freshness guards then mis-diagnosed because the
        timestamps in meta no longer matched the params snapshot.

        Atomic write pattern: write to ``*.tmp`` files, then
        ``os.replace`` for both. ``os.replace`` is atomic on POSIX
        and Windows.
        """
        if self._params_df is None or self._meta is None:
            return
        import os as _os
        params_tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        meta_path = self.cache_path.with_suffix(".meta.parquet")
        meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._params_df.to_parquet(params_tmp)
            pd.DataFrame([{
                "last_recalibrated_at": self._meta.last_recalibrated_at,
                "n_buckets": self._meta.n_buckets,
                "n_params_rows": self._meta.n_params_rows,
                "window_start": self._meta.window_start,
                "window_end": self._meta.window_end,
            }]).to_parquet(meta_tmp)
            # Both temp files exist now → replace atomically.
            _os.replace(params_tmp, self.cache_path)
            _os.replace(meta_tmp, meta_path)
        except Exception as exc:  # noqa: BLE001 - log + continue, cache is best-effort
            logger.warning("Failed to persist SVI cache: %s", exc)
            # Clean up any leftover temp files so we don't accumulate cruft.
            for _tmp in (params_tmp, meta_tmp):
                try:
                    if _tmp.exists():
                        _tmp.unlink()
                except OSError:
                    pass


def _empty_meta(at: pd.Timestamp) -> SVICacheMeta:
    return SVICacheMeta(
        last_recalibrated_at=at,
        n_buckets=0,
        n_params_rows=0,
        window_start=at,
        window_end=at,
    )
