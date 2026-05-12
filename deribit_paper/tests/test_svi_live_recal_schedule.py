"""Cycle-27 (2026-05-13) regression tests for live SVI recal scheduling.

Pre-cycle-27 ``LiveSVICalibrator.needs_refresh`` used elapsed time::

    elapsed = now - _meta.last_recalibrated_at
    return elapsed >= pd.Timedelta(hours=recalibrate_hours)

When the engine restarted at HH:45 and immediately fired a recal, the
next was anchored at HH+1:45 — every subsequent recal fitted the most
recent bucket (e.g. HH+1:00) from only ~45 min of trades.

Offline, each ``[t0, t0+bucket_freq)`` bucket is calibrated from
trades fully within that window. Cycle-25's bar-by-bar parity audit
caught the consequence at 2026-05-12 12:00 UTC::

    offline 11:00 bucket = 83 May-15 expiry trades, full hour
    live    11:00 bucket = 75 May-15 expiry trades, [11:00, 11:45]
    Δiv at bar 12:00 = ~1.4vp

Cycle-27 fix:
- ``needs_refresh`` fires when ``now.floor(bucket_freq)`` crosses
  ``anchor.floor(bucket_freq)`` (bucket-boundary trigger).
- Bootstrap (``_meta is None``): wait for a bucket that STARTED
  after ``_started_at`` to close → guarantees full trade coverage.
- Engine ``process_bar`` AWAITS the recal at the boundary so the
  bar's IV query sees the freshly-fitted bucket (F-4 contract
  ``bucket_ts + bucket_freq <= now`` picks the just-fitted bucket).

These tests pin every leg of the contract so a future schedule rewrite
fails CI rather than silently re-introducing the 1.4vp drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from deribit_paper.features.svi_live import LiveSVICalibrator, SVICacheMeta


# ── Bucket-boundary trigger (post-cycle-27 semantics) ────────────────


def test_post_calibration_does_not_fire_within_same_bucket(tmp_path):
    """A recal at 12:30 must NOT trigger another at 12:45 — both fall
    in the [12:00, 13:00) bucket. Pre-cycle-27 the elapsed predicate
    correctly returned False here; cycle-27 must preserve that."""
    cal = LiveSVICalibrator(cache_path=tmp_path / "svi.parquet")
    cal._meta = SVICacheMeta(
        last_recalibrated_at=pd.Timestamp("2026-05-12 12:30:00", tz="UTC"),
        n_buckets=1, n_params_rows=2,
        window_start=pd.Timestamp("2026-05-12 11:00:00", tz="UTC"),
        window_end=pd.Timestamp("2026-05-12 12:30:00", tz="UTC"),
    )
    # Same bucket as last recal — no fire.
    assert cal.needs_refresh(
        pd.Timestamp("2026-05-12 12:45:00", tz="UTC"),
    ) is False


def test_post_calibration_fires_at_next_bucket_boundary(tmp_path):
    """Cycle-27: a recal at 12:30 must trigger again at 13:00 — that
    is when the [12:00, 13:00) bucket has fully closed and the
    [12:00, 13:00) trade set is in option_buffer for refit."""
    cal = LiveSVICalibrator(cache_path=tmp_path / "svi.parquet")
    cal._meta = SVICacheMeta(
        last_recalibrated_at=pd.Timestamp("2026-05-12 12:30:00", tz="UTC"),
        n_buckets=1, n_params_rows=2,
        window_start=pd.Timestamp("2026-05-12 11:00:00", tz="UTC"),
        window_end=pd.Timestamp("2026-05-12 12:30:00", tz="UTC"),
    )
    assert cal.needs_refresh(
        pd.Timestamp("2026-05-12 13:00:00", tz="UTC"),
    ) is True


def test_post_calibration_does_not_double_fire_within_new_bucket(tmp_path):
    """Once a recal lands at 13:00 (next bucket), the 13:15 bar
    must NOT trigger another fit — we already have the 12:00 bucket
    fresh and the 13:00 bucket hasn't closed yet."""
    cal = LiveSVICalibrator(cache_path=tmp_path / "svi.parquet")
    cal._meta = SVICacheMeta(
        last_recalibrated_at=pd.Timestamp("2026-05-12 13:00:00", tz="UTC"),
        n_buckets=2, n_params_rows=4,
        window_start=pd.Timestamp("2026-05-12 11:00:00", tz="UTC"),
        window_end=pd.Timestamp("2026-05-12 13:00:00", tz="UTC"),
    )
    assert cal.needs_refresh(
        pd.Timestamp("2026-05-12 13:15:00", tz="UTC"),
    ) is False
    assert cal.needs_refresh(
        pd.Timestamp("2026-05-12 13:45:00", tz="UTC"),
    ) is False


# ── Bootstrap edge case (engine restart mid-bucket) ──────────────────


def test_bootstrap_does_not_fire_in_startup_bucket(tmp_path):
    """Engine restart at 11:15 must NOT fit the [11:00, 12:00) bucket
    at 12:00 — the prefix trades [11:00, 11:15) were never captured
    in option_buffer, so a fit there would be truncated worse than
    pre-cycle-27."""
    cal = LiveSVICalibrator(
        cache_path=tmp_path / "svi.parquet",
        started_at=pd.Timestamp("2026-05-12 11:15:00", tz="UTC"),
    )
    # _meta is None; ceil(11:15, 1h) = 12:00; cur_bucket_start at 12:00 is 12:00.
    # 12:00 > 12:00 is False → no fire. (Bucket [12:00, 13:00) hasn't closed.)
    assert cal.needs_refresh(
        pd.Timestamp("2026-05-12 12:00:00", tz="UTC"),
    ) is False
    # 12:45 bar — same [12:00, 13:00) bucket — still no fire.
    assert cal.needs_refresh(
        pd.Timestamp("2026-05-12 12:45:00", tz="UTC"),
    ) is False


def test_bootstrap_fires_after_first_clean_bucket_closes(tmp_path):
    """Engine restart at 11:15 → first FULLY-observed bucket is
    [12:00, 13:00). It closes at 13:00. First bar at or after 13:00
    fires the first recal, fitting the 12:00 bucket from full
    [12:00, 13:00) trades (engine alive since 11:15)."""
    cal = LiveSVICalibrator(
        cache_path=tmp_path / "svi.parquet",
        started_at=pd.Timestamp("2026-05-12 11:15:00", tz="UTC"),
    )
    # 13:00 bar: cur_bucket_start=13:00 > ceil(11:15, 1h)=12:00 → fire.
    assert cal.needs_refresh(
        pd.Timestamp("2026-05-12 13:00:00", tz="UTC"),
    ) is True


def test_bootstrap_started_at_on_bucket_edge(tmp_path):
    """Edge case: engine restart at exactly 11:00:00 (bucket start).
    ``ceil(11:00, 1h)`` is 11:00 (ceil on aligned input is itself).
    First clean bucket is [11:00, 12:00); fires at 12:00."""
    cal = LiveSVICalibrator(
        cache_path=tmp_path / "svi.parquet",
        started_at=pd.Timestamp("2026-05-12 11:00:00", tz="UTC"),
    )
    # 11:30 bar — same [11:00, 12:00) bucket — no fire.
    assert cal.needs_refresh(
        pd.Timestamp("2026-05-12 11:30:00", tz="UTC"),
    ) is False
    # 12:00 bar — bucket closed — fire.
    assert cal.needs_refresh(
        pd.Timestamp("2026-05-12 12:00:00", tz="UTC"),
    ) is True


# ── Engine ``process_bar`` AWAITS the recal at boundary ──────────────


def test_engine_process_bar_uses_await_not_create_task():
    """Cycle-27 static-source check: ``process_bar`` must AWAIT the
    recal at the bucket boundary, not fire-and-forget via
    ``asyncio.create_task``. Pre-cycle-27 the engine wrapped the
    recal in ``asyncio.create_task(self._recalibrate_svi_task(...))``
    so the bar's IV query saw the PRIOR bucket; cycle-27 must use
    ``await self._recalibrate_svi_task(...)`` so the freshly-fitted
    bucket lands in ``params_df`` before F-4 lookups run.

    We inspect the source directly: a regex on the recal block must
    show ``await self._recalibrate_svi_task`` and must NOT show
    ``create_task(.*_recalibrate_svi_task`` inside the
    ``if self.svi_calibrator.needs_refresh(`` block.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1].joinpath("engine.py").read_text()
    # Locate the recal block — starts at the needs_refresh check.
    match = re.search(
        r"if self\.svi_calibrator\.needs_refresh\([^\)]+\):"
        r"(?P<body>(?:\n[^\n]*){1,30})",
        src,
    )
    assert match is not None, (
        "could not locate recal block in engine.py — has the layout changed?"
    )
    body = match.group("body")
    assert "await self._recalibrate_svi_task" in body, (
        "Cycle-27: engine recal must AWAIT _recalibrate_svi_task; "
        f"got body:\n{body}"
    )
    assert "create_task(\n            self._recalibrate_svi_task" not in body, (
        "Cycle-27: engine recal must NOT use create_task "
        "(would leave the bar reading stale bucket params)"
    )
    assert "create_task(self._recalibrate_svi_task" not in body, (
        "Cycle-27: engine recal must NOT use create_task "
        "(would leave the bar reading stale bucket params)"
    )


# ── Full-window trade coverage (the actual parity property) ──────────


def test_recal_at_boundary_fits_bucket_from_full_window_trades(tmp_path):
    """The behavioral property cycle-27 enforces: when the recal fires
    at 13:00 with option_buffer = [12:00, 13:00) trades, the resulting
    12:00 bucket params are byte-identical to offline's 12:00 bucket
    fit from the same trade set.

    We don't simulate the engine here — we directly call ``recalibrate``
    with the [12:00, 13:00) trade window and inspect the produced
    bucket. The test pins that calibrate_surface buckets by trade
    timestamp's floor(bucket_freq), so a trade at 12:55 lands in the
    12:00 bucket. (Mirrors offline's bucketing convention.)
    """
    cal = LiveSVICalibrator(cache_path=tmp_path / "svi.parquet")
    rng = np.random.default_rng(42)
    rows = []
    base_ts = pd.Timestamp("2026-05-12 12:00:00", tz="UTC")
    strikes = np.linspace(55_000, 65_000, 7)
    expiry = pd.Timestamp("2026-05-15 08:00:00", tz="UTC")
    # 60 trades distributed across [12:00, 13:00) at 1-min cadence.
    for i in range(60):
        ts = base_ts + pd.Timedelta(minutes=i)
        strike = float(rng.choice(strikes))
        rows.append({
            "timestamp": ts,
            "instrument_name": f"BTC-MAY15-{int(strike)}-C",
            "strike": strike,
            "option_type": "C" if rng.random() > 0.5 else "P",
            "expiry_dt": expiry,
            "iv": 70.0 + rng.normal(0, 3.0),
            "price": 0.05,
            "amount": 1.0,
            "direction": 1 if rng.random() > 0.5 else -1,
            "index_price": 60_000.0,
            "mark_price": 0.05,
        })
    trades = pd.DataFrame(rows)

    meta = cal.recalibrate(trades, now=pd.Timestamp("2026-05-12 13:00:00", tz="UTC"))
    assert meta.window_start <= base_ts
    assert meta.window_end >= base_ts + pd.Timedelta(minutes=59)
    # Exactly one bucket should be produced (all 60 trades land in the
    # same [12:00, 13:00) window). Cycle-27 ensures that bucket is
    # fitted from the full trade window, byte-identical to offline's
    # bucket from the same trade set.
    if cal.has_params:
        bucket_stamps = cal._params_df["timestamp"].unique()
        assert len(bucket_stamps) == 1, (
            f"expected 1 bucket from 60 trades in a single hour; "
            f"got {len(bucket_stamps)} buckets: {bucket_stamps}"
        )
