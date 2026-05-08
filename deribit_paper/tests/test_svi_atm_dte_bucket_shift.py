"""F-4 regression test (round-3 parity audit, fix 2026-05-08).

The offline SVI feature pipeline shifts ``bucket_ts`` forward by
``bucket_freq`` (default 1h) before reindexing onto the 15-min decision
grid (see ``vol_arb/rebuild_svi_features_with_spot.py:117``). A bar at
wall-clock T uses the most recent bucket whose
``bucket_ts + bucket_freq <= T``.

Pre-fix, ``LiveSVICalibrator.atm_vol_at_dte`` always returned the
freshest bucket regardless of decision-time T — so live IV could be
up to ~1 hour fresher than the offline-calibrated model expected.

Post-fix, the caller passes ``now=bar.timestamp`` and
``atm_vol_at_dte`` filters ``params_df`` accordingly. This test
verifies:

1. With ``now=None`` (legacy), behaviour is unchanged (back-compat).
2. With ``now`` set, only buckets that respect ``bucket_ts +
   bucket_freq <= now`` are visible.
3. Edge case: when ``now`` is BEFORE every bucket's observability
   timestamp, the calibrator returns ``NaN`` (forces baseline fallback).
"""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.features.svi_live import LiveSVICalibrator


def _stub_svi_calibrator(
    bucket_freq: str = "1h",
    expiry_dts: list[pd.Timestamp] | None = None,
) -> LiveSVICalibrator:
    """Build a calibrator with two buckets at 1h apart and stable params."""
    cal = LiveSVICalibrator()
    cal.svi_config.bucket_freq = bucket_freq

    if expiry_dts is None:
        expiry_dts = [
            pd.Timestamp("2026-05-15 08:00:00", tz="UTC"),  # ~1w
            pd.Timestamp("2026-05-22 08:00:00", tz="UTC"),  # ~2w
        ]

    # Two buckets: bucket1 at 13:00 UTC, bucket2 at 14:00 UTC.
    # Each has a row per expiry so atm_vol_pct can interpolate.
    rows = []
    bucket_ts1 = pd.Timestamp("2026-05-07 13:00:00", tz="UTC")
    bucket_ts2 = pd.Timestamp("2026-05-07 14:00:00", tz="UTC")
    for bucket_ts, sigma in [(bucket_ts1, 0.55), (bucket_ts2, 0.65)]:
        for exp_dt in expiry_dts:
            T_years = (exp_dt - bucket_ts).total_seconds() / (365.25 * 24 * 3600)
            rows.append({
                "timestamp": bucket_ts,
                "expiry_dt": exp_dt,
                "a": sigma * sigma,        # ATM total var anchor
                "b": 0.0,
                "rho": 0.0,
                "m": 0.0,
                "sigma": 1e-6,
                "T": T_years,
                "F": 80_000.0,
            })
    cal._params_df = pd.DataFrame(rows)
    return cal


# ── Behaviour ──────────────────────────────────────────────────────


def test_now_none_uses_freshest_bucket_unchanged_behaviour():
    """Pre-fix path: ``now=None`` → freshest bucket (14:00) wins.

    We don't care about the exact ATM-vol value (surface stub units are
    arbitrary); the assertion is "the 14:00 bucket was selected, NOT
    the 13:00 bucket" — verified by giving the two buckets DIFFERENT
    ``a`` values and confirming the value matches the 14:00 fingerprint.
    """
    cal = _stub_svi_calibrator()
    out_no_now = cal.atm_vol_at_dte(7.0)
    assert pd.notna(out_no_now)

    # Replace the 14:00 bucket with the 13:00-only set and re-evaluate.
    only_13 = cal._params_df[
        cal._params_df["timestamp"] == pd.Timestamp("2026-05-07 13:00", tz="UTC")
    ].copy()
    cal_13_only = LiveSVICalibrator()
    cal_13_only.svi_config.bucket_freq = "1h"
    cal_13_only._params_df = only_13
    out_13 = cal_13_only.atm_vol_at_dte(7.0)
    # 14:00-fresh and 13:00-only must produce different results
    # (different ``a`` anchors → different ATM vol output).
    assert out_no_now != pytest.approx(out_13, rel=1e-9), (
        f"freshest-bucket selection seems broken: 14:00 path = {out_no_now}, "
        f"13:00-only path = {out_13}, but they're approximately equal"
    )


def test_now_filters_to_observable_buckets():
    """At now=14:30, only the 13:00 bucket is observable (freshest's
    observability window not yet elapsed). Result must match the
    13:00-only baseline. At now=15:30, both observable → freshest wins."""
    cal = _stub_svi_calibrator(bucket_freq="1h")

    # Build a "13:00-only" calibrator as the baseline target.
    only_13 = cal._params_df[
        cal._params_df["timestamp"] == pd.Timestamp("2026-05-07 13:00", tz="UTC")
    ].copy()
    cal_baseline_13 = LiveSVICalibrator()
    cal_baseline_13.svi_config.bucket_freq = "1h"
    cal_baseline_13._params_df = only_13
    expected_13_value = cal_baseline_13.atm_vol_at_dte(7.0)

    now_1430 = pd.Timestamp("2026-05-07 14:30:00", tz="UTC")
    out_filtered = cal.atm_vol_at_dte(7.0, now=now_1430)
    assert pd.notna(out_filtered)
    # At 14:30, the 14:00 bucket isn't observable yet (14+1=15 > 14:30),
    # so the calibrator must fall back to the 13:00 bucket.
    assert out_filtered == pytest.approx(expected_13_value, rel=1e-12), (
        f"now=14:30 should filter to 13:00 bucket "
        f"(value={expected_13_value}), got {out_filtered}"
    )

    # And at now=15:30, both buckets are observable; the freshest (14:00)
    # is selected. Result should match the now=None case.
    now_1530 = pd.Timestamp("2026-05-07 15:30:00", tz="UTC")
    out_after = cal.atm_vol_at_dte(7.0, now=now_1530)
    out_no_now = cal.atm_vol_at_dte(7.0)
    assert pd.notna(out_after)
    assert out_after == pytest.approx(out_no_now, rel=1e-12), (
        "now after both buckets observable should match now=None"
    )


def test_now_before_all_buckets_returns_nan():
    """When the bar timestamp is before every bucket's observability
    timestamp, the calibrator must return NaN (baseline fallback)."""
    cal = _stub_svi_calibrator(bucket_freq="1h")
    # 13:00 + 1h = 14:00 — before that, no bucket is observable.
    now_too_early = pd.Timestamp("2026-05-07 13:30:00", tz="UTC")
    out = cal.atm_vol_at_dte(7.0, now=now_too_early)
    assert pd.isna(out), (
        f"now before any bucket's observability should return NaN, got {out}"
    )


def test_now_at_observability_boundary_includes_bucket():
    """The shift contract is ``bucket_ts + bucket_freq <= now``
    (inclusive on the boundary)."""
    cal = _stub_svi_calibrator(bucket_freq="1h")
    # 13:00 + 1h = 14:00 — at exactly 14:00 the 13:00 bucket is observable.
    now_boundary = pd.Timestamp("2026-05-07 14:00:00", tz="UTC")
    out = cal.atm_vol_at_dte(7.0, now=now_boundary)
    assert pd.notna(out), (
        "at boundary now == bucket_ts + bucket_freq, bucket must be visible"
    )


def test_bucket_freq_2h_yields_different_observability():
    """Changing bucket_freq must change which buckets are observable."""
    cal_1h = _stub_svi_calibrator(bucket_freq="1h")
    cal_2h = _stub_svi_calibrator(bucket_freq="2h")
    # at 14:30:
    #   bucket_freq=1h → 13:00 bucket observable (13+1=14 <= 14:30)
    #   bucket_freq=2h → no bucket observable (13+2=15 > 14:30)
    now = pd.Timestamp("2026-05-07 14:30:00", tz="UTC")
    out_1h = cal_1h.atm_vol_at_dte(7.0, now=now)
    out_2h = cal_2h.atm_vol_at_dte(7.0, now=now)
    assert pd.notna(out_1h)
    assert pd.isna(out_2h)


def test_unparseable_bucket_freq_raises():
    """Operator-detection: a typo in bucket_freq should fail loud,
    not silently degrade to pre-fix behaviour."""
    cal = _stub_svi_calibrator()
    cal.svi_config.bucket_freq = "this is not a timedelta"
    with pytest.raises(ValueError, match="could not be parsed as a Timedelta"):
        cal.atm_vol_at_dte(7.0, now=pd.Timestamp("2026-05-07 15:00:00", tz="UTC"))


def test_empty_params_df_returns_nan_with_now():
    """No cached params → NaN regardless of ``now``."""
    cal = LiveSVICalibrator()
    out = cal.atm_vol_at_dte(7.0, now=pd.Timestamp("2026-05-07 14:00:00", tz="UTC"))
    assert pd.isna(out)
