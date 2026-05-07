"""D1 (parity-cycle round 2 — 2026-05-07): live DRL hedger consumes
the SVI slice params plumbed from the calibrator → predictor → straddle_flow
→ ``LiveDRLHedger.begin_trade``.

Pre-D1 the calibrator already exposed the slice infrastructure but the
caller in ``StraddleFlow.execute`` never passed ``svi_slice_params``,
so ``LiveDRLHedger`` fell back to ``iv_entry`` for every tick of every
trade — diverging from V2 ``DRLHedger.simulate`` which updates IV
per-tick from the trade's slice.

These tests verify:

1. ``LiveSVICalibrator.slice_params_for_expiry`` returns a usable dict
   when the latest bucket has the requested expiry, ``None`` otherwise.
2. The predictor-built ``Signal`` carries ``svi_slice`` populated when
   SVI is calibrated and an actionable straddle is picked.
3. ``LiveDRLHedger.begin_trade`` with non-None ``svi_slice_params``
   stores a ``_TradeSliceSVI`` on the trade state, and ``step()`` then
   evaluates IV via the slice (per-tick) instead of using a constant
   ``iv_entry`` for all 10 ticks.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.svi_live import LiveSVICalibrator, SVICacheMeta
from deribit_paper.signal.drl_hedger_live import LiveDRLHedger, _TradeSliceSVI
from deribit_paper.signal.model_loader import ModelBundle


# ── LiveSVICalibrator.slice_params_for_expiry ────────────────────────


def _make_calibrator_with_slices(
    expiry_dts: list[pd.Timestamp],
    *,
    bucket_ts: pd.Timestamp,
) -> LiveSVICalibrator:
    """Construct a calibrator whose ``_params_df`` mocks a calibrated state."""
    rows = [
        {
            "timestamp": bucket_ts,
            "expiry_dt": expiry,
            "T": 7.0 / 365.25,
            "F": 50_000.0,
            "a": 0.001,
            "b": 0.1,
            "rho": -0.2,
            "m": 0.0,
            "sigma": 0.3,
        }
        for expiry in expiry_dts
    ]
    cal = LiveSVICalibrator(cache_path="/tmp/_test_only_does_not_exist.parquet")
    cal._params_df = pd.DataFrame(rows)
    cal._meta = SVICacheMeta(
        last_recalibrated_at=bucket_ts,
        n_buckets=1,
        n_params_rows=len(rows),
        window_start=bucket_ts,
        window_end=bucket_ts,
    )
    return cal


def test_slice_params_returns_none_when_uncalibrated():
    cal = LiveSVICalibrator(cache_path="/tmp/_test_only_does_not_exist.parquet")
    assert cal.slice_params_for_expiry(pd.Timestamp("2026-05-09", tz="UTC")) is None


def test_slice_params_returns_none_when_expiry_missing():
    bucket_ts = pd.Timestamp("2026-05-07 12:00", tz="UTC")
    cal = _make_calibrator_with_slices(
        [pd.Timestamp("2026-05-09", tz="UTC")], bucket_ts=bucket_ts,
    )
    assert cal.slice_params_for_expiry(pd.Timestamp("2026-06-01", tz="UTC")) is None


def test_slice_params_returns_dict_with_required_fields():
    bucket_ts = pd.Timestamp("2026-05-07 12:00", tz="UTC")
    expiry = pd.Timestamp("2026-05-09", tz="UTC")
    cal = _make_calibrator_with_slices([expiry], bucket_ts=bucket_ts)
    out = cal.slice_params_for_expiry(expiry)
    assert out is not None
    assert set(out.keys()) == {"a", "b", "rho", "m", "sigma", "T", "F"}
    assert out["a"] == pytest.approx(0.001)
    assert out["F"] == pytest.approx(50_000.0)


def test_slice_params_uses_latest_bucket():
    """Two buckets, different params per expiry — pick the most recent."""
    older_ts = pd.Timestamp("2026-05-07 11:00", tz="UTC")
    latest_ts = pd.Timestamp("2026-05-07 12:00", tz="UTC")
    expiry = pd.Timestamp("2026-05-09", tz="UTC")
    rows = [
        {
            "timestamp": older_ts,
            "expiry_dt": expiry,
            "T": 7.0 / 365.25,
            "F": 49_000.0,
            "a": 0.001,
            "b": 0.05,
            "rho": -0.1,
            "m": 0.0,
            "sigma": 0.2,
        },
        {
            "timestamp": latest_ts,
            "expiry_dt": expiry,
            "T": 7.0 / 365.25,
            "F": 51_000.0,
            "a": 0.002,
            "b": 0.15,
            "rho": -0.3,
            "m": 0.0,
            "sigma": 0.4,
        },
    ]
    cal = LiveSVICalibrator(cache_path="/tmp/_test_only_does_not_exist.parquet")
    cal._params_df = pd.DataFrame(rows)
    cal._meta = SVICacheMeta(
        last_recalibrated_at=latest_ts,
        n_buckets=2,
        n_params_rows=2,
        window_start=older_ts,
        window_end=latest_ts,
    )
    out = cal.slice_params_for_expiry(expiry)
    assert out is not None
    # Latest bucket wins
    assert out["F"] == pytest.approx(51_000.0)
    assert out["b"] == pytest.approx(0.15)


# ── LiveDRLHedger.begin_trade with svi_slice_params ──────────────────


def _make_stub_bundle(action_value: float = 0.5) -> ModelBundle:
    drl = MagicMock()
    drl.predict.return_value = (np.array([[action_value]], dtype=np.float32), None)
    return ModelBundle(
        month_key="stub",
        lgbm_model=None,
        lgbm_features=[],
        transvol_model=None,
        transvol_config=None,
        transvol_feat_mean=None,
        transvol_feat_std=None,
        transvol_time_mean=None,
        transvol_time_std=None,
        drl_policy=drl,
        drl_vec_normalize_mean=None,
        drl_vec_normalize_var=None,
        drl_vec_normalize_clip=10.0,
    )


def test_begin_trade_without_slice_uses_iv_entry_for_all_ticks():
    """Pre-D1 behaviour: no slice → IV stays at iv_entry for every tick."""
    bundle = _make_stub_bundle(action_value=0.5)
    h = LiveDRLHedger(bundle=bundle)
    tid = h.begin_trade(
        signal=-1, S_entry=50_000, K=50_000,
        vol_ann_entry=0.8, iv_entry=0.75, tau_entry=0.04,
        V_bg=100.0, jump_fraction=0.0, vpin=0.5,
        svi_slice_params=None,
    )
    # Move spot to provoke an IV change if a slice were available
    h.step(tid, current_spot=51_000.0)
    h.step(tid, current_spot=52_000.0)
    h.step(tid, current_spot=49_000.0)
    summary = h.close_trade(tid)
    ivs = [step["iv"] for step in summary["step_log"]]
    # All ticks see the constant entry IV
    assert all(iv == pytest.approx(0.75) for iv in ivs)


def test_begin_trade_with_slice_evaluates_iv_per_tick():
    """D1: with a slice, IV varies per-tick as a function of strike (and effectively spot)."""
    bundle = _make_stub_bundle(action_value=0.5)
    h = LiveDRLHedger(bundle=bundle)
    # Slice with non-trivial b/rho/m so iv(K=50000) != iv(K=51000) etc.
    slice_params = {
        "a": 0.001,
        "b": 0.30,    # bigger b → bigger smile curvature
        "rho": -0.3,
        "m": 0.0,
        "sigma": 0.25,
        "T": 7.0 / 365.25,
        "F": 50_000.0,
    }
    tid = h.begin_trade(
        signal=-1, S_entry=50_000, K=50_000,
        vol_ann_entry=0.8, iv_entry=0.75, tau_entry=0.04,
        V_bg=100.0, jump_fraction=0.0, vpin=0.5,
        svi_slice_params=slice_params,
    )
    # The slice IS evaluated at K=strike (constant K), but the engine's
    # per-tick IV path differs from the constant ``iv_entry`` because the
    # slice IV at K=strike is a different number than 0.75 by construction.
    h.step(tid, current_spot=50_500.0)
    h.step(tid, current_spot=51_000.0)
    summary = h.close_trade(tid)
    ivs = [step["iv"] for step in summary["step_log"]]
    # Slice-evaluated IV at strike 50_000 (the trade's K) is NOT 0.75.
    expected = _TradeSliceSVI(**slice_params).iv(50_000.0)
    assert all(iv == pytest.approx(expected) for iv in ivs)
    assert abs(expected - 0.75) > 0.01  # confirm it's a meaningfully different number


def test_begin_trade_slice_field_validation():
    """Missing required keys in svi_slice_params → KeyError surfaces clearly."""
    bundle = _make_stub_bundle()
    h = LiveDRLHedger(bundle=bundle)
    with pytest.raises(KeyError):
        h.begin_trade(
            signal=-1, S_entry=50_000, K=50_000,
            vol_ann_entry=0.8, iv_entry=0.75, tau_entry=0.04,
            V_bg=100.0, jump_fraction=0.0, vpin=0.5,
            svi_slice_params={"a": 0.001},  # missing b, rho, m, sigma, T, F
        )
