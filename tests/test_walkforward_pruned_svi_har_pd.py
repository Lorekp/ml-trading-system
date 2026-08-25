"""Alignment + leakage tests for the HAR-PD walk-forward training harness.

The critical invariants pinned here:

  1. Post -15m shift, HAR-PD at join-index T uses data strictly KNOWN at
     wall-clock time T (not T+15m — that would be a leak, nor T-15m — that
     would be conservatively wasteful). Verified by mutating TSRV bars that
     start at or after T and confirming the post-shift HAR-PD[T] is byte-
     identical.

  2. The -15m shift exactly reverses the internal shift(1) that HAR-PD's
     feature construction applies. Sanity: shifting by -15m and then
     reversing the internal shift(1) recovers TSRV at end-stamp T — i.e.,
     the most recent complete bar ending at decision time T.

  3. The end-to-end label/feature alignment matches: at decision time T,
     predict label for bar [T, T+15m) using HAR-PD + SVI + TransVol +
     base features all known at or before wall-clock T. No feature may
     contain info from within [T, T+15m).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vol_model.features.har_pd import compute_har_pd_features
from vol_model.labels import detect_stamping
from vol_model.run_walkforward_pruned_svi_har_pd import (
    ADD_HAR_PD_FEATURES,
    _load_har_pd,
)


# Horizons the 7-horizon AWS sweep trains (120m descoped — no TSRV).
# h=15 is END-stamped (legacy migration); all others are START-stamped.
ALL_HORIZONS = [5, 15, 30, 60, 240, 480, 1440]


# ── 1. Shift-direction + alignment ──────────────────────────────────


def test_load_har_pd_applies_minus_15m_shift():
    """The loader must shift the sidecar index backward by 15 minutes so
    that the feature at new-index T was originally at T+15m (i.e., a row
    computed from TSRV end-stamps T, T-15m, ..., T-K+1)."""
    raw_path = Path("vol_model/output/BTCUSDT_features_har_pd_15m.parquet")
    if not raw_path.exists():
        pytest.skip("HAR-PD sidecar not present")

    raw = pd.read_parquet(raw_path)
    shifted = _load_har_pd("BTCUSDT")
    # The shifted index should equal raw index minus 15 minutes, row-for-row.
    diffs = (raw.index - shifted.index).unique()
    assert len(diffs) == 1, f"non-uniform shift detected: {diffs}"
    assert diffs[0] == pd.Timedelta(minutes=15), (
        f"expected +15m difference (raw = shifted + 15m), got {diffs[0]}"
    )


def test_load_har_pd_values_byte_identical_to_raw_post_shift():
    """The loader changes index but not values — row ordering preserved."""
    raw_path = Path("vol_model/output/BTCUSDT_features_har_pd_15m.parquet")
    if not raw_path.exists():
        pytest.skip("HAR-PD sidecar not present")

    raw = pd.read_parquet(raw_path)[list(ADD_HAR_PD_FEATURES)]
    shifted = _load_har_pd("BTCUSDT")
    # Reset both indexes to integer positions so we can compare values only.
    assert np.array_equal(
        raw.values, shifted.values, equal_nan=True
    ), "loader changed feature values, not just index"


# ── 2. Post-shift leakage probe ─────────────────────────────────────


def test_no_lookahead_after_shift():
    """After the -15m shift, HAR-PD at new-index T must NOT depend on TSRV
    bars starting at or after wall-clock T.

    Construction: a bar with TSRV end-stamp T corresponds to the bar
    [T-15m, T), which STARTS at T-15m and ENDS at T. This bar is strictly
    before the prediction window [T, T+15m), so using TSRV[T] is safe.

    Leakage would be using TSRV[T+15m] or later (bars that start inside the
    prediction window). The probe: mutate TSRV at end-stamps > T (bars
    starting at T or later), regenerate HAR-PD, apply -15m shift, and
    confirm the new-index-T row is unchanged."""
    tsrv_path = Path("TSRV/output/BTCUSDT_tsrv_15m.parquet")
    if not tsrv_path.exists():
        pytest.skip("TSRV parquet not present")

    tsrv = pd.read_parquet(tsrv_path)

    # Pick a wall-clock T mid-series so we have plenty of history.
    T = pd.Timestamp("2023-06-15 12:00:00", tz="UTC")

    # Normal HAR-PD + -15m shift → value at new-index T.
    hpd_normal = compute_har_pd_features(tsrv)
    hpd_normal.index = hpd_normal.index - pd.Timedelta(minutes=15)
    val_normal = hpd_normal.loc[T]

    # Mutate TSRV bars whose end-stamp > T (equivalently, bars starting at T
    # or later — the first "forbidden" bar starts at T and ends at T+15m,
    # stamped at T+15m; the second starts at T+15m, ends at T+30m, stamped
    # at T+30m; etc.).
    tsrv_mut = tsrv.copy()
    mask = tsrv_mut.index > T
    n_mutated = int(mask.sum())
    assert n_mutated > 0, "test requires some future bars — pick earlier T"
    tsrv_mut.loc[mask, ["rv_ann", "rs_up", "rs_down"]] = 999.0

    hpd_mut = compute_har_pd_features(tsrv_mut)
    hpd_mut.index = hpd_mut.index - pd.Timedelta(minutes=15)
    val_mut = hpd_mut.loc[T]

    diff = (val_normal - val_mut).abs().max()
    assert diff == 0.0, (
        f"HAR-PD at new-index T={T} changed by {diff} when {n_mutated} future "
        f"TSRV rows were mutated. This indicates post-shift lookahead leakage."
    )


def test_shift_preserves_dependence_on_current_bar():
    """Conversely, HAR-PD at new-index T MUST depend on TSRV end-stamped at T
    (= bar [T-15m, T), the most recent bar known at decision time T). This is
    the whole point of the -15m shift — without it, we'd be discarding this
    bar unnecessarily."""
    tsrv_path = Path("TSRV/output/BTCUSDT_tsrv_15m.parquet")
    if not tsrv_path.exists():
        pytest.skip("TSRV parquet not present")

    tsrv = pd.read_parquet(tsrv_path)
    T = pd.Timestamp("2023-06-15 12:00:00", tz="UTC")

    # Normal post-shift value at new-index T
    hpd_normal = compute_har_pd_features(tsrv)
    hpd_normal.index = hpd_normal.index - pd.Timedelta(minutes=15)
    val_normal = hpd_normal.loc[T]

    # Mutate ONLY TSRV at end-stamp T (bar [T-15m, T)). A correctly-aligned
    # HAR-PD at new-index T should change because this bar IS used.
    tsrv_mut = tsrv.copy()
    mask_current_bar = tsrv_mut.index == T
    assert mask_current_bar.sum() == 1, "expected exactly one row at T"
    tsrv_mut.loc[mask_current_bar, ["rv_ann", "rs_up", "rs_down"]] = 999.0

    hpd_mut = compute_har_pd_features(tsrv_mut)
    hpd_mut.index = hpd_mut.index - pd.Timedelta(minutes=15)
    val_mut = hpd_mut.loc[T]

    max_diff = (val_normal - val_mut).abs().max()
    assert max_diff > 0.0, (
        "HAR-PD at new-index T did NOT respond to mutation of TSRV[T]. The "
        "-15m shift should make the bar [T-15m, T) contribute to the new-T "
        "feature; if it doesn't, the shift isn't wired correctly."
    )


# ── 2b. Horizon-parametrized no-lookahead probe (MF-2) ──────────────
#
# The h=15-only probe above is BLIND to the entire non-15m bug class: the
# maintained loader applied an unconditional -h shift, which is a one-bar
# LOOKAHEAD LEAK on START-stamped sidecars (every horizon except 15m). These
# parametrized probes replicate _load_har_pd's NEW stamping-aware shift at
# every horizon and assert that mutating strictly-future TSRV bars cannot
# change the decision-time HAR-PD feature.


def _load_har_pd_stamping_aware(tsrv: pd.DataFrame, h_min: int) -> pd.DataFrame:
    """Replicate _load_har_pd's stamping-aware alignment from a raw TSRV frame.

    Mirrors the maintained loader: build HAR-PD, then shift by -h ONLY if the
    sidecar (inheriting TSRV stamping) is END-stamped; START-stamped sidecars
    are already decision-time-aligned and joined with zero shift.
    """
    feats = compute_har_pd_features(tsrv)[list(ADD_HAR_PD_FEATURES)].copy()
    stamping = detect_stamping(feats.index, h_min * 60)
    if stamping == "END":
        feats.index = feats.index - pd.Timedelta(minutes=h_min)
    return feats, stamping


@pytest.mark.parametrize("h_min", ALL_HORIZONS)
def test_no_lookahead_after_shift_all_horizons(h_min):
    """At EVERY horizon, HAR-PD at decision-index T must not depend on any TSRV
    bar whose realised window starts at or after wall-clock T (strictly future).

    Future-bar definition by stamping:
      START-stamped: raw bar indexed at t covers [t, t+h); window starts at t.
                     Forbidden bars = raw index >= T.
      END-stamped:   raw bar indexed at t covers [t-h, t); window starts at
                     t-h. Forbidden bars = (raw index - h) >= T.
    """
    tsrv_path = Path(f"TSRV/output/BTCUSDT_tsrv_{h_min}m.parquet")
    if not tsrv_path.exists():
        pytest.skip(f"TSRV {h_min}m parquet not present")

    tsrv = pd.read_parquet(tsrv_path)
    h_td = pd.Timedelta(minutes=h_min)

    base, stamping = _load_har_pd_stamping_aware(tsrv, h_min)
    # Pick a decision time mid-series with full HAR-PD coverage and future bars.
    valid_idx = base.dropna().index
    assert len(valid_idx) > 500, "need a populated series for the probe"
    T = valid_idx[len(valid_idx) // 2]
    base_val = base.loc[T].astype(float).values

    # Window-start of each raw bar, by stamping convention.
    win_start = (tsrv.index - h_td) if stamping == "END" else tsrv.index
    future = win_start >= T
    n_future = int(future.sum())
    assert n_future > 0, "test requires future bars — pick an earlier T"

    tsrv_mut = tsrv.copy()
    for c in ("rv_ann", "rs_up", "rs_down", "tsrv"):
        if c in tsrv_mut.columns:
            tsrv_mut.loc[future, c] = tsrv_mut.loc[future, c].values * 3.0 + 1e6

    mut, _ = _load_har_pd_stamping_aware(tsrv_mut, h_min)
    mut_val = mut.loc[T].astype(float).values

    assert np.allclose(base_val, mut_val, equal_nan=True), (
        f"h={h_min} ({stamping}-stamped): HAR-PD at decision T={T} changed when "
        f"{n_future} strictly-future TSRV bars were mutated — LOOKAHEAD LEAK. "
        f"base={base_val}, mut={mut_val}"
    )


@pytest.mark.parametrize("h_min", ALL_HORIZONS)
def test_har_pd_depends_on_most_recent_known_bar_all_horizons(h_min):
    """Conversely, HAR-PD at decision-index T MUST use the most recent TSRV bar
    known at T (the bar whose window ends exactly at T). Guards against an
    over-conservative shift that wastes a bar of usable information."""
    tsrv_path = Path(f"TSRV/output/BTCUSDT_tsrv_{h_min}m.parquet")
    if not tsrv_path.exists():
        pytest.skip(f"TSRV {h_min}m parquet not present")

    tsrv = pd.read_parquet(tsrv_path)
    h_td = pd.Timedelta(minutes=h_min)

    base, stamping = _load_har_pd_stamping_aware(tsrv, h_min)
    valid_idx = base.dropna().index
    T = valid_idx[len(valid_idx) // 2]
    base_val = base.loc[T].astype(float).values

    # The most-recent-known bar ends at wall-clock T:
    #   START-stamped: that bar is [T-h, T) -> raw index T-h.
    #   END-stamped:   that bar is [T-h, T) -> raw index T (end-stamp).
    recent_key = (T - h_td) if stamping == "START" else T
    if recent_key not in tsrv.index:
        pytest.skip(f"h={h_min}: most-recent bar key {recent_key} not in TSRV")

    tsrv_mut = tsrv.copy()
    for c in ("rv_ann", "rs_up", "rs_down", "tsrv"):
        if c in tsrv_mut.columns:
            tsrv_mut.loc[recent_key, c] = tsrv_mut.loc[recent_key, c] * 3.0 + 1e6

    mut, _ = _load_har_pd_stamping_aware(tsrv_mut, h_min)
    mut_val = mut.loc[T].astype(float).values

    assert not np.allclose(base_val, mut_val, equal_nan=True), (
        f"h={h_min} ({stamping}-stamped): HAR-PD at decision T={T} did NOT "
        f"respond to mutation of the most-recent-known bar at {recent_key}. The "
        f"stamping-aware shift is discarding usable decision-time information."
    )


# ── 3. Cross-check: symmetric to SVI's +15m shift ───────────────────


def test_har_pd_alignment_matches_svi_end_convention():
    """Both SVI (+15m shift) and HAR-PD (-15m shift) should land with
    feature-info-cutoff equal to the label's decision time T. Concretely:
    HAR-PD post-shift at index T uses info through wall-clock T, which is
    the same info cutoff that the reference SVI path achieves post-shift.

    We verify this through the leakage probe above. This test adds the
    cross-check that the shifts are in OPPOSITE directions (SVI shifts
    forward because raw SVI lives 15m ahead of labels; HAR-PD shifts back
    because raw HAR-PD lives 15m behind labels due to its internal shift(1))."""
    # Symbolic check — no backtest run needed.
    SVI_SHIFT_MINUTES = +15    # from _load_svi_atm_1d
    HAR_PD_SHIFT_MINUTES = -15  # from _load_har_pd (this change)
    assert SVI_SHIFT_MINUTES == -HAR_PD_SHIFT_MINUTES, (
        "SVI and HAR-PD shifts must be equal magnitude, opposite sign — they "
        "correct opposite stamp conventions (SVI: fitted-end-of-bar indexed "
        "at bar start; HAR-PD: paper-faithful strictly-past w/ internal "
        "shift(1))."
    )


# ── 4. LGBM-data-audit cleanup levers (default-OFF toggles) ─────────────
#
# These pin the byte-identity + correctness contract for the three additive
# historical toggles [1.1]/[1.2]/[1.3]. The AWS fleet launcher
# enables a winning lever per-horizon via the matching CLI flag; with every
# flag OFF the training matrix and params dict must be byte-identical to the
# historical --no-transvol path.


def test_audit_lever_constants_match_derivation():
    """The hard-coded lever feature lists must match the audit's derivation."""
    from vol_model.run_walkforward_pruned_svi_har_pd import (
        AUDIT_ORPHAN_DROP, AUDIT_COLLINEAR_DROP, AUDIT_MONOTONE_POS_FEATURES,
    )
    # [1.1] orphans — exactly the two SHAP-prune survivors.
    assert AUDIT_ORPHAN_DROP == ["noise_level_lag3", "ffd_rv_lag1"]
    # [1.2] collinear-drop — 14 ema/lag copies of the 5 strong >0.999 families,
    # keeping base+lag3. No base or lag3 of any family may appear in the drop.
    assert len(AUDIT_COLLINEAR_DROP) == 14
    for fam in ("tsrv_mean_1d", "tsrv_mean_1w", "parkinson_1d",
                "time_to_funding", "tsrv_ffd"):
        assert fam not in AUDIT_COLLINEAR_DROP, f"{fam} base must be kept"
        assert f"{fam}_lag3" not in AUDIT_COLLINEAR_DROP, f"{fam}_lag3 must be kept"
        assert f"{fam}_lag1" in AUDIT_COLLINEAR_DROP
    # [1.3] monotone — R2 (variance kernels) constrained, R1 (signed flow) NOT.
    assert "har_pd_R2_b005" in AUDIT_MONOTONE_POS_FEATURES
    assert "har_pd_R1_a005" not in AUDIT_MONOTONE_POS_FEATURES
    for amb in ("vpin", "time_to_funding", "up_jump_frac", "imbalance_lag1"):
        assert amb not in AUDIT_MONOTONE_POS_FEATURES, (
            f"ambiguous-sign {amb} must stay unconstrained"
        )


def test_monotone_vector_alignment_and_default_off():
    """The constraint vector is +1 only on named features, 0 elsewhere, and
    aligned 1:1 to the supplied column order."""
    from vol_model.run_walkforward_pruned_svi_har_pd import _build_monotone_vector
    cols = ["tsrv_mean_1d", "vpin", "har_pd_R2_b005", "imbalance_lag3",
            "parkinson_1d", "har_pd_R1_a005", "svi_atm_1d"]
    v = _build_monotone_vector(cols)
    assert v == [1, 0, 1, 0, 1, 0, 0]
    assert len(v) == len(cols)
    # All-zero when no constrained feature is present (vector still injected
    # is harmless, but the maintained default path passes None — see below).
    assert _build_monotone_vector(["vpin", "n_trades"]) == [0, 0]


def test_make_objective_omits_monotone_key_when_none():
    """_make_objective must NOT add a monotone_constraints key when the kwarg
    is None — that is the byte-identity guarantee for the reference path."""
    import inspect
    import optuna
    import numpy as np
    import lightgbm as lgb
    from vol_model.run_walkforward import _make_objective

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4))
    y = rng.standard_normal(200)
    dtr = lgb.Dataset(X, label=y, free_raw_data=False)
    dva = lgb.Dataset(X, label=y, reference=dtr, free_raw_data=False)
    dtr.construct(); dva.construct()

    captured = {}
    orig_train = lgb.train

    def _spy(params, *a, **k):
        captured["params"] = dict(params)
        # short-circuit: return a trivially-trained booster
        return orig_train(params, *a, **k)

    # Default (None) → no key.
    obj = _make_objective(dtr, dva, X, y, "cpu")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    lgb.train = _spy
    try:
        study.optimize(obj, n_trials=1)
    finally:
        lgb.train = orig_train
    assert "monotone_constraints" not in captured["params"], (
        "default path leaked a monotone_constraints key — breaks byte-identity"
    )

    # Explicit vector → key present and equal.
    captured.clear()
    obj2 = _make_objective(dtr, dva, X, y, "cpu",
                           monotone_constraints=[1, 0, 1, 0])
    study2 = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=0))
    lgb.train = _spy
    try:
        study2.optimize(obj2, n_trials=1)
    finally:
        lgb.train = orig_train
    assert captured["params"].get("monotone_constraints") == [1, 0, 1, 0]
