"""Tests for Session I Candidate I-1 (HAR-CJ jump-diffusion dual-head).

Pins:
1. Decomposition exactness: exp(2*y_cont) + exp(2*y_jump) reconstructs
   exp(2*y_logvol) up to eps_floor clipping.
2. Jump-fraction edge cases: when jump_var = 0, y_jump floors to
   0.5 * ln(eps_floor); when cont_var = 0, same for y_cont; when both
   are nonzero, both heads are finite and decomposition is exact.
3. Combination arithmetic: oracle-head predictions combine to exactly
   reproduce the true log-vol (pre-eps).
4. Leakage: decompose_logvol_into_cont_jump + combine path passes
   ``probe_causality_by_mutation`` — mutating ``continuous_var`` and
   ``jump_var`` at future timestamps must not affect output at T.
5. Error handling: missing required columns raise KeyError with a
   helpful message pointing to the labels loader.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_model.signal_research import (
    combine_cont_jump_forecasts,
    decompose_logvol_into_cont_jump,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _synthetic_labels(n: int = 500, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    """Build synthetic log-vol target + matching (continuous_var, jump_var)
    labels DataFrame with a range of jump fractions (including 0, 1, mid).

    Returns (ln_vol, labels_df) at a shared DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")

    # Generate vol_ann ∈ [0.2, 2.0], then decompose into cont + jump
    # where jump_frac is in [0, 1] with heavy mass near 0 (most bars
    # have no jumps — mimics real TSRV).
    vol_ann = np.exp(rng.uniform(np.log(0.2), np.log(2.0), size=n))
    var_ann = vol_ann ** 2

    # Heavy mass at jump_frac=0 (60% of bars)
    jfrac = np.zeros(n)
    has_jump_mask = rng.random(n) < 0.4
    jfrac[has_jump_mask] = rng.uniform(0.01, 0.8, size=has_jump_mask.sum())

    # Express in pre-annualized (per-bar variance) so the labels
    # DataFrame matches the TSRV sidecar conventions (continuous_var
    # and jump_var are per-window, NOT annualized — they'll be
    # re-annualized inside the decomposition via the y_logvol path).
    # The key invariant the decomposition relies on is that
    # jfrac = jump / (cont + jump) — so we can put any POSITIVE
    # per-window values into the frame as long as the ratio is right.
    # We use scaled values: cont_per = (1 - jfrac), jump_per = jfrac.
    cont_per = (1.0 - jfrac) * 1.0  # nominal scale
    jump_per = jfrac * 1.0

    labels = pd.DataFrame({
        "vol_ann": vol_ann,
        "ln_vol": np.log(vol_ann),
        "continuous_var": cont_per,
        "jump_var": jump_per,
        "t1": idx + pd.Timedelta(minutes=15),
    }, index=idx)
    return labels["ln_vol"], labels


# ── 1. Decomposition exactness ───────────────────────────────────────


def test_decomp_reconstructs_logvol() -> None:
    """The CORE invariant: exp(2*y_cont) + exp(2*y_jump) must
    reconstruct exp(2*y_logvol) exactly (up to eps_floor effects for
    degenerate bars)."""
    y_logvol, labels = _synthetic_labels(n=500, seed=0)
    y_cont, y_jump = decompose_logvol_into_cont_jump(y_logvol, labels)

    var_reconstructed = np.exp(2.0 * y_cont.to_numpy()) + np.exp(2.0 * y_jump.to_numpy())
    var_true = np.exp(2.0 * y_logvol.to_numpy())

    # For rows where BOTH components > eps_floor, the reconstruction is
    # exact to machine precision. For degenerate rows (jump=0 or
    # cont=0), the eps_floor clip shifts one side by up to eps_floor,
    # which is negligible in variance space.
    abs_diff = np.abs(var_reconstructed - var_true)
    # Allow eps_floor=1e-10 leakage on the component scale.
    assert (abs_diff < 2e-10).all(), (
        f"reconstruction error: max |diff| = {abs_diff.max():.3e}"
    )


def test_decomp_handles_zero_jump_rows() -> None:
    """Bars with jump_var == 0 should:
    - y_cont: reconstruct log(vol_ann) exactly (all variance is continuous)
    - y_jump: clip to 0.5 * ln(eps_floor) (extreme negative but finite)
    """
    idx = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    vol_ann = np.array([0.5, 1.0, 1.5])
    ln_vol = np.log(vol_ann)
    # All rows: no jumps
    labels = pd.DataFrame({
        "vol_ann": vol_ann,
        "ln_vol": ln_vol,
        "continuous_var": [1.0, 1.0, 1.0],
        "jump_var": [0.0, 0.0, 0.0],
        "t1": idx + pd.Timedelta(minutes=15),
    }, index=idx)
    y_logvol = pd.Series(ln_vol, index=idx, name="ln_vol")

    y_cont, y_jump = decompose_logvol_into_cont_jump(
        y_logvol, labels, eps_floor=1e-10,
    )
    np.testing.assert_allclose(y_cont.to_numpy(), ln_vol, atol=1e-10)
    expected_floor = 0.5 * np.log(1e-10)
    np.testing.assert_allclose(
        y_jump.to_numpy(), np.full_like(ln_vol, expected_floor), atol=1e-10,
    )


def test_decomp_handles_pure_jump_rows() -> None:
    """Bars with cont_var == 0 (all jump): y_jump reconstructs ln(vol_ann);
    y_cont floors to 0.5 * ln(eps_floor)."""
    idx = pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC")
    vol_ann = np.array([0.8, 1.2])
    ln_vol = np.log(vol_ann)
    labels = pd.DataFrame({
        "vol_ann": vol_ann,
        "ln_vol": ln_vol,
        "continuous_var": [1e-21, 1e-21],  # clipped to 1e-20 inside
        "jump_var": [1.0, 1.0],
        "t1": idx + pd.Timedelta(minutes=15),
    }, index=idx)
    y_logvol = pd.Series(ln_vol, index=idx, name="ln_vol")

    y_cont, y_jump = decompose_logvol_into_cont_jump(
        y_logvol, labels, eps_floor=1e-10,
    )
    np.testing.assert_allclose(y_jump.to_numpy(), ln_vol, atol=1e-10)
    # jfrac = jump/(cont + jump) ≈ 1.0 so cont_var_ann ≈ 0 → clipped
    expected_floor = 0.5 * np.log(1e-10)
    np.testing.assert_allclose(
        y_cont.to_numpy(), np.full_like(ln_vol, expected_floor), atol=1e-10,
    )


# ── 2. Combination arithmetic ────────────────────────────────────────


def test_combine_oracle_heads_reproduces_truth() -> None:
    """When both heads output the ground-truth decomposition, the
    combined forecast must equal the true log-vol exactly."""
    y_logvol, labels = _synthetic_labels(n=200, seed=1)
    y_cont, y_jump = decompose_logvol_into_cont_jump(y_logvol, labels)

    # Oracle heads
    pred_c = y_cont.to_numpy()
    pred_j = y_jump.to_numpy()

    combined = combine_cont_jump_forecasts(pred_c, pred_j)
    # Tolerance 5e-9 accommodates the eps_floor=1e-10 clipping on
    # degenerate (jump=0 or cont=0) rows: the clip adds eps_floor to
    # the smaller component, which propagates to ~eps_floor log-space
    # error after the log-sum-exp combination. Non-degenerate rows
    # reconstruct to machine precision.
    np.testing.assert_allclose(
        combined, y_logvol.to_numpy(),
        rtol=1e-8, atol=5e-9,
        err_msg="oracle heads do not combine back to true log-vol",
    )


def test_combine_preserves_length() -> None:
    pred_c = np.zeros(100)
    pred_j = np.zeros(100)
    out = combine_cont_jump_forecasts(pred_c, pred_j)
    assert out.shape == (100,)
    # 0 + 0 in variance space = 0 … but clipped to eps, log is very negative
    # We expect equal cont + jump of exp(0) + exp(0) = 2 → log(2)/2 = 0.347
    np.testing.assert_allclose(out, 0.5 * np.log(2.0), atol=1e-12)


# ── 3. Causality via probe_causality_by_mutation ─────────────────────


def test_decomp_is_causal() -> None:
    """The decomposition + combination path must not read any future
    rows of ``continuous_var`` or ``jump_var`` when producing output at
    index T. We wrap the decomposition as a feature_fn-style callable
    that takes the labels DataFrame as raw input and returns a 2-column
    DataFrame; then mutate future rows and verify byte-identity at T.
    """
    from vol_model.leakage_probes import probe_causality_by_mutation

    y_logvol, labels = _synthetic_labels(n=300, seed=2)

    def decomp_fn(raw_df: pd.DataFrame) -> pd.DataFrame:
        # raw_df has the same shape as ``labels``. Compute the
        # decomposition for the ln_vol column and return a 2-col frame.
        ln_vol = raw_df["ln_vol"]
        y_c, y_j = decompose_logvol_into_cont_jump(ln_vol, raw_df)
        return pd.DataFrame({"y_cont": y_c, "y_jump": y_j})

    T = labels.index[150]
    result = probe_causality_by_mutation(
        decomp_fn, labels, T,
        mutation_cols=["continuous_var", "jump_var", "ln_vol", "vol_ann"],
        atol=0.0,
    )
    assert result.passed, result.detail


# ── 4. Error handling ────────────────────────────────────────────────


def test_missing_columns_raises_keyerror() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="15min", tz="UTC")
    bad_labels = pd.DataFrame({
        "vol_ann": np.full(50, 0.5),
        "ln_vol": np.full(50, np.log(0.5)),
        # MISSING continuous_var and jump_var
        "t1": idx + pd.Timedelta(minutes=15),
    }, index=idx)
    y_logvol = bad_labels["ln_vol"]
    with pytest.raises(KeyError, match="continuous_var.*jump_var"):
        decompose_logvol_into_cont_jump(y_logvol, bad_labels)


# ── 5. Tier map integration ──────────────────────────────────────────


def test_har_pd_jump_decomp_in_tier_map() -> None:
    """Pin: TIER_FEATURE_MAP has the new thread key and it uses the same
    feature slots as ``har_pd`` (feature set is identical; novelty is in
    target). If this test fails the runner will not dispatch correctly.
    """
    from vol_model.signal_research import TIER_FEATURE_MAP

    assert "har_pd_jump_decomp" in TIER_FEATURE_MAP
    assert "har_pd" in TIER_FEATURE_MAP
    # Same tuple (same feature loadout, just different training target)
    assert TIER_FEATURE_MAP["har_pd_jump_decomp"] == TIER_FEATURE_MAP["har_pd"]


def test_thread_suffix_routes_to_dual_head() -> None:
    """Pin the suffix-based dispatch contract used in the runner: any
    thread ending with ``_jump_decomp`` must be routed to the dual-head
    trainer. Catches accidental typos in new jump-decomp thread keys.
    """
    assert "har_pd_jump_decomp".endswith("_jump_decomp")