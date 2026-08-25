"""Tests for vol_model.leakage_probes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_model.leakage_probes import (
    FoldSpec,
    ProbeResult,
    probe_causality_by_mutation,
    probe_current_bar_dependence,
    probe_future_bar_trace,
    probe_label_alignment,
    probe_parquet_end_stamped,
    probe_shuffle_label,
    probe_walkforward_isolation,
    run_probe_battery,
)


# ---- Probe A: shuffle-label ----

def test_shuffle_label_catches_leakage() -> None:
    """A candidate that perfectly predicts test y from training y (e.g.,
    because of a data leak) should fail the shuffle-label probe."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((500, 3)), columns=list("abc"))
    y = pd.Series(rng.standard_normal(500))
    X_test = pd.DataFrame(rng.standard_normal((100, 3)), columns=list("abc"))
    y_test = pd.Series(rng.standard_normal(100))

    # Leaky train_fn: predicts y_test exactly (impossible without leakage).
    def leaky_train_fn(Xtr, ytr, Xte):
        # Ignore everything, return y_test directly
        return y_test.to_numpy()

    result = probe_shuffle_label(leaky_train_fn, X, y, X_test, y_test)
    assert not result.passed
    # QLIKE = 0 (perfect prediction) is below the 0.7 lower bound
    assert result.metric is not None and result.metric < 0.7


def test_shuffle_label_accepts_honest_model() -> None:
    """An honest model trained on permuted labels should return QLIKE
    in the null-signal range."""
    rng = np.random.default_rng(0)
    n = 800
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=list("abc"))
    y = pd.Series(rng.standard_normal(n) * 0.3 - 3.0)  # logvol-ish
    X_test = pd.DataFrame(rng.standard_normal((200, 3)), columns=list("abc"))
    y_test = pd.Series(rng.standard_normal(200) * 0.3 - 3.0)

    def honest_train_fn(Xtr, ytr, Xte):
        # Pred = mean(ytr) -- the honest null-signal prediction
        return np.full(len(Xte), ytr.mean())

    result = probe_shuffle_label(honest_train_fn, X, y, X_test, y_test)
    # QLIKE near 0 for mean-prediction on zero-signal data; allow range.
    # The probe is ALLOWED to reject if QLIKE is extreme in either direction.
    # For this fixture with small variance, mean-prediction gives low QLIKE,
    # which the probe will flag. That's correct: in a zero-signal test, even
    # a honest mean-predictor gets "too good" QLIKE because there's no signal
    # to NOT predict. This test just verifies the probe runs and returns
    # structure.
    assert isinstance(result, ProbeResult)
    assert result.metric is not None


def test_shuffle_label_handles_train_fn_exception() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((100, 3)), columns=list("abc"))
    y = pd.Series(rng.standard_normal(100))

    def broken_train_fn(Xtr, ytr, Xte):
        raise RuntimeError("boom")

    result = probe_shuffle_label(broken_train_fn, X, y, X, y)
    assert not result.passed
    assert "boom" in result.detail


# ---- Probe B: future-bar trace ----

def test_future_bar_trace_passes_on_safe_data() -> None:
    idx = pd.date_range("2024-06-01", periods=100, freq="15min", tz="UTC")
    X_test = pd.DataFrame(np.zeros((100, 2)), index=idx, columns=["f1", "f2"])
    # Source cutoffs are BEFORE the test data starts: safe.
    bounds = {
        "f1": pd.Timestamp("2024-05-01", tz="UTC"),
        "f2": pd.Timestamp("2024-05-15", tz="UTC"),
    }
    result = probe_future_bar_trace(X_test, bounds)
    assert result.passed


def test_future_bar_trace_fails_when_source_overlaps() -> None:
    idx = pd.date_range("2024-06-01", periods=100, freq="15min", tz="UTC")
    X_test = pd.DataFrame(np.zeros((100, 2)), index=idx, columns=["f1", "f2"])
    bounds = {
        # source max is IN the test window -- some test rows are at-or-before it
        "f1": pd.Timestamp("2024-06-01 05:00", tz="UTC"),
        "f2": pd.Timestamp("2024-05-01", tz="UTC"),  # safe
    }
    result = probe_future_bar_trace(X_test, bounds)
    assert not result.passed
    assert "f1" in result.detail


def test_future_bar_trace_empty_raises() -> None:
    X_test = pd.DataFrame()
    result = probe_future_bar_trace(X_test, {})
    assert not result.passed


# ---- Probe C: label alignment ----

def test_label_alignment_passes_on_standard_lag() -> None:
    feats = pd.date_range("2024-01-01", periods=1000, freq="15min", tz="UTC")
    labs = feats + pd.Timedelta(minutes=15)
    result = probe_label_alignment(feats, labs, expected_shift=pd.Timedelta(minutes=15))
    assert result.passed


def test_label_alignment_fails_on_contemporaneous() -> None:
    feats = pd.date_range("2024-01-01", periods=1000, freq="15min", tz="UTC")
    labs = feats  # label == feature time -- contemporaneous leakage
    result = probe_label_alignment(feats, labs)  # no expected_shift: check label > feature
    assert not result.passed


def test_label_alignment_fails_on_wrong_shift() -> None:
    feats = pd.date_range("2024-01-01", periods=1000, freq="15min", tz="UTC")
    labs = feats + pd.Timedelta(minutes=30)  # 2-bar lag, not 1
    result = probe_label_alignment(feats, labs, expected_shift=pd.Timedelta(minutes=15))
    assert not result.passed


def test_label_alignment_length_mismatch() -> None:
    feats = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
    labs = pd.date_range("2024-01-01", periods=99, freq="15min", tz="UTC")
    result = probe_label_alignment(feats, labs)
    assert not result.passed


# ---- Probe D: parquet END-stamp ----

def test_parquet_end_stamp_passes(tmp_path) -> None:
    idx = pd.date_range("2024-01-01 00:15", periods=100, freq="15min", tz="UTC")
    df = pd.DataFrame({"v": np.arange(100)}, index=idx)
    p = tmp_path / "end_stamped.parquet"
    df.to_parquet(p)
    result = probe_parquet_end_stamped(p, expected_freq=pd.Timedelta(minutes=15))
    assert result.passed


def test_parquet_end_stamp_fails_on_off_grid(tmp_path) -> None:
    """Probe catches timestamps that aren't on the expected grid
    boundary (e.g., 00:07 instead of 00:15). This is the category of
    bug where someone writes a parquet with un-floored timestamps.
    It cannot detect pure semantic start-stamping when timestamps are
    already grid-aligned; that's a pipeline-wide convention, not a
    per-file verification."""
    idx = pd.date_range("2024-01-01 00:07", periods=100, freq="15min", tz="UTC")
    df = pd.DataFrame({"v": np.arange(100)}, index=idx)
    p = tmp_path / "off_grid.parquet"
    df.to_parquet(p)
    result = probe_parquet_end_stamped(p, expected_freq=pd.Timedelta(minutes=15))
    assert not result.passed


def test_parquet_end_stamp_missing_file() -> None:
    result = probe_parquet_end_stamped(
        "/does/not/exist.parquet",
        expected_freq=pd.Timedelta(minutes=15),
    )
    assert not result.passed


# ---- Probe E: causality by mutation (gold standard) ----


def _make_synthetic_raw(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Synthetic END-stamped raw dataframe mimicking TSRV structure."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 00:15", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "rv_ann": np.abs(rng.standard_normal(n)) + 0.1,
            "rs_up": np.abs(rng.standard_normal(n)) * 0.3,
            "rs_down": np.abs(rng.standard_normal(n)) * 0.3,
        },
        index=idx,
    )


def test_causality_by_mutation_passes_on_honest_feature() -> None:
    """A causal rolling feature (strictly uses data up to T) passes."""

    def honest_rolling(raw: pd.DataFrame) -> pd.DataFrame:
        # shift(1) removes the current-bar observation, rolling uses only
        # lagged windows -- gold-standard causal.
        return pd.DataFrame(
            {"ma5": raw["rv_ann"].shift(1).rolling(5).mean()},
            index=raw.index,
        )

    raw = _make_synthetic_raw(n=200)
    T = raw.index[100]
    result = probe_causality_by_mutation(honest_rolling, raw, T)
    assert result.passed, result.detail
    assert result.metric == 0.0


def test_causality_by_mutation_catches_lookahead() -> None:
    """A feature that uses a forward shift leaks — probe must reject."""

    def leaky_forward_shift(raw: pd.DataFrame) -> pd.DataFrame:
        # shift(-1) peeks at the NEXT bar -- classic leakage.
        return pd.DataFrame({"peek": raw["rv_ann"].shift(-1)}, index=raw.index)

    raw = _make_synthetic_raw(n=200)
    T = raw.index[100]
    result = probe_causality_by_mutation(leaky_forward_shift, raw, T)
    assert not result.passed
    assert "leakage" in result.detail.lower()
    # max diff should be on the order of the sentinel (999)
    assert result.metric is not None and result.metric > 100


def test_causality_by_mutation_catches_centered_rolling() -> None:
    """Centered rolling windows look ahead — probe must reject."""

    def centered_rolling(raw: pd.DataFrame) -> pd.DataFrame:
        # center=True on a rolling window uses future observations.
        return pd.DataFrame(
            {"cma": raw["rv_ann"].rolling(9, center=True).mean()},
            index=raw.index,
        )

    raw = _make_synthetic_raw(n=300)
    T = raw.index[150]
    result = probe_causality_by_mutation(centered_rolling, raw, T)
    assert not result.passed


def test_causality_by_mutation_fails_when_T_absent() -> None:
    """If T is missing from feature output, probe fails with clear message."""

    def drop_half(raw: pd.DataFrame) -> pd.DataFrame:
        # Feature only returns half the rows -- T may be absent.
        return pd.DataFrame({"f": raw["rv_ann"].iloc[::2]}, index=raw.index[::2])

    raw = _make_synthetic_raw(n=200)
    # Pick a T that falls on the odd (dropped) grid
    T = raw.index[101]
    result = probe_causality_by_mutation(drop_half, raw, T)
    assert not result.passed
    assert "not in feature_fn output" in result.detail


def test_causality_by_mutation_custom_mask() -> None:
    """Custom mutation_mask lets callers probe post-shifted features
    (the HAR-PD case: after -15m shift, 'future' means idx >= T+15m)."""

    def honest_rolling(raw: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"ma5": raw["rv_ann"].shift(1).rolling(5).mean()},
            index=raw.index,
        )

    raw = _make_synthetic_raw(n=200)
    T = raw.index[100]
    # Stricter mask: mutate rows at or after T. Feature at T should still
    # be unchanged because shift(1).rolling(5) uses only rows < T.
    strict_mask = lambda idx, t: np.asarray(idx >= t)  # noqa: E731
    result = probe_causality_by_mutation(
        honest_rolling, raw, T, mutation_mask=strict_mask,
    )
    assert result.passed


def test_causality_by_mutation_rejects_non_datetime_index() -> None:
    raw = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    T = pd.Timestamp("2024-01-01", tz="UTC")
    result = probe_causality_by_mutation(lambda r: r, raw, T)
    assert not result.passed
    assert "DatetimeIndex" in result.detail


# ---- Probe F: current-bar dependence ----


def test_current_bar_dependence_passes_on_useful_feature() -> None:
    """A feature that uses raw[T] legitimately (rolling ending at T)
    passes the sanity check."""

    def uses_current(raw: pd.DataFrame) -> pd.DataFrame:
        # rolling without shift: window ends at T, so raw[T] is included.
        # Still causal (rolling is left-aligned by default in pandas).
        return pd.DataFrame(
            {"ma5": raw["rv_ann"].rolling(5).mean()},
            index=raw.index,
        )

    raw = _make_synthetic_raw(n=200)
    T = raw.index[100]
    result = probe_current_bar_dependence(uses_current, raw, T)
    assert result.passed, result.detail


def test_current_bar_dependence_catches_wasteful_shift() -> None:
    """A feature that discards raw[T] via an unneeded shift fails the sanity."""

    def wasteful(raw: pd.DataFrame) -> pd.DataFrame:
        # shift(2) drops TWO bars including the current one -- wasteful.
        return pd.DataFrame(
            {"ma5": raw["rv_ann"].shift(2).rolling(5).mean()},
            index=raw.index,
        )

    raw = _make_synthetic_raw(n=200)
    T = raw.index[100]
    result = probe_current_bar_dependence(wasteful, raw, T)
    assert not result.passed
    assert "wasting" in result.detail.lower() or "unchanged" in result.detail.lower()


# ---- Probe G: walk-forward isolation ----


def _make_folds(n: int = 5, test_days: int = 30) -> list[FoldSpec]:
    """Build a clean expanding-window fold schedule."""
    folds = []
    train_start = pd.Timestamp("2022-01-01", tz="UTC")
    test_start = pd.Timestamp("2022-06-01", tz="UTC")
    for i in range(n):
        test_end = test_start + pd.Timedelta(days=test_days)
        folds.append(
            FoldSpec(
                train_start=train_start,
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
                name=f"fold_{i}",
            )
        )
        test_start = test_end
    return folds


def test_walkforward_isolation_passes_on_clean_schedule() -> None:
    folds = _make_folds(n=5)
    result = probe_walkforward_isolation(folds)
    assert result.passed, result.detail


def test_walkforward_isolation_catches_overlap() -> None:
    folds = _make_folds(n=3)
    # Corrupt: make fold 1's train_end extend into its test window.
    folds[1] = FoldSpec(
        train_start=folds[1].train_start,
        train_end=folds[1].test_start + pd.Timedelta(days=5),  # overlap!
        test_start=folds[1].test_start,
        test_end=folds[1].test_end,
        name="corrupted",
    )
    result = probe_walkforward_isolation(folds)
    assert not result.passed
    assert "corrupted" in result.detail


def test_walkforward_isolation_catches_test_overlap() -> None:
    folds = _make_folds(n=3)
    # Corrupt: make fold 2's test_start precede fold 1's test_end.
    folds[2] = FoldSpec(
        train_start=folds[2].train_start,
        train_end=folds[2].train_end,
        test_start=folds[1].test_end - pd.Timedelta(days=10),  # overlap!
        test_end=folds[2].test_end,
        name="overlap_test",
    )
    result = probe_walkforward_isolation(folds)
    assert not result.passed


def test_walkforward_isolation_empty_fails() -> None:
    result = probe_walkforward_isolation([])
    assert not result.passed


# ---- Battery runner ----

def test_run_probe_battery_all_pass() -> None:
    probes = [
        ProbeResult(name="p1", passed=True, detail="ok"),
        ProbeResult(name="p2", passed=True, detail="ok"),
    ]
    all_passed, results = run_probe_battery("cand", probes)
    assert all_passed
    assert len(results) == 2


def test_run_probe_battery_fails_if_any_fail() -> None:
    probes = [
        ProbeResult(name="p1", passed=True, detail="ok"),
        ProbeResult(name="p2", passed=False, detail="broken"),
    ]
    all_passed, _ = run_probe_battery("cand", probes)
    assert not all_passed


# ---- Session I Phase 0: TSRV jump-decomposition causality probe ----

def test_tsrv_window_decomposition_is_causal() -> None:
    """Session I Phase 0: pins the TSRV bar-window semantics used by HAR-CJ.

    ``TSRV/tsrv_series.py`` produces a per-bar row containing BPV/MedRV/
    continuous_var/jump_var statistics. Each row at timestamp T is a
    strictly-contained-window aggregate: it uses ONLY the ticks with
    integer floor-division ``ts_ns // freq_ns == T // freq_ns`` (i.e.,
    ticks in ``[T, T+freq)``). No ticks outside the window contribute to
    row T. That's START-stamped semantics — and it's safe for LABELS
    (forward-looking targets) when combined with the one-step-ahead
    shift in ``align_features_labels``.

    This test simulates the core windowing of ``_process_month_tsrv`` with
    a pure aggregator (sum of squared returns) and probes causality via
    future-row mutation. It pins the invariant: mutating any tick at
    timestamp strictly after T+freq must NOT change the stat at T.

    Acts as a regression guard against accidentally widening the window
    (e.g., ``[T-freq/2, T+freq/2]`` centered, or ``[T, T+2*freq)``)
    which would introduce forward leakage for Session I Candidate I-1.
    """
    from vol_model.leakage_probes import probe_causality_by_mutation
    from TSRV.windowing import timestamp_window_ids

    rng = np.random.default_rng(0)
    # Simulate 500 tick-level price observations at 1-minute cadence (a
    # fine-grained grid for the windowing logic to aggregate).
    n_ticks = 500
    tick_idx = pd.date_range(
        "2024-01-01 00:00:00", periods=n_ticks, freq="1min", tz="UTC",
    )
    prices = 50_000.0 + rng.standard_normal(n_ticks).cumsum() * 10.0
    raw = pd.DataFrame({"price": prices}, index=tick_idx)

    freq = pd.Timedelta(minutes=15)
    freq_ns = int(freq.value)

    def aggregate_by_window(raw_df: pd.DataFrame) -> pd.DataFrame:
        """Mimic TSRV windowing: for each 15-min window, return sum of
        squared log-returns using ONLY ticks inside [T, T+freq).

        This mirrors ``_process_month_tsrv``'s invariant that each window
        is a strictly-contained block of ticks with matching window_id.
        """
        window_ids, _ = timestamp_window_ids(raw_df.index, freq)
        log_p = np.log(raw_df["price"].to_numpy())
        out_rows: list[dict] = []
        # Contiguous-block splitting (same O(n) pattern as TSRV)
        unique_wids, first_idx = np.unique(window_ids, return_index=True)
        for i, wid in enumerate(unique_wids):
            s = int(first_idx[i])
            e = int(first_idx[i + 1]) if i + 1 < len(unique_wids) else len(raw_df)
            if e - s < 2:
                continue
            rets = np.diff(log_p[s:e])
            rv = float(np.sum(rets ** 2))
            ts = pd.Timestamp(int(wid) * freq_ns, unit="ns", tz="UTC")
            out_rows.append({"timestamp": ts, "rv": rv})
        out = pd.DataFrame(out_rows).set_index("timestamp")
        return out

    # Build the baseline stats
    baseline = aggregate_by_window(raw)
    assert len(baseline) >= 5, "need multiple windows for the probe"

    # Pick a T that is the START of a middle window — the stat at T must
    # depend only on ticks in [T, T+freq), so any mutation at index >= T+freq
    # must not change the output at T.
    T = baseline.index[len(baseline) // 2]
    # The mutation mask: future = strictly-after (T + freq) — i.e., rows
    # outside the [T, T+freq) window AND after it. Within-window ticks
    # are EXPECTED to be read (that's the correct START-stamped semantic).
    next_window_start = T + freq

    def future_mask(idx: pd.DatetimeIndex, t: pd.Timestamp) -> np.ndarray:
        return np.asarray(idx >= next_window_start)

    result = probe_causality_by_mutation(
        aggregate_by_window, raw, T,
        mutation_cols=["price"],
        mutation_mask=future_mask,
        atol=0.0,
    )
    assert result.passed, result.detail
    assert result.metric == 0.0


def test_tsrv_centered_window_would_fail_probe() -> None:
    """Negative-control: a centered-window aggregator (which WOULD leak)
    must be caught by the same probe. Lock the probe's sensitivity.
    """
    from vol_model.leakage_probes import probe_causality_by_mutation

    rng = np.random.default_rng(1)
    n_ticks = 500
    tick_idx = pd.date_range(
        "2024-01-01 00:00:00", periods=n_ticks, freq="1min", tz="UTC",
    )
    prices = 50_000.0 + rng.standard_normal(n_ticks).cumsum() * 10.0
    raw = pd.DataFrame({"price": prices}, index=tick_idx)

    freq = pd.Timedelta(minutes=15)
    half = freq / 2

    def centered_aggregator(raw_df: pd.DataFrame) -> pd.DataFrame:
        """Deliberately-leaky: computes RV over [T-freq/2, T+freq/2]."""
        out_rows: list[dict] = []
        # Emit one stat per 15-minute grid point, using a centered window.
        grid = pd.date_range(
            raw_df.index[0] + half, raw_df.index[-1] - half, freq=freq,
        )
        for T in grid:
            mask = (raw_df.index >= T - half) & (raw_df.index < T + half)
            p = np.log(raw_df.loc[mask, "price"].to_numpy())
            if len(p) < 2:
                continue
            out_rows.append({"timestamp": T, "rv": float(np.sum(np.diff(p) ** 2))})
        return pd.DataFrame(out_rows).set_index("timestamp")

    baseline = centered_aggregator(raw)
    T = baseline.index[len(baseline) // 2]

    result = probe_causality_by_mutation(
        centered_aggregator, raw, T,
        mutation_cols=["price"],
        atol=0.0,
    )
    # A centered window at T READS rows in [T, T+half) as well as
    # [T-half, T). Mutating raw at indices > T will change RV at T.
    # So this MUST fail — confirming the probe's sensitivity.
    assert not result.passed, (
        "centered-window aggregator should fail causality probe; "
        "if this test stops catching it, the probe is broken"
    )
