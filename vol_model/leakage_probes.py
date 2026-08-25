"""Leakage probe battery for signal-research candidates.

Hard gate: runs BEFORE each candidate's spot-check. Any probe failure
is an automatic rejection -- a candidate that fails leakage hygiene is
not allowed to produce QLIKE numbers the user might otherwise trust.

These are targeted regression probes, not a complete leakage audit. They cover
the named lag, shift, timestamp, and purged-fold mechanisms only. Passing them
does not certify the wider pipeline or any result.

Probes:
    A. Shuffle-label: permute y in training, expect QLIKE near HAR baseline
       (model with no real signal). Failure = candidate is learning from
       label information that shouldn't be in its features.
    B. Future-bar source trace: sample test-month rows, assert every
       feature value's source timestamp < feature row timestamp.
    C. Label alignment: label_timestamp > feature_timestamp on every row.
    D. Parquet END-stamp: any parquet written by this session's code uses
       ceiling-aligned timestamps (matches pipeline convention).
    E. Causality by mutation: mutate raw data at timestamps AFTER T, regenerate
       the feature end-to-end, assert feature[T] is BYTE-IDENTICAL. This is
       the gold-standard probe -- unlike B, it doesn't rely on metadata; it
       verifies the feature FUNCTION is causal. New in 2026-04 (Session E).
    F. Current-bar dependence: mutate raw data AT T, regenerate the feature,
       assert feature[T] DOES change. Complementary to E: catches wastefully
       conservative features that are throwing away information they could
       legally use. Not a leakage probe per se; a feature-quality sanity.
    G. Walk-forward isolation: given a fold schedule, assert no row in any
       fold's training window appears in its test window. Catches schedule
       bugs without rerunning the training loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    name: str
    passed: bool
    detail: str
    metric: float | None = None


# ---------- Probe A: shuffle-label ----------

def probe_shuffle_label(
    train_fn,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    qlike_upper_bound: float = 1.0,
    qlike_lower_bound: float = 0.7,
    seed: int = 13,
) -> ProbeResult:
    """Train the candidate on permuted y, expect QLIKE near HAR baseline.

    If the candidate achieves QLIKE below `qlike_lower_bound` on
    shuffled labels, it has learned label information that leaked
    through feature engineering -- FAIL.

    If the QLIKE is between the bounds, the candidate is behaving
    correctly: with no real signal, it defaults to predicting
    population mean, which gives QLIKE ~= variance of y.

    Parameters
    ----------
    train_fn
        Callable ``(X_train, y_train, X_test) -> np.ndarray[preds]``.
        Must be self-contained: train on X/y, predict on X_test.
    """
    from vol_model.model import qlike_loss

    rng = np.random.default_rng(seed)
    y_perm = pd.Series(rng.permutation(y_train.to_numpy()), index=y_train.index)

    try:
        preds = train_fn(X_train, y_perm, X_test)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            name="shuffle_label",
            passed=False,
            detail=f"train_fn raised {type(exc).__name__}: {exc}",
        )

    try:
        q = qlike_loss(y_test.to_numpy(), np.asarray(preds))
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            name="shuffle_label",
            passed=False,
            detail=f"qlike computation failed: {exc}",
        )

    if q < qlike_lower_bound:
        return ProbeResult(
            name="shuffle_label",
            passed=False,
            detail=(
                f"Shuffled-label QLIKE={q:.4f} is BELOW lower bound "
                f"{qlike_lower_bound:.4f} -- suspicious label leakage "
                f"(candidate learned from shuffled labels better than "
                f"HAR baseline)."
            ),
            metric=q,
        )
    if q > qlike_upper_bound * 2:
        # Very high QLIKE = numerically unstable or broken model; fail
        # defensively so the real spot-check doesn't report garbage.
        return ProbeResult(
            name="shuffle_label",
            passed=False,
            detail=f"Shuffled-label QLIKE={q:.4f} is numerically unstable.",
            metric=q,
        )

    return ProbeResult(
        name="shuffle_label",
        passed=True,
        detail=f"QLIKE={q:.4f} in expected null-signal range.",
        metric=q,
    )


# ---------- Probe B: future-bar source trace ----------

def probe_future_bar_trace(
    X_test: pd.DataFrame,
    source_index_bounds: dict[str, pd.Timestamp],
    *,
    n_samples: int = 50,
    seed: int = 13,
) -> ProbeResult:
    """For N random test rows, assert every feature's source-parquet max
    timestamp is strictly less than the row's feature timestamp.

    Parameters
    ----------
    source_index_bounds
        Mapping ``{feature_name: max_source_timestamp}``. Caller builds
        this once by loading each source parquet and recording its max
        index. The probe checks that no test-row timestamp is AFTER the
        source max (which would indicate future data in the feature).

    Rationale: THIS is the probe you run BEFORE training a new feature,
    not after. The source-parquet max is "the latest timestamp this
    feature can possibly know about"; every test row must post-date it
    for the feature to be non-forward-looking.
    """
    if X_test.empty:
        return ProbeResult(
            name="future_bar_trace",
            passed=False,
            detail="X_test is empty; cannot sample.",
        )

    rng = np.random.default_rng(seed)
    n = min(n_samples, len(X_test))
    sample_idx = rng.choice(len(X_test), size=n, replace=False)
    rows = X_test.iloc[sample_idx]

    failures: list[str] = []
    for feature, src_max in source_index_bounds.items():
        # Test rows must have timestamp >= src_max (feature value
        # available by that time) but their OWN timestamp must also be
        # > src_max for the feature to have been computed causally.
        # Here we check: for each sampled test row, the feature column
        # in that row was computed from data at or before src_max, and
        # the test row timestamp > src_max is the safety margin.
        violated = rows.index[rows.index <= src_max]
        if len(violated) > 0:
            failures.append(
                f"{feature}: {len(violated)}/{n} test rows at-or-before "
                f"source max {src_max}"
            )

    if failures:
        return ProbeResult(
            name="future_bar_trace",
            passed=False,
            detail="; ".join(failures),
        )

    return ProbeResult(
        name="future_bar_trace",
        passed=True,
        detail=f"All {len(source_index_bounds)} features pass future-bar trace on {n} rows.",
    )


# ---------- Probe C: label alignment ----------

def probe_label_alignment(
    feature_index: pd.DatetimeIndex,
    label_index: pd.DatetimeIndex,
    *,
    expected_shift: pd.Timedelta | None = None,
    n_samples: int = 1000,
    seed: int = 13,
) -> ProbeResult:
    """Verify label_timestamp > feature_timestamp for every aligned row.

    If `expected_shift` is provided, additionally asserts
    ``label_timestamp == feature_timestamp + expected_shift``.

    Note: in the existing vol_model pipeline, ``X.index`` is the LABEL
    timestamp (t+1) and feature values come from t. So the "feature
    index" fed here should be constructed as ``X.index - freq`` if the
    caller wants to verify the shift explicitly; otherwise pass X.index
    twice and only verify `label > feature` by construction of the
    training dataset.
    """
    if len(feature_index) != len(label_index):
        return ProbeResult(
            name="label_alignment",
            passed=False,
            detail=f"length mismatch: features={len(feature_index)} labels={len(label_index)}",
        )
    if len(feature_index) == 0:
        return ProbeResult(
            name="label_alignment",
            passed=False,
            detail="empty index",
        )

    rng = np.random.default_rng(seed)
    n = min(n_samples, len(feature_index))
    sample = rng.choice(len(feature_index), size=n, replace=False)

    ft = feature_index[sample]
    lb = label_index[sample]

    if expected_shift is not None:
        diffs = lb - ft
        mismatch = (diffs != expected_shift).sum()
        if mismatch:
            return ProbeResult(
                name="label_alignment",
                passed=False,
                detail=(
                    f"{mismatch}/{n} rows have label-feature shift != "
                    f"{expected_shift}"
                ),
            )
    else:
        not_after = (lb <= ft).sum()
        if not_after:
            return ProbeResult(
                name="label_alignment",
                passed=False,
                detail=f"{not_after}/{n} rows have label <= feature timestamp",
            )

    return ProbeResult(
        name="label_alignment",
        passed=True,
        detail=f"All {n} sampled rows have label > feature timestamp.",
    )


# ---------- Probe D: parquet END-stamp ----------

def probe_parquet_end_stamped(
    parquet_path,
    expected_freq: pd.Timedelta,
) -> ProbeResult:
    """Assert every row in `parquet_path` has a ceiling-aligned timestamp
    relative to `expected_freq`.

    Production convention (TSRV pipeline Stage 12.8): END-stamped means
    the bar covering ``[T-freq, T)`` is stored at timestamp ``T``. This
    is the ONLY convention the pipeline's merge_asof(backward) calls
    assume -- a parquet written with START-stamped timestamps would
    silently look ahead by one bar when merged.
    """
    from pathlib import Path

    p = Path(parquet_path)
    if not p.exists():
        return ProbeResult(
            name="parquet_end_stamp",
            passed=False,
            detail=f"parquet not found at {p}",
        )

    df = pd.read_parquet(p)
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    elif "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], utc=True)
    else:
        return ProbeResult(
            name="parquet_end_stamp",
            passed=False,
            detail=f"{p.name} has no DatetimeIndex or 'timestamp' column",
        )

    # "END-stamped": idx == idx.ceil(freq). If a row's timestamp does
    # not round up to itself at the expected frequency, it's START- or
    # mid-bucket-stamped.
    ceiled = idx.ceil(expected_freq)
    mismatched = (ceiled != idx).sum()
    if mismatched:
        return ProbeResult(
            name="parquet_end_stamp",
            passed=False,
            detail=(
                f"{mismatched}/{len(idx)} rows in {p.name} are NOT "
                f"END-stamped at freq {expected_freq}"
            ),
        )

    return ProbeResult(
        name="parquet_end_stamp",
        passed=True,
        detail=f"{p.name}: all {len(idx)} rows END-stamped at {expected_freq}.",
    )


# ---------- Probe E: causality by mutation (gold standard) ----------

def probe_causality_by_mutation(
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    raw_df: pd.DataFrame,
    T: pd.Timestamp,
    *,
    mutation_cols: Sequence[str] | None = None,
    mutation_mask: Callable[[pd.DatetimeIndex, pd.Timestamp], np.ndarray] | None = None,
    sentinel: float = 999.0,
    atol: float = 0.0,
) -> ProbeResult:
    """Mutation-based causality probe: asserts feature_fn(raw)[T] is byte-
    identical to feature_fn(mutate_future(raw))[T].

    This is the gold-standard leakage check. Unlike ``probe_future_bar_trace``
    (which only checks timestamps), this probe actually verifies that the
    feature FUNCTION does not read from rows newer than T by mutating them
    and re-running end-to-end. A feature that passes this probe cannot be
    leaky at index T, regardless of how its implementation looks.

    This generalises the one-off ``test_no_lookahead_after_shift`` pattern
    that was used to validate HAR-PD's -15m alignment shift. Every new
    feature family in Session E onwards should include a test that calls
    this probe.

    Parameters
    ----------
    feature_fn
        Callable ``(raw_df) -> feature_df``. Must be deterministic and
        must include any alignment shifts the feature family applies
        (e.g. for HAR-PD: ``feature_fn = lambda raw: _load_har_pd_from_raw(raw)``,
        which internally does ``compute_har_pd_features(raw); index -= 15m``).
    raw_df
        Raw input DataFrame with DatetimeIndex. Must contain enough
        history at timestamps < T for ``feature_fn`` to compute a value
        at T.
    T
        Wall-clock decision time. The probe asserts feature_df.loc[T]
        does not depend on any raw row whose timestamp the ``mutation_mask``
        flags as "in the future".
    mutation_cols
        Columns of raw_df to mutate. If None, all numeric columns (int or
        float dtype) are mutated. Object/string columns are skipped.
    mutation_mask
        Callable ``(raw_index, T) -> bool array`` selecting the "forbidden"
        rows that the feature at T must not depend on. Default is
        ``raw_index > T`` — correct for END-stamped feature families
        where raw-index T means "bar ending at T" (i.e., the most recent
        bar legally known at decision time T). Override for features with
        unusual conventions; e.g., a feature whose raw input is START-
        stamped would want ``raw_index >= T``.
    sentinel
        Value to inject into future rows. 999.0 is far from typical vol
        feature magnitudes; any seepage would be trivially detectable.
    atol
        Absolute tolerance for byte-identity. Default 0.0 (strict).
        Raise only if the feature has unavoidable float non-determinism
        (e.g. numerical optimisers); 1e-12 is a reasonable relaxation.

    Returns
    -------
    ProbeResult
        ``passed=True`` with max-diff 0 on honest features. ``passed=False``
        with the observed max-diff and the list of feature columns that
        changed on mutation.
    """
    # Validate inputs
    if not isinstance(raw_df.index, pd.DatetimeIndex):
        return ProbeResult(
            name="causality_by_mutation",
            passed=False,
            detail="raw_df must have a DatetimeIndex",
        )
    if T not in raw_df.index and T > raw_df.index.max():
        return ProbeResult(
            name="causality_by_mutation",
            passed=False,
            detail=f"T={T} is beyond raw_df.index.max()={raw_df.index.max()}",
        )

    # Default mutation mask: rows strictly after T. Using np.asarray so the
    # caller can return either a numpy array or a pandas Series.
    if mutation_mask is None:
        mask_fn = lambda idx, t: np.asarray(idx > t)  # noqa: E731
    else:
        mask_fn = mutation_mask

    # Default mutation cols: all numeric
    if mutation_cols is None:
        mutation_cols = [
            c for c in raw_df.columns
            if pd.api.types.is_numeric_dtype(raw_df[c])
        ]
    if not mutation_cols:
        return ProbeResult(
            name="causality_by_mutation",
            passed=False,
            detail="no numeric columns to mutate",
        )

    # Baseline feature
    try:
        feat_normal = feature_fn(raw_df)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            name="causality_by_mutation",
            passed=False,
            detail=f"feature_fn raised on normal input: {type(exc).__name__}: {exc}",
        )

    if not isinstance(feat_normal.index, pd.DatetimeIndex):
        return ProbeResult(
            name="causality_by_mutation",
            passed=False,
            detail="feature_fn output must have a DatetimeIndex",
        )
    if T not in feat_normal.index:
        return ProbeResult(
            name="causality_by_mutation",
            passed=False,
            detail=(
                f"T={T} not in feature_fn output index "
                f"(range {feat_normal.index.min()}..{feat_normal.index.max()}). "
                "Pick a T with enough warmup history, or check the feature "
                "function's alignment shift."
            ),
        )

    # Mutated feature
    raw_mut = raw_df.copy()
    mask = mask_fn(raw_mut.index, T)
    if isinstance(mask, pd.Series):
        mask = mask.to_numpy()
    n_mutated = int(np.asarray(mask, dtype=bool).sum())
    if n_mutated == 0:
        return ProbeResult(
            name="causality_by_mutation",
            passed=False,
            detail=(
                f"mutation_mask selected 0 rows at T={T}. "
                "Pick an earlier T so there is future data to mutate."
            ),
        )
    raw_mut.loc[mask, list(mutation_cols)] = sentinel

    try:
        feat_mut = feature_fn(raw_mut)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            name="causality_by_mutation",
            passed=False,
            detail=f"feature_fn raised on mutated input: {type(exc).__name__}: {exc}",
        )

    if T not in feat_mut.index:
        return ProbeResult(
            name="causality_by_mutation",
            passed=False,
            detail=(
                f"T={T} fell out of feature index after mutation -- "
                "feature_fn is nondeterministic in its output alignment."
            ),
        )

    # Compare row at T
    row_normal = feat_normal.loc[T]
    row_mut = feat_mut.loc[T]

    # Align columns explicitly (feature fn might emit the same cols but
    # we don't assume order).
    common = [c for c in row_normal.index if c in row_mut.index]
    if len(common) != len(row_normal.index):
        missing = set(row_normal.index) - set(row_mut.index)
        return ProbeResult(
            name="causality_by_mutation",
            passed=False,
            detail=f"mutated feature lost columns: {sorted(missing)}",
        )

    v_normal = row_normal[common].to_numpy(dtype=float)
    v_mut = row_mut[common].to_numpy(dtype=float)
    # NaN handling: NaN == NaN counts as identical for causality purposes
    nan_mask = np.isnan(v_normal) & np.isnan(v_mut)
    diff = np.where(nan_mask, 0.0, np.abs(v_normal - v_mut))
    max_diff = float(np.nanmax(diff)) if len(diff) else 0.0

    if max_diff > atol:
        changed = [c for c, d in zip(common, diff) if d > atol]
        return ProbeResult(
            name="causality_by_mutation",
            passed=False,
            detail=(
                f"feature at T={T} changed by max {max_diff:.3e} "
                f"when {n_mutated} future rows were mutated; "
                f"changed cols: {changed[:8]}{'...' if len(changed) > 8 else ''}. "
                "This indicates lookahead leakage."
            ),
            metric=max_diff,
        )

    return ProbeResult(
        name="causality_by_mutation",
        passed=True,
        detail=(
            f"feature at T={T} is byte-identical under mutation of "
            f"{n_mutated} future rows across {len(mutation_cols)} cols."
        ),
        metric=max_diff,
    )


# ---------- Probe F: current-bar dependence (sanity) ----------

def probe_current_bar_dependence(
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    raw_df: pd.DataFrame,
    T: pd.Timestamp,
    *,
    mutation_cols: Sequence[str] | None = None,
    sentinel: float = 999.0,
) -> ProbeResult:
    """Sanity check: the feature at T DOES depend on raw data at T.

    Complementary to ``probe_causality_by_mutation``. If that probe passes
    with ``mask = (idx > T)`` but this probe ALSO fails (feature at T
    unchanged when raw at T is mutated), the feature is being wastefully
    conservative — it's throwing away the most recent bar it could legally
    use. This catches the "HAR-PD without the -15m shift" failure mode
    that motivated adding the shift in Session D.

    Not a leakage probe — a feature-quality sanity. Treat a failure as a
    warning, not necessarily a reject.
    """
    if not isinstance(raw_df.index, pd.DatetimeIndex):
        return ProbeResult(
            name="current_bar_dependence",
            passed=False,
            detail="raw_df must have a DatetimeIndex",
        )

    if mutation_cols is None:
        mutation_cols = [
            c for c in raw_df.columns
            if pd.api.types.is_numeric_dtype(raw_df[c])
        ]

    try:
        feat_normal = feature_fn(raw_df)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            name="current_bar_dependence",
            passed=False,
            detail=f"feature_fn raised: {type(exc).__name__}: {exc}",
        )

    if T not in feat_normal.index or T not in raw_df.index:
        return ProbeResult(
            name="current_bar_dependence",
            passed=False,
            detail=f"T={T} missing from feature or raw index",
        )

    raw_mut = raw_df.copy()
    raw_mut.loc[T, list(mutation_cols)] = sentinel

    try:
        feat_mut = feature_fn(raw_mut)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            name="current_bar_dependence",
            passed=False,
            detail=f"feature_fn raised on mutated input: {type(exc).__name__}: {exc}",
        )

    row_normal = feat_normal.loc[T]
    row_mut = feat_mut.loc[T]
    common = [c for c in row_normal.index if c in row_mut.index]
    v_normal = row_normal[common].to_numpy(dtype=float)
    v_mut = row_mut[common].to_numpy(dtype=float)
    nan_mask = np.isnan(v_normal) & np.isnan(v_mut)
    diff = np.where(nan_mask, 0.0, np.abs(v_normal - v_mut))
    max_diff = float(np.nanmax(diff)) if len(diff) else 0.0

    if max_diff == 0.0:
        return ProbeResult(
            name="current_bar_dependence",
            passed=False,
            detail=(
                f"feature at T={T} is UNCHANGED when raw at T is mutated. "
                "Feature is wasting the most recent bar it could legally use — "
                "check alignment shifts (see HAR-PD -15m Session D writeup)."
            ),
            metric=max_diff,
        )

    return ProbeResult(
        name="current_bar_dependence",
        passed=True,
        detail=f"feature at T={T} changes by {max_diff:.3e} when raw at T is mutated.",
        metric=max_diff,
    )


# ---------- Probe G: walk-forward isolation ----------

@dataclass
class FoldSpec:
    """A single walk-forward fold. All four timestamps are inclusive on
    start, exclusive on end; ``calib`` is optional for simple train/test.

    Canonical ordering: train_start <= train_end <= test_start < test_end,
    with train_end <= test_start required (no overlap).
    """
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    name: str = ""


def probe_walkforward_isolation(folds: Sequence[FoldSpec]) -> ProbeResult:
    """Assert no fold's training window overlaps its test window, and
    that the sequence of test windows is monotonic (expanding window).

    This is a schedule-level check; it does NOT verify that the actual
    training code respects the schedule. Combine with the shuffle-label
    probe for a full train/test-isolation guarantee.
    """
    if not folds:
        return ProbeResult(
            name="walkforward_isolation",
            passed=False,
            detail="no folds provided",
        )

    failures: list[str] = []
    for i, f in enumerate(folds):
        tag = f.name or f"fold_{i}"
        # Per-fold ordering
        if not (f.train_start <= f.train_end <= f.test_start < f.test_end):
            failures.append(
                f"{tag}: bad ordering "
                f"(train [{f.train_start}, {f.train_end}] "
                f"test [{f.test_start}, {f.test_end}))"
            )
            continue
        # No overlap between train and test
        if f.train_end > f.test_start:
            failures.append(
                f"{tag}: train_end {f.train_end} > test_start {f.test_start}"
            )

    # Monotonicity: expanding window means test_start[i+1] >= test_end[i]
    # (otherwise one fold's test overlaps another's).
    for i in range(len(folds) - 1):
        if folds[i + 1].test_start < folds[i].test_end:
            tag_i = folds[i].name or f"fold_{i}"
            tag_j = folds[i + 1].name or f"fold_{i + 1}"
            failures.append(
                f"{tag_i}.test_end {folds[i].test_end} > "
                f"{tag_j}.test_start {folds[i + 1].test_start} -- test overlap"
            )

    if failures:
        return ProbeResult(
            name="walkforward_isolation",
            passed=False,
            detail="; ".join(failures[:5]) + ("..." if len(failures) > 5 else ""),
        )

    return ProbeResult(
        name="walkforward_isolation",
        passed=True,
        detail=f"{len(folds)} folds: no train/test overlap, monotonic expanding windows.",
    )


# ---------- Probe battery runner ----------

def run_probe_battery(
    candidate_name: str,
    probes: list[ProbeResult],
) -> tuple[bool, list[ProbeResult]]:
    """Execute a set of probes; return (all_passed, results)."""
    all_passed = all(p.passed for p in probes)
    for p in probes:
        status = "PASS" if p.passed else "FAIL"
        logger.info("[%s] probe %s: %s -- %s", candidate_name, p.name, status, p.detail)
    return all_passed, probes
