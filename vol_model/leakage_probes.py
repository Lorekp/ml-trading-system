"""Leakage probe battery for signal-research candidates.

Hard gate: runs BEFORE each candidate's spot-check. Any probe failure
is an automatic rejection -- a candidate that fails leakage hygiene is
not allowed to produce QLIKE numbers the user might otherwise trust.

Design principle: these probes are VERIFICATION, not audit. The codebase
already has strong leakage guards (align_features_labels 1-bar lag,
+15-min SVI shift, DVOL +1h shift, END-stamping convention, PurgedKFold).
Prior audits (project_leakage_audit.md) cleared TSRV/vol_model/vol_arb
with one micro-leak fixed in slippage surface. We TRUST those guards;
the probes here catch regressions introduced by new code added in this
session -- not re-audit the existing pipeline.

Probes:
    A. Shuffle-label: permute y in training, expect QLIKE near HAR baseline
       (model with no real signal). Failure = candidate is learning from
       label information that shouldn't be in its features.
    B. Future-bar source trace: sample test-month rows, assert every
       feature value's source timestamp < feature row timestamp.
    C. Label alignment: label_timestamp > feature_timestamp on every row.
    D. Parquet END-stamp: any parquet written by this session's code uses
       ceiling-aligned timestamps (matches pipeline convention).
"""

from __future__ import annotations

import logging
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
