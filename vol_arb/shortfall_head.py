"""Conditional-shortfall classifier on the DFL decision panel + σ²_epi.

Cycle 4 Phase A.1 / B core (plan: ``now-check-empirically-from-eager-bird.md``).

Question Cycle 4 answers
------------------------
Cycle 3 found that ``σ²_epi`` (deep-ensemble epistemic uncertainty) is
*anti-correlated* with |trade NC PnL| on V2's fired trades — collider-bias
from the entry filter that already conditions on |edge|. The Stage-37
critique reframed the question: instead of measuring "model disagreement"
(parameter-space), measure "is this trade likely to be a big loser given
the DFL decision-feature panel" (decision-space). The Cycle 3 DFL Head
(Stage 28) shipped a positive-Sharpe filter on a 10-feature panel with
target ``P(win)``. Cycle 4 reuses the DFL panel verbatim with an
asymmetric target — ``P(net_pnl_nc < -X)`` — and asks the controlled
question: does ``σ²_epi`` add information *beyond* what DFL captures?

Two classifiers, identical hyperparameters, identical walk-forward year
folds, only the feature set differs:

- **BASELINE**: 10 DFL-Head features (``vol_arb.dfl_head.DFL_HEAD_FEATURES``)
- **TREATMENT**: same 10 + ``sigma_epi_squared``

Plus a leakage-defence-in-depth **PLACEBO**: TREATMENT trained with
``sigma_epi_squared`` permuted *within each test year*. Pre-committed
gate: AUC delta on the placebo must collapse to |Δ| < 0.005.

Pre-committed gate (CONJUNCTIVE, all four required)
---------------------------------------------------
- G1: pooled-OOS AUC(TREATMENT) − AUC(BASELINE) ≥ +0.02
- G2: 95% lower bound of paired stationary block bootstrap (mean block
  length 5 days, paired by day) of the AUC delta > 0
- G3: paired DeLong p < 0.01 (one-sided, treatment > baseline)
- G4: |AUC(PLACEBO) − AUC(BASELINE)| < 0.005

Failure on any → halt; honest reject; σ²_epi adds nothing the DFL panel
already encodes for downside-asymmetric prediction. Pass → Phase C
sidecar filter under Cycle 1 Tier C-2 triple-gate.

Anti-leakage discipline
-----------------------
- DFL features use ``shift(1).rolling(...)`` (already proven by
  ``test_dfl_head.py::test_features_are_strictly_lagged``).
- ``sigma_epi_squared`` MUST come from a walk-forward parquet
  (``BTC_v2_har_pd_ensemble_var_walkforward.parquet``), produced by
  Phase A.0 — NOT the Cycle 3 K-fold parquet which leaks across years.
- Walk-forward year folds (expanding-window) match
  ``vol_arb.dfl_head.compute_walkforward_scores``: train < test_y,
  2022 pass-through (no train history), 2026 excluded (partial year).
- Placebo permutation is *within each test year* with a frozen seed;
  it does not alter cross-year temporal causality.

Anti-overfitting discipline
---------------------------
- Hyperparameters frozen at ``DFL_LGBM_PARAMS`` for ALL three
  classifiers — parity is the experimental control.
- Threshold X = $200 locked as PRIMARY hypothesis (5.91% positive rate
  globally; X=$500 sparse / X=$1000 degenerate per pre-experiment power
  analysis). X=$500 reported as descriptive secondary, no decision axis.
- Gate thresholds (+0.02, p<0.01, |Δ|<0.005, LCB>0) locked BEFORE any
  classifier training. No post-hoc tuning under any failure mode.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from vol_arb.dfl_head import DFL_HEAD_FEATURES, DFL_LGBM_PARAMS, build_head_features

logger = logging.getLogger(__name__)


# ─── Pre-committed parameters (locked in plan) ─────────────────────────

SHORTFALL_THRESHOLD_USD: float = 200.0          # PRIMARY — locked pre-training
SHORTFALL_DESCRIPTIVE_USD: tuple[float, ...] = (500.0,)  # secondary, no decision axis
SHORTFALL_TEST_YEARS: tuple[int, ...] = (2023, 2024, 2025)
SHORTFALL_FEATURE_EXTRA: str = "sigma_epi_squared"
SHORTFALL_FEATURES_TREATMENT: tuple[str, ...] = tuple(DFL_HEAD_FEATURES) + (
    SHORTFALL_FEATURE_EXTRA,
)
SHORTFALL_PLACEBO_SEED: int = 4242

# AUC-delta gate constants (locked)
GATE_G1_DELTA_MIN: float = 0.02            # pooled-OOS AUC delta ≥ +0.02
GATE_G2_BLOCK_DAYS: int = 5                # mean block length, days
GATE_G2_BLOCK_BOOTSTRAP_REPS: int = 1000   # n_reps
GATE_G2_BLOCK_BOOTSTRAP_SEED: int = 42
GATE_G3_DELONG_P_MAX: float = 0.01         # one-sided p < 0.01
GATE_G4_PLACEBO_MAX: float = 0.005         # |Δ_placebo| < 0.005


FeatureSetTag = Literal["baseline", "treatment", "placebo"]


# ─── Feature panel + target ────────────────────────────────────────────


def build_shortfall_features(
    trade_log: pd.DataFrame,
    predictions: pd.DataFrame,
    walkforward_uncertainty: pd.DataFrame,
    *,
    horizon_minutes: int = 15,
) -> pd.DataFrame:
    """Build the 11-feature panel: 10 DFL features + ``sigma_epi_squared``.

    Reuses ``vol_arb.dfl_head.build_head_features`` for the 10 DFL
    features (identical leakage discipline). Joins
    ``walkforward_uncertainty["sigma_epi"] ** 2`` as the 11th column.

    Parameters
    ----------
    trade_log : indexed by entry timestamp, must contain ``signal``,
        ``iv_used``, ``net_pnl_nc``.
    predictions : matching index, must contain ``y_pred`` and ``y_true``.
    walkforward_uncertainty : indexed by bar timestamp, must contain
        ``sigma_epi`` (standard deviation across K members; we square it).
    horizon_minutes : bar size (default 15 = production V2).

    Returns
    -------
    pd.DataFrame indexed by trade_log.index with 11 columns:
    ``DFL_HEAD_FEATURES + ['sigma_epi_squared']``. NaN propagated where
    the input columns lack history (warmup, year-2022 σ²_epi, etc).
    """
    if SHORTFALL_FEATURE_EXTRA in trade_log.columns:
        raise ValueError(
            f"trade_log already contains a column called "
            f"{SHORTFALL_FEATURE_EXTRA!r} — refusing to overwrite. "
            "Pass a clean trade log."
        )

    dfl_feats = build_head_features(
        trade_log, predictions, horizon_minutes=horizon_minutes,
    )

    if "sigma_epi" not in walkforward_uncertainty.columns:
        raise KeyError(
            "walkforward_uncertainty missing 'sigma_epi' column; got "
            f"{list(walkforward_uncertainty.columns)}"
        )
    sigma_epi_sq = (
        walkforward_uncertainty["sigma_epi"].astype(float) ** 2
    ).rename(SHORTFALL_FEATURE_EXTRA)

    feats = dfl_feats.join(sigma_epi_sq, how="left")
    # NaN fill is NOT applied here — the classifier driver decides per-fold
    # whether to drop NaN rows or impute. Pass-through is the disciplined
    # default for this module.
    return feats[list(SHORTFALL_FEATURES_TREATMENT)].copy()


def compute_shortfall_target(
    trade_log: pd.DataFrame,
    *,
    threshold_usd: float = SHORTFALL_THRESHOLD_USD,
) -> pd.Series:
    """Binary target: 1 ⇔ trade lost more than ``threshold_usd``.

    Defined ONLY on fired trades (``net_pnl_nc != 0``). Non-fired bars
    are excluded from the returned Series. Threshold is in absolute USD;
    the convention is ``net_pnl_nc < -threshold_usd``.
    """
    if "net_pnl_nc" not in trade_log.columns:
        raise KeyError("trade_log missing 'net_pnl_nc' column")
    fired = trade_log.loc[trade_log["net_pnl_nc"] != 0]
    target = (fired["net_pnl_nc"] < -float(threshold_usd)).astype(int)
    return target.rename("y_short")


# ─── Walk-forward classifier training ──────────────────────────────────


def _select_feature_columns(
    feature_set: FeatureSetTag,
) -> tuple[str, ...]:
    """Return the column order for a given feature set."""
    if feature_set == "baseline":
        return tuple(DFL_HEAD_FEATURES)
    if feature_set == "treatment":
        return SHORTFALL_FEATURES_TREATMENT
    if feature_set == "placebo":
        return SHORTFALL_FEATURES_TREATMENT
    raise ValueError(f"unknown feature_set: {feature_set!r}")


def _placebo_shuffle_within_year(
    series: pd.Series, *, seed: int = SHORTFALL_PLACEBO_SEED,
) -> pd.Series:
    """Shuffle ``series`` values WITHIN each calendar year, preserving the index.

    Used for the leakage placebo: TREATMENT with σ²_epi locally permuted
    must have AUC delta indistinguishable from BASELINE if our walk-forward
    discipline holds. Cross-year temporal causality is preserved.
    """
    rng = np.random.default_rng(seed)
    out = series.copy()
    for yr in sorted(set(series.index.year)):
        yr_mask = series.index.year == yr
        idx_yr = series.index[yr_mask]
        if len(idx_yr) == 0:
            continue
        values = series.loc[idx_yr].to_numpy().copy()
        rng.shuffle(values)
        out.loc[idx_yr] = values
    return out


def train_walkforward_shortfall_classifier(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    feature_set: FeatureSetTag,
    test_years: tuple[int, ...] = SHORTFALL_TEST_YEARS,
    min_train_trades: int = 500,
    placebo_seed: int = SHORTFALL_PLACEBO_SEED,
    early_stopping_rounds: int = 25,
    fairness_columns: tuple[str, ...] | None = None,
) -> pd.Series:
    """Train an LGBM shortfall classifier walk-forward by year.

    Mirrors ``vol_arb.dfl_head.compute_walkforward_scores`` exactly:
    expanding-window by calendar year, train on rows with
    ``index.year < test_y``, predict on rows with ``index.year == test_y``.
    Years not in ``test_years`` are skipped (NaN). 2022 is automatically
    skipped because it has no train history under the DFL contract.

    Parameters
    ----------
    features : 11-column panel from ``build_shortfall_features``.
    target : binary target from ``compute_shortfall_target``. Indexed
        by FIRED-TRADE timestamps; non-fired bars are absent.
    feature_set :
        - ``"baseline"`` → 10 DFL features
        - ``"treatment"`` → 10 DFL + ``sigma_epi_squared``
        - ``"placebo"`` → 10 DFL + within-year permuted ``sigma_epi_squared``
    test_years : year subset to score (frozen at {2023, 2024, 2025}).
    min_train_trades : skip year if fewer prior fired trades available.
    placebo_seed : within-year permutation seed (deterministic placebo).
    early_stopping_rounds : LightGBM early-stop patience.
    fairness_columns : optional sequence of column names (subset of
        ``features.columns``); rows where ANY of these columns is NaN
        are dropped from BOTH train and test sets. The default
        ``None`` means dropna on the active feature subset only — which
        is UNFAIR across feature sets when the active subset varies.
        Pass ``SHORTFALL_FEATURES_TREATMENT`` (the union of all 11 cols)
        to enforce parity: BASELINE / TREATMENT / PLACEBO all see
        exactly the same training rows. STRONGLY RECOMMENDED for the
        Phase B AUC-delta gate to avoid sample-size confounds.

    Returns
    -------
    pd.Series of P(shortfall) indexed by ``target.index`` (fired trades
    only). NaN for trades outside ``test_years`` or with insufficient
    train history.
    """
    from lightgbm import LGBMClassifier, early_stopping

    cols = _select_feature_columns(feature_set)
    # Apply fairness mask BEFORE subsetting to active cols so that all
    # feature sets see the same rows (fair sample-size comparison).
    if fairness_columns is not None:
        fairness_cols = list(fairness_columns)
        missing = set(fairness_cols) - set(features.columns)
        if missing:
            raise KeyError(
                f"fairness_columns missing from features: {sorted(missing)}"
            )
        fairness_mask = features[fairness_cols].notna().all(axis=1)
        feats = features.loc[fairness_mask, list(cols)].copy()
    else:
        feats = features[list(cols)].copy()

    if feature_set == "placebo":
        if SHORTFALL_FEATURE_EXTRA not in feats.columns:
            raise KeyError(
                f"placebo requires {SHORTFALL_FEATURE_EXTRA} in features"
            )
        feats[SHORTFALL_FEATURE_EXTRA] = _placebo_shuffle_within_year(
            feats[SHORTFALL_FEATURE_EXTRA], seed=placebo_seed,
        )

    # Restrict to fired trades with full feature coverage.
    common_idx = target.index.intersection(feats.index)
    feats_fired = feats.loc[common_idx].dropna()
    target_fired = target.loc[feats_fired.index]

    scores = pd.Series(
        np.nan, index=target.index, name=f"p_short_{feature_set}", dtype=float,
    )

    test_years_set = set(test_years)
    years_in_data = sorted(set(feats_fired.index.year))
    for test_y in years_in_data:
        if test_y not in test_years_set:
            continue
        train_mask = feats_fired.index.year < test_y
        test_mask = feats_fired.index.year == test_y
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        if n_train < min_train_trades or n_test < 1:
            logger.info(
                "Skipping test_y=%d (n_train=%d < %d, or n_test=%d)",
                test_y, n_train, min_train_trades, n_test,
            )
            continue

        X_train = feats_fired.loc[train_mask]
        X_test = feats_fired.loc[test_mask]
        y_train = target_fired.loc[X_train.index]
        y_test = target_fired.loc[X_test.index]

        clf = LGBMClassifier(**DFL_LGBM_PARAMS)
        clf.fit(
            X_train, y_train, eval_set=[(X_test, y_test)],
            callbacks=[early_stopping(early_stopping_rounds, verbose=False)],
        )
        p = clf.predict_proba(X_test)[:, 1]
        scores.loc[X_test.index] = p
        n_pos_train = int(y_train.sum())
        n_pos_test = int(y_test.sum())
        logger.info(
            "feature_set=%s test_y=%d n_train=%d (n+=%d, %.2f%%) "
            "n_test=%d (n+=%d, %.2f%%)",
            feature_set, test_y, n_train, n_pos_train,
            100.0 * n_pos_train / max(n_train, 1),
            n_test, n_pos_test,
            100.0 * n_pos_test / max(n_test, 1),
        )

    return scores


# ─── DeLong paired AUC test ────────────────────────────────────────────


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Mid-rank with ties; used by DeLong's covariance formula."""
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # 1-indexed mean rank
        i = j
    out = np.empty(N, dtype=np.float64)
    out[J] = T
    return out


def delong_paired_p_value(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    *,
    alternative: Literal["greater", "two-sided"] = "greater",
) -> tuple[float, float, float, float]:
    """Paired-AUC DeLong test (DeLong-DeLong-Clarke-Pearson 1988, Sun-Xu 2014).

    Tests H0: AUC(B) − AUC(A) = 0 vs:
    - ``"greater"``: AUC(B) > AUC(A) (one-sided; used in Phase B G3)
    - ``"two-sided"``: AUC(B) ≠ AUC(A)

    Returns ``(auc_a, auc_b, var_delta, p_value)``. The variance is the
    DeLong covariance-based variance of (AUC_b − AUC_a) accounting for
    the fact that both classifiers score the same observations.

    Parameters
    ----------
    y_true : binary labels {0, 1}, shape (N,).
    p_a, p_b : classifier scores, shape (N,).
    alternative : direction of one-sided alternative (default "greater"
        for Phase B, where TREATMENT is expected to exceed BASELINE).
    """
    from scipy.stats import norm

    y_true = np.asarray(y_true).astype(int)
    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)

    if len(y_true) != len(p_a) or len(y_true) != len(p_b):
        raise ValueError("y_true / p_a / p_b length mismatch")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must be binary {0, 1}")

    pos_mask = y_true == 1
    neg_mask = ~pos_mask
    m = int(pos_mask.sum())
    n = int(neg_mask.sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong requires both positives and negatives")

    # Per-classifier "structural components" (Sun-Xu 2014 fast form)
    def _components(scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        x_pos = scores[pos_mask]
        x_neg = scores[neg_mask]
        # Auc via mid-rank trick
        all_scores = np.concatenate([x_pos, x_neg])
        T_all = _compute_midrank(all_scores)
        T_pos = _compute_midrank(x_pos)
        T_neg = _compute_midrank(x_neg)
        # AUC = (sum(T_all_pos) - m*(m+1)/2) / (m*n)
        sum_T_pos_in_all = T_all[:m].sum()
        auc = (sum_T_pos_in_all - m * (m + 1) / 2.0) / (m * n)
        # Structural components V10 (per positive), V01 (per negative)
        V10 = (T_all[:m] - T_pos) / n        # length m
        V01 = 1.0 - (T_all[m:] - T_neg) / m  # length n
        return float(auc), V10, V01

    auc_a, V10_a, V01_a = _components(p_a)
    auc_b, V10_b, V01_b = _components(p_b)

    # Covariance matrix of (AUC_a, AUC_b) via Sun-Xu Eq. (5)
    S10 = np.cov(np.vstack([V10_a, V10_b]), bias=False)
    S01 = np.cov(np.vstack([V01_a, V01_b]), bias=False)
    cov = S10 / m + S01 / n   # 2x2 covariance of the AUC pair

    var_a = float(cov[0, 0])
    var_b = float(cov[1, 1])
    cov_ab = float(cov[0, 1])
    var_delta = var_a + var_b - 2.0 * cov_ab
    if var_delta <= 0:
        # Numerical fallback: degenerate (e.g. identical classifiers)
        return auc_a, auc_b, max(var_delta, 0.0), 1.0 if alternative == "greater" else 1.0

    delta = auc_b - auc_a
    z = delta / np.sqrt(var_delta)
    if alternative == "greater":
        p_value = float(1.0 - norm.cdf(z))
    elif alternative == "two-sided":
        p_value = float(2.0 * (1.0 - norm.cdf(abs(z))))
    else:
        raise ValueError(f"unknown alternative: {alternative!r}")
    return auc_a, auc_b, float(var_delta), p_value


# ─── Block-bootstrap of paired AUC delta ───────────────────────────────


def block_bootstrap_auc_delta_ci(
    y_true: pd.Series,
    p_baseline: pd.Series,
    p_treatment: pd.Series,
    *,
    n_reps: int = GATE_G2_BLOCK_BOOTSTRAP_REPS,
    block_mean_days: int = GATE_G2_BLOCK_DAYS,
    seed: int = GATE_G2_BLOCK_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float, np.ndarray]:
    """Stationary block bootstrap (Politis-Romano 1994), paired by DAY.

    Bootstrap unit = calendar day; each rep resamples days with
    geometric block length (mean = ``block_mean_days``). Same resampled
    day-set is applied to baseline and treatment so the delta is paired.

    AUC is computed on the union of trades belonging to the resampled
    days. Returns ``(delta_lcb, delta_median, delta_ucb, deltas_sample)``
    where lcb/ucb are the (alpha/2, 1-alpha/2) percentiles.

    Parameters
    ----------
    y_true : binary target, indexed by trade timestamp.
    p_baseline, p_treatment : aligned classifier scores, same index.
    n_reps : number of bootstrap reps.
    block_mean_days : geometric distribution parameter for block length
        (mean length in days).
    seed : RNG seed.
    alpha : two-sided alpha (0.05 → 95% CI).
    """
    from sklearn.metrics import roc_auc_score

    common = y_true.index.intersection(p_baseline.index).intersection(p_treatment.index)
    y_arr = y_true.loc[common].to_numpy(dtype=int)
    pb_arr = p_baseline.loc[common].to_numpy(dtype=np.float64)
    pt_arr = p_treatment.loc[common].to_numpy(dtype=np.float64)

    days = pd.DatetimeIndex(common).normalize()
    unique_days = pd.DatetimeIndex(sorted(set(days)))
    n_days = len(unique_days)
    if n_days < 30:
        raise ValueError(
            f"Need ≥ 30 unique days for block bootstrap; got {n_days}"
        )

    # Build day → trade-row indices
    day_to_rows: dict[pd.Timestamp, np.ndarray] = {}
    for i, d in enumerate(days):
        day_to_rows.setdefault(d, []).append(i)
    day_to_rows_arr = {d: np.asarray(rows, dtype=np.intp)
                       for d, rows in day_to_rows.items()}

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_reps, dtype=np.float64)
    p_block = 1.0 / float(block_mean_days)

    for r in range(n_reps):
        # Stationary bootstrap: build a sample of ~n_days days using
        # geometric block lengths.
        sampled_day_positions: list[int] = []
        while len(sampled_day_positions) < n_days:
            start = int(rng.integers(0, n_days))
            block_len = int(rng.geometric(p_block))
            for k in range(block_len):
                sampled_day_positions.append((start + k) % n_days)
                if len(sampled_day_positions) >= n_days:
                    break

        # Trade indices in the resample
        rows_collected: list[np.ndarray] = []
        for pos in sampled_day_positions:
            d = unique_days[pos]
            rows_collected.append(day_to_rows_arr[d])
        sample_rows = np.concatenate(rows_collected) if rows_collected else np.empty(0, dtype=np.intp)

        y_s = y_arr[sample_rows]
        # AUC requires both classes present
        if y_s.sum() == 0 or y_s.sum() == len(y_s):
            deltas[r] = 0.0
            continue
        pb_s = pb_arr[sample_rows]
        pt_s = pt_arr[sample_rows]
        auc_b = float(roc_auc_score(y_s, pb_s))
        auc_t = float(roc_auc_score(y_s, pt_s))
        deltas[r] = auc_t - auc_b

    lcb = float(np.percentile(deltas, 100 * (alpha / 2)))
    median = float(np.percentile(deltas, 50.0))
    ucb = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return lcb, median, ucb, deltas


# ─── Per-year + pooled AUC summary ─────────────────────────────────────


def per_year_auc_summary(
    y_true: pd.Series,
    p_scores: pd.Series,
    *,
    test_years: tuple[int, ...] = SHORTFALL_TEST_YEARS,
) -> dict[str, float]:
    """Compute per-year + pooled AUC for a single classifier.

    NaN scores (years skipped due to no train history) are excluded from
    pooling. Returns dict with keys ``year_<Y>`` and ``pooled``.
    """
    from sklearn.metrics import roc_auc_score

    common = y_true.index.intersection(p_scores.index)
    y = y_true.loc[common]
    p = p_scores.loc[common]

    out: dict[str, float] = {}
    pooled_y: list[int] = []
    pooled_p: list[float] = []
    for yr in test_years:
        yr_mask = y.index.year == yr
        y_yr = y.loc[yr_mask].dropna()
        p_yr = p.loc[y_yr.index].dropna()
        if len(p_yr) < 1:
            out[f"year_{yr}"] = float("nan")
            continue
        y_yr_aligned = y_yr.loc[p_yr.index].astype(int)
        if y_yr_aligned.sum() == 0 or y_yr_aligned.sum() == len(y_yr_aligned):
            out[f"year_{yr}"] = float("nan")
        else:
            out[f"year_{yr}"] = float(
                roc_auc_score(y_yr_aligned.values, p_yr.values)
            )
        pooled_y.extend(y_yr_aligned.tolist())
        pooled_p.extend(p_yr.tolist())

    if len(pooled_y) > 0 and 0 < sum(pooled_y) < len(pooled_y):
        out["pooled"] = float(roc_auc_score(np.asarray(pooled_y), np.asarray(pooled_p)))
    else:
        out["pooled"] = float("nan")
    return out


# ─── Composite gate evaluator ──────────────────────────────────────────


@dataclass(frozen=True)
class AucDeltaGateResult:
    """Result of the four-part Phase B gate evaluation."""

    # Per-classifier AUC
    auc_baseline_pooled: float
    auc_treatment_pooled: float
    auc_placebo_pooled: float

    # G1
    delta_pooled: float                 # AUC_treatment − AUC_baseline
    g1_pass: bool                       # delta ≥ +0.02
    g1_threshold: float

    # G2
    delta_lcb_95: float                 # 95% block-bootstrap LCB
    delta_ucb_95: float
    delta_median_bootstrap: float
    g2_pass: bool                       # LCB > 0
    g2_block_mean_days: int
    g2_n_reps: int

    # G3
    delong_p_one_sided: float
    delong_var_delta: float
    g3_pass: bool                       # p < 0.01
    g3_p_threshold: float

    # G4
    delta_placebo: float                # AUC_placebo − AUC_baseline
    g4_pass: bool                       # |Δ_placebo| < 0.005
    g4_threshold: float

    # Conjunction
    all_pass: bool

    def summary_lines(self) -> list[str]:
        """Plain-text bullet summary for the V4_SHORTFALL_HEAD.md report.

        ASCII-safe (Windows cp1252 stdout): no unicode characters.
        """
        return [
            f"AUC pooled - baseline = {self.auc_baseline_pooled:.4f}",
            f"AUC pooled - treatment = {self.auc_treatment_pooled:.4f}",
            f"AUC pooled - placebo = {self.auc_placebo_pooled:.4f}",
            "",
            f"G1 (delta >= +{self.g1_threshold:.3f}): "
            f"delta = {self.delta_pooled:+.4f}  "
            f"{'PASS' if self.g1_pass else 'FAIL'}",
            f"G2 (95% bootstrap LCB > 0): "
            f"LCB = {self.delta_lcb_95:+.4f}, "
            f"median = {self.delta_median_bootstrap:+.4f}, "
            f"UCB = {self.delta_ucb_95:+.4f}  "
            f"{'PASS' if self.g2_pass else 'FAIL'}",
            f"G3 (DeLong one-sided p < {self.g3_p_threshold:.3f}): "
            f"p = {self.delong_p_one_sided:.4g}  "
            f"{'PASS' if self.g3_pass else 'FAIL'}",
            f"G4 (|placebo delta| < {self.g4_threshold:.3f}): "
            f"placebo delta = {self.delta_placebo:+.4f}  "
            f"{'PASS' if self.g4_pass else 'FAIL'}",
            "",
            f"VERDICT: {'PASS - proceed to Phase C' if self.all_pass else 'FAIL - honest reject'}",
        ]


def auc_delta_gate(
    target: pd.Series,
    p_baseline: pd.Series,
    p_treatment: pd.Series,
    p_placebo: pd.Series,
    *,
    test_years: tuple[int, ...] = SHORTFALL_TEST_YEARS,
    block_bootstrap_n_reps: int = GATE_G2_BLOCK_BOOTSTRAP_REPS,
    block_mean_days: int = GATE_G2_BLOCK_DAYS,
    seed: int = GATE_G2_BLOCK_BOOTSTRAP_SEED,
) -> AucDeltaGateResult:
    """Evaluate the four pre-committed Phase B sub-gates.

    All thresholds are LOCKED at module-level constants. Re-running with
    different thresholds would invalidate the pre-commitment.

    Parameters
    ----------
    target : binary y_short Series, indexed by fired-trade timestamps.
    p_baseline / p_treatment / p_placebo : classifier scores aligned to
        ``target``. NaN where year was skipped.

    Returns
    -------
    AucDeltaGateResult — frozen dataclass with all gate diagnostics.
    """
    from sklearn.metrics import roc_auc_score

    # Restrict to test-year fired trades with non-NaN scores in ALL three
    # classifier outputs (paired AUC requires same support).
    common_idx = (
        target.index
        .intersection(p_baseline.index)
        .intersection(p_treatment.index)
        .intersection(p_placebo.index)
    )
    test_years_set = set(test_years)
    in_test_years = pd.Series(
        common_idx.year.isin(test_years_set), index=common_idx,
    )
    common_idx = common_idx[in_test_years.to_numpy()]

    y = target.loc[common_idx]
    pb = p_baseline.loc[common_idx]
    pt = p_treatment.loc[common_idx]
    pp = p_placebo.loc[common_idx]

    valid_mask = ~(pb.isna() | pt.isna() | pp.isna())
    y = y.loc[valid_mask]
    pb = pb.loc[valid_mask]
    pt = pt.loc[valid_mask]
    pp = pp.loc[valid_mask]

    if len(y) < 100:
        raise ValueError(
            f"Pooled-OOS sample too small after alignment: {len(y)} trades; need ≥ 100"
        )

    # Pooled AUC per classifier
    auc_b = float(roc_auc_score(y.values, pb.values))
    auc_t = float(roc_auc_score(y.values, pt.values))
    auc_p = float(roc_auc_score(y.values, pp.values))
    delta_pooled = auc_t - auc_b

    # G1
    g1_pass = bool(delta_pooled >= GATE_G1_DELTA_MIN)

    # G2 — block bootstrap
    lcb, median_boot, ucb, _deltas = block_bootstrap_auc_delta_ci(
        y, pb, pt,
        n_reps=block_bootstrap_n_reps,
        block_mean_days=block_mean_days,
        seed=seed,
    )
    g2_pass = bool(lcb > 0.0)

    # G3 — DeLong
    _, _, var_delta, p_delong = delong_paired_p_value(
        y.values, pb.values, pt.values, alternative="greater",
    )
    g3_pass = bool(p_delong < GATE_G3_DELONG_P_MAX)

    # G4 — placebo
    delta_placebo = auc_p - auc_b
    g4_pass = bool(abs(delta_placebo) < GATE_G4_PLACEBO_MAX)

    all_pass = bool(g1_pass and g2_pass and g3_pass and g4_pass)

    return AucDeltaGateResult(
        auc_baseline_pooled=auc_b,
        auc_treatment_pooled=auc_t,
        auc_placebo_pooled=auc_p,
        delta_pooled=delta_pooled,
        g1_pass=g1_pass,
        g1_threshold=GATE_G1_DELTA_MIN,
        delta_lcb_95=lcb,
        delta_ucb_95=ucb,
        delta_median_bootstrap=median_boot,
        g2_pass=g2_pass,
        g2_block_mean_days=block_mean_days,
        g2_n_reps=block_bootstrap_n_reps,
        delong_p_one_sided=p_delong,
        delong_var_delta=var_delta,
        g3_pass=g3_pass,
        g3_p_threshold=GATE_G3_DELONG_P_MAX,
        delta_placebo=delta_placebo,
        g4_pass=g4_pass,
        g4_threshold=GATE_G4_PLACEBO_MAX,
        all_pass=all_pass,
    )
