"""Evaluation metrics and proper baselines for volatility prediction.

Primary metric: QLIKE (proper volatility loss).
Key baseline: HAR model (Corsi 2009) — the gold standard for RV forecasting.
Statistical tests: Diebold-Mariano, Mincer-Zarnowitz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from vol_model.model import qlike_loss


# ── HAR model (Heterogeneous Autoregressive) ────────────────────

def har_horizons(freq_s: int = 900) -> list[int]:
    """Compute HAR horizons (in bars) from timeframe.

    Returns deduplicated [1 bar, 1 hour, 1 day, 1 week] in bar counts.
    For daily+ timeframes, 1h collapses to 1 bar — deduplicated automatically.
    """
    bars_per_hour = max(3600 // freq_s, 1)
    raw = [1, bars_per_hour, bars_per_hour * 24, bars_per_hour * 168]
    # Deduplicate while preserving order
    seen = set()
    return [h for h in raw if h not in seen and not seen.add(h)]


def fit_har(y: pd.Series, horizons: list[int] | None = None, freq_s: int = 900) -> dict:
    """Fit HAR model: ln(vol_{t+1}) = α + Σ β_h · mean(y_{t-h+1:t}) + ε

    Horizons computed dynamically from freq_s: [1 bar, 1h, 1d, 1w].
    Uses OLS. Returns out-of-sample predictions via expanding window.

    This is THE benchmark for realized vol forecasting (Corsi 2009).
    If LightGBM can't beat this 4-variable OLS, microstructure features are noise.
    """
    if horizons is None:
        horizons = har_horizons(freq_s)
    max_h = max(horizons)
    n = len(y)
    y_arr = y.values

    # Build HAR features (rolling means at each horizon)
    X_har = np.column_stack([
        pd.Series(y_arr).rolling(h, min_periods=1).mean().values
        for h in horizons
    ])

    # One-step-ahead: features at t predict y at t+1
    X = X_har[max_h:-1]
    y_target = y_arr[max_h + 1:]

    if len(X) < 20:
        return {"predictions": np.full(n, np.nan), "coefficients": None}

    # Expanding window OOS predictions (no leakage)
    min_train = max(50, len(X) // 5)
    predictions = np.full(n, np.nan)

    for t in range(min_train, len(X)):
        X_train = X[:t]
        y_train = y_target[:t]

        # OLS with intercept
        X_aug = np.column_stack([np.ones(t), X_train])
        try:
            beta = np.linalg.lstsq(X_aug, y_train, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        x_test = np.concatenate([[1.0], X[t]])
        pred = x_test @ beta
        # Map back to original index
        orig_idx = max_h + 1 + t
        if orig_idx < n:
            predictions[orig_idx] = pred

    # Final coefficients (full sample for reporting)
    X_aug_full = np.column_stack([np.ones(len(X)), X])
    beta_full = np.linalg.lstsq(X_aug_full, y_target, rcond=None)[0]

    return {
        "predictions": predictions,
        "coefficients": {
            "intercept": beta_full[0],
            **{f"RV_{h}bar": beta_full[i + 1] for i, h in enumerate(horizons)},
        },
    }


# ── Statistical tests ───────────────────────────────────────────

def diebold_mariano_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict:
    """Diebold-Mariano test: are forecast A's losses significantly different from B's?

    Uses QLIKE loss differences. H0: equal predictive accuracy.
    Negative t-stat = A is better. p < 0.05 = significant.
    """
    diff_a = y_true - pred_a
    diff_b = y_true - pred_b
    loss_a = np.exp(np.clip(2 * diff_a, -20, 20)) - 2 * diff_a - 1
    loss_b = np.exp(np.clip(2 * diff_b, -20, 20)) - 2 * diff_b - 1

    d = loss_a - loss_b  # negative means A is better
    n = len(d)

    d_mean = d.mean()
    # Newey-West HAC variance (lag = floor(n^{1/3}))
    lag = int(np.floor(n ** (1.0 / 3)))
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for j in range(1, lag + 1):
        gamma_j = np.cov(d[j:], d[:-j])[0, 1]
        gamma_sum += 2 * (1 - j / (lag + 1)) * gamma_j

    variance = (gamma_0 + gamma_sum) / n
    if variance <= 0:
        return {"t_stat": 0.0, "p_value": 1.0, "significant": False}

    t_stat = d_mean / np.sqrt(variance)
    p_value = 2 * stats.norm.sf(abs(t_stat))

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "a_better": d_mean < 0,
    }


def mincer_zarnowitz(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Mincer-Zarnowitz regression: y_true = α + β·y_pred + ε.

    Unbiased forecast: α ≈ 0, β ≈ 1.
    R² = forecast efficiency (how much variance is explained).
    """
    n = len(y_true)
    X = np.column_stack([np.ones(n), y_pred])
    beta = np.linalg.lstsq(X, y_true, rcond=None)[0]
    y_hat = X @ beta
    ss_res = np.sum((y_true - y_hat) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "alpha": float(beta[0]),
        "beta": float(beta[1]),
        "r_squared": float(r_squared),
        "alpha_near_zero": abs(beta[0]) < 0.1,
        "beta_near_one": abs(beta[1] - 1.0) < 0.2,
        "well_calibrated": abs(beta[0]) < 0.1 and abs(beta[1] - 1.0) < 0.2,
    }


# ── Baselines ───────────────────────────────────────────────────

def compute_baselines(
    y: pd.Series,
    tsrv_path: str | None = None,
    freq_s: int = 900,
) -> dict[str, np.ndarray]:
    """Compute baseline predictions including HAR model."""
    y_arr = y.values

    baselines = {
        "last_value": np.concatenate([[y_arr[0]], y_arr[:-1]]),
        "rolling_mean_10": y.rolling(10, min_periods=1).mean().values,
        "ema_50": y.ewm(span=50, min_periods=1).mean().values,
    }

    # HAR model (the real benchmark)
    har = fit_har(y, freq_s=freq_s)
    baselines["HAR"] = har["predictions"]

    # AR(1): simple autoregressive
    # y_{t+1} = α + β·y_t (expanding window OLS)
    ar1_pred = np.full(len(y_arr), np.nan)
    ar1_min = max(50, len(y_arr) // 5)
    for t in range(ar1_min, len(y_arr) - 1):
        X = y_arr[:t].reshape(-1, 1)
        X_aug = np.column_stack([np.ones(t), X])
        yt = y_arr[1:t + 1]
        try:
            beta = np.linalg.lstsq(X_aug, yt, rcond=None)[0]
            ar1_pred[t + 1] = beta[0] + beta[1] * y_arr[t]
        except Exception:
            continue
    baselines["AR1"] = ar1_pred

    # Naive RV (noisy, no TSRV correction)
    if tsrv_path:
        try:
            tsrv = pd.read_parquet(tsrv_path)
            rv_fast = tsrv["rv_fast"].reindex(y.index)
            periods_per_year = 365.25 * 24 * 3600 / freq_s
            rv_ann = rv_fast * periods_per_year
            naive_vol = np.sqrt(np.maximum(rv_ann, 0))
            naive_ln_vol = np.log(np.maximum(naive_vol, 1e-10))
            baselines["naive_rv_fast"] = np.concatenate(
                [[naive_ln_vol.iloc[0]], naive_ln_vol.values[:-1]])
        except Exception:
            pass

    return baselines


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baselines: dict[str, np.ndarray] | None = None,
) -> dict:
    """Full evaluation: metrics, baselines, statistical tests, calibration."""
    metrics = {}

    # Core metrics
    metrics["qlike"] = qlike_loss(y_true, y_pred)
    metrics["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))

    # RankIC (Spearman correlation)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    metrics["rank_ic"] = float(spearman_r)
    metrics["rank_ic_pval"] = float(spearman_p)

    # Pearson correlation
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    metrics["pearson_r"] = float(pearson_r)

    # Directional accuracy
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    valid_dir = (true_dir != 0) & (pred_dir != 0)
    metrics["directional_accuracy"] = float(
        (true_dir[valid_dir] == pred_dir[valid_dir]).mean()) if valid_dir.sum() > 0 else 0.5

    # Mincer-Zarnowitz regression (calibration)
    mz = mincer_zarnowitz(y_true, y_pred)
    metrics["mz_alpha"] = mz["alpha"]
    metrics["mz_beta"] = mz["beta"]
    metrics["mz_r_squared"] = mz["r_squared"]
    metrics["mz_well_calibrated"] = mz["well_calibrated"]

    # Baseline comparisons + Diebold-Mariano tests
    if baselines:
        for name, bl_pred in baselines.items():
            # Handle NaN in baseline predictions (HAR, AR1 have warmup NaN)
            valid_mask = ~np.isnan(bl_pred) & ~np.isnan(y_pred)
            if valid_mask.sum() < 30:
                continue

            bl_qlike = qlike_loss(y_true[valid_mask], bl_pred[valid_mask])
            model_qlike = qlike_loss(y_true[valid_mask], y_pred[valid_mask])
            improvement = 1.0 - model_qlike / bl_qlike if bl_qlike > 0 else 0.0
            metrics[f"baseline_{name}_qlike"] = bl_qlike
            metrics[f"vs_{name}_improvement"] = improvement

            # Diebold-Mariano test: is the improvement significant?
            dm = diebold_mariano_test(y_true[valid_mask], y_pred[valid_mask], bl_pred[valid_mask])
            metrics[f"dm_{name}_tstat"] = dm["t_stat"]
            metrics[f"dm_{name}_pval"] = dm["p_value"]
            metrics[f"dm_{name}_significant"] = dm["significant"]

    return metrics


def print_evaluation(metrics: dict, fold_metrics: list[dict] | None = None) -> None:
    """Print full evaluation results."""
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    print(f"\n  QLIKE:      {metrics['qlike']:.4f}")
    print(f"  RMSE:       {metrics['rmse']:.4f}")
    print(f"  MAE:        {metrics['mae']:.4f}")
    print(f"  RankIC:     {metrics['rank_ic']:.4f} (p={metrics['rank_ic_pval']:.2e})")
    print(f"  Pearson r:  {metrics['pearson_r']:.4f}")
    print(f"  Dir. Acc:   {metrics['directional_accuracy']:.1%}")

    # Mincer-Zarnowitz calibration
    print(f"\n  Mincer-Zarnowitz (y_true = a + b*y_pred):")
    print(f"    a = {metrics['mz_alpha']:.4f} (ideal: 0)")
    print(f"    b = {metrics['mz_beta']:.4f} (ideal: 1)")
    print(f"    R² = {metrics['mz_r_squared']:.4f}")
    cal = "YES" if metrics["mz_well_calibrated"] else "NO"
    print(f"    Well-calibrated: {cal}")

    # Baselines + Diebold-Mariano
    baseline_keys = sorted([k for k in metrics if k.startswith("baseline_") and k.endswith("_qlike")])
    if baseline_keys:
        print(f"\n  {'Baseline':>20s}  {'QLIKE':>8s}  {'vs Model':>10s}  {'DM t-stat':>10s}  {'p-val':>8s}  Sig?")
        print("  " + "-" * 72)
        for key in baseline_keys:
            name = key.replace("baseline_", "").replace("_qlike", "")
            bl_qlike = metrics[key]
            improvement = metrics.get(f"vs_{name}_improvement", 0)
            dm_t = metrics.get(f"dm_{name}_tstat", float("nan"))
            dm_p = metrics.get(f"dm_{name}_pval", float("nan"))
            dm_sig = metrics.get(f"dm_{name}_significant", False)
            beat = "BEAT" if improvement > 0 else "LOST"
            sig = "***" if dm_sig and improvement > 0 else ("n.s." if not dm_sig else "---")
            print(f"  {name:>20s}  {bl_qlike:>8.4f}  {beat} {abs(improvement):>5.1%}  {dm_t:>10.2f}  {dm_p:>8.4f}  {sig}")

    # Fold details
    if fold_metrics:
        print("\n  Per-fold QLIKE:")
        for fm in fold_metrics:
            print(f"    Fold {fm['fold']}: {fm['qlike']:.4f}  (n_test={fm['n_test']})")
        qlike_values = [fm["qlike"] for fm in fold_metrics]
        print(f"    Mean: {np.mean(qlike_values):.4f} +/- {np.std(qlike_values):.4f}")
