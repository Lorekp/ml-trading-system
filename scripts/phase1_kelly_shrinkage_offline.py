"""Phase 1 offline evaluation of long-side Kelly-shrinkage sizing boost.

Plan: C:\\Users\\laure\\.claude\\plans\\starry-rolling-rabin.md
Project memory reference: project_kelly_sizing_2026_04_21.md (to be written).

Purpose
-------
Evaluate three pre-committed asymmetric-long Kelly-shrinkage sizing rules
(K1/K2/K3) on the existing V2 trade log, WITHOUT running the backtester.
Apply walk-forward multipliers to ``net_pnl_nc`` and ``net_pnl`` and
measure bootstrap/permutation/jackknife significance of PnL gain vs
baseline, plus a pre-aligner-log cross-check.

The offline eval is leakage-safe by construction:
- sigma^2_M uses only residuals in [M-3, M) (strictly trailing).
- trailing-12m median of f_shrunk uses only values from [M-12, M).
- edge-quintile boundaries for K2/K3 come from trailing-12m longs only.
- EVERY walk-forward scalar is recomputed per target month, never
  peeking across months.

Decision gate (pre-committed in the plan):
A variant passes Phase 1 iff ALL four hold:
  (a) bootstrap lower 95% CI on delta-NC-PnL > $0
  (b) permutation-null p < 0.05
  (c) monthly jackknife: >= 60% of month-drops still positive
  (d) baseline-log cross-check: delta-NC-PnL same sign
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
N_BOOTSTRAP = 1000
N_PERMUTATION = 1000

TRADE_LOG_PRIMARY = "vol_arb/output/BTC_v2_trade_log.parquet"
TRADE_LOG_BASELINE = "vol_arb/output/BTC_v2_trade_log_baseline.parquet"
PREDS_ALIGNED = "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
PREDS_RAW = "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet"

REPORT_PATH = "vol_arb/output/KELLY_SHRINKAGE_PHASE1_REPORT.md"

# Kelly-shrinkage constants
KELLY_FLOOR = 1.0   # never scale DOWN (upside-only asymmetric)
KELLY_CAP = 2.0     # hard risk cap on any single boost
CAL_MONTHS = 3      # trailing residual-variance window
TRAIL_MONTHS_MEDIAN = 12  # trailing median-of-f_shrunk window
TRAIL_MONTHS_QUINTILE = 12  # trailing quintile-boundary window


# ---------- Walk-forward primitives ----------


def month_key(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


def compute_sigma2_trailing(
    preds_df: pd.DataFrame, cal_months: int = CAL_MONTHS
) -> pd.Series:
    """For each month M in preds_df.index, variance of residuals in
    [M-cal_months, M). Residuals use the 'y_pred' column (aligned or
    raw, matching the trade log). Returns Series indexed by 'YYYY-MM'.
    """
    residuals = preds_df["y_true"] - preds_df["y_pred"]
    residuals = residuals.dropna()
    # Group residuals by month for fast windowed aggregation.
    months = sorted({month_key(ts) for ts in residuals.index})
    out = {}
    for m in months:
        start = pd.Timestamp(f"{m}-01", tz="UTC")
        cal_start = start - pd.DateOffset(months=cal_months)
        cal_end = start  # [cal_start, start)
        mask = (residuals.index >= cal_start) & (residuals.index < cal_end)
        slice_ = residuals.loc[mask]
        if len(slice_) < 500:  # insufficient; skip
            out[m] = np.nan
        else:
            out[m] = float(slice_.var(ddof=1))
    return pd.Series(out, name="sigma2")


def compute_ntrail_longs(
    tl: pd.DataFrame, trail_months: int = TRAIL_MONTHS_MEDIAN
) -> pd.Series:
    """Per-month trailing count of LONG firing trades in [M-trail_months, M)."""
    long_ts = tl.index[tl["signal"] == 1]
    months = sorted({month_key(ts) for ts in tl.index})
    out = {}
    for m in months:
        start = pd.Timestamp(f"{m}-01", tz="UTC")
        tstart = start - pd.DateOffset(months=trail_months)
        tend = start
        mask = (long_ts >= tstart) & (long_ts < tend)
        out[m] = int(mask.sum())
    return pd.Series(out, name="n_trail_longs")


def compute_kelly_raw_and_shrunk(
    tl: pd.DataFrame,
    sigma2_by_month: pd.Series,
    n_trail_by_month: pd.Series,
    d: int = 1,
) -> pd.DataFrame:
    """For each bar: add columns f_star, shrink, f_shrunk. f_star = NaN
    for NaN sigma2 / non-long bars (we only boost longs).
    Rising-Wyner shrinkage: shrink = 1 - (d+2)/(n_trail * lambda_max)
    where lambda_max = sigma2 (1-D case).
    """
    out = tl.copy()
    month_idx = pd.Index([month_key(ts) for ts in out.index])
    sigma2 = month_idx.map(sigma2_by_month.to_dict()).to_numpy(dtype=float)
    n_trail = month_idx.map(n_trail_by_month.to_dict()).to_numpy(dtype=float)

    abs_edge = (out["y_pred"] - out["iv_used"]).abs().to_numpy()
    signal = out["signal"].to_numpy()

    # f_star = |edge| / sigma2 — only meaningful for longs with valid sigma2
    with np.errstate(divide="ignore", invalid="ignore"):
        f_star = np.where(
            (signal == 1) & (sigma2 > 0) & np.isfinite(sigma2),
            abs_edge / sigma2, np.nan
        )
        # Rising-Wyner shrinkage
        shrink = 1.0 - (d + 2) / (n_trail * sigma2)
        shrink = np.where(np.isfinite(shrink) & (shrink > 0), shrink, 0.0)
    f_shrunk = np.where(np.isnan(f_star), np.nan, np.maximum(0.0, f_star * shrink))

    out["sigma2"] = sigma2
    out["n_trail_longs"] = n_trail
    out["f_star"] = f_star
    out["shrink"] = shrink
    out["f_shrunk"] = f_shrunk
    return out


def compute_trailing_long_median(
    tl_with_kelly: pd.DataFrame, trail_months: int = TRAIL_MONTHS_MEDIAN
) -> pd.Series:
    """For each month M: median of f_shrunk over LONG firing trades in
    [M-trail_months, M) (non-NaN only). Returns Series indexed 'YYYY-MM'.
    """
    longs = tl_with_kelly.loc[
        (tl_with_kelly["signal"] == 1) & tl_with_kelly["f_shrunk"].notna()
    ]
    months = sorted({month_key(ts) for ts in tl_with_kelly.index})
    out = {}
    for m in months:
        start = pd.Timestamp(f"{m}-01", tz="UTC")
        tstart = start - pd.DateOffset(months=trail_months)
        tend = start
        mask = (longs.index >= tstart) & (longs.index < tend)
        vals = longs.loc[mask, "f_shrunk"].to_numpy()
        if len(vals) < 50:
            out[m] = np.nan
        else:
            out[m] = float(np.median(vals))
    return pd.Series(out, name="median_fshrunk_trail")


def compute_trailing_long_quintile_bounds(
    tl: pd.DataFrame, trail_months: int = TRAIL_MONTHS_QUINTILE
) -> pd.DataFrame:
    """For each month M: compute Q4 and Q5 edge-magnitude boundaries
    from the trailing-12m LONG |pred_edge| distribution. Walk-forward.
    Returns DataFrame(index='YYYY-MM', cols=['q80', 'q60'])
    where a trade in the target month is Q5 if its |edge| >= q80,
    Q4 if q60 <= |edge| < q80, etc.
    """
    abs_edge_all = (tl["y_pred"] - tl["iv_used"]).abs()
    long_mask_all = tl["signal"] == 1
    months = sorted({month_key(ts) for ts in tl.index})
    out = {"q60": {}, "q80": {}}
    for m in months:
        start = pd.Timestamp(f"{m}-01", tz="UTC")
        tstart = start - pd.DateOffset(months=trail_months)
        tend = start
        mask = (tl.index >= tstart) & (tl.index < tend) & long_mask_all
        edges = abs_edge_all.loc[mask].dropna().to_numpy()
        if len(edges) < 100:
            out["q60"][m] = np.nan
            out["q80"][m] = np.nan
        else:
            out["q60"][m] = float(np.quantile(edges, 0.60))
            out["q80"][m] = float(np.quantile(edges, 0.80))
    return pd.DataFrame(out)


# ---------- Sizing rules ----------


def size_mult_K1(tl_with_kelly: pd.DataFrame, median_by_month: pd.Series) -> np.ndarray:
    """K1: Kelly-shrunk continuous. mult = clip(f_shrunk / median_trail, 1.0, 2.0)
    for longs with valid f_shrunk and median; else 1.0.
    """
    mult = np.ones(len(tl_with_kelly), dtype=float)
    month_idx = pd.Index([month_key(ts) for ts in tl_with_kelly.index])
    med = month_idx.map(median_by_month.to_dict()).to_numpy(dtype=float)
    signal = tl_with_kelly["signal"].to_numpy()
    f_shrunk = tl_with_kelly["f_shrunk"].to_numpy()
    mask = (signal == 1) & np.isfinite(f_shrunk) & np.isfinite(med) & (med > 0)
    raw = np.where(mask, f_shrunk / med, np.nan)
    mult = np.where(mask, np.clip(raw, KELLY_FLOOR, KELLY_CAP), mult)
    return mult


def size_mult_K2(
    tl_with_kelly: pd.DataFrame,
    median_by_month: pd.Series,
    qbounds: pd.DataFrame,
) -> np.ndarray:
    """K2: K1 rule but applied only to LONG Q4+Q5 (|edge| >= q60). Else 1.0."""
    mult = size_mult_K1(tl_with_kelly, median_by_month)
    month_idx = pd.Index([month_key(ts) for ts in tl_with_kelly.index])
    q60 = month_idx.map(qbounds["q60"].to_dict()).to_numpy(dtype=float)
    abs_edge = (tl_with_kelly["y_pred"] - tl_with_kelly["iv_used"]).abs().to_numpy()
    signal = tl_with_kelly["signal"].to_numpy()
    in_q4q5 = (signal == 1) & np.isfinite(q60) & (abs_edge >= q60)
    # outside Q4+Q5: force to 1.0
    return np.where(in_q4q5, mult, 1.0)


def size_mult_K3(
    tl_with_kelly: pd.DataFrame, qbounds: pd.DataFrame,
) -> np.ndarray:
    """K3: fixed 1.5x on LONG Q5 (|edge| >= q80); else 1.0. Kelly-free ablation."""
    month_idx = pd.Index([month_key(ts) for ts in tl_with_kelly.index])
    q80 = month_idx.map(qbounds["q80"].to_dict()).to_numpy(dtype=float)
    abs_edge = (tl_with_kelly["y_pred"] - tl_with_kelly["iv_used"]).abs().to_numpy()
    signal = tl_with_kelly["signal"].to_numpy()
    in_q5 = (signal == 1) & np.isfinite(q80) & (abs_edge >= q80)
    return np.where(in_q5, 1.5, 1.0)


# ---------- Metrics ----------


def compute_metrics(
    tl: pd.DataFrame, size_mult: np.ndarray, capital: float = 100_000.0
) -> dict:
    """Scale PnL columns by size_mult; compute NC + compound metrics.

    Compound PnL after scaling is an APPROXIMATION (true compounding
    path depends on order of trades; bar-wise scaling is close but not
    exact for margin-gated sizing). NC is exact and is the primary metric.
    """
    t = tl.loc[tl["signal"] != 0].copy()
    mult_trades = size_mult[tl["signal"] != 0]
    t["net_pnl_nc_scaled"] = t["net_pnl_nc"] * mult_trades
    t["net_pnl_scaled"] = t["net_pnl"] * mult_trades

    # NC metrics
    nc_daily = t["net_pnl_nc_scaled"].resample("1D").sum()
    nc_daily_active = nc_daily.loc[nc_daily != 0]
    nc_eq = nc_daily.cumsum()
    nc_dd = float((nc_eq - nc_eq.cummax()).min())
    nc_sharpe = (
        float(nc_daily_active.mean() / nc_daily_active.std() * np.sqrt(365))
        if nc_daily_active.std() > 0 else float("nan")
    )
    # Compound (approximate with scaled net_pnl)
    cp_daily = t["net_pnl_scaled"].resample("1D").sum()
    cp_eq = cp_daily.cumsum() + capital
    cp_dd_series = (cp_eq - cp_eq.cummax()) / cp_eq.cummax()
    cp_dd = float(cp_dd_series.min())
    cp_daily_active = cp_daily.loc[cp_daily != 0]
    cp_sharpe = (
        float(cp_daily_active.mean() / cp_daily_active.std() * np.sqrt(365))
        if cp_daily_active.std() > 0 else float("nan")
    )

    return dict(
        nc_pnl=float(t["net_pnl_nc_scaled"].sum()),
        nc_sharpe=nc_sharpe,
        nc_mdd=nc_dd,
        nc_worst_day=float(nc_daily.min()),
        cp_pnl=float(t["net_pnl_scaled"].sum()),
        cp_sharpe=cp_sharpe,
        cp_mdd_pct=float(cp_dd * 100),
        n_trades_boosted=int((mult_trades > 1.0 + 1e-9).sum()),
        n_trades_total=int(len(t)),
        mean_mult=float(mult_trades.mean()),
        mean_mult_longs=float(mult_trades[t["signal"] == 1].mean()),
        long_pnl_nc=float(t.loc[t["signal"] == 1, "net_pnl_nc_scaled"].sum()),
        short_pnl_nc=float(t.loc[t["signal"] == -1, "net_pnl_nc_scaled"].sum()),
    )


# ---------- Statistical tests ----------


def block_bootstrap_delta_nc_pnl(
    tl: pd.DataFrame, mult_base: np.ndarray, mult_new: np.ndarray,
    n_boot: int = N_BOOTSTRAP, seed: int = SEED,
) -> tuple[float, float, float]:
    """Block bootstrap CI on (NC PnL new - NC PnL base), resampling
    trading DAYS (block size = 1 day) to preserve intraday autocorrelation.
    Returns (lower_95, mean, upper_95).
    """
    t = tl.loc[tl["signal"] != 0]
    mult_base_t = mult_base[tl["signal"] != 0]
    mult_new_t = mult_new[tl["signal"] != 0]
    pnl_base_per_bar = t["net_pnl_nc"].to_numpy() * mult_base_t
    pnl_new_per_bar = t["net_pnl_nc"].to_numpy() * mult_new_t
    # Assign each bar to a day
    day_key = t.index.floor("1D")
    unique_days = day_key.unique()
    day_to_idx = {d: np.where(day_key == d)[0] for d in unique_days}

    rng = np.random.default_rng(seed)
    deltas = np.zeros(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(len(unique_days), size=len(unique_days), replace=True)
        idxs = np.concatenate([day_to_idx[unique_days[d]] for d in sampled])
        deltas[b] = float(
            pnl_new_per_bar[idxs].sum() - pnl_base_per_bar[idxs].sum()
        )
    return (
        float(np.quantile(deltas, 0.025)),
        float(deltas.mean()),
        float(np.quantile(deltas, 0.975)),
    )


def permutation_test_edge_quintiles(
    tl_with_kelly: pd.DataFrame,
    median_by_month: pd.Series,
    qbounds: pd.DataFrame,
    variant: str,
    n_perm: int = N_PERMUTATION, seed: int = SEED,
) -> tuple[float, float]:
    """Null: shuffle the |edge| values among LONG firing bars (keeping
    month assignment intact). Re-run variant's rule on shuffled log.
    Report p = fraction of null-delta >= observed-delta.

    Shuffling |edge| kills ONLY the |edge| -> $/tr signal; month
    counts, bar times, PnL per bar all preserved.

    Returns (observed_delta_nc_pnl, p_value).
    """
    # Observed delta
    tl_obs = tl_with_kelly.copy()
    if variant == "K1":
        mult_obs = size_mult_K1(tl_obs, median_by_month)
    elif variant == "K2":
        mult_obs = size_mult_K2(tl_obs, median_by_month, qbounds)
    elif variant == "K3":
        mult_obs = size_mult_K3(tl_obs, qbounds)
    else:
        raise ValueError(variant)
    base_ones = np.ones(len(tl_obs))
    obs_delta = float(
        ((mult_obs - base_ones) * tl_obs["net_pnl_nc"].to_numpy()).sum()
    )

    # Null: shuffle |edge| among LONG bars.
    rng = np.random.default_rng(seed)
    long_mask = (tl_with_kelly["signal"] == 1).to_numpy()
    long_idx = np.where(long_mask)[0]
    orig_edge = (tl_with_kelly["y_pred"] - tl_with_kelly["iv_used"]).abs().to_numpy()
    orig_ypred = tl_with_kelly["y_pred"].to_numpy()
    orig_iv = tl_with_kelly["iv_used"].to_numpy()
    orig_fshrunk = tl_with_kelly["f_shrunk"].to_numpy()

    null_deltas = np.zeros(n_perm)
    # Perf shortcut: K2/K3 depend on (|edge| vs quintile) AND f_shrunk.
    # For K3 (|edge| only) shuffle |edge|. For K1/K2 we need f_shrunk too.
    # f_shrunk depends on abs_edge linearly (f_shrunk = shrink * abs_edge / sigma2),
    # so shuffling abs_edge on longs induces a corresponding shuffle of f_shrunk.
    shrink_arr = tl_with_kelly["shrink"].to_numpy()
    sigma2_arr = tl_with_kelly["sigma2"].to_numpy()

    for b in range(n_perm):
        perm = rng.permutation(long_idx)
        abs_edge_perm = orig_edge.copy()
        abs_edge_perm[long_idx] = orig_edge[perm]
        # rebuild f_shrunk on longs from shuffled abs_edge
        fsh_perm = orig_fshrunk.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            new_fstar = np.where(
                (sigma2_arr > 0) & np.isfinite(sigma2_arr),
                abs_edge_perm / sigma2_arr, np.nan
            )
            new_fsh = np.where(
                np.isnan(new_fstar), np.nan,
                np.maximum(0.0, new_fstar * shrink_arr)
            )
        fsh_perm[long_idx] = new_fsh[long_idx]

        tl_perm = tl_with_kelly.copy()
        # We need to write back the permuted |edge| and f_shrunk.
        # Overwrite via y_pred (keep iv_used, recompute y_pred such that
        # |y_pred - iv_used| = abs_edge_perm but sign same as original).
        sign_edge = np.sign(orig_ypred - orig_iv)
        y_pred_perm = orig_ypred.copy()
        y_pred_perm[long_idx] = (
            orig_iv[long_idx] + sign_edge[long_idx] * abs_edge_perm[long_idx]
        )
        tl_perm["y_pred"] = y_pred_perm
        tl_perm["f_shrunk"] = fsh_perm

        if variant == "K1":
            mult_perm = size_mult_K1(tl_perm, median_by_month)
        elif variant == "K2":
            mult_perm = size_mult_K2(tl_perm, median_by_month, qbounds)
        elif variant == "K3":
            mult_perm = size_mult_K3(tl_perm, qbounds)
        null_deltas[b] = float(
            ((mult_perm - base_ones) * tl_with_kelly["net_pnl_nc"].to_numpy()).sum()
        )

    p = float((null_deltas >= obs_delta).mean())
    return obs_delta, p


def monthly_jackknife_delta(
    tl: pd.DataFrame, mult_base: np.ndarray, mult_new: np.ndarray,
) -> tuple[int, int, list[tuple[str, float]]]:
    """Drop each month M in turn, compute delta-NC-PnL on the rest.
    Returns (n_positive, n_total, per_month_deltas).
    """
    t = tl.loc[tl["signal"] != 0]
    mult_base_t = mult_base[tl["signal"] != 0]
    mult_new_t = mult_new[tl["signal"] != 0]
    pnl_delta_per_bar = (mult_new_t - mult_base_t) * t["net_pnl_nc"].to_numpy()
    month_tags = pd.Index([month_key(ts) for ts in t.index])

    months = sorted(set(month_tags))
    per_month = []
    for m in months:
        keep = np.asarray(month_tags != m)
        per_month.append((m, float(pnl_delta_per_bar[keep].sum())))
    n_pos = sum(1 for _, d in per_month if d > 0)
    return n_pos, len(per_month), per_month


# ---------- Orchestration ----------


def run_full_variant(
    tl: pd.DataFrame, preds: pd.DataFrame, variant: str, label: str,
) -> dict:
    """End-to-end walk-forward evaluation of one variant on one trade log."""
    sigma2 = compute_sigma2_trailing(preds, cal_months=CAL_MONTHS)
    n_trail = compute_ntrail_longs(tl, trail_months=TRAIL_MONTHS_MEDIAN)
    tl_k = compute_kelly_raw_and_shrunk(tl, sigma2, n_trail)
    median_fsh = compute_trailing_long_median(tl_k, TRAIL_MONTHS_MEDIAN)
    qbounds = compute_trailing_long_quintile_bounds(tl_k, TRAIL_MONTHS_QUINTILE)

    if variant == "K1":
        mult = size_mult_K1(tl_k, median_fsh)
    elif variant == "K2":
        mult = size_mult_K2(tl_k, median_fsh, qbounds)
    elif variant == "K3":
        mult = size_mult_K3(tl_k, qbounds)
    else:
        raise ValueError(variant)
    mult_base = np.ones(len(tl_k))

    metrics_new = compute_metrics(tl_k, mult)
    metrics_base = compute_metrics(tl_k, mult_base)

    # Bootstrap
    boot_lo, boot_mean, boot_hi = block_bootstrap_delta_nc_pnl(tl_k, mult_base, mult)
    # Permutation
    obs_delta, p_perm = permutation_test_edge_quintiles(
        tl_k, median_fsh, qbounds, variant,
    )
    # Jackknife
    jk_pos, jk_n, jk_rows = monthly_jackknife_delta(tl_k, mult_base, mult)

    return dict(
        label=label, variant=variant,
        metrics_new=metrics_new, metrics_base=metrics_base,
        delta_nc_pnl=metrics_new["nc_pnl"] - metrics_base["nc_pnl"],
        delta_nc_sharpe=metrics_new["nc_sharpe"] - metrics_base["nc_sharpe"],
        delta_nc_mdd=metrics_new["nc_mdd"] - metrics_base["nc_mdd"],
        delta_cp_pnl=metrics_new["cp_pnl"] - metrics_base["cp_pnl"],
        delta_cp_sharpe=metrics_new["cp_sharpe"] - metrics_base["cp_sharpe"],
        boot_lower_95=boot_lo, boot_mean=boot_mean, boot_upper_95=boot_hi,
        obs_delta_pnl=obs_delta, p_permutation=p_perm,
        jk_n_positive=jk_pos, jk_n_total=jk_n, jk_fraction_positive=jk_pos / jk_n,
    )


def format_report(results_primary: list[dict], results_baseline: list[dict]) -> str:
    lines = []
    lines.append("# Phase 1: Long-side Kelly-shrinkage sizing — offline evaluation")
    lines.append("")
    lines.append("**Run date**: 2026-04-21. **Seed**: 42. **N_bootstrap**: 1000. **N_permutation**: 1000.")
    lines.append("")
    lines.append("Plan: `C:\\Users\\laure\\.claude\\plans\\starry-rolling-rabin.md`")
    lines.append("")
    lines.append("## Phase 0 verdict — profit_scaling is orthogonal, not redundant")
    lines.append("")
    lines.append("Verified from `vol_arb/backtester.py:1830-1858`:")
    lines.append("")
    lines.append("- `profit_scaling_cap = 1.0` hard-codes the maximum to 1.0x — "
                 "**no upside boost ever applied**. This is a scale-down-only mechanism.")
    lines.append("- `est_ratio = est_net / est_gross` ranges over cost-ratio; the "
                 "scale factor is driven by `(gross - costs)/gross`, i.e. "
                 "**cost-awareness**, not edge-magnitude awareness. High-|edge| bars "
                 "and low-|edge| bars can have similar est_ratio if cost fractions match.")
    lines.append("- Kelly-shrinkage boost proposed here scales UP high-|edge| bars "
                 "(mult >= 1.0x). The mechanism is **orthogonal** to profit_scaling; "
                 "both can co-exist in the same position size (profit_scaling applied "
                 "first, then Kelly mult, then dynamic_btc_cap clip).")
    lines.append("")
    lines.append("## Results — Primary (post-aligner) log")
    lines.append("")
    lines.append(_format_results_block(results_primary))
    lines.append("")
    lines.append("## Results — Baseline (pre-aligner) log — cross-check")
    lines.append("")
    lines.append(_format_results_block(results_baseline))
    lines.append("")
    lines.append("## Pre-committed Phase 1 pass criteria (from plan)")
    lines.append("")
    lines.append("A variant passes Phase 1 iff ALL of:")
    lines.append("- (a) bootstrap lower-95% delta-NC-PnL > $0")
    lines.append("- (b) permutation p < 0.05")
    lines.append("- (c) monthly jackknife: >= 60% of month-drops still positive")
    lines.append("- (d) baseline-log cross-check: delta-NC-PnL same sign")
    lines.append("")
    lines.append("## Decision matrix")
    lines.append("")
    lines.append("| variant | (a) boot LCL>0 | (b) p<0.05 | (c) jk>=60% | (d) baseline sign | PASS? |")
    lines.append("|---|---|---|---|---|---|")
    prim_by_var = {r["variant"]: r for r in results_primary}
    base_by_var = {r["variant"]: r for r in results_baseline}
    for v in ["K1", "K2", "K3"]:
        p = prim_by_var[v]
        b = base_by_var[v]
        a_pass = p["boot_lower_95"] > 0
        b_pass = p["p_permutation"] < 0.05
        c_pass = p["jk_fraction_positive"] >= 0.60
        d_pass = (p["delta_nc_pnl"] > 0) == (b["delta_nc_pnl"] > 0)
        all_pass = a_pass and b_pass and c_pass and d_pass
        lines.append(
            f"| {v} "
            f"| {'Y' if a_pass else 'N'} (${p['boot_lower_95']:+,.0f}) "
            f"| {'Y' if b_pass else 'N'} (p={p['p_permutation']:.3f}) "
            f"| {'Y' if c_pass else 'N'} ({p['jk_fraction_positive']*100:.0f}%) "
            f"| {'Y' if d_pass else 'N'} (pri {p['delta_nc_pnl']:+.0f} / base {b['delta_nc_pnl']:+.0f}) "
            f"| **{'PASS' if all_pass else 'FAIL'}** |"
        )
    lines.append("")
    lines.append("Phase 2 proceeds only on a PASS verdict.")
    return "\n".join(lines)


def _format_results_block(results: list[dict]) -> str:
    lines = []
    lines.append("| variant | dNC PnL | dNC Sharpe | dNC MDD | dCP PnL (approx) | boot 95% CI | perm p | jk +ve |  boosted trades  | mean mult (longs) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    base = results[0]["metrics_base"]
    for r in results:
        mb = r["metrics_new"]
        lines.append(
            f"| {r['variant']} "
            f"| ${r['delta_nc_pnl']:+,.0f} "
            f"| {r['delta_nc_sharpe']:+.2f} "
            f"| ${r['delta_nc_mdd']:+,.0f} "
            f"| ${r['delta_cp_pnl']:+,.0f} "
            f"| [${r['boot_lower_95']:+,.0f}, ${r['boot_upper_95']:+,.0f}] "
            f"| {r['p_permutation']:.3f} "
            f"| {r['jk_n_positive']}/{r['jk_n_total']} ({r['jk_fraction_positive']*100:.0f}%) "
            f"| {mb['n_trades_boosted']:,}/{mb['n_trades_total']:,} "
            f"| {mb['mean_mult_longs']:.3f} |"
        )
    lines.append("")
    lines.append(f"Baseline metrics (mult=1.0 everywhere): "
                 f"NC PnL ${base['nc_pnl']:,.0f}, "
                 f"NC Sharpe {base['nc_sharpe']:.2f}, "
                 f"NC MDD ${base['nc_mdd']:,.0f}, "
                 f"Compound PnL ${base['cp_pnl']:,.0f}.")
    return "\n".join(lines)


def main():
    print("Loading data...")
    tl_pri = pd.read_parquet(TRADE_LOG_PRIMARY)
    preds_pri = pd.read_parquet(PREDS_ALIGNED)
    tl_base = pd.read_parquet(TRADE_LOG_BASELINE)
    preds_base = pd.read_parquet(PREDS_RAW)

    print(f"Primary trade log: {tl_pri.shape}; preds aligned: {preds_pri.shape}")
    print(f"Baseline trade log: {tl_base.shape}; preds raw: {preds_base.shape}")

    results_pri = []
    for v in ["K1", "K2", "K3"]:
        print(f"\n--- PRIMARY {v} ---")
        r = run_full_variant(tl_pri, preds_pri, v, label="primary")
        results_pri.append(r)
        m = r["metrics_new"]
        print(f"  dNC PnL ${r['delta_nc_pnl']:+,.0f}, dNC Sharpe {r['delta_nc_sharpe']:+.2f}, "
              f"dNC MDD ${r['delta_nc_mdd']:+,.0f}")
        print(f"  bootstrap 95% CI: [${r['boot_lower_95']:+,.0f}, ${r['boot_upper_95']:+,.0f}]")
        print(f"  permutation p={r['p_permutation']:.3f}")
        print(f"  jackknife +ve: {r['jk_n_positive']}/{r['jk_n_total']} ({r['jk_fraction_positive']*100:.0f}%)")
        print(f"  boosted: {m['n_trades_boosted']}/{m['n_trades_total']} trades, mean mult on longs={m['mean_mult_longs']:.3f}")

    results_base = []
    for v in ["K1", "K2", "K3"]:
        print(f"\n--- BASELINE-LOG CROSS-CHECK {v} ---")
        r = run_full_variant(tl_base, preds_base, v, label="baseline")
        results_base.append(r)
        print(f"  dNC PnL ${r['delta_nc_pnl']:+,.0f} (sign match with primary: {(r['delta_nc_pnl'] > 0) == (results_pri[len(results_base)-1]['delta_nc_pnl'] > 0)})")

    # Write report
    report = format_report(results_pri, results_base)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
