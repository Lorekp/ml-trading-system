"""h=5m vs h=15m deep-dive: regime-conditional / longs-only / ensemble.

Plan: ``starry-rolling-rabin.md`` Phase 2h5.7.D.

Closes the h=5 question by stress-testing the headline NC PnL gain
(+$420K) along three orthogonal axes. The audit revealed the canonical
h=5 backtest is leakage-clean (Phase A: embargo bug found in research-
path code only; Phase B: DRL hedger horizon mismatch is non-material).
The Sharpe drop (6.29 → 3.61) is therefore a real signal-quality cost,
not a pipeline bug. This script asks: is there a SUBSET (regime / side
/ portfolio mix) where h=5's $/Sharpe ratio survives the strict gates?

Pre-committed gates (locked in plan, not in code; see plan §Phase
2h5.7.D for the full table). Every threshold below was declared BEFORE
the analysis ran:

  Regime cell ships as conditional preset iff ALL hold:
    - dPnL >= +$50K
    - dSharpe >= -0.7
    - dMDD >= -$3K
    - bootstrap LCL > $0 (primary axis #1) or > $25K (secondary #2-5)
    - permutation p < 0.0033 (primary) or < 0.001 (secondary)
    - cell holds >= 250 trading days of joint activity

  Longs-only ships as opt-in preset iff ALL hold:
    - dPnL >= +$50K, dSharpe >= -0.7, dMDD >= -$3K (longs subset)
    - bootstrap LCL on d(longs PnL) > $0
    - per-month jackknife: >= 90% positive (>= 45/50 months)

  Ensemble ships as opt-in preset iff ALL hold (walk-forward):
    - WF blended Sharpe >= max(h=5 solo, h=15 solo) + 0.30
    - WF blended NC PnL >= h=15 solo NC PnL + $50K
    - WF blended NC MDD >= h=15 solo NC MDD - $3K
    - bootstrap LCL on (blended - h=15) NC PnL > $0
    - std(w_5* across months) <= 0.30  (weight stability)

Reuses every utility from ``compare_h5_vs_h15.py`` (no copy-paste).

Outputs ``vol_arb/output/H5_DEEP_DIVE_REPORT.md``.

Usage::

    cd C:/Claude/ml-trading-system
    python scripts/h5_deep_dive.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# Reuse utilities from the production comparison script. Keeps a single
# source of truth for daily aggregation, bootstrap CI, and permutation
# null — if those are ever fixed in compare_h5_vs_h15.py, this script
# inherits the fix automatically.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_h5_vs_h15 import (  # noqa: E402  (deliberate post-path-insert import)
    CAPITAL,
    H15,
    H5,
    SEED,
    annual_sharpe,
    block_bootstrap_delta,
    daily_pnl,
    headline,
    max_drawdown,
    monthly_jackknife,
    permutation_test,
    trade_economics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
H5_ALIGNED_PATH = (
    REPO_ROOT
    / "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias_h5m.parquet"
)
VPIN_5M_PATH = REPO_ROOT / "vol_arb/output/vpin_btc_5m.parquet"
VPIN_15M_PATH = REPO_ROOT / "vol_arb/output/vpin_btc_15m.parquet"
REPORT_PATH = REPO_ROOT / "vol_arb/output/H5_DEEP_DIVE_REPORT.md"


# ---------------------------------------------------------------------
# Regime-conditional analysis
# ---------------------------------------------------------------------


def _expanding_quantile_label(
    s: pd.Series, q_low: float = 0.33, q_high: float = 0.66
) -> pd.Series:
    """Tercile labels {-1, 0, +1} via EXPANDING-window quantile.

    Strictly leakage-safe: each bar's tercile is determined by the
    distribution of the series UP TO and INCLUDING that bar (no future
    information). Equivalent to ``rank by past observations``.

    A ``min_periods`` warm-up of 5,000 bars (~ 1 month at h=5) prevents
    early-sample noise from picking spurious terciles. Bars before the
    warm-up are labeled NaN and dropped from the cell partition.
    """
    n = len(s)
    out = np.full(n, np.nan)
    arr = s.values.astype(float)
    min_periods = 5_000  # ~ 1 month at h=5; ~ 1 quarter at h=15
    for i in range(min_periods, n):
        past = arr[: i + 1]
        ql = np.nanquantile(past, q_low)
        qh = np.nanquantile(past, q_high)
        v = arr[i]
        if np.isnan(v):
            continue
        out[i] = -1.0 if v <= ql else (1.0 if v >= qh else 0.0)
    return pd.Series(out, index=s.index, name=f"{s.name}_tercile")


def _attach_regime_axes(
    tl: pd.DataFrame, vpin: pd.DataFrame, beta_monthly: pd.Series
) -> pd.DataFrame:
    """Attach 5 regime labels to a fired-bar trade log.

    Axis 1 (PRIMARY): term-structure sign — `svi_term_slope_1w_2w`
        in trade log. Sign: contango (>0) vs backwardation (<0).
    Axis 2: ATM IV level — `iv_used` expanding-window terciles.
    Axis 3: VPIN — merge_asof backward from per-horizon VPIN parquet,
        expanding-window terciles. Use h=5 VPIN for h=5 trade log,
        h=15 VPIN for h=15 trade log (`vpin` is passed in).
    Axis 4: BTC trend — `index_price` 30-day rolling return sign.
    Axis 5: aligner-beta quality (h=5 only) — month's mean
        ``y_pred - y_pred_raw`` from the h=5 aligner output. Strong
        (beta >= 0.05) vs weak (beta < 0.05). Always 'na' for h=15.
    """
    df = tl.copy()
    df = df.loc[df["signal"] != 0]  # only fired bars

    # Axis 1: term-structure sign
    df["regime_term"] = np.where(
        df["svi_term_slope_1w_2w"] > 0, "contango",
        np.where(df["svi_term_slope_1w_2w"] < 0, "backwardation", "flat"),
    )

    # Axis 2: ATM IV tercile
    iv_t = _expanding_quantile_label(df["iv_used"].rename("iv"))
    df["regime_iv"] = np.where(
        iv_t < 0, "low_iv", np.where(iv_t > 0, "high_iv", "mid_iv")
    )

    # Axis 3: VPIN tercile (merge_asof backward)
    if vpin is not None and not vpin.empty:
        vpin_aligned = pd.merge_asof(
            df.index.to_frame(name="ts").reset_index(drop=True).sort_values("ts"),
            vpin.reset_index().sort_values("timestamp").rename(
                columns={"timestamp": "ts"}
            ),
            on="ts",
            direction="backward",
        )
        vpin_series = pd.Series(
            vpin_aligned["vpin_true"].values, index=df.index, name="vpin"
        )
        v_t = _expanding_quantile_label(vpin_series)
        df["regime_vpin"] = np.where(
            v_t < 0, "low_vpin", np.where(v_t > 0, "high_vpin", "mid_vpin")
        )
    else:
        df["regime_vpin"] = "na"

    # Axis 4: BTC trend — 30d rolling index_price return sign
    if "index_price" in df.columns:
        # Resample to daily for a stable 30d return; broadcast back.
        ts = df.index.tz_convert(None) if df.index.tz is not None else df.index
        ip_daily = df["index_price"].groupby(pd.Grouper(freq="1D")).last()
        ret_30d = ip_daily.pct_change(30)
        ret_aligned = ret_30d.reindex(df.index, method="ffill")
        df["regime_trend"] = np.where(
            ret_aligned > 0, "uptrend",
            np.where(ret_aligned < 0, "downtrend", "flat"),
        )
    else:
        df["regime_trend"] = "na"

    # Axis 5: aligner-β quality (h=5 only)
    if beta_monthly is not None and not beta_monthly.empty:
        # Map each bar's calendar month to that month's mean beta.
        ts_idx = df.index
        month_keys = ts_idx.tz_localize(None).to_period("M") if ts_idx.tz is not None \
            else ts_idx.to_period("M")
        beta_by_month = {p: float(v) for p, v in beta_monthly.items()}
        beta_for_bar = pd.Series(
            [beta_by_month.get(m, np.nan) for m in month_keys],
            index=df.index,
            name="beta",
        )
        df["regime_beta"] = np.where(
            beta_for_bar >= 0.05, "beta_strong",
            np.where(beta_for_bar < 0.05, "beta_weak", "beta_na"),
        )
    else:
        df["regime_beta"] = "na"

    return df


def _per_cell_metrics(
    tl5_cell: pd.DataFrame, tl15_cell: pd.DataFrame
) -> dict:
    """Compute headline + bootstrap + permutation for one regime cell.

    Both ``tl5_cell`` and ``tl15_cell`` should already be filtered to
    the same regime label. Returns dict with NC PnL/Sharpe/MDD per
    horizon, deltas, bootstrap CI, permutation p, joint-active-day
    count.
    """
    if tl5_cell.empty or tl15_cell.empty:
        return dict(empty=True, n5=0, n15=0)

    h5 = headline(tl5_cell)
    h15 = headline(tl15_cell)
    d5 = daily_pnl(tl5_cell, "net_pnl_nc")
    d15 = daily_pnl(tl15_cell, "net_pnl_nc")
    joint = pd.concat([d5.rename("h5"), d15.rename("h15")], axis=1).fillna(0.0)
    joint = joint.loc[(joint != 0).any(axis=1)]
    n_joint_days = int(len(joint))

    if n_joint_days < 30:  # too small for bootstrap to mean anything
        return dict(
            empty=False,
            n5=int(h5["n_trades"]),
            n15=int(h15["n_trades"]),
            pnl_5=float(h5["nc_pnl"]),
            pnl_15=float(h15["nc_pnl"]),
            delta_pnl=float(h5["nc_pnl"] - h15["nc_pnl"]),
            sharpe_5=float(h5["nc_sharpe"]),
            sharpe_15=float(h15["nc_sharpe"]),
            delta_sharpe=float(h5["nc_sharpe"] - h15["nc_sharpe"]),
            mdd_5=float(h5["nc_mdd"]),
            mdd_15=float(h15["nc_mdd"]),
            delta_mdd=float(h5["nc_mdd"] - h15["nc_mdd"]),
            n_joint_days=n_joint_days,
            bootstrap_lcl=float("nan"),
            bootstrap_ucl=float("nan"),
            perm_p=float("nan"),
            cell_too_small=True,
        )

    lcl, _, ucl = block_bootstrap_delta(joint["h5"], joint["h15"], n=1000)
    p = permutation_test(joint["h5"], joint["h15"], n=1000)

    return dict(
        empty=False,
        n5=int(h5["n_trades"]),
        n15=int(h15["n_trades"]),
        pnl_5=float(h5["nc_pnl"]),
        pnl_15=float(h15["nc_pnl"]),
        delta_pnl=float(h5["nc_pnl"] - h15["nc_pnl"]),
        sharpe_5=float(h5["nc_sharpe"]),
        sharpe_15=float(h15["nc_sharpe"]),
        delta_sharpe=float(h5["nc_sharpe"] - h15["nc_sharpe"]),
        mdd_5=float(h5["nc_mdd"]),
        mdd_15=float(h15["nc_mdd"]),
        delta_mdd=float(h5["nc_mdd"] - h15["nc_mdd"]),
        n_joint_days=n_joint_days,
        bootstrap_lcl=lcl,
        bootstrap_ucl=ucl,
        perm_p=p,
        cell_too_small=False,
    )


def _gate_pass_regime(cell: dict, axis_id: int) -> tuple[bool, list[str]]:
    """Return (pass, reasons-failed) per pre-committed regime gate.

    Axis 1 (term-structure) is PRIMARY — relaxed bootstrap LCL > $0,
    p < 0.0033. Axes 2-5 are SECONDARY — bootstrap LCL > $25K and
    p < 0.001.
    """
    reasons: list[str] = []
    if cell.get("empty") or cell.get("cell_too_small"):
        return False, ["sample too small (< 250 days or empty)"]
    if cell["n_joint_days"] < 250:
        reasons.append(f"n_joint_days={cell['n_joint_days']} < 250")
    if cell["delta_pnl"] < 50_000:
        reasons.append(f"dPnL ${cell['delta_pnl']:,.0f} < $50K")
    if cell["delta_sharpe"] < -0.7:
        reasons.append(f"dSharpe {cell['delta_sharpe']:+.2f} < -0.7")
    if cell["delta_mdd"] < -3_000:
        reasons.append(f"dMDD ${cell['delta_mdd']:,.0f} < -$3K")
    lcl_floor = 0 if axis_id == 1 else 25_000
    if cell["bootstrap_lcl"] <= lcl_floor:
        reasons.append(
            f"bootstrap LCL ${cell['bootstrap_lcl']:,.0f} <= ${lcl_floor:,}"
        )
    p_floor = 0.0033 if axis_id == 1 else 0.001
    if cell["perm_p"] >= p_floor:
        reasons.append(f"perm p {cell['perm_p']:.4f} >= {p_floor}")
    return len(reasons) == 0, reasons


def regime_conditional_analysis(
    tl5: pd.DataFrame, tl15: pd.DataFrame, beta_monthly_h5: pd.Series
) -> dict:
    """Run the 5-axis regime-conditional analysis. Returns nested dict
    keyed by (axis_id, axis_label) -> per-cell-label -> metrics dict.
    """
    vpin5 = pd.read_parquet(VPIN_5M_PATH)
    vpin15 = pd.read_parquet(VPIN_15M_PATH)

    tl5_lab = _attach_regime_axes(tl5, vpin5, beta_monthly_h5)
    # h=15 has NO aligner-beta axis — pass an empty Series so the
    # axis_5 column is filled with 'na' for h=15.
    tl15_lab = _attach_regime_axes(tl15, vpin15, pd.Series(dtype=float))

    axes = [
        (1, "term", "regime_term", ["contango", "backwardation"]),
        (2, "iv_level", "regime_iv", ["low_iv", "mid_iv", "high_iv"]),
        (3, "vpin", "regime_vpin", ["low_vpin", "mid_vpin", "high_vpin"]),
        (4, "trend", "regime_trend", ["uptrend", "downtrend"]),
        (5, "beta_quality", "regime_beta", ["beta_strong", "beta_weak"]),
    ]
    results: dict = {}
    for axis_id, axis_name, col, labels in axes:
        results[(axis_id, axis_name)] = {}
        for lab in labels:
            mask5 = tl5_lab[col] == lab
            if axis_id == 5:
                # h=15 has no β axis — gate becomes a within-h5 split
                # (h=5 in regime vs h=5 overall) for context only.
                # We still compute the comparison vs h=15 ALL trades
                # filtered to the bars where h=5 was in this regime;
                # this isn't a perfect partition but provides context.
                mask15 = tl15_lab.index.isin(tl5_lab.loc[mask5].index)
            else:
                mask15 = tl15_lab[col] == lab
            cell5 = tl5_lab.loc[mask5]
            cell15 = tl15_lab.loc[mask15]
            metrics = _per_cell_metrics(cell5, cell15)
            ok, reasons = _gate_pass_regime(metrics, axis_id)
            metrics["pass"] = ok
            metrics["fail_reasons"] = reasons
            results[(axis_id, axis_name)][lab] = metrics
    return results


# ---------------------------------------------------------------------
# Longs-only / shorts-only analysis
# ---------------------------------------------------------------------


def longs_only_analysis(tl5: pd.DataFrame, tl15: pd.DataFrame) -> dict:
    """Per-side h=5 vs h=15 metrics + bootstrap + monthly jackknife."""
    out: dict = {}
    for side, side_name in [(1, "longs"), (-1, "shorts")]:
        s5 = tl5.loc[tl5["signal"] == side]
        s15 = tl15.loc[tl15["signal"] == side]
        h5 = headline(s5)
        h15 = headline(s15)
        d5 = daily_pnl(s5, "net_pnl_nc")
        d15 = daily_pnl(s15, "net_pnl_nc")
        joint = pd.concat([d5.rename("h5"), d15.rename("h15")], axis=1).fillna(0.0)
        lcl, _, ucl = block_bootstrap_delta(joint["h5"], joint["h15"])
        p = permutation_test(joint["h5"], joint["h15"])
        # Per-month jackknife on this side's daily PnL difference
        jk = monthly_jackknife(s5, s15)
        n_pos = int((jk["delta_nc"] >= 0).sum())
        n_total = int(len(jk))
        pct_pos = (n_pos / n_total * 100) if n_total else 0.0

        side_dict = dict(
            n5=int(h5["n_trades"]),
            n15=int(h15["n_trades"]),
            pnl_5=float(h5["nc_pnl"]),
            pnl_15=float(h15["nc_pnl"]),
            delta_pnl=float(h5["nc_pnl"] - h15["nc_pnl"]),
            sharpe_5=float(h5["nc_sharpe"]),
            sharpe_15=float(h15["nc_sharpe"]),
            delta_sharpe=float(h5["nc_sharpe"] - h15["nc_sharpe"]),
            mdd_5=float(h5["nc_mdd"]),
            mdd_15=float(h15["nc_mdd"]),
            delta_mdd=float(h5["nc_mdd"] - h15["nc_mdd"]),
            bootstrap_lcl=lcl,
            bootstrap_ucl=ucl,
            perm_p=p,
            jackknife_pct_positive=pct_pos,
            jackknife_n_pos=n_pos,
            jackknife_n_total=n_total,
        )
        # Pre-committed gate (longs-only opt-in)
        reasons: list[str] = []
        if side_name == "longs":
            if side_dict["delta_pnl"] < 50_000:
                reasons.append(f"dPnL ${side_dict['delta_pnl']:,.0f} < $50K")
            if side_dict["delta_sharpe"] < -0.7:
                reasons.append(
                    f"dSharpe {side_dict['delta_sharpe']:+.2f} < -0.7"
                )
            if side_dict["delta_mdd"] < -3_000:
                reasons.append(f"dMDD ${side_dict['delta_mdd']:,.0f} < -$3K")
            if side_dict["bootstrap_lcl"] <= 0:
                reasons.append(
                    f"bootstrap LCL ${side_dict['bootstrap_lcl']:,.0f} <= $0"
                )
            if side_dict["jackknife_pct_positive"] < 90:
                reasons.append(
                    f"jackknife {side_dict['jackknife_pct_positive']:.1f}% < 90%"
                )
            side_dict["pass"] = len(reasons) == 0
            side_dict["fail_reasons"] = reasons
        out[side_name] = side_dict
    return out


# ---------------------------------------------------------------------
# Ensemble (portfolio) analysis
# ---------------------------------------------------------------------


def _markowitz_w5_clipped(
    d5: pd.Series, d15: pd.Series, *, eps: float = 1e-12
) -> float:
    """Sharpe-maximizing w_5 in [0, 1] for blend w_5*d5 + (1-w_5)*d15.

    Closed-form for two-asset Sharpe maximization on demeaned returns.
    Clipped to [0, 1]; degenerate cases (zero variance, zero mean
    difference) → 0.5 fallback.
    """
    a = d5.fillna(0.0)
    b = d15.fillna(0.0)
    if len(a) < 5 or len(b) < 5:
        return 0.5
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov_ab = a.cov(b)
    # max Sharpe weight (closed-form, two-asset, no risk-free):
    num = mu_a * var_b - mu_b * cov_ab
    den = mu_a * var_b + mu_b * var_a - (mu_a + mu_b) * cov_ab
    if abs(den) < eps:
        return 0.5
    w = num / den
    return float(np.clip(w, 0.0, 1.0))


def ensemble_analysis(d5: pd.Series, d15: pd.Series) -> dict:
    """Daily-blend grid sweep + walk-forward Sharpe-optimal weight."""
    # Align on union of trading days
    joint = pd.concat([d5.rename("h5"), d15.rename("h15")], axis=1).fillna(0.0)
    h5_solo_sharpe = annual_sharpe(joint["h5"])
    h15_solo_sharpe = annual_sharpe(joint["h15"])
    h5_solo_pnl = float(joint["h5"].sum())
    h15_solo_pnl = float(joint["h15"].sum())
    h5_solo_mdd, _ = max_drawdown(joint["h5"])
    h15_solo_mdd, _ = max_drawdown(joint["h15"])

    # In-sample grid sweep (ceiling)
    grid = np.arange(0.0, 1.05, 0.05)
    grid_rows = []
    for w5 in grid:
        blended = w5 * joint["h5"] + (1 - w5) * joint["h15"]
        s = annual_sharpe(blended)
        mdd, _ = max_drawdown(blended)
        grid_rows.append(dict(
            w5=float(w5),
            pnl=float(blended.sum()),
            sharpe=float(s),
            mdd=float(mdd),
        ))
    grid_df = pd.DataFrame(grid_rows)
    in_sample_sharpe_max = grid_df.iloc[grid_df["sharpe"].idxmax()].to_dict()

    # Walk-forward: trailing-12m Sharpe-optimal weight, applied OOS to
    # the next month. Need >= 13 months for the first OOS month.
    months = sorted({(d.year, d.month) for d in joint.index})
    if len(months) < 13:
        return dict(
            n_months=len(months),
            insufficient_history=True,
            in_sample_sharpe_max=in_sample_sharpe_max,
            h5_solo_sharpe=h5_solo_sharpe,
            h15_solo_sharpe=h15_solo_sharpe,
        )

    oos_blend: list[pd.Series] = []
    w5_history: list[float] = []
    for i in range(12, len(months)):
        train_months = months[i - 12 : i]
        test_y, test_m = months[i]
        train_idx = pd.Index(joint.index).map(
            lambda d: (d.year, d.month) in set(train_months)
        )
        test_idx = pd.Index(joint.index).map(
            lambda d: d.year == test_y and d.month == test_m
        )
        if not train_idx.any() or not test_idx.any():
            continue
        d5_tr = joint.loc[train_idx, "h5"]
        d15_tr = joint.loc[train_idx, "h15"]
        w5 = _markowitz_w5_clipped(d5_tr, d15_tr)
        w5_history.append(w5)
        d5_te = joint.loc[test_idx, "h5"]
        d15_te = joint.loc[test_idx, "h15"]
        blended = w5 * d5_te + (1 - w5) * d15_te
        oos_blend.append(blended)

    oos_series = pd.concat(oos_blend).sort_index()
    wf_sharpe = annual_sharpe(oos_series)
    wf_pnl = float(oos_series.sum())
    wf_mdd, _ = max_drawdown(oos_series)
    wf_w5_mean = float(np.mean(w5_history)) if w5_history else float("nan")
    wf_w5_std = float(np.std(w5_history, ddof=1)) if len(w5_history) > 1 else float("nan")
    wf_w5_median = float(np.median(w5_history)) if w5_history else float("nan")

    # Bootstrap CI on (blended - h=15 solo) NC PnL, day-block
    h15_oos = joint.loc[oos_series.index, "h15"]
    lcl, obs, ucl = block_bootstrap_delta(oos_series, h15_oos)

    # Pre-committed ensemble gate
    target_solo_sharpe = max(h5_solo_sharpe, h15_solo_sharpe)
    reasons: list[str] = []
    if wf_sharpe < target_solo_sharpe + 0.30:
        reasons.append(
            f"WF Sharpe {wf_sharpe:.2f} < max(solo) + 0.30 = "
            f"{target_solo_sharpe + 0.30:.2f}"
        )
    if wf_pnl < h15_solo_pnl + 50_000:
        reasons.append(
            f"WF PnL ${wf_pnl:,.0f} < h15 solo + $50K = "
            f"${h15_solo_pnl + 50_000:,.0f}"
        )
    if wf_mdd < h15_solo_mdd - 3_000:
        reasons.append(
            f"WF MDD ${wf_mdd:,.0f} < h15 solo - $3K = "
            f"${h15_solo_mdd - 3_000:,.0f}"
        )
    if lcl <= 0:
        reasons.append(f"bootstrap LCL ${lcl:,.0f} <= $0")
    if not np.isnan(wf_w5_std) and wf_w5_std > 0.30:
        reasons.append(f"std(w5) {wf_w5_std:.2f} > 0.30 (unstable weight)")

    return dict(
        n_months=len(months),
        insufficient_history=False,
        h5_solo_sharpe=float(h5_solo_sharpe),
        h15_solo_sharpe=float(h15_solo_sharpe),
        h5_solo_pnl=h5_solo_pnl,
        h15_solo_pnl=h15_solo_pnl,
        h5_solo_mdd=float(h5_solo_mdd),
        h15_solo_mdd=float(h15_solo_mdd),
        in_sample_grid=grid_df.to_dict(orient="records"),
        in_sample_sharpe_max=in_sample_sharpe_max,
        wf_sharpe=float(wf_sharpe),
        wf_pnl=wf_pnl,
        wf_mdd=float(wf_mdd),
        wf_w5_mean=wf_w5_mean,
        wf_w5_std=wf_w5_std,
        wf_w5_median=wf_w5_median,
        wf_n_oos_months=len(w5_history),
        bootstrap_lcl=float(lcl),
        bootstrap_obs=float(obs),
        bootstrap_ucl=float(ucl),
        pass_gate=len(reasons) == 0,
        fail_reasons=reasons,
    )


# ---------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------


def _fmt_money(x: float) -> str:
    if np.isnan(x):
        return "    nan"
    return f"${x:,.0f}"


def _fmt_money_signed(x: float) -> str:
    if np.isnan(x):
        return "    nan"
    return f"${x:+,.0f}"


def _format_regime(results: dict) -> str:
    lines: list[str] = []
    lines.append("\n## 1. Regime-conditional analysis")
    lines.append(
        "\nFive pre-committed axes; primary axis #1 (term structure) at "
        "Bonferroni alpha = 0.0033, secondary axes 2-5 at alpha = 0.001 "
        "AND bootstrap LCL > $25K floor."
    )
    for (axis_id, axis_name), cells in results.items():
        primary = " (PRIMARY)" if axis_id == 1 else ""
        lines.append(f"\n### Axis {axis_id}: {axis_name}{primary}\n")
        lines.append(
            "| label | n5 | n15 | dPnL | dSharpe | dMDD | LCL | UCL | "
            "perm p | days | gate |"
        )
        lines.append(
            "|-------|----|----:|-----:|--------:|-----:|----:|----:|------:|----:|------|"
        )
        for lab, m in cells.items():
            if m.get("empty"):
                lines.append(
                    f"| {lab} | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | 0 | EMPTY |"
                )
                continue
            verdict = "PASS" if m.get("pass") else "fail"
            lines.append(
                f"| {lab} | {m['n5']:,} | {m['n15']:,} | "
                f"{_fmt_money_signed(m['delta_pnl'])} | "
                f"{m['delta_sharpe']:+.2f} | "
                f"{_fmt_money_signed(m['delta_mdd'])} | "
                f"{_fmt_money_signed(m['bootstrap_lcl'])} | "
                f"{_fmt_money_signed(m['bootstrap_ucl'])} | "
                f"{m['perm_p']:.4f} | "
                f"{m['n_joint_days']} | {verdict} |"
            )
            if not m.get("pass") and m.get("fail_reasons"):
                lines.append(
                    f"|     | | | | | | | | | | | _{'; '.join(m['fail_reasons'])}_ |"
                )
    return "\n".join(lines)


def _format_longs(out: dict) -> str:
    lines = ["\n## 2. Longs-only / shorts-only analysis\n"]
    for side_name, m in out.items():
        lines.append(f"### {side_name}\n")
        lines.append(f"- n trades h=5: **{m['n5']:,}**, h=15: **{m['n15']:,}**")
        lines.append(
            f"- NC PnL h=5: **{_fmt_money(m['pnl_5'])}**, "
            f"h=15: **{_fmt_money(m['pnl_15'])}**, "
            f"d: **{_fmt_money_signed(m['delta_pnl'])}**"
        )
        lines.append(
            f"- NC Sharpe h=5: **{m['sharpe_5']:.2f}**, "
            f"h=15: **{m['sharpe_15']:.2f}**, "
            f"d: **{m['delta_sharpe']:+.2f}**"
        )
        lines.append(
            f"- NC MDD h=5: **{_fmt_money(m['mdd_5'])}**, "
            f"h=15: **{_fmt_money(m['mdd_15'])}**, "
            f"d: **{_fmt_money_signed(m['delta_mdd'])}**"
        )
        lines.append(
            f"- bootstrap 95% CI on dPnL: "
            f"**[{_fmt_money_signed(m['bootstrap_lcl'])}, "
            f"{_fmt_money_signed(m['bootstrap_ucl'])}]**"
        )
        lines.append(f"- permutation p: **{m['perm_p']:.4f}**")
        lines.append(
            f"- monthly jackknife: **{m['jackknife_pct_positive']:.1f}%** positive "
            f"({m['jackknife_n_pos']}/{m['jackknife_n_total']})"
        )
        if "pass" in m:
            lines.append(
                f"- **gate: {'PASS' if m['pass'] else 'FAIL'}**"
            )
            if not m["pass"]:
                lines.append(f"  - reasons: {'; '.join(m['fail_reasons'])}")
        lines.append("")
    return "\n".join(lines)


def _format_ensemble(out: dict) -> str:
    lines = ["\n## 3. Ensemble (portfolio) analysis\n"]
    if out.get("insufficient_history"):
        lines.append(
            f"Insufficient history ({out['n_months']} months); ensemble "
            "analysis requires >= 13 months for trailing-12m walk-forward."
        )
        return "\n".join(lines)

    lines.append(f"### Solo benchmarks\n")
    lines.append(f"- h=5 solo Sharpe: **{out['h5_solo_sharpe']:.2f}**, "
                 f"PnL: **{_fmt_money(out['h5_solo_pnl'])}**, "
                 f"MDD: **{_fmt_money(out['h5_solo_mdd'])}**")
    lines.append(f"- h=15 solo Sharpe: **{out['h15_solo_sharpe']:.2f}**, "
                 f"PnL: **{_fmt_money(out['h15_solo_pnl'])}**, "
                 f"MDD: **{_fmt_money(out['h15_solo_mdd'])}**")

    lines.append(f"\n### In-sample weight-grid sweep (ceiling)\n")
    lines.append("| w_5 | NC PnL | Sharpe | MDD |")
    lines.append("|----:|-------:|-------:|----:|")
    for r in out["in_sample_grid"]:
        if int(round(r["w5"] * 20)) % 1 == 0:  # every step of 0.05
            lines.append(
                f"| {r['w5']:.2f} | {_fmt_money(r['pnl'])} | "
                f"{r['sharpe']:.2f} | {_fmt_money(r['mdd'])} |"
            )
    sm = out["in_sample_sharpe_max"]
    lines.append(
        f"\n**In-sample Sharpe-max**: w5 = **{sm['w5']:.2f}**, "
        f"Sharpe = **{sm['sharpe']:.2f}**, "
        f"PnL = **{_fmt_money(sm['pnl'])}**"
    )

    lines.append(f"\n### Walk-forward (trailing-12m Markowitz, OOS apply to next month)\n")
    lines.append(f"- WF blended Sharpe: **{out['wf_sharpe']:.2f}** "
                 f"(h=15 solo + 0.30 = {out['h15_solo_sharpe'] + 0.30:.2f})")
    lines.append(f"- WF blended NC PnL: **{_fmt_money(out['wf_pnl'])}** "
                 f"(h=15 solo + $50K = {_fmt_money(out['h15_solo_pnl'] + 50_000)})")
    lines.append(f"- WF blended NC MDD: **{_fmt_money(out['wf_mdd'])}** "
                 f"(h=15 solo - $3K = {_fmt_money(out['h15_solo_mdd'] - 3_000)})")
    lines.append(
        f"- bootstrap CI on (WF - h15 solo) NC PnL: "
        f"**[{_fmt_money_signed(out['bootstrap_lcl'])}, "
        f"{_fmt_money_signed(out['bootstrap_ucl'])}]**"
    )
    lines.append(
        f"- w_5 walk-forward stats: mean **{out['wf_w5_mean']:.3f}**, "
        f"median **{out['wf_w5_median']:.3f}**, "
        f"std **{out['wf_w5_std']:.3f}**"
    )
    lines.append(
        f"- WF n OOS months: **{out['wf_n_oos_months']}**"
    )
    lines.append(
        f"\n**Gate: {'PASS' if out['pass_gate'] else 'FAIL'}**"
    )
    if not out["pass_gate"]:
        lines.append(f"\nReasons: {'; '.join(out['fail_reasons'])}")
    return "\n".join(lines)


def _format_tldr(regime_results: dict, longs_out: dict, ensemble_out: dict) -> str:
    """One-paragraph verdict per analysis at the top of the report."""
    # Regime: any cell pass?
    any_regime_pass = any(
        m.get("pass")
        for cells in regime_results.values()
        for m in cells.values()
    )
    regime_v = "SHIP-AS-OPT-IN" if any_regime_pass else "REJECT"

    longs_v = "SHIP-AS-OPT-IN" if longs_out["longs"].get("pass") else "REJECT"

    if ensemble_out.get("insufficient_history"):
        ens_v = "INSUFFICIENT-HISTORY"
    else:
        ens_v = "SHIP-AS-OPT-IN" if ensemble_out["pass_gate"] else "REJECT"

    return (
        f"## TL;DR\n\n"
        f"- **Regime-conditional**: {regime_v}\n"
        f"- **Longs-only**: {longs_v}\n"
        f"- **Ensemble (h=5 + h=15)**: {ens_v}\n"
    )


def main() -> int:
    print("Loading trade logs...")
    tl5 = pd.read_parquet(H5)
    tl15 = pd.read_parquet(H15)

    # Per-month aligner-β at h=5 (axis 5)
    print("Computing per-month aligner-beta at h=5...")
    pred = pd.read_parquet(H5_ALIGNED_PATH)
    beta = pred["y_pred"] - pred["y_pred_raw"]
    beta_monthly_h5 = beta.groupby(pd.Grouper(freq="1ME")).mean()
    beta_monthly_h5.index = beta_monthly_h5.index.to_period("M")

    print("Running regime-conditional analysis (5 axes)...")
    regime_results = regime_conditional_analysis(tl5, tl15, beta_monthly_h5)

    print("Running longs-only / shorts-only analysis...")
    longs_out = longs_only_analysis(tl5, tl15)

    print("Running ensemble (portfolio) analysis...")
    d5 = daily_pnl(tl5)
    d15 = daily_pnl(tl15)
    ensemble_out = ensemble_analysis(d5, d15)

    # Build report
    parts = [
        "# h=5m Deep-Dive Report\n",
        "Generated by `scripts/h5_deep_dive.py`. Plan: "
        "`starry-rolling-rabin.md` Phase 2h5.7.D.\n",
        _format_tldr(regime_results, longs_out, ensemble_out),
        _format_regime(regime_results),
        _format_longs(longs_out),
        _format_ensemble(ensemble_out),
    ]
    REPORT_PATH.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")

    # Also print TL;DR to stdout
    print("\n" + _format_tldr(regime_results, longs_out, ensemble_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
