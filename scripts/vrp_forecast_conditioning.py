"""VRP forecast-conditioning test — does the HAR-PD RV forecast (and regime
signals) rescue the marginal UNCONDITIONAL delta-hedged short-straddle carry?

WALK-FORWARD, honest, no lookahead, no cherry-pick. ADD-ONLY: reuses
vol_arb.realistic_eval.evaluate_realistic at hold=to-expiry + the candidate_table
forecast features. Never edits realistic_eval.py / backtester.py / honest_pnl.py.

THE QUESTION
------------
The UNCONDITIONAL to-expiry delta-hedged SHORT straddle (band re-tuned ~0.2-0.75)
is economically positive (hedged NC +$276K..+$367K, overlap-adj NW Sharpe
~+1.3..+1.9) but only 3-4/5 years positive (2023 LOSES at every band) and
marginal under a DSR penalty. Does conditioning on the entry-time forecast
(forecast_edge = rv_pred - DVOL) and/or a DVOL-percentile regime signal:
  1) ENTRY FILTER  — short only when forecast says RV << implied
  2) SIZING        — scale by forecast edge (cap)
  3) REGIME GATE   — stand down in the crisis regime (2023)
lift the overlap-adjusted Sharpe AND fix the negative 2023, WALK-FORWARD?

WALK-FORWARD PROTOCOL (no lookahead, no in-sample tuning)
--------------------------------------------------------
The fill-covered window is 2023-05+. Folds = calendar years {2023,2024,2025,2026}.
For each OOS year Y, the threshold/parameter is selected ONLY on years < Y
(the past), then applied to year Y. 2023 has NO prior covered year -> it runs on
a FIXED first-principles default (forecast_edge < 0, i.e. only short when the
forecast itself says RV < implied; regime gate at the 1-yr-DVOL 80th pct floor),
NOT a tuned value. The headline rests on the CONCATENATED OOS series across all
folds (genuinely out-of-sample) and its per-year signs.

The verdict rests on the OVERLAP-ADJUSTED Sharpe (Newey-West + non-overlapping),
the per-year signs (does 2023 flip or get skipped?), MaxDD, gate_no_fake_profit,
and an honest DSR with K = 6 bands already tried + the forecast configs tried.

Usage:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/vrp_forecast_conditioning.py
  PYTHONPATH=. .venv/Scripts/python.exe scripts/vrp_forecast_conditioning.py --smoke
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic
from vol_arb.honest_pnl import gate_no_fake_profit
from scripts.longhold_hedged_vrp import (
    _sharpe_iid, _max_drawdown, _newey_west_sharpe, _nonoverlap_sharpe,
)

TRADE_LOG = "vol_arb/output/BTC_v2_trade_log.parquet"
CAND_TABLE = "vol_arb/output/candidate_table.parquet"
OUT_MD = "vol_arb/output/VRP_FORECAST_CONDITIONING.md"

# Bands the unconditional sweep already tried (sets part of the DSR K).
BANDS_TRIED = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75]
# The economically-positive band region. We anchor the forecast test on the
# project-default band 0.35 (mid of the positive region) and also report 0.20/0.50.
PRIMARY_BAND = 0.35

# Folds: calendar years in the fill-covered window.
FOLDS = [2023, 2024, 2025, 2026]


# ──────────────────────────────────────────────────────────────────────────
# Eval: covered to-expiry hedged trades + joined forecast features
# ──────────────────────────────────────────────────────────────────────────
def eval_covered(tl: pd.DataFrame, band: float) -> pd.DataFrame:
    """Run the realistic to-expiry hedged eval at one band; return the COVERED
    trades with hedged PnL, entry/exit ts, signal, and joined forecast features."""
    cfg = RealisticEvalConfig(
        hold="expiry", real_bs_delta=True, fill_tol_minutes=60.0,
        spread_source="tape", delta_hedge_band=band,
    )
    res = evaluate_realistic(tl, cfg)
    f = res.trade_log
    cov = f[f["premium_tier"] > 0].sort_index().copy()
    cov["entry_ts"] = pd.DatetimeIndex(pd.to_datetime(cov.index, utc=True))
    cov["exit_ts"] = pd.DatetimeIndex(pd.to_datetime(cov["expiry_dt"].values, utc=True))
    cov["year"] = cov["entry_ts"].dt.year

    # Join forecast features on the entry timestamp (causal, entry-time only).
    ct = pd.read_parquet(CAND_TABLE)
    feat_cols = [
        "rv_pred_pct", "dvol_close", "iv", "svi_gated",
        "forecast_edge_dvol", "forecast_edge_iv", "forecast_edge_svi",
        "jump_fraction", "term_slope", "dvol_pct_1y", "dte_days",
    ]
    feat = ct[[c for c in feat_cols if c in ct.columns]].copy()
    feat.index = pd.DatetimeIndex(pd.to_datetime(feat.index, utc=True))
    # Drop any trade-log columns that collide with the candidate-table forecast
    # features so the JOINED value is unambiguously the entry-time forecast
    # feature from candidate_table (the documented signal), not a same-named
    # trade-log column. _ct suffix-free.
    overlap = [c for c in feat.columns if c in cov.columns]
    cov = cov.drop(columns=overlap)
    cov = cov.join(feat, how="left")
    return cov, res, f


# ──────────────────────────────────────────────────────────────────────────
# Metrics on a (possibly masked / sized) PnL vector
# ──────────────────────────────────────────────────────────────────────────
def metrics(pnl: np.ndarray, entry_ts: pd.DatetimeIndex, exit_ts: pd.DatetimeIndex,
            year: np.ndarray) -> dict:
    pnl = np.asarray(pnl, float)
    fin = np.isfinite(pnl)
    pnl_f = pnl[fin]
    e_f = entry_ts[fin]
    x_f = exit_ts[fin]
    y_f = year[fin]
    n = len(pnl_f)
    per_year = {int(y): float(np.nansum(pnl_f[y_f == y])) for y in sorted(set(y_f))}
    return {
        "n": n,
        "total": float(np.nansum(pnl_f)),
        "mean": float(np.nanmean(pnl_f)) if n else float("nan"),
        "naive_sharpe": _sharpe_iid(pnl_f),
        "nw_sharpe": _newey_west_sharpe(pnl_f, e_f, x_f),
        "nonov_sharpe": _nonoverlap_sharpe(pnl_f, e_f, x_f)[0],
        "n_keep": _nonoverlap_sharpe(pnl_f, e_f, x_f)[1],
        "maxdd": _max_drawdown(pnl_f[np.argsort(e_f.values)]),
        "worst": float(np.nanmin(pnl_f)) if n else float("nan"),
        "per_year": per_year,
        "pos_years": sum(1 for v in per_year.values() if v > 0),
        "tot_years": len(per_year),
    }


# ──────────────────────────────────────────────────────────────────────────
# Walk-forward driver: a strategy is a per-trade weight in [0, cap] chosen from a
# PARAM, where PARAM is tuned on past years only.
# ──────────────────────────────────────────────────────────────────────────
def wf_apply(cov: pd.DataFrame, weight_fn, param_grid, default_param,
             select_metric="nw_sharpe", cond_col="forecast_edge_dvol"):
    """Walk-forward apply of a parametrized weighting rule.

    weight_fn(cov_subset, param) -> per-trade weight array in [0, cap] aligned to
    cov_subset rows (0 = skip trade, 1 = full size, >1 = upsize).

    For each fold year Y: pick param* maximizing `select_metric` of the WEIGHTED
    hedged PnL over years < Y (the training span). The training span must contain
    >=30 trades with a NON-NaN conditioning feature (`cond_col`) — otherwise the
    rule cannot be tuned causally (e.g. 2023 has no prior forecast-covered year,
    so it falls back to the FIXED first-principles default, NOT a degenerate tune
    on the forecast-NaN 2022 trades). Apply param* to year Y -> OOS weighted PnL.
    Returns (oos_pnl Series aligned to cov.index, chosen params per fold).
    """
    base = cov["total_hedged_usd"].to_numpy(float)
    oos_w = np.full(len(cov), np.nan)
    chosen = {}
    yr = cov["year"].to_numpy()
    cond_ok = cov[cond_col].notna().to_numpy() if cond_col in cov.columns else np.ones(len(cov), bool)
    for Y in FOLDS:
        test_mask = yr == Y
        if test_mask.sum() == 0:
            continue
        train_mask = yr < Y
        # require >=30 prior-year trades that actually carry the conditioning feature
        n_train_cond = int((train_mask & cond_ok).sum())
        if n_train_cond < 30:
            param = default_param
            chosen[Y] = {"param": _param_repr(param),
                         "source": f"default (only {n_train_cond} prior forecast-covered trades)"}
        else:
            ctr = cov[train_mask]
            best_param, best_score = default_param, -np.inf
            for p in param_grid:
                w = weight_fn(ctr, p)
                pnl = ctr["total_hedged_usd"].to_numpy(float) * w
                m = metrics(pnl, pd.DatetimeIndex(ctr["entry_ts"]),
                            pd.DatetimeIndex(ctr["exit_ts"]), ctr["year"].to_numpy())
                score = m[select_metric]
                # tie-break: require the weighting actually keep trades
                if np.isfinite(score) and score > best_score and m["n"] >= 20:
                    best_score, best_param = score, p
            param = best_param
            chosen[Y] = {"param": _param_repr(param), "train_score": float(best_score)}
        w_test = weight_fn(cov[test_mask], param)
        oos_w[test_mask] = base[test_mask] * w_test
    return pd.Series(oos_w, index=cov.index), chosen


def _param_repr(p):
    if isinstance(p, (tuple, list)):
        return [round(float(x), 4) for x in p]
    return round(float(p), 4)


# ──────────────────────────────────────────────────────────────────────────
# Weight functions for the three families
# ──────────────────────────────────────────────────────────────────────────
def w_entry_filter(cov: pd.DataFrame, tau: float) -> np.ndarray:
    """ENTRY FILTER: short only when forecast_edge_dvol < tau (forecast RV below
    implied by margin tau). tau in vol-points; tau<0 => only when forecast itself
    says RV < implied. Trades with NaN forecast keep size 1 (can't condition)."""
    fe = cov["forecast_edge_dvol"].to_numpy(float)
    keep = ~(fe >= tau)  # keep if fe < tau; NaN -> NaN>=tau is False -> keep
    return keep.astype(float)


def w_regime_gate(cov: pd.DataFrame, pct: float) -> np.ndarray:
    """REGIME GATE: stand down when trailing-1y DVOL percentile > pct (crisis).
    NaN percentile (early window) keeps size 1."""
    dp = cov["dvol_pct_1y"].to_numpy(float)
    stand_down = dp > pct  # NaN > pct -> False -> keep
    return (~stand_down).astype(float)


def w_sizing_edge(cov: pd.DataFrame, params) -> np.ndarray:
    """SIZING: scale by forecast edge. weight = clip(1 + k*max(edge_norm,0), 0, cap)
    where edge_norm = (-forecast_edge_dvol)/scale so MORE size when forecast RV <<
    implied (edge_dvol very negative). params=(k, cap). scale fixed = 10 vp.
    NaN forecast -> weight 1."""
    k, cap = params
    fe = cov["forecast_edge_dvol"].to_numpy(float)
    edge_norm = np.where(np.isfinite(fe), -fe / 10.0, 0.0)  # +ve when RV<<implied
    w = np.clip(1.0 + k * np.maximum(edge_norm, 0.0), 0.0, cap)
    w = np.where(np.isfinite(fe), w, 1.0)
    return w


def w_filter_and_regime(cov: pd.DataFrame, params) -> np.ndarray:
    """COMBINED: entry filter AND regime gate."""
    tau, pct = params
    return w_entry_filter(cov, tau) * w_regime_gate(cov, pct)


# ──────────────────────────────────────────────────────────────────────────
# DSR (Deflated Sharpe Ratio) — Bailey & Lopez de Prado haircut for K trials
# ──────────────────────────────────────────────────────────────────────────
def deflated_sharpe(sr_obs: float, n_obs: int, n_trials: int,
                    skew: float = 0.0, kurt: float = 3.0,
                    sr_trial_std: float | None = None) -> dict:
    """DSR (Bailey & Lopez de Prado 2014): probability the OOS Sharpe is > 0 after
    a multiple-testing haircut. sr_obs is a PER-OBSERVATION Sharpe (mean/sd, NOT
    the *sqrt(n) t-stat form).

    The deflation benchmark sr0 = expected MAX of K trial Sharpes:
        sr0 = sr_trial_std * [ (1-g)*Z^-1(1-1/K) + g*Z^-1(1-1/(K*e)) ],  g=Euler.
    CRITICAL: the bracket term is in UNITS OF THE TRIAL-SHARPE CROSS-SECTIONAL STD.
    Without multiplying by sr_trial_std the benchmark is nonsensically large. When
    the cross-sectional std of the K trial Sharpes is unknown we use the analytic
    per-observation Sharpe ESTIMATION std under the null, sqrt(1/n) (Lo 2002:
    Var(SR_hat) ~ (1 + SR^2/2)/n; at SR~0 -> 1/n), which is the standard fallback.

    DSR = Phi( (sr_obs - sr0)*sqrt(n-1) / sqrt(1 - skew*sr_obs + (kurt-1)/4*sr_obs^2) ).
    """
    if not np.isfinite(sr_obs) or n_obs < 3 or n_trials < 1:
        return {"dsr": float("nan"), "sr0_perobs": float("nan")}
    g = 0.5772156649
    if sr_trial_std is None:
        sr_trial_std = np.sqrt(1.0 / n_obs)   # analytic null SR estimation std
    if n_trials == 1:
        sr0 = 0.0                              # no multiple-testing inflation
    else:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr0 = sr_trial_std * ((1.0 - g) * z1 + g * z2)
    denom = np.sqrt(max(1.0 - skew * sr_obs + (kurt - 1.0) / 4.0 * sr_obs ** 2, 1e-9))
    dsr = stats.norm.cdf((sr_obs - sr0) * np.sqrt(n_obs - 1.0) / denom)
    return {"dsr": float(dsr), "sr0_perobs": float(sr0),
            "sr_trial_std": float(sr_trial_std)}


def _daily_perobs_sharpe(pnl: np.ndarray, entry_ts: pd.DatetimeIndex) -> tuple[float, int]:
    """Per-observation (NOT *sqrt(n)) DAILY Sharpe + n_days, for DSR. Aggregates
    per-trade PnL onto entry-day grid (same grid the NW Sharpe uses)."""
    fin = np.isfinite(pnl)
    df = pd.DataFrame({"pnl": np.asarray(pnl, float)[fin]},
                      index=entry_ts[fin])
    daily = df["pnl"].groupby(df.index.floor("D")).sum().to_numpy(float)
    if len(daily) < 3 or daily.std(ddof=1) == 0:
        return float("nan"), len(daily)
    return float(daily.mean() / daily.std(ddof=1)), len(daily)


# ──────────────────────────────────────────────────────────────────────────
def random_thinning_null(cov: pd.DataFrame, n_keep: int, n_draws: int = 300,
                         seed: int = 0) -> dict:
    """NULL: randomly KEEP n_keep of the covered trades (matching the count a
    forecast filter keeps), n_draws times; return the NW-Sharpe distribution.

    Thinning an OVERLAPPING-trade book mechanically raises the overlap-adjusted
    Sharpe (fewer concurrent positions -> less autocorrelation penalty). A real
    forecast edge must beat this random-thinning null, not just the full-book
    Sharpe. Restricted to the conditionable (forecast non-NaN, 2023+) universe so
    it is apples-to-apples with the filters.
    """
    c = cov[(cov["year"] >= 2023)].copy()
    base = c["total_hedged_usd"].to_numpy(float)
    e = pd.DatetimeIndex(c["entry_ts"]); x = pd.DatetimeIndex(c["exit_ts"])
    rng = np.random.RandomState(seed)
    n = len(base)
    n_keep = min(max(n_keep, 3), n)
    nws = []
    for _ in range(n_draws):
        idx = rng.choice(n, n_keep, replace=False)
        w = np.zeros(n); w[idx] = 1.0
        s = _newey_west_sharpe(base * w, e, x)
        if np.isfinite(s):
            nws.append(s)
    nws = np.asarray(nws, float)
    return {"mean": float(nws.mean()) if len(nws) else float("nan"),
            "p95": float(np.percentile(nws, 95)) if len(nws) else float("nan"),
            "max": float(nws.max()) if len(nws) else float("nan"),
            "n_keep": n_keep, "n_universe": n}


def summarize(name: str, pnl: np.ndarray, cov: pd.DataFrame) -> dict:
    e = pd.DatetimeIndex(cov["entry_ts"])
    x = pd.DatetimeIndex(cov["exit_ts"])
    y = cov["year"].to_numpy()
    m = metrics(pnl, e, x, y)
    m["name"] = name
    sr_perobs, n_days = _daily_perobs_sharpe(pnl, e)
    m["sr_perobs_daily"] = sr_perobs
    m["n_days"] = n_days
    return m


def fmt_year(per_year: dict) -> str:
    return ", ".join(f"{k}:${v:,.0f}" for k, v in per_year.items())


# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="single band only, coarse grids")
    args = ap.parse_args()

    tl = pd.read_parquet(TRADE_LOG)
    bands = [PRIMARY_BAND] if args.smoke else [0.20, 0.35, 0.50]

    report = []  # md lines
    report.append("# VRP Forecast-Conditioning — does the HAR-PD forecast rescue the carry?\n")
    report.append(f"_Generated by `scripts/vrp_forecast_conditioning.py`. Walk-forward, "
                  f"OOS folds {FOLDS}, primary band {PRIMARY_BAND}._\n")

    all_configs_tried = []   # for DSR K count
    results_by_band = {}

    for band in bands:
        print("#" * 78)
        print(f"BAND = {band:.2f}")
        cov, res, f_full = eval_covered(tl, band)
        n = len(cov)
        e = pd.DatetimeIndex(cov["entry_ts"])
        x = pd.DatetimeIndex(cov["exit_ts"])
        y = cov["year"].to_numpy()
        base = cov["total_hedged_usd"].to_numpy(float)

        # ---- UNCONDITIONAL baseline ----
        base_m = summarize("unconditional", base, cov)
        print(f"  UNCOND n={base_m['n']} total=${base_m['total']:,.0f} "
              f"NW={base_m['nw_sharpe']:.3f} nonov={base_m['nonov_sharpe']:.3f} "
              f"+yrs={base_m['pos_years']}/{base_m['tot_years']}")
        print(f"         per-year: {fmt_year(base_m['per_year'])}")

        # ---- forecast feature coverage on covered trades ----
        fe_cov = cov["forecast_edge_dvol"].notna().mean()
        dp_cov = cov["dvol_pct_1y"].notna().mean()
        print(f"  feature coverage on covered trades: forecast_edge_dvol "
              f"{fe_cov:.1%}, dvol_pct_1y {dp_cov:.1%}")

        # ---- WALK-FORWARD families ----
        # 1) ENTRY FILTER: tau grid (vol-points). default = 0 (only short when RV<implied).
        tau_grid = [-15, -10, -5, 0, 5, 10] if not args.smoke else [-10, 0, 10]
        oos_filt, ch_filt = wf_apply(cov, w_entry_filter, tau_grid, default_param=0.0)
        m_filt = summarize("wf_entry_filter", oos_filt.to_numpy(), cov)

        # 2) REGIME GATE: percentile grid. default = 0.80 (stand down top-20% DVOL).
        pct_grid = [0.70, 0.80, 0.90, 0.95] if not args.smoke else [0.80, 0.90]
        oos_reg, ch_reg = wf_apply(cov, w_regime_gate, pct_grid, default_param=0.80)
        m_reg = summarize("wf_regime_gate", oos_reg.to_numpy(), cov)

        # 3) SIZING: (k, cap) grid. default = (1.0, 2.0).
        size_grid = ([(0.5, 1.5), (1.0, 2.0), (2.0, 3.0), (1.0, 1.0)]
                     if not args.smoke else [(1.0, 2.0)])
        oos_size, ch_size = wf_apply(cov, w_sizing_edge, size_grid, default_param=(1.0, 2.0))
        m_size = summarize("wf_sizing_edge", oos_size.to_numpy(), cov)

        # 4) COMBINED filter + regime: (tau, pct) grid. default = (0, 0.80).
        comb_grid = ([(0, 0.90), (0, 0.80), (-5, 0.85), (5, 0.90)]
                     if not args.smoke else [(0, 0.80)])
        oos_comb, ch_comb = wf_apply(cov, w_filter_and_regime, comb_grid,
                                     default_param=(0.0, 0.80))
        m_comb = summarize("wf_filter_regime", oos_comb.to_numpy(), cov)

        fam = [base_m, m_filt, m_reg, m_size, m_comb]
        chosen = {"entry_filter": ch_filt, "regime_gate": ch_reg,
                  "sizing_edge": ch_size, "filter_regime": ch_comb}
        results_by_band[band] = {"families": fam, "chosen": chosen,
                                 "fe_cov": float(fe_cov), "dp_cov": float(dp_cov)}

        # count configs for DSR K: each family's grid is one search per fold but
        # the *strategy family* is the trial unit; count distinct families tried.
        all_configs_tried.extend([f"{band}:entry_filter", f"{band}:regime_gate",
                                  f"{band}:sizing_edge", f"{band}:filter_regime"])

        for m in fam:
            print(f"  {m['name']:18s} n={m['n']:4d} total=${m['total']:>11,.0f} "
                  f"NW={m['nw_sharpe']:6.3f} nonov={m['nonov_sharpe']:6.3f} "
                  f"MDD=${m['maxdd']:>9,.0f} +yrs={m['pos_years']}/{m['tot_years']}")
            print(f"      per-year: {fmt_year(m['per_year'])}")

        # ---- gate_no_fake_profit on the covered subset (model vs real fills) ----
        # The realistic hedged total IS the honest mark; check real_option_pnl_nc.
        gate_passed, gate_detail = _run_gate(f_full)
        results_by_band[band]["gate"] = {"passed": gate_passed, "detail": gate_detail}
        print(f"  gate_no_fake_profit: {gate_passed} — {gate_detail}")

        # ---- random-thinning NULL: best filter keeps n_keep; how much of its NW
        #      Sharpe is just mechanical book-thinning vs real forecast signal? ----
        n_keep_filt = int(np.isfinite(oos_filt.to_numpy()).sum()
                          - (oos_filt.to_numpy() == 0).sum())
        thin = random_thinning_null(cov, n_keep_filt)
        results_by_band[band]["thinning_null"] = thin
        print(f"  random-thinning null (keep {thin['n_keep']}/{thin['n_universe']}): "
              f"NW mean {thin['mean']:.3f}, p95 {thin['p95']:.3f}, max {thin['max']:.3f} "
              f"-- vs entry_filter NW {m_filt['nw_sharpe']:.3f}")

    # ── DSR: K = 6 bands already tried + forecast families tried ──
    K = len(BANDS_TRIED) + len(set(all_configs_tried))
    print("\n" + "#" * 78)
    print(f"DSR penalty K = {len(BANDS_TRIED)} bands + {len(set(all_configs_tried))} "
          f"forecast configs = {K}")

    # Headline = best OOS family by NW Sharpe at the primary band, vs uncond.
    prim = results_by_band[PRIMARY_BAND]
    uncond = prim["families"][0]
    cond_fams = prim["families"][1:]
    best_cond = max(cond_fams, key=lambda m: (m["nw_sharpe"] if np.isfinite(m["nw_sharpe"]) else -np.inf))

    dsr_uncond = deflated_sharpe(uncond["sr_perobs_daily"], uncond["n_days"], K)
    dsr_best = deflated_sharpe(best_cond["sr_perobs_daily"], best_cond["n_days"], K)
    print(f"\nUNCOND  (band {PRIMARY_BAND}): NW={uncond['nw_sharpe']:.3f} "
          f"sr_perobs_daily={uncond['sr_perobs_daily']:.4f} n_days={uncond['n_days']} "
          f"DSR={dsr_uncond['dsr']:.3f}")
    print(f"BEST COND ({best_cond['name']}): NW={best_cond['nw_sharpe']:.3f} "
          f"sr_perobs_daily={best_cond['sr_perobs_daily']:.4f} n_days={best_cond['n_days']} "
          f"DSR={dsr_best['dsr']:.3f}")

    # ── Write report ──
    _write_report(report, results_by_band, K, dsr_uncond, dsr_best,
                  uncond, best_cond, PRIMARY_BAND)
    print(f"\nwrote {OUT_MD}")

    # ── final verdict line ──
    print("\n" + "=" * 78)
    _verdict(uncond, best_cond, dsr_uncond, dsr_best)


def _run_gate(f_full: pd.DataFrame):
    if "real_option_pnl_nc" not in f_full.columns:
        return None, "real_option_pnl_nc absent"
    try:
        gr = gate_no_fake_profit(f_full, model_col="total_hedged_usd",
                                 real_col="real_option_pnl_nc")
        return bool(gr.passed), (gr.detail if isinstance(gr.detail, str) else json.dumps(gr.detail, default=str)[:200])
    except Exception as ex:  # noqa
        return None, f"gate error: {ex}"


def _verdict(uncond, best_cond, dsr_uncond, dsr_best):
    improves = (np.isfinite(best_cond["nw_sharpe"]) and np.isfinite(uncond["nw_sharpe"])
                and best_cond["nw_sharpe"] > uncond["nw_sharpe"])
    fixes_2023 = best_cond["per_year"].get(2023, -1) >= 0 or 2023 not in best_cond["per_year"]
    clears_dsr = np.isfinite(dsr_best["dsr"]) and dsr_best["dsr"] > 0.95
    print("VERDICT:")
    print(f"  forecast-cond improves overlap NW Sharpe vs uncond? {improves} "
          f"({best_cond['nw_sharpe']:.3f} vs {uncond['nw_sharpe']:.3f})")
    print(f"  2023 fixed/skipped to non-negative? {fixes_2023} "
          f"(2023 OOS=${best_cond['per_year'].get(2023, float('nan')):,.0f})")
    print(f"  best-cond DSR clears 0.95? {clears_dsr} (DSR={dsr_best['dsr']:.3f})")
    if improves and fixes_2023 and clears_dsr:
        print("  => YES (provisional) — see report for the exact config.")
    else:
        print("  => NO — forecast-conditioning does not robustly rescue the carry.")


def _write_report(report, results_by_band, K, dsr_uncond, dsr_best,
                  uncond, best_cond, primary_band):
    L = report
    L.append("\n## Method\n")
    L.append("- Engine: `evaluate_realistic(hold='expiry', real_bs_delta=True, "
             "fill_tol_minutes=60, spread_source='tape', delta_hedge_band=B)`.\n")
    L.append("- Forecast features joined on entry timestamp from "
             "`candidate_table.parquet`: `forecast_edge_dvol = rv_pred_pct - "
             "dvol_close` (entry-time HAR-PD 15-min forecast minus exchange DVOL), "
             "`dvol_pct_1y` (trailing-1y DVOL percentile = crisis marker).\n")
    L.append("- **Walk-forward**: for OOS year Y, the threshold/param maximizing the "
             "training-span (years<Y) NW Sharpe is applied to year Y. 2023 has no "
             "prior covered year -> fixed first-principles default (not tuned). "
             "Headline = concatenated OOS series.\n")
    L.append(f"- DSR penalty K = {len(BANDS_TRIED)} bands + forecast families = **{K}**.\n")

    for band, r in results_by_band.items():
        L.append(f"\n## Band {band:.2f}\n")
        L.append(f"_forecast_edge_dvol coverage on covered trades: {r['fe_cov']:.1%}; "
                 f"dvol_pct_1y coverage: {r['dp_cov']:.1%}_\n")
        L.append("\n| strategy | n | total $ | NW Sharpe | nonov Sharpe | MaxDD $ | "
                 "+years | per-year |\n")
        L.append("|---|---|---|---|---|---|---|---|\n")
        for m in r["families"]:
            L.append(f"| {m['name']} | {m['n']} | {m['total']:,.0f} | "
                     f"{m['nw_sharpe']:.3f} | {m['nonov_sharpe']:.3f} | "
                     f"{m['maxdd']:,.0f} | {m['pos_years']}/{m['tot_years']} | "
                     f"{fmt_year(m['per_year'])} |\n")
        gate = r.get("gate", {})
        L.append(f"\ngate_no_fake_profit: **{gate.get('passed')}** — {gate.get('detail')}\n")
        thin = r.get("thinning_null", {})
        if thin:
            L.append(f"\n**Random-thinning null** (randomly keep {thin.get('n_keep')} of "
                     f"{thin.get('n_universe')} conditionable trades, 300 draws): NW Sharpe "
                     f"mean {thin.get('mean'):.3f}, p95 {thin.get('p95'):.3f}, max "
                     f"{thin.get('max'):.3f}. The entry-filter's lift must clear this null "
                     f"(thinning an overlapping book mechanically raises overlap-adj Sharpe).\n")
        L.append("\nWalk-forward chosen params per fold:\n")
        L.append("```\n" + json.dumps(r["chosen"], indent=1, default=str) + "\n```\n")

    L.append(f"\n## DSR (deflated Sharpe, K={K})\n")
    L.append(f"- Unconditional (band {primary_band}): per-obs daily Sharpe "
             f"{uncond['sr_perobs_daily']:.4f} over {uncond['n_days']} days -> "
             f"**DSR {dsr_uncond['dsr']:.3f}** (null bench per-obs {dsr_uncond.get('sr0_perobs', float('nan')):.4f}).\n")
    L.append(f"- Best forecast-conditioned ({best_cond['name']}): per-obs daily "
             f"Sharpe {best_cond['sr_perobs_daily']:.4f} over {best_cond['n_days']} "
             f"days -> **DSR {dsr_best['dsr']:.3f}**.\n")

    improves = best_cond["nw_sharpe"] > uncond["nw_sharpe"]
    fixes_2023 = best_cond["per_year"].get(2023, -1) >= 0 or 2023 not in best_cond["per_year"]
    clears_dsr = np.isfinite(dsr_best["dsr"]) and dsr_best["dsr"] > 0.95
    verdict = "YES" if (improves and fixes_2023 and clears_dsr) else "NO"
    L.append("\n## VERDICT\n")
    outcome = (
        "meets this historical script's candidate thresholds"
        if verdict == "YES"
        else "does not robustly rescue the carry"
    )
    L.append(f"**{verdict}** — forecast-conditioning {outcome}. ")
    L.append("This does not override the canonical S3 negative status.\n\n")
    L.append(f"- improves overlap NW Sharpe vs unconditional: {improves} "
             f"({best_cond['nw_sharpe']:.3f} vs {uncond['nw_sharpe']:.3f})\n")
    L.append(f"- 2023 handled (flipped non-negative or skipped): {fixes_2023} "
             f"(2023 OOS = ${best_cond['per_year'].get(2023, float('nan')):,.0f})\n")
    L.append(f"- best-cond DSR clears 0.95 bar: {clears_dsr} ({dsr_best['dsr']:.3f})\n")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("".join(L))


if __name__ == "__main__":
    main()
