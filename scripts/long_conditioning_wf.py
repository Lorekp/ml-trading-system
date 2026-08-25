"""Conditioned-LONG straddle: walk-forward threshold selection vs THREE nulls.

Thesis: go LONG only when a signal predicts realized > implied (vol spike/jump).
Universe = FULL fired set forced signal=+1 (the truly unconditional long null),
evaluated on the realistic Deribit engine at a given horizon + hedge band.

For each candidate conditioning signal s (entry-time, causal) and rule "keep trade
iff s >= thr": choose thr on TRAIN years to maximise train objective, apply to
HELD-OUT TEST years. Judge OOS total PnL + non-overlap Sharpe vs:
  (i)   unconditional long OOS (the structural null)
  (ii)  random-thinning null at the SAME OOS trade count (1000 draws)
  (iii) shuffled-label placebo: same thr-selection pipeline on a signal whose
        values are permuted across trades (kills any real signal-PnL link).

Reports per-year, MaxDD, gate_no_fake_profit, DSR with K = all combos tried.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic
from vol_arb.honest_pnl import gate_no_fake_profit
from scripts.longhold_hedged_vrp import (
    _sharpe_iid, _newey_west_sharpe, _nonoverlap_sharpe, _max_drawdown,
)

TRADE_LOG = "vol_arb/output/BTC_v2_trade_log.parquet"
HOLD_MIN = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}

# Candidate conditioning signals (entry-time, causal). All are "higher => more
# likely realized>implied / a move", so rule is keep iff s >= thr.
SIGNALS = ["edge", "rv_dvol_spread", "rv_iv_spread", "rv_pred_pct",
           "jump_fraction", "down_jump_fraction", "up_jump_fraction"]

RNG = np.random.default_rng(12345)


def build_forced_long(tl):
    out = tl.copy()
    out["signal"] = np.where(out["signal"].to_numpy() != 0, 1.0, 0.0)
    return out


def eval_horizon(tl_long, hold, band):
    cfg = RealisticEvalConfig(hold=hold, real_bs_delta=True, fill_tol_minutes=60.0,
                              spread_source="tape", delta_hedge_band=band)
    res = evaluate_realistic(tl_long, cfg)
    f = res.trade_log
    cov = f[f["premium_tier"] > 0].copy().sort_index()
    return res, f, cov


def make_exit_ts(cov, hold):
    entry_ts = pd.DatetimeIndex(pd.to_datetime(cov.index, utc=True))
    if hold == "expiry":
        exit_ts = pd.DatetimeIndex(pd.to_datetime(cov["expiry_dt"].values, utc=True))
    else:
        m = HOLD_MIN[hold]
        exit_ts = entry_ts + pd.Timedelta(minutes=m)
        expiry = pd.DatetimeIndex(pd.to_datetime(cov["expiry_dt"].values, utc=True))
        exit_ts = pd.DatetimeIndex(np.where(exit_ts < expiry, exit_ts,
                                            expiry - pd.Timedelta(minutes=1)))
    return entry_ts, exit_ts


def stats(pnl, e, x):
    pnl = np.asarray(pnl, float)
    m = np.isfinite(pnl)
    pnl, e, x = pnl[m], e[m], x[m]
    if len(pnl) == 0:
        return dict(n=0, total=0.0, nonov=float("nan"), nw=float("nan"),
                    maxdd=0.0, n_nonov=0)
    nonov, n_nonov = _nonoverlap_sharpe(pnl, e, x)
    return dict(n=len(pnl), total=float(np.nansum(pnl)),
                nonov=nonov, nw=_newey_west_sharpe(pnl, e, x),
                maxdd=_max_drawdown(pnl), n_nonov=n_nonov)


def per_year(pnl, e):
    yr = e.year
    return {int(y): float(np.nansum(np.asarray(pnl)[yr == y])) for y in sorted(set(yr))}


def main():
    hold = sys.argv[1] if len(sys.argv) > 1 else "4h"
    band = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
    metric_col = "total_hedged_usd"  # gamma-scalp; also report unhedged
    print("=" * 80)
    print(f"CONDITIONED LONG — hold={hold} band={band}  metric={metric_col}")
    print("=" * 80)

    tl = pd.read_parquet(TRADE_LOG)
    tl_long = build_forced_long(tl)
    res, f, cov = eval_horizon(tl_long, hold, band)
    print(f"n_fired={res.n_fired}  coverage={res.coverage_frac:.2%}  covered_long={len(cov)}")

    entry_ts, exit_ts = make_exit_ts(cov, hold)
    cov = cov.assign(_entry=entry_ts, _exit=exit_ts)
    cov["_year"] = entry_ts.year
    pnl_h = cov["total_hedged_usd"].to_numpy(float)
    pnl_u = cov["total_unhedged_usd"].to_numpy(float)

    # ---- UNCONDITIONAL NULL (full covered set) ----
    uh = stats(pnl_h, entry_ts, exit_ts)
    uu = stats(pnl_u, entry_ts, exit_ts)
    print(f"\n[UNCONDITIONAL LONG full set]")
    print(f"  hedged   total=${uh['total']:,.0f}  nonov_S={uh['nonov']:.3f}  n={uh['n']}  MaxDD=${uh['maxdd']:,.0f}")
    print(f"  unhedged total=${uu['total']:,.0f}  nonov_S={uu['nonov']:.3f}")
    print(f"  per-year hedged: {per_year(pnl_h, entry_ts)}")

    # gate on full set
    try:
        gr = gate_no_fake_profit(cov, model_col="total_hedged_usd", real_col="real_option_pnl_nc")
        print(f"  gate_no_fake_profit(full)={gr.passed}")
    except Exception as ex:
        print(f"  gate err: {ex}")

    # ---- Train/Test split: TRAIN 2023-2024, TEST 2025-2026 ----
    yr = entry_ts.year
    train_mask = (yr <= 2024)
    test_mask = (yr >= 2025)
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    print(f"\nTRAIN(<=2024) n={n_train}   TEST(>=2025) n={n_test}")

    # candidate thresholds = quantiles of the TRAIN signal (choose-insensitivity-friendly)
    QS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    K_combos = 0  # multiple-testing count: signal x threshold x (this hold,band)
    results = []
    for sig in SIGNALS:
        if sig not in cov.columns:
            continue
        sv = cov[sig].to_numpy(float)
        valid = np.isfinite(sv)
        # TRAIN selection: maximise train total PnL (hedged) subject to >= 30 trades
        best = None
        tr_sv = sv[train_mask & valid]
        if len(tr_sv) < 30:
            continue
        for q in QS:
            K_combos += 1
            thr = np.quantile(tr_sv, q)
            keep_tr = train_mask & valid & (sv >= thr)
            if keep_tr.sum() < 30:
                continue
            tr_total = float(np.nansum(pnl_h[keep_tr]))
            tr_e = entry_ts[keep_tr]; tr_x = exit_ts[keep_tr]
            tr_nonov, _ = _nonoverlap_sharpe(pnl_h[keep_tr], tr_e, tr_x)
            score = tr_total  # objective = train hedged PnL
            if best is None or score > best["score"]:
                best = dict(q=q, thr=thr, score=score, tr_total=tr_total, tr_nonov=tr_nonov)
        if best is None:
            continue
        thr = best["thr"]
        keep_te = test_mask & valid & (sv >= thr)
        te = stats(pnl_h[keep_te], entry_ts[keep_te], exit_ts[keep_te])
        te_u = stats(pnl_u[keep_te], entry_ts[keep_te], exit_ts[keep_te])
        # unconditional TEST null
        te_uncond = stats(pnl_h[test_mask], entry_ts[test_mask], exit_ts[test_mask])
        # random-thinning null: 1000 draws of te['n'] from full TEST set
        idx_test = np.where(test_mask & valid)[0]
        n_keep = int(keep_te.sum())
        rand_totals = []
        if n_keep > 0 and len(idx_test) > n_keep:
            for _ in range(1000):
                pick = RNG.choice(idx_test, size=n_keep, replace=False)
                rand_totals.append(float(np.nansum(pnl_h[pick])))
            rand_totals = np.array(rand_totals)
            rand_mean = rand_totals.mean()
            rand_pct = float((rand_totals >= te["total"]).mean())  # P(random >= conditioned)
        else:
            rand_mean, rand_pct = float("nan"), float("nan")
        # shuffled-label placebo: permute signal over ALL trades, redo full pipeline
        plc_totals = []
        for _ in range(200):
            perm = RNG.permutation(np.where(valid)[0])
            sv_perm = sv.copy()
            sv_perm[valid] = sv[perm]
            tr_p = sv_perm[train_mask & valid]
            bestp = None
            for q in QS:
                thrp = np.quantile(tr_p, q)
                ktr = train_mask & valid & (sv_perm >= thrp)
                if ktr.sum() < 30:
                    continue
                tot = float(np.nansum(pnl_h[ktr]))
                if bestp is None or tot > bestp[0]:
                    bestp = (tot, thrp)
            if bestp is None:
                continue
            kte = test_mask & valid & (sv_perm >= bestp[1])
            plc_totals.append(float(np.nansum(pnl_h[kte])))
        plc_totals = np.array(plc_totals) if plc_totals else np.array([np.nan])
        plc_pct = float(np.nanmean(plc_totals >= te["total"]))
        results.append(dict(sig=sig, q=best["q"], thr=thr, tr_total=best["tr_total"],
                            tr_nonov=best["tr_nonov"], te=te, te_u=te_u,
                            te_uncond=te_uncond, rand_mean=rand_mean, rand_pct=rand_pct,
                            plc_mean=float(np.nanmean(plc_totals)), plc_pct=plc_pct,
                            n_keep=n_keep, py=per_year(pnl_h[keep_te], entry_ts[keep_te])))

    print(f"\n--- OOS RESULTS (TEST>=2025), K_combos this hold/band = {K_combos} ---")
    for r in results:
        te = r["te"]
        print(f"\nSIGNAL={r['sig']}  thr@q{r['q']:.1f}={r['thr']:.4g}  (train PnL ${r['tr_total']:,.0f}, train nonov_S {r['tr_nonov']:.2f})")
        print(f"  OOS hedged  total=${te['total']:,.0f}  nonov_S={te['nonov']:.3f}  n={te['n']}  MaxDD=${te['maxdd']:,.0f}")
        print(f"  OOS unhedged total=${r['te_u']['total']:,.0f}  nonov_S={r['te_u']['nonov']:.3f}")
        print(f"  NULL(i) uncond TEST: ${r['te_uncond']['total']:,.0f}  nonov_S={r['te_uncond']['nonov']:.3f}  n={r['te_uncond']['n']}")
        print(f"  NULL(ii) random-thin mean=${r['rand_mean']:,.0f}  P(rand>=cond)={r['rand_pct']:.3f}")
        print(f"  NULL(iii) placebo mean=${r['plc_mean']:,.0f}  P(placebo>=cond)={r['plc_pct']:.3f}")
        beats = (te['total'] > r['te_uncond']['total'] and r['rand_pct'] < 0.05 and r['plc_pct'] < 0.05 and te['total'] > 0)
        print(f"  per-year OOS: {r['py']}")
        print(f"  >>> BEATS ALL 3 NULLS & POSITIVE: {beats}")


if __name__ == "__main__":
    main()
