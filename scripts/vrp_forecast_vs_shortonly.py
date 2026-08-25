"""Does the HAR-PD forecast filter add value OVER simple short-only?

The forecast-conditioning agent compared against the BLENDED unconditional book
(NW 1.89). But short-only unconditional is already NW 6.86 / nonov 7.74 / 5-5 years.
Per the historical anti-overfit protocol (prefer the simplest candidate unless it beats
the null on reserved years AND beats random thinning), the correct test is:

   short-only + forecast filter   vs   short-only alone   vs   random-thinning null

All on the TRUE held-out years (2024-2026); the forecast threshold is chosen on the
TRAIN years (2022-2023) only. If the filter does not robustly beat BOTH nulls OOS,
historically prefer the simpler unconditional short-only and drop the forecast layer.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from scripts.longhold_hedged_vrp import _sharpe_iid, _newey_west_sharpe, _nonoverlap_sharpe, _max_drawdown

BANDS = {0.20: "verify_frame_band20.parquet", 0.35: "verify_frame_band35.parquet",
         0.75: "verify_frame_band75.parquet"}
DIR = "vol_arb/output/"
TRAIN = {2022, 2023}
RNG = np.random.RandomState(0)


def load_short(fn):
    f = pd.read_parquet(DIR + fn)
    f = f[(f["premium_tier"] > 0) & (f["signal"] < 0)].copy()
    f["entry_ts"] = pd.to_datetime(f["entry_ts"], utc=True)
    f["exit_ts"] = pd.to_datetime(f["expiry_dt"], utc=True)
    f = f.sort_values("entry_ts").reset_index(drop=True)
    f["year"] = f["entry_ts"].dt.year
    return f


def met(d):
    pnl = d["total_hedged_usd"].to_numpy(float)
    e = pd.DatetimeIndex(d["entry_ts"]); x = pd.DatetimeIndex(d["exit_ts"])
    nonov, nk = _nonoverlap_sharpe(pnl, e, x)
    return dict(n=len(d), total=float(np.nansum(pnl)), nw=_newey_west_sharpe(pnl, e, x),
                nonov=nonov, nk=nk, maxdd=_max_drawdown(pnl),
                pos_years=sum(1 for y in sorted(d.year.unique())
                              if d.loc[d.year == y, "total_hedged_usd"].sum() > 0),
                n_years=d.year.nunique())


def thinning_null(d_oos, keep_n, iters=2000):
    """Random-thin the OOS short-only book to keep_n trades; return NW Sharpe dist.
    This is the honest null: thinning an overlapping book mechanically lifts Sharpe."""
    pnl = d_oos["total_hedged_usd"].to_numpy(float)
    e = pd.DatetimeIndex(d_oos["entry_ts"]); x = pd.DatetimeIndex(d_oos["exit_ts"])
    out = []
    idx = np.arange(len(d_oos))
    for _ in range(iters):
        sel = np.sort(RNG.choice(idx, size=keep_n, replace=False))
        out.append(_newey_west_sharpe(pnl[sel], e[sel], x[sel]))
    out = np.array([o for o in out if np.isfinite(o)])
    return out


print("Q: does forecast-filtering BEAT simple short-only (the real null), OOS 2024-2026?\n")
for b, fn in BANDS.items():
    f = load_short(fn)
    if "rv_pred_pct" not in f.columns:
        print(f"band {b}: rv_pred_pct missing"); continue
    # forecast covered subset (some trades lack a forecast)
    f = f[np.isfinite(f["rv_pred_pct"])].copy()
    tr = f[f.year.isin(TRAIN)]
    oos = f[~f.year.isin(TRAIN)]

    # Threshold chosen on TRAIN only: keep low-forecast-vol shorts. Sweep train
    # quantiles, pick the one maximizing TRAIN nonov Sharpe (forecast's own rule).
    best_q, best_s = None, -np.inf
    for q in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        thr = tr["rv_pred_pct"].quantile(q)
        sub = tr[tr["rv_pred_pct"] <= thr]
        if len(sub) < 10:
            continue
        s = _nonoverlap_sharpe(sub["total_hedged_usd"].to_numpy(float),
                               pd.DatetimeIndex(sub["entry_ts"]), pd.DatetimeIndex(sub["exit_ts"]))[0]
        if np.isfinite(s) and s > best_s:
            best_s, best_q = s, q
    thr = tr["rv_pred_pct"].quantile(best_q)        # threshold frozen from TRAIN

    base = met(oos)                                  # short-only unconditional OOS
    filt = met(oos[oos["rv_pred_pct"] <= thr])       # short-only + forecast filter OOS

    null = thinning_null(oos, filt["n"])             # random thin to filter's n
    null_p95 = np.percentile(null, 95); null_med = np.percentile(null, 50)
    beats_null = filt["nw"] > null_p95

    print("=" * 78)
    print(f"BAND {b:.2f}  (train q*={best_q}, thr rv_pred_pct<={thr:.1f})")
    print(f"  short-only UNCOND OOS : n={base['n']}  ${base['total']:,.0f}  "
          f"NW {base['nw']:.2f}  nonov {base['nonov']:.2f}  MaxDD ${base['maxdd']:,.0f}  "
          f"{base['pos_years']}/{base['n_years']}yr+")
    print(f"  short-only +FORECAST  : n={filt['n']}  ${filt['total']:,.0f}  "
          f"NW {filt['nw']:.2f}  nonov {filt['nonov']:.2f}  MaxDD ${filt['maxdd']:,.0f}  "
          f"{filt['pos_years']}/{filt['n_years']}yr+")
    print(f"  random-thinning NULL  : NW median {null_med:.2f}, p95 {null_p95:.2f}  "
          f"(thinned to n={filt['n']})")
    print(f"  --> filter NW {filt['nw']:.2f} vs uncond {base['nw']:.2f} vs thin-null-p95 {null_p95:.2f}  "
          f"==> {'BEATS both' if (filt['nw']>base['nw'] and beats_null) else 'does NOT beat both'}")

print("\nHISTORICAL RULE: retain the forecast layer only if it beats BOTH the unconditional "
      "short-only AND the thinning-null p95 OOS. This does not validate either strategy.")
