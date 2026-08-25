"""EXTRA — is the SHORT-only carry robust across bands, and DSR-honest?

The signal-side decomposition showed SHORT-only = +$856K (NW 6.857, 5/5 +years)
vs LONG-only = -$516K. Adversarial follow-ups:
  1. Splitting by side is a researcher DOF. Recompute SHORT-only across ALL bands
     {0.20,0.35,0.75} -> is the short carry band-stable (not a band-tune artifact)?
  2. Honest DSR on the SHORT-only nonoverlap t-stat with an INFLATED config count
     K (bands x sides = 12, or 20) -> does it clear unconditionally?
  3. SHORT-only un-hedged vs hedged: is the hedge the source of profit or is it the
     raw premium carry? (raw premium carry surviving = the edge is the VRP, not a
     gamma/marking artifact.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.longhold_hedged_vrp import _newey_west_sharpe, _nonoverlap_sharpe, _sharpe_iid

BANDS = [0.20, 0.35, 0.75]


def expected_max_sharpe(K, var_sr):
    g = 0.5772156649
    z1 = norm.ppf(1 - 1.0 / K)
    z2 = norm.ppf(1 - 1.0 / (K * np.e))
    return np.sqrt(var_sr) * ((1 - g) * z1 + g * z2)


def main():
    for b in BANDS:
        f = pd.read_parquet(f"vol_arb/output/verify_frame_band{int(b*100):02d}.parquet")
        cov = f[f["premium_tier"] > 0].copy()
        s = cov["signal"].to_numpy(float)
        sh = cov[s < 0].sort_index()
        h = sh["total_hedged_usd"].to_numpy(float)
        u = sh["total_unhedged_usd"].to_numpy(float)
        e = pd.DatetimeIndex(pd.to_datetime(sh["entry_ts"], utc=True))
        x = pd.DatetimeIndex(pd.to_datetime(sh["expiry_dt"].values, utc=True))
        nw = _newey_west_sharpe(h, e, x)
        nonov, nkeep = _nonoverlap_sharpe(h, e, x)
        yr = e.year
        py = {int(y): float(np.nansum(h[yr == y])) for y in sorted(set(yr))}
        posyrs = sum(1 for v in py.values() if v > 0)
        print(f"BAND {b:.2f} SHORT-ONLY n={len(sh)}: "
              f"hedged=${np.nansum(h):,.0f}  unhedged=${np.nansum(u):,.0f}  "
              f"NW={nw:.2f}  nonov={nonov:.2f}(nk={nkeep})  {posyrs}/{len(py)}+yrs")

    # DSR on band=0.35 short-only
    f = pd.read_parquet("vol_arb/output/verify_frame_band35.parquet")
    cov = f[f["premium_tier"] > 0].copy()
    s = cov["signal"].to_numpy(float)
    sh = cov[s < 0].sort_index()
    h = sh["total_hedged_usd"].to_numpy(float)
    e = pd.DatetimeIndex(pd.to_datetime(sh["entry_ts"], utc=True))
    x = pd.DatetimeIndex(pd.to_datetime(sh["expiry_dt"].values, utc=True))
    nonov, nkeep = _nonoverlap_sharpe(h, e, x)
    sr_perobs = nonov / np.sqrt(nkeep)
    print(f"\nSHORT-ONLY band=0.35 nonov t={nonov:.3f} over nkeep={nkeep} "
          f"-> per-obs SR={sr_perobs:.4f}")
    for K in (12, 20, 40):
        var_sr = (1.0 + 0.5 * sr_perobs ** 2) / nkeep
        sr0 = expected_max_sharpe(K, var_sr)
        sd = np.sqrt(var_sr)
        t_hurdle = sr0 / sd
        dsr = norm.cdf((sr_perobs - sr0) / sd)
        print(f"  K={K}: DSR t-hurdle={t_hurdle:.3f}  best nonov t={nonov:.3f}  "
              f"DSR p={dsr:.4f}  {'CLEARS' if nonov > t_hurdle else 'FAILS'}")
        bonf = norm.ppf(1 - 0.05 / (2 * K))
        print(f"        Bonferroni z(5%, K={K})={bonf:.3f}  "
              f"{'CLEARS' if nonov > bonf else 'FAILS'} (nonov)  "
              f"{'CLEARS' if 6.857 > bonf else 'FAILS'} (NW=6.857)")


if __name__ == "__main__":
    main()
