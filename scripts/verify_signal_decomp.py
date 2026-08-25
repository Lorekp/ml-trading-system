"""EXTRA — decompose the band=0.35 hedged carry by SIGNAL SIDE across all years.

The 'short-vol VRP carry' claim implies the SHORT straddles drive the profit. But
the covered set is 666 short + 279 long. If the long straddles contribute a large
share (or the carry is actually a directional long-vol bull-market artifact), the
'VRP carry' framing is wrong. Quantify per-year per-side, plus an overlap-aware
t-stat on the SHORT-ONLY subset (the genuine VRP carry).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.longhold_hedged_vrp import _newey_west_sharpe, _nonoverlap_sharpe, _sharpe_iid

BAND = 0.35


def main():
    f = pd.read_parquet(f"vol_arb/output/verify_frame_band{int(BAND*100):02d}.parquet")
    cov = f[f["premium_tier"] > 0].copy()
    cov["entry_ts"] = pd.to_datetime(cov["entry_ts"], utc=True)
    cov["year"] = cov["entry_ts"].dt.year
    h = cov["total_hedged_usd"].to_numpy(float)
    s = cov["signal"].to_numpy(float)

    print(f"covered n={len(cov)}  short={int((s<0).sum())}  long={int((s>0).sum())}")
    print(f"TOTAL hedged = ${np.nansum(h):,.0f}")
    print(f"  SHORT subset total = ${np.nansum(h[s<0]):,.0f}  mean=${np.nanmean(h[s<0]):.1f}/tr")
    print(f"  LONG  subset total = ${np.nansum(h[s>0]):,.0f}  mean=${np.nanmean(h[s>0]):.1f}/tr")

    print("\nPER-YEAR x SIDE hedged:")
    for y, g in cov.groupby("year"):
        hs = g["total_hedged_usd"].to_numpy(float)
        sg = g["signal"].to_numpy(float)
        print(f"  {y}: SHORT ${np.nansum(hs[sg<0]):>10,.0f} (n={int((sg<0).sum()):>3})   "
              f"LONG ${np.nansum(hs[sg>0]):>10,.0f} (n={int((sg>0).sum()):>3})")

    # Overlap-aware Sharpe on SHORT-ONLY (the actual VRP carry)
    sh = cov[s < 0].sort_index()
    hsh = sh["total_hedged_usd"].to_numpy(float)
    e = pd.DatetimeIndex(pd.to_datetime(sh.index if sh.index.name else sh["entry_ts"], utc=True))
    e = pd.DatetimeIndex(pd.to_datetime(sh["entry_ts"], utc=True))
    x = pd.DatetimeIndex(pd.to_datetime(sh["expiry_dt"].values, utc=True))
    nw = _newey_west_sharpe(hsh, e, x)
    nonov, nkeep = _nonoverlap_sharpe(hsh, e, x)
    print(f"\nSHORT-ONLY overlap-aware: naive {_sharpe_iid(hsh):.3f}  NW {nw:.3f}  "
          f"nonov {nonov:.3f} (n_keep={nkeep})")
    yr = e.year
    py = {int(y): float(np.nansum(hsh[yr == y])) for y in sorted(set(yr))}
    print("SHORT-ONLY per-year:", {y: round(v) for y, v in py.items()},
          f"  ({sum(1 for v in py.values() if v>0)}/{len(py)} +)")

    # LONG-ONLY for completeness
    lo = cov[s > 0].sort_index()
    hlo = lo["total_hedged_usd"].to_numpy(float)
    el = pd.DatetimeIndex(pd.to_datetime(lo["entry_ts"], utc=True))
    xl = pd.DatetimeIndex(pd.to_datetime(lo["expiry_dt"].values, utc=True))
    print(f"\nLONG-ONLY overlap-aware: naive {_sharpe_iid(hlo):.3f}  "
          f"NW {_newey_west_sharpe(hlo, el, xl):.3f}  "
          f"nonov {_nonoverlap_sharpe(hlo, el, xl)[0]:.3f}")


if __name__ == "__main__":
    main()
