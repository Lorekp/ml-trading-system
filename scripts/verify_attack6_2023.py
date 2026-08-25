"""ATTACK 6 — diagnose the 2023 structural loss (negative at every band).

Single-trade blowup or a regime (realized >> implied)? Quantify per-trade, by
month, by signal side, and the RV-vs-IV relationship for 2023 covered trades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, PricePanel, _to_i8_ns

BAND = 0.35


def main():
    f = pd.read_parquet(f"vol_arb/output/verify_frame_band{int(BAND*100):02d}.parquet")
    cov = f[f["premium_tier"] > 0].copy()
    cov["entry_ts"] = pd.to_datetime(cov["entry_ts"], utc=True)
    cov["year"] = cov["entry_ts"].dt.year
    cov["month"] = cov["entry_ts"].dt.to_period("M").astype(str)
    hedged = cov["total_hedged_usd"].to_numpy(float)

    print("PER-YEAR hedged NC (band=0.35):")
    for y, g in cov.groupby("year"):
        h = g["total_hedged_usd"].to_numpy(float)
        print(f"  {y}: n={len(g):>4}  sum=${np.nansum(h):>12,.0f}  "
              f"mean=${np.nanmean(h):>8.1f}  worst=${np.nanmin(h):>10,.0f}  "
              f"best=${np.nanmax(h):>10,.0f}")

    c23 = cov[cov["year"] == 2023].copy()
    h23 = c23["total_hedged_usd"].to_numpy(float)
    print(f"\n=== 2023 deep dive: n={len(c23)}, total=${np.nansum(h23):,.0f} ===")
    print(f"  positive trades: {(h23>0).sum()}  negative: {(h23<0).sum()}")
    print(f"  worst single = ${np.nanmin(h23):,.0f}  best single = ${np.nanmax(h23):,.0f}")
    # is it one blowup? remove the worst trade and recompute
    order = np.argsort(h23)
    for k in (1, 3, 5):
        excl = np.nansum(h23) - np.nansum(np.sort(h23)[:k])
        print(f"  2023 total excluding {k} worst trades = ${excl:,.0f}")

    print("\n  2023 by MONTH:")
    for m, g in c23.groupby("month"):
        h = g["total_hedged_usd"].to_numpy(float)
        print(f"    {m}: n={len(g):>3}  sum=${np.nansum(h):>10,.0f}")

    print("\n  2023 by SIGNAL side:")
    for s, g in c23.groupby("signal"):
        h = g["total_hedged_usd"].to_numpy(float)
        side = "SHORT" if s < 0 else "LONG"
        print(f"    {side}: n={len(g):>4}  sum=${np.nansum(h):>10,.0f}  "
              f"mean=${np.nanmean(h):>8.1f}")

    # RV vs IV regime in 2023 (short straddle loses when RV>IV)
    cfg = RealisticEvalConfig(hold="expiry")
    panel = PricePanel.load(cfg, c23["entry_ts"].min(),
                            pd.to_datetime(c23["expiry_dt"], utc=True).max() + pd.Timedelta(days=1))
    entry_i8 = _to_i8_ns(pd.DatetimeIndex(c23["entry_ts"]))
    exit_i8 = _to_i8_ns(pd.DatetimeIndex(pd.to_datetime(c23["expiry_dt"], utc=True)))
    lo = np.searchsorted(panel.ts_i8, entry_i8, "left")
    hi = np.searchsorted(panel.ts_i8, exit_i8, "right")
    _YEAR_MIN = 365.25 * 24 * 60
    rv = np.full(len(c23), np.nan)
    for i in range(len(c23)):
        a, b = lo[i], hi[i]
        if b - a >= 3:
            r = np.diff(np.log(panel.close[a:b]))
            rv[i] = np.sqrt(np.mean(r ** 2) * _YEAR_MIN)
    iv = c23["iv_used"].to_numpy(float)
    iv = np.where(np.nanmedian(np.abs(iv)) > 5, iv / 100.0, iv)
    print(f"\n  2023 mean realized vol over hold = {np.nanmean(rv):.3f}")
    print(f"  2023 mean entry IV (decimal)      = {np.nanmean(iv):.3f}")
    print(f"  2023 mean VRP (IV-RV)             = {np.nanmean(iv-rv):+.4f}")
    print(f"  2023 frac(RV>IV)                  = {np.nanmean((rv>iv).astype(float)):.3f}")
    # compare to a profitable year (2025)
    c25 = cov[cov["year"] == 2025]
    print(f"\n  (ref) 2025 hedged total = ${np.nansum(c25['total_hedged_usd'].to_numpy(float)):,.0f}, "
          f"n={len(c25)}")


if __name__ == "__main__":
    main()
