"""ATTACK 3 — SELECTION BIAS: are the 945 fill-covered trades systematically
CALMER / cheaper / lower-realized-vol than the 11,282 uncovered ones?

If covered trades are systematically calm, a short-vol carry measured only on
them is a coverage artifact, not a deployable edge. Compare covered vs uncovered
on realized vol over the hold, entry IV/DVOL, vega, premium, calendar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, PricePanel, _to_i8_ns

BAND = 0.35


def realized_vol_over_hold(cov_idx_ts, expiry, index_price, panel):
    """Annualized realized vol of the real BTC path from entry to expiry per trade."""
    entry_i8 = _to_i8_ns(pd.DatetimeIndex(cov_idx_ts))
    exit_i8 = _to_i8_ns(pd.DatetimeIndex(expiry))
    lo = np.searchsorted(panel.ts_i8, entry_i8, "left")
    hi = np.searchsorted(panel.ts_i8, exit_i8, "right")
    out = np.full(len(entry_i8), np.nan)
    _YEAR_MIN = 365.25 * 24 * 60
    for i in range(len(entry_i8)):
        a, b = lo[i], hi[i]
        if b - a < 3:
            continue
        seg = panel.close[a:b]
        r = np.diff(np.log(seg))
        out[i] = np.sqrt(np.mean(r ** 2) * _YEAR_MIN)  # annualized
    return out


def summ(name, x):
    x = x[np.isfinite(x)]
    return (f"{name:>22}: n={len(x):>6}  mean={np.mean(x):>12.4f}  "
            f"med={np.median(x):>12.4f}  p25={np.percentile(x,25):>12.4f}  "
            f"p75={np.percentile(x,75):>12.4f}")


def main():
    f = pd.read_parquet(f"vol_arb/output/verify_frame_band{int(BAND*100):02d}.parquet")
    covered = f["premium_tier"].to_numpy() > 0
    cov = f[covered].copy()
    unc = f[~covered].copy()
    print(f"covered n={len(cov)}  uncovered n={len(unc)}  "
          f"coverage={covered.mean():.2%}")

    # Build panel over the FULL fired window to compute realized vol for BOTH groups
    entry_all = pd.to_datetime(f["entry_ts"], utc=True)
    expiry_all = pd.to_datetime(f["expiry_dt"], utc=True)
    cfg = RealisticEvalConfig(hold="expiry")
    panel = PricePanel.load(cfg, entry_all.min(), expiry_all.max() + pd.Timedelta(days=1))

    rv_cov = realized_vol_over_hold(cov["entry_ts"], cov["expiry_dt"],
                                    cov["index_price"].to_numpy(float), panel)
    rv_unc = realized_vol_over_hold(unc["entry_ts"], unc["expiry_dt"],
                                    unc["index_price"].to_numpy(float), panel)

    print("\n--- REALIZED VOL over hold (annualized) ---")
    print(summ("RV covered", rv_cov))
    print(summ("RV uncovered", rv_unc))

    def col(d, c):
        return pd.to_numeric(d[c], errors="coerce").to_numpy() if c in d else np.array([])

    for c in ["iv_used", "iv", "dvol_close", "vega", "mark_price", "T_years",
              "dte_days", "N_contracts_nc", "index_price", "rv_iv_spread",
              "rv_dvol_spread", "edge"]:
        if c in f.columns:
            print("\n--- " + c + " ---")
            print(summ(c + " covered", col(cov, c)))
            print(summ(c + " uncovered", col(unc, c)))

    # entry IV minus realized vol over hold (the VRP per trade) — the carry driver
    iv_cov = col(cov, "iv_used")
    iv_unc = col(unc, "iv_used")
    # iv_used is in vol-points (~49); RV is decimal -> normalize iv to decimal
    iv_cov_dec = np.where(np.nanmedian(np.abs(iv_cov)) > 5, iv_cov / 100.0, iv_cov)
    iv_unc_dec = np.where(np.nanmedian(np.abs(iv_unc)) > 5, iv_unc / 100.0, iv_unc)
    vrp_cov = iv_cov_dec - rv_cov
    vrp_unc = iv_unc_dec - rv_unc
    print("\n--- VRP per trade (entry IV[dec] - realized vol over hold) ---")
    print(summ("VRP covered", vrp_cov))
    print(summ("VRP uncovered", vrp_unc))
    print(f"  >>> covered VRP mean {np.nanmean(vrp_cov):+.4f} vs "
          f"uncovered {np.nanmean(vrp_unc):+.4f}  "
          f"(higher covered VRP = short-vol carry inflated by coverage)")

    # realized > implied fraction (short straddle LOSES when RV>IV)
    rv_gt_iv_cov = float(np.nanmean((rv_cov > iv_cov_dec).astype(float)))
    rv_gt_iv_unc = float(np.nanmean((rv_unc > iv_unc_dec).astype(float)))
    print(f"\nfrac(RV>IV) covered={rv_gt_iv_cov:.3f}  uncovered={rv_gt_iv_unc:.3f}  "
          "(short straddle loses when RV>IV; lower covered frac = calmer covered set)")

    # calendar distribution
    print("\n--- CALENDAR (entry year fractions) ---")
    yc = pd.to_datetime(cov["entry_ts"], utc=True).dt.year.value_counts(normalize=True).sort_index()
    yu = pd.to_datetime(unc["entry_ts"], utc=True).dt.year.value_counts(normalize=True).sort_index()
    cal = pd.DataFrame({"covered_frac": yc, "uncovered_frac": yu}).fillna(0.0)
    print(cal.round(3).to_string())

    # DTE distribution — short-dated contracts trade more (better coverage) and
    # have different vol behaviour
    print("\n--- DTE bucket fractions ---")
    def dte_buckets(d):
        dte = pd.to_numeric(d["dte_days"], errors="coerce")
        b = pd.cut(dte, [0, 1, 3, 7, 14, 30, 1000],
                   labels=["0-1", "1-3", "3-7", "7-14", "14-30", "30+"])
        return b.value_counts(normalize=True).sort_index()
    print(pd.DataFrame({"covered": dte_buckets(cov), "uncovered": dte_buckets(unc)})
          .fillna(0.0).round(3).to_string())


if __name__ == "__main__":
    main()
