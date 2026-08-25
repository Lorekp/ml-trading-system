"""ATTACK 1 — gate_no_fake_profit at historical candidate bands {0.20,0.35,0.75}.

Does total_hedged_usd agree in SIGN and correlate (per-trade) with the real-fill
ground truth real_option_pnl_nc? Report corr, signs, gate.passed.

Also: a SHARPER cross-check the gate itself does NOT do — compare total_hedged
against the UN-hedged real-fill option PnL on the SAME exit valuation. The gate's
real_option_pnl_nc marks the exit to a TAPE FILL near expiry; the eval's
total_unhedged uses the SETTLEMENT INTRINSIC. We report both and their corr so we
know whether 'agreement' is real or an artifact of two different exit marks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_arb.honest_pnl import gate_no_fake_profit

BANDS = [0.20, 0.35, 0.75]


def main():
    for b in BANDS:
        path = f"vol_arb/output/verify_frame_band{int(b*100):02d}.parquet"
        f = pd.read_parquet(path)
        cov = f[f["premium_tier"] > 0].copy()
        n = len(cov)

        hedged = pd.to_numeric(cov["total_hedged_usd"], errors="coerce").to_numpy()
        unhedged = pd.to_numeric(cov["total_unhedged_usd"], errors="coerce").to_numpy()
        realfill = pd.to_numeric(cov["real_option_pnl_nc"], errors="coerce").to_numpy()

        ok = np.isfinite(hedged) & np.isfinite(realfill)
        h, u, r = hedged[ok], unhedged[ok], realfill[ok]

        # The gate as the pipeline runs it (model_col=total_hedged_usd):
        gr = gate_no_fake_profit(f, model_col="total_hedged_usd",
                                 real_col="real_option_pnl_nc")

        corr_hr = float(np.corrcoef(h, r)[0, 1]) if h.std() > 0 and r.std() > 0 else float("nan")
        corr_ur = float(np.corrcoef(u, r)[0, 1]) if u.std() > 0 and r.std() > 0 else float("nan")
        # sign agreement per-trade
        sign_agree = float(np.mean(np.sign(h) == np.sign(r)))

        print("=" * 78)
        print(f"BAND {b:.2f}  n_covered={n}")
        print(f"  total_hedged_usd  sum = ${h.sum():,.0f}   mean=${h.mean():.1f}/tr")
        print(f"  total_unhedged    sum = ${u.sum():,.0f}   mean=${u.mean():.1f}/tr")
        print(f"  real_option_pnl_nc sum= ${r.sum():,.0f}   mean=${r.mean():.1f}/tr "
              f"(GROUND TRUTH, tape-fill exit near expiry)")
        print(f"  corr(hedged, real_fill)   = {corr_hr:.4f}")
        print(f"  corr(unhedged, real_fill) = {corr_ur:.4f}")
        print(f"  per-trade sign agreement(hedged vs real_fill) = {sign_agree:.3f}")
        print(f"  GATE passed = {gr.passed}")
        print(f"  GATE detail = {gr.detail}")
        # The key adversarial point: aggregate signs
        print(f"  >>> aggregate SIGN: hedged {'+' if h.sum()>0 else '-'}  "
              f"real_fill {'+' if r.sum()>0 else '-'}  "
              f"{'AGREE' if (h.sum()>0)==(r.sum()>0) else 'DISAGREE'}")


if __name__ == "__main__":
    main()
