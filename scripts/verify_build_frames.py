"""ADVERSARIAL VERIFICATION — build per-trade eval frames once, cache to parquet.

Runs evaluate_realistic at hold='expiry', real_bs_delta=True, fill_tol=60, tape
spread, for bands {0.20, 0.35, 0.75}. Dumps each result trade_log to parquet so
the individual attack scripts can read cached per-trade columns without re-running
the (slow) path eval. Verification-only; no engine edits.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic

TRADE_LOG = "vol_arb/output/BTC_v2_trade_log.parquet"
BANDS = [0.20, 0.35, 0.75]
OUT = "vol_arb/output"


def main():
    tl = pd.read_parquet(TRADE_LOG)
    for b in BANDS:
        t0 = time.time()
        cfg = RealisticEvalConfig(
            hold="expiry", real_bs_delta=True, fill_tol_minutes=60.0,
            spread_source="tape", delta_hedge_band=b,
        )
        res = evaluate_realistic(tl, cfg)
        f = res.trade_log
        # Persist the per-trade frame (reset index so timestamp survives parquet).
        out = f.copy()
        out.index.name = "entry_ts"
        out = out.reset_index()
        path = f"{OUT}/verify_frame_band{int(b*100):02d}.parquet"
        out.to_parquet(path)
        cov = f[f["premium_tier"] > 0]
        print(f"band={b:.2f}  n_fired={res.n_fired}  n_cov={res.n_covered} "
              f"({res.coverage_frac:.2%})  hedged=${res.hedged_total_usd:,.0f}  "
              f"unhedged=${res.unhedged_total_usd:,.0f}  "
              f"opt_prem=${res.option_premium_total_usd:,.0f}  "
              f"opt_fee=${res.option_fee_total_usd:,.0f}  "
              f"[{time.time()-t0:.1f}s] -> {path}")


if __name__ == "__main__":
    main()
