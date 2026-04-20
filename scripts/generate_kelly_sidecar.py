"""Generate Kelly-shrinkage per-bar multiplier sidecar parquet for V2.

Phase 2 Step 1: reuse the walk-forward multipliers computed in
``scripts/phase1_kelly_shrinkage_offline.py`` and save them as a parquet
aligned to the trade log's bar index. The backtester loads this sidecar
at run time when ``BacktestConfig.kelly_mult_path`` is set.

Saves one sidecar per pre-committed variant:
  - vol_arb/output/BTC_v2_kelly_mult_K1.parquet  (continuous Kelly-shrunk)
  - vol_arb/output/BTC_v2_kelly_mult_K2.parquet  (Q4+Q5 only)
  - vol_arb/output/BTC_v2_kelly_mult_K3.parquet  (Q5 fixed 1.5x)

Multipliers are 1.0 for shorts, zero-signal bars, and longs that don't
meet the variant's trigger condition. Always >= 1.0, <= 2.0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from phase1_kelly_shrinkage_offline import (  # noqa: E402
    CAL_MONTHS, TRAIL_MONTHS_MEDIAN, TRAIL_MONTHS_QUINTILE,
    compute_sigma2_trailing, compute_ntrail_longs,
    compute_kelly_raw_and_shrunk,
    compute_trailing_long_median,
    compute_trailing_long_quintile_bounds,
    size_mult_K1, size_mult_K2, size_mult_K3,
)

TRADE_LOG = "vol_arb/output/BTC_v2_trade_log.parquet"
PREDS = "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"


def main():
    tl = pd.read_parquet(TRADE_LOG)
    preds = pd.read_parquet(PREDS)
    print(f"Trade log: {tl.shape}, preds: {preds.shape}")

    sigma2 = compute_sigma2_trailing(preds, cal_months=CAL_MONTHS)
    n_trail = compute_ntrail_longs(tl, trail_months=TRAIL_MONTHS_MEDIAN)
    tl_k = compute_kelly_raw_and_shrunk(tl, sigma2, n_trail)
    median_fsh = compute_trailing_long_median(tl_k, TRAIL_MONTHS_MEDIAN)
    qbounds = compute_trailing_long_quintile_bounds(tl_k, TRAIL_MONTHS_QUINTILE)

    for variant, fn in [
        ("K1", lambda: size_mult_K1(tl_k, median_fsh)),
        ("K2", lambda: size_mult_K2(tl_k, median_fsh, qbounds)),
        ("K3", lambda: size_mult_K3(tl_k, qbounds)),
    ]:
        mult = fn()
        # Sanity checks
        assert np.all(mult >= 1.0 - 1e-9), f"{variant}: floor violated"
        assert np.all(mult <= 2.0 + 1e-9), f"{variant}: cap violated"
        n_boost = int((mult > 1.0 + 1e-9).sum())
        print(f"{variant}: {n_boost} boosted, mean={mult.mean():.4f}, max={mult.max():.4f}")

        out_path = f"vol_arb/output/BTC_v2_kelly_mult_{variant}.parquet"
        out = pd.DataFrame({"size_mult": mult}, index=tl_k.index)
        out.to_parquet(out_path)
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
