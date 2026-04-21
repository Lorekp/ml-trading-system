"""Generate Kelly-shrinkage per-bar multiplier sidecar parquet for V2.

Phase 2 Step 1: reuse the walk-forward multipliers computed in
``scripts/phase1_kelly_shrinkage_offline.py`` and save them as a parquet
aligned to the trade log's bar index. The backtester loads this sidecar
at run time when ``BacktestConfig.kelly_mult_path`` is set.

Saves one sidecar per pre-committed variant at the horizon-tagged path:
  - vol_arb/output/BTC_v2_kelly_mult_K1[_h{N}m].parquet  (continuous Kelly-shrunk)
  - vol_arb/output/BTC_v2_kelly_mult_K2[_h{N}m].parquet  (Q4+Q5 only)
  - vol_arb/output/BTC_v2_kelly_mult_K3[_h{N}m].parquet  (Q5 fixed 1.5x)

Multipliers are 1.0 for shorts, zero-signal bars, and longs that don't
meet the variant's trigger condition. Always >= 1.0, <= 2.0.

Default inputs / outputs resolve to the h=15 production files so
pre-horizon callers stay byte-identical. Pass ``--horizon-minutes N``
to regenerate sidecars at other horizons (requires trade log and
aligned-bias predictions to already exist at that horizon).
"""
from __future__ import annotations

import argparse
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

# Path helpers live at repo-root/vol_arb; make sure we can import them.
sys.path.insert(0, str(Path(__file__).parent.parent))
from vol_arb.paths import (  # noqa: E402
    har_pd_predictions_path, kelly_sidecar_path, trade_log_path,
)


def main():
    ap = argparse.ArgumentParser(
        description="Generate Kelly-shrinkage sidecar parquets (K1/K2/K3)."
    )
    ap.add_argument("--horizon-minutes", type=int, default=15,
                    help="Bar size in minutes; drives input trade-log + "
                         "aligned-bias predictions paths and output "
                         "kelly sidecar paths. Default 15 (legacy).")
    ap.add_argument("--trade-log", type=str, default="",
                    help="Override trade-log parquet path.")
    ap.add_argument("--preds-path", type=str, default="",
                    help="Override aligned-bias predictions parquet path.")
    ap.add_argument("--currency", type=str, default="BTC",
                    help="Currency for output sidecar naming (default BTC).")
    ap.add_argument("--asset", type=str, default="BTCUSDT",
                    help="Asset for predictions path (default BTCUSDT).")
    args = ap.parse_args()

    h = args.horizon_minutes
    if 1440 % h != 0:
        ap.error(f"--horizon-minutes={h} must divide 1440 (e.g., 15, 30, 60).")

    trade_log_str = args.trade_log or trade_log_path(
        args.currency, h, variant="v2",
    ).as_posix()
    preds_str = args.preds_path or har_pd_predictions_path(
        args.asset, h, aligned_bias=True,
    ).as_posix()

    tl = pd.read_parquet(trade_log_str)
    preds = pd.read_parquet(preds_str)
    print(f"Trade log: {trade_log_str}  shape={tl.shape}")
    print(f"Preds:     {preds_str}  shape={preds.shape}")
    print(f"Horizon:   {h}m")

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

        out_path = kelly_sidecar_path(args.currency, h, variant=variant).as_posix()
        out = pd.DataFrame({"size_mult": mult}, index=tl_k.index)
        out.to_parquet(out_path)
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
