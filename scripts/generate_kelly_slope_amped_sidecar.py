"""Generate the K1+slope walk-forward Kelly sidecar.

Composes on top of the existing K1 sidecar by multiplying the K1 size_mult
by `MULT` on bars where:

* `signal == 1`
* `svi_term_slope_1w_2w < threshold_M`

where `threshold_M` is the 20th-percentile of `svi_term_slope_1w_2w` over
LONG bars in the trailing window [M-24m, M-1m]. This is fully walk-forward
(no global threshold is used; each month's threshold uses only past data).

Justification: Phase 2 audit (2026-05-10) found long-side at extreme
backwardation has 2x per-trade Sharpe; signal stable across all 5 years;
Sharpe ordering preserved OOS; bootstrap p<0.0005 vs random null.
$/Sharpe-pt $204K (real backtest) beats K1 adoption template ($121K).

Output:

* `vol_arb/output/BTC_v2_kelly_mult_K1_slope.parquet` — new sidecar
  (matches K1 byte-identity for un-amplified bars; K1×{MULT} for the
  ~20% of long bars below the per-month threshold).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

K1_PATH = REPO_ROOT / "vol_arb" / "output" / "BTC_v2_kelly_mult_K1.parquet"
TRADE_LOG = REPO_ROOT / "vol_arb" / "output" / "BTC_v2_trade_log.parquet"
OUT_PATH = REPO_ROOT / "vol_arb" / "output" / "BTC_v2_kelly_mult_K1_slope.parquet"

TRAIL_MONTHS = 24
PERCENTILE = 0.20
MULT = 1.25
CAP = 2.5  # raise K1's 2.0 cap slightly to allow stacked amplification

WARMUP_MONTHS = TRAIL_MONTHS  # don't amplify before this many months of history


def main() -> int:
    print(f"[generate] Reading {K1_PATH.name}")
    k1 = pd.read_parquet(K1_PATH)
    if k1.index.tz is None:
        k1.index = pd.to_datetime(k1.index, utc=True)
    print(f"[generate]  K1: {len(k1):,} rows, mean={k1['size_mult'].mean():.4f}")

    print(f"[generate] Reading {TRADE_LOG.name}")
    tl = pd.read_parquet(TRADE_LOG)

    aligned = k1.join(
        tl[["signal", "svi_term_slope_1w_2w"]], how="left",
    )
    aligned["month"] = aligned.index.to_period("M")
    months = sorted(aligned["month"].unique())

    new_mult = aligned["size_mult"].copy().astype(np.float64)
    n_amp_total = 0
    threshold_by_month: dict[str, float] = {}
    n_amp_by_month: dict[str, int] = {}

    for i, m in enumerate(months):
        if i < WARMUP_MONTHS:
            continue
        m_start = pd.Timestamp(m.start_time, tz="UTC")
        m_end = (pd.Timestamp(m.end_time, tz="UTC")
                 + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
        train_start = m_start - pd.DateOffset(months=TRAIL_MONTHS)

        train_window = aligned.loc[
            (aligned.index >= train_start) & (aligned.index < m_start)
        ]
        train_longs = train_window[train_window["signal"] == 1]
        slopes = train_longs["svi_term_slope_1w_2w"].dropna()
        if len(slopes) < 30:
            continue

        threshold = float(np.quantile(slopes, PERCENTILE))
        threshold_by_month[str(m)] = threshold

        cur = aligned.loc[(aligned.index >= m_start) & (aligned.index <= m_end)]
        amp_mask = (
            (cur["signal"] == 1) &
            (cur["svi_term_slope_1w_2w"] < threshold)
        )
        n_amp = int(amp_mask.sum())
        n_amp_by_month[str(m)] = n_amp
        n_amp_total += n_amp

        # Apply MULT and cap
        amped_idx = cur.loc[amp_mask].index
        new_mult.loc[amped_idx] = (new_mult.loc[amped_idx] * MULT).clip(upper=CAP)

    out = pd.DataFrame({"size_mult": new_mult.astype(np.float64)})
    out.index.name = k1.index.name
    out.to_parquet(OUT_PATH)
    print(f"[generate] Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"[generate]  New: mean={out['size_mult'].mean():.4f}, "
          f"max={out['size_mult'].max():.4f}, "
          f"n_at_cap={(out['size_mult'] >= CAP - 1e-6).sum()}")
    print(f"[generate]  Total bars amplified: {n_amp_total:,}")

    # Diagnostic — per-month threshold and amp count
    if threshold_by_month:
        thresh_arr = np.array(list(threshold_by_month.values()))
        amp_arr = np.array(list(n_amp_by_month.values()))
        print(f"[generate]  Per-month threshold: mean={thresh_arr.mean():.2f}, "
              f"min={thresh_arr.min():.2f}, max={thresh_arr.max():.2f}")
        print(f"[generate]  Per-month amp count: mean={amp_arr.mean():.1f}, "
              f"max={amp_arr.max()}")
        print(f"[generate]  First applicable month: {list(threshold_by_month.keys())[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
