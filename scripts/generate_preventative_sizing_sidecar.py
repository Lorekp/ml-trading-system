"""Generate the per-bar preventative-sizing multiplier sidecar.

Phase P artifact (plan: ``now-check-empirically-from-eager-bird.md``).

Reads:
  - ``vol_arb/output/BTC_v2_trade_log.parquet``: canonical V2 bars with
    (equity, index_price, iv_used, gamma, vega) columns.
  - ``data/deribit/dvol/BTC_dvol_60m.parquet``: DVOL hourly series for the
    frozen ``pme_shock_iv_pct`` calibration.

Writes:
  - ``vol_arb/output/BTC_v2_preventative_size_mult.parquet``: per-bar
    ``size_mult`` ∈ [0.10, 1.0] for every bar in the canonical trade log.

The pme_shock_iv_pct anchor is computed ONCE on the calibration window
2021-04-01 → 2023-09-30 (UTC, exclusive end). The script refuses to run
without an explicit ``--allow-overwrite`` flag if the sidecar already
exists — the sidecar is itself a frozen artifact.

Run::

    PYTHONPATH=. python scripts/generate_preventative_sizing_sidecar.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.preventative_sizing import (
    PME_IV_CAL_WINDOW,
    build_pme_anchors,
    compute_size_scaler_vectorised,
    write_sidecar,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_preventative_sidecar")

TRADE_LOG_PATH = Path("vol_arb/output/BTC_v2_trade_log.parquet")
DVOL_PATH = Path("data/deribit/dvol/BTC_dvol_60m.parquet")
SIDECAR_OUT = Path("vol_arb/output/BTC_v2_preventative_size_mult.parquet")


def _load_dvol_series() -> pd.Series:
    """Load Deribit DVOL hourly series.

    The on-disk parquet stores ``timestamp`` as a column on a RangeIndex
    (start-stamped per pipeline convention). For the anchor we just need
    a consistent daily resampling — both start- and end-stamping
    give the same daily quantile because the full UTC day is sliced.
    """
    df = pd.read_parquet(DVOL_PATH)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if "close" in df.columns:
        return df["close"].rename("dvol")
    if "dvol_close" in df.columns:
        return df["dvol_close"].rename("dvol")
    if "dvol" in df.columns:
        return df["dvol"]
    raise KeyError(f"{DVOL_PATH}: no DVOL column found "
                    f"(have {list(df.columns)})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="Overwrite existing sidecar (defeat safety)")
    parser.add_argument("--out", type=Path, default=SIDECAR_OUT)
    args = parser.parse_args()

    out: Path = args.out
    if out.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Sidecar already exists at {out}. Use --allow-overwrite "
            "only if you have a NEW plan file authorising regeneration."
        )

    logger.info("Loading canonical V2 trade log: %s", TRADE_LOG_PATH)
    tl = pd.read_parquet(TRADE_LOG_PATH)
    if not isinstance(tl.index, pd.DatetimeIndex):
        tl.index = pd.to_datetime(tl.index, utc=True)
    logger.info("Trade log: %d bars, span %s → %s",
                len(tl), tl.index.min(), tl.index.max())
    required = {"equity", "index_price", "iv_used", "gamma", "vega"}
    missing = required - set(tl.columns)
    if missing:
        raise KeyError(f"Trade log missing required columns: {sorted(missing)}")

    logger.info("Loading DVOL: %s", DVOL_PATH)
    dvol = _load_dvol_series()
    logger.info("DVOL: %d rows, span %s → %s",
                len(dvol), dvol.index.min(), dvol.index.max())

    logger.info("Building PME anchors with calibration window %s → %s",
                PME_IV_CAL_WINDOW[0].date(), PME_IV_CAL_WINDOW[1].date())
    anchors = build_pme_anchors(dvol)
    logger.info("Frozen anchors:")
    logger.info("  loss_budget_pct  = %.4f (1%% policy)", anchors.loss_budget_pct)
    logger.info("  pme_shock_spot   = %.4f (Deribit ±17.5%%)",
                anchors.pme_shock_spot)
    logger.info("  pme_shock_iv_pct = %.4f (99th-pct |ΔDVOL_1d|, %d obs)",
                anchors.pme_shock_iv_pct, anchors.pme_iv_anchor_n_obs)

    size_mult = compute_size_scaler_vectorised(tl, anchors=anchors)

    n_bind = int((size_mult < 1.0 - 1e-9).sum())
    n_floor = int((size_mult <= 0.10 + 1e-9).sum())
    n_total = len(size_mult)
    logger.info("Size multiplier statistics:")
    logger.info("  mean  = %.4f", float(size_mult.mean()))
    logger.info("  p1    = %.4f", float(size_mult.quantile(0.01)))
    logger.info("  p10   = %.4f", float(size_mult.quantile(0.10)))
    logger.info("  p50   = %.4f", float(size_mult.quantile(0.50)))
    logger.info("  bars binding (<1.0):  %d / %d (%.2f%%)",
                n_bind, n_total, 100.0 * n_bind / n_total)
    logger.info("  bars at floor:         %d (%.2f%%)",
                n_floor, 100.0 * n_floor / n_total)

    write_sidecar(size_mult, out)
    logger.info("Wrote sidecar to %s", out)

    print()
    print("=" * 70)
    print(" Preventative sizing sidecar generated")
    print("=" * 70)
    print(f"  pme_shock_iv_pct = {anchors.pme_shock_iv_pct:.4f}  "
          f"(99th-pct |dDVOL_1d| relative move "
          f"on {anchors.pme_iv_anchor_n_obs} cal-window days)")
    print(f"  bars binding (<1.0): {n_bind:,} / {n_total:,} "
          f"({100.0 * n_bind / n_total:.2f}%)")
    print(f"  bars at f_min=0.10:  {n_floor:,} "
          f"({100.0 * n_floor / n_total:.2f}%)")
    print(f"  mean size_mult: {float(size_mult.mean()):.4f}")
    print(f"  Sidecar: {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
