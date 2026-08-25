"""Build Tier 1 extra features (ETH-TSRV + RQ) as a sidecar parquet.

Why a sidecar instead of rebuilding the full features parquet?
- The production ``BTCUSDT_features_15m.parquet`` is 183K rows × 126 cols and
  was built from 6.6M dollar bars via the full FeatureEngineer pipeline.
  Rebuilding takes ~30-60 min.
- For the 2026-04-17 spot-check we only need 3 ETH-TSRV features and 2 RQ
  features added, each with its windowing variants.
- The sidecar computes them directly at 15m resolution (skips the dollar-bar
  roundtrip), which is DIRECTIONALLY identical to what the engineer.py
  blocks would produce. Values differ slightly (rolling at 15m vs bar grid),
  but the causal structure — shift(1) for point-in-time, past-only FFill, 1d
  and 1w rolling means — is identical.

Causal safety invariants (mirror engineer.py ETH/RQ blocks):
- ETH TSRV (START-stamped): at parquet-index T, value = vol of [T, T+freq).
  shift(1) gives value from T-freq at new-index T, i.e. vol of [T-freq, T).
  ffill + reindex to 15m grid: feature at 15m-index T has vol info strictly
  from [..., T-freq). No look-ahead. ✓
- BTC TSRV (END-stamped): at parquet-index T, value = vol of [T-freq, T).
  shift(1) gives value from T-freq at new-index T, i.e. vol of [T-2·freq, T-freq).
  ffill + reindex: feature at 15m-index T uses info strictly from [..., T-freq).
  No look-ahead. ✓ It is slightly staler under END than under START.

Usage:
    python -m vol_model.run_build_extra_features --asset BTCUSDT

Output: vol_model/output/BTCUSDT_features_extras_15m.parquet
Schema: eth_tsrv_mean_{1d,1w}, eth_tsrv_ema50, rq_{1d,1w} + their lag/EMA
        variants (25 columns total after apply_windowing).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_model.config import VolModelConfig
from vol_model.features.windowing import apply_windowing


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_extras")

# 15m-resolution rolling windows (days × 96 bars/day)
WIN_1D = 96
WIN_1W = 7 * 96
MIN_1D = 24
MIN_1W = 168


def build_eth_tsrv_raw(idx_15m: pd.DatetimeIndex) -> pd.DataFrame:
    """Build raw ETH-TSRV features at 15m resolution on the given index."""
    path = Path("TSRV/output/ETHUSDT_tsrv_15m.parquet")
    if not path.exists():
        raise FileNotFoundError(f"ETH TSRV parquet missing: {path}")
    eth_df = pd.read_parquet(path)

    # ETH is START-stamped (not migrated): at parquet-index T, value = vol of
    # [T, T+freq). shift(1) is essential to avoid look-ahead.
    eth_ln = np.log(eth_df["vol_ann"].clip(lower=1e-10))
    eth_ln_lagged = eth_ln.shift(1)  # strict point-in-time

    # FFill to the 15m target index (the bar-grid ffill in engineer.py collapses
    # to this at 15m resolution anyway).
    ci = eth_ln_lagged.dropna().index.union(idx_15m).sort_values()
    eth_at_15m = eth_ln_lagged.reindex(ci).ffill().reindex(idx_15m)

    raw = pd.DataFrame(index=idx_15m)
    raw["eth_tsrv_mean_1d"] = eth_at_15m.rolling(WIN_1D, min_periods=MIN_1D).mean()
    raw["eth_tsrv_mean_1w"] = eth_at_15m.rolling(WIN_1W, min_periods=MIN_1W).mean()
    raw["eth_tsrv_ema50"] = eth_at_15m.ewm(span=50, min_periods=10).mean()

    return raw


def build_rq_raw(idx_15m: pd.DatetimeIndex, asset: str = "BTCUSDT") -> pd.DataFrame:
    """Build raw RQ features at 15m resolution on the given index.

    Uses ``log1p(rq * 1e6)`` transform (matches engineer.py). RQ spans many
    orders of magnitude so log-scaling is essential for LightGBM stability.
    """
    path = Path(f"TSRV/output/{asset}_tsrv_15m.parquet")
    if not path.exists():
        raise FileNotFoundError(f"TSRV parquet missing: {path}")
    tsrv_df = pd.read_parquet(path)
    if "rq" not in tsrv_df.columns:
        raise KeyError(f"'rq' column missing from {path}; run TSRV.backfill_rq first")

    rq_series = tsrv_df["rq"].clip(lower=0.0)
    log_rq = np.log1p(rq_series * 1e6)
    log_rq_lagged = log_rq.shift(1)  # strict point-in-time

    ci = log_rq_lagged.dropna().index.union(idx_15m).sort_values()
    rq_at_15m = log_rq_lagged.reindex(ci).ffill().reindex(idx_15m)

    raw = pd.DataFrame(index=idx_15m)
    raw["rq_1d"] = rq_at_15m.rolling(WIN_1D, min_periods=MIN_1D).mean()
    raw["rq_1w"] = rq_at_15m.rolling(WIN_1W, min_periods=MIN_1W).mean()

    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTCUSDT")
    args = parser.parse_args()

    cfg = VolModelConfig(asset=args.asset)

    # Base on the EXISTING features parquet's index so rows align one-to-one
    # when the sidecar is joined into ctx.X_full downstream.
    base_path = Path(cfg.output_dir) / f"{args.asset}_features_15m.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"base features parquet missing: {base_path}")
    base = pd.read_parquet(base_path, columns=["n_bars_in_window"])
    idx_15m = base.index
    logger.info("Base index: %d rows, %s → %s", len(idx_15m), idx_15m[0], idx_15m[-1])

    # Build raw blocks
    eth_raw = build_eth_tsrv_raw(idx_15m)
    logger.info("ETH-TSRV raw: %s NaN per col: %s",
                eth_raw.shape, eth_raw.isna().sum().to_dict())
    rq_raw = build_rq_raw(idx_15m, asset=args.asset)
    logger.info("RQ raw: %s NaN per col: %s",
                rq_raw.shape, rq_raw.isna().sum().to_dict())

    raw = pd.concat([eth_raw, rq_raw], axis=1)

    # Apply the same windowing (lag1, lag3, ema10, ema50) used by the main
    # feature pipeline so downstream concat with X_full yields a uniform set
    # of columns per feature family.
    windowed = apply_windowing(raw, lags=cfg.lags, ema_spans=cfg.ema_spans)
    logger.info("After windowing: %d cols", windowed.shape[1])
    logger.info("Columns: %s", list(windowed.columns))

    out_path = Path(cfg.output_dir) / f"{args.asset}_features_extras_15m.parquet"
    windowed.to_parquet(out_path)
    logger.info("Saved: %s (%d rows × %d cols)", out_path, len(windowed), windowed.shape[1])

    # Final sanity: no leakage from the future (shift(1) + ffill only look
    # backward). The apply_windowing lags are also past-only. Spot-check a
    # single row to verify non-NaN and finite.
    sample = windowed.dropna().iloc[-1]
    logger.info("Sample last-row (all non-NaN): %s", sample.to_dict())


if __name__ == "__main__":
    main()
