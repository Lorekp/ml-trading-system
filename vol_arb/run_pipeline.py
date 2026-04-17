"""CLI entry point for the vol-arb merge pipeline.

Usage:
    PYTHONPATH=. python vol_arb/run_pipeline.py
    PYTHONPATH=. python vol_arb/run_pipeline.py --currency ETH
    PYTHONPATH=. python vol_arb/run_pipeline.py --include-features
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import math

from vol_arb.config import VolArbConfig
from vol_arb.paths import (
    dvol_path, option_trade_dir, tsrv_path, vol_arb_path,
)
from vol_arb.pipeline import build_vol_arb_dataset, verify_alignment


# Per-currency slippage bucket calibration. BTC defaults reproduce the
# legacy module-level constants in vol_arb/pipeline.py exactly; ETH edges
# shift by roughly 10x in size (ETH ~10x more units per trade than BTC) and
# upward in IV to match its higher baseline.
_SLIPPAGE_EDGES = {
    "BTC": dict(
        size_edges=[0, 1, 3, 10, math.inf],
        vol_edges=[0, 45, 65, math.inf],
    ),
    "ETH": dict(
        size_edges=[0, 10, 30, 100, math.inf],
        vol_edges=[0, 55, 85, math.inf],
    ),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Vol-arb options-ML merge pipeline")
    parser.add_argument("--currency", default="BTC", help="BTC or ETH")
    parser.add_argument("--output", default="", help="Output parquet path (auto-generated if empty)")
    parser.add_argument("--include-features", action="store_true",
                        help="Join full 116-col ML features")
    parser.add_argument("--ffill", type=int, default=None,
                        help="Forward-fill limit for option data (number of 15m bars)")
    parser.add_argument("--moneyness-lo", type=float, default=0.95)
    parser.add_argument("--moneyness-hi", type=float, default=1.05)
    parser.add_argument("--dte-min", type=int, default=1)
    parser.add_argument("--dte-max", type=int, default=14)
    parser.add_argument("--svi", action="store_true",
                        help="Enable SVI surface calibration and svi_iv_matched column "
                             "(produces the BTC_vol_arb_svi.parquet variant)")
    parser.add_argument("--svi-bucket-freq", type=str, default=None,
                        help="SVI calibration bucket frequency (e.g. '1h' or "
                             "'15min'). Default None → uses VolArbConfig default "
                             "('1h'). Set to '15min' to match a 15m holding "
                             "strategy — the resulting svi_features cadence "
                             "matches the TSRV 15m grid.")
    args = parser.parse_args()

    ccy = args.currency.upper()
    asset = f"{ccy}USDT"

    if ccy not in _SLIPPAGE_EDGES:
        raise ValueError(f"Unsupported currency {ccy!r}; add to _SLIPPAGE_EDGES first.")
    edges = _SLIPPAGE_EDGES[ccy]

    cfg_kwargs = dict(
        currency=ccy,
        option_trade_dir=str(option_trade_dir(ccy)),
        dvol_path=str(dvol_path(ccy)),
        tsrv_path=str(tsrv_path(asset)),
        moneyness_lo=args.moneyness_lo,
        moneyness_hi=args.moneyness_hi,
        dte_min_days=args.dte_min,
        dte_max_days=args.dte_max,
        ffill_limit_bars=args.ffill,
        svi_enabled=args.svi,
        slippage_size_edges=edges["size_edges"],
        slippage_vol_edges=edges["vol_edges"],
    )
    if args.svi_bucket_freq is not None:
        cfg_kwargs["svi_bucket_freq"] = args.svi_bucket_freq
    config = VolArbConfig(**cfg_kwargs)

    if args.include_features:
        config.features_path = f"vol_model/output/{asset}_features_15m.parquet"

    # Build (SVI outputs persisted internally per config.currency — see
    # build_vol_arb_dataset for the resolver-driven path handling)
    merged = build_vol_arb_dataset(config)

    # Verify
    verify_alignment(merged)

    # Save vol-arb dataset via resolver (BTC keeps legacy unsuffixed filename).
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = vol_arb_path(ccy, svi=args.svi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    logger.info("Saved %d rows x %d cols to %s", len(merged), merged.shape[1], out_path)


if __name__ == "__main__":
    main()
