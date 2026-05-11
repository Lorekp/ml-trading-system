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
import pandas as pd

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
    parser.add_argument("--bar-minutes", type=int, default=15,
                        help="Target bar cadence for the WHOLE pipeline: TSRV "
                             "input parquet, SVI calibration buckets, composite-"
                             "straddle bucketing, and output filenames. Default "
                             "15 (legacy BTC production). At h=5 pass 5; at h=60 "
                             "pass 60. Must divide 1440. See plan "
                             "starry-rolling-rabin.md Phase 2h5.2a.")
    parser.add_argument("--end-date", type=str, default="",
                        help="Optional ISO timestamp upper bound for the merged "
                             "output. When set, rows with index > end_date are "
                             "filtered out BEFORE writing the parquet. Use for "
                             "partial-window parity runs against the live "
                             "engine. Default empty (full sample).")
    args = parser.parse_args()
    if 1440 % args.bar_minutes != 0:
        parser.error(
            f"--bar-minutes={args.bar_minutes} must divide 1440; "
            f"choose 5, 15, 30, 60, etc."
        )

    ccy = args.currency.upper()
    asset = f"{ccy}USDT"

    if ccy not in _SLIPPAGE_EDGES:
        raise ValueError(f"Unsupported currency {ccy!r}; add to _SLIPPAGE_EDGES first.")
    edges = _SLIPPAGE_EDGES[ccy]

    # Unified cadence: --bar-minutes drives TSRV / SVI-bucket / composite-
    # straddle bucketing. --svi-bucket-freq can still override SVI alone
    # (for studies where SVI runs at a DIFFERENT cadence than the bar grid),
    # but the default is "match bar_minutes" so the pipeline is coherent.
    bar_freq_str = f"{args.bar_minutes}min"
    cfg_kwargs = dict(
        currency=ccy,
        option_trade_dir=str(option_trade_dir(ccy)),
        dvol_path=str(dvol_path(ccy)),
        tsrv_path=str(tsrv_path(asset, tf_min=args.bar_minutes)),
        moneyness_lo=args.moneyness_lo,
        moneyness_hi=args.moneyness_hi,
        dte_min_days=args.dte_min,
        dte_max_days=args.dte_max,
        ffill_limit_bars=args.ffill,
        svi_enabled=args.svi,
        slippage_size_edges=edges["size_edges"],
        slippage_vol_edges=edges["vol_edges"],
        bucket_freq=bar_freq_str,
    )
    # Default SVI calibration cadence = bar cadence (coherent); user can
    # override explicitly to decouple.
    if args.svi_bucket_freq is not None:
        cfg_kwargs["svi_bucket_freq"] = args.svi_bucket_freq
    else:
        cfg_kwargs["svi_bucket_freq"] = bar_freq_str
    config = VolArbConfig(**cfg_kwargs)

    if args.include_features:
        config.features_path = f"vol_model/output/{asset}_features_15m.parquet"

    # Build (SVI outputs persisted internally per config.currency — see
    # build_vol_arb_dataset for the resolver-driven path handling)
    merged = build_vol_arb_dataset(config)

    # Partial-window slice: trim the output to [start, end_date] before
    # verification + save. Rows beyond end_date are dropped so downstream
    # consumers (predictions extraction, backtester) see a clean cutoff.
    if args.end_date:
        ed = pd.Timestamp(args.end_date)
        if merged.index.tz is not None and ed.tz is None:
            ed = ed.tz_localize(merged.index.tz)
        n_pre = len(merged)
        merged = merged.loc[merged.index <= ed]
        logger.info("Partial-window slice: end_date=%s, %d -> %d rows",
                    ed, n_pre, len(merged))

    # Verify
    verify_alignment(merged)

    # Save vol-arb dataset via resolver. BTC @ h=15 keeps legacy unsuffixed
    # filename for byte-identity; other horizons get _h{N}m suffix so h=5 SVI
    # regen does not clobber the h=15 production file.
    if args.output:
        out_path = Path(args.output)
    else:
        _merged_bar_minutes = int(
            round(pd.Timedelta(config.bucket_freq).total_seconds() / 60.0)
        )
        out_path = vol_arb_path(
            ccy, svi=args.svi, horizon_minutes=_merged_bar_minutes,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    logger.info("Saved %d rows x %d cols to %s", len(merged), merged.shape[1], out_path)


if __name__ == "__main__":
    main()
