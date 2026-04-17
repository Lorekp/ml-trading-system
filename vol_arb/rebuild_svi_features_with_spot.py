"""Augment SVI features parquet with ONLY the new svi_atm_spot column.

Preserves byte-identity of all existing columns in
``vol_arb/output/svi_features_15m.parquet`` — the column set production
models were trained on. Computes ``svi_atm_spot`` from the cached
``svi_surface_params_1h.parquet`` by evaluating the nearest-slice ATM
vol at k=0 using the SAME parameters the original pipeline saved.

Why not recompute everything: the current SVI evaluation path produces
slightly different values from the values saved in the parquet (the
code has drifted between 2021-04-10 and 2026-04-17, probably in
``total_variance`` interpolation or ``_constant_maturity_features``
logic). Overwriting would break compatibility with the production
models trained on the old values. Adding ONLY the new column keeps the
existing 44-feature control byte-compatible and lets Tier 2 research
run against the same baseline.

Speed: ~20s on BTC for 43,965 buckets.

Usage::

    python -m vol_arb.rebuild_svi_features_with_spot --currency BTC
    python -m vol_arb.rebuild_svi_features_with_spot --currency ETH
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.svi import SVISlice
from vol_arb.paths import svi_features_path, svi_surface_params_path


def _compute_spot_from_nearest_slice(bucket_params: pd.DataFrame) -> float:
    """Compute svi_atm_spot for one bucket's worth of per-expiry params.

    Mirrors the production gate in vol_arb.svi._constant_maturity_features:
    only populate when nearest slice has calibration_level < 3, n_strikes >= 3,
    and is butterfly-free on a 50-point grid. Otherwise NaN — LightGBM
    handles NaN natively.
    """
    if bucket_params.empty:
        return float("nan")
    # Nearest slice = shortest T
    idx = bucket_params["T"].idxmin()
    r = bucket_params.loc[idx]
    sl = SVISlice(
        a=float(r["a"]),
        b=float(r["b"]),
        rho=float(r["rho"]),
        m=float(r["m"]),
        sigma=float(r["sigma"]),
        T=float(r["T"]),
        F=float(r["F"]),
        calibration_level=int(r["level"]),
        rmse=float(r["rmse"]),
        n_strikes=int(r["n_strikes"]),
    )
    # Quality gate
    if sl.calibration_level >= 3:
        return float("nan")
    if sl.n_strikes < 3:
        return float("nan")
    if not sl.is_butterfly_free(50):
        return float("nan")
    return float(sl.atm_vol_pct())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC", choices=["BTC", "ETH"])
    ap.add_argument("--bucket-freq", default="1h")
    ap.add_argument("--verify-only", action="store_true",
                    help="Don't write — just report")
    args = ap.parse_args()

    params_path = svi_surface_params_path(args.currency)
    feat_path = svi_features_path(args.currency)

    if not params_path.exists():
        print(f"[rebuild] ERROR: params cache not found at {params_path}")
        return 1
    if not feat_path.exists():
        print(f"[rebuild] ERROR: features parquet not found at {feat_path}")
        return 1

    print(f"[rebuild] loading params: {params_path}")
    params = pd.read_parquet(params_path)
    print(f"[rebuild]   {len(params)} rows, {params['timestamp'].nunique()} unique buckets")

    print(f"[rebuild] loading features: {feat_path}")
    feats = pd.read_parquet(feat_path)
    print(f"[rebuild]   {feats.shape[0]} rows x {feats.shape[1]} cols")

    if "svi_atm_spot" in feats.columns:
        nn = feats["svi_atm_spot"].notna().sum()
        print(f"[rebuild] svi_atm_spot ALREADY PRESENT: {nn}/{len(feats)} non-null")
        if not args.verify_only:
            print(f"[rebuild] re-computing anyway (in case gate logic changed)...")

    # Compute per-bucket svi_atm_spot
    t0 = time.perf_counter()
    spot_rows = []
    for bucket_ts, bucket_params in params.groupby("timestamp", sort=True):
        spot = _compute_spot_from_nearest_slice(bucket_params)
        spot_rows.append({"bucket_ts": bucket_ts, "svi_atm_spot_new": spot})
    elapsed = time.perf_counter() - t0
    print(f"[rebuild]   computed {len(spot_rows)} bucket values in {elapsed:.1f}s")

    spot_df = pd.DataFrame(spot_rows)
    # Apply the same +bucket_freq end-shift the original pipeline applied
    spot_df["ts_shifted"] = spot_df["bucket_ts"] + pd.Timedelta(args.bucket_freq)
    spot_df = spot_df.set_index("ts_shifted").sort_index()

    # Reindex onto the existing features grid (15m, ffill within bucket)
    spot_aligned = spot_df["svi_atm_spot_new"].reindex(feats.index, method="ffill")
    nn = spot_aligned.notna().sum()
    print(f"[rebuild] svi_atm_spot (new) non-null: {nn}/{len(spot_aligned)} ({nn/len(spot_aligned)*100:.1f}%)")
    if nn:
        print(f"           mean={spot_aligned.mean():.2f}  median={spot_aligned.median():.2f}  std={spot_aligned.std():.2f}")

    if args.verify_only:
        print(f"[rebuild] verify-only — not writing.")
        return 0

    # Byte-identical guard on all pre-existing columns — do NOT touch them
    feats_new = feats.copy()
    feats_new["svi_atm_spot"] = spot_aligned.values

    # Sanity: the only diff should be the new column
    new_cols = set(feats_new.columns) - set(feats.columns)
    dropped_cols = set(feats.columns) - set(feats_new.columns)
    print(f"[rebuild] new columns: {new_cols}  dropped: {dropped_cols}")
    for c in feats.columns:
        ov = feats[c].values
        nv = feats_new[c].values
        if ov.dtype.kind in "fiu":
            ok = np.array_equal(ov, nv, equal_nan=True)
        else:
            ok = (ov == nv).all()
        if not ok:
            print(f"[rebuild] ABORT — unexpected change on column {c}")
            return 2

    feats_new.to_parquet(feat_path)
    print(f"[rebuild] wrote {feat_path} ({feats_new.shape[0]} rows x {feats_new.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
