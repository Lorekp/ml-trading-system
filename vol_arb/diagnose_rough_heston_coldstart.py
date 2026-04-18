"""Diagnostic: measure cold-start RMSE distribution on real Deribit panels.

Samples N random (date, hour) pairs from the 2022-01 → 2026-03 trade archive,
loads the bucket panel via the sidecar builder's `_panel_for_bucket`, runs
`calibrate_rough_heston(..., use_global_search=True, init=None)` once per
bucket, records RMSE + timing + param fit.

Purpose: establish the baseline RMSE distribution before the Session G DE
upgrade, then re-run post-upgrade to verify the quality gate
(p50 RMSE < 5vp AND p95 RMSE < 10vp).

Usage (pre-DE baseline)::

    python -m vol_arb.diagnose_rough_heston_coldstart \\
        --n-buckets 50 --workers 4 --label nm --seed 42

Usage (post-DE, after G.2)::

    python -m vol_arb.diagnose_rough_heston_coldstart \\
        --n-buckets 50 --workers 4 --label de --seed 42 \\
        --global-method de

The ``--seed`` is used to pick the same (date, hour) pairs across runs, so
NM vs DE can be compared bucket-for-bucket.

Output
------
Writes ``vol_arb/output/rough_heston_coldstart_diagnostic_{label}.json``:

.. code-block:: json

    {
      "label": "nm",
      "global_method": null,
      "n_buckets_attempted": 50,
      "n_buckets_panel_ok": 44,
      "n_buckets_calibrated": 42,
      "wall_clock_seconds": 2103.4,
      "rmse_stats": {"p10": 0.012, "p50": 0.139, "p90": 0.218, "p95": 0.243, "mean": 0.148},
      "n_rmse_gt_10vp": 31,
      "per_bucket": [...]  # list of per-bucket records
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.build_rough_heston_sidecar import (
    OPTIONS_DIR,
    WINDOW_SIDE_MINUTES,
    _load_day_options,
    _panel_for_bucket,
)
from vol_arb.rough_heston import RoughHestonParams, calibrate_rough_heston

logger = logging.getLogger("rough_heston_diagnose")

OUTPUT_DIR = Path("vol_arb/output")


def _sample_dates_and_hours(
    n_buckets: int,
    seed: int,
    options_dir: Path,
    start: str = "2022-01",
    end: str = "2026-03",
) -> list[tuple[str, int]]:
    """Sample (date, hour) pairs uniformly from available trade files."""
    rng = random.Random(seed)
    avail_files = sorted(options_dir.glob("*.parquet"))
    avail_dates = [
        f.stem for f in avail_files if start <= f.stem[:7] <= end
    ]
    if len(avail_dates) < n_buckets:
        raise RuntimeError(
            f"only {len(avail_dates)} trade files in [{start}, {end}]; "
            f"cannot sample {n_buckets} buckets"
        )
    # Sample with replacement is allowed (different hours same day are different
    # panels), but we want some date diversity — so cap 1 bucket per date.
    sampled_dates = rng.sample(avail_dates, n_buckets)
    pairs = [(d, rng.randint(1, 23)) for d in sampled_dates]
    return sorted(pairs)


def _diagnose_one_bucket(
    date: str,
    hour: int,
    options_dir: Path,
    global_method: str | None,
    de_workers: int = 1,
) -> dict:
    """Load + calibrate a single bucket. Returns a record dict."""
    record = {
        "date": date,
        "hour": hour,
        "panel_ok": False,
        "calibrated": False,
        "rmse": None,
        "calibration_level": 0,
        "n_options": 0,
        "tau_min_days": None,
        "tau_max_days": None,
        "V_0": None,
        "H": None,
        "eta": None,
        "rho": None,
        "wall_clock_seconds": 0.0,
        "error": None,
    }
    try:
        trades = _load_day_options(date, options_dir)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"load_day_options: {exc}"
        return record
    if trades.empty:
        record["error"] = "empty day"
        return record

    bucket_end = pd.Timestamp(f"{date} {hour:02d}:00:00", tz="UTC")
    res = _panel_for_bucket(trades, bucket_end, window_side_min=WINDOW_SIDE_MINUTES)
    if res is None:
        record["error"] = "panel insufficient"
        return record
    panel, F = res
    record["panel_ok"] = True
    record["n_options"] = int(len(panel))
    record["tau_min_days"] = float(panel["tau"].min() * 365.25)
    record["tau_max_days"] = float(panel["tau"].max() * 365.25)

    t0 = time.time()
    try:
        kwargs = {"use_global_search": True, "init": None, "F": F}
        if global_method is not None:
            kwargs["global_method"] = global_method
        if global_method == "de" and de_workers > 1:
            kwargs["de_workers"] = de_workers
        fit = calibrate_rough_heston(panel, **kwargs)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"calibrate: {exc}"
        record["wall_clock_seconds"] = time.time() - t0
        return record

    record["wall_clock_seconds"] = time.time() - t0
    record["calibrated"] = True
    record["rmse"] = float(fit.calibration_rmse) if np.isfinite(fit.calibration_rmse) else None
    record["calibration_level"] = int(fit.calibration_level)
    record["V_0"] = float(fit.V_0)
    record["H"] = float(fit.H)
    record["eta"] = float(fit.eta)
    record["rho"] = float(fit.rho)
    return record


def _aggregate(records: list[dict]) -> dict:
    """Compute summary stats over per-bucket records."""
    panel_ok = [r for r in records if r["panel_ok"]]
    calib = [r for r in records if r["calibrated"] and r["rmse"] is not None]
    rmses = np.array([r["rmse"] for r in calib])
    elapsed = np.array([r["wall_clock_seconds"] for r in calib])
    if len(rmses) == 0:
        rmse_stats = {}
    else:
        rmse_stats = {
            "min": float(rmses.min()),
            "p10": float(np.quantile(rmses, 0.10)),
            "p25": float(np.quantile(rmses, 0.25)),
            "p50": float(np.quantile(rmses, 0.50)),
            "p75": float(np.quantile(rmses, 0.75)),
            "p90": float(np.quantile(rmses, 0.90)),
            "p95": float(np.quantile(rmses, 0.95)),
            "p99": float(np.quantile(rmses, 0.99)),
            "max": float(rmses.max()),
            "mean": float(rmses.mean()),
            "std": float(rmses.std()),
        }
    return {
        "n_buckets_attempted": len(records),
        "n_buckets_panel_ok": len(panel_ok),
        "n_buckets_calibrated": len(calib),
        "n_rmse_lt_5vp": int((rmses < 0.05).sum()) if len(rmses) else 0,
        "n_rmse_lt_10vp": int((rmses < 0.10).sum()) if len(rmses) else 0,
        "n_rmse_gt_10vp": int((rmses >= 0.10).sum()) if len(rmses) else 0,
        "n_rmse_gt_20vp": int((rmses >= 0.20).sum()) if len(rmses) else 0,
        "wall_clock_stats": {
            "mean_sec": float(elapsed.mean()) if len(elapsed) else 0.0,
            "median_sec": float(np.median(elapsed)) if len(elapsed) else 0.0,
            "max_sec": float(elapsed.max()) if len(elapsed) else 0.0,
        },
        "rmse_stats": rmse_stats,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-buckets", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", required=True, help="output file label, e.g. 'nm' or 'de'")
    parser.add_argument(
        "--global-method",
        choices=["nm", "de"],
        default=None,
        help="if set, passed as global_method kwarg to calibrate_rough_heston (requires G.2 calibrator)",
    )
    parser.add_argument(
        "--de-workers",
        type=int,
        default=1,
        help="inner DE parallelism (number of processes for scipy DE). "
             "Use with --workers 1 for single-bucket diagnostics; do not combine "
             "with high --workers (CPU oversubscription).",
    )
    parser.add_argument("--options-dir", default=str(OPTIONS_DIR))
    parser.add_argument("--start", default="2022-01")
    parser.add_argument("--end", default="2026-03")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    options_dir = Path(args.options_dir)

    pairs = _sample_dates_and_hours(
        args.n_buckets, args.seed, options_dir, args.start, args.end,
    )
    logger.info(
        "sampled %d (date,hour) pairs in [%s, %s], label=%s, global_method=%s",
        len(pairs), args.start, args.end, args.label, args.global_method,
    )

    t0 = time.time()
    records: list[dict] = []
    if args.workers <= 1:
        for i, (date, hour) in enumerate(pairs):
            r = _diagnose_one_bucket(
                date, hour, options_dir, args.global_method, args.de_workers,
            )
            records.append(r)
            if r["calibrated"] and r["rmse"] is not None:
                logger.info(
                    "[%3d/%d] %s@%02d: n_opt=%d tau=[%.1f,%.1f]d rmse=%.4f V0=%.3f H=%.3f eta=%.2f rho=%+.2f t=%.1fs",
                    i + 1, len(pairs), date, hour,
                    r["n_options"], r["tau_min_days"] or 0, r["tau_max_days"] or 0,
                    r["rmse"], r["V_0"] or 0, r["H"] or 0, r["eta"] or 0, r["rho"] or 0,
                    r["wall_clock_seconds"],
                )
            else:
                logger.info(
                    "[%3d/%d] %s@%02d: SKIP (%s)",
                    i + 1, len(pairs), date, hour, r["error"] or "unknown",
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _diagnose_one_bucket,
                    d, h, options_dir, args.global_method, args.de_workers,
                ): (d, h)
                for d, h in pairs
            }
            for i, fut in enumerate(as_completed(futures)):
                d, h = futures[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001
                    r = {
                        "date": d, "hour": h, "panel_ok": False,
                        "calibrated": False, "rmse": None,
                        "error": f"worker crash: {exc}",
                        "wall_clock_seconds": 0.0,
                    }
                records.append(r)
                if r.get("calibrated") and r.get("rmse") is not None:
                    logger.info(
                        "[%3d/%d done] %s@%02d: n_opt=%d rmse=%.4f t=%.1fs",
                        i + 1, len(pairs), d, h,
                        r["n_options"], r["rmse"], r["wall_clock_seconds"],
                    )
                else:
                    logger.info(
                        "[%3d/%d done] %s@%02d: SKIP (%s)",
                        i + 1, len(pairs), d, h, r.get("error", "unknown"),
                    )
    wall_clock = time.time() - t0
    agg = _aggregate(records)

    summary = {
        "label": args.label,
        "global_method": args.global_method,
        "seed": args.seed,
        "wall_clock_seconds": wall_clock,
        **agg,
        "per_bucket": records,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"rough_heston_coldstart_diagnostic_{args.label}.json"
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("wrote %s", out_path)

    logger.info("\n=== Summary (label=%s) ===", args.label)
    logger.info("wall clock: %.1f s (%d workers)", wall_clock, args.workers)
    logger.info("buckets attempted: %d", agg["n_buckets_attempted"])
    logger.info("panels OK:         %d", agg["n_buckets_panel_ok"])
    logger.info("calibrated:        %d", agg["n_buckets_calibrated"])
    if agg["rmse_stats"]:
        s = agg["rmse_stats"]
        logger.info("RMSE (vol-points): p10=%.1fvp p50=%.1fvp p90=%.1fvp p95=%.1fvp mean=%.1fvp",
                    100 * s["p10"], 100 * s["p50"], 100 * s["p90"], 100 * s["p95"], 100 * s["mean"])
    logger.info("buckets with RMSE >10vp: %d/%d",
                agg["n_rmse_gt_10vp"], agg["n_buckets_calibrated"])
    logger.info("buckets with RMSE >20vp: %d/%d",
                agg["n_rmse_gt_20vp"], agg["n_buckets_calibrated"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
