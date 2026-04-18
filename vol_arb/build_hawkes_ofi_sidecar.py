"""One-off driver to build the Hawkes-OFI sidecar parquet for Session F.

For each 1-hour bucket timestamp T in the target window (default 2021-12 →
2026-03, matching production coverage), fit a bivariate exp-kernel Hawkes
process to the trailing 4 hours of 1-second-binned buy/sell trade counts
from Binance BTCUSDT perp aggTrades. Emit 7 fitted parameters + diagnostics
per bucket.

Runtime: ~1-3 minutes on 4 workers thanks to warm-starting across buckets
within a month (benchmark: 8ms warm vs 100ms cold per calibration).

Outputs
-------
- ``vol_arb/output/hawkes_ofi_params_1h.parquet`` — 1-hour sidecar with one
  row per bucket containing the 7 raw Hawkes params + branching-ratio summary
  stats + convergence diagnostics. Not used directly by the gauntlet; kept
  for diagnostics / upgrade paths.
- ``vol_arb/output/hawkes_ofi_features_15m.parquet`` — 15-minute sidecar
  consumed by the gauntlet. Reindexed to 15m grid by forward-fill within
  each 1-hour bucket (matches SVI pattern). Contains 7 columns: 5 level
  features (mu_b, mu_s, branching_buy, branching_sell, asymmetry) and 2
  1-hour-delta features (branching_buy_d1h, branching_sell_d1h).

CLI
---
Run from repo root::

    python -m vol_arb.build_hawkes_ofi_sidecar \
        --start 2021-12 --end 2026-03 --workers 4

or (smoke test on one month)::

    python -m vol_arb.build_hawkes_ofi_sidecar \
        --start 2023-07 --end 2023-07 --workers 1
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from TSRV.data_loader import available_months, load_trades
from vol_arb.hawkes_ofi import (
    HawkesOFIParams,
    bin_trades_to_counts,
    calibrate_hawkes_ofi,
)

logger = logging.getLogger("hawkes_ofi_sidecar")


# ── Constants ────────────────────────────────────────────────────────

TRADE_DIR = Path("data/binance-aggTrades-parquet")
SYMBOL = "BTCUSDT"
OUTPUT_DIR = Path("vol_arb/output")
PARAMS_PATH = OUTPUT_DIR / "hawkes_ofi_params_1h.parquet"
FEATURES_1H_PATH = OUTPUT_DIR / "hawkes_ofi_features_1h.parquet"
FEATURES_15M_PATH = OUTPUT_DIR / "hawkes_ofi_features_15m.parquet"

# Window + bin configuration (fixed, not tunable per the signal-research plan)
BUCKET_FREQ = "1h"
WINDOW_HOURS = 4
BIN_SECONDS = 1.0

# Diagnostic / feature columns written to parquet
PARAM_COLS = (
    "mu_b", "mu_s",
    "alpha_bb", "alpha_bs", "alpha_sb", "alpha_ss",
    "beta",
)
DIAG_COLS = (
    "branching_buy", "branching_sell", "spectral_radius",
    "converged", "n_iter", "final_neg_ll", "final_grad_norm",
)
ALL_COLS = PARAM_COLS + DIAG_COLS

# Session F feature set emitted to the 15m sidecar (before .diff(4) deltas)
LEVEL_FEATURES = ("hawkes_mu_buy", "hawkes_mu_sell",
                  "hawkes_branching_buy", "hawkes_branching_sell",
                  "hawkes_asymmetry")
DELTA_FEATURES = ("hawkes_branching_buy_d1h", "hawkes_branching_sell_d1h")
ALL_FEATURES = LEVEL_FEATURES + DELTA_FEATURES


# ── Per-month worker ─────────────────────────────────────────────────


def _process_month(
    month: str,
    prev_month: str | None,
    trade_dir: Path = TRADE_DIR,
    symbol: str = SYMBOL,
) -> pd.DataFrame:
    """Fit Hawkes on trailing 4h for every hourly bucket in ``month``.

    Loads ``month`` + ``prev_month`` (latter only for the first 4h of buckets
    in ``month``) and returns a DataFrame indexed by UTC bucket-end timestamps.

    Robust to missing parquets and to buckets with no trades (emits NaN row
    with converged=False). Warm-starts across successive buckets to keep
    per-fit wall-clock near the benchmark 8ms.
    """
    t0 = time.time()

    # Load current month + previous month's tail (last 4h + buffer)
    try:
        cur_df = load_trades(trade_dir, symbol, month)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not load month %s: %s -- skipping", month, exc)
        return pd.DataFrame()

    prev_tail_df = pd.DataFrame()
    if prev_month is not None:
        try:
            prev_df = load_trades(trade_dir, symbol, prev_month)
            # Keep only the last WINDOW_HOURS + a little buffer
            cutoff = prev_df.index.max() - pd.Timedelta(hours=WINDOW_HOURS + 1)
            prev_tail_df = prev_df.loc[prev_df.index >= cutoff]
        except Exception:  # noqa: BLE001
            pass

    trades = pd.concat([prev_tail_df, cur_df]).sort_index()
    if trades.empty:
        return pd.DataFrame()

    # Vectorize into int64 ns + bool aggressor side
    ts_ns = trades.index.astype("int64").values
    ibm = trades["is_buyer_maker"].values.astype(bool)

    # Hourly bucket timestamps (end-stamped) within ``month``
    month_start = pd.Timestamp(month + "-01", tz="UTC")
    month_end = month_start + pd.DateOffset(months=1)
    buckets = pd.date_range(
        month_start + pd.Timedelta(hours=1),  # first close is 1h after month start
        month_end,
        freq=BUCKET_FREQ,
        inclusive="left",
    )
    # Drop buckets that can't be fit because the trailing 4h reaches before
    # any data we have on disk.
    earliest_window_start = (
        trades.index.min() if not trades.empty else pd.Timestamp.max.tz_localize("UTC")
    )

    rows: list[dict] = []
    prev_fit: HawkesOFIParams | None = None
    for T in buckets:
        window_start = T - pd.Timedelta(hours=WINDOW_HOURS)
        row: dict = {"timestamp": T}
        if window_start < earliest_window_start:
            # Insufficient history; emit NaN
            for c in ALL_COLS:
                row[c] = np.nan if c != "converged" else False
            rows.append(row)
            continue

        N_b, N_s = bin_trades_to_counts(
            ts_ns, ibm,
            window_start_ns=int(window_start.value),
            window_end_ns=int(T.value),
            dt_seconds=BIN_SECONDS,
        )
        # Fit
        if N_b.sum() + N_s.sum() < 10:
            # Essentially no trades — can't fit; emit NaN
            for c in ALL_COLS:
                row[c] = np.nan if c != "converged" else False
            rows.append(row)
            prev_fit = None
            continue

        fit = calibrate_hawkes_ofi(N_b, N_s, dt=BIN_SECONDS, init=prev_fit)
        if fit.converged:
            prev_fit = fit
        else:
            # Retry once from default init if warm-start failed
            fit2 = calibrate_hawkes_ofi(N_b, N_s, dt=BIN_SECONDS, init=None)
            if fit2.converged:
                fit = fit2
                prev_fit = fit2
            else:
                prev_fit = None  # force cold restart next bucket

        d = asdict(fit)
        for c in PARAM_COLS:
            row[c] = d[c] if fit.converged else np.nan
        row["branching_buy"] = fit.branching_buy if fit.converged else np.nan
        row["branching_sell"] = fit.branching_sell if fit.converged else np.nan
        row["spectral_radius"] = fit.spectral_radius if fit.converged else np.nan
        row["converged"] = fit.converged
        row["n_iter"] = fit.n_iter
        row["final_neg_ll"] = fit.final_neg_ll
        row["final_grad_norm"] = fit.final_grad_norm
        rows.append(row)

    out = pd.DataFrame(rows).set_index("timestamp")
    n_ok = int(out["converged"].sum())
    t_elapsed = time.time() - t0
    logger.info(
        "month %s: %d buckets, %d converged (%.1f%%), %.1fs wall",
        month, len(out), n_ok, 100 * n_ok / max(len(out), 1), t_elapsed,
    )
    return out


# ── Driver ───────────────────────────────────────────────────────────


def build_sidecar(
    start_month: str,
    end_month: str,
    *,
    workers: int = 4,
    trade_dir: Path = TRADE_DIR,
    symbol: str = SYMBOL,
) -> pd.DataFrame:
    """Build the 1-hour Hawkes-OFI sidecar for [start_month, end_month] inclusive.

    Returns the full DataFrame (also written to disk at PARAMS_PATH).
    """
    avail = available_months(trade_dir, symbol)
    months = [m for m in avail if start_month <= m <= end_month]
    if not months:
        raise RuntimeError(
            f"no months found in range [{start_month}, {end_month}] "
            f"under {trade_dir / symbol}"
        )
    logger.info("processing %d months with %d workers", len(months), workers)

    prev_of: dict[str, str | None] = {}
    for i, m in enumerate(months):
        prev_of[m] = avail[avail.index(m) - 1] if avail.index(m) > 0 else None

    frames: list[pd.DataFrame] = []
    if workers <= 1:
        for m in months:
            frames.append(_process_month(m, prev_of[m], trade_dir=trade_dir, symbol=symbol))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fut_to_month = {
                pool.submit(_process_month, m, prev_of[m], trade_dir, symbol): m
                for m in months
            }
            for fut in as_completed(fut_to_month):
                m = fut_to_month[fut]
                try:
                    frames.append(fut.result())
                except Exception:  # noqa: BLE001
                    logger.exception("month %s failed", m)
                    continue

    if not frames:
        raise RuntimeError("no months produced output")
    all_df = pd.concat(frames).sort_index()
    # Strip duplicate timestamps (shouldn't exist, but defensive)
    all_df = all_df[~all_df.index.duplicated(keep="first")]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(PARAMS_PATH)
    logger.info(
        "wrote %s (%d rows, %d converged)",
        PARAMS_PATH, len(all_df), int(all_df["converged"].sum()),
    )
    return all_df


def derive_15m_features(params_df: pd.DataFrame) -> pd.DataFrame:
    """Derive the gauntlet-facing 15m feature sidecar from the 1h params.

    Steps
    -----
    1. Build 5 level features per 1h bucket (mu_b, mu_s, branching_buy,
       branching_sell, asymmetry). NaN at non-converged buckets.
    2. Reindex to 15m grid and forward-fill within the 1h bucket — each
       calibrated hour produces 4 identical 15m rows (SVI pattern).
    3. Compute .diff(4) on the 15m grid for the 2 delta features. Strictly
       past-only — no leakage.
    """
    if params_df.empty:
        return pd.DataFrame(columns=list(ALL_FEATURES))

    levels = pd.DataFrame(index=params_df.index)
    levels["hawkes_mu_buy"] = params_df["mu_b"]
    levels["hawkes_mu_sell"] = params_df["mu_s"]
    levels["hawkes_branching_buy"] = params_df["branching_buy"]
    levels["hawkes_branching_sell"] = params_df["branching_sell"]
    levels["hawkes_asymmetry"] = params_df["alpha_bb"] - params_df["alpha_ss"]

    # Reindex to 15m grid covering the full 1h index, then forward-fill up
    # to 45 minutes within each 1h bucket.
    grid_15m = pd.date_range(
        levels.index.min(),
        levels.index.max() + pd.Timedelta(minutes=45),
        freq="15min",
    )
    levels_15m = levels.reindex(grid_15m).ffill(limit=3)

    # Deltas via .diff(4) = 1-hour lookback on 15m grid (strictly past).
    deltas = pd.DataFrame(index=grid_15m)
    deltas["hawkes_branching_buy_d1h"] = levels_15m["hawkes_branching_buy"].diff(4)
    deltas["hawkes_branching_sell_d1h"] = levels_15m["hawkes_branching_sell"].diff(4)

    out = pd.concat([levels_15m, deltas], axis=1)[list(ALL_FEATURES)]
    return out


def write_15m_sidecar(params_df: pd.DataFrame) -> pd.DataFrame:
    feats_15m = derive_15m_features(params_df)
    # Also write 1h-cadence frame (raw features pre-ffill) for diagnostics
    feats_1h = pd.DataFrame(index=params_df.index)
    feats_1h["hawkes_mu_buy"] = params_df["mu_b"]
    feats_1h["hawkes_mu_sell"] = params_df["mu_s"]
    feats_1h["hawkes_branching_buy"] = params_df["branching_buy"]
    feats_1h["hawkes_branching_sell"] = params_df["branching_sell"]
    feats_1h["hawkes_asymmetry"] = params_df["alpha_bb"] - params_df["alpha_ss"]
    feats_1h["hawkes_branching_buy_d1h"] = feats_1h["hawkes_branching_buy"].diff(1)
    feats_1h["hawkes_branching_sell_d1h"] = feats_1h["hawkes_branching_sell"].diff(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feats_1h.to_parquet(FEATURES_1H_PATH)
    feats_15m.to_parquet(FEATURES_15M_PATH)

    coverage = feats_15m.notna().mean()
    logger.info("15m sidecar coverage:\n%s", coverage.to_string())
    logger.info("wrote %s (%d rows)", FEATURES_1H_PATH, len(feats_1h))
    logger.info("wrote %s (%d rows)", FEATURES_15M_PATH, len(feats_15m))
    return feats_15m


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", default="2021-12", help="first month (YYYY-MM)")
    parser.add_argument("--end", default="2026-03", help="last month (YYYY-MM)")
    parser.add_argument("--workers", type=int, default=4, help="parallel workers")
    parser.add_argument("--trade-dir", default=str(TRADE_DIR), help="aggTrades dir")
    parser.add_argument("--symbol", default=SYMBOL, help="symbol (e.g. BTCUSDT)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="if the params parquet already exists with full coverage, skip build",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.skip_existing and PARAMS_PATH.exists():
        existing = pd.read_parquet(PARAMS_PATH)
        logger.info("existing params parquet at %s has %d rows; skipping build",
                    PARAMS_PATH, len(existing))
        write_15m_sidecar(existing)
        return 0

    t0 = time.time()
    params_df = build_sidecar(
        args.start, args.end,
        workers=args.workers,
        trade_dir=Path(args.trade_dir),
        symbol=args.symbol,
    )
    t_build = time.time() - t0
    logger.info("build_sidecar took %.1fs for %d rows", t_build, len(params_df))

    feats_15m = write_15m_sidecar(params_df)
    logger.info("total wall clock: %.1fs", time.time() - t0)

    logger.info("\n=== Coverage summary ===")
    for c in feats_15m.columns:
        cov = feats_15m[c].notna().mean()
        logger.info("  %s: %.1f%%", c, 100 * cov)
    return 0


if __name__ == "__main__":
    sys.exit(main())
