"""One-off driver to build the rough-Heston sidecar parquet for Session F.

For each 1-hour bucket timestamp T in the target window (default 2021-12 →
2026-03), calibrate the 4-parameter rough-Heston model (V_0, H, eta, rho,
with kappa + theta pinned per Bayer-Friz-Gatheral 2016 convention) to the
trade-by-trade Deribit BTC options panel in the ±1h window around T.

Runtime on 4 workers: ~30-45 min (warm-started calibration ~240ms/bucket,
44K buckets total).

Outputs
-------
- ``vol_arb/output/rough_heston_params_1h.parquet`` — 1-hour sidecar with
  6 param cols + diagnostics (RMSE, n_iter, convergence flag, calibration
  level, n_options).
- ``vol_arb/output/rough_heston_features_15m.parquet`` — 15m gauntlet-facing
  sidecar with 4 level features (rh_v0, rh_hurst, rh_eta, rh_rho) and 4
  1-hour-delta features (computed via `.diff(4)` on the 15m ffill).

CLI
---
Full build::

    python -m vol_arb.build_rough_heston_sidecar \\
        --start 2021-12 --end 2026-03 --workers 4

Smoke test (1 month)::

    python -m vol_arb.build_rough_heston_sidecar \\
        --start 2023-07 --end 2023-07 --workers 1
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.rough_heston import (
    OPTION_DTYPE,
    RoughHestonParams,
    _bs_vega,
    calibrate_rough_heston,
)

logger = logging.getLogger("rough_heston_sidecar")


# ── Constants ────────────────────────────────────────────────────────

OPTIONS_DIR = Path("data/deribit/trades/BTC_option")
OUTPUT_DIR = Path("vol_arb/output")
PARAMS_PATH = OUTPUT_DIR / "rough_heston_params_1h.parquet"
FEATURES_1H_PATH = OUTPUT_DIR / "rough_heston_features_1h.parquet"
FEATURES_15M_PATH = OUTPUT_DIR / "rough_heston_features_15m.parquet"

BUCKET_FREQ = "1h"
WINDOW_SIDE_MINUTES = 30   # ±30min around bucket → 1h window width
DTE_MIN_DAYS = 0.5
DTE_MAX_DAYS = 90.0

# Instrument-name parser: "BTC-{EXPIRY}-{STRIKE}-{C|P}"
_INSTRUMENT_RE = re.compile(r"^BTC-(\d{1,2}[A-Z]{3}\d{2,4})-(\d+(?:\.\d+)?)-([CP])$")
_EXPIRY_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _parse_instrument(name: str) -> tuple[pd.Timestamp, float, str] | None:
    """Parse 'BTC-DDMMMYY-STRIKE-C|P' into (expiry_UTC, strike, option_type).

    Deribit expiry is 08:00 UTC on the expiry date.
    """
    m = _INSTRUMENT_RE.match(name)
    if m is None:
        return None
    expiry_str, strike_str, opt = m.group(1), m.group(2), m.group(3)
    # expiry_str: e.g. "28JUL23" or "28JUL2023"
    if re.match(r"^\d{1,2}[A-Z]{3}\d{2}$", expiry_str):
        # 2-digit year
        day = int(expiry_str[:-5])
        mon = _EXPIRY_MONTHS.get(expiry_str[-5:-2])
        yr = 2000 + int(expiry_str[-2:])
    else:
        # 4-digit year
        day = int(expiry_str[:-7])
        mon = _EXPIRY_MONTHS.get(expiry_str[-7:-4])
        yr = int(expiry_str[-4:])
    if mon is None:
        return None
    try:
        expiry = pd.Timestamp(year=yr, month=mon, day=day, hour=8, tz="UTC")
    except (ValueError, TypeError):
        return None
    return expiry, float(strike_str), opt


# ── Per-day processing ───────────────────────────────────────────────


def _load_day_options(date: str, options_dir: Path = OPTIONS_DIR) -> pd.DataFrame:
    """Load one day of Deribit option trades, parse instrument names.

    Returns DataFrame with columns: (timestamp, price, iv, instrument_name,
    index_price, expiry, strike, option_type, tau_days). Trades with
    unparseable names are dropped.
    """
    path = options_dir / f"{date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    # Parse instrument
    parsed = df["instrument_name"].apply(_parse_instrument)
    df = df[parsed.notna()].copy()
    if df.empty:
        return df
    parsed = parsed[parsed.notna()]
    df["expiry"] = pd.Series([p[0] for p in parsed], index=df.index)
    df["strike"] = pd.Series([p[1] for p in parsed], index=df.index)
    df["option_type"] = pd.Series([p[2] for p in parsed], index=df.index)
    # tau in days (exclude calendar day 0.5 floor)
    df["tau_days"] = (df["expiry"] - df["timestamp"]).dt.total_seconds() / 86400.0
    df = df[(df["tau_days"] >= DTE_MIN_DAYS) & (df["tau_days"] <= DTE_MAX_DAYS)]
    # Cast numeric columns
    df["iv"] = df["iv"].astype(float) / 100.0  # Deribit IV is in percent
    df["index_price"] = df["index_price"].astype(float)
    df = df[df["iv"] > 0.01]  # drop trades with absurd IV
    return df[
        ["timestamp", "iv", "strike", "option_type", "tau_days", "index_price"]
    ].reset_index(drop=True)


def _panel_for_bucket(
    trades: pd.DataFrame,
    bucket_end: pd.Timestamp,
    window_side_min: int = WINDOW_SIDE_MINUTES,
) -> np.ndarray | None:
    """Aggregate trades within ±window_side_min of bucket_end into a panel.

    For each (tau, strike), takes the MEDIAN IV across trades in the window.
    Returns a structured OPTION_DTYPE array or None if the panel is empty /
    insufficient for calibration.
    """
    if trades.empty:
        return None
    t_start = bucket_end - pd.Timedelta(minutes=window_side_min)
    t_end = bucket_end
    mask = (trades["timestamp"] >= t_start) & (trades["timestamp"] < t_end)
    window = trades[mask]
    if len(window) < 8:  # need minimum trades to define a smile
        return None
    # Forward from the most recent index_price in the window
    F = float(window["index_price"].iloc[-1])
    if not (F > 0 and np.isfinite(F)):
        return None
    # Aggregate by (tau_days rounded to nearest day, strike, option_type)
    window = window.copy()
    window["tau_year"] = window["tau_days"].values / 365.25
    grp = window.groupby(["tau_year", "strike", "option_type"], sort=False)
    agg = grp["iv"].median().reset_index()

    # Filter by moneyness: keep options within ±40% of forward (Deribit-liquid region)
    agg["moneyness"] = np.log(agg["strike"].astype(float).values / F)
    agg = agg[np.abs(agg["moneyness"]) <= 0.4]
    if len(agg) < 8:
        return None

    # For strikes where BOTH call and put trade, prefer the OTM leg (call at
    # K>F, put at K<F) — tighter vega, less bid-ask. Dedupe by (tau, strike).
    agg["is_otm"] = (
        ((agg["option_type"] == "C") & (agg["strike"] > F))
        | ((agg["option_type"] == "P") & (agg["strike"] < F))
    )
    # Sort so OTM comes first within each (tau, strike) group
    agg = agg.sort_values(["tau_year", "strike", "is_otm"], ascending=[True, True, False])
    agg = agg.drop_duplicates(subset=["tau_year", "strike"], keep="first")

    rows = []
    for _, r in agg.iterrows():
        tau_year = float(r["tau_year"])
        K = float(r["strike"])
        iv = float(r["iv"])
        if not (0.05 < iv < 5.0):
            continue
        # Vega weight for loss (we calibrate against CALL-equivalent IVs;
        # BS vega is the same for put and call at same K, tau, sigma)
        vega = _bs_vega(F, K, tau_year, iv)
        # Extra quality: downweight very far-OTM options (low liquidity)
        mk = abs(np.log(K / F))
        quality = np.exp(-(mk / 0.25) ** 2)  # Gaussian taper around ATM
        rows.append((tau_year, K, iv, vega * quality))
    if len(rows) < 8:
        return None
    panel = np.array(rows, dtype=OPTION_DTYPE)
    return panel, F


def _process_month(
    month: str,
    prev_month: str | None,
    options_dir: Path = OPTIONS_DIR,
) -> pd.DataFrame:
    """Fit rough-Heston on 1h-window panels for every hourly bucket in ``month``.

    Strategy: warm-start across successive buckets (uses 238ms fast-path).
    First bucket of the month uses global search (15s) to escape any bad
    init; subsequent buckets warm-start from previous fit.
    """
    t0 = time.time()
    # Load all trade days for this month + previous month's last day (for
    # the 30-min look-back of the first bucket of each month).
    days_list = []
    if prev_month is not None:
        # Last day of prev_month
        try:
            prev_start = pd.Timestamp(prev_month + "-01")
            prev_end = prev_start + pd.DateOffset(months=1)
            last_day = (prev_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            df_last = _load_day_options(last_day, options_dir)
            if not df_last.empty:
                days_list.append(df_last)
        except Exception:  # noqa: BLE001
            pass

    month_start = pd.Timestamp(month + "-01", tz="UTC")
    month_end = month_start + pd.DateOffset(months=1)
    for d in pd.date_range(month_start, month_end - pd.Timedelta(days=1), freq="D"):
        df_day = _load_day_options(d.strftime("%Y-%m-%d"), options_dir)
        if not df_day.empty:
            days_list.append(df_day)

    if not days_list:
        return pd.DataFrame()

    trades = pd.concat(days_list, ignore_index=True).sort_values("timestamp")

    buckets = pd.date_range(
        month_start + pd.Timedelta(hours=1),
        month_end,
        freq=BUCKET_FREQ,
        inclusive="left",
    )

    rows: list[dict] = []
    prev_fit: RoughHestonParams | None = None
    cold_count = 0
    warm_count = 0
    for T in buckets:
        res = _panel_for_bucket(trades, T, window_side_min=WINDOW_SIDE_MINUTES)
        row: dict = {"timestamp": T}
        if res is None:
            # Empty or insufficient panel
            for c in ("V_0", "H", "kappa", "theta", "eta", "rho"):
                row[c] = np.nan
            row["calibration_rmse"] = np.nan
            row["n_options"] = 0
            row["converged"] = False
            row["n_iter"] = 0
            row["calibration_level"] = 0
            rows.append(row)
            continue

        panel, F = res
        # Use global search if no warm start or previous fit failed
        use_global = prev_fit is None
        if use_global:
            cold_count += 1
        else:
            warm_count += 1

        try:
            fit = calibrate_rough_heston(
                panel, F=F,
                init=prev_fit if prev_fit is not None else None,
                max_iter=40,
                fix_kappa_theta=True,
                use_global_search=use_global,
            )
        except Exception:  # noqa: BLE001
            fit = RoughHestonParams(calibration_level=0)

        if fit.calibration_level > 0 and fit.calibration_rmse < 0.15:
            prev_fit = fit  # Good fit — warm-start next bucket
        else:
            prev_fit = None  # Cold-start next bucket

        d = asdict(fit)
        for c in ("V_0", "H", "kappa", "theta", "eta", "rho"):
            row[c] = d[c]
        row["calibration_rmse"] = d["calibration_rmse"]
        row["n_options"] = d["n_options"]
        row["converged"] = d["converged"]
        row["n_iter"] = d["n_iter"]
        row["calibration_level"] = d["calibration_level"]
        rows.append(row)

    out = pd.DataFrame(rows).set_index("timestamp")
    n_ok = int(out["calibration_level"].gt(0).sum())
    t_elapsed = time.time() - t0
    logger.info(
        "month %s: %d buckets, %d ok (%.1f%%), cold=%d warm=%d, %.1fs wall",
        month, len(out), n_ok, 100 * n_ok / max(len(out), 1),
        cold_count, warm_count, t_elapsed,
    )
    return out


# ── 15m feature derivation ───────────────────────────────────────────


LEVEL_FEATURES = ("rh_v0", "rh_hurst", "rh_eta", "rh_rho")
DELTA_FEATURES = ("rh_v0_d1h", "rh_hurst_d1h", "rh_eta_d1h", "rh_rho_d1h")
ALL_FEATURES = LEVEL_FEATURES + DELTA_FEATURES


def derive_15m_features(params_df: pd.DataFrame) -> pd.DataFrame:
    """Derive the 15m gauntlet sidecar from the 1h params parquet.

    Columns:
      Levels: rh_v0, rh_hurst, rh_eta, rh_rho (NaN where calibration failed)
      Deltas: .diff(4) of each level on the 15m grid (1h lookback, strict past)
    """
    if params_df.empty:
        return pd.DataFrame(columns=list(ALL_FEATURES))

    levels = pd.DataFrame(index=params_df.index)
    bad = params_df["calibration_level"].eq(0) | params_df["calibration_rmse"].gt(0.10)
    levels["rh_v0"] = params_df["V_0"].where(~bad, np.nan)
    levels["rh_hurst"] = params_df["H"].where(~bad, np.nan)
    levels["rh_eta"] = params_df["eta"].where(~bad, np.nan)
    levels["rh_rho"] = params_df["rho"].where(~bad, np.nan)

    # Reindex to 15m grid and forward-fill within the 1h bucket (up to 45min)
    grid_15m = pd.date_range(
        levels.index.min(),
        levels.index.max() + pd.Timedelta(minutes=45),
        freq="15min",
    )
    levels_15m = levels.reindex(grid_15m).ffill(limit=3)

    deltas = pd.DataFrame(index=grid_15m)
    for src in LEVEL_FEATURES:
        deltas[f"{src}_d1h"] = levels_15m[src].diff(4)

    out = pd.concat([levels_15m, deltas], axis=1)[list(ALL_FEATURES)]
    return out


# ── Driver ───────────────────────────────────────────────────────────


def build_sidecar(
    start_month: str,
    end_month: str,
    *,
    workers: int = 4,
    options_dir: Path = OPTIONS_DIR,
) -> pd.DataFrame:
    """Build the 1h rough-Heston sidecar for [start_month, end_month]."""
    avail_files = sorted(options_dir.glob("*.parquet"))
    avail_months = sorted({f.stem[:7] for f in avail_files})
    months = [m for m in avail_months if start_month <= m <= end_month]
    if not months:
        raise RuntimeError(
            f"no months found in range [{start_month}, {end_month}] "
            f"under {options_dir}"
        )
    logger.info("processing %d months with %d workers", len(months), workers)

    prev_of: dict[str, str | None] = {}
    for m in months:
        idx = avail_months.index(m)
        prev_of[m] = avail_months[idx - 1] if idx > 0 else None

    frames: list[pd.DataFrame] = []
    if workers <= 1:
        for m in months:
            frames.append(_process_month(m, prev_of[m], options_dir=options_dir))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fut_to_month = {
                pool.submit(_process_month, m, prev_of[m], options_dir): m
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
    all_df = all_df[~all_df.index.duplicated(keep="first")]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(PARAMS_PATH)
    logger.info(
        "wrote %s (%d rows, %d with calibration_level>0)",
        PARAMS_PATH, len(all_df), int(all_df["calibration_level"].gt(0).sum()),
    )
    return all_df


def write_15m_sidecar(params_df: pd.DataFrame) -> pd.DataFrame:
    feats_15m = derive_15m_features(params_df)

    # Also write 1h-cadence frame for diagnostics
    feats_1h = pd.DataFrame(index=params_df.index)
    bad = params_df["calibration_level"].eq(0) | params_df["calibration_rmse"].gt(0.10)
    feats_1h["rh_v0"] = params_df["V_0"].where(~bad, np.nan)
    feats_1h["rh_hurst"] = params_df["H"].where(~bad, np.nan)
    feats_1h["rh_eta"] = params_df["eta"].where(~bad, np.nan)
    feats_1h["rh_rho"] = params_df["rho"].where(~bad, np.nan)
    for src in LEVEL_FEATURES:
        feats_1h[f"{src}_d1h"] = feats_1h[src].diff(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feats_1h.to_parquet(FEATURES_1H_PATH)
    feats_15m.to_parquet(FEATURES_15M_PATH)

    coverage = feats_15m.notna().mean()
    logger.info("15m coverage per column:\n%s", coverage.to_string())
    logger.info("wrote %s (%d rows)", FEATURES_15M_PATH, len(feats_15m))
    return feats_15m


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", default="2021-12")
    parser.add_argument("--end", default="2026-03")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--options-dir", default=str(OPTIONS_DIR))
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="if the params parquet exists, skip build and rewrite 15m features only",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.skip_existing and PARAMS_PATH.exists():
        existing = pd.read_parquet(PARAMS_PATH)
        logger.info("existing params at %s has %d rows; skipping build",
                    PARAMS_PATH, len(existing))
        write_15m_sidecar(existing)
        return 0

    t0 = time.time()
    params_df = build_sidecar(
        args.start, args.end,
        workers=args.workers,
        options_dir=Path(args.options_dir),
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
