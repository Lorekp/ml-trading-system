"""Compute volatility signature plots and find optimal sampling frequency."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from TSRV.config import TSRVConfig
from TSRV.data_loader import load_day, iter_trade_days
from TSRV.sampling import resample_time, resample_tick_bars, resample_dollar_bars
from TSRV.realized_vol import realized_variance, tsrv, noise_variance, annualize_rv

logger = logging.getLogger(__name__)


def compute_time_signature(
    tick_dir: str | Path,
    symbol: str,
    dates: list[str],
    intervals: list[float],
    config: TSRVConfig,
) -> pd.DataFrame:
    """Compute RV at each time interval, averaged over dates.

    Returns DataFrame with columns:
        interval, rv_mean, rv_std, rv_median, n_bars_mean, tsrv_mean, noise_var_mean
    """
    # results[interval_idx][date_idx] = rv
    results = {iv: [] for iv in intervals}
    tsrv_vals = []
    noise_vals = []

    for di, date in enumerate(dates):
        try:
            ticks = load_day(tick_dir, symbol, date)
        except Exception as e:
            logger.warning("Skip %s: %s", date, e)
            continue

        if len(ticks) < 100:
            continue

        # TSRV on raw ticks (benchmark)
        tsrv_vals.append(tsrv(ticks["mid"], K=config.tsrv_K))
        noise_vals.append(noise_variance(ticks["mid"]))

        for iv in intervals:
            sampled = resample_time(ticks, iv)
            if len(sampled) < 2:
                continue
            rv = realized_variance(sampled, config.rv_method)
            results[iv].append(rv)

        if (di + 1) % 30 == 0:
            logger.info("  Time sig: %d/%d dates", di + 1, len(dates))

        del ticks

    rows = []
    for iv in intervals:
        vals = np.array(results[iv])
        if len(vals) == 0:
            continue
        rows.append({
            "interval": iv,
            "rv_mean": float(vals.mean()),
            "rv_std": float(vals.std()),
            "rv_median": float(np.median(vals)),
            "n_days": len(vals),
        })

    df = pd.DataFrame(rows)
    if tsrv_vals:
        df.attrs["tsrv_mean"] = float(np.mean(tsrv_vals))
        df.attrs["noise_var_mean"] = float(np.mean(noise_vals))
    return df


def compute_tick_signature(
    tick_dir: str | Path,
    symbol: str,
    dates: list[str],
    thresholds: list[int],
    config: TSRVConfig,
) -> pd.DataFrame:
    """Compute RV at each tick-bar threshold, averaged over dates."""
    results = {t: [] for t in thresholds}

    for di, date in enumerate(dates):
        try:
            ticks = load_day(tick_dir, symbol, date)
        except Exception:
            continue
        if len(ticks) < 100:
            continue

        for t in thresholds:
            sampled = resample_tick_bars(ticks, t)
            if len(sampled) < 2:
                continue
            results[t].append(realized_variance(sampled, config.rv_method))

        if (di + 1) % 30 == 0:
            logger.info("  Tick sig: %d/%d dates", di + 1, len(dates))
        del ticks

    rows = []
    for t in thresholds:
        vals = np.array(results[t])
        if len(vals) == 0:
            continue
        rows.append({
            "threshold": t,
            "rv_mean": float(vals.mean()),
            "rv_std": float(vals.std()),
            "rv_median": float(np.median(vals)),
            "n_days": len(vals),
        })
    return pd.DataFrame(rows)


def compute_dollar_signature(
    trade_dir: str | Path,
    symbol: str,
    thresholds: list[float],
    config: TSRVConfig,
    max_days: int | None = None,
) -> pd.DataFrame:
    """Compute RV at each dollar-bar threshold using real aggTrades volume.

    Uses actual trade data (price * quantity) for true dollar bars.
    Processes one day at a time from monthly parquets.
    """
    results = {t: [] for t in thresholds}
    day_count = 0

    for date_str, day_df in iter_trade_days(trade_dir, symbol):
        if len(day_df) < 100:
            continue

        for t in thresholds:
            sampled = resample_dollar_bars(day_df, t)
            if len(sampled) < 2:
                continue
            results[t].append(realized_variance(sampled, config.rv_method))

        day_count += 1
        if day_count % 30 == 0:
            logger.info("  Dollar sig: %d days processed", day_count)

        if max_days and day_count >= max_days:
            break

        del day_df

    rows = []
    for t in thresholds:
        vals = np.array(results[t])
        if len(vals) == 0:
            continue
        rows.append({
            "threshold": t,
            "rv_mean": float(vals.mean()),
            "rv_std": float(vals.std()),
            "rv_median": float(np.median(vals)),
            "n_days": len(vals),
        })
    return pd.DataFrame(rows)


def compute_per_window_dollar_signature(
    trade_dir: str | Path,
    symbol: str,
    thresholds: list[float],
    window_minutes: int,
    config: TSRVConfig,
    max_days: int | None = None,
) -> pd.DataFrame:
    """Per-slow-window dollar-bar RV signature.

    Unlike :func:`compute_dollar_signature` which aggregates one RV per
    DAY per threshold, this computes RV per ``window_minutes``-wide slow
    bar (e.g., every 5-min window). Designed to find the TSRV-optimal
    threshold at a specific slow-bar HORIZON — the daily signature knee
    is correct for daily-RV estimation but NOT for TSRV at short
    slow-bar windows (where the Zhang (2006) asymptotic K* ≈ n^{2/3}
    optimum is different).

    For each dollar threshold, returns the per-window RV distribution
    plus the median sub-bars-per-window — the latter directly tells us
    whether K* is being met at this threshold / horizon combination.

    Parameters
    ----------
    trade_dir : path to daily aggTrades parquets
    symbol : e.g. "BTCUSDT"
    thresholds : dollar thresholds to sweep (USD)
    window_minutes : slow-bar duration in minutes (e.g., 5, 15, 60)
    config : TSRVConfig (uses config.rv_method)
    max_days : cap iteration for a fast approximation (e.g., 180 for 6mo)

    Returns
    -------
    pd.DataFrame with columns: threshold, rv_mean, rv_median, rv_std,
    n_windows, n_sub_bars_mean, n_sub_bars_median, n_sub_bars_p5,
    n_sub_bars_p95.
    """
    results: dict[float, list[float]] = {t: [] for t in thresholds}
    sub_bar_counts: dict[float, list[int]] = {t: [] for t in thresholds}
    day_count = 0
    window_label = f"{window_minutes}min"

    for date_str, day_df in iter_trade_days(trade_dir, symbol):
        if len(day_df) < 100:
            continue

        for t in thresholds:
            sampled = resample_dollar_bars(day_df, t)
            if len(sampled) < 2:
                continue

            # Group sampled bar closes by floor(timestamp, window_minutes)
            # so every slow-window gets an independent RV from its own
            # internal sub-bar returns. Consecutive sub-bars within a
            # window contribute one squared log-return each.
            window_ids = sampled.index.floor(window_label)
            for _, win_group in sampled.groupby(window_ids):
                if len(win_group) < 2:
                    continue
                rv = realized_variance(win_group, config.rv_method)
                results[t].append(rv)
                sub_bar_counts[t].append(len(win_group))

        day_count += 1
        if day_count % 30 == 0:
            total_windows = sum(len(v) for v in results.values())
            logger.info(
                "  Dollar sig (per %dmin window): %d days, %d cumulative windows",
                window_minutes, day_count, total_windows,
            )

        if max_days and day_count >= max_days:
            break

        del day_df

    rows = []
    for t in thresholds:
        vals = np.array(results[t], dtype=np.float64)
        sb = np.array(sub_bar_counts[t], dtype=np.float64)
        if len(vals) == 0:
            continue
        rows.append({
            "threshold": t,
            "rv_mean": float(vals.mean()),
            "rv_median": float(np.median(vals)),
            "rv_std": float(vals.std()),
            "n_windows": int(len(vals)),
            "n_sub_bars_mean": float(sb.mean()),
            "n_sub_bars_median": float(np.median(sb)),
            "n_sub_bars_p5": float(np.quantile(sb, 0.05)),
            "n_sub_bars_p95": float(np.quantile(sb, 0.95)),
        })
    return pd.DataFrame(rows)


def find_knee(x: np.ndarray, y: np.ndarray, config: TSRVConfig) -> dict:
    """Find optimal sampling point using KneeLocator.

    Returns dict with knee_x, knee_y, and the KneeLocator object.
    """
    from kneed import KneeLocator

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    kl = KneeLocator(
        x, y,
        S=config.knee_sensitivity,
        curve=config.knee_curve,
        direction=config.knee_direction,
        interp_method=config.knee_interp,
        polynomial_degree=config.knee_poly_degree,
    )
    return {
        "knee_x": kl.knee,
        "knee_y": kl.knee_y,
        "kl": kl,
    }


def plot_signature(
    sig_df: pd.DataFrame,
    x_col: str,
    knee_result: dict,
    title: str,
    output_path: Path,
    x_label: str = "Sampling interval",
    tsrv_ref: float | None = None,
) -> None:
    """Plot signature curve with knee annotated and save to file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    x = sig_df[x_col].values
    y = sig_df["rv_mean"].values
    y_std = sig_df["rv_std"].values

    # Annualized vol for readability
    y_vol = np.sqrt(y * 365.25)
    y_vol_std = 0.5 * y_std * 365.25 / (y_vol + 1e-12)

    ax.plot(x, y_vol, "o-", color="steelblue", linewidth=2, markersize=6, label="RV (annualized)")
    ax.fill_between(x, y_vol - y_vol_std, y_vol + y_vol_std,
                     alpha=0.2, color="steelblue")

    # Knee
    if knee_result.get("knee_x") is not None:
        kx = knee_result["knee_x"]
        ky_vol = np.sqrt(knee_result["knee_y"] * 365.25)
        ax.axvline(kx, color="red", linestyle="--", alpha=0.7, label=f"Knee = {kx:g}")
        ax.plot(kx, ky_vol, "D", color="red", markersize=10, zorder=5)

    # TSRV reference
    if tsrv_ref is not None and tsrv_ref > 0:
        tsrv_vol = np.sqrt(tsrv_ref * 365.25)
        ax.axhline(tsrv_vol, color="green", linestyle=":", alpha=0.7,
                    label=f"TSRV = {tsrv_vol:.1%}")

    ax.set_xscale("log")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Annualized Volatility", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_path)
