"""TSRV Label Quality Diagnostic — Phase 0.

Quantifies:
1. Negative TSRV frequency (pre-clip)
2. Subgrid averaging variance reduction
3. Noise-to-signal ratio per window
4. Jump contamination (BPV on sub-bar returns)
5. Finite-sample bias magnitude
6. Label distribution analysis

Run:
    PYTHONPATH=. python TSRV/diagnostic.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from TSRV.config import TSRVConfig
from TSRV.data_loader import iter_trade_days
from TSRV.sampling import _accumulate_bars

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Estimator primitives ──────────────────────────────────────

def bipower_variation(log_prices: np.ndarray) -> float:
    """BPV on log-price array (Barndorff-Nielsen & Shephard 2004).

    BPV = (pi/2) * (n/(n-1)) * sum(|r_{i-1}| * |r_i|)
    Converges to integrated variance, robust to finite-activity jumps.
    """
    n = len(log_prices) - 1
    if n < 3:
        return 0.0
    returns = np.diff(log_prices)
    mu1_inv_sq = np.pi / 2  # (sqrt(2/pi))^{-2}
    bpv = mu1_inv_sq * np.sum(np.abs(returns[1:]) * np.abs(returns[:-1]))
    return float(bpv * n / (n - 1))


def medrv(log_prices: np.ndarray) -> float:
    """MedRV (Andersen, Dobrev, Schaumburg 2012).

    MedRV = c * (M/(M-2)) * sum median(|r_{i-1}|, |r_i|, |r_{i+1}|)^2
    More robust than BPV to clustered jumps.
    """
    n = len(log_prices) - 1
    if n < 4:
        return 0.0
    returns = np.abs(np.diff(log_prices))
    c = np.pi / (6 - 4 * np.sqrt(3) + np.pi)  # ~2.610
    M = len(returns)
    med_sum = 0.0
    for i in range(1, M - 1):
        med_sum += np.median([returns[i - 1], returns[i], returns[i + 1]]) ** 2
    return float(c * (M / (M - 2)) * med_sum)


def subgrid_tsrv(log_prices: np.ndarray, sub_bar_indices: np.ndarray, K: int) -> float:
    """Proper K-subgrid-averaged TSRV (Zhang et al. 2005).

    Averages RV(slow) over K offset sub-grids of dollar sub-bar indices.
    Includes finite-sample bias correction.
    """
    n = len(log_prices) - 1
    if n < 2:
        return 0.0
    rv_fast = float(np.sum(np.diff(log_prices) ** 2))

    M = len(sub_bar_indices)
    if M < 2:
        return rv_fast

    K = min(K, M // 2) if M >= 4 else 1
    if K < 1:
        K = 1

    rv_slow_sum = 0.0
    valid_grids = 0
    for k in range(K):
        grid_idx = sub_bar_indices[k::K]
        if len(grid_idx) < 2:
            continue
        sub_ret = np.diff(log_prices[grid_idx])
        rv_slow_sum += float(np.sum(sub_ret ** 2))
        valid_grids += 1

    if valid_grids == 0:
        return rv_fast

    rv_slow_avg = rv_slow_sum / valid_grids
    n_bar_avg = (M - K + 1) / K
    tsrv_raw = rv_slow_avg - (n_bar_avg / n) * rv_fast

    # Finite-sample bias correction (Zhang 2005 Eq.25)
    denom = 1.0 - n_bar_avg / n
    if denom > 0.01:
        tsrv_corrected = tsrv_raw / denom
    else:
        tsrv_corrected = tsrv_raw

    return tsrv_corrected


# ── Per-window analysis ───────────────────────────────────────

def analyze_window(prices: np.ndarray, dollar_vol: np.ndarray,
                   threshold: float) -> dict | None:
    """Compute all diagnostic metrics for one window."""
    n_ticks = len(prices)
    if n_ticks < 10:
        return None

    log_p = np.log(prices)
    rv_fast = float(np.sum(np.diff(log_p) ** 2))

    # Dollar sub-bars
    sub_bar_indices = _accumulate_bars(dollar_vol, threshold)
    if len(sub_bar_indices) < 2:
        return None

    sub_idx = np.array(sub_bar_indices)
    n_sub = len(sub_idx)

    # RV(slow) — single grid (current implementation)
    sub_log_p = log_p[sub_idx]
    rv_slow = float(np.sum(np.diff(sub_log_p) ** 2))

    # Current simplified TSRV
    n_dash = (n_ticks - 1) / n_sub
    tsrv_simple = rv_slow - (1.0 / n_dash) * rv_fast

    # Proper K-subgrid TSRV
    K = max(1, int(np.ceil(n_sub ** (2 / 3))))
    tsrv_proper = subgrid_tsrv(log_p, sub_idx, K)

    # BPV on sub-bar returns
    bpv = bipower_variation(sub_log_p)
    mrv = medrv(sub_log_p)

    # Jump decomposition
    continuous_bpv = min(rv_slow, bpv)
    jump_bpv = max(rv_slow - bpv, 0.0)
    continuous_med = min(rv_slow, mrv)
    jump_med = max(rv_slow - mrv, 0.0)

    # Noise estimate (Hansen & Lunde)
    noise_var = rv_fast / (2 * (n_ticks - 1)) if n_ticks > 1 else 0.0

    # Bias correction factor
    n_bar_avg = (n_sub - K + 1) / K
    bias_factor = 1.0 / (1.0 - n_bar_avg / (n_ticks - 1)) if (n_ticks - 1) > n_bar_avg else 1.0

    return {
        "n_ticks": n_ticks,
        "n_sub_bars": n_sub,
        "K": K,
        "rv_fast": rv_fast,
        "rv_slow": rv_slow,
        "tsrv_simple": tsrv_simple,
        "tsrv_proper": tsrv_proper,
        "tsrv_simple_clipped": max(tsrv_simple, 0.0),
        "bpv": bpv,
        "medrv": mrv,
        "continuous_bpv": continuous_bpv,
        "jump_bpv": jump_bpv,
        "continuous_med": continuous_med,
        "jump_med": jump_med,
        "jump_fraction_bpv": jump_bpv / max(rv_slow, 1e-20),
        "jump_fraction_med": jump_med / max(rv_slow, 1e-20),
        "noise_var": noise_var,
        "noise_ratio": noise_var * 2 * (n_ticks - 1) / max(tsrv_simple, 1e-20),
        "bias_factor": bias_factor,
    }


# ── Main diagnostic ──────────────────────────────────────────

def run_diagnostic(
    trade_dir: str = "data/binance-aggTrades-parquet",
    symbol: str = "BTCUSDT",
    threshold: float = 500_000,
    slow_tf_s: int = 900,
    max_windows: int = 2000,
) -> pd.DataFrame:
    """Run full diagnostic on sampled windows."""
    freq = pd.Timedelta(seconds=slow_tf_s)
    rows = []
    window_count = 0

    logger.info("Sampling up to %d windows (threshold=$%.0fK, tf=%ds)...",
                max_windows, threshold / 1000, slow_tf_s)

    for date_str, day_df in iter_trade_days(trade_dir, symbol):
        if len(day_df) < 100:
            continue

        prices = day_df["price"].values.astype(np.float64)
        dollar_vol = day_df["dollar_volume"].values.astype(np.float64)
        timestamps = day_df.index

        day_start = timestamps[0].floor(freq)
        day_end = timestamps[-1]
        window_start = day_start

        while window_start < day_end:
            window_end = window_start + freq
            mask = (timestamps >= window_start) & (timestamps < window_end)
            w_idx = np.where(mask)[0]

            if len(w_idx) >= 10:
                result = analyze_window(prices[w_idx], dollar_vol[w_idx], threshold)
                if result is not None:
                    result["timestamp"] = window_start
                    rows.append(result)
                    window_count += 1

            window_start = window_end

            if window_count >= max_windows:
                break
        if window_count >= max_windows:
            break

        if window_count % 500 == 0 and window_count > 0:
            logger.info("  %d windows analyzed...", window_count)

    logger.info("Done: %d windows analyzed", window_count)
    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame, tsrv_path: str | None = None) -> None:
    """Print comprehensive diagnostic report."""
    print("\n" + "=" * 70)
    print("  TSRV LABEL QUALITY DIAGNOSTIC")
    print("=" * 70)

    n = len(df)
    print(f"\n  Windows analyzed: {n}")
    print(f"  Mean ticks/window: {df['n_ticks'].mean():.0f}")
    print(f"  Mean sub-bars/window: {df['n_sub_bars'].mean():.0f}")
    print(f"  Mean K (subgrids): {df['K'].mean():.0f}")

    # ── 1. Negative TSRV frequency ────────────────────────────
    print(f"\n{'-' * 70}")
    print("  1. NEGATIVE TSRV FREQUENCY (pre-clip)")
    print(f"{'-' * 70}")
    neg_simple = (df["tsrv_simple"] < 0).sum()
    neg_proper = (df["tsrv_proper"] < 0).sum()
    print(f"  Simplified TSRV < 0:  {neg_simple:>5d} / {n} ({100*neg_simple/n:.2f}%)")
    print(f"  Proper TSRV < 0:      {neg_proper:>5d} / {n} ({100*neg_proper/n:.2f}%)")
    if neg_simple > 0:
        neg_vals = df.loc[df["tsrv_simple"] < 0, "tsrv_simple"]
        print(f"  Negative magnitude:   mean={neg_vals.mean():.6f}, min={neg_vals.min():.6f}")
        neg_subs = df.loc[df["tsrv_simple"] < 0, "n_sub_bars"]
        print(f"  Negative window sub-bars: mean={neg_subs.mean():.0f} (vs overall {df['n_sub_bars'].mean():.0f})")

    # ── 2. Subgrid variance reduction ─────────────────────────
    print(f"\n{'-' * 70}")
    print("  2. SUBGRID AVERAGING VARIANCE REDUCTION")
    print(f"{'-' * 70}")
    diff = df["tsrv_proper"] - df["tsrv_simple"]
    corr = df["tsrv_simple"].corr(df["tsrv_proper"])
    print(f"  Correlation (simple vs proper): {corr:.6f}")
    print(f"  Mean difference (proper - simple): {diff.mean():.8f}")
    print(f"  Std of difference: {diff.std():.8f}")
    # Rolling variance comparison
    var_simple = df["tsrv_simple"].rolling(20, min_periods=10).std()
    var_proper = df["tsrv_proper"].rolling(20, min_periods=10).std()
    ratio = (var_proper / var_simple.clip(lower=1e-15)).dropna()
    print(f"  Variance ratio (proper/simple): mean={ratio.mean():.4f}, median={ratio.median():.4f}")
    if ratio.mean() < 0.95:
        print(f"  -> Subgrid averaging REDUCES variance by {100*(1-ratio.mean()):.1f}%")
    else:
        print(f"  -> Subgrid averaging has MINIMAL effect ({100*(1-ratio.mean()):.1f}% change)")

    # ── 3. Noise-to-signal ratio ──────────────────────────────
    print(f"\n{'-' * 70}")
    print("  3. NOISE-TO-SIGNAL RATIO")
    print(f"{'-' * 70}")
    nr = df["noise_ratio"]
    print(f"  Mean:   {nr.mean():.4f}")
    print(f"  Median: {nr.median():.4f}")
    print(f"  P90:    {nr.quantile(0.90):.4f}")
    print(f"  P99:    {nr.quantile(0.99):.4f}")
    high_noise = (nr > 0.5).sum()
    print(f"  Windows with noise_ratio > 0.5: {high_noise} ({100*high_noise/n:.2f}%)")

    # ── 4. Jump contamination ─────────────────────────────────
    print(f"\n{'-' * 70}")
    print("  4. JUMP CONTAMINATION")
    print(f"{'-' * 70}")
    jf_bpv = df["jump_fraction_bpv"]
    jf_med = df["jump_fraction_med"]
    print(f"  Jump fraction (BPV):  mean={jf_bpv.mean():.4f}, median={jf_bpv.median():.4f}")
    print(f"  Jump fraction (MedRV): mean={jf_med.mean():.4f}, median={jf_med.median():.4f}")

    # RV/BPV ratio test
    rv_bpv_ratio = df["rv_slow"] / df["bpv"].clip(lower=1e-20)
    jump_windows_bpv = (rv_bpv_ratio > 1.5).sum()
    jump_windows_med = (df["rv_slow"] / df["medrv"].clip(lower=1e-20) > 1.5).sum()
    print(f"  Windows with RV/BPV > 1.5 (jumps):  {jump_windows_bpv} ({100*jump_windows_bpv/n:.1f}%)")
    print(f"  Windows with RV/MedRV > 1.5 (jumps): {jump_windows_med} ({100*jump_windows_med/n:.1f}%)")
    print(f"  BPV vs MedRV correlation: {df['bpv'].corr(df['medrv']):.4f}")

    # ── 5. Bias magnitude ─────────────────────────────────────
    print(f"\n{'-' * 70}")
    print("  5. FINITE-SAMPLE BIAS")
    print(f"{'-' * 70}")
    bf = df["bias_factor"]
    print(f"  Bias correction factor: mean={bf.mean():.6f}, range=[{bf.min():.6f}, {bf.max():.6f}]")
    pct_bias = 100 * (bf.mean() - 1)
    print(f"  Systematic upward bias: {pct_bias:.3f}%")

    # ── 6. Label distribution ─────────────────────────────────
    print(f"\n{'-' * 70}")
    print("  6. LABEL DISTRIBUTION")
    print(f"{'-' * 70}")

    if tsrv_path and Path(tsrv_path).exists():
        tsrv_df = pd.read_parquet(tsrv_path)
        valid = tsrv_df[tsrv_df["vol_ann"] > 0]
        ln_vol = np.log(valid["vol_ann"])

        print(f"  Total windows: {len(tsrv_df)}")
        print(f"  Zero vol_ann: {(tsrv_df['vol_ann'] == 0).sum()}")
        print(f"  ln(vol_ann) stats:")
        print(f"    mean={ln_vol.mean():.4f}, std={ln_vol.std():.4f}")
        print(f"    min={ln_vol.min():.4f}, max={ln_vol.max():.4f}")
        print(f"    skew={ln_vol.skew():.4f}, kurtosis={ln_vol.kurtosis():.4f}")

        # Normality test
        if len(ln_vol) > 100:
            _, p_normal = stats.normaltest(ln_vol.dropna())
            print(f"    D'Agostino normality p={p_normal:.4e} ({'normal' if p_normal > 0.05 else 'non-normal'})")

        # Autocorrelation
        acf_1 = ln_vol.autocorr(1)
        acf_5 = ln_vol.autocorr(5)
        acf_20 = ln_vol.autocorr(20)
        print(f"    ACF(1)={acf_1:.4f}, ACF(5)={acf_5:.4f}, ACF(20)={acf_20:.4f}")
        print(f"    -> {'Long memory (slow decay)' if acf_20 > 0.3 else 'Moderate persistence'}")

    # ── Summary & Recommendations ─────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY & PHASE PRIORITIZATION")
    print(f"{'=' * 70}")

    recommendations = []
    if neg_simple / n > 0.02:
        recommendations.append(("HIGH", "Phase 3 (RK)", f"Negative TSRV = {100*neg_simple/n:.1f}% — non-negativity matters"))
    if ratio.mean() < 0.90:
        recommendations.append(("HIGH", "Phase 1 (subgrid)", f"Variance reduction = {100*(1-ratio.mean()):.1f}%"))
    elif ratio.mean() < 0.97:
        recommendations.append(("MEDIUM", "Phase 1 (subgrid)", f"Variance reduction = {100*(1-ratio.mean()):.1f}%"))
    else:
        recommendations.append(("LOW", "Phase 1 (subgrid)", f"Variance reduction = {100*(1-ratio.mean()):.1f}% — minimal"))

    if jump_windows_bpv / n > 0.10:
        recommendations.append(("HIGH", "Phase 2 (jumps)", f"{100*jump_windows_bpv/n:.1f}% of windows have jumps"))
    elif jump_windows_bpv / n > 0.05:
        recommendations.append(("MEDIUM", "Phase 2 (jumps)", f"{100*jump_windows_bpv/n:.1f}% of windows have jumps"))

    recommendations.append(("HIGH", "Phase 4 (features)", "Always high — jump/noise features add new information"))

    for priority, phase, reason in sorted(recommendations, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x[0]]):
        print(f"  [{priority:>6s}] {phase:<25s} {reason}")


if __name__ == "__main__":
    t0 = time.time()

    config = TSRVConfig()
    tsrv_path = f"TSRV/output/{config.asset}_tsrv_15m.parquet"

    df = run_diagnostic(
        trade_dir=config.trade_dir,
        symbol=config.asset,
        threshold=500_000,
        slow_tf_s=900,
        max_windows=2000,
    )

    # Save raw diagnostic data
    out_path = Path(config.output_dir) / "diagnostic.parquet"
    df.to_parquet(out_path)
    logger.info("Saved diagnostic data: %s", out_path)

    print_report(df, tsrv_path=tsrv_path)

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.0f}s")
