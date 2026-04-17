"""Rough memory features — Hurst exponent and fractionally differenced RV.

These features capture the memory structure of the volatility process:
- Hurst > 0.5: persistent (trending vol regime)
- Hurst < 0.5: anti-persistent (mean-reverting vol)
- FFD: stationary representation that preserves long memory

Both are strictly backward-looking — no leakage by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vol_model.config import VolModelConfig


def _hurst_rs(series: np.ndarray) -> float:
    """Compute Hurst exponent via R/S (Rescaled Range) analysis.

    Uses sub-sample sizes [8, 16, 32, 64] and regresses log(R/S) on log(n).
    Returns H estimate. If computation fails, returns 0.5 (random walk).
    """
    n = len(series)
    if n < 20:
        return 0.5

    # Sub-sample sizes (powers of 2 that fit within the series)
    max_k = int(np.log2(n))
    sizes = [2 ** i for i in range(3, max_k + 1) if 2 ** i <= n // 2]

    if len(sizes) < 2:
        return 0.5

    log_rs = []
    log_n = []

    for size in sizes:
        n_chunks = n // size
        if n_chunks < 1:
            continue

        rs_values = []
        for i in range(n_chunks):
            chunk = series[i * size: (i + 1) * size]
            if len(chunk) < 2:
                continue

            mean = chunk.mean()
            std = chunk.std(ddof=1)
            if std < 1e-15:
                continue

            cumdev = np.cumsum(chunk - mean)
            r = cumdev.max() - cumdev.min()
            rs_values.append(r / std)

        if rs_values:
            log_rs.append(np.log(np.mean(rs_values)))
            log_n.append(np.log(size))

    if len(log_rs) < 2:
        return 0.5

    # OLS regression: log(R/S) = H * log(n) + c
    log_n = np.array(log_n)
    log_rs = np.array(log_rs)
    slope = np.polyfit(log_n, log_rs, 1)[0]

    return float(np.clip(slope, 0.0, 1.0))


def compute_memory_features(config: VolModelConfig) -> pd.DataFrame:
    """Compute rough memory features from TSRV output.

    Features:
        hurst_100: Rolling Hurst exponent over last 100 bars of ln(vol_ann)
        ffd_rv: Fractionally differenced ln(vol_ann) (d=0.4)

    All features are strictly backward-looking (use only past completed bars).
    """
    from src.afml.fracdiff import frac_diff_ffd

    tsrv = pd.read_parquet(config.get_tsrv_path())
    valid = tsrv[tsrv["vol_ann"] > 0].copy()
    ln_vol = np.log(valid["vol_ann"])

    features = pd.DataFrame(index=valid.index)

    # 1. Rolling Hurst exponent
    window = config.hurst_window
    hurst_values = np.full(len(ln_vol), np.nan)
    values = ln_vol.values

    for i in range(window, len(values)):
        # Use returns (differences of log-vol) for Hurst
        chunk = values[i - window: i]
        returns = np.diff(chunk)
        hurst_values[i] = _hurst_rs(returns)

    features["hurst_100"] = hurst_values

    # 2. Fractionally differenced RV
    ln_vol_df = pd.DataFrame({"ln_vol": ln_vol})
    ffd = frac_diff_ffd(ln_vol_df, d=config.fracdiff_d, threshold=config.fracdiff_threshold)

    if len(ffd) > 0 and "ln_vol" in ffd.columns:
        features["ffd_rv"] = ffd["ln_vol"]
    else:
        features["ffd_rv"] = np.nan

    # 3. Rolling Z-score of TSRV — regime normalization
    # 60% ann. vol means different things in different regimes.
    # Z-score tells the model whether current vol is HIGH or LOW relative to recent history.
    tsrv_raw = valid["tsrv"]
    for window, label in [(96, "1d"), (672, "1w"), (2880, "1m")]:
        mu = tsrv_raw.rolling(window, min_periods=max(window // 4, 2)).mean()
        sigma = tsrv_raw.rolling(window, min_periods=max(window // 4, 2)).std()
        features[f"tsrv_zscore_{label}"] = (tsrv_raw - mu) / sigma.clip(lower=1e-15)

    return features
