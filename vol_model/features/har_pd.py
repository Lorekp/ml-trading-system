"""HAR-PD path-dependent features (Xiao et al. 2025, arxiv:2503.00851).

HAR-PD augments the classic HAR-RV baseline with two path-dependent features
whose exponential-decay kernels are learned from data rather than fixed at
the 1/5/22-day horizons HAR uses. The original paper demonstrates a 37.6%
MSE reduction vs plain HAR on Chinese equity indices.

Definitions (this implementation, 15-min bar adaptation):

    R1(t; alpha)  = sum_{k=1..K} w_k(alpha) * signed_flow_{t-k}
    R2(t; beta)   = sum_{k=1..K} w_k(beta)  * realized_var_{t-k}

    w_k(p) = p * (1 - p)^(k-1)  (exponential kernel, normalized to sum = 1)

where:
    signed_flow_t  = sqrt(rs_up_t) - sqrt(rs_down_t)    (signed realized flow proxy)
    realized_var_t = rv_ann_t                            (annualized realized variance)

Rationale for substituting signed_flow for raw log-returns: we don't have
close-to-close 15m prices natively on the TSRV grid, but we have upside/downside
realized semivariances that sum to realized variance. sqrt(rs_up) - sqrt(rs_down)
is a bar-level signed flow proxy proportional-in-expectation to the signed
return magnitude (positive = net buying pressure, negative = net selling).

Leakage guarantee: R1(t), R2(t) use ONLY values at indices strictly less than t
via pandas rolling + shift(1). Verified by causality unit test.

Why multiple alpha values rather than a single learned one: Xiao's method is
a LASSO over a bank of R1/R2 features at several kernel decays, letting the
linear model select. We emit a fixed bank of kernel decays and let LightGBM's
tree-based feature selection play the role of the LASSO.

Output schema (sidecar parquet `BTCUSDT_features_har_pd_15m.parquet`):

    R1_{a005,a01,a03} : 3 columns, alpha in {0.05, 0.1, 0.3}
    R2_{b005,b01,b03} : 3 columns, beta in {0.05, 0.1, 0.3}

Six raw features total. No windowing variants (the R1/R2 families ARE
different windowings by design).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Kernel decay values — fast (0.05, half-life ~14 bars = 3.5h), medium (0.1,
# half-life ~7 bars = 1.75h), slow (0.3, half-life ~2 bars = 30min).
# These span the range where Corsi-HAR uses 1/5/22-day fixed windows.
ALPHA_GRID: tuple[float, ...] = (0.05, 0.1, 0.3)
BETA_GRID: tuple[float, ...] = (0.05, 0.1, 0.3)

# Truncate the exponential kernel at 96 bars (24 hours at 15-min cadence).
# At alpha=0.05 (slowest), the weight at bar 96 is 0.05 * 0.95^95 ~ 0.0004 —
# negligible relative to the sum. A finite K keeps the computation vectorizable.
K_BARS: int = 96

HAR_PD_RAW_FEATURES: tuple[str, ...] = (
    "har_pd_R1_a005", "har_pd_R1_a01", "har_pd_R1_a03",
    "har_pd_R2_b005", "har_pd_R2_b01", "har_pd_R2_b03",
)


def _exponential_kernel_weights(alpha: float, k_bars: int = K_BARS) -> np.ndarray:
    """Return the exponential decay weights w_k = alpha*(1-alpha)^(k-1) for k=1..k_bars.

    Normalized so sum(w_k) == 1 (removes dependence on the sum-truncation).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    k = np.arange(1, k_bars + 1)
    w = alpha * (1 - alpha) ** (k - 1)
    return w / w.sum()


def _weighted_rolling_sum(series: pd.Series, weights: np.ndarray) -> pd.Series:
    """Compute sum_{k=1..K} w_k * series_{t-k} using numpy convolution.

    STRICTLY past: at time t, weight w_1 multiplies series at t-1, w_2 at t-2,
    ..., w_K at t-K. The current value series_t is NOT used — this is the
    core leakage guard.
    """
    values = series.values
    n = len(values)
    K = len(weights)
    # np.convolve(a, v) internally flips v, so out[n] = sum_m a[m] * v[n-m].
    # At n = K-1 + t-K = t-1 (for t >= K): out[t-1] = sum_{m} a[m] * v[t-1-m]
    # Reparameterize with k = t - m (m = t - k):
    #   out[t-1] = sum_{k=1..K} a[t-k] * v[k-1]
    # With a=values, v=weights: result[t] = sum_{k=1..K} weights[k-1] * values[t-k].
    # This is exactly what we want. DO NOT reverse weights — np.convolve handles it.
    conv = np.convolve(values, weights, mode="full")
    # conv has length n + K - 1; result at position t (0-indexed) for t >= K is
    # conv[t-1]: this sums values[t-1], values[t-2], ..., values[t-K] with the
    # reversed kernel. At t=0..K-1 the sum is undefined (not enough history).
    out = np.full(n, np.nan, dtype=np.float64)
    # result[t] = conv[t-1]; so for t in [1..n], result[t] = conv[t-1]
    # But we want enough history (K bars), so t must be >= K.
    # conv[t-1] for t >= K → conv[K-1:K-1+n-K+1] = conv[K-1:n]
    out[K:] = conv[K - 1 : n - 1]
    return pd.Series(out, index=series.index)


def compute_har_pd_features(tsrv: pd.DataFrame) -> pd.DataFrame:
    """Build the HAR-PD feature block from a TSRV parquet.

    Parameters
    ----------
    tsrv : DataFrame indexed by 15-min timestamp with columns
        rs_up, rs_down, rv_ann. Only these columns are used.

    Returns
    -------
    DataFrame indexed identically to tsrv, with six columns listed in
    HAR_PD_RAW_FEATURES. First K_BARS rows are NaN by construction.
    """
    required = {"rs_up", "rs_down", "rv_ann"}
    missing = required - set(tsrv.columns)
    if missing:
        raise ValueError(f"TSRV parquet missing required cols: {missing}")

    # Signed realized flow proxy
    signed_flow = np.sqrt(np.clip(tsrv["rs_up"].values, 0, None)) - np.sqrt(
        np.clip(tsrv["rs_down"].values, 0, None)
    )
    signed_flow = pd.Series(signed_flow, index=tsrv.index)

    realized_var = tsrv["rv_ann"].astype(float)

    out: dict[str, pd.Series] = {}
    for alpha, suffix in zip(ALPHA_GRID, ("a005", "a01", "a03")):
        w = _exponential_kernel_weights(alpha)
        out[f"har_pd_R1_{suffix}"] = _weighted_rolling_sum(signed_flow, w)
    for beta, suffix in zip(BETA_GRID, ("b005", "b01", "b03")):
        w = _exponential_kernel_weights(beta)
        out[f"har_pd_R2_{suffix}"] = _weighted_rolling_sum(realized_var, w)

    return pd.DataFrame(out, index=tsrv.index)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTCUSDT", choices=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--tsrv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.tsrv is None:
        args.tsrv = Path(f"TSRV/output/{args.asset}_tsrv_15m.parquet")
    if args.out is None:
        args.out = Path(f"vol_model/output/{args.asset}_features_har_pd_15m.parquet")

    print(f"[har_pd] loading TSRV: {args.tsrv}")
    tsrv = pd.read_parquet(args.tsrv)
    print(f"[har_pd]   {len(tsrv)} rows, {len(tsrv.columns)} cols")

    print(f"[har_pd] computing 6 features at K={K_BARS} bars, alpha/beta grids {ALPHA_GRID} / {BETA_GRID}...")
    feats = compute_har_pd_features(tsrv)

    # Stats
    for c in feats.columns:
        s = feats[c].dropna()
        print(f"  {c}: non-null={len(s)}/{len(feats)}  mean={s.mean():.3e}  std={s.std():.3e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out)
    print(f"[har_pd] wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
