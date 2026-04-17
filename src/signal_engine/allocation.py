"""Portfolio allocation methods.

Supports Kelly, HRP (AFML Ch16), NCO (ML for AM Ch7), and CVaR-NCO.
All methods return per-asset weight dicts normalized to mean=1.

CVaR (Conditional Value at Risk / Expected Shortfall):
    Unlike variance which penalizes upside and downside equally,
    CVaR focuses on the average loss in the worst alpha% of days.
    This directly targets the tail risk that causes drawdowns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


def compute_kelly_weights(
    signals: pd.DataFrame,
    min_trades: int = 10,
    scale: float = 7.5,
    floor: float = 0.2,
    cap: float = 3.0,
) -> dict[str, float]:
    """Per-asset Kelly weights from trade returns.

    Args:
        signals: Must have columns: asset, direction, fwd_return.
        min_trades: Minimum trades per asset to compute Kelly.
        scale: Multiplier for raw Kelly fraction.
        floor: Minimum weight.
        cap: Maximum weight.

    Returns:
        Dict of asset -> weight, normalized to mean=1.
    """
    from afml.bet_sizing import kelly_size

    weights: dict[str, float] = {}
    for asset in signals["asset"].unique():
        asset_sigs = signals[signals["asset"] == asset]
        if len(asset_sigs) < min_trades:
            weights[asset] = 1.0
            continue
        trade_rets = asset_sigs["direction"].values * asset_sigs["fwd_return"].values
        wins = trade_rets[trade_rets > 0]
        losses = trade_rets[trade_rets <= 0]
        if len(wins) < 3 or len(losses) < 3:
            weights[asset] = 1.0
            continue
        w_prob = len(wins) / len(trade_rets)
        w_ret = float(wins.mean())
        l_ret = float(abs(losses.mean()))
        if l_ret <= 0:
            weights[asset] = 1.0
            continue
        f_star = kelly_size(w_prob, w_ret, l_ret)
        weights[asset] = max(floor, min(cap, f_star * scale))

    return _normalize(weights)


def compute_nco_weights(
    corr: np.ndarray,
    expected_returns: np.ndarray,
    asset_names: list[str],
    n_clusters: int = 8,
) -> dict[str, float]:
    """Nested Clustered Optimization (ML for AM Ch7).

    Two-stage allocation:
    1. Within-cluster: inverse-variance weights
    2. Across-cluster: Sharpe-ratio-weighted

    Args:
        corr: Correlation matrix (n x n).
        expected_returns: Per-asset expected returns (n,).
        asset_names: Asset names matching matrix indices.
        n_clusters: Number of clusters.

    Returns:
        Dict of asset -> weight, normalized to mean=1.
    """
    n = len(asset_names)
    if n <= 1:
        return {asset_names[0]: 1.0} if n == 1 else {}

    # Denoise correlation matrix (Marcenko-Pastur)
    corr = _denoise_corr(corr, n)

    # Cluster assets
    dist = np.sqrt(np.maximum(0.5 * (1 - corr), 0))
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="ward")
    clusters = fcluster(link, t=min(n_clusters, n), criterion="maxclust")

    weights = np.zeros(n)
    cluster_ret: dict[int, float] = {}
    cluster_vol: dict[int, float] = {}

    for c in np.unique(clusters):
        members = np.where(clusters == c)[0]
        if len(members) == 1:
            weights[members[0]] = 1.0
            cluster_ret[c] = expected_returns[members[0]]
            cluster_vol[c] = np.sqrt(max(corr[members[0], members[0]], 1e-10))
        else:
            variances = np.maximum(np.array([corr[i, i] for i in members]), 1e-10)
            inv_var = 1.0 / variances
            w_inner = inv_var / inv_var.sum()
            for idx, m in enumerate(members):
                weights[m] = w_inner[idx]
            cluster_ret[c] = sum(
                expected_returns[m] * w_inner[idx] for idx, m in enumerate(members)
            )
            cluster_vol[c] = np.sqrt(sum(variances * w_inner**2))

    # Across-cluster: weight by Sharpe ratio
    cids = sorted(cluster_ret.keys())
    sharpes = [max(cluster_ret[c] / max(cluster_vol[c], 1e-10), 0) for c in cids]
    total = sum(sharpes)

    if total <= 0:
        for c in cids:
            for m in np.where(clusters == c)[0]:
                weights[m] *= 1.0 / len(cids)
    else:
        for i, c in enumerate(cids):
            cw = sharpes[i] / total
            for m in np.where(clusters == c)[0]:
                weights[m] *= cw

    result = {asset_names[i]: float(weights[i]) for i in range(n)}
    return _normalize(result)


def compute_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Conditional Value at Risk (Expected Shortfall) at alpha level.

    CVaR = average loss on the worst alpha% of days.
    Positive return = bigger tail risk.

    Args:
        returns: Array of daily returns (can be negative).
        alpha: Tail probability (0.05 = worst 5%).

    Returns:
        CVaR as a positive number (higher = worse tail risk).
    """
    if len(returns) < 5:
        return 0.0
    var_threshold = np.percentile(returns, alpha * 100)
    tail = returns[returns <= var_threshold]
    if len(tail) == 0:
        return -var_threshold  # no tail data, use VaR
    return -tail.mean()


def _min_cvar_weights(
    daily_returns: np.ndarray,
    n_assets: int,
    alpha: float = 0.05,
) -> np.ndarray:
    """Find portfolio weights that minimize CVaR.

    Uses scipy.optimize.minimize with SLSQP to find the weight
    vector that minimizes the expected shortfall of the portfolio.

    Args:
        daily_returns: (T, n_assets) array of daily returns.
        n_assets: Number of assets.
        alpha: CVaR tail probability.

    Returns:
        Optimal weight vector (n_assets,), sums to 1.
    """
    if n_assets == 1:
        return np.array([1.0])

    def _portfolio_cvar(w):
        port_returns = daily_returns @ w
        return compute_cvar(port_returns, alpha)

    # Constraints: weights sum to 1, all non-negative
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.01, 1.0)] * n_assets  # floor at 1% to avoid zeros

    # Start with equal weights
    w0 = np.ones(n_assets) / n_assets

    try:
        result = minimize(
            _portfolio_cvar,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-8},
        )
        if result.success:
            return result.x / result.x.sum()  # renormalize
    except Exception:
        pass

    return w0  # fallback to equal weights


def compute_nco_cvar_weights(
    daily_returns_df: pd.DataFrame,
    expected_returns: np.ndarray,
    asset_names: list[str],
    n_clusters: int = 8,
    alpha: float = 0.05,
) -> dict[str, float]:
    """NCO with CVaR minimization instead of inverse-variance.

    Same two-stage structure as standard NCO:
    1. Within-cluster: minimize CVaR (instead of inverse-variance)
    2. Across-cluster: weight by return/CVaR ratio (instead of Sharpe)

    Args:
        daily_returns_df: DataFrame with columns = asset names, rows = daily returns.
        expected_returns: Per-asset expected returns (n,).
        asset_names: Asset names matching matrix columns.
        n_clusters: Number of clusters.
        alpha: CVaR tail probability.

    Returns:
        Dict of asset -> weight, normalized to mean=1.
    """
    n = len(asset_names)
    if n <= 1:
        return {asset_names[0]: 1.0} if n == 1 else {}

    # Compute and denoise correlation for clustering
    corr = daily_returns_df[asset_names].corr().values
    corr = _denoise_corr(corr, n)

    # Cluster assets (same as standard NCO)
    dist = np.sqrt(np.maximum(0.5 * (1 - corr), 0))
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="ward")
    clusters = fcluster(link, t=min(n_clusters, n), criterion="maxclust")

    weights = np.zeros(n)
    cluster_ret: dict[int, float] = {}
    cluster_cvar: dict[int, float] = {}

    daily_arr = daily_returns_df[asset_names].values  # (T, n)

    for c in np.unique(clusters):
        members = np.where(clusters == c)[0]
        if len(members) == 1:
            weights[members[0]] = 1.0
            cluster_ret[c] = expected_returns[members[0]]
            asset_rets = daily_arr[:, members[0]]
            cluster_cvar[c] = max(compute_cvar(asset_rets, alpha), 1e-10)
        else:
            # Minimize CVaR within cluster
            cluster_daily = daily_arr[:, members]
            w_inner = _min_cvar_weights(cluster_daily, len(members), alpha)
            for idx, m in enumerate(members):
                weights[m] = w_inner[idx]

            # Cluster-level stats
            cluster_port = cluster_daily @ w_inner
            cluster_ret[c] = float(np.mean(cluster_port))
            cluster_cvar[c] = max(compute_cvar(cluster_port, alpha), 1e-10)

    # Across-cluster: weight by return / CVaR (like Sharpe but with CVaR)
    cids = sorted(cluster_ret.keys())
    ratios = [max(cluster_ret[c] / cluster_cvar[c], 0) for c in cids]
    total = sum(ratios)

    if total <= 0:
        for c in cids:
            for m in np.where(clusters == c)[0]:
                weights[m] *= 1.0 / len(cids)
    else:
        for i, c in enumerate(cids):
            cw = ratios[i] / total
            for m in np.where(clusters == c)[0]:
                weights[m] *= cw

    result = {asset_names[i]: float(weights[i]) for i in range(n)}
    return _normalize(result)


def _denoise_corr(corr: np.ndarray, n: int) -> np.ndarray:
    """Denoise correlation via Marcenko-Pastur (ML for AM Ch2)."""
    evals, evecs = np.linalg.eigh(corr)
    # Assume T/N ~ 50 (typical for our data)
    q = 50.0
    lp = (1 + np.sqrt(1 / q)) ** 2
    noise = evals < lp
    if noise.any():
        evals[noise] = evals[noise].mean()
    d = evecs @ np.diag(evals) @ evecs.T
    diag = np.sqrt(np.maximum(np.diag(d), 1e-10))
    d = d / np.outer(diag, diag)
    np.fill_diagonal(d, 1.0)
    return d


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    """Normalize weights to mean=1."""
    if not weights:
        return weights
    vals = list(weights.values())
    mean_w = np.mean(vals)
    if mean_w > 0:
        return {k: v / mean_w for k, v in weights.items()}
    return weights
