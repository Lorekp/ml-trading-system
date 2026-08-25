"""Tier 3: Portfolio Construction Experiments.

EXP-G: Denoised correlation (Marcenko-Pastur)
EXP-H: HRP allocation (AFML Ch16)
EXP-I: Per-asset Kelly sizing
EXP-J: NCO allocation (ML for AM Ch7)

All share the same signal generation — only allocation/sizing differs.
"""
import sys
sys.path.insert(0, "src")

import copy, warnings, os, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats

W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")

PROD_ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "ZEC": "ZECUSDT", "BNX": "BNXUSDT", "BNB": "BNBUSDT", "ENJ": "ENJUSDT",
    "TAO": "TAOUSDT", "ADA": "ADAUSDT", "ANKR": "ANKRUSDT", "SUI": "SUIUSDT",
    "AVAX": "AVAXUSDT", "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "FIL": "FILUSDT",
    "DOT": "DOTUSDT", "SXP": "SXPUSDT", "ENA": "ENAUSDT", "BCH": "BCHUSDT",
    "LTC": "LTCUSDT", "UNI": "UNIUSDT", "WLD": "WLDUSDT", "TRX": "TRXUSDT",
    "FTM": "FTMUSDT", "AAVE": "AAVEUSDT", "FET": "FETUSDT", "APT": "APTUSDT",
    "HBAR": "HBARUSDT", "XLM": "XLMUSDT",
}


def _gen_asset_signals(args):
    """Generate signals + daily returns for one asset. Subprocess."""
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")
    from signal_engine.config import load_config
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.model import WalkForwardModel
    from signal_engine.filters import apply_confidence_filter

    cfg = load_config(Path("configs/default.yaml"))
    cfg.regime_filter.enabled = False
    cfg.confidence_filter.mode = "absolute"
    cfg.confidence_filter.threshold = 0.08

    try:
        ohlcv_1m = load_binance_1m(symbol, [Path("data/binance-1m")])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        features = compute_features(ohlcv_15m, cfg.features)
        wf = WalkForwardModel(cfg)
        raw = wf.generate_signals_batch(ohlcv_15m, features, asset)
        if len(raw) == 0:
            return asset, None, None
        filtered = apply_confidence_filter(raw, cfg.confidence_filter)
        # Daily returns for correlation matrix
        daily_ret = ohlcv_15m["close"].resample("D").last().pct_change().dropna()
        return asset, filtered.to_dict("list"), daily_ret.to_dict()
    except Exception:
        return asset, None, None


def get_prod_portfolio():
    cfg = load_config(Path("configs/default.yaml"))
    p = copy.deepcopy(cfg.portfolio)
    p.base_sizing_pct = 0.04
    p.slippage = 0.0005
    p.max_exposure = 0.90
    p.max_per_asset = 0.03
    return p


def apply_weights_to_signals(signals_dict, weights):
    """Scale signal_strength by per-asset weight."""
    df = pd.DataFrame(signals_dict)
    asset = df["asset"].iloc[0] if "asset" in df.columns and len(df) > 0 else None
    if asset and asset in weights:
        w = weights[asset]
        df["signal_strength"] = df["signal_strength"] * w
    return df.to_dict("list")


def combine_and_sim(signal_dicts, portfolio_cfg):
    """Combine signals and run portfolio sim."""
    frames = [pd.DataFrame(d) for d in signal_dicts]
    if not frames:
        return None
    combined = pd.concat(frames).sort_values("timestamp")
    sim = simulate_portfolio(combined, portfolio_cfg)
    if sim is None or sim["n_trades"] < 10:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    stats = compute_full_stats(dpnl, portfolio_cfg.starting_capital,
                                sim["n_trades"], sim["n_trades"] / len(dpnl))
    stats["_pnl_day"] = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
    return stats


def print_result(label, stats):
    if stats is None:
        print(f"  {label:<50s} FAILED", flush=True)
        return
    print(
        f"  {label:<50s} Sharpe={stats['sharpe']:>6.3f}  Ret={stats['total_return_pct']:>7.2f}%  "
        f"DD={stats['max_dd_pct']:>5.2f}%  Trades={stats['n_trades']:>6d}  "
        f"PnL/day=${stats['_pnl_day']:>6.2f}",
        flush=True,
    )


# =========================================================================
# HRP Implementation (AFML Ch16)
# =========================================================================

def compute_hrp_weights(corr_matrix, asset_names):
    """Hierarchical Risk Parity allocation weights."""
    n = len(asset_names)
    if n <= 1:
        return {asset_names[0]: 1.0} if n == 1 else {}

    # Distance matrix
    dist = np.sqrt(0.5 * (1 - corr_matrix))
    np.fill_diagonal(dist, 0)

    # Hierarchical clustering
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")

    # Quasi-diagonalize (reorder by leaves)
    order = leaves_list(link).astype(int)

    # Recursive bisection
    cov = np.diag(np.diag(corr_matrix))  # use diagonal (variance) only for simplicity
    weights = np.ones(n)

    def _bisect(items):
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]

        # Cluster variance
        var_left = sum(cov[i, i] for i in left)
        var_right = sum(cov[i, i] for i in right)
        total = var_left + var_right
        if total <= 0:
            return

        # Inverse variance allocation
        alpha = 1 - var_left / total  # more to lower-variance cluster
        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= (1 - alpha)

        _bisect(left)
        _bisect(right)

    _bisect(list(order))

    # Normalize to mean=1 (so total budget unchanged)
    weights = weights / weights.mean()

    return {asset_names[i]: weights[i] for i in range(n)}


# =========================================================================
# Marcenko-Pastur Denoising (ML for AM Ch2)
# =========================================================================

def denoise_corr(corr, q):
    """Denoise correlation matrix using Marcenko-Pastur."""
    evals, evecs = np.linalg.eigh(corr)
    # MP upper bound
    lambda_plus = (1 + np.sqrt(1 / q)) ** 2
    # Replace noise eigenvalues with mean of noise eigenvalues
    noise_mask = evals < lambda_plus
    if noise_mask.any():
        noise_mean = evals[noise_mask].mean()
        evals[noise_mask] = noise_mean
    # Reconstruct
    denoised = evecs @ np.diag(evals) @ evecs.T
    # Normalize to correlation (diagonal = 1)
    d = np.sqrt(np.diag(denoised))
    d[d == 0] = 1
    denoised = denoised / np.outer(d, d)
    np.fill_diagonal(denoised, 1.0)
    return denoised


# =========================================================================
# NCO (ML for AM Ch7)
# =========================================================================

def compute_nco_weights(corr_matrix, expected_returns, asset_names, n_clusters=5):
    """Nested Clustered Optimization."""
    from scipy.cluster.hierarchy import linkage, fcluster

    n = len(asset_names)
    if n <= 1:
        return {asset_names[0]: 1.0} if n == 1 else {}

    # Cluster
    dist = np.sqrt(0.5 * (1 - corr_matrix))
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="ward")
    clusters = fcluster(link, t=min(n_clusters, n), criterion="maxclust")

    weights = np.zeros(n)

    # Within-cluster: inverse-variance weights
    cluster_returns = {}
    cluster_vols = {}
    for c in np.unique(clusters):
        members = np.where(clusters == c)[0]
        if len(members) == 1:
            weights[members[0]] = 1.0
            cluster_returns[c] = expected_returns[members[0]]
            cluster_vols[c] = np.sqrt(corr_matrix[members[0], members[0]])
        else:
            # Inverse-variance within cluster
            variances = np.array([corr_matrix[i, i] for i in members])
            variances = np.maximum(variances, 1e-10)
            inv_var = 1.0 / variances
            w_inner = inv_var / inv_var.sum()
            for idx, m in enumerate(members):
                weights[m] = w_inner[idx]
            cluster_returns[c] = sum(expected_returns[m] * w_inner[idx] for idx, m in enumerate(members))
            cluster_vols[c] = np.sqrt(sum(variances * w_inner ** 2))

    # Across clusters: weight by Sharpe-like ratio
    cluster_ids = sorted(cluster_returns.keys())
    cluster_sharpes = []
    for c in cluster_ids:
        vol = max(cluster_vols[c], 1e-10)
        cluster_sharpes.append(max(cluster_returns[c] / vol, 0))

    total_sharpe = sum(cluster_sharpes)
    if total_sharpe <= 0:
        # Equal weight across clusters
        for c in cluster_ids:
            members = np.where(clusters == c)[0]
            for m in members:
                weights[m] *= 1.0 / len(cluster_ids)
    else:
        for i, c in enumerate(cluster_ids):
            cluster_weight = cluster_sharpes[i] / total_sharpe
            members = np.where(clusters == c)[0]
            for m in members:
                weights[m] *= cluster_weight

    # Normalize to mean=1
    if weights.sum() > 0:
        weights = weights / weights.mean()

    return {asset_names[i]: weights[i] for i in range(n)}


def main():
    portfolio_cfg = get_prod_portfolio()

    print("=" * 80)
    print(f"TIER 3: PORTFOLIO CONSTRUCTION EXPERIMENTS ({W} workers)")
    print("Baseline: Sharpe 2.819, 30 assets, equal-weight allocation")
    print("=" * 80, flush=True)

    # Phase 1: Generate ALL signals in parallel
    print("\nPhase 1: Generating signals...", flush=True)
    t0 = time.time()

    asset_signals = {}
    asset_daily_rets = {}

    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_gen_asset_signals, (a, s)): a for a, s in PROD_ASSETS.items()}
        done = 0
        for future in as_completed(futures):
            asset, sig, ret_dict = future.result()
            done += 1
            if sig is not None:
                asset_signals[asset] = sig
            if ret_dict is not None:
                asset_daily_rets[asset] = pd.Series(ret_dict)

    print(f"Signals generated in {time.time()-t0:.0f}s ({len(asset_signals)} assets)", flush=True)

    # Phase 2: Build correlation matrix from daily returns
    common_assets = sorted(set(asset_signals.keys()) & set(asset_daily_rets.keys()))
    print(f"\nAssets with both signals and returns: {len(common_assets)}")

    # Align daily returns
    ret_df = pd.DataFrame({a: asset_daily_rets[a] for a in common_assets})
    ret_df = ret_df.dropna(how="all").fillna(0)
    corr = ret_df.corr().values
    n = len(common_assets)
    q = len(ret_df) / n  # T/N ratio for Marcenko-Pastur

    print(f"Correlation matrix: {n}x{n}, T/N ratio: {q:.1f}")
    print(f"Avg pairwise corr: {corr[np.triu_indices(n, k=1)].mean():.3f}")

    # =====================================================================
    # BASELINE
    # =====================================================================
    print("\n--- BASELINE (equal-weight) ---", flush=True)
    baseline_signals = [asset_signals[a] for a in common_assets]
    baseline = combine_and_sim(baseline_signals, portfolio_cfg)
    print_result("Equal-weight baseline", baseline)

    # =====================================================================
    # EXP-G: Denoised Correlation + Inverse-Variance
    # =====================================================================
    print("\n" + "=" * 80)
    print("EXP-G: DENOISED CORRELATION (Marcenko-Pastur)")
    print("=" * 80, flush=True)

    denoised = denoise_corr(corr, q)
    # Inverse-variance weights from denoised matrix
    variances = np.diag(denoised)
    variances = np.maximum(variances, 1e-10)
    inv_var = 1.0 / variances
    inv_var_weights = inv_var / inv_var.mean()  # normalize to mean=1
    g_weights = {common_assets[i]: inv_var_weights[i] for i in range(n)}

    weighted_g = [apply_weights_to_signals(asset_signals[a], g_weights) for a in common_assets]
    stats_g = combine_and_sim(weighted_g, portfolio_cfg)
    print_result("Denoised inv-variance", stats_g)

    print("  Top 5 weights:", sorted(g_weights.items(), key=lambda x: x[1], reverse=True)[:5])
    print("  Bot 5 weights:", sorted(g_weights.items(), key=lambda x: x[1])[:5])

    # =====================================================================
    # EXP-H: HRP Allocation
    # =====================================================================
    print("\n" + "=" * 80)
    print("EXP-H: HRP ALLOCATION (AFML Ch16)")
    print("=" * 80, flush=True)

    hrp_weights = compute_hrp_weights(corr, common_assets)

    weighted_h = [apply_weights_to_signals(asset_signals[a], hrp_weights) for a in common_assets]
    stats_h = combine_and_sim(weighted_h, portfolio_cfg)
    print_result("HRP allocation", stats_h)

    print("  Top 5 weights:", sorted(hrp_weights.items(), key=lambda x: x[1], reverse=True)[:5])
    print("  Bot 5 weights:", sorted(hrp_weights.items(), key=lambda x: x[1])[:5])

    # =====================================================================
    # EXP-I: Per-Asset Kelly
    # =====================================================================
    print("\n" + "=" * 80)
    print("EXP-I: PER-ASSET KELLY SIZING")
    print("=" * 80, flush=True)

    from afml.bet_sizing import kelly_size

    kelly_weights = {}
    for asset in common_assets:
        sig_df = pd.DataFrame(asset_signals[asset])
        if "fwd_return" not in sig_df.columns or "direction" not in sig_df.columns:
            kelly_weights[asset] = 1.0
            continue
        # Compute win rate and payoff from signals
        trade_rets = sig_df["direction"] * sig_df["fwd_return"]
        wins = trade_rets[trade_rets > 0]
        losses = trade_rets[trade_rets <= 0]
        if len(wins) < 5 or len(losses) < 5:
            kelly_weights[asset] = 1.0
            continue
        win_prob = len(wins) / len(trade_rets)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        if avg_loss <= 0:
            kelly_weights[asset] = 1.0
            continue
        f_star = kelly_size(win_prob, avg_win, avg_loss)
        # Quarter-Kelly, capped at 3.0, floored at 0.2
        kelly_weights[asset] = max(0.2, min(3.0, f_star / 4 * 30))  # scale to mean ~1

    # Normalize to mean=1
    mean_w = np.mean(list(kelly_weights.values()))
    if mean_w > 0:
        kelly_weights = {k: v / mean_w for k, v in kelly_weights.items()}

    weighted_i = [apply_weights_to_signals(asset_signals[a], kelly_weights) for a in common_assets]
    stats_i = combine_and_sim(weighted_i, portfolio_cfg)
    print_result("Per-asset Kelly (quarter)", stats_i)

    print("  Top 5 weights:", sorted(kelly_weights.items(), key=lambda x: x[1], reverse=True)[:5])
    print("  Bot 5 weights:", sorted(kelly_weights.items(), key=lambda x: x[1])[:5])

    # =====================================================================
    # EXP-J: NCO Allocation
    # =====================================================================
    print("\n" + "=" * 80)
    print("EXP-J: NCO ALLOCATION (ML for AM Ch7)")
    print("=" * 80, flush=True)

    # Expected returns per asset (from signal win rates)
    expected_rets = np.zeros(n)
    for i, asset in enumerate(common_assets):
        sig_df = pd.DataFrame(asset_signals[asset])
        if "fwd_return" in sig_df.columns and "direction" in sig_df.columns:
            trade_rets = sig_df["direction"] * sig_df["fwd_return"]
            expected_rets[i] = trade_rets.mean()

    for n_clusters in [3, 5, 8]:
        nco_weights = compute_nco_weights(denoised, expected_rets, common_assets, n_clusters=n_clusters)

        weighted_j = [apply_weights_to_signals(asset_signals[a], nco_weights) for a in common_assets]
        stats_j = combine_and_sim(weighted_j, portfolio_cfg)
        print_result(f"NCO (K={n_clusters})", stats_j)

    # =====================================================================
    print("\n" + "=" * 80)
    print("TIER 3 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
