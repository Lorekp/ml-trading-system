"""Sector Momentum Strategy — Crypto Factor Baskets

Hypothesis: Crypto assets cluster into effective sectors (L1, DeFi, AI, etc).
Narrative-driven flows are sector-specific, creating persistent sector momentum.
Long strongest sectors, short weakest.

Key insight from 2A: crypto residuals are MOMENTUM (not MR). This strategy
explicitly exploits that by going long recent winners at the sector level.

Two approaches tested:
A) Manual sector definitions
B) Correlation-based clustering (dynamic)

Kill criteria:
- SR < 0.5
- Correlation with LogReg 15m > 0.5
"""

import sys, os, time, warnings
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.stats import compute_full_stats

warnings.filterwarnings("ignore")

DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]

# Manual sector definitions (fixed before test period)
SECTORS = {
    "L1": ["SOLUSDT", "AVAXUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT", "DOTUSDT", "ADAUSDT"],
    "DeFi": ["UNIUSDT", "AAVEUSDT", "LINKUSDT"],
    "BTC_adj": ["BTCUSDT", "LTCUSDT", "BCHUSDT"],
    "ETH_eco": ["ETHUSDT"],
    "AI_data": ["FETUSDT", "TAOUSDT"],
    "Misc": ["XRPUSDT", "BNBUSDT", "TRXUSDT", "XLMUSDT", "HBARUSDT",
             "FILUSDT", "ENJUSDT", "SXPUSDT", "ZECUSDT", "FTMUSDT",
             "WLDUSDT", "BNXUSDT", "ANKRUSDT", "ENAUSDT"],
}

# Broader universe for clustering approach
ALL_SYMBOLS = {}
for sector, symbols in SECTORS.items():
    for s in symbols:
        name = s.replace("USDT", "")
        ALL_SYMBOLS[name] = s


def load_daily_returns(symbols: dict) -> pd.DataFrame:
    """Load daily close returns for all symbols."""
    daily_rets = {}
    for name, symbol in sorted(symbols.items()):
        try:
            ohlcv_1m = load_binance_1m(symbol, DATA_DIRS)
            daily = resample_ohlcv(ohlcv_1m, "1D")
            daily_rets[name] = daily["close"].pct_change()
        except Exception:
            continue
    return pd.DataFrame(daily_rets)


def compute_sector_returns(daily_rets: pd.DataFrame, sectors: dict) -> pd.DataFrame:
    """Compute equal-weighted sector returns."""
    sector_rets = {}
    for sector, symbols in sectors.items():
        names = [s.replace("USDT", "") for s in symbols]
        available = [n for n in names if n in daily_rets.columns]
        if len(available) >= 1:
            sector_rets[sector] = daily_rets[available].mean(axis=1)
    return pd.DataFrame(sector_rets)


def backtest_sector_momentum(
    daily_rets: pd.DataFrame,
    sector_rets: pd.DataFrame,
    momentum_window: int = 5,
    rebalance_freq: int = 7,
    n_long: int = 2,
    n_short: int = 2,
    cost_bps: float = 10.0,
    start_date: str = "2021-01-01",
) -> dict:
    """Backtest sector momentum L/S strategy.

    Every rebalance_freq days:
    - Rank sectors by trailing momentum_window-day return
    - Long top n_long sectors, short bottom n_short
    - Equal weight within long/short legs
    """
    # Filter to start date
    start_ts = pd.Timestamp(start_date)
    if sector_rets.index.tz is not None:
        start_ts = start_ts.tz_localize(sector_rets.index.tz)
    mask = sector_rets.index >= start_ts
    sector_rets = sector_rets[mask].dropna(how="all")

    if len(sector_rets) < 60:
        return {"error": "insufficient_data"}

    sectors = sector_rets.columns.tolist()
    n_sectors = len(sectors)

    if n_sectors < n_long + n_short:
        return {"error": f"need {n_long+n_short} sectors, have {n_sectors}"}

    # Trailing momentum signal
    momentum = sector_rets.rolling(momentum_window).sum()

    # Daily PnL
    daily_pnl = []
    dates = []
    last_rebal = -rebalance_freq  # force rebalance on first day
    long_sectors = []
    short_sectors = []
    prev_long = []
    prev_short = []

    for i in range(momentum_window, len(sector_rets)):
        date = sector_rets.index[i]

        # Rebalance check
        if i - last_rebal >= rebalance_freq:
            mom_today = momentum.iloc[i].dropna()
            if len(mom_today) < n_long + n_short:
                continue

            ranked = mom_today.sort_values(ascending=False)
            prev_long = long_sectors.copy()
            prev_short = short_sectors.copy()
            long_sectors = ranked.index[:n_long].tolist()
            short_sectors = ranked.index[-n_short:].tolist()
            last_rebal = i

            # Turnover cost
            new_longs = set(long_sectors) - set(prev_long)
            new_shorts = set(short_sectors) - set(prev_short)
            turnover = (len(new_longs) + len(new_shorts)) / max(n_long + n_short, 1)
            cost = turnover * cost_bps / 10000.0
        else:
            cost = 0.0

        # Daily return: long leg - short leg
        if long_sectors and short_sectors:
            long_ret = sector_rets.loc[date, long_sectors].mean()
            short_ret = sector_rets.loc[date, short_sectors].mean()
            ls_ret = (long_ret - short_ret) / 2.0  # scale to ~half vol
            daily_pnl.append(float(ls_ret) - cost)
            dates.append(date)

    if len(daily_pnl) < 30:
        return {"error": "too_few_days"}

    pnl_series = pd.Series(daily_pnl, index=dates)
    pnl_nonzero = pnl_series[pnl_series != 0]

    if len(pnl_nonzero) < 20:
        return {"error": "too_few_active_days"}

    # Stats
    mean_daily = pnl_nonzero.mean()
    std_daily = pnl_nonzero.std()
    sr = mean_daily / std_daily * np.sqrt(365.25) if std_daily > 0 else 0
    cumulative = (1 + pnl_nonzero).cumprod()
    max_dd = float((cumulative / cumulative.cummax() - 1).min())
    total_ret = float(cumulative.iloc[-1] - 1) * 100
    win_rate = (pnl_nonzero > 0).mean() * 100

    # T-stat on mean daily return
    t_stat = mean_daily / (std_daily / np.sqrt(len(pnl_nonzero))) if std_daily > 0 else 0

    return {
        "sharpe": sr,
        "total_ret": total_ret,
        "max_dd": max_dd * 100,
        "win_rate": win_rate,
        "t_stat": t_stat,
        "n_days": len(pnl_nonzero),
        "mean_daily_bps": mean_daily * 10000,
        "pnl_series": pnl_series,
    }


def main():
    t0 = time.time()

    print("=" * 100)
    print("  SECTOR MOMENTUM STRATEGY")
    print("=" * 100)

    # Load data
    print("\nLoading daily returns...")
    daily_rets = load_daily_returns(ALL_SYMBOLS)
    print(f"  Loaded {len(daily_rets.columns)} assets, {len(daily_rets)} days")
    print(f"  Date range: {daily_rets.index.min()} to {daily_rets.index.max()}")

    # Compute sector returns
    sector_rets = compute_sector_returns(daily_rets, SECTORS)
    print(f"  Sectors: {list(sector_rets.columns)}")

    # ================================================================
    # Approach A: Manual sectors
    # ================================================================
    print(f"\n{'='*80}")
    print("APPROACH A: MANUAL SECTOR DEFINITIONS")
    print(f"{'='*80}")

    configs = [
        {"momentum_window": 5, "rebalance_freq": 7, "n_long": 2, "n_short": 2, "cost_bps": 10},
        {"momentum_window": 5, "rebalance_freq": 5, "n_long": 2, "n_short": 2, "cost_bps": 10},
        {"momentum_window": 10, "rebalance_freq": 7, "n_long": 2, "n_short": 2, "cost_bps": 10},
        {"momentum_window": 20, "rebalance_freq": 7, "n_long": 2, "n_short": 2, "cost_bps": 10},
        {"momentum_window": 5, "rebalance_freq": 7, "n_long": 1, "n_short": 1, "cost_bps": 10},
        {"momentum_window": 5, "rebalance_freq": 14, "n_long": 2, "n_short": 2, "cost_bps": 10},
        {"momentum_window": 5, "rebalance_freq": 7, "n_long": 2, "n_short": 2, "cost_bps": 5},
        {"momentum_window": 5, "rebalance_freq": 7, "n_long": 2, "n_short": 2, "cost_bps": 0},
    ]

    header = (f"{'Config':<35} {'SR':>8} {'Ret%':>8} {'DD%':>8} {'Win%':>8} "
              f"{'t-stat':>8} {'Days':>8} {'bps/d':>8}")
    print(header)
    print("-" * len(header))

    best_sr = -999
    best_pnl = None
    best_label = ""

    for cfg in configs:
        label = (f"mom={cfg['momentum_window']}d reb={cfg['rebalance_freq']}d "
                 f"L{cfg['n_long']}S{cfg['n_short']} c={cfg['cost_bps']}bp")

        result = backtest_sector_momentum(
            daily_rets, sector_rets,
            momentum_window=cfg["momentum_window"],
            rebalance_freq=cfg["rebalance_freq"],
            n_long=cfg["n_long"],
            n_short=cfg["n_short"],
            cost_bps=cfg["cost_bps"],
            start_date="2021-01-01",
        )

        if "error" in result:
            print(f"{label:<35} ERROR: {result['error']}")
            continue

        print(f"{label:<35} {result['sharpe']:>8.2f} {result['total_ret']:>7.1f}% "
              f"{result['max_dd']:>7.1f}% {result['win_rate']:>7.1f}% "
              f"{result['t_stat']:>8.2f} {result['n_days']:>8d} {result['mean_daily_bps']:>7.1f}")

        if result["sharpe"] > best_sr:
            best_sr = result["sharpe"]
            best_pnl = result["pnl_series"]
            best_label = label

    # ================================================================
    # Approach B: Correlation-based clustering
    # ================================================================
    print(f"\n{'='*80}")
    print("APPROACH B: CORRELATION-BASED CLUSTERING (DYNAMIC)")
    print(f"{'='*80}")

    # Use 60-day rolling correlation, cluster into 4-6 groups
    from scipy.cluster.hierarchy import linkage, fcluster

    # Use data before 2024 for cluster definition (avoid test period contamination)
    cutoff = pd.Timestamp("2024-01-01")
    if daily_rets.index.tz is not None:
        cutoff = cutoff.tz_localize(daily_rets.index.tz)
    train_rets = daily_rets[daily_rets.index < cutoff].dropna(axis=1, thresh=200)
    if len(train_rets.columns) >= 10:
        corr_matrix = train_rets.corr()
        # Convert correlation to distance
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        # Hierarchical clustering
        from scipy.spatial.distance import squareform
        condensed = squareform(dist_matrix.values, checks=False)
        Z = linkage(condensed, method="ward")

        for n_clusters in [4, 5, 6]:
            cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
            cluster_map = {}
            for i, col in enumerate(train_rets.columns):
                c = int(cluster_labels[i])
                if c not in cluster_map:
                    cluster_map[c] = []
                cluster_map[c].append(col + "USDT")

            # Build sector returns from clusters
            cluster_sectors = {f"C{k}": v for k, v in cluster_map.items()}
            cluster_sector_rets = compute_sector_returns(daily_rets, cluster_sectors)

            result = backtest_sector_momentum(
                daily_rets, cluster_sector_rets,
                momentum_window=5, rebalance_freq=7,
                n_long=max(1, n_clusters // 3),
                n_short=max(1, n_clusters // 3),
                cost_bps=10, start_date="2024-01-01",
            )

            n_l = max(1, n_clusters // 3)
            label = f"k={n_clusters} L{n_l}S{n_l} (2024+)"

            if "error" in result:
                print(f"  {label}: ERROR ({result['error']})")
                continue

            print(f"  {label}: SR={result['sharpe']:.2f}, Ret={result['total_ret']:.1f}%, "
                  f"DD={result['max_dd']:.1f}%, t={result['t_stat']:.2f}, "
                  f"bps/d={result['mean_daily_bps']:.1f}")

            # Print cluster composition
            for ck, members in sorted(cluster_map.items()):
                names = [m.replace("USDT", "") for m in members]
                print(f"    Cluster {ck}: {', '.join(names)}")

    # ================================================================
    # Approach C: Individual asset momentum (not sector-level)
    # ================================================================
    print(f"\n{'='*80}")
    print("APPROACH C: INDIVIDUAL ASSET CROSS-SECTIONAL MOMENTUM")
    print("  (Long top quintile performers, short bottom quintile)")
    print(f"{'='*80}")

    for mom_window in [5, 10, 20]:
        for reb_freq in [5, 7]:
            for cost in [10, 5, 0]:
                momentum = daily_rets.rolling(mom_window).sum()
                n_assets = len(daily_rets.columns)
                n_leg = max(n_assets // 5, 3)

                pnl_list = []
                dates = []
                last_reb = -reb_freq

                for i in range(mom_window, len(daily_rets)):
                    if i - last_reb >= reb_freq:
                        mom_i = momentum.iloc[i].dropna()
                        if len(mom_i) < 2 * n_leg:
                            continue
                        ranked = mom_i.sort_values(ascending=False)
                        longs = ranked.index[:n_leg].tolist()
                        shorts = ranked.index[-n_leg:].tolist()
                        last_reb = i
                        turnover_cost = cost / 10000.0
                    else:
                        turnover_cost = 0.0

                    date = daily_rets.index[i]
                    if longs and shorts:
                        l_ret = daily_rets.loc[date, longs].mean()
                        s_ret = daily_rets.loc[date, shorts].mean()
                        ls = (l_ret - s_ret) / 2.0
                        pnl_list.append(float(ls) - turnover_cost)
                        dates.append(date)

                if len(pnl_list) < 50:
                    continue

                pnl_s = pd.Series(pnl_list, index=dates)
                pnl_nz = pnl_s[pnl_s != 0]
                if len(pnl_nz) < 30:
                    continue

                m = pnl_nz.mean()
                s = pnl_nz.std()
                sr = m / s * np.sqrt(365.25) if s > 0 else 0
                t = m / (s / np.sqrt(len(pnl_nz))) if s > 0 else 0
                cum = (1 + pnl_nz).cumprod()
                dd = float((cum / cum.cummax() - 1).min()) * 100

                label = f"mom={mom_window}d reb={reb_freq}d n={n_leg} c={cost}bp"
                print(f"  {label}: SR={sr:.2f}, DD={dd:.1f}%, t={t:.2f}, "
                      f"bps/d={m*10000:.1f}")

                if best_pnl is None or sr > best_sr:
                    best_sr = sr
                    best_pnl = pnl_s
                    best_label = f"CS_mom {label}"

    # ================================================================
    # Correlation with LogReg (if we have 15m LogReg daily PnL)
    # ================================================================
    print(f"\n{'='*80}")
    print("DECISION")
    print(f"{'='*80}")

    if best_sr < 0.5:
        print(f"  KILL: Best SR = {best_sr:.2f} < 0.5 threshold")
        print(f"  Best config: {best_label}")
    else:
        print(f"  PASS: Best SR = {best_sr:.2f} >= 0.5")
        print(f"  Best config: {best_label}")
        print(f"  (Correlation with LogReg needs to be measured against production daily PnL)")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
