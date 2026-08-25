"""Cointegration Pair Expansion — Screen all 435 pairs independently.

Uses our validated bar-level simulation (not Codex code) to find
additional Sharpe-positive pairs beyond the top 3.

For each candidate pair:
1. Engle-Granger cointegration test (p < 0.05)
2. Half-life filter (1-20 days)
3. Beta stability (max/min < 3.0)
4. Bar-level walk-forward simulation with embargo
5. Keep pairs with Sharpe > 0.8 over full test period

Parallelized across CPU cores for speed.
"""

import sys, os, time, warnings, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")
DATA_DIR = Path("data/binance-1m")

# All assets with sufficient data
ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "ENJ": "ENJUSDT", "TAO": "TAOUSDT", "ADA": "ADAUSDT",
    "ANKR": "ANKRUSDT", "SUI": "SUIUSDT", "AVAX": "AVAXUSDT", "LINK": "LINKUSDT",
    "NEAR": "NEARUSDT", "FIL": "FILUSDT", "DOT": "DOTUSDT", "SXP": "SXPUSDT",
    "ENA": "ENAUSDT", "BCH": "BCHUSDT", "LTC": "LTCUSDT", "UNI": "UNIUSDT",
    "WLD": "WLDUSDT", "TRX": "TRXUSDT", "FTM": "FTMUSDT", "AAVE": "AAVEUSDT",
    "FET": "FETUSDT", "APT": "APTUSDT", "HBAR": "HBARUSDT", "XLM": "XLMUSDT",
    "DOGE": "DOGEUSDT", "ETC": "ETCUSDT",
}

EMBARGO = 96 * 7
MAX_HOLD = 96 * 30
CAPITAL = 10000.0
SIZE_PCT = 0.05
TEST_START = "2024-07-01"
TEST_END = "2026-02-28"
CAL_DAYS = 611
MIN_BARS_TRAIN = 96 * 30  # 30 days min training
ENTRY_Z_GRID = [1.5, 2.0, 2.5]
LB_GRID = [1, 2, 4]
COST_BPS = 5.0  # conservative default


def load_15m_close(symbol):
    from signal_engine.data import load_binance_1m, resample_ohlcv
    try:
        o = load_binance_1m(symbol, [DATA_DIR])
        return resample_ohlcv(o, "15min")["close"]
    except Exception:
        return None


def ols_fit(y, x):
    xm, ym = x.mean(), y.mean()
    xd = x - xm
    d = np.dot(xd, xd)
    if d <= 0: return np.nan, np.nan
    beta = np.dot(xd, y - ym) / d
    return float(beta), float(ym - beta * xm)


def half_life_bars(resid):
    if len(resid) < 50: return float("inf")
    lag = resid[:-1]; delta = resid[1:] - resid[:-1]
    d = np.dot(lag - lag.mean(), lag - lag.mean())
    if d <= 0: return float("inf")
    phi = np.dot(lag - lag.mean(), delta - delta.mean()) / d
    if phi >= 0: return float("inf")
    return -math.log(2) / phi


def beta_stability(y, x, splits=4):
    if len(y) < splits * 250: return float("inf")
    betas = []
    for ys, xs in zip(np.array_split(y, splits), np.array_split(x, splits)):
        b, _ = ols_fit(ys, xs)
        if not np.isfinite(b) or b <= 0: return float("inf")
        betas.append(abs(b))
    return max(betas) / min(betas) if min(betas) > 0 else float("inf")


def screen_pair(y_arr, x_arr):
    """Quick cointegration screen. Returns (p_value, half_life_days, beta_stab)."""
    if len(y_arr) < 5000: return None
    try:
        _, p, _ = coint(y_arr, x_arr, maxlag=10)
    except Exception:
        return None
    if p > 0.05: return None

    log_y, log_x = np.log(y_arr), np.log(x_arr)
    beta, intercept = ols_fit(log_y, log_x)
    if not np.isfinite(beta) or beta <= 0: return None

    resid = log_y - beta * log_x - intercept
    hl = half_life_bars(resid) / 96  # to days
    if hl < 0.25 or hl > 20: return None

    bs = beta_stability(log_y, log_x)
    if bs > 3.0: return None

    return {"p_value": float(p), "half_life_days": hl, "beta_stability": bs}


def simulate_pair_quick(y_close, x_close, lb_months, entry_z, cost_bps):
    """Quick bar-level simulation. Returns Sharpe and metrics."""
    common = y_close.index.intersection(x_close.index)
    y = y_close.reindex(common).astype(float)
    x = x_close.reindex(common).astype(float)

    log_y = np.log(y.values); log_x = np.log(x.values)
    y_ret = np.zeros(len(y)); x_ret = np.zeros(len(x))
    y_ret[1:] = y.values[1:] / y.values[:-1] - 1
    x_ret[1:] = x.values[1:] / x.values[:-1] - 1

    ws = pd.Timestamp(TEST_START, tz="UTC")
    we = pd.Timestamp(TEST_END, tz="UTC")
    test_mask = (y.index >= ws) & (y.index <= we)
    test_idx = y.index[test_mask]
    if len(test_idx) < 96 * 60: return None  # need 2+ months

    idx_map = {ts: i for i, ts in enumerate(y.index)}
    pos = 0; entry_bar = 0
    beta = np.nan; intercept = np.nan; mu = np.nan; sigma = np.nan
    last_month = None
    daily_pnl = {}; n_trades = 0; n_wins = 0

    for ts in test_idx:
        i = idx_map[ts]
        cm = ts.to_period("M")

        if cm != last_month:
            te = ts - pd.Timedelta(minutes=15 * EMBARGO)
            t_start = te - pd.DateOffset(months=lb_months)
            tm = (y.index >= t_start) & (y.index < te)
            if tm.sum() >= MIN_BARS_TRAIN:
                ty = log_y[tm]; tx = log_x[tm]
                beta, intercept = ols_fit(ty, tx)
                if np.isfinite(beta) and beta > 0:
                    tr = ty - beta * tx - intercept
                    mu = float(np.mean(tr)); sigma = float(np.std(tr))
                    if sigma <= 1e-12: sigma = np.nan
                else: beta = np.nan
            last_month = cm

        if not np.isfinite(beta) or not np.isfinite(sigma): continue

        spread = log_y[i] - beta * log_x[i] - intercept
        z = (spread - mu) / sigma
        notional = CAPITAL * SIZE_PCT
        bar_pnl = 0.0
        if pos != 0:
            hr = y_ret[i] - beta * x_ret[i]
            bar_pnl = pos * hr * notional

        cost = cost_bps / 10000 * (1 + abs(beta)) * notional

        if pos != 0:
            bh = i - entry_bar
            exit_it = False
            if pos == 1 and z >= 0: exit_it = True
            elif pos == -1 and z <= 0: exit_it = True
            elif bh >= MAX_HOLD: exit_it = True
            if exit_it:
                bar_pnl -= cost; pos = 0
                if bar_pnl > 0: n_wins += 1

        if pos == 0:
            if z <= -entry_z: pos = 1; entry_bar = i; bar_pnl -= cost; n_trades += 1
            elif z >= entry_z: pos = -1; entry_bar = i; bar_pnl -= cost; n_trades += 1

        day = ts.normalize()
        daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl

    if n_trades < 3: return None

    dpnl = pd.Series(daily_pnl)
    dr = dpnl / CAPITAL
    sharpe = dr.mean() / dr.std() * np.sqrt(365.25) if len(dr) > 10 and dr.std() > 0 else 0
    cum = dpnl.cumsum()
    peak = (cum + CAPITAL).cummax()
    dd = abs(((cum + CAPITAL - peak) / peak).min())
    wd = abs(dr.min()) if len(dr) > 0 else 0
    pnl = cum.iloc[-1]
    wr = n_wins / max(n_trades, 1) * 100

    return {
        "sharpe": sharpe, "pnl": pnl, "pnl_cal": pnl / CAL_DAYS,
        "max_dd": dd, "worst_day": wd, "n_trades": n_trades, "win_rate": wr,
    }


def evaluate_pair(args):
    """Worker: screen + simulate one pair across param grid."""
    y_name, x_name, y_data_path, x_data_path = args

    # Lazy load in worker
    sys.path.insert(0, os.path.join(_root, "src"))
    from signal_engine.data import load_binance_1m, resample_ohlcv
    try:
        y_close = pd.read_parquet(y_data_path) if y_data_path.endswith(".parquet") else None
        x_close = pd.read_parquet(x_data_path) if x_data_path.endswith(".parquet") else None
        if y_close is None or x_close is None: return []
        y_close = y_close.iloc[:, 0]; x_close = x_close.iloc[:, 0]
    except Exception:
        return []

    # Quick screen on overlapping data
    common = y_close.index.intersection(x_close.index)
    if len(common) < 5000: return []
    y_arr = y_close.reindex(common).values.astype(float)
    x_arr = x_close.reindex(common).values.astype(float)

    screen = screen_pair(y_arr, x_arr)
    if screen is None: return []

    # Simulate across param grid
    results = []
    for lb in LB_GRID:
        for ez in ENTRY_Z_GRID:
            r = simulate_pair_quick(y_close, x_close, lb, ez, COST_BPS)
            if r and r["sharpe"] > 0:
                results.append({
                    "pair": f"{y_name}/{x_name}",
                    "y": y_name, "x": x_name,
                    "lb_months": lb, "entry_z": ez,
                    "coint_p": screen["p_value"],
                    "hl_days": screen["half_life_days"],
                    "beta_stab": screen["beta_stability"],
                    **r,
                })
    return results


def main():
    t0 = time.time()
    print("=" * 110)
    print("COINTEGRATION PAIR EXPANSION — Screen All 435 Pairs")
    print("=" * 110)

    # Pre-cache 15m close as parquet for fast worker loading
    cache_dir = Path("data/coint_cache")
    cache_dir.mkdir(exist_ok=True)

    print("\nPre-caching 15m close data...", flush=True)
    asset_paths = {}
    for name, symbol in ASSETS.items():
        cache_path = cache_dir / f"{name}_15m_close.parquet"
        if not cache_path.exists():
            s = load_15m_close(symbol)
            if s is not None:
                pd.DataFrame({"close": s}).to_parquet(cache_path)
                asset_paths[name] = str(cache_path)
                print(f"  {name}: {len(s)} bars", flush=True)
        else:
            asset_paths[name] = str(cache_path)

    print(f"\n{len(asset_paths)} assets cached. Generating pair tasks...")

    # Generate all pairs
    names = sorted(asset_paths.keys())
    tasks = []
    for y_name, x_name in combinations(names, 2):
        tasks.append((y_name, x_name, asset_paths[y_name], asset_paths[x_name]))

    print(f"{len(tasks)} pairs to screen across {len(LB_GRID)}x{len(ENTRY_Z_GRID)} = {len(LB_GRID)*len(ENTRY_Z_GRID)} configs")

    # Parallel execution
    n_workers = min(10, os.cpu_count() or 4)
    all_results = []
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(evaluate_pair, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)} pairs done ({time.time()-t0:.0f}s)", flush=True)
            try:
                results = future.result()
                all_results.extend(results)
            except Exception:
                pass

    print(f"\nScreening done in {time.time()-t0:.0f}s. {len(all_results)} positive configs found.")

    if not all_results:
        print("NO COINTEGRATED PAIRS FOUND")
        return

    df = pd.DataFrame(all_results)

    # Best config per pair (highest Sharpe)
    best = df.loc[df.groupby("pair")["sharpe"].idxmax()].sort_values("sharpe", ascending=False)

    print(f"\n{'='*110}")
    print(f"TOP PAIRS BY SHARPE (best config per pair, Sharpe > 0.8)")
    print(f"{'='*110}")

    strong = best[best["sharpe"] > 0.8]
    print(f"\n{'Pair':<15s} {'LB':>3s} {'Z':>4s} {'Sharpe':>7s} {'$/cal':>7s} {'DD%':>6s} "
          f"{'WD%':>6s} {'#Tr':>5s} {'WR%':>5s} {'p_coint':>8s} {'HL(d)':>6s} {'BetaSt':>7s}")
    print("-" * 110)

    for _, r in strong.iterrows():
        ok = r["max_dd"] < 0.10 and r["worst_day"] < 0.05
        print(f"{r['pair']:<15s} {r['lb_months']:>3d} {r['entry_z']:>4.1f} {r['sharpe']:>7.2f} "
              f"${r['pnl_cal']:>6.2f} {r['max_dd']*100:>5.1f}% {r['worst_day']*100:>5.2f}% "
              f"{r['n_trades']:>5.0f} {r['win_rate']:>4.0f}% {r['coint_p']:>8.4f} "
              f"{r['hl_days']:>5.1f}d {r['beta_stab']:>6.2f}  {'OK' if ok else 'FAIL'}")

    # Portfolio: top 5 pairs equal-weight
    print(f"\n{'='*110}")
    print("PORTFOLIO: Top 5 Pairs Combined (5% per leg)")
    print(f"{'='*110}")

    top5 = strong.head(5)
    if len(top5) > 0:
        total_pnl = top5["pnl"].sum()
        avg_sharpe = top5["sharpe"].mean()
        max_dd = top5["max_dd"].max()
        max_wd = top5["worst_day"].max()
        total_trades = top5["n_trades"].sum()
        print(f"\nPairs: {', '.join(top5['pair'].values)}")
        print(f"Total PnL: ${total_pnl:.0f}")
        print(f"$/cal: ${total_pnl/CAL_DAYS:.2f}")
        print(f"Avg pair Sharpe: {avg_sharpe:.2f}")
        print(f"Max DD (per pair): {max_dd*100:.1f}%")
        print(f"Max WD (per pair): {max_wd*100:.2f}%")
        print(f"Total trades: {total_trades:.0f}")

    # Also show overlap — which assets appear most
    print(f"\n{'='*110}")
    print("ASSET CONCENTRATION (in top Sharpe>0.8 pairs)")
    print(f"{'='*110}")

    if len(strong) > 0:
        all_assets = list(strong["y"]) + list(strong["x"])
        from collections import Counter
        counts = Counter(all_assets)
        for asset, count in counts.most_common(10):
            print(f"  {asset}: appears in {count} pairs")

    # Save results
    out_dir = Path("results/coint_optimization")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "pair_expansion_all.csv", index=False)
    best.to_csv(out_dir / "pair_expansion_best.csv", index=False)
    if len(strong) > 0:
        strong.to_csv(out_dir / "pair_expansion_strong.csv", index=False)

    print(f"\nResults saved to results/coint_optimization/")
    print(f"Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
