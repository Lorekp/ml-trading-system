"""Phase 1: Exhaustive Pair Scan — Find the best cointegrated pairs pre-2025.

Scans all C(57,2) = 1,596 pairs across 8 rolling 6-month windows.
Computes persistence, half-life, Hurst, beta stability, profitability.
Outputs ranked list for manual selection.
"""

import sys, os, time, warnings, math
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/cache_15m")
OUT_DIR = Path("results/coint_ml")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")
MIN_BARS = 96 * 180  # 180 days minimum
ENTRY_Z = 2.0; MIN_HOLD = 96; MAX_HOLD = 96 * 30

SECTORS = {
    "L1": ["BTC","ETH","SOL","ADA","AVAX","NEAR","DOT","APT","SUI"],
    "DeFi": ["UNI","AAVE","LINK","FTM"],
    "AI": ["FET","TAO"],
    "Storage": ["FIL","ANKR"],
    "Gaming": ["ENJ","AXS","GALA"],
    "Exchange": ["BNB","TRX"],
    "Payment": ["XRP","XLM","LTC","BCH"],
    "Meme": ["DOGE","1000PEPE","WIF","1000SHIB","1000BONK","PENGU"],
    "Privacy": ["ZEC","DASH"],
    "L2": ["ARB","WLD"],
    "Misc": ["ENA","ETC","SXP","HBAR","BNX","POLYX","VANRY","ALPHA",
             "LINA","OMNI","LIT","HIFI","LEVER","AMB","PIXEL"],
}
ASSET_SECTOR = {}
for s, assets in SECTORS.items():
    for a in assets: ASSET_SECTOR[a] = s


def ols_fit(y, x):
    xm, ym = x.mean(), y.mean(); xd = x - xm; d = np.dot(xd, xd)
    if d <= 0: return np.nan, np.nan
    b = np.dot(xd, y - ym) / d; return float(b), float(ym - b * xm)


def half_life_bars(resid):
    if len(resid) < 50: return float("inf")
    lag = resid[:-1]; delta = np.diff(resid)
    d = np.dot(lag - lag.mean(), lag - lag.mean())
    if d <= 0: return float("inf")
    phi = np.dot(lag - lag.mean(), delta - delta.mean()) / d
    return -math.log(2) / phi if phi < 0 else float("inf")


def hurst_exponent(series, max_lag=64):
    if len(series) < max_lag * 3: return 0.5
    lags = np.arange(2, min(max_lag + 1, len(series) // 3))
    tau = []
    for lag in lags:
        s = np.std(series[lag:] - series[:-lag])
        tau.append(s if s > 0 and np.isfinite(s) else np.nan)
    tau_arr = np.array(tau); keep = np.isfinite(tau_arr) & (tau_arr > 0)
    if keep.sum() < 5: return 0.5
    return float(np.clip(np.polyfit(np.log(lags[keep]), np.log(tau_arr[keep]), 1)[0], 0, 1))


def simulate_pair_pnl(y_close, x_close, beta, intercept, mu, sigma):
    """Quick z-score sim on arrays. Returns total hedged return (not $)."""
    if len(y_close) < 500 or sigma <= 0: return 0.0
    log_y = np.log(y_close); log_x = np.log(x_close)
    y_ret = np.diff(y_close) / y_close[:-1]
    x_ret = np.diff(x_close) / x_close[:-1]
    pos = 0; pnl = 0.0; entry_i = 0
    for i in range(1, len(y_close)):
        spread = log_y[i] - beta * log_x[i] - intercept
        z = (spread - mu) / sigma
        if pos != 0:
            pnl += pos * (y_ret[i-1] - beta * x_ret[i-1])
            bh = i - entry_i
            ex = False
            if bh >= MIN_HOLD:
                if pos == 1 and z >= 0: ex = True
                elif pos == -1 and z <= 0: ex = True
            if bh >= MAX_HOLD: ex = True
            if ex: pos = 0
        if pos == 0:
            if z <= -ENTRY_Z: pos = 1; entry_i = i
            elif z >= ENTRY_Z: pos = -1; entry_i = i
    return pnl


def scan_pair_window(y_arr, x_arr):
    """Scan one pair on one window. Returns dict of metrics or None."""
    from statsmodels.tsa.stattools import coint as _coint
    if len(y_arr) < 3000: return None

    # Correlation prescreen
    ry = np.diff(y_arr) / y_arr[:-1]; rx = np.diff(x_arr) / x_arr[:-1]
    v = np.isfinite(ry) & np.isfinite(rx)
    if v.sum() < 2000: return None
    corr = float(np.corrcoef(ry[v], rx[v])[0, 1])
    if not np.isfinite(corr) or corr < 0.30: return None

    # EG test
    try:
        _, p, _ = _coint(y_arr, x_arr, maxlag=10)
    except: return None

    # OLS on log prices
    log_y = np.log(y_arr); log_x = np.log(x_arr)
    beta, intercept = ols_fit(log_y, log_x)
    if not np.isfinite(beta) or beta <= 0: return None
    resid = log_y - beta * log_x - intercept
    hl = half_life_bars(resid) / 96  # days
    h = hurst_exponent(resid)
    mu = float(np.mean(resid)); sigma = float(np.std(resid))

    # Profitability
    pnl = simulate_pair_pnl(y_arr, x_arr, beta, intercept, mu, sigma)

    return {
        "eg_p": float(p), "passes": p < 0.05,
        "hl_days": min(hl, 100), "hurst": h,
        "beta": beta, "sigma": sigma, "corr": corr,
        "pnl": pnl,
    }


def process_pair(args):
    """Worker: scan one pair across all windows."""
    y_name, x_name, y_series, x_series, windows = args
    common = y_series.index.intersection(x_series.index)
    if len(common) < 5000: return None

    y_full = y_series.reindex(common).values.astype(float)
    x_full = x_series.reindex(common).values.astype(float)
    idx = y_series.reindex(common).index

    results = []
    betas = []
    for ws, we in windows:
        mask = (idx >= ws) & (idx < we)
        y_w = y_full[mask]; x_w = x_full[mask]
        if len(y_w) < 3000: continue
        r = scan_pair_window(y_w, x_w)
        if r is not None:
            results.append(r)
            if r["passes"]: betas.append(r["beta"])

    if len(results) < 3: return None  # need at least 3 windows

    n_pass = sum(1 for r in results if r["passes"])
    persistence = n_pass / len(results)
    avg_hl = np.mean([r["hl_days"] for r in results])
    avg_hurst = np.mean([r["hurst"] for r in results])
    avg_corr = np.mean([r["corr"] for r in results])
    total_pnl = sum(r["pnl"] for r in results)
    avg_eg_p = np.mean([r["eg_p"] for r in results])

    # Beta stability
    if len(betas) >= 2:
        beta_stab = max(betas) / min(betas) if min(betas) > 0 else 99
    else:
        beta_stab = 99

    # Sector match
    sy = ASSET_SECTOR.get(y_name, "?"); sx = ASSET_SECTOR.get(x_name, "?")
    sector_match = 1 if sy == sx and sy != "?" and sy != "Misc" else 0

    return {
        "pair": f"{y_name}/{x_name}", "y": y_name, "x": x_name,
        "n_windows": len(results), "n_pass": n_pass,
        "persistence": persistence, "avg_hl": avg_hl,
        "avg_hurst": avg_hurst, "avg_corr": avg_corr,
        "beta_stability": min(beta_stab, 10),
        "total_pnl": total_pnl, "avg_eg_p": avg_eg_p,
        "sector_y": sy, "sector_x": sx, "sector_match": sector_match,
    }


def main():
    t0 = time.time()
    print("=" * 110)
    print("EXHAUSTIVE PAIR SCAN - Pre-2025 Data")
    print("=" * 110)

    # Load data
    print("\nLoading...", end=" ", flush=True)
    close_dict = {}; vol_dict = {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        name = f.stem.replace("USDT", "")
        try:
            df = pd.read_parquet(f, columns=["close", "volume"])
            pre = df[df.index < CUTOFF]
            if len(pre) >= MIN_BARS:
                close_dict[name] = pre["close"]
                dv = (pre["close"] * pre["volume"]).resample("1D").sum()
                vol_dict[name] = dv.iloc[-90:].median() if len(dv) >= 90 else dv.median()
        except: pass
    print(f"{len(close_dict)} assets")

    # Define rolling 6-month windows (quarterly steps)
    windows = []
    for start_q in pd.date_range("2023-01-01", "2024-07-01", freq="QS", tz="UTC"):
        end = start_q + pd.DateOffset(months=6)
        if end <= CUTOFF:
            windows.append((start_q, end))
    print(f"Windows: {len(windows)} (6-month, quarterly steps)")
    for ws, we in windows:
        print(f"  {ws.strftime('%Y-%m')} to {we.strftime('%Y-%m')}")

    # Build pair tasks
    assets = sorted(close_dict.keys())
    pair_args = []
    for a_i, a_j in combinations(assets, 2):
        pair_args.append((a_i, a_j, close_dict[a_i], close_dict[a_j], windows))
    print(f"\nScanning {len(pair_args)} pairs...", flush=True)

    # Parallel scan
    results = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(process_pair, args): args[:2] for args in pair_args}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(pair_args)}...", flush=True)
            r = fut.result()
            if r is not None:
                results.append(r)

    print(f"\n{len(results)} pairs with 3+ valid windows")

    # Compute composite score
    for r in results:
        # Only score pairs that have SOME cointegration
        if r["persistence"] == 0:
            r["score"] = 0
            continue
        # Score components (all [0,1] or positive)
        pers = r["persistence"]  # higher = better
        hurst_inv = max(0, 1 - r["avg_hurst"])  # lower hurst = better
        pnl_pos = max(0, r["total_pnl"])  # only count positive
        beta_inv = 1.0 / max(r["beta_stability"], 1.01)  # lower instability = better
        liq_y = vol_dict.get(r["y"], 0); liq_x = vol_dict.get(r["x"], 0)
        liq_ok = 1 if min(liq_y, liq_x) >= 10e6 else 0.5 if min(liq_y, liq_x) >= 5e6 else 0.2
        sector = 1.2 if r["sector_match"] else 1.0

        r["score"] = pers * hurst_inv * (1 + pnl_pos * 10) * beta_inv * liq_ok * sector
        r["min_vol"] = min(liq_y, liq_x) / 1e6

    # Sort by score
    results.sort(key=lambda r: -r["score"])

    # Save full results
    df = pd.DataFrame(results)
    df.to_parquet(OUT_DIR / "pair_scan_full.parquet", index=False)

    # Print top 40
    print(f"\n{'Rank':<5s} {'Pair':<15s} {'Score':>7s} {'Pers%':>6s} {'HL':>5s} "
          f"{'Hurst':>6s} {'Corr':>5s} {'BetaS':>6s} {'PnL':>8s} {'Vol$M':>6s} "
          f"{'Sector':>8s} {'EG_p':>6s}")
    print("-" * 105)

    for i, r in enumerate(results[:40]):
        sect = f"{r['sector_y']}/{r['sector_x']}" if not r["sector_match"] else r["sector_y"]
        print(f"{i+1:<5d} {r['pair']:<15s} {r['score']:>7.2f} {r['persistence']*100:>5.0f}% "
              f"{r['avg_hl']:>5.1f} {r['avg_hurst']:>6.3f} {r['avg_corr']:>5.2f} "
              f"{r['beta_stability']:>6.2f} {r['total_pnl']:>8.4f} {r['min_vol']:>6.0f} "
              f"{sect:>8s} {r['avg_eg_p']:>6.3f}")

    # Summary stats
    passing = [r for r in results if r["persistence"] > 0]
    high_pers = [r for r in results if r["persistence"] >= 0.5]
    profitable = [r for r in results if r["total_pnl"] > 0]
    both = [r for r in results if r["persistence"] >= 0.3 and r["total_pnl"] > 0]

    print(f"\nSummary:")
    print(f"  Total scanned: {len(pair_args)}")
    print(f"  With 3+ windows: {len(results)}")
    print(f"  Any coint: {len(passing)} ({len(passing)/max(len(results),1)*100:.0f}%)")
    print(f"  Persistence >= 50%: {len(high_pers)}")
    print(f"  Total PnL > 0: {len(profitable)}")
    print(f"  Pers >= 30% AND PnL > 0: {len(both)}")

    # Print candidate pairs for selection (persistence > 30%, PnL > 0, diverse)
    print(f"\n{'='*110}")
    print("CANDIDATE PAIRS FOR MANUAL SELECTION")
    print(f"(persistence >= 30%, profitable, ranked by score)")
    print(f"{'='*110}")

    selected_assets = set()
    candidates = [r for r in results if r["persistence"] >= 0.25 and r["total_pnl"] > 0]
    candidates.sort(key=lambda r: -r["score"])

    for i, r in enumerate(candidates[:25]):
        dup = "*" if r["y"] in selected_assets or r["x"] in selected_assets else " "
        sect = f"{r['sector_y']}/{r['sector_x']}" if not r["sector_match"] else f"[{r['sector_y']}]"
        print(f"  {i+1:2d}.{dup} {r['pair']:<15s} pers={r['persistence']*100:.0f}% "
              f"hl={r['avg_hl']:.1f}d hurst={r['avg_hurst']:.3f} "
              f"pnl={r['total_pnl']:.4f} vol=${r['min_vol']:.0f}M {sect}")
        selected_assets.add(r["y"]); selected_assets.add(r["x"])

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
