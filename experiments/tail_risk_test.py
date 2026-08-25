"""Test 3 tail-risk reduction methods on T165.

1. Intraday loss cap (2%, 3%, 4%)
2. Correlation regime throttle
3. HRP allocation (replace NCO)
+ Combinations

All use T165's cached signals — no signal generation needed.
"""

import sys, os, time, warnings, pickle
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np, pandas as pd
from signal_engine.config import PortfolioConfig
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
import test_original_pipeline as orig
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/optuna/signal_cache")
W = [("W1", 245), ("W2", 244), ("W3", 122)]
C_STR = "0.02000"; LB = 2; THR = 0.10; FWD = 6; LEV = 5.0; DLV = 0.05
CAPITAL = 10000.0


def load_signals():
    """Load and filter T165 signals from cache."""
    all_fi = {}
    for wl, cd in W:
        cf = CACHE_DIR / f"C{C_STR}_lb{LB}_{wl}.pkl"
        with open(cf, "rb") as f: raw = pickle.load(f)
        fi = raw[raw["confidence"] > THR].copy()
        fi["fwd_return"] = fi[f"ret_{FWD}"].values
        all_fi[wl] = fi
    return all_fi


def simulate_with_daily_cap(signals_df, pcfg, daily_cap_pct):
    """Portfolio sim with intraday loss cap. Stops opening new positions
    when cumulative daily loss exceeds threshold."""
    sub = signals_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sub)
    if n < 10: return None

    timestamps = sub["timestamp"].values
    directions = sub["direction"].values.astype(np.float64)
    fwd_returns = sub["fwd_return"].values.astype(np.float64)
    sig_strengths = sub["signal_strength"].values.astype(np.float64) if "signal_strength" in sub.columns else np.ones(n)

    equity = pcfg.starting_capital
    peak_equity = equity
    halted = False
    active_positions = []
    fwd_bars = 6

    unique_ts = pd.Series(timestamps).unique()
    ts_to_idx = {ts: idx for idx, ts in enumerate(unique_ts)}

    equity_arr = np.empty(n)
    pnl_arr = np.zeros(n)
    size_arr = np.zeros(n)
    skipped = np.ones(n, dtype=bool)

    # Daily tracking
    current_date = None
    daily_pnl_accum = 0.0

    for i in range(n):
        current_ts = timestamps[i]
        current_ts_idx = ts_to_idx.get(current_ts, 0)
        active_positions = [(e, s, a) for e, s, a in active_positions if e > current_ts_idx]

        # Reset daily accumulator on new day
        ts_date = pd.Timestamp(current_ts).date()
        if ts_date != current_date:
            current_date = ts_date
            daily_pnl_accum = 0.0

        current_exposure = sum(s for _, s, _ in active_positions)

        if halted:
            equity_arr[i] = equity
            continue

        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd >= pcfg.max_dd:
            halted = True
            equity_arr[i] = equity
            continue

        # Daily cap check: skip new positions if daily loss exceeds cap
        if daily_cap_pct > 0 and daily_pnl_accum / max(equity, 1) < -daily_cap_pct:
            equity_arr[i] = equity
            continue

        # Dynamic deleveraging
        if dd > pcfg.delever_start:
            dd_ratio = (dd - pcfg.delever_start) / (pcfg.max_dd - pcfg.delever_start)
            size_mult = max(1.0 - dd_ratio, 0.0)
        else:
            size_mult = 1.0

        sig_abs = abs(sig_strengths[i])
        raw_size = equity * pcfg.base_sizing_pct * sig_abs * size_mult
        raw_size = min(raw_size, equity * pcfg.base_sizing_pct * pcfg.max_signal_mult * size_mult)

        asset = sub.iloc[i].get("asset", "X")
        asset_exp = sum(s for _, s, a in active_positions if a == asset)
        if asset_exp + raw_size > equity * pcfg.max_per_asset:
            raw_size = max(equity * pcfg.max_per_asset - asset_exp, 0.0)
        if current_exposure + raw_size > equity * pcfg.max_exposure:
            raw_size = max(equity * pcfg.max_exposure - current_exposure, 0.0)

        if raw_size < 1.0:
            equity_arr[i] = equity
            continue

        pnl = raw_size * (directions[i] * fwd_returns[i] - pcfg.slippage)
        equity += pnl
        daily_pnl_accum += pnl

        if equity > peak_equity:
            peak_equity = equity

        pnl_arr[i] = pnl
        size_arr[i] = raw_size
        skipped[i] = False
        equity_arr[i] = equity

        expiry_idx = current_ts_idx + fwd_bars
        active_positions.append((expiry_idx, raw_size, asset))

    n_trades = (~skipped).sum()
    return {
        "equity": equity_arr, "pnl": pnl_arr, "sizes": size_arr,
        "skipped": skipped, "timestamps": timestamps,
        "halted": halted, "final_equity": equity, "n_trades": int(n_trades),
    }


def apply_corr_throttle(signals_df, high_corr=0.7, reduction=0.5):
    """Reduce signal_strength when recent trade returns are highly correlated."""
    df = signals_df.copy()
    ts = pd.to_datetime(df["timestamp"])

    # Compute per-asset returns and rolling correlation
    asset_returns = {}
    for asset in df["asset"].unique():
        mask = df["asset"] == asset
        rets = pd.Series(
            df.loc[mask, "direction"].values * df.loc[mask, "fwd_return"].values,
            index=ts[mask],
        )
        asset_returns[asset] = rets

    if len(asset_returns) < 3:
        return df

    ret_df = pd.DataFrame(asset_returns).resample("D").sum().fillna(0)
    rolling_corr = ret_df.rolling(20).corr()

    # For each day, compute mean pairwise correlation
    daily_mean_corr = {}
    for date in ret_df.index:
        try:
            corr_matrix = rolling_corr.loc[date]
            if isinstance(corr_matrix, pd.DataFrame) and len(corr_matrix) > 1:
                mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                vals = corr_matrix.values[mask]
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    daily_mean_corr[date.date()] = np.mean(vals)
        except Exception:
            pass

    # Apply throttle
    dates = pd.to_datetime(df["timestamp"]).dt.date
    for idx in df.index:
        d = dates[idx]
        mc = daily_mean_corr.get(d, 0.3)
        if mc > high_corr:
            df.at[idx, "signal_strength"] = df.at[idx, "signal_strength"] * reduction

    return df


def apply_hrp_wf(combined):
    """Walk-forward HRP allocation (replace NCO)."""
    from signal_engine.allocation import _denoise_corr, _normalize

    combined = combined.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(combined["timestamp"])
    months = sorted(ts.dt.to_period("M").unique())
    parts = []

    for km in months:
        km_sigs = combined[ts.dt.to_period("M") == km]
        past_sigs = combined[ts.dt.to_period("M") < km]

        if len(past_sigs) < 50:
            parts.append(km_sigs)
            continue

        assets_in_month = sorted(km_sigs["asset"].unique())
        daily_rets = {}
        for a in assets_in_month:
            a_past = past_sigs[past_sigs["asset"] == a]
            if len(a_past) < 10: continue
            ts_a = pd.to_datetime(a_past["timestamp"])
            rets = pd.Series(a_past["direction"].values * a_past["fwd_return"].values, index=ts_a)
            daily_rets[a] = rets.resample("D").sum()

        if len(daily_rets) < 3:
            parts.append(km_sigs)
            continue

        ret_df = pd.DataFrame(daily_rets).fillna(0)
        cov = ret_df.cov().values
        corr = ret_df.corr().values
        names = list(ret_df.columns)
        n = len(names)

        # HRP: cluster → quasi-diagonalize → bisect allocation
        dist = np.sqrt(np.maximum(0.5 * (1 - corr), 0))
        np.fill_diagonal(dist, 0)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="single")
        sort_ix = leaves_list(link).astype(int)

        # Bisection allocation
        weights = np.ones(n)
        clusters = [sort_ix.tolist()]

        while len(clusters) > 0:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Variance of each half
                var_left = np.mean([cov[i, i] for i in left]) if left else 1.0
                var_right = np.mean([cov[i, i] for i in right]) if right else 1.0

                # Inverse-variance split
                total = 1.0 / max(var_left, 1e-10) + 1.0 / max(var_right, 1e-10)
                w_left = (1.0 / max(var_left, 1e-10)) / total
                w_right = (1.0 / max(var_right, 1e-10)) / total

                for i in left:
                    weights[i] *= w_left
                for i in right:
                    weights[i] *= w_right

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        # Normalize
        weight_dict = {names[i]: float(weights[i]) for i in range(n)}
        mean_w = np.mean(list(weight_dict.values()))
        if mean_w > 0:
            weight_dict = {k: v / mean_w for k, v in weight_dict.items()}

        kmc = km_sigs.copy()
        kmc["signal_strength"] = kmc.apply(
            lambda r: r["signal_strength"] * weight_dict.get(r["asset"], 1.0), axis=1)
        parts.append(kmc)

    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def run_variant(all_fi, label, alloc_fn=None, daily_cap=0.0, corr_throttle=None):
    """Run one variant across all windows."""
    total_pnl = 0; total_cal = 0; worst_dd = 0; worst_day = 0
    total_trades = 0; total_days = 0

    for (wl, cd) in W:
        fi = all_fi[wl]
        if len(fi) < 10: continue

        # Apply correlation throttle if specified
        if corr_throttle:
            fi = apply_corr_throttle(fi, **corr_throttle)

        # Apply allocation
        if alloc_fn:
            combined = alloc_fn(fi)
        else:
            combined = orig.apply_nco_wf(fi)

        pcfg = PortfolioConfig(
            starting_capital=CAPITAL, max_dd=0.10, delever_start=DLV,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=0.04 * LEV, max_signal_mult=3.0,
        )

        # Use custom sim with daily cap, or standard
        if daily_cap > 0:
            sim = simulate_with_daily_cap(combined, pcfg, daily_cap)
        else:
            from signal_engine.portfolio import simulate_portfolio
            sim = simulate_portfolio(combined, pcfg)

        if sim is None or sim["n_trades"] < 5: continue
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 1: continue

        eq = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
        dd = ((eq - eq.expanding().max()) / eq.expanding().max())
        daily_ret = dpnl / CAPITAL

        total_pnl += sim["equity"][-1] - CAPITAL
        total_cal += cd; total_trades += sim["n_trades"]; total_days += len(dpnl)
        worst_dd = max(worst_dd, abs(dd.min()))
        worst_day = max(worst_day, abs(daily_ret.min()))

    if total_cal == 0: return None
    pcal = total_pnl / total_cal
    d2t = int(CAPITAL * 0.10 / pcal) if pcal > 0 else 9999
    act = total_days / total_cal * 100
    ok = worst_dd < 0.10 and worst_day < 0.05
    return {
        "label": label, "pnl_cal": pcal, "d2t": d2t, "dd": worst_dd,
        "wd": worst_day, "act": act, "trades": total_trades, "ok": ok,
    }


def main():
    t0 = time.time()
    print("=" * 95)
    print("TAIL-RISK REDUCTION TEST ON T165")
    print(f"  Baseline: C=0.02, lb=2, thr=0.10, fixed T+6, 5x, dlv=0.05")
    print("=" * 95)

    all_fi = load_signals()
    print(f"  Loaded signals: {', '.join(f'{wl}={len(fi)}' for wl, fi in all_fi.items())}")

    results = []

    # Baseline
    print("\nRunning variants...", flush=True)
    r = run_variant(all_fi, "Baseline (NCO)")
    if r: results.append(r); print(f"  {r['label']:<40s} ${r['pnl_cal']:>6.2f}/cal ~{r['d2t']:>3d}d DD={r['dd']:.1%} WD={r['wd']:.1%} {'OK' if r['ok'] else 'no'}", flush=True)

    # Method 1: Intraday loss cap
    for cap in [0.02, 0.03, 0.04]:
        r = run_variant(all_fi, f"Daily cap {cap:.0%}", daily_cap=cap)
        if r: results.append(r); print(f"  {r['label']:<40s} ${r['pnl_cal']:>6.2f}/cal ~{r['d2t']:>3d}d DD={r['dd']:.1%} WD={r['wd']:.1%} {'OK' if r['ok'] else 'no'}", flush=True)

    # Method 2: Correlation throttle
    for hc in [0.6, 0.7, 0.8]:
        for red in [0.3, 0.5]:
            r = run_variant(all_fi, f"Corr>{hc} x{red}", corr_throttle={"high_corr": hc, "reduction": red})
            if r: results.append(r); print(f"  {r['label']:<40s} ${r['pnl_cal']:>6.2f}/cal ~{r['d2t']:>3d}d DD={r['dd']:.1%} WD={r['wd']:.1%} {'OK' if r['ok'] else 'no'}", flush=True)

    # Method 3: HRP allocation
    r = run_variant(all_fi, "HRP allocation", alloc_fn=apply_hrp_wf)
    if r: results.append(r); print(f"  {r['label']:<40s} ${r['pnl_cal']:>6.2f}/cal ~{r['d2t']:>3d}d DD={r['dd']:.1%} WD={r['wd']:.1%} {'OK' if r['ok'] else 'no'}", flush=True)

    # Combinations
    for cap in [0.03, 0.04]:
        r = run_variant(all_fi, f"HRP + cap {cap:.0%}", alloc_fn=apply_hrp_wf, daily_cap=cap)
        if r: results.append(r); print(f"  {r['label']:<40s} ${r['pnl_cal']:>6.2f}/cal ~{r['d2t']:>3d}d DD={r['dd']:.1%} WD={r['wd']:.1%} {'OK' if r['ok'] else 'no'}", flush=True)

    for cap in [0.03, 0.04]:
        r = run_variant(all_fi, f"Corr>0.7 x0.5 + cap {cap:.0%}",
                        corr_throttle={"high_corr": 0.7, "reduction": 0.5}, daily_cap=cap)
        if r: results.append(r); print(f"  {r['label']:<40s} ${r['pnl_cal']:>6.2f}/cal ~{r['d2t']:>3d}d DD={r['dd']:.1%} WD={r['wd']:.1%} {'OK' if r['ok'] else 'no'}", flush=True)

    # Triple combo
    r = run_variant(all_fi, "HRP + Corr>0.7 x0.5 + cap 3%",
                    alloc_fn=apply_hrp_wf,
                    corr_throttle={"high_corr": 0.7, "reduction": 0.5}, daily_cap=0.03)
    if r: results.append(r); print(f"  {r['label']:<40s} ${r['pnl_cal']:>6.2f}/cal ~{r['d2t']:>3d}d DD={r['dd']:.1%} WD={r['wd']:.1%} {'OK' if r['ok'] else 'no'}", flush=True)

    # Summary
    print(f"\n{'='*95}")
    print("RESULTS SORTED BY PnL/cal_day (feasible only)")
    print(f"{'='*95}")
    print(f"\n  {'Method':<40s} {'$/cal':>7s} {'~Days':>5s} {'DD':>6s} {'WD':>6s} {'Act%':>5s}")
    print("  " + "-" * 70)

    results.sort(key=lambda x: -x["pnl_cal"])
    for r in results:
        marker = ">>>" if r["ok"] else "   "
        print(f"  {marker} {r['label']:<37s} ${r['pnl_cal']:>6.2f} {r['d2t']:>5d} "
              f"{r['dd']*100:>5.1f}% {r['wd']*100:>5.1f}% {r['act']:>4.0f}%")

    # Best feasible
    feasible = [r for r in results if r["ok"]]
    if feasible:
        b = max(feasible, key=lambda x: x["pnl_cal"])
        print(f"\n  BEST FEASIBLE: {b['label']}")
        print(f"    ${b['pnl_cal']:.2f}/cal_day, ~{b['d2t']} days, DD={b['dd']:.1%}, WD={b['wd']:.1%}")

    baseline = next((r for r in results if r["label"] == "Baseline (NCO)"), None)
    if baseline and feasible:
        pnl_loss = (baseline["pnl_cal"] - b["pnl_cal"]) / baseline["pnl_cal"] * 100
        wd_improve = (baseline["wd"] - b["wd"]) / baseline["wd"] * 100
        print(f"    vs baseline: PnL {pnl_loss:+.1f}%, worst day {wd_improve:+.1f}% improvement")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
