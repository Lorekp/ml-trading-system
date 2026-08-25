"""
Portfolio combo search — PARALLELIZED for fast CPU.

Loads cached signals, runs all 4095 combos with realistic shared-capital
simulation using multiprocessing.
"""
import numpy as np
import pandas as pd
import json, itertools, time, sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SIGNAL_CACHE = Path("C:/Claude/ml-trading-system/v3-results/signal_cache.csv")
OUTPUT_DIR = Path("C:/Claude/ml-trading-system/v3-results")

ALL_MONTHS = pd.period_range("2025-01", "2026-02", freq="M")
IS_MONTHS = ALL_MONTHS[:7]
OOS_MONTHS = ALL_MONTHS[7:]

STARTING_CAPITAL = 10000.0
MAX_DD = 0.15
DELEVER_START = 0.075
SLIPPAGE = 0.0001
MAX_EXPOSURE = 0.60
MAX_PER_ASSET = 0.20
SIZING_PCT = 0.02

CONFIG_IDS = ["15m_1b", "15m_2b", "15m_4b", "15m_8b",
              "30m_1b", "30m_2b", "30m_4b", "30m_6b", "30m_8b",
              "1H_1b", "1H_2b", "1H_4b"]


def simulate_combo(args):
    """Simulate one combo. Designed for multiprocessing (no shared state)."""
    combo, signals_np, ts_np, assets_np, configs_np, period_mask = args

    # Filter signals for this combo
    config_mask = np.isin(configs_np, list(combo))
    mask = config_mask & period_mask
    indices = np.where(mask)[0]

    if len(indices) < 20:
        return None

    # Sort by timestamp
    sort_order = np.argsort(ts_np[indices])
    idx_sorted = indices[sort_order]

    # Simulate with shared capital
    equity = STARTING_CAPITAL
    peak = equity
    halted = False

    n = len(idx_sorted)
    pnls = np.empty(n)
    equities = np.empty(n)
    dds = np.empty(n)
    active_count = 0

    asset_exp = {}
    total_exp = 0.0

    for j in range(n):
        i = idx_sorted[j]

        if halted:
            pnls[j] = 0
            equities[j] = equity
            dds[j] = (peak - equity) / peak if peak > 0 else 0
            continue

        dd = (peak - equity) / peak if peak > 0 else 0

        if dd >= MAX_DD:
            halted = True
            pnls[j] = 0
            equities[j] = equity
            dds[j] = dd
            continue

        # Deleveraging
        if dd > DELEVER_START:
            size_mult = max(1.0 - (dd - DELEVER_START) / (MAX_DD - DELEVER_START), 0.0)
        else:
            size_mult = 1.0

        raw_size = equity * SIZING_PCT * size_mult

        # Exposure limits
        asset = assets_np[i]
        ae = asset_exp.get(asset, 0.0) * 0.9  # Decay
        asset_exp[asset] = ae

        if ae + raw_size > equity * MAX_PER_ASSET:
            raw_size = max(equity * MAX_PER_ASSET - ae, 0)

        total_exp *= 0.9
        if total_exp + raw_size > equity * MAX_EXPOSURE:
            raw_size = max(equity * MAX_EXPOSURE - total_exp, 0)

        if raw_size < 1.0:
            pnls[j] = 0
            equities[j] = equity
            dds[j] = dd
            continue

        # Trade
        trade_ret = signals_np[i] - SLIPPAGE
        pnl = raw_size * trade_ret

        equity += pnl
        if equity > peak:
            peak = equity

        asset_exp[asset] = ae + raw_size
        total_exp += raw_size

        pnls[j] = pnl
        equities[j] = equity
        dds[j] = (peak - equity) / peak if peak > 0 else 0

    # Compute metrics
    active = pnls != 0
    if active.sum() < 20:
        return None

    active_pnls = pnls[active]
    active_ts = ts_np[idx_sorted[active]]

    # Daily aggregation
    ts_days = pd.to_datetime(active_ts).normalize()
    daily_df = pd.DataFrame({"pnl": active_pnls, "day": ts_days})
    daily = daily_df.groupby("day")["pnl"].sum()

    if len(daily) < 5:
        return None

    mean_d = daily.mean()
    std_d = daily.std()
    down = daily[daily < 0]
    down_std = down.std() if len(down) > 1 else std_d

    sharpe = mean_d / std_d * np.sqrt(365.25) if std_d > 0 else 0
    sortino = mean_d / down_std * np.sqrt(365.25) if down_std > 0 else 0
    max_dd = dds.max()
    total_ret = equity / STARTING_CAPITAL - 1
    days = max((pd.Timestamp(active_ts[-1]) - pd.Timestamp(active_ts[0])).total_seconds() / 86400, 1)
    years = days / 365.25
    cagr = (equity / STARTING_CAPITAL) ** (1/years) - 1 if years > 0 else 0
    calmar = cagr / max_dd if max_dd > 0 else 0

    wins = active_pnls[active_pnls > 0]
    losses = active_pnls[active_pnls < 0]
    wr = len(wins) / len(active_pnls)
    pf = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 99
    wr_d = (daily > 0).sum() / len(daily)

    monthly = daily.resample("ME").sum()
    mo_pos = (monthly > 0).sum()

    return {
        "label": "+".join(combo),
        "configs": list(combo),
        "n_configs": len(combo),
        "n_trades": int(active.sum()),
        "trades_per_day": round(active.sum() / days, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "total_return_pct": round(total_ret * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "max_dd_pct": round(max_dd * 100, 1),
        "win_rate_trades": round(wr, 3),
        "win_rate_days": round(wr_d, 3),
        "profit_factor": round(min(pf, 99), 2),
        "months_positive": int(mo_pos),
        "months_total": int(len(monthly)),
        "final_equity": round(equity, 0),
        "halted": halted,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("PORTFOLIO COMBO SEARCH — PARALLELIZED")
    print(f"Started: {datetime.now()}")
    print(f"Capital: ${STARTING_CAPITAL}, MaxDD: {MAX_DD*100:.0f}%, Sizing: {SIZING_PCT*100:.0f}%/trade")
    print(f"Searching ALL 4095 combos of {len(CONFIG_IDS)} configs")
    print("=" * 100)

    # Load signals
    print("\nLoading cached signals...")
    signals = pd.read_csv(SIGNAL_CACHE, parse_dates=["timestamp"])
    print(f"  {len(signals)} signals loaded")

    # Pre-compute numpy arrays for speed
    ts_np = signals["timestamp"].values.astype("datetime64[ns]")
    assets_np = signals["asset"].values
    configs_np = signals["config"].values
    directions = signals["direction"].values
    fwd_returns = signals["fwd_return"].values
    # Net return per trade (direction × fwd_return)
    signals_np = directions * fwd_returns

    # Period masks
    is_periods = set(IS_MONTHS)
    oos_periods = set(OOS_MONTHS)
    sig_months = pd.to_datetime(ts_np).to_period("M")
    is_mask = np.array([m in is_periods for m in sig_months])
    oos_mask = np.array([m in oos_periods for m in sig_months])

    # Generate all combos
    all_combos = []
    for n in range(1, len(CONFIG_IDS) + 1):
        all_combos.extend(itertools.combinations(CONFIG_IDS, n))
    print(f"  Total combos: {len(all_combos)}")

    # IS search with multiprocessing
    print(f"\nSearching IS period ({len(IS_MONTHS)} months)...")
    t0 = time.time()

    args_list = [(combo, signals_np, ts_np, assets_np, configs_np, is_mask)
                 for combo in all_combos]

    is_results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(simulate_combo, args): args[0] for args in args_list}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(all_combos)} combos done...", flush=True)
            result = future.result()
            if result:
                is_results.append(result)

    elapsed = time.time() - t0
    print(f"  Completed {len(all_combos)} combos in {elapsed:.1f}s ({len(is_results)} valid)")

    is_results.sort(key=lambda x: -x["sharpe"])

    # Top 30 IS
    print(f"\n{'='*120}")
    print(f"TOP 30 IN-SAMPLE (shared capital, realistic sim)")
    print(f"{'='*120}")
    print(f"{'#':>3} {'Combo':<60} {'N':>2} {'Sh':>5} {'So':>5} {'Ret':>7} {'DD':>5} {'PF':>4} "
          f"{'WR':>5} {'T/d':>4} {'Mo+':>4} {'H':>2}")
    print("-" * 120)
    for i, r in enumerate(is_results[:30]):
        h = "H" if r["halted"] else ""
        print(f"{i+1:>3} {r['label']:<60} {r['n_configs']:>2} {r['sharpe']:>5.2f} {r['sortino']:>5.2f} "
              f"{r['total_return_pct']:>+6.1f}% {r['max_dd_pct']:>4.1f}% {r['profit_factor']:>4.2f} "
              f"{r['win_rate_trades']:>5.3f} {r['trades_per_day']:>4.1f} "
              f"{r['months_positive']:>2}/{r['months_total']:<2} {h:>2}")

    # OOS validation of top 15
    print(f"\n{'='*120}")
    print(f"OOS VALIDATION (top 15 IS combos)")
    print(f"{'='*120}")
    print(f"{'#':>3} {'Combo':<60} {'IS_Sh':>5} {'OOS_Sh':>6} {'OOS_Ret':>8} {'OOS_DD':>6} "
          f"{'PF':>5} {'Mo+':>4} {'H':>2}")
    print("-" * 120)

    oos_args = [(is_r["configs"], signals_np, ts_np, assets_np, configs_np, oos_mask)
                for is_r in is_results[:15]]
    oos_results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures_oos = [pool.submit(simulate_combo, (tuple(a[0]),) + a[1:]) for a in oos_args]
        for i, future in enumerate(futures_oos):
            result = future.result()
            is_r = is_results[i]
            if result:
                oos_results.append(result)
                h = "H" if result["halted"] else ""
                print(f"{i+1:>3} {is_r['label']:<60} {is_r['sharpe']:>5.2f} {result['sharpe']:>6.2f} "
                      f"{result['total_return_pct']:>+7.1f}% {result['max_dd_pct']:>5.1f}% "
                      f"{result['profit_factor']:>5.2f} {result['months_positive']:>2}/{result['months_total']:<2} {h:>2}")

    # Sizing sweep on best IS combo
    best_configs = tuple(is_results[0]["configs"])
    print(f"\nSIZING SWEEP: {'+'.join(best_configs)}")
    print(f"{'Size':>6} {'Sharpe':>7} {'Return':>8} {'MaxDD':>6} {'PF':>5} {'Calmar':>7} {'H':>2}")
    print("-" * 50)
    for pct in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10]:
        old_sizing = SIZING_PCT
        # Temporarily change sizing — but simulate_combo uses global. Pass it differently.
        # Actually, our function uses the global. Let me just run it inline.
        sub = signals[signals["config"].isin(best_configs)].sort_values("timestamp")
        equity = STARTING_CAPITAL; peak = equity; halted = False
        asset_exp = {}; total_exp = 0.0
        pnl_list = []; ts_list = []
        max_dd_seen = 0

        for _, row in sub.iterrows():
            if halted: continue
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd >= MAX_DD: halted = True; continue
            max_dd_seen = max(max_dd_seen, dd)
            sm = max(1.0 - max(dd - DELEVER_START, 0) / (MAX_DD - DELEVER_START), 0) if dd > DELEVER_START else 1.0
            raw = equity * pct * sm
            a = row["asset"]
            ae = asset_exp.get(a, 0) * 0.9; asset_exp[a] = ae
            if ae + raw > equity * MAX_PER_ASSET: raw = max(equity * MAX_PER_ASSET - ae, 0)
            total_exp *= 0.9
            if total_exp + raw > equity * MAX_EXPOSURE: raw = max(equity * MAX_EXPOSURE - total_exp, 0)
            if raw < 1: continue
            ret = row["direction"] * row["fwd_return"] - SLIPPAGE
            pnl = raw * ret
            equity += pnl
            if equity > peak: peak = equity
            asset_exp[a] = ae + raw; total_exp += raw
            pnl_list.append(pnl); ts_list.append(row["timestamp"])

        if pnl_list:
            daily = pd.Series(pnl_list, index=pd.to_datetime(ts_list)).resample("1D").sum()
            dn = daily[daily != 0]
            sh = dn.mean() / dn.std() * np.sqrt(365.25) if len(dn) > 5 and dn.std() > 0 else 0
            tr = equity / STARTING_CAPITAL - 1
            mdd = max_dd_seen
            days = max((pd.Timestamp(ts_list[-1]) - pd.Timestamp(ts_list[0])).total_seconds()/86400, 1)
            cagr = (equity/STARTING_CAPITAL)**(1/(days/365.25))-1 if days > 0 else 0
            cal = cagr / mdd if mdd > 0 else 0
            g = sum(p for p in pnl_list if p > 0)
            l = abs(sum(p for p in pnl_list if p < 0))
            pf = g/l if l > 0 else 99
            h = "H" if halted else ""
            v = " <--" if mdd <= MAX_DD and sh > 0 and not halted else ""
            print(f"{pct*100:>5.1f}% {sh:>7.2f} {tr*100:>+7.1f}% {mdd*100:>5.1f}% {min(pf,99):>5.2f} {cal:>7.2f} {h:>2}{v}")

    # Stress tests
    print(f"\nSTRESS TESTS: {'+'.join(best_configs)}")
    scenarios = [
        ("base", signals),
    ]
    # Decay
    d = signals.copy(); d["fwd_return"] = d["fwd_return"] * 0.7
    scenarios.append(("decay_30%", d))
    # Extra cost
    e = signals.copy(); e["fwd_return"] = e["fwd_return"] - 0.0003 * e["direction"]
    scenarios.append(("extra_3bp", e))
    # 60% fills
    np.random.seed(42)
    p = signals[np.random.random(len(signals)) < 0.6].copy()
    scenarios.append(("60%_fills", p))

    print(f"{'Scenario':<15} {'Sharpe':>7} {'Return':>8} {'MaxDD':>6} {'PF':>5} {'H':>2}")
    print("-" * 45)
    for name, sigs in scenarios:
        r = simulate_combo((best_configs, (sigs["direction"] * sigs["fwd_return"]).values,
                           sigs["timestamp"].values.astype("datetime64[ns]"),
                           sigs["asset"].values, sigs["config"].values,
                           np.ones(len(sigs), dtype=bool)))
        if r:
            h = "H" if r["halted"] else ""
            print(f"{name:<15} {r['sharpe']:>7.2f} {r['total_return_pct']:>+7.1f}% "
                  f"{r['max_dd_pct']:>5.1f}% {r['profit_factor']:>5.2f} {h:>2}")

    # Save
    save = {
        "timestamp": datetime.now().isoformat(),
        "top_30_is": is_results[:30],
        "oos_top_15": oos_results[:15],
    }
    with open(OUTPUT_DIR / "portfolio_realistic_results.json", "w") as f:
        json.dump(save, f, indent=2, default=str)

    print(f"\nCompleted: {datetime.now()}")


if __name__ == "__main__":
    main()
