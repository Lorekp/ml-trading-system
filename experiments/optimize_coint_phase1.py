"""Cointegration Phase 1 Optimization — Quick Wins

Tests 4 optimizations INDEPENDENTLY against the validated base strategy:
1A. CUSUM event filter (only enter on structural break events)
1B. Half-life adaptive exit (fast spreads exit earlier)
1C. Dynamic volatility thresholds (adapt entry z to vol regime)
1D. Z-score proportional sizing (bigger when more extreme)

Each is tested in isolation. Only improvements with positive OOS delta are kept.
Base: Sharpe 1.58, 89 trades, $3.25/cal, DD 4.0%, WD 2.23% (5% sizing, 3 pairs)
"""

import sys, os, time, warnings, math
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from signal_engine.data import load_binance_1m, resample_ohlcv
from afml.filters import cusum_filter, get_daily_vol

warnings.filterwarnings("ignore")
DATA_DIR = Path("data/binance-1m")

PAIRS = [
    {"y": "LTC", "x": "ENA", "lb_months": 2, "entry_z": 1.5, "cost_bps": 2.0, "label": "ENA/LTC"},
    {"y": "TAO", "x": "FET", "lb_months": 1, "entry_z": 2.0, "cost_bps": 10.0, "label": "TAO/FET"},
    {"y": "LINK", "x": "ENA", "lb_months": 4, "entry_z": 1.5, "cost_bps": 2.0, "label": "LINK/ENA"},
]

EMBARGO = 96 * 7
MAX_HOLD = 96 * 30
CAPITAL = 10000.0
BASE_SIZE_PCT = 0.05
TEST_START = "2024-07-01"
TEST_END = "2026-02-28"
CAL_DAYS = 611


def load_15m_close(asset):
    o = load_binance_1m(f"{asset}USDT", [DATA_DIR])
    return resample_ohlcv(o, "15min")["close"]


def ols_fit(y, x):
    xm, ym = x.mean(), y.mean()
    xd = x - xm
    d = np.dot(xd, xd)
    if d <= 0: return np.nan, np.nan
    beta = np.dot(xd, y - ym) / d
    return float(beta), float(ym - beta * xm)


def half_life_from_resid(resid):
    """AR(1) half-life in bars."""
    if len(resid) < 50: return float("inf")
    lag = resid[:-1]
    delta = resid[1:] - resid[:-1]
    d = np.dot(lag - lag.mean(), lag - lag.mean())
    if d <= 0: return float("inf")
    phi = np.dot(lag - lag.mean(), delta - delta.mean()) / d
    if phi >= 0: return float("inf")
    return -math.log(2) / phi  # in bars


def simulate_pair(
    y_close, x_close, pair_cfg,
    # Optimization flags
    use_cusum=False, cusum_h=1.5, cusum_lookback=96,
    use_halflife_exit=False,
    use_dynamic_vol=False, vol_scale=0.2,
    use_zscore_sizing=False,
    size_pct=BASE_SIZE_PCT,
):
    """Simulate one pair with optional optimizations. Returns metrics dict."""
    lb_months = pair_cfg["lb_months"]
    entry_z = pair_cfg["entry_z"]
    cost_bps = pair_cfg["cost_bps"]

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
    if len(test_idx) < 96 * 30: return None

    idx_map = {ts: i for i, ts in enumerate(y.index)}

    # Pre-compute CUSUM events on spread if needed
    cusum_events = set()
    if use_cusum:
        # We'll compute CUSUM per-month after refit
        pass  # Done inline below

    pos = 0; entry_bar = 0
    beta = np.nan; intercept = np.nan; mu = np.nan; sigma = np.nan
    hl_bars = float("inf")
    last_month = None
    spread_vol_mean = np.nan; spread_vol_std = np.nan

    equity = 0.0; equity_curve = []
    daily_pnl = {}; n_trades = 0; n_wins = 0

    # Rolling spread for CUSUM and vol
    spread_history = []

    for ts in test_idx:
        i = idx_map[ts]
        cm = ts.to_period("M")

        # Monthly refit with embargo
        if cm != last_month:
            te = ts - pd.Timedelta(minutes=15 * EMBARGO)
            t_start = te - pd.DateOffset(months=lb_months)
            tm = (y.index >= t_start) & (y.index < te)
            if tm.sum() >= 96 * 30:
                ty = log_y[tm]; tx = log_x[tm]
                beta, intercept = ols_fit(ty, tx)
                if np.isfinite(beta) and beta > 0:
                    tr = ty - beta * tx - intercept
                    mu = float(np.mean(tr)); sigma = float(np.std(tr))
                    if sigma <= 1e-12: sigma = np.nan

                    # Half-life
                    hl_bars = half_life_from_resid(tr)

                    # CUSUM: compute events on training spread returns
                    if use_cusum and np.isfinite(sigma):
                        train_spread = pd.Series(tr, index=y.index[tm])
                        vol = get_daily_vol(pd.Series(np.exp(tr), index=y.index[tm]), span=50)
                        h = vol.mean() * cusum_h if len(vol.dropna()) > 10 else sigma * cusum_h
                        # Also compute CUSUM on recent test spread
                        recent_start = te - pd.Timedelta(days=30)
                        recent_mask = (y.index >= recent_start) & (y.index < ts)
                        if recent_mask.sum() > 50:
                            recent_spread = pd.Series(
                                log_y[recent_mask] - beta * log_x[recent_mask] - intercept,
                                index=y.index[recent_mask]
                            )
                            cusum_events = set(cusum_filter(recent_spread, threshold=sigma * cusum_h))

                    # Dynamic vol: compute spread vol stats from training
                    if use_dynamic_vol:
                        train_spread_ret = np.diff(tr)
                        if len(train_spread_ret) > 50:
                            spread_vol_mean = float(np.std(train_spread_ret))
                            # Rolling vol on training to get distribution
                            _sv = pd.Series(train_spread_ret).rolling(96).std().dropna()
                            if len(_sv) > 20:
                                spread_vol_std = float(_sv.std())

                else:
                    beta = np.nan
            last_month = cm

        if not np.isfinite(beta) or not np.isfinite(sigma):
            equity_curve.append(equity)
            continue

        # Current spread and z-score
        spread = log_y[i] - beta * log_x[i] - intercept
        z = (spread - mu) / sigma
        spread_history.append(spread)

        # Dynamic entry threshold
        eff_entry_z = entry_z
        if use_dynamic_vol and len(spread_history) > 96:
            recent_vol = float(np.std(np.diff(spread_history[-96:])))
            if np.isfinite(spread_vol_mean) and spread_vol_mean > 0 and np.isfinite(spread_vol_std) and spread_vol_std > 0:
                vol_z = (recent_vol - spread_vol_mean) / spread_vol_std
                eff_entry_z = entry_z * (1 + vol_z * vol_scale)
                eff_entry_z = max(1.0, min(eff_entry_z, 4.0))  # clamp

        # Dynamic exit threshold
        exit_z = 0.0
        max_hold = MAX_HOLD
        if use_halflife_exit and np.isfinite(hl_bars) and hl_bars > 0:
            hl_days = hl_bars / 96
            exit_z = 0.3 * (1 - min(hl_days / 5, 1.0))
            max_hold = max(int(hl_bars * 2), 96)

        # Position sizing
        notional = CAPITAL * size_pct
        if use_zscore_sizing and pos == 0:
            z_mult = min(abs(z) / entry_z, 2.0) if entry_z > 0 else 1.0
            notional = CAPITAL * size_pct * z_mult

        # PnL
        bar_pnl = 0.0
        if pos != 0:
            hr = y_ret[i] - beta * x_ret[i]
            bar_pnl = pos * hr * notional

        equity += bar_pnl
        cost = cost_bps / 10000 * (1 + abs(beta)) * notional

        # Exit
        if pos != 0:
            bh = i - entry_bar
            should_exit = False
            if pos == 1 and z >= exit_z: should_exit = True
            elif pos == -1 and z <= -exit_z: should_exit = True
            elif bh >= max_hold: should_exit = True
            if should_exit:
                equity -= cost; pos = 0
                if bar_pnl > 0: n_wins += 1

        # Entry
        if pos == 0:
            # CUSUM gate: require recent event
            cusum_ok = True
            if use_cusum:
                # Check if any CUSUM event in last cusum_lookback bars
                cusum_ok = False
                for ev_ts in cusum_events:
                    if hasattr(ev_ts, 'tz') and ev_ts.tz is None:
                        pass
                    if abs((ts - ev_ts).total_seconds()) < cusum_lookback * 15 * 60:
                        cusum_ok = True
                        break

            if cusum_ok:
                if z <= -eff_entry_z:
                    pos = 1; entry_bar = i; equity -= cost; n_trades += 1
                elif z >= eff_entry_z:
                    pos = -1; entry_bar = i; equity -= cost; n_trades += 1

        equity_curve.append(equity)
        day = ts.normalize()
        daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl

    if len(equity_curve) < 100 or n_trades < 3: return None

    dpnl = pd.Series(daily_pnl)
    dr = dpnl / CAPITAL
    sharpe = dr.mean() / dr.std() * np.sqrt(365.25) if len(dr) > 10 and dr.std() > 0 else 0
    cum = dpnl.cumsum()
    peak = (cum + CAPITAL).cummax()
    dd = ((cum + CAPITAL - peak) / peak).min()
    worst_day = abs(dr.min()) if len(dr) > 0 else 0
    pnl_cal = cum.iloc[-1] / CAL_DAYS
    wr = n_wins / max(n_trades, 1) * 100

    return {
        "pnl_cal": pnl_cal, "sharpe": sharpe, "max_dd": abs(dd),
        "worst_day": worst_day, "n_trades": n_trades, "win_rate": wr,
        "total_pnl": cum.iloc[-1],
    }


def run_portfolio(close_data, config_label, **kwargs):
    """Run all 3 pairs with given optimization kwargs, return portfolio metrics."""
    port_daily = {}
    total_trades = 0; total_wins = 0

    for pair in PAIRS:
        y_s = close_data[pair["y"]]; x_s = close_data[pair["x"]]
        r = simulate_pair(y_s, x_s, pair, **kwargs)
        if r is None: continue
        total_trades += r["n_trades"]

    # Re-run with combined PnL tracking
    total_pnl = 0
    port_daily = {}
    all_trades = 0

    for pair in PAIRS:
        y_s = close_data[pair["y"]]; x_s = close_data[pair["x"]]
        common = y_s.index.intersection(x_s.index)
        y = y_s.reindex(common).astype(float); x = x_s.reindex(common).astype(float)
        r = simulate_pair(y_s, x_s, pair, **kwargs)
        if r: total_pnl += r["total_pnl"]; all_trades += r["n_trades"]

    # Simplified portfolio: sum individual pair PnLs
    # Re-simulate to get combined daily
    combined = {}
    for pair in PAIRS:
        r = simulate_pair(close_data[pair["y"]], close_data[pair["x"]], pair, **kwargs)
        if r is None: continue
        # We need daily PnL per pair — re-simulate just to get it
        # Actually, let's just report the summed metrics
        pass

    # Quick aggregate: just sum per-pair stats
    results = []
    for pair in PAIRS:
        r = simulate_pair(close_data[pair["y"]], close_data[pair["x"]], pair, **kwargs)
        if r: results.append(r)

    if not results: return None

    total_pnl = sum(r["total_pnl"] for r in results)
    total_trades = sum(r["n_trades"] for r in results)
    avg_sharpe = np.mean([r["sharpe"] for r in results])
    max_dd = max(r["max_dd"] for r in results)
    max_wd = max(r["worst_day"] for r in results)
    pnl_cal = total_pnl / CAL_DAYS

    return {
        "label": config_label,
        "pnl_cal": pnl_cal, "sharpe": avg_sharpe, "max_dd": max_dd,
        "worst_day": max_wd, "n_trades": total_trades,
        "total_pnl": total_pnl,
    }


def main():
    t0 = time.time()
    print("=" * 100)
    print("COINTEGRATION PHASE 1 OPTIMIZATION — QUICK WINS")
    print("=" * 100)

    # Load data
    print("\nLoading data...", end=" ", flush=True)
    close_data = {}
    for a in ["LTC", "ENA", "TAO", "FET", "LINK"]:
        close_data[a] = load_15m_close(a)
    print(f"done in {time.time()-t0:.0f}s")

    configs = [
        # (label, kwargs)
        ("BASE (control)", {}),
        # 1A: CUSUM filter
        ("1A: CUSUM h=1.0", {"use_cusum": True, "cusum_h": 1.0}),
        ("1A: CUSUM h=1.5", {"use_cusum": True, "cusum_h": 1.5}),
        ("1A: CUSUM h=2.0", {"use_cusum": True, "cusum_h": 2.0}),
        # 1B: Half-life exit
        ("1B: HL-exit", {"use_halflife_exit": True}),
        # 1C: Dynamic vol thresholds
        ("1C: DynVol s=0.1", {"use_dynamic_vol": True, "vol_scale": 0.1}),
        ("1C: DynVol s=0.2", {"use_dynamic_vol": True, "vol_scale": 0.2}),
        ("1C: DynVol s=0.3", {"use_dynamic_vol": True, "vol_scale": 0.3}),
        # 1D: Z-score proportional sizing
        ("1D: ZscoreSizing", {"use_zscore_sizing": True}),
        # Combinations of winners
        ("1B+1D: HL+ZSize", {"use_halflife_exit": True, "use_zscore_sizing": True}),
        ("1B+1C: HL+DynVol", {"use_halflife_exit": True, "use_dynamic_vol": True, "vol_scale": 0.2}),
        ("1A+1B+1D", {"use_cusum": True, "cusum_h": 1.5, "use_halflife_exit": True, "use_zscore_sizing": True}),
        ("ALL", {"use_cusum": True, "cusum_h": 1.5, "use_halflife_exit": True, "use_dynamic_vol": True, "vol_scale": 0.2, "use_zscore_sizing": True}),
    ]

    print(f"\n{'Config':<25s} {'$/cal':>7s} {'Sharpe':>7s} {'DD%':>6s} {'WD%':>6s} {'#Tr':>5s} {'Delta SR':>9s}")
    print("-" * 75)

    base_sharpe = None
    for label, kwargs in configs:
        r = run_portfolio(close_data, label, **kwargs)
        if r is None:
            print(f"{label:<25s} NO DATA")
            continue

        delta = ""
        if base_sharpe is not None:
            d = r["sharpe"] - base_sharpe
            delta = f"{d:>+8.2f}"
        else:
            base_sharpe = r["sharpe"]
            delta = "    base"

        ok = r["max_dd"] < 0.10 and r["worst_day"] < 0.05
        print(f"{label:<25s} ${r['pnl_cal']:>6.2f} {r['sharpe']:>7.2f} "
              f"{r['max_dd']*100:>5.1f}% {r['worst_day']*100:>5.2f}% "
              f"{r['n_trades']:>5d} {delta}  {'OK' if ok else 'FAIL'}")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
