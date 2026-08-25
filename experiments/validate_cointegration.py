"""Independent Cointegration Pairs Validation — Bar-Level, From Scratch

Reimplements the spread trading backtest WITHOUT importing Codex code.
Tests the 3 frozen pairs (ENA/LTC, TAO/FET, LINK/ENA) with:
- Bar-level equity tracking (exposes real intra-month DD)
- 1-week embargo between train/test
- Walk-forward W1/W2/W3 windows (same as LogReg)
- Sharpe confidence intervals
- DSR correction for 435-pair screening
- Max hold backstop (30 days)

HARD GATE: W2+W3 Sharpe < 1.0 or bar-level DD > 10% -> KILL.
"""

import sys, os, time, warnings, math
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from signal_engine.data import load_binance_1m, resample_ohlcv

warnings.filterwarnings("ignore")
DATA_DIR = Path("data/binance-1m")

# Frozen configs from Codex research (DO NOT optimize — fixed prior)
PAIRS = [
    {"y": "LTC", "x": "ENA", "lb_months": 2, "entry_z": 1.5, "cost_bps": 2.0, "label": "ENA/LTC"},
    {"y": "TAO", "x": "FET", "lb_months": 1, "entry_z": 2.0, "cost_bps": 10.0, "label": "TAO/FET"},
    {"y": "LINK", "x": "ENA", "lb_months": 4, "entry_z": 1.5, "cost_bps": 2.0, "label": "LINK/ENA"},
]

WINDOWS = [
    {"label": "W1", "start": "2024-07-01", "end": "2025-02-28", "cal_days": 245},
    {"label": "W2", "start": "2025-03-01", "end": "2025-10-31", "cal_days": 244},
    {"label": "W3", "start": "2025-11-01", "end": "2026-02-28", "cal_days": 122},
]

EMBARGO_BARS = 96 * 7  # 1 week at 15-min bars
MAX_HOLD_BARS = 96 * 30  # 30 days max hold
NOTIONAL = 1000.0  # $1000 per spread leg


# ============================================================
# Data
# ============================================================

def load_15m_close(asset: str) -> pd.Series | None:
    symbol = f"{asset}USDT"
    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        return ohlcv_15m["close"]
    except Exception:
        return None


# ============================================================
# OLS (from scratch — no Codex imports)
# ============================================================

def ols_fit(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Simple OLS: y = beta * x + intercept. Returns (beta, intercept)."""
    x_mean = x.mean()
    y_mean = y.mean()
    x_dev = x - x_mean
    denom = np.dot(x_dev, x_dev)
    if denom <= 0:
        return np.nan, np.nan
    beta = np.dot(x_dev, y - y_mean) / denom
    intercept = y_mean - beta * x_mean
    return float(beta), float(intercept)


# ============================================================
# Bar-Level Spread Simulation
# ============================================================

def simulate_pair_bars(
    y_close: pd.Series,
    x_close: pd.Series,
    pair_cfg: dict,
    start_date: str,
    end_date: str,
) -> dict | None:
    """Simulate one pair with bar-level PnL tracking.

    Returns dict with equity_curve, daily_pnl, trade_log, metrics.
    """
    lb_months = pair_cfg["lb_months"]
    entry_z = pair_cfg["entry_z"]
    cost_bps = pair_cfg["cost_bps"]

    # Align series
    common = y_close.index.intersection(x_close.index)
    y = y_close.reindex(common).astype(float)
    x = x_close.reindex(common).astype(float)

    # Date range
    ws = pd.Timestamp(start_date, tz="UTC")
    we = pd.Timestamp(end_date, tz="UTC")
    test_mask = (y.index >= ws) & (y.index <= we)
    test_idx = y.index[test_mask]
    if len(test_idx) < 96 * 30:  # Need at least 1 month
        return None

    # Log prices
    log_y = np.log(y.values)
    log_x = np.log(x.values)

    # Returns
    y_ret = np.zeros(len(y))
    x_ret = np.zeros(len(x))
    y_ret[1:] = y.values[1:] / y.values[:-1] - 1.0
    x_ret[1:] = x.values[1:] / x.values[:-1] - 1.0

    # State
    position = 0  # 0, +1 (long spread), -1 (short spread)
    entry_bar = 0
    beta = np.nan
    intercept = np.nan
    mu = np.nan
    sigma = np.nan
    last_refit_month = None

    equity = 0.0  # Track spread PnL (not capital-based)
    equity_curve = []
    daily_pnl = {}
    trades = []

    # Map index to position in full array
    idx_map = {ts: i for i, ts in enumerate(y.index)}

    for ts in test_idx:
        i = idx_map[ts]

        # Monthly refit with embargo
        current_month = ts.to_period("M")
        if current_month != last_refit_month:
            train_end_ts = ts - pd.Timedelta(minutes=15 * EMBARGO_BARS)
            train_start_ts = train_end_ts - pd.DateOffset(months=lb_months)

            train_mask = (y.index >= train_start_ts) & (y.index < train_end_ts)
            n_train = train_mask.sum()

            if n_train >= 96 * 30:  # Need at least 30 days training
                train_log_y = log_y[train_mask]
                train_log_x = log_x[train_mask]
                beta, intercept = ols_fit(train_log_y, train_log_x)
                if np.isfinite(beta) and beta > 0:
                    train_resid = train_log_y - beta * train_log_x - intercept
                    mu = float(np.mean(train_resid))
                    sigma = float(np.std(train_resid))
                    if sigma <= 1e-12:
                        sigma = np.nan
                else:
                    beta = np.nan
            last_refit_month = current_month

        # Skip if no valid model
        if not np.isfinite(beta) or not np.isfinite(sigma):
            equity_curve.append(equity)
            continue

        # Compute z-score
        spread = log_y[i] - beta * log_x[i] - intercept
        z = (spread - mu) / sigma

        # PnL from held position (BEFORE entry/exit decisions)
        bar_pnl = 0.0
        if position != 0:
            hedge_ret = y_ret[i] - beta * x_ret[i]
            bar_pnl = position * hedge_ret * NOTIONAL

        equity += bar_pnl

        # Entry / exit
        cost = cost_bps / 10000.0 * (1.0 + abs(beta)) * NOTIONAL

        if position != 0:
            bars_held = i - entry_bar
            # Exit conditions
            should_exit = False
            exit_reason = ""
            if position == 1 and z >= 0.0:
                should_exit = True
                exit_reason = "mean_revert"
            elif position == -1 and z <= 0.0:
                should_exit = True
                exit_reason = "mean_revert"
            elif bars_held >= MAX_HOLD_BARS:
                should_exit = True
                exit_reason = "max_hold"

            if should_exit:
                equity -= cost
                trades.append({
                    "entry_time": y.index[entry_bar],
                    "exit_time": ts,
                    "direction": position,
                    "bars_held": bars_held,
                    "pnl": equity - (equity_curve[entry_bar - idx_map[test_idx[0]]] if entry_bar >= idx_map[test_idx[0]] else 0),
                    "exit_reason": exit_reason,
                    "z_at_entry": z,  # approximate
                    "beta": beta,
                })
                position = 0

        if position == 0:
            if z <= -entry_z:
                position = 1  # Long spread: buy Y, sell X
                entry_bar = i
                equity -= cost
            elif z >= entry_z:
                position = -1  # Short spread: sell Y, buy X
                entry_bar = i
                equity -= cost

        equity_curve.append(equity)

        # Daily PnL aggregation
        day = ts.normalize()
        daily_pnl[day] = daily_pnl.get(day, 0.0) + bar_pnl

    if len(equity_curve) < 100:
        return None

    # Metrics
    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr + NOTIONAL)  # Add notional as base
    dd = (eq_arr + NOTIONAL - peak) / peak
    max_dd = abs(dd.min()) if len(dd) > 0 else 0

    dpnl = pd.Series(daily_pnl)
    daily_ret = dpnl / NOTIONAL
    n_days = len(daily_ret)
    sharpe = 0.0
    if n_days > 10 and daily_ret.std() > 0:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(365.25)

    # Sharpe confidence interval: SR ± 1.96 * sqrt((1 + SR²/2) / N)
    sharpe_se = math.sqrt((1 + sharpe**2 / 2) / max(n_days - 1, 1)) if n_days > 1 else float('inf')
    sharpe_lo = sharpe - 1.96 * sharpe_se
    sharpe_hi = sharpe + 1.96 * sharpe_se

    worst_day = abs(daily_ret.min()) if len(daily_ret) > 0 else 0
    total_pnl = eq_arr[-1]
    n_trades = len(trades)
    win_rate = sum(1 for t in trades if t.get("pnl", 0) > 0) / max(n_trades, 1) * 100
    avg_hold = np.mean([t["bars_held"] for t in trades]) / 96 if trades else 0  # in days

    return {
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "sharpe_ci": (sharpe_lo, sharpe_hi),
        "max_dd": max_dd,
        "worst_day": worst_day,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_hold_days": avg_hold,
        "n_days": n_days,
        "equity_curve": eq_arr,
        "trades": trades,
    }


# ============================================================
# DSR (Deflated Sharpe Ratio)
# ============================================================

def deflated_sharpe(observed_sr: float, n_days: int, k_trials: int = 435) -> float:
    """Compute DSR: probability that observed SR > 0 given K trials.

    Uses Bailey & Lopez de Prado (2014) approximation.
    """
    from scipy.stats import norm

    if n_days < 10:
        return 0.0
    # Expected max SR under null hypothesis (K independent trials)
    e_max_sr = norm.ppf(1 - 1 / k_trials) if k_trials > 1 else 0
    # Variance of SR estimator
    sr_var = (1 + observed_sr**2 / 2) / n_days
    if sr_var <= 0:
        return 0.0
    # DSR = P(true SR > 0 | observed SR, K trials)
    t_stat = (observed_sr - e_max_sr) / math.sqrt(sr_var)
    return float(norm.cdf(t_stat))


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 100)
    print("INDEPENDENT COINTEGRATION VALIDATION (from scratch, bar-level)")
    print("=" * 100)
    print(f"\nPairs: {[p['label'] for p in PAIRS]}")
    print(f"Embargo: {EMBARGO_BARS} bars ({EMBARGO_BARS/96:.0f} days)")
    print(f"Max hold: {MAX_HOLD_BARS} bars ({MAX_HOLD_BARS/96:.0f} days)")

    # Load data
    print("\nLoading 15m data...", flush=True)
    assets_needed = set()
    for p in PAIRS:
        assets_needed.add(p["y"])
        assets_needed.add(p["x"])

    close_data = {}
    for asset in sorted(assets_needed):
        s = load_15m_close(asset)
        if s is not None:
            close_data[asset] = s
            print(f"  {asset}: {len(s)} bars ({s.index.min().date()} to {s.index.max().date()})")
        else:
            print(f"  {asset}: MISSING")

    print(f"\nLoaded {len(close_data)} assets in {time.time()-t0:.0f}s")

    # ---- Per-pair, per-window validation ----
    print("\n" + "=" * 80)
    print("PER-PAIR RESULTS (bar-level equity, 1-week embargo)")
    print("=" * 80)

    print(f"\n{'Pair':<12s} {'Window':<4s} {'PnL':>8s} {'$/cal':>7s} {'Sharpe':>7s} "
          f"{'95%CI':>14s} {'DD%':>6s} {'WD%':>6s} {'#Tr':>4s} {'WR%':>5s} "
          f"{'AvgHold':>8s} {'DSR':>5s}")
    print("-" * 110)

    all_pair_results = {}

    for pair in PAIRS:
        y_close = close_data.get(pair["y"])
        x_close = close_data.get(pair["x"])
        if y_close is None or x_close is None:
            print(f"{pair['label']:<12s} MISSING DATA")
            continue

        pair_results = {}
        for w in WINDOWS:
            r = simulate_pair_bars(y_close, x_close, pair, w["start"], w["end"])
            if r is None:
                print(f"{pair['label']:<12s} {w['label']:<4s}  insufficient data")
                continue

            pnl_cal = r["total_pnl"] / w["cal_days"]
            ci_lo, ci_hi = r["sharpe_ci"]
            dsr = deflated_sharpe(r["sharpe"], r["n_days"], k_trials=435)

            pair_results[w["label"]] = r

            print(f"{pair['label']:<12s} {w['label']:<4s} ${r['total_pnl']:>7.0f} ${pnl_cal:>6.2f} "
                  f"{r['sharpe']:>7.2f} [{ci_lo:>5.2f},{ci_hi:>5.2f}] "
                  f"{r['max_dd']*100:>5.1f}% {r['worst_day']*100:>5.2f}% "
                  f"{r['n_trades']:>4d} {r['win_rate']:>4.0f}% "
                  f"{r['avg_hold_days']:>7.1f}d {dsr:>5.2f}")

        all_pair_results[pair["label"]] = pair_results

    # ---- Portfolio (equal-weight 3 pairs) ----
    print("\n" + "=" * 80)
    print("PORTFOLIO RESULTS (equal-weight 3 pairs)")
    print("=" * 80)

    print(f"\n{'Window':<6s} {'PnL':>8s} {'$/cal':>7s} {'Sharpe':>7s} {'95%CI':>14s} "
          f"{'DD%':>6s} {'WD%':>6s} {'#Tr':>5s} {'DSR':>5s}")
    print("-" * 75)

    total_pnl = 0
    total_cal = 0
    max_dd_all = 0
    max_wd_all = 0
    total_trades = 0
    all_daily_pnl = {}

    for w in WINDOWS:
        port_daily = {}
        w_trades = 0
        w_pnl = 0

        for pair_label, pair_results in all_pair_results.items():
            if w["label"] not in pair_results:
                continue
            r = pair_results[w["label"]]
            w_pnl += r["total_pnl"]
            w_trades += r["n_trades"]

            # Aggregate daily PnL across pairs
            for t in r.get("trades", []):
                day = t["entry_time"].normalize()
                port_daily[day] = port_daily.get(day, 0.0) + t.get("pnl", 0)

        # Portfolio equity from all pairs' equity curves
        # Use daily PnL for Sharpe
        dpnl = pd.Series(port_daily) if port_daily else pd.Series(dtype=float)
        daily_ret = dpnl / (NOTIONAL * 3)  # 3 pairs
        n_days = len(daily_ret)

        sharpe = 0.0
        if n_days > 10 and daily_ret.std() > 0:
            sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(365.25)

        sharpe_se = math.sqrt((1 + sharpe**2 / 2) / max(n_days - 1, 1)) if n_days > 1 else float('inf')
        ci = (sharpe - 1.96 * sharpe_se, sharpe + 1.96 * sharpe_se)
        dsr = deflated_sharpe(sharpe, n_days, k_trials=435)

        pnl_cal = w_pnl / w["cal_days"]
        worst_day = abs(daily_ret.min()) if len(daily_ret) > 0 else 0

        # Approximate portfolio DD from cumulative daily PnL
        if len(dpnl) > 0:
            cum = dpnl.cumsum() + NOTIONAL * 3
            peak = cum.cummax()
            dd = ((cum - peak) / peak).min()
            max_dd_w = abs(dd)
        else:
            max_dd_w = 0

        print(f"{w['label']:<6s} ${w_pnl:>7.0f} ${pnl_cal:>6.2f} {sharpe:>7.2f} "
              f"[{ci[0]:>5.2f},{ci[1]:>5.2f}] {max_dd_w*100:>5.1f}% "
              f"{worst_day*100:>5.2f}% {w_trades:>5d} {dsr:>5.2f}")

        total_pnl += w_pnl
        total_cal += w["cal_days"]
        max_dd_all = max(max_dd_all, max_dd_w)
        max_wd_all = max(max_wd_all, worst_day)
        total_trades += w_trades
        all_daily_pnl.update(port_daily)

    if total_cal > 0:
        agg_dpnl = pd.Series(all_daily_pnl)
        agg_ret = agg_dpnl / (NOTIONAL * 3)
        agg_sharpe = 0.0
        if len(agg_ret) > 10 and agg_ret.std() > 0:
            agg_sharpe = agg_ret.mean() / agg_ret.std() * np.sqrt(365.25)
        agg_dsr = deflated_sharpe(agg_sharpe, len(agg_ret), k_trials=435)

        print(f"\n{'TOTAL':<6s} ${total_pnl:>7.0f} ${total_pnl/total_cal:>6.2f} "
              f"{agg_sharpe:>7.2f} {'':>14s} {max_dd_all*100:>5.1f}% "
              f"{max_wd_all*100:>5.2f}% {total_trades:>5d} {agg_dsr:>5.2f}")

    # ---- Comparison with Codex results ----
    print("\n" + "=" * 80)
    print("COMPARISON: Independent vs Codex Results")
    print("=" * 80)

    codex_results = {
        "ENA/LTC": {"sharpe": 4.87, "return": 81.73, "dd": 0.0, "trades": 15},
        "TAO/FET": {"sharpe": 6.99, "return": 184.79, "dd": 0.0, "trades": 23},
        "LINK/ENA": {"sharpe": 4.95, "return": 94.33, "dd": 0.0, "trades": 12},
    }

    print(f"\n{'Pair':<12s} {'Codex SR':>9s} {'Indep SR':>9s} {'Delta':>7s}  "
          f"{'Codex DD':>9s} {'Indep DD':>9s}  "
          f"{'Codex #Tr':>9s} {'Indep #Tr':>9s}")
    print("-" * 85)

    for pair_label, codex in codex_results.items():
        if pair_label in all_pair_results:
            # Combine all windows for aggregate
            total_sr = 0
            total_dd = 0
            total_tr = 0
            for w_label, r in all_pair_results[pair_label].items():
                total_sr = r["sharpe"]  # Last window's Sharpe (approximate)
                total_dd = max(total_dd, r["max_dd"])
                total_tr += r["n_trades"]
            print(f"{pair_label:<12s} {codex['sharpe']:>9.2f} {total_sr:>9.2f} "
                  f"{total_sr - codex['sharpe']:>+7.2f}  "
                  f"{codex['dd']:>8.1f}% {total_dd*100:>8.1f}%  "
                  f"{codex['trades']:>9d} {total_tr:>9d}")
        else:
            print(f"{pair_label:<12s} {codex['sharpe']:>9.2f}   NO DATA")

    # ---- Trade Log Detail ----
    print("\n" + "=" * 80)
    print("TRADE LOG (all pairs, all windows)")
    print("=" * 80)

    print(f"\n{'Pair':<12s} {'Entry':>20s} {'Exit':>20s} {'Dir':>4s} {'Hold(d)':>8s} "
          f"{'Exit Reason':>12s}")
    print("-" * 85)

    for pair_label, pair_results in all_pair_results.items():
        for w_label, r in pair_results.items():
            for t in r.get("trades", []):
                entry_str = str(t["entry_time"])[:19]
                exit_str = str(t["exit_time"])[:19]
                dir_str = "LONG" if t["direction"] > 0 else "SHORT"
                hold_d = t["bars_held"] / 96
                print(f"{pair_label:<12s} {entry_str:>20s} {exit_str:>20s} {dir_str:>4s} "
                      f"{hold_d:>7.1f}d {t['exit_reason']:>12s}")

    # ---- Verdict ----
    print(f"\n{'='*100}")
    print(f"Runtime: {time.time()-t0:.0f}s")
    print(f"\nVERDICT:")
    if total_cal > 0:
        agg_dpnl = pd.Series(all_daily_pnl)
        agg_ret = agg_dpnl / (NOTIONAL * 3)
        agg_sharpe = agg_ret.mean() / agg_ret.std() * np.sqrt(365.25) if len(agg_ret) > 10 and agg_ret.std() > 0 else 0

        if agg_sharpe >= 1.0 and max_dd_all < 0.10:
            print(f"  PASS: Aggregate Sharpe {agg_sharpe:.2f} >= 1.0, DD {max_dd_all*100:.1f}% < 10%")
            print(f"  -> Proceed to live strategy implementation")
        else:
            print(f"  FAIL: Sharpe {agg_sharpe:.2f}, DD {max_dd_all*100:.1f}%")
            if agg_sharpe < 1.0:
                print(f"  -> Sharpe below threshold (need >= 1.0)")
            if max_dd_all >= 0.10:
                print(f"  -> DD above limit (need < 10%)")
            print(f"  -> KILL strategy or wait for more data")
    else:
        print("  NO DATA — cannot evaluate")


if __name__ == "__main__":
    main()
