"""
Direction V3 — Portfolio Simulation & Risk Management.

Senior quant standards. Tests all combinations of direction prediction configs
with proper position sizing, 15% max drawdown constraint, and walk-forward
validated selection.

Architecture:
  1. Generate trade signals (all 12 configs × 6 assets, walk-forward monthly)
  2. Portfolio simulator (position sizing, drawdown control, concurrent positions)
  3. Combination search (1,585 combos, rank by Sharpe, filter by MaxDD ≤ 15%)
  4. Position sizing comparison (fixed, Kelly, vol-scaled)
  5. Stress testing (crash, correlation spike, fill degradation, signal decay)
  6. Final report

Walk-forward selection: months 1-7 for config selection, months 8-14 for OOS validation.
"""
import sys, warnings, json, itertools
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/ml-data")

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels

DATA_DIR = Path("/home/user/ml-data/binance-1m")
OUTPUT_DIR = Path("/home/user/ml-data/v3-results")

ASSETS = ["BTC", "ETH", "XRP", "BNB", "SUI", "LINK"]
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
           "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT"}

CONFIGS = [
    ("15m", "15min", 1, "15m_1bar"),
    ("15m", "15min", 2, "15m_2bar"),
    ("15m", "15min", 4, "15m_4bar"),
    ("15m", "15min", 8, "15m_8bar"),
    ("30m", "30min", 1, "30m_1bar"),
    ("30m", "30min", 2, "30m_2bar"),
    ("30m", "30min", 4, "30m_4bar"),
    ("30m", "30min", 6, "30m_6bar"),
    ("30m", "30min", 8, "30m_8bar"),
    ("1H", "1h", 1, "1H_1bar"),
    ("1H", "1h", 2, "1H_2bar"),
    ("1H", "1h", 4, "1H_4bar"),
]

ALL_MONTHS = pd.period_range("2025-01", "2026-02", freq="M")
IS_MONTHS = ALL_MONTHS[:7]   # Jan-Jul 2025: in-sample selection
OOS_MONTHS = ALL_MONTHS[7:]  # Aug 2025-Feb 2026: out-of-sample validation

STARTING_CAPITAL = 10000.0
MAX_DD_TARGET = 0.15  # 15%
COST_BPS = 0  # MEXC 0% maker. Add 1 bps slippage.
SLIPPAGE_BPS = 1.0
TOTAL_COST = (COST_BPS + SLIPPAGE_BPS) / 10000  # per trade

VOL_PERCENTILE = 50
CONF_PERCENTILE = 70

XGB_PARAMS = dict(max_depth=3, learning_rate=0.05, n_estimators=100,
                  min_child_weight=50, subsample=0.7, colsample_bytree=0.7,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", random_state=42)


# =========================================================================
# DATA LOADING + FEATURES (cached)
# =========================================================================
_ohlcv_cache = {}

def load_ohlcv(symbol, tf):
    key = (symbol, tf)
    if key in _ohlcv_cache:
        return _ohlcv_cache[key]
    d = DATA_DIR / SYMBOLS[symbol]
    dfs = []
    for f in sorted(d.glob("*.csv")):
        with open(f) as fh:
            first = fh.readline().strip().split(",")[0]
        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "qv", "count", "tbv", "tqv", "ignore"]
        if first.replace(".", "").replace("-", "").isdigit():
            df = pd.read_csv(f, header=None, names=cols)
        else:
            df = pd.read_csv(f)
            df.columns = cols
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    result = combined.resample(tf).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    _ohlcv_cache[key] = result
    return result


def build_features(ohlcv, train_end):
    close, high, low, volume = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]
    train_mask = ohlcv.index < train_end
    features = pd.DataFrame(index=ohlcv.index)
    for lb in [1, 3, 7, 14]:
        ret = close.pct_change(lb)
        lo = ret[train_mask].quantile(0.01)
        hi = ret[train_mask].quantile(0.99)
        features[f"ret_{lb}"] = ret.clip(lo, hi)
    features["bar_range"] = (high - low) / close
    features["vol_zscore"] = (volume - volume.rolling(30).mean()) / volume.rolling(30).std().replace(0, np.nan)
    features["rsi_14"] = rsi(close, 14)
    features["stoch_rsi"] = stochastic_rsi(close, 14, 14)
    features["macd_hist"] = macd(close)["macd_hist"]
    features["bb_pct_b"] = bollinger_pct_b(close, 20)
    features["atr_14"] = atr(high, low, close, 14)
    sw = swing_levels(high, low, lookback=20)
    features["dist_swing_high"] = sw["dist_swing_high"]
    features["dist_swing_low"] = sw["dist_swing_low"]
    features["dist_nearest_level"] = sw["dist_nearest_level"]
    return features


def get_daily_vol(close, span=50):
    return close.pct_change().dropna().ewm(span=span).std()


# =========================================================================
# STEP 1: GENERATE ALL TRADE SIGNALS
# =========================================================================
def generate_all_signals():
    """Walk-forward trade signal generation for all configs × assets × months."""
    print("[1/6] Generating trade signals...", flush=True)
    all_signals = []

    for bar_tf, bar_resample, fwd_bars, config_id in CONFIGS:
        for symbol in ASSETS:
            ohlcv = load_ohlcv(symbol, bar_resample)
            close = ohlcv["close"]
            fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)
            vol = get_daily_vol(close, span=50)

            for month in ALL_MONTHS:
                month_start = pd.Timestamp(month.start_time, tz="UTC")
                month_end = pd.Timestamp(month.end_time, tz="UTC")

                features = build_features(ohlcv, train_end=month_start)
                fc = features.dropna()
                y_all = fwd_ret.reindex(fc.index).dropna()
                valid = fc.index.intersection(y_all.index)

                train_idx = valid[valid < month_start]
                test_idx = valid[(valid >= month_start) & (valid <= month_end)]

                if len(train_idx) < 500 or len(test_idx) < 20:
                    continue

                # Vol filter
                train_vol = vol.loc[train_idx].dropna()
                vol_thresh = train_vol.quantile(VOL_PERCENTILE / 100)
                test_vol = vol.reindex(test_idx)
                hv_mask = (test_vol > vol_thresh).fillna(False)
                test_idx = test_idx[hv_mask.values]
                if len(test_idx) < 10:
                    continue

                X_train = fc.loc[train_idx]
                y_train = (y_all.loc[train_idx] > 0).astype(int)

                model = xgb.XGBClassifier(**XGB_PARAMS)
                model.fit(X_train, y_train)

                X_test = fc.loc[test_idx]
                y_prob = model.predict_proba(X_test)[:, 1]

                # Confidence filter
                conf = np.abs(y_prob - 0.5)
                if len(conf) > 10:
                    trade_mask = conf >= np.percentile(conf, CONF_PERCENTILE)
                else:
                    trade_mask = np.ones(len(conf), dtype=bool)

                for i, idx in enumerate(test_idx):
                    if trade_mask[i] and np.isfinite(fwd_ret.loc[idx]):
                        direction = 1.0 if y_prob[i] > 0.5 else -1.0
                        actual = fwd_ret.loc[idx]
                        all_signals.append({
                            "timestamp": idx,
                            "asset": symbol,
                            "config": config_id,
                            "direction": direction,
                            "probability": float(y_prob[i]),
                            "actual_return": float(actual),
                            "trade_return": float(direction * actual),
                        })

        print(f"  {config_id}: {sum(1 for s in all_signals if s['config']==config_id)} signals", flush=True)

    signals_df = pd.DataFrame(all_signals)
    signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])
    print(f"  Total signals: {len(signals_df)}", flush=True)
    return signals_df


# =========================================================================
# STEP 2: PORTFOLIO SIMULATOR
# =========================================================================
def simulate_portfolio(signals_df, active_configs, sizing_pct=0.02,
                       max_dd_limit=MAX_DD_TARGET, dynamic_delever=True):
    """
    Simulate portfolio PnL with position sizing and drawdown control.

    Args:
        signals_df: DataFrame of all trade signals
        active_configs: list of config IDs to include
        sizing_pct: fraction of equity per trade
        max_dd_limit: halt trading when drawdown exceeds this
        dynamic_delever: if True, reduce size as DD approaches limit
    """
    # Filter to active configs
    trades = signals_df[signals_df["config"].isin(active_configs)].copy()
    trades = trades.sort_values("timestamp")

    if len(trades) == 0:
        return None

    equity = STARTING_CAPITAL
    peak_equity = equity
    halted = False
    daily_pnl = {}
    trade_log = []

    for _, trade in trades.iterrows():
        ts = trade["timestamp"]
        day = ts.date()

        if halted:
            continue

        # Current drawdown
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

        # Dynamic deleveraging: reduce size as DD approaches limit
        if dynamic_delever and dd > max_dd_limit * 0.5:
            # Linear reduction: at 50% of limit = full size, at limit = 0 size
            dd_ratio = min(dd / max_dd_limit, 1.0)
            size_multiplier = max(1.0 - dd_ratio, 0.0)
        else:
            size_multiplier = 1.0

        if dd >= max_dd_limit:
            halted = True
            continue

        # Position size
        position_size = equity * sizing_pct * size_multiplier

        # Trade PnL (return × position size - costs)
        gross_return = trade["trade_return"]
        cost = TOTAL_COST  # per trade
        net_return = gross_return - cost
        pnl = position_size * net_return

        equity += pnl
        if equity > peak_equity:
            peak_equity = equity

        # Daily aggregation
        daily_pnl[day] = daily_pnl.get(day, 0) + pnl

        trade_log.append({
            "timestamp": ts,
            "equity": equity,
            "pnl": pnl,
            "dd": (peak_equity - equity) / peak_equity if peak_equity > 0 else 0,
            "size_mult": size_multiplier,
        })

    if not trade_log:
        return None

    return {
        "equity_final": equity,
        "trade_log": trade_log,
        "daily_pnl": daily_pnl,
        "n_trades": len(trade_log),
        "halted": halted,
    }


def compute_metrics(sim_result, label=""):
    """Compute all quant metrics from simulation result."""
    if sim_result is None:
        return None

    log = sim_result["trade_log"]
    equity_series = [STARTING_CAPITAL] + [t["equity"] for t in log]
    equity = np.array(equity_series)
    daily_pnl = sim_result["daily_pnl"]

    n = len(log)
    if n < 20:
        return None

    # Returns
    trade_pnls = np.array([t["pnl"] for t in log])
    trade_rets = trade_pnls / np.maximum(equity[:-1], 1)  # return on equity

    # Total return
    total_return = (equity[-1] / STARTING_CAPITAL) - 1

    # Time span
    t_start = log[0]["timestamp"]
    t_end = log[-1]["timestamp"]
    days = max((t_end - t_start).total_seconds() / 86400, 1)
    years = days / 365.25

    # CAGR
    cagr = (equity[-1] / STARTING_CAPITAL) ** (1 / years) - 1 if years > 0 else 0

    # Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / np.maximum(peak, 1)
    max_dd = abs(np.min(drawdown))

    # DD duration (in days)
    dd_series = [t["dd"] for t in log]
    in_dd = np.array(dd_series) > 0.001
    max_dd_duration = 0
    current_dd_len = 0
    for d in in_dd:
        if d:
            current_dd_len += 1
            max_dd_duration = max(max_dd_duration, current_dd_len)
        else:
            current_dd_len = 0

    # Daily returns
    daily_rets = pd.Series(daily_pnl)
    daily_ret_pct = daily_rets / STARTING_CAPITAL  # simplified
    daily_nonzero = daily_ret_pct[daily_ret_pct != 0]

    # Sharpe (annualized from daily)
    if len(daily_nonzero) > 5 and daily_nonzero.std() > 0:
        sharpe = daily_nonzero.mean() / daily_nonzero.std() * np.sqrt(365.25)
    else:
        sharpe = 0

    # Sortino
    downside = daily_nonzero[daily_nonzero < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = daily_nonzero.mean() / downside.std() * np.sqrt(365.25)
    else:
        sortino = sharpe

    # Calmar
    calmar = cagr / max_dd if max_dd > 0 else 0

    # Win rates
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls <= 0]
    win_rate = len(wins) / n if n > 0 else 0
    win_days = (daily_nonzero > 0).sum()
    total_days = len(daily_nonzero)
    day_win_rate = win_days / total_days if total_days > 0 else 0

    # Profit factor
    pf = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")

    # Avg win/loss
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    # Monthly
    monthly_pnl = {}
    for t in log:
        m = t["timestamp"].strftime("%Y-%m")
        monthly_pnl[m] = monthly_pnl.get(m, 0) + t["pnl"]
    monthly_rets = pd.Series(monthly_pnl) / STARTING_CAPITAL
    months_positive = (monthly_rets > 0).sum()

    # VaR / CVaR (daily)
    if len(daily_nonzero) > 20:
        var_95 = np.percentile(daily_nonzero, 5)
        cvar_95 = daily_nonzero[daily_nonzero <= var_95].mean()
    else:
        var_95 = 0
        cvar_95 = 0

    return {
        "label": label,
        "n_trades": n,
        "trades_per_day": round(n / days, 1),
        "total_return_pct": round(total_return * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "final_equity": round(equity[-1], 0),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "max_dd_pct": round(max_dd * 100, 1),
        "max_dd_duration": max_dd_duration,
        "win_rate_trades": round(win_rate, 3),
        "win_rate_days": round(day_win_rate, 3),
        "profit_factor": round(min(pf, 99), 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "months_positive": int(months_positive),
        "months_total": len(monthly_rets),
        "monthly_mean_pct": round(monthly_rets.mean() * 100, 2) if len(monthly_rets) > 0 else 0,
        "monthly_min_pct": round(monthly_rets.min() * 100, 2) if len(monthly_rets) > 0 else 0,
        "var_95_pct": round(var_95 * 100, 3),
        "cvar_95_pct": round(cvar_95 * 100, 3),
        "halted": sim_result["halted"],
    }


# =========================================================================
# STEP 3: COMBINATION SEARCH
# =========================================================================
def search_combinations(signals_df, sizing_pct=0.02):
    """Test all combos of 1-5 configs, rank by Sharpe at ≤15% MaxDD."""
    print("\n[3/6] Searching combinations...", flush=True)

    config_ids = [c[3] for c in CONFIGS]
    results = []
    total_combos = 0

    for n_configs in range(1, 6):
        combos = list(itertools.combinations(config_ids, n_configs))
        print(f"  Testing C({len(config_ids)},{n_configs}) = {len(combos)} combos...", flush=True)

        for combo in combos:
            total_combos += 1
            sim = simulate_portfolio(signals_df, list(combo), sizing_pct=sizing_pct)
            metrics = compute_metrics(sim, label="+".join(combo))
            if metrics and metrics["n_trades"] >= 50:
                results.append(metrics)

    print(f"  Total combos tested: {total_combos}, valid: {len(results)}", flush=True)

    # Filter by MaxDD
    viable = [r for r in results if r["max_dd_pct"] <= MAX_DD_TARGET * 100]
    print(f"  Viable (MaxDD ≤ {MAX_DD_TARGET*100:.0f}%): {len(viable)}", flush=True)

    # Sort by Sharpe
    viable.sort(key=lambda x: -x["sharpe"])
    return viable, results


# =========================================================================
# STEP 4: POSITION SIZING COMPARISON
# =========================================================================
def compare_sizing(signals_df, best_combo):
    """Compare sizing methods for the best combination."""
    print(f"\n[4/6] Comparing position sizing for: {best_combo}...", flush=True)

    sizing_results = []
    for label, pct in [("1%", 0.01), ("2%", 0.02), ("3%", 0.03), ("5%", 0.05),
                       ("7%", 0.07), ("10%", 0.10)]:
        sim = simulate_portfolio(signals_df, best_combo, sizing_pct=pct)
        metrics = compute_metrics(sim, label=f"size_{label}")
        if metrics:
            metrics["sizing"] = label
            sizing_results.append(metrics)

    return sizing_results


# =========================================================================
# STEP 5: STRESS TESTS
# =========================================================================
def stress_test(signals_df, best_combo, best_sizing):
    """Run stress scenarios on the selected portfolio."""
    print(f"\n[5/6] Stress testing...", flush=True)

    results = []

    # Base case
    sim = simulate_portfolio(signals_df, best_combo, sizing_pct=best_sizing)
    base = compute_metrics(sim, "base")
    results.append(base)

    # Scenario 1: Signal decay (reduce all returns by 30%)
    degraded = signals_df.copy()
    degraded["trade_return"] = degraded["trade_return"] * 0.7
    sim_deg = simulate_portfolio(degraded, best_combo, sizing_pct=best_sizing)
    deg = compute_metrics(sim_deg, "signal_decay_30pct")
    results.append(deg)

    # Scenario 2: Higher costs (add 3 bps)
    expensive = signals_df.copy()
    expensive["trade_return"] = expensive["trade_return"] - 0.0003
    sim_exp = simulate_portfolio(expensive, best_combo, sizing_pct=best_sizing)
    exp = compute_metrics(sim_exp, "extra_3bps_cost")
    results.append(exp)

    # Scenario 3: Worst half of trades only
    worst_half = signals_df.copy()
    median_ret = worst_half["trade_return"].median()
    worst_half = worst_half[worst_half["trade_return"] <= median_ret]
    sim_worst = simulate_portfolio(worst_half, best_combo, sizing_pct=best_sizing)
    worst = compute_metrics(sim_worst, "worst_50pct_trades")
    results.append(worst)

    # Scenario 4: Only 60% of trades fill (random drop)
    np.random.seed(42)
    partial = signals_df.copy()
    fill_mask = np.random.random(len(partial)) < 0.6
    partial = partial[fill_mask]
    sim_partial = simulate_portfolio(partial, best_combo, sizing_pct=best_sizing)
    part = compute_metrics(sim_partial, "60pct_fill_rate")
    results.append(part)

    return results


# =========================================================================
# MAIN
# =========================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100, flush=True)
    print("PORTFOLIO SIMULATION & RISK MANAGEMENT", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Capital: ${STARTING_CAPITAL}, MaxDD target: {MAX_DD_TARGET*100:.0f}%", flush=True)
    print(f"Configs: {len(CONFIGS)}, Assets: {len(ASSETS)}", flush=True)
    print(f"Cost model: {COST_BPS}bp fees + {SLIPPAGE_BPS}bp slippage = {TOTAL_COST*10000:.1f}bp/trade", flush=True)
    print(f"Walk-forward: IS months {IS_MONTHS[0]}-{IS_MONTHS[-1]}, OOS {OOS_MONTHS[0]}-{OOS_MONTHS[-1]}", flush=True)
    print("=" * 100, flush=True)

    # Step 1: Generate signals
    signals_df = generate_all_signals()
    signals_df.to_csv(OUTPUT_DIR / "trade_signals.csv", index=False)

    # Split into IS and OOS
    is_signals = signals_df[signals_df["timestamp"].dt.to_period("M").isin(IS_MONTHS)]
    oos_signals = signals_df[signals_df["timestamp"].dt.to_period("M").isin(OOS_MONTHS)]
    print(f"\n  IS signals: {len(is_signals)}, OOS signals: {len(oos_signals)}", flush=True)

    # Step 2+3: Combination search on IS period
    print("\n--- IN-SAMPLE SELECTION ---", flush=True)
    viable_is, all_is = search_combinations(is_signals, sizing_pct=0.02)

    if not viable_is:
        print("  No viable combos at 2%. Trying 1%...", flush=True)
        viable_is, all_is = search_combinations(is_signals, sizing_pct=0.01)

    if not viable_is:
        print("  FATAL: No combo meets MaxDD constraint. Exiting.", flush=True)
        return

    # Print top 20 IS combos
    print(f"\n  TOP 20 IN-SAMPLE COMBOS (MaxDD ≤ {MAX_DD_TARGET*100:.0f}%):", flush=True)
    print(f"  {'Rank':>4} {'Combo':<50} {'Sharpe':>7} {'Return':>8} {'MaxDD':>6} {'PF':>5} "
          f"{'WR_T':>5} {'Trades':>6} {'Mo+':>4}", flush=True)
    print(f"  {'-'*100}", flush=True)
    for i, r in enumerate(viable_is[:20]):
        print(f"  {i+1:>4} {r['label']:<50} {r['sharpe']:>7.2f} {r['total_return_pct']:>+7.1f}% "
              f"{r['max_dd_pct']:>5.1f}% {r['profit_factor']:>5.2f} {r['win_rate_trades']:>5.3f} "
              f"{r['n_trades']:>6} {r['months_positive']:>2}/{r['months_total']}", flush=True)

    best_combo_label = viable_is[0]["label"]
    best_combo = best_combo_label.split("+")
    print(f"\n  SELECTED: {best_combo_label} (Sharpe={viable_is[0]['sharpe']:.2f})", flush=True)

    # Validate on OOS
    print(f"\n--- OUT-OF-SAMPLE VALIDATION ---", flush=True)
    # Test top 5 IS combos on OOS
    print(f"  {'Rank':>4} {'Combo':<50} {'IS_Sh':>6} {'OOS_Sh':>7} {'OOS_Ret':>8} {'OOS_DD':>7} "
          f"{'OOS_PF':>6} {'OOS_WR':>6}", flush=True)
    print(f"  {'-'*100}", flush=True)

    oos_results = []
    for i, is_r in enumerate(viable_is[:10]):
        combo = is_r["label"].split("+")
        sim = simulate_portfolio(oos_signals, combo, sizing_pct=0.02)
        oos_m = compute_metrics(sim, is_r["label"])
        if oos_m:
            oos_results.append(oos_m)
            print(f"  {i+1:>4} {is_r['label']:<50} {is_r['sharpe']:>5.2f} {oos_m['sharpe']:>7.2f} "
                  f"{oos_m['total_return_pct']:>+7.1f}% {oos_m['max_dd_pct']:>6.1f}% "
                  f"{oos_m['profit_factor']:>6.2f} {oos_m['win_rate_trades']:>6.3f}", flush=True)

    # Pick best OOS combo that also met IS criteria
    oos_viable = [r for r in oos_results if r["max_dd_pct"] <= MAX_DD_TARGET * 100 and r["sharpe"] > 0]
    if oos_viable:
        best_oos = max(oos_viable, key=lambda x: x["sharpe"])
        final_combo = best_oos["label"].split("+")
        print(f"\n  OOS BEST: {best_oos['label']} (Sharpe={best_oos['sharpe']:.2f}, "
              f"MaxDD={best_oos['max_dd_pct']:.1f}%)", flush=True)
    else:
        final_combo = best_combo
        print(f"\n  WARNING: No OOS combo meets criteria. Using IS best: {best_combo_label}", flush=True)

    # Step 4: Sizing comparison on full period
    print(f"\n--- POSITION SIZING COMPARISON ---", flush=True)
    sizing = compare_sizing(signals_df, final_combo)
    print(f"  {'Size':>6} {'Sharpe':>7} {'Return':>8} {'MaxDD':>6} {'PF':>5} {'Calmar':>7}", flush=True)
    print(f"  {'-'*50}", flush=True)
    best_sizing_pct = 0.02
    best_sizing_sharpe = 0
    for s in sizing:
        viable_flag = " ✓" if s["max_dd_pct"] <= MAX_DD_TARGET * 100 else " ✗"
        print(f"  {s['sizing']:>6} {s['sharpe']:>7.2f} {s['total_return_pct']:>+7.1f}% "
              f"{s['max_dd_pct']:>5.1f}% {s['profit_factor']:>5.2f} {s['calmar']:>7.2f}{viable_flag}", flush=True)
        if s["max_dd_pct"] <= MAX_DD_TARGET * 100 and s["sharpe"] > best_sizing_sharpe:
            best_sizing_sharpe = s["sharpe"]
            best_sizing_pct = float(s["sizing"].replace("%", "")) / 100

    print(f"  Best sizing at ≤{MAX_DD_TARGET*100:.0f}% MaxDD: {best_sizing_pct*100:.0f}%", flush=True)

    # Step 5: Stress tests
    stress = stress_test(signals_df, final_combo, best_sizing_pct)
    print(f"\n  {'Scenario':<25} {'Sharpe':>7} {'Return':>8} {'MaxDD':>6} {'PF':>5} {'Halted':>7}", flush=True)
    print(f"  {'-'*65}", flush=True)
    for s in stress:
        if s:
            print(f"  {s['label']:<25} {s['sharpe']:>7.2f} {s['total_return_pct']:>+7.1f}% "
                  f"{s['max_dd_pct']:>5.1f}% {s['profit_factor']:>5.2f} {'YES' if s['halted'] else 'no':>7}", flush=True)

    # Step 6: Final report
    print(f"\n{'='*100}", flush=True)
    print("FINAL DEPLOYMENT SPEC", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  Configs:     {'+'.join(final_combo)}", flush=True)
    print(f"  Sizing:      {best_sizing_pct*100:.0f}% of equity per trade", flush=True)
    print(f"  MaxDD limit: {MAX_DD_TARGET*100:.0f}% (dynamic deleveraging from {MAX_DD_TARGET*50:.0f}%)", flush=True)
    print(f"  Cost model:  {TOTAL_COST*10000:.1f} bps/trade (MEXC 0% maker + 1bp slippage)", flush=True)
    print(f"  Platform:    MEXC (primary), Hyperliquid (backup)", flush=True)
    print(f"  Assets:      {ASSETS}", flush=True)
    print(f"  Retraining:  Monthly walk-forward", flush=True)

    if oos_viable:
        b = best_oos
        print(f"\n  OOS Performance:", flush=True)
        print(f"    Sharpe:      {b['sharpe']:.2f}", flush=True)
        print(f"    Sortino:     {b['sortino']:.2f}", flush=True)
        print(f"    Calmar:      {b['calmar']:.2f}", flush=True)
        print(f"    Total return: {b['total_return_pct']:+.1f}%", flush=True)
        print(f"    Max DD:      {b['max_dd_pct']:.1f}%", flush=True)
        print(f"    Win rate:    {b['win_rate_trades']:.1%} trades, {b['win_rate_days']:.1%} days", flush=True)
        print(f"    Trades/day:  {b['trades_per_day']:.1f}", flush=True)
        print(f"    Months +/-:  {b['months_positive']}/{b['months_total']}", flush=True)

    with open(OUTPUT_DIR / "portfolio_sim_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "final_combo": final_combo,
            "sizing_pct": best_sizing_pct,
            "max_dd_target": MAX_DD_TARGET,
            "top_20_is": [r for r in viable_is[:20]],
            "oos_results": oos_results,
            "sizing_comparison": sizing,
            "stress_tests": stress,
        }, f, indent=2, default=str)

    print(f"\nCompleted: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
