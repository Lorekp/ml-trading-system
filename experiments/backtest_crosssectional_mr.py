"""Cross-Sectional BTC-Neutral Residual Mean Reversion — Backtest

Hypothesis: After stripping BTC beta, altcoins that deviated the most
from expected returns tend to mean-revert over the next 5-7 days.

Methodology:
1. Load daily OHLCV for all available altcoins + BTC
2. Estimate rolling beta to BTC using 60-day OLS window
3. Compute residual = actual_ret - beta * BTC_ret
4. Rank assets by cumulative residual over lookback window
5. Long bottom quintile (oversold residuals), short top quintile
6. Dollar-neutral, equal-weight within each leg, weekly rebalance
7. Walk-forward: W1 (2024-07..2025-02), W2 (2025-03..2025-10), W3 (2025-11..2026-02)

Evidence: JFQA 2024 — Sharpe ~2.3 for residual mean reversion leg.
Weekly alpha 2.62% (t=4.22). Strong post-2021.

Correlation with LogReg: VERY LOW (cross-sectional relative vs directional).
"""

import sys, os, time, warnings
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.asset_selection import compute_monthly_dollar_volume

warnings.filterwarnings("ignore")
DATA_DIR = Path("data/binance-1m")

# Walk-forward windows (same as existing experiments for comparability)
WINDOWS = [
    {"label": "W1", "test_start": "2024-07-01", "test_end": "2025-02-28", "cal_days": 245},
    {"label": "W2", "test_start": "2025-03-01", "test_end": "2025-10-31", "cal_days": 244},
    {"label": "W3", "test_start": "2025-11-01", "test_end": "2026-02-28", "cal_days": 122},
]


# ============================================================
# Data Loading
# ============================================================

def load_all_daily(min_bars: int = 500) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all USDT pairs, resample to daily, return dict + BTC separately.

    Returns:
        asset_daily: {symbol: daily OHLCV DataFrame}
        btc_daily: BTC daily OHLCV DataFrame
    """
    asset_daily = {}
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith("USDT"):
            continue
        symbol = d.name
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            daily = resample_ohlcv(ohlcv_1m, "1D")
            if len(daily) < min_bars:
                continue
            asset_daily[symbol] = daily
        except Exception:
            continue
    btc_daily = asset_daily.get("BTCUSDT")
    return asset_daily, btc_daily


# ============================================================
# Beta Estimation
# ============================================================

def estimate_rolling_betas(
    daily_returns: dict[str, pd.Series],
    btc_returns: pd.Series,
    window: int = 60,
) -> dict[str, pd.Series]:
    """Estimate rolling OLS beta of each asset to BTC.

    beta_i(t) = cov(r_i, r_btc) / var(r_btc) over trailing window.
    Vectorized using rolling covariance/variance.
    """
    betas = {}
    btc_var = btc_returns.rolling(window, min_periods=30).var()

    for symbol, ret in daily_returns.items():
        if symbol == "BTCUSDT":
            continue
        # Align indices
        common = ret.index.intersection(btc_returns.index)
        if len(common) < window:
            continue
        r_aligned = ret.reindex(common)
        btc_aligned = btc_returns.reindex(common)

        cov = r_aligned.rolling(window, min_periods=30).cov(btc_aligned)
        var = btc_aligned.rolling(window, min_periods=30).var()
        beta = cov / var.replace(0, np.nan)
        beta = beta.clip(-3, 5)  # Sanity: beta in [-3, 5]
        betas[symbol] = beta

    return betas


# ============================================================
# Residual Computation
# ============================================================

def compute_residuals(
    daily_returns: dict[str, pd.Series],
    btc_returns: pd.Series,
    betas: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """Compute idiosyncratic residual: r_i - beta_i * r_btc."""
    residuals = {}
    for symbol, beta_s in betas.items():
        ret = daily_returns[symbol]
        common = ret.index.intersection(btc_returns.index).intersection(beta_s.index)
        r_aligned = ret.reindex(common)
        btc_aligned = btc_returns.reindex(common)
        beta_aligned = beta_s.reindex(common)

        resid = r_aligned - beta_aligned * btc_aligned
        residuals[symbol] = resid
    return residuals


def compute_cumulative_residuals(
    residuals: dict[str, pd.Series],
    lookback: int = 5,
) -> dict[str, pd.Series]:
    """Cumulative residual over lookback days — the signal."""
    cum_resid = {}
    for symbol, resid in residuals.items():
        cum_resid[symbol] = resid.rolling(lookback, min_periods=max(3, lookback - 1)).sum()
    return cum_resid


# ============================================================
# Cross-Sectional Ranking & Portfolio Construction
# ============================================================

def construct_portfolio(
    cum_residuals: dict[str, pd.Series],
    eligible_assets: list[str],
    date: pd.Timestamp,
    n_per_leg: int = 6,
) -> dict[str, float]:
    """Rank eligible assets by cumulative residual, build L/S portfolio.

    Returns {symbol: weight} where positive = long, negative = short.
    Weights sum to zero (dollar-neutral). Each leg sums to 1/n_per_leg.
    """
    # Get cumulative residual at this date for each eligible asset
    scores = {}
    for symbol in eligible_assets:
        if symbol not in cum_residuals:
            continue
        cr = cum_residuals[symbol]
        # Find the value at or before this date
        valid = cr.loc[:date].dropna()
        if len(valid) == 0:
            continue
        scores[symbol] = float(valid.iloc[-1])

    if len(scores) < 2 * n_per_leg:
        return {}

    # Rank: lowest residual = most oversold (long), highest = most overbought (short)
    ranked = sorted(scores.items(), key=lambda x: x[1])
    longs = [s for s, _ in ranked[:n_per_leg]]
    shorts = [s for s, _ in ranked[-n_per_leg:]]

    # Equal weight within each leg, dollar-neutral
    w_long = 1.0 / n_per_leg
    w_short = -1.0 / n_per_leg

    weights = {}
    for s in longs:
        weights[s] = w_long
    for s in shorts:
        weights[s] = w_short

    return weights


# ============================================================
# Portfolio Simulation
# ============================================================

def simulate_ls_portfolio(
    daily_returns: dict[str, pd.Series],
    cum_residuals: dict[str, pd.Series],
    asset_volumes: dict[str, pd.Series],
    window: dict,
    params: dict,
) -> dict | None:
    """Walk-forward L/S simulation for one window.

    Returns dict with equity curve, metrics, or None if insufficient data.
    """
    beta_window = params.get("beta_window", 60)
    residual_lookback = params.get("residual_lookback", 5)
    n_per_leg = params.get("n_per_leg", 6)
    rebalance_days = params.get("rebalance_days", 7)
    universe_size = params.get("universe_size", 30)
    starting_capital = params.get("starting_capital", 10000.0)
    slippage_bps = params.get("slippage_bps", 10)  # 10 bps roundtrip
    leverage = params.get("leverage", 1.0)
    min_history_days = params.get("min_history_days", 180)

    test_start = pd.Timestamp(window["test_start"], tz="UTC")
    test_end = pd.Timestamp(window["test_end"], tz="UTC")

    # Build date range of trading days (all days in range with BTC data)
    btc_ret = daily_returns.get("BTCUSDT")
    if btc_ret is None:
        return None
    all_dates = btc_ret.index[(btc_ret.index >= test_start) & (btc_ret.index <= test_end)]
    if len(all_dates) < 30:
        return None

    # Track state
    equity = starting_capital
    equity_curve = []
    daily_pnl_list = []
    dates_list = []
    positions: dict[str, float] = {}  # {symbol: weight}
    last_rebalance_idx = -rebalance_days  # Force first rebalance
    n_trades = 0
    total_turnover = 0.0

    for i, date in enumerate(all_dates):
        # Daily P&L from held positions
        day_pnl = 0.0
        for symbol, weight in positions.items():
            if symbol in daily_returns:
                ret_s = daily_returns[symbol]
                if date in ret_s.index:
                    # P&L = equity * leverage * weight * return
                    day_pnl += equity * leverage * weight * float(ret_s.loc[date])

        equity += day_pnl
        equity_curve.append(equity)
        daily_pnl_list.append(day_pnl)
        dates_list.append(date)

        # Rebalance check
        if i - last_rebalance_idx >= rebalance_days:
            # Select universe for this month
            lookback_end = date - pd.Timedelta(days=1)
            eligible = []
            for symbol, vol_s in asset_volumes.items():
                if symbol == "BTCUSDT":
                    continue
                avail = vol_s.loc[:lookback_end]
                if len(avail) < 45:
                    continue
                latest = avail.iloc[-1]
                if np.isnan(latest) or latest <= 0:
                    continue
                # Age filter
                first_date = daily_returns[symbol].index.min()
                age = (date - first_date).days
                if age < min_history_days:
                    continue
                # Spike filter
                if len(avail) >= 90:
                    m90 = avail.iloc[-90:].median()
                    if m90 > 0 and latest / m90 > 5.0:
                        continue
                eligible.append((symbol, float(latest)))

            # Top N by volume
            eligible.sort(key=lambda x: -x[1])
            universe = [s for s, _ in eligible[:universe_size]]

            # Construct new portfolio
            new_positions = construct_portfolio(
                cum_residuals, universe, date, n_per_leg=n_per_leg
            )

            if new_positions:
                # Compute turnover and transaction costs
                old_set = set(positions.keys())
                new_set = set(new_positions.keys())
                turnover = 0.0
                for s in old_set | new_set:
                    old_w = positions.get(s, 0.0)
                    new_w = new_positions.get(s, 0.0)
                    turnover += abs(new_w - old_w)

                # Transaction cost: slippage * turnover * leverage * equity
                tc = slippage_bps / 10000.0 * turnover * leverage * equity
                equity -= tc
                total_turnover += turnover

                # Count trades
                for s in new_set:
                    if s not in old_set or abs(new_positions[s] - positions.get(s, 0)) > 0.001:
                        n_trades += 1

                positions = new_positions
                last_rebalance_idx = i

        # Halt on max drawdown
        peak = max(equity_curve) if equity_curve else starting_capital
        dd = (equity - peak) / peak
        if dd < -0.10:
            break

    if len(equity_curve) < 10:
        return None

    equity_arr = np.array(equity_curve)
    pnl_arr = np.array(daily_pnl_list)
    dates_arr = np.array(dates_list)

    # Metrics
    daily_ret_pct = pnl_arr / starting_capital
    peak = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peak) / peak
    max_dd = abs(dd.min())
    worst_day = abs(daily_ret_pct.min()) if len(daily_ret_pct) > 0 else 0
    total_pnl = equity_arr[-1] - starting_capital
    sharpe = 0.0
    if len(daily_ret_pct) > 5:
        mean_r = daily_ret_pct.mean()
        std_r = daily_ret_pct.std()
        if std_r > 0:
            sharpe = mean_r / std_r * np.sqrt(365.25)

    n_days = len(equity_curve)
    pnl_per_cal = total_pnl / window["cal_days"] if window["cal_days"] > 0 else 0

    return {
        "pnl": total_pnl,
        "pnl_cal": pnl_per_cal,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "worst_day": worst_day,
        "n_trades": n_trades,
        "n_days": n_days,
        "turnover": total_turnover,
        "equity": equity_arr,
        "dates": dates_arr,
    }


# ============================================================
# Exploration: Is the signal real?
# ============================================================

def analyze_signal_quality(
    cum_residuals: dict[str, pd.Series],
    daily_returns: dict[str, pd.Series],
    btc_returns: pd.Series,
    eligible_symbols: list[str],
    fwd_days: int = 7,
) -> dict:
    """Test whether residual rank predicts forward returns.

    For each date, rank assets by cumulative residual, then check
    if bottom quintile outperforms top quintile over next fwd_days.
    """
    # Build panel: date × asset residual & forward return
    all_dates = sorted(set().union(*(
        cum_residuals[s].dropna().index for s in eligible_symbols if s in cum_residuals
    )))

    long_wins = 0
    short_wins = 0
    spread_returns = []
    n_obs = 0

    for date in all_dates:
        scores = {}
        fwd_rets = {}
        for symbol in eligible_symbols:
            if symbol not in cum_residuals or symbol not in daily_returns:
                continue
            cr = cum_residuals[symbol]
            ret = daily_returns[symbol]
            if date not in cr.index or pd.isna(cr.loc[date]):
                continue

            # Forward return over next fwd_days
            future_dates = ret.index[(ret.index > date)][:fwd_days]
            if len(future_dates) < fwd_days - 1:
                continue
            fwd_r = ret.reindex(future_dates).sum()
            # Hedge: subtract BTC return over same period
            btc_future = btc_returns.reindex(future_dates)
            # Don't hedge here — we measure the raw L/S spread

            scores[symbol] = float(cr.loc[date])
            fwd_rets[symbol] = float(fwd_r)

        if len(scores) < 12:
            continue

        # Rank
        ranked = sorted(scores.items(), key=lambda x: x[1])
        n = len(ranked)
        n_leg = max(3, n // 5)
        longs = [s for s, _ in ranked[:n_leg]]
        shorts = [s for s, _ in ranked[-n_leg:]]

        # Mean forward return per leg
        long_ret = np.mean([fwd_rets[s] for s in longs])
        short_ret = np.mean([fwd_rets[s] for s in shorts])
        spread = long_ret - short_ret  # L/S spread return

        spread_returns.append(spread)
        if long_ret > 0:
            long_wins += 1
        if short_ret < 0:
            short_wins += 1
        n_obs += 1

    if n_obs < 20:
        return {"n_obs": n_obs, "error": "insufficient_data"}

    spread_arr = np.array(spread_returns)
    mean_spread = spread_arr.mean()
    std_spread = spread_arr.std()
    t_stat = mean_spread / (std_spread / np.sqrt(n_obs)) if std_spread > 0 else 0
    annualized_sharpe = mean_spread / std_spread * np.sqrt(365.25 / fwd_days) if std_spread > 0 else 0

    return {
        "n_obs": n_obs,
        "mean_spread_bps": mean_spread * 10000,
        "t_stat": t_stat,
        "annualized_sharpe": annualized_sharpe,
        "long_win_pct": long_wins / n_obs * 100,
        "short_win_pct": short_wins / n_obs * 100,
        "spread_std_bps": std_spread * 10000,
    }


# ============================================================
# XGBoost Enhancement (Phase 2)
# ============================================================

def xgboost_enhanced_portfolio(
    daily_returns: dict[str, pd.Series],
    cum_residuals: dict[str, pd.Series],
    betas: dict[str, pd.Series],
    eligible_assets: list[str],
    date: pd.Timestamp,
    lookback_train: int = 120,
    n_per_leg: int = 6,
) -> dict[str, float]:
    """Use XGBoost to predict which residual reversals will succeed.

    Features per asset at each date:
    - cum_residual (the raw signal)
    - beta stability (std of beta over last 30d)
    - residual volatility (std of residual over last 30d)
    - asset return momentum (30d cumulative return)
    - cross-sectional rank of residual

    Target: whether residual mean-reverts (residual shrinks) over next 7 days.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return construct_portfolio(cum_residuals, eligible_assets, date, n_per_leg)

    # Build training data from past lookback_train days
    btc_ret = daily_returns.get("BTCUSDT")
    if btc_ret is None:
        return {}

    train_start = date - pd.Timedelta(days=lookback_train + 30)
    train_end = date - pd.Timedelta(days=8)  # 7-day gap to avoid lookahead

    features_list = []
    labels_list = []

    train_dates = btc_ret.index[(btc_ret.index >= train_start) & (btc_ret.index <= train_end)]

    for d in train_dates[::7]:  # Weekly samples to avoid overlap
        for symbol in eligible_assets:
            if symbol not in cum_residuals or symbol not in daily_returns or symbol not in betas:
                continue
            cr = cum_residuals[symbol]
            ret = daily_returns[symbol]
            beta = betas[symbol]

            if d not in cr.index or pd.isna(cr.loc[d]):
                continue

            # Features
            cr_val = float(cr.loc[d])
            cr_30 = cr.loc[:d].tail(30)
            ret_30 = ret.loc[:d].tail(30)
            beta_30 = beta.loc[:d].tail(30)

            if len(cr_30) < 20 or len(ret_30) < 20 or len(beta_30) < 20:
                continue

            feat = [
                cr_val,
                float(cr_30.std()),
                float(ret_30.sum()),
                float(ret_30.std()),
                float(beta_30.std()),
                float(beta_30.iloc[-1]),
            ]

            # Label: does residual mean-revert over next 7 days?
            future_dates = cr.index[(cr.index > d)][:7]
            if len(future_dates) < 5:
                continue
            future_cr = cr.reindex(future_dates).dropna()
            if len(future_cr) == 0:
                continue
            # Mean reversion = residual moved toward zero
            label = 1 if abs(float(future_cr.iloc[-1])) < abs(cr_val) else 0

            features_list.append(feat)
            labels_list.append(label)

    if len(features_list) < 100:
        return construct_portfolio(cum_residuals, eligible_assets, date, n_per_leg)

    X_train = np.array(features_list)
    y_train = np.array(labels_list)

    xgb = XGBClassifier(
        max_depth=3, n_estimators=50, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss", verbosity=0,
    )
    xgb.fit(X_train, y_train)

    # Predict for current eligible assets
    scores = {}
    for symbol in eligible_assets:
        if symbol not in cum_residuals or symbol not in daily_returns or symbol not in betas:
            continue
        cr = cum_residuals[symbol]
        ret = daily_returns[symbol]
        beta = betas[symbol]

        if date not in cr.index or pd.isna(cr.loc[date]):
            continue

        cr_val = float(cr.loc[date])
        cr_30 = cr.loc[:date].tail(30)
        ret_30 = ret.loc[:date].tail(30)
        beta_30 = beta.loc[:date].tail(30)

        if len(cr_30) < 20 or len(ret_30) < 20 or len(beta_30) < 20:
            continue

        feat = np.array([[
            cr_val,
            float(cr_30.std()),
            float(ret_30.sum()),
            float(ret_30.std()),
            float(beta_30.std()),
            float(beta_30.iloc[-1]),
        ]])
        prob = float(xgb.predict_proba(feat)[0, 1])
        # Score = direction signal * ML confidence
        direction = -1.0 if cr_val > 0 else 1.0  # Short if positive residual
        scores[symbol] = (direction, prob, cr_val)

    if len(scores) < 2 * n_per_leg:
        return construct_portfolio(cum_residuals, eligible_assets, date, n_per_leg)

    # Select: top n_per_leg longs with highest reversal probability
    long_candidates = [(s, p, cr) for s, (d, p, cr) in scores.items() if d > 0]
    short_candidates = [(s, p, cr) for s, (d, p, cr) in scores.items() if d < 0]

    long_candidates.sort(key=lambda x: -x[1])
    short_candidates.sort(key=lambda x: -x[1])

    longs = [s for s, _, _ in long_candidates[:n_per_leg]]
    shorts = [s for s, _, _ in short_candidates[:n_per_leg]]

    weights = {}
    for s in longs:
        weights[s] = 1.0 / max(len(longs), 1)
    for s in shorts:
        weights[s] = -1.0 / max(len(shorts), 1)
    return weights


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 100)
    print("CROSS-SECTIONAL BTC-NEUTRAL RESIDUAL MEAN REVERSION — BACKTEST")
    print("=" * 100)

    # ---- Load data ----
    print("\nLoading data...", end=" ", flush=True)
    asset_daily, btc_daily = load_all_daily(min_bars=400)
    if btc_daily is None:
        print("ERROR: No BTC data")
        return
    print(f"loaded {len(asset_daily)} assets in {time.time()-t0:.0f}s")

    # ---- Compute daily returns ----
    daily_returns = {}
    for symbol, df in asset_daily.items():
        daily_returns[symbol] = df["close"].pct_change().dropna()

    btc_returns = daily_returns["BTCUSDT"]
    print(f"BTC daily returns: {len(btc_returns)} days "
          f"({btc_returns.index.min().date()} to {btc_returns.index.max().date()})")

    # ---- Compute volumes for asset selection ----
    asset_volumes = {}
    for symbol, df in asset_daily.items():
        dollar_vol = (df["close"] * df["volume"]).rolling(30, min_periods=20).median()
        asset_volumes[symbol] = dollar_vol

    # ---- Phase 1: Signal Quality Analysis ----
    print("\n" + "=" * 80)
    print("PHASE 1: SIGNAL QUALITY ANALYSIS")
    print("=" * 80)

    # Pre-compute betas and residuals for signal analysis
    print("\nEstimating betas (60-day rolling OLS)...", end=" ", flush=True)
    betas = estimate_rolling_betas(daily_returns, btc_returns, window=60)
    print(f"done ({len(betas)} assets)")

    print("Computing residuals...", end=" ", flush=True)
    residuals = compute_residuals(daily_returns, btc_returns, betas)
    print(f"done")

    # Test at different residual lookbacks
    print(f"\n{'Lookback':>10s} {'N_obs':>7s} {'Spread(bps)':>12s} {'t-stat':>8s} "
          f"{'Ann.Sharpe':>10s} {'Long Win%':>10s} {'Short Win%':>11s}")
    print("-" * 80)

    for lb in [3, 5, 7, 10, 15, 20]:
        cum_resid = compute_cumulative_residuals(residuals, lookback=lb)
        # Use all available symbols for signal analysis
        eligible = [s for s in cum_resid.keys() if s != "BTCUSDT"]
        stats = analyze_signal_quality(
            cum_resid, daily_returns, btc_returns, eligible, fwd_days=7
        )
        if "error" in stats:
            print(f"{lb:>10d}   {stats.get('error', 'N/A')}")
            continue
        print(f"{lb:>10d} {stats['n_obs']:>7d} {stats['mean_spread_bps']:>12.1f} "
              f"{stats['t_stat']:>8.2f} {stats['annualized_sharpe']:>10.2f} "
              f"{stats['long_win_pct']:>9.1f}% {stats['short_win_pct']:>10.1f}%")

    # ---- Phase 2: Walk-Forward Backtest ----
    print("\n" + "=" * 80)
    print("PHASE 2: WALK-FORWARD BACKTEST")
    print("=" * 80)

    # Default parameters
    default_params = {
        "beta_window": 60,
        "residual_lookback": 5,
        "n_per_leg": 6,
        "rebalance_days": 7,
        "universe_size": 30,
        "starting_capital": 10000.0,
        "slippage_bps": 10,
        "leverage": 1.0,
        "min_history_days": 180,
    }

    # Parameter grid to test
    configs = [
        # (label, param_overrides)
        ("Base (lb5, reb7, 1x)", {}),
        ("lb3, reb5", {"residual_lookback": 3, "rebalance_days": 5}),
        ("lb10, reb7", {"residual_lookback": 10}),
        ("lb20, reb14", {"residual_lookback": 20, "rebalance_days": 14}),
        ("lb5, reb5", {"rebalance_days": 5}),
        ("lb5, reb14", {"rebalance_days": 14}),
        ("n_leg=3 (tercile)", {"n_per_leg": 3}),
        ("n_leg=10 (decile)", {"n_per_leg": 10}),
        ("beta30", {"beta_window": 30}),
        ("beta90", {"beta_window": 90}),
        ("universe20", {"universe_size": 20}),
        ("universe50", {"universe_size": 50}),
        ("2x leverage", {"leverage": 2.0}),
        ("3x leverage", {"leverage": 3.0}),
        ("5x leverage", {"leverage": 5.0}),
        ("age90d", {"min_history_days": 90}),
        ("5bps slippage", {"slippage_bps": 5}),
    ]

    print(f"\n{'Config':<25s} {'$/cal':>7s} {'Sharpe':>7s} {'DD%':>6s} {'WD%':>6s} "
          f"{'Trades':>6s}  W1$/d  W2$/d  W3$/d  Status")
    print("-" * 110)

    for label, overrides in configs:
        params = {**default_params, **overrides}

        # Pre-compute with this config's beta window
        bw = params["beta_window"]
        rl = params["residual_lookback"]
        b = estimate_rolling_betas(daily_returns, btc_returns, window=bw)
        r = compute_residuals(daily_returns, btc_returns, b)
        cr = compute_cumulative_residuals(r, lookback=rl)

        tp = tc = tt = 0
        mdd = mwd = 0.0
        per_window = {}

        for w in WINDOWS:
            result = simulate_ls_portfolio(daily_returns, cr, asset_volumes, w, params)
            if result is None:
                per_window[w["label"]] = 0
                continue
            per_window[w["label"]] = result["pnl_cal"]
            tp += result["pnl"]
            tc += w["cal_days"]
            tt += result["n_trades"]
            mdd = max(mdd, result["max_dd"])
            mwd = max(mwd, result["worst_day"])

        if tc == 0:
            print(f"{label:<25s} NO DATA")
            continue

        pnl_cal = tp / tc
        d2t = int(1000 / pnl_cal) if pnl_cal > 0 else 9999

        ok = mdd < 0.10 and mwd < 0.05
        w1 = per_window.get("W1", 0)
        w2 = per_window.get("W2", 0)
        w3 = per_window.get("W3", 0)

        print(f"{label:<25s} ${pnl_cal:>6.2f} {0:>7.2f} {mdd*100:>5.1f}% "
              f"{mwd*100:>5.2f}% {tt:>6d}  "
              f"${w1:>5.2f} ${w2:>5.2f} ${w3:>5.2f}  "
              f"{'OK' if ok else 'FAIL'}")

    # ---- Phase 3: XGBoost Enhancement (if base works) ----
    print("\n" + "=" * 80)
    print("PHASE 3: XGBOOST ENHANCEMENT")
    print("=" * 80)

    # Only run if base strategy shows promise (signal quality t > 2)
    cum_resid_base = compute_cumulative_residuals(residuals, lookback=5)
    eligible_base = [s for s in cum_resid_base.keys() if s != "BTCUSDT"]
    sq = analyze_signal_quality(cum_resid_base, daily_returns, btc_returns, eligible_base, fwd_days=7)

    if sq.get("t_stat", 0) < 1.5:
        print("Base signal too weak for XGBoost enhancement (t < 1.5)")
        print(f"Signal quality: t={sq.get('t_stat', 0):.2f}, spread={sq.get('mean_spread_bps', 0):.1f}bps")
    else:
        print("Base signal viable, testing XGBoost enhancement...")
        # This would use xgboost_enhanced_portfolio in the simulation loop
        # For now, just report signal quality
        print(f"Base signal: t={sq['t_stat']:.2f}, spread={sq['mean_spread_bps']:.1f}bps, "
              f"Sharpe={sq['annualized_sharpe']:.2f}")
        print("(XGBoost integration reserved for Phase 2 if base validates)")

    # ---- Summary ----
    print(f"\n{'='*100}")
    print(f"Total runtime: {time.time()-t0:.0f}s")
    print(f"\nNext steps:")
    print(f"  - If signal quality t > 2.0: proceed to live strategy implementation")
    print(f"  - If Sharpe > 1.5 on walk-forward: deploy alongside LogReg")
    print(f"  - Compare correlation with LogReg PnL to confirm diversification")


if __name__ == "__main__":
    main()
