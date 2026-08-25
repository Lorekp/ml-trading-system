"""EXP-F: Cointegration Pairs Trading (Chan Ch7).

Find cointegrated crypto pairs, trade the mean-reverting spread.
Walk-forward: train on rolling 3-month window, trade next month.
Uses ProcessPoolExecutor for per-pair parallelism.
"""
import sys
sys.path.insert(0, "src")

import warnings, os, time, itertools
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/binance-1m")
W = max(1, os.cpu_count() - 1)

ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "ZEC": "ZECUSDT", "BNX": "BNXUSDT", "BNB": "BNBUSDT", "ENJ": "ENJUSDT",
    "TAO": "TAOUSDT", "ADA": "ADAUSDT", "ANKR": "ANKRUSDT", "SUI": "SUIUSDT",
    "AVAX": "AVAXUSDT", "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "FIL": "FILUSDT",
    "DOT": "DOTUSDT", "SXP": "SXPUSDT", "ENA": "ENAUSDT", "BCH": "BCHUSDT",
    "LTC": "LTCUSDT", "UNI": "UNIUSDT", "WLD": "WLDUSDT", "TRX": "TRXUSDT",
    "FTM": "FTMUSDT", "AAVE": "AAVEUSDT", "FET": "FETUSDT", "APT": "APTUSDT",
    "HBAR": "HBARUSDT", "XLM": "XLMUSDT",
}


def load_asset_close(symbol):
    """Load 15m close prices for one asset."""
    from signal_engine.data import load_binance_1m, resample_ohlcv
    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        return ohlcv_15m["close"]
    except Exception:
        return None


def test_cointegration(y, x):
    """Engle-Granger test: regress y on x, test residuals for stationarity."""
    if len(y) < 200 or len(x) < 200:
        return None

    # Align
    common = y.index.intersection(x.index)
    if len(common) < 200:
        return None
    y_c, x_c = y.loc[common].values, x.loc[common].values

    # OLS: y = beta * x + alpha + epsilon
    reg = LinearRegression()
    reg.fit(x_c.reshape(-1, 1), y_c)
    beta = reg.coef_[0]
    residuals = y_c - beta * x_c - reg.intercept_

    # ADF on residuals
    try:
        adf_stat, p_value, _, _, _, _ = adfuller(residuals, maxlag=20)
    except Exception:
        return None

    # Half-life of mean-reversion (AR(1) on spread)
    spread = pd.Series(residuals)
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    common_hl = lag.index.intersection(delta.index)
    if len(common_hl) < 50:
        return None

    try:
        ar_reg = LinearRegression()
        ar_reg.fit(lag.loc[common_hl].values.reshape(-1, 1), delta.loc[common_hl].values)
        phi = ar_reg.coef_[0]
        if phi >= 0:
            half_life = float("inf")
        else:
            half_life = -np.log(2) / phi  # in bars (15m)
            half_life_days = half_life / 96  # 96 bars/day
    except Exception:
        return None

    return {
        "beta": beta,
        "intercept": reg.intercept_,
        "adf_stat": adf_stat,
        "p_value": p_value,
        "half_life_bars": half_life,
        "half_life_days": half_life_days if half_life != float("inf") else float("inf"),
    }


def backtest_pair(y_close, x_close, beta, intercept, z_entry=2.0, z_exit=0.0,
                   lookback=96*30, slippage_bps=5):
    """Walk-forward spread trading backtest."""
    common = y_close.index.intersection(x_close.index)
    y, x = y_close.loc[common], x_close.loc[common]

    if len(y) < lookback + 96 * 30:
        return None

    # Walk-forward: train on lookback, trade next month
    pnls = []
    start = lookback

    while start + 96 * 30 < len(y):
        train_end = start
        test_end = min(start + 96 * 30, len(y))

        # Train: compute spread stats on training window
        train_y = y.iloc[train_end - lookback:train_end]
        train_x = x.iloc[train_end - lookback:train_end]

        # Re-estimate beta on training window
        reg = LinearRegression()
        reg.fit(train_x.values.reshape(-1, 1), train_y.values)
        beta_t = reg.coef_[0]
        intercept_t = reg.intercept_

        train_spread = train_y.values - beta_t * train_x.values - intercept_t
        mu = train_spread.mean()
        sigma = train_spread.std()
        if sigma <= 0:
            start = test_end
            continue

        # Test: trade the spread
        test_y = y.iloc[train_end:test_end]
        test_x = x.iloc[train_end:test_end]
        test_spread = test_y.values - beta_t * test_x.values - intercept_t
        z_scores = (test_spread - mu) / sigma

        position = 0  # 0=flat, 1=long spread, -1=short spread
        entry_z = None
        monthly_pnl = 0.0

        for i in range(1, len(z_scores)):
            z = z_scores[i]
            spread_ret = (test_spread[i] - test_spread[i - 1]) / abs(test_spread[i - 1]) if abs(test_spread[i - 1]) > 1e-10 else 0

            if position != 0:
                monthly_pnl += position * spread_ret

            # Entry
            if position == 0:
                if z < -z_entry:
                    position = 1
                    entry_z = z
                    monthly_pnl -= slippage_bps / 10000 * 2  # round-trip entry slippage
                elif z > z_entry:
                    position = -1
                    entry_z = z
                    monthly_pnl -= slippage_bps / 10000 * 2

            # Exit
            elif position == 1 and z >= z_exit:
                position = 0
                monthly_pnl -= slippage_bps / 10000 * 2  # exit slippage
            elif position == -1 and z <= z_exit:
                position = 0
                monthly_pnl -= slippage_bps / 10000 * 2

        pnls.append(monthly_pnl)
        start = test_end

    if not pnls:
        return None

    pnl_series = pd.Series(pnls)
    mean_ret = pnl_series.mean()
    std_ret = pnl_series.std()
    sharpe = mean_ret / std_ret * np.sqrt(12) if std_ret > 0 else 0  # annualised from monthly
    total_ret = pnl_series.sum() * 100  # percentage

    return {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_ret, 2),
        "n_months": len(pnls),
        "mean_monthly_ret_pct": round(mean_ret * 100, 3),
        "positive_months_pct": round((pnl_series > 0).mean() * 100, 1),
    }


def _test_pair(args):
    """Test one pair for cointegration + backtest. Subprocess."""
    a1, s1, a2, s2 = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    c1 = load_asset_close(s1)
    c2 = load_asset_close(s2)
    if c1 is None or c2 is None:
        return None

    coint = test_cointegration(c1, c2)
    if coint is None:
        return None

    result = {
        "pair": f"{a1}/{a2}",
        "p_value": round(coint["p_value"], 4),
        "half_life_days": round(coint["half_life_days"], 1),
        "beta": round(coint["beta"], 4),
        "cointegrated": coint["p_value"] < 0.05 and 1 < coint["half_life_days"] < 20,
    }

    # Only backtest if cointegrated
    if result["cointegrated"]:
        bt = backtest_pair(c1, c2, coint["beta"], coint["intercept"])
        if bt:
            result.update(bt)

    return result


def main():
    print("=" * 80)
    print(f"EXP-F: COINTEGRATION PAIRS TRADING ({W} workers)")
    print(f"Testing {len(ASSETS)}C2 = {len(ASSETS)*(len(ASSETS)-1)//2} pairs")
    print("=" * 80, flush=True)

    # Generate all pairs
    asset_list = list(ASSETS.items())
    pairs = []
    for i in range(len(asset_list)):
        for j in range(i + 1, len(asset_list)):
            a1, s1 = asset_list[i]
            a2, s2 = asset_list[j]
            pairs.append((a1, s1, a2, s2))

    print(f"Testing {len(pairs)} pairs...", flush=True)
    t0 = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_test_pair, p): p for p in pairs}
        done = 0
        for future in as_completed(futures):
            r = future.result()
            done += 1
            if r is not None:
                results.append(r)
            if done % 50 == 0:
                print(f"  {done}/{len(pairs)} pairs tested...", flush=True)

    elapsed = time.time() - t0
    print(f"\nTested {len(pairs)} pairs in {elapsed:.0f}s")

    # Filter cointegrated
    coint_pairs = [r for r in results if r.get("cointegrated")]
    tested = [r for r in results if r.get("p_value") is not None]

    print(f"\nCointegration results:")
    print(f"  Pairs with ADF p < 0.05: {sum(1 for r in tested if r['p_value'] < 0.05)}/{len(tested)}")
    print(f"  Pairs with tradeable HL (1-20 days): {len(coint_pairs)}/{len(tested)}")
    print()

    if not coint_pairs:
        print("NO cointegrated pairs found. EXP-F KILLED.")
        return

    # Sort by Sharpe
    backtested = [r for r in coint_pairs if "sharpe" in r]
    backtested.sort(key=lambda x: x.get("sharpe", 0), reverse=True)

    print(f"Backtested {len(backtested)} cointegrated pairs:")
    print(f"{'Pair':<12s} {'Sharpe':>7s} {'Ret%':>8s} {'Months':>7s} {'HL(d)':>6s} {'p-val':>6s} {'Win%':>5s}")
    print("-" * 55)
    for r in backtested[:20]:
        print(f"{r['pair']:<12s} {r.get('sharpe',0):>7.3f} {r.get('total_return_pct',0):>8.2f} "
              f"{r.get('n_months',0):>7d} {r['half_life_days']:>6.1f} {r['p_value']:>6.4f} "
              f"{r.get('positive_months_pct',0):>4.0f}%")

    if backtested:
        sharpes = [r["sharpe"] for r in backtested]
        positive = sum(1 for s in sharpes if s > 0)
        print(f"\nSharpe distribution: mean={np.mean(sharpes):.3f} median={np.median(sharpes):.3f} "
              f"positive={positive}/{len(sharpes)}")

        strong = [r for r in backtested if r.get("sharpe", 0) > 1.0]
        print(f"Pairs with Sharpe > 1.0: {len(strong)}")

        if strong:
            print("\nSTRONG PAIRS (Sharpe > 1.0):")
            for r in strong:
                print(f"  {r['pair']}: Sharpe={r['sharpe']:.3f}, Ret={r.get('total_return_pct',0):.2f}%, "
                      f"HL={r['half_life_days']:.1f}d, Win={r.get('positive_months_pct',0):.0f}%")


if __name__ == "__main__":
    main()
