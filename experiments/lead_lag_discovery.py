"""Lead-Lag Network Discovery — BTC/ETH Lead, Alts Follow

Hypothesis: BTC and ETH price moves lead altcoin moves by 1-5 minutes
due to information propagation asymmetry. When BTC's 1-min return exceeds
2-sigma, go long alts with high BTC beta.

Phase 1: Signal discovery — compute cross-correlations at various lags
Phase 2: If signal exists, build and backtest a trading strategy

Kill criteria:
- Peak lag correlation < 0.05 for most assets (no lead-lag structure)
- Signal decays by 5 minutes (not exploitable at realistic latency)
- Net SR < 1.0 after 3 bps cost
"""

import sys, os, time, warnings
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from signal_engine.data import load_binance_1m, resample_ohlcv

warnings.filterwarnings("ignore")

DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]

# Top 20 liquid alts (excluding BTC)
ALTS = {
    "ETH": "ETHUSDT", "XRP": "XRPUSDT", "BNB": "BNBUSDT",
    "SOL": "SOLUSDT", "ADA": "ADAUSDT", "AVAX": "AVAXUSDT",
    "DOT": "DOTUSDT", "LINK": "LINKUSDT", "NEAR": "NEARUSDT",
    "UNI": "UNIUSDT", "LTC": "LTCUSDT", "BCH": "BCHUSDT",
    "FIL": "FILUSDT", "HBAR": "HBARUSDT", "TRX": "TRXUSDT",
    "SUI": "SUIUSDT", "APT": "APTUSDT", "FET": "FETUSDT",
    "AAVE": "AAVEUSDT", "ENJ": "ENJUSDT",
}


def compute_lead_lag_correlations(btc_ret, alt_ret, max_lag=10):
    """Compute cross-correlations at various lags.

    corr(BTC_ret_t, ALT_ret_{t+k}) for k = -max_lag to +max_lag
    Positive k means BTC leads ALT (our signal).
    """
    results = {}
    # Align on common timestamps
    combined = pd.DataFrame({"btc": btc_ret, "alt": alt_ret}).dropna()
    if len(combined) < 10000:
        return results

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = combined["btc"].iloc[:len(combined)-lag].corr(
                combined["alt"].iloc[lag:].reset_index(drop=True)
            )
        else:
            corr = combined["btc"].iloc[-lag:].reset_index(drop=True).corr(
                combined["alt"].iloc[:len(combined)+lag]
            )
        results[lag] = corr

    return results


def backtest_lead_lag_signal(btc_1m, alt_1m, entry_sigma=2.0,
                              hold_bars=5, cost_bps=3.0):
    """Backtest lead-lag strategy on 1-minute data.

    Signal: when BTC 1-min return > entry_sigma * rolling_std,
    go long the alt on the next bar. Hold for hold_bars.
    """
    btc_close = btc_1m["close"]
    alt_close = alt_1m["close"]

    # Align
    common_idx = btc_close.index.intersection(alt_close.index)
    if len(common_idx) < 50000:
        return None

    btc_c = btc_close.reindex(common_idx)
    alt_c = alt_close.reindex(common_idx)

    btc_ret = btc_c.pct_change()
    alt_ret = alt_c.pct_change()

    # Rolling volatility of BTC 1-min returns (60-min window)
    btc_vol = btc_ret.rolling(60).std()

    # Entry signal: BTC return exceeds threshold
    threshold_up = btc_vol * entry_sigma
    threshold_dn = -btc_vol * entry_sigma

    # Generate trades
    trades = []
    i = 0
    n = len(common_idx)
    btc_ret_vals = btc_ret.values
    alt_ret_vals = alt_ret.values
    btc_vol_vals = btc_vol.values

    while i < n - hold_bars - 1:
        if not np.isfinite(btc_vol_vals[i]) or btc_vol_vals[i] < 1e-8:
            i += 1
            continue

        br = btc_ret_vals[i]
        thresh = btc_vol_vals[i] * entry_sigma

        if abs(br) > thresh:
            # Direction: follow BTC
            direction = 1.0 if br > 0 else -1.0

            # Alt return over hold period (entry at i+1, exit at i+1+hold_bars)
            entry_price = alt_c.iloc[i + 1]
            exit_price = alt_c.iloc[min(i + 1 + hold_bars, n - 1)]

            if entry_price > 0:
                alt_trade_ret = (exit_price / entry_price - 1.0) * direction
                alt_trade_ret -= cost_bps / 10000.0  # roundtrip cost

                trades.append({
                    "timestamp": common_idx[i],
                    "direction": direction,
                    "btc_ret": br,
                    "alt_ret": float(alt_trade_ret),
                    "hold_bars": hold_bars,
                })

            # Skip forward to avoid overlapping trades
            i += hold_bars + 1
        else:
            i += 1

    if not trades:
        return None

    df = pd.DataFrame(trades)
    return df


def main():
    t0 = time.time()

    print("=" * 100)
    print("  LEAD-LAG NETWORK DISCOVERY")
    print("  BTC -> Alt cross-correlation at 1-minute resolution")
    print("=" * 100)

    # Load BTC 1-minute data
    print("\nLoading BTC 1-minute data...")
    btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
    btc_ret_1m = btc_1m["close"].pct_change()
    print(f"  BTC: {len(btc_1m)} bars ({btc_1m.index.min()} to {btc_1m.index.max()})")

    # ================================================================
    # Phase 1: Cross-correlation discovery
    # ================================================================
    print(f"\n{'='*80}")
    print("PHASE 1: CROSS-CORRELATION AT 1-MINUTE LAGS")
    print(f"{'='*80}")

    lag_results = {}

    for name, symbol in sorted(ALTS.items()):
        try:
            alt_1m = load_binance_1m(symbol, DATA_DIRS)
            alt_ret = alt_1m["close"].pct_change()
            correlations = compute_lead_lag_correlations(btc_ret_1m, alt_ret, max_lag=10)
            lag_results[name] = correlations
        except Exception as e:
            print(f"  {name}: SKIP ({e})")
            continue

    # Print results table
    print(f"\n{'Asset':<8}", end="")
    for lag in range(-5, 6):
        label = f"k={lag:+d}"
        print(f" {label:>8}", end="")
    print(f" {'Peak_k':>8} {'Peak_r':>8}")
    print("-" * 110)

    pass_count = 0
    for name in sorted(lag_results.keys()):
        corrs = lag_results[name]
        if not corrs:
            continue

        print(f"{name:<8}", end="")
        for lag in range(-5, 6):
            r = corrs.get(lag, float("nan"))
            print(f" {r:>8.4f}", end="")

        # Find peak positive lag (BTC leads alt)
        positive_lags = {k: v for k, v in corrs.items() if k > 0}
        if positive_lags:
            peak_k = max(positive_lags, key=lambda k: abs(positive_lags[k]))
            peak_r = positive_lags[peak_k]
        else:
            peak_k, peak_r = 0, 0

        marker = " *" if peak_r > 0.05 else ""
        print(f" {peak_k:>8d} {peak_r:>8.4f}{marker}")
        if peak_r > 0.05:
            pass_count += 1

    print(f"\n  Assets with peak lag correlation > 0.05: {pass_count}/{len(lag_results)}")

    # ================================================================
    # Phase 2: Signal strength analysis
    # ================================================================
    print(f"\n{'='*80}")
    print("PHASE 2: CONDITIONAL RETURN ANALYSIS")
    print("  When BTC moves > 2-sigma in 1 min, what do alts do next?")
    print(f"{'='*80}")

    # Test: after large BTC moves, do alts follow with a lag?
    btc_vol = btc_ret_1m.rolling(60).std()
    big_up = btc_ret_1m > btc_vol * 2.0
    big_dn = btc_ret_1m < -btc_vol * 2.0

    print(f"\n  BTC big moves (>2 sigma): {big_up.sum()} up, {big_dn.sum()} down")

    print(f"\n{'Asset':<8} {'Fwd1m':>8} {'Fwd2m':>8} {'Fwd3m':>8} {'Fwd5m':>8}"
          f" {'Fwd10m':>8} {'t_1m':>8} {'N':>8}")
    print("-" * 80)

    for name, symbol in sorted(ALTS.items()):
        try:
            alt_1m = load_binance_1m(symbol, DATA_DIRS)
        except:
            continue

        alt_ret = alt_1m["close"].pct_change()

        # Align
        common = big_up.index.intersection(alt_ret.index)
        big_up_aligned = big_up.reindex(common).fillna(False)
        big_dn_aligned = big_dn.reindex(common).fillna(False)
        alt_ret_aligned = alt_ret.reindex(common)

        # Forward alt returns at various horizons after BTC big move
        fwd_rets = {}
        for fwd in [1, 2, 3, 5, 10]:
            alt_fwd = alt_ret_aligned.shift(-fwd)
            # After big up: expect alt to go up (positive following return)
            up_ret = alt_fwd[big_up_aligned].mean()
            # After big down: expect alt to go down (negative following return)
            dn_ret = alt_fwd[big_dn_aligned].mean()
            # Average directional following return
            avg_fwd = (up_ret - dn_ret) / 2.0  # should be positive if lead-lag exists
            fwd_rets[fwd] = avg_fwd * 10000  # convert to bps

        # t-stat for 1-min forward (most relevant)
        alt_fwd_1 = alt_ret_aligned.shift(-1)
        up_rets_1 = alt_fwd_1[big_up_aligned].dropna()
        dn_rets_1 = alt_fwd_1[big_dn_aligned].dropna()
        if len(up_rets_1) > 30 and len(dn_rets_1) > 30:
            combined_rets = pd.concat([up_rets_1, -dn_rets_1])
            t_stat = combined_rets.mean() / (combined_rets.std() / np.sqrt(len(combined_rets)))
            n_events = len(combined_rets)
        else:
            t_stat = 0
            n_events = 0

        print(f"{name:<8} {fwd_rets.get(1,0):>8.2f} {fwd_rets.get(2,0):>8.2f} "
              f"{fwd_rets.get(3,0):>8.2f} {fwd_rets.get(5,0):>8.2f} "
              f"{fwd_rets.get(10,0):>8.2f} {t_stat:>8.2f} {n_events:>8,}")

    # ================================================================
    # Phase 3: Backtest top candidates
    # ================================================================
    print(f"\n{'='*80}")
    print("PHASE 3: BACKTEST (top 10 alts, 2-sigma entry, 3bps cost)")
    print(f"{'='*80}")

    configs = [
        {"sigma": 2.0, "hold": 3, "cost": 3.0},
        {"sigma": 2.0, "hold": 5, "cost": 3.0},
        {"sigma": 3.0, "hold": 5, "cost": 3.0},
        {"sigma": 2.0, "hold": 5, "cost": 0.0},  # zero cost (maker rebate scenario)
    ]

    for cfg in configs:
        print(f"\n  Config: sigma={cfg['sigma']}, hold={cfg['hold']}bars, cost={cfg['cost']}bps")
        print(f"  {'Asset':<8} {'Trades':>8} {'MeanRet':>10} {'SR':>8} {'WinRate':>8} {'t-stat':>8}")
        print(f"  {'-'*60}")

        total_trades = 0
        all_rets = []

        for name, symbol in sorted(ALTS.items()):
            try:
                alt_1m = load_binance_1m(symbol, DATA_DIRS)
            except:
                continue

            trades_df = backtest_lead_lag_signal(
                btc_1m, alt_1m,
                entry_sigma=cfg["sigma"],
                hold_bars=cfg["hold"],
                cost_bps=cfg["cost"],
            )

            if trades_df is None or len(trades_df) < 50:
                print(f"  {name:<8} {'<50 trades':>8}")
                continue

            rets = trades_df["alt_ret"]
            mean_ret = rets.mean() * 10000  # bps
            n_trades = len(trades_df)
            win_rate = (rets > 0).mean() * 100
            t_stat = rets.mean() / (rets.std() / np.sqrt(n_trades)) if rets.std() > 0 else 0

            # Approximate annualized Sharpe
            daily_rets = trades_df.set_index("timestamp")["alt_ret"].resample("1D").sum()
            daily_rets = daily_rets[daily_rets != 0]
            if len(daily_rets) > 30:
                sr = daily_rets.mean() / daily_rets.std() * np.sqrt(365.25) if daily_rets.std() > 0 else 0
            else:
                sr = 0

            total_trades += n_trades
            all_rets.extend(rets.tolist())

            print(f"  {name:<8} {n_trades:>8,} {mean_ret:>10.2f}bps {sr:>8.2f} "
                  f"{win_rate:>7.1f}% {t_stat:>8.2f}")

        if all_rets:
            all_rets = np.array(all_rets)
            agg_mean = all_rets.mean() * 10000
            agg_t = all_rets.mean() / (all_rets.std() / np.sqrt(len(all_rets)))
            agg_win = (all_rets > 0).mean() * 100
            print(f"\n  AGGREGATE: {total_trades:,} trades, mean={agg_mean:.2f}bps, "
                  f"t={agg_t:.2f}, win={agg_win:.1f}%")

    # ================================================================
    # Decision
    # ================================================================
    print(f"\n{'='*80}")
    print("DECISION CRITERIA")
    print(f"{'='*80}")
    print("  - Peak lag correlation > 0.05 for multiple assets? (Phase 1)")
    print("  - Conditional returns > 3 bps at 1-5 min horizon? (Phase 2)")
    print("  - Net SR > 1.0 after costs? (Phase 3)")
    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
