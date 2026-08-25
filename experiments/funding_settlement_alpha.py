"""Experiment 3: Funding Settlement Micro-Alpha

Hypothesis: HL funding settles every 8 hours. Extreme funding creates
predictable price pressure around settlement times. We test:
1. Does extreme funding predict post-settlement price move?
2. Is there a tradeable edge in the 2h window around settlement?
"""

import sys, os, time, warnings
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PCACHE = Path("data/cache_15m")
HL_DIR = Path("data/hl-asset-ctxs/hl-asset-ctxs")
HL_CACHE = Path("data/hl-asset-ctxs/cache_15m")
HL_CACHE.mkdir(parents=True, exist_ok=True)

ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK", "APT",
          "SUI", "ADA", "BNB", "NEAR", "DOT", "FIL", "ARB", "FET",
          "WLD", "UNI", "AAVE", "FTM"]

# Settlement hours on HL
SETTLEMENT_HOURS = [0, 8, 16]


def load_data(coin):
    """Load 15m OHLCV + HL derivatives for one coin."""
    from data.hl_derivatives import load_hl_asset_ctxs, resample_to_ohlcv_freq

    # OHLCV
    ohlcv_path = PCACHE / f"{coin}USDT.parquet"
    if not ohlcv_path.exists(): return None
    ohlcv = pd.read_parquet(ohlcv_path)
    if len(ohlcv) < 5000: return None

    # Derivatives
    cache_path = HL_CACHE / f"{coin}.parquet"
    try:
        deriv = load_hl_asset_ctxs(str(HL_DIR), coin=coin, cache_path=str(cache_path))
    except Exception:
        return None
    if len(deriv) < 1000: return None

    # Align derivatives to 15m bars
    deriv_15m = resample_to_ohlcv_freq(deriv, ohlcv.index, method="last")

    return ohlcv, deriv_15m


def analyze_settlement_alpha(ohlcv, deriv, coin):
    """Analyze price behavior around funding settlements."""
    close = ohlcv["close"]
    funding = deriv["funding"] if "funding" in deriv.columns else None
    if funding is None or funding.isna().all():
        return None

    # Funding z-score (50-bar rolling = ~12.5 hours)
    f_mean = funding.rolling(50, min_periods=20).mean()
    f_std = funding.rolling(50, min_periods=20).std()
    f_zscore = (funding - f_mean) / f_std.replace(0, np.nan)

    # Find settlement bars (within 15min of settlement hour)
    settlement_mask = close.index.hour.isin(SETTLEMENT_HOURS) & (close.index.minute == 0)
    settlement_times = close.index[settlement_mask]

    if len(settlement_times) < 100:
        return None

    results = []
    for st in settlement_times:
        # Get funding z-score at settlement
        if st not in f_zscore.index or pd.isna(f_zscore.loc[st]):
            continue
        fz = float(f_zscore.loc[st])
        fr = float(funding.loc[st]) if st in funding.index else 0

        # Forward returns at various horizons (in 15m bars)
        idx = close.index.get_loc(st)
        if idx + 8 >= len(close) or idx < 8:
            continue

        # Pre-settlement returns (T-2h to T)
        pre_ret = float(close.iloc[idx] / close.iloc[idx - 8] - 1)
        # Post-settlement returns
        ret_1h = float(close.iloc[idx + 4] / close.iloc[idx] - 1)
        ret_2h = float(close.iloc[idx + 8] / close.iloc[idx] - 1)
        ret_30m = float(close.iloc[idx + 2] / close.iloc[idx] - 1) if idx + 2 < len(close) else 0

        results.append({
            "time": st, "coin": coin,
            "funding_zscore": fz, "funding_raw": fr,
            "pre_ret": pre_ret,
            "ret_30m": ret_30m, "ret_1h": ret_1h, "ret_2h": ret_2h,
        })

    return pd.DataFrame(results) if results else None


def main():
    t0 = time.time()
    print("=" * 110)
    print("FUNDING SETTLEMENT MICRO-ALPHA")
    print("=" * 110)

    # Load data
    print("\nLoading data...", flush=True)
    all_data = []
    for coin in ASSETS:
        result = load_data(coin)
        if result is None:
            print(f"  {coin}: SKIP", flush=True)
            continue
        ohlcv, deriv = result
        df = analyze_settlement_alpha(ohlcv, deriv, coin)
        if df is not None and len(df) > 50:
            all_data.append(df)
            print(f"  {coin}: {len(df)} settlements", flush=True)
        else:
            print(f"  {coin}: insufficient data", flush=True)

    if not all_data:
        print("NO DATA"); return

    df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(df)} settlement events across {df['coin'].nunique()} assets")

    # ---- Signal Analysis ----
    print(f"\n{'='*110}")
    print("SIGNAL ANALYSIS: Does extreme funding predict post-settlement returns?")
    print(f"{'='*110}")

    # Quintile analysis by funding z-score
    bins = [(-np.inf, -3), (-3, -2), (-2, -1), (-1, 1), (1, 2), (2, 3), (3, np.inf)]
    labels = ["< -3", "-3 to -2", "-2 to -1", "-1 to 1", "1 to 2", "2 to 3", "> 3"]

    print(f"\n{'Funding Z':<12s} {'N':>6s} {'Pre2h%':>8s} {'Post30m%':>9s} {'Post1h%':>8s} {'Post2h%':>8s} {'t(1h)':>7s} {'t(2h)':>7s}")
    print("-" * 75)

    for (lo, hi), label in zip(bins, labels):
        mask = (df["funding_zscore"] >= lo) & (df["funding_zscore"] < hi)
        sub = df[mask]
        if len(sub) < 20:
            print(f"{label:<12s} {len(sub):>6d}   insufficient"); continue

        pre = sub["pre_ret"].mean() * 100
        r30 = sub["ret_30m"].mean() * 100
        r1h = sub["ret_1h"].mean() * 100
        r2h = sub["ret_2h"].mean() * 100
        t1h = r1h / (sub["ret_1h"].std() * 100 / np.sqrt(len(sub))) if sub["ret_1h"].std() > 0 else 0
        t2h = r2h / (sub["ret_2h"].std() * 100 / np.sqrt(len(sub))) if sub["ret_2h"].std() > 0 else 0
        sig1 = '***' if abs(t1h) > 2.58 else ('**' if abs(t1h) > 1.96 else ('*' if abs(t1h) > 1.65 else ''))
        sig2 = '***' if abs(t2h) > 2.58 else ('**' if abs(t2h) > 1.96 else ('*' if abs(t2h) > 1.65 else ''))

        print(f"{label:<12s} {len(sub):>6d} {pre:>+7.3f}% {r30:>+8.3f}% {r1h:>+7.3f}% {r2h:>+7.3f}% "
              f"{t1h:>+6.2f}{sig1} {t2h:>+6.2f}{sig2}")

    # Contrarian signal: extreme funding → trade opposite
    print(f"\n--- Contrarian Edge ---")
    extreme_neg = df[df["funding_zscore"] < -2]
    extreme_pos = df[df["funding_zscore"] > 2]

    if len(extreme_neg) >= 20 and len(extreme_pos) >= 20:
        # Long when funding extremely negative (shorts paying longs)
        long_ret = extreme_neg["ret_2h"].mean()
        # Short when funding extremely positive (longs paying shorts)
        short_ret = -extreme_pos["ret_2h"].mean()
        spread = long_ret + short_ret

        t_long = long_ret / (extreme_neg["ret_2h"].std() / np.sqrt(len(extreme_neg))) if extreme_neg["ret_2h"].std() > 0 else 0
        t_short = -extreme_pos["ret_2h"].mean() / (extreme_pos["ret_2h"].std() / np.sqrt(len(extreme_pos))) if extreme_pos["ret_2h"].std() > 0 else 0

        print(f"  LONG on extreme neg funding (z<-2): avg 2h ret = {long_ret*100:+.3f}% (N={len(extreme_neg)}, t={t_long:.2f})")
        print(f"  SHORT on extreme pos funding (z>2): avg 2h ret = {short_ret*100:+.3f}% (N={len(extreme_pos)}, t={t_short:.2f})")
        print(f"  Combined spread: {spread*100:+.3f}% per settlement")
        print(f"  Annualized (3 settlements/day × 365): {spread*3*365*100:.1f}%")

    # Per-asset breakdown
    print(f"\n--- Per-Asset: Extreme Funding Contrarian (|z|>2, 2h return) ---")
    print(f"{'Asset':<8s} {'N_neg':>6s} {'Long2h%':>8s} {'N_pos':>6s} {'Short2h%':>9s} {'Spread%':>8s}")
    print("-" * 55)

    for coin in sorted(df["coin"].unique()):
        cdf = df[df["coin"] == coin]
        neg = cdf[cdf["funding_zscore"] < -2]
        pos = cdf[cdf["funding_zscore"] > 2]
        if len(neg) < 5 or len(pos) < 5: continue
        lr = neg["ret_2h"].mean() * 100
        sr = -pos["ret_2h"].mean() * 100
        print(f"{coin:<8s} {len(neg):>6d} {lr:>+7.3f}% {len(pos):>6d} {sr:>+8.3f}% {lr+sr:>+7.3f}%")

    # ---- Walk-forward backtest ----
    print(f"\n{'='*110}")
    print("WALK-FORWARD BACKTEST")
    print(f"{'='*110}")

    # Split: pre-2025 = IS, 2025+ = OOS
    df["time"] = pd.to_datetime(df["time"])
    oos_start = pd.Timestamp("2025-01-01", tz="UTC")

    configs = [
        ("z>2 contrarian 2h", 2.0, 8),   # 8 bars = 2h hold
        ("z>2 contrarian 1h", 2.0, 4),
        ("z>1.5 contrarian 2h", 1.5, 8),
        ("z>3 contrarian 2h", 3.0, 8),
        ("z>2 contrarian 4h", 2.0, 16),
    ]

    capital = 10000.0; cost_bps = 5.0; base_size = 0.03

    for name, threshold, hold_bars in configs:
        # Simulate: at each settlement, if |funding_z| > threshold,
        # trade contrarian direction for hold_bars
        for period, period_label in [(None, "FULL"), (oos_start, "OOS")]:
            sub = df.copy()
            if period is not None:
                sub = sub[sub["time"] >= period]
            if len(sub) < 50: continue

            daily_pnl = {}; n_trades = 0

            for _, row in sub.iterrows():
                fz = row["funding_zscore"]
                if abs(fz) < threshold: continue

                # Direction: contrarian to funding
                direction = 1.0 if fz < 0 else -1.0

                # Return based on hold period
                if hold_bars == 4:
                    ret = row["ret_30m"] if hold_bars == 2 else row["ret_1h"]
                elif hold_bars == 8:
                    ret = row["ret_2h"]
                elif hold_bars == 16:
                    ret = row["ret_2h"]  # approximate
                else:
                    ret = row["ret_1h"]

                notional = capital * base_size
                pnl = direction * ret * notional
                cost = cost_bps / 10000 * notional * 2  # roundtrip
                pnl -= cost

                day = pd.Timestamp(row["time"]).normalize()
                if hasattr(day, "tz") and day.tz is not None:
                    day = day.tz_localize(None)
                daily_pnl[day] = daily_pnl.get(day, 0) + pnl
                n_trades += 1

            if n_trades < 10: continue
            dpnl = pd.Series(daily_pnl).sort_index()
            cal = (dpnl.index[-1] - dpnl.index[0]).days + 1
            dr = dpnl / capital
            sr = dr.mean() / dr.std() * np.sqrt(365.25) if dr.std() > 0 else 0
            pnl_cal = dpnl.sum() / cal

            print(f"  {name:<25s} [{period_label}] SR={sr:>5.2f} ${pnl_cal:>6.2f}/cal "
                  f"Tr={n_trades:>5d} WR={(dpnl>0).sum()/max((dpnl!=0).sum(),1)*100:.0f}%")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
