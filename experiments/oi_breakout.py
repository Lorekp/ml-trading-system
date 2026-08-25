"""Experiment 4: OI-Price Divergence Breakout

Hypothesis: When OI rises sharply without price movement, it signals
directional buildup. The resolution (breakout) is predictable.
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


def load_data(coin):
    """Load 15m OHLCV + HL derivatives for one coin."""
    from data.hl_derivatives import load_hl_asset_ctxs, resample_to_ohlcv_freq

    ohlcv_path = PCACHE / f"{coin}USDT.parquet"
    if not ohlcv_path.exists(): return None
    ohlcv = pd.read_parquet(ohlcv_path)
    if len(ohlcv) < 5000: return None

    cache_path = HL_CACHE / f"{coin}.parquet"
    try:
        deriv = load_hl_asset_ctxs(str(HL_DIR), coin=coin, cache_path=str(cache_path))
    except Exception:
        return None
    if len(deriv) < 1000: return None

    deriv_15m = resample_to_ohlcv_freq(deriv, ohlcv.index, method="last")
    return ohlcv, deriv_15m


def analyze_oi_divergence(ohlcv, deriv, coin):
    """Find OI-price divergence events and measure forward returns."""
    close = ohlcv["close"]
    oi = deriv["open_interest"] if "open_interest" in deriv.columns else None
    funding = deriv["funding"] if "funding" in deriv.columns else None

    if oi is None or oi.isna().all(): return None
    if funding is None: funding = pd.Series(0, index=oi.index)

    # OI change (5-bar = 1.25h)
    oi_pct = oi.pct_change(4).clip(-0.5, 0.5)
    oi_mean = oi_pct.rolling(96, min_periods=50).mean()
    oi_std = oi_pct.rolling(96, min_periods=50).std()
    oi_zscore = (oi_pct - oi_mean) / oi_std.replace(0, np.nan)

    # Price change
    price_ret = close.pct_change(4)
    price_mean = price_ret.rolling(96, min_periods=50).mean()
    price_std = price_ret.rolling(96, min_periods=50).std()
    price_zscore = (price_ret - price_mean) / price_std.replace(0, np.nan)

    # Funding z-score
    f_mean = funding.rolling(50, min_periods=20).mean()
    f_std = funding.rolling(50, min_periods=20).std()
    f_zscore = (funding - f_mean) / f_std.replace(0, np.nan)

    results = []
    valid_idx = close.index[200:-24]  # need lookback + forward

    for ts in valid_idx:
        if ts not in oi_zscore.index or ts not in price_zscore.index: continue
        oz = oi_zscore.get(ts, np.nan)
        pz = price_zscore.get(ts, np.nan)
        fz = f_zscore.get(ts, np.nan) if ts in f_zscore.index else 0

        if not np.isfinite(oz) or not np.isfinite(pz): continue

        # Divergence: OI moved significantly but price didn't
        if abs(oz) < 2.0 or abs(pz) > 1.0: continue

        # Forward returns
        idx = close.index.get_loc(ts)
        if idx + 24 >= len(close): continue

        fwd_2h = float(close.iloc[idx + 8] / close.iloc[idx] - 1)
        fwd_4h = float(close.iloc[idx + 16] / close.iloc[idx] - 1)
        fwd_6h = float(close.iloc[idx + 24] / close.iloc[idx] - 1)

        results.append({
            "time": ts, "coin": coin,
            "oi_zscore": float(oz), "price_zscore": float(pz),
            "funding_zscore": float(fz) if np.isfinite(fz) else 0,
            "oi_direction": 1 if oz > 0 else -1,  # 1=OI rising, -1=OI falling
            "fwd_2h": fwd_2h, "fwd_4h": fwd_4h, "fwd_6h": fwd_6h,
        })

    return pd.DataFrame(results) if results else None


def main():
    t0 = time.time()
    print("=" * 110)
    print("OI-PRICE DIVERGENCE BREAKOUT")
    print("=" * 110)

    print("\nLoading data...", flush=True)
    all_data = []
    for coin in ASSETS:
        result = load_data(coin)
        if result is None:
            print(f"  {coin}: SKIP", flush=True); continue
        ohlcv, deriv = result
        df = analyze_oi_divergence(ohlcv, deriv, coin)
        if df is not None and len(df) > 30:
            all_data.append(df)
            print(f"  {coin}: {len(df)} divergence events", flush=True)
        else:
            print(f"  {coin}: insufficient events", flush=True)

    if not all_data:
        print("NO DATA"); return

    df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(df)} divergence events across {df['coin'].nunique()} assets")

    # ---- Signal Analysis ----
    print(f"\n{'='*110}")
    print("SIGNAL: Does OI divergence predict breakout direction?")
    print(f"{'='*110}")

    # Split by OI direction
    print(f"\n--- By OI Direction ---")
    print(f"{'OI Direction':<15s} {'N':>6s} {'Fwd2h%':>8s} {'Fwd4h%':>8s} {'Fwd6h%':>8s} {'t(4h)':>7s}")
    print("-" * 60)

    for oi_dir, label in [(1, "OI Rising"), (-1, "OI Falling")]:
        sub = df[df["oi_direction"] == oi_dir]
        if len(sub) < 20: continue
        r2 = sub["fwd_2h"].mean() * 100; r4 = sub["fwd_4h"].mean() * 100; r6 = sub["fwd_6h"].mean() * 100
        t4 = r4 / (sub["fwd_4h"].std() * 100 / np.sqrt(len(sub))) if sub["fwd_4h"].std() > 0 else 0
        sig = '***' if abs(t4) > 2.58 else ('**' if abs(t4) > 1.96 else ('*' if abs(t4) > 1.65 else ''))
        print(f"{label:<15s} {len(sub):>6d} {r2:>+7.3f}% {r4:>+7.3f}% {r6:>+7.3f}% {t4:>+6.2f}{sig}")

    # Split by OI direction AND funding sign
    print(f"\n--- By OI Direction + Funding Sign ---")
    print(f"{'Condition':<30s} {'N':>6s} {'Fwd4h%':>8s} {'t(4h)':>7s} {'WR':>5s}")
    print("-" * 65)

    conditions = [
        ("OI rise + pos funding", (df["oi_direction"] == 1) & (df["funding_zscore"] > 0.5)),
        ("OI rise + neg funding", (df["oi_direction"] == 1) & (df["funding_zscore"] < -0.5)),
        ("OI fall + pos funding", (df["oi_direction"] == -1) & (df["funding_zscore"] > 0.5)),
        ("OI fall + neg funding", (df["oi_direction"] == -1) & (df["funding_zscore"] < -0.5)),
        ("OI rise strong (z>3)", df["oi_zscore"] > 3),
        ("OI fall strong (z<-3)", df["oi_zscore"] < -3),
    ]

    for label, mask in conditions:
        sub = df[mask]
        if len(sub) < 20: continue
        r4 = sub["fwd_4h"].mean() * 100
        t4 = r4 / (sub["fwd_4h"].std() * 100 / np.sqrt(len(sub))) if sub["fwd_4h"].std() > 0 else 0
        wr = (sub["fwd_4h"] > 0).mean() * 100
        sig = '***' if abs(t4) > 2.58 else ('**' if abs(t4) > 1.96 else ('*' if abs(t4) > 1.65 else ''))
        print(f"{label:<30s} {len(sub):>6d} {r4:>+7.3f}% {t4:>+6.2f}{sig} {wr:>4.0f}%")

    # Trade the OI direction (breakout hypothesis)
    print(f"\n--- Breakout Trade: Enter in OI direction ---")
    df["trade_ret"] = df["oi_direction"] * df["fwd_4h"]
    overall_ret = df["trade_ret"].mean() * 100
    overall_t = overall_ret / (df["trade_ret"].std() * 100 / np.sqrt(len(df))) if df["trade_ret"].std() > 0 else 0
    wr = (df["trade_ret"] > 0).mean() * 100
    print(f"  Direction = OI direction, hold 4h")
    print(f"  Avg return: {overall_ret:+.3f}%, t={overall_t:.2f}, WR={wr:.0f}%, N={len(df)}")

    # Contrarian trade (OI falling = shorts unwinding = price goes up)
    print(f"\n--- Contrarian Trade: Enter opposite to OI direction ---")
    df["contra_ret"] = -df["oi_direction"] * df["fwd_4h"]
    contra_ret = df["contra_ret"].mean() * 100
    contra_t = contra_ret / (df["contra_ret"].std() * 100 / np.sqrt(len(df))) if df["contra_ret"].std() > 0 else 0
    wr_c = (df["contra_ret"] > 0).mean() * 100
    print(f"  Avg return: {contra_ret:+.3f}%, t={contra_t:.2f}, WR={wr_c:.0f}%, N={len(df)}")

    # ---- Per-asset breakdown ----
    print(f"\n--- Per-Asset: Trade in OI direction (4h) ---")
    print(f"{'Asset':<8s} {'N':>6s} {'Avg%':>8s} {'t':>7s} {'WR':>5s}")
    print("-" * 40)

    for coin in sorted(df["coin"].unique()):
        cdf = df[df["coin"] == coin]
        if len(cdf) < 10: continue
        ret = cdf["trade_ret"].mean() * 100
        t = ret / (cdf["trade_ret"].std() * 100 / np.sqrt(len(cdf))) if cdf["trade_ret"].std() > 0 else 0
        wr = (cdf["trade_ret"] > 0).mean() * 100
        sig = '*' if abs(t) > 1.65 else ''
        print(f"{coin:<8s} {len(cdf):>6d} {ret:>+7.3f}% {t:>+6.2f}{sig} {wr:>4.0f}%")

    # ---- OOS walk-forward ----
    print(f"\n{'='*110}")
    print("WALK-FORWARD (IS: pre-2025, OOS: 2025+)")
    print(f"{'='*110}")

    oos_start = pd.Timestamp("2025-01-01", tz="UTC")
    capital = 10000.0; cost_bps = 5.0

    for size_pct in [0.03, 0.05]:
        for direction in ["oi", "contra"]:
            for period_label, pmask in [("IS", df["time"] < oos_start), ("OOS", df["time"] >= oos_start)]:
                sub = df[pmask]
                if len(sub) < 30: continue

                daily_pnl = {}; n_trades = 0
                for _, row in sub.iterrows():
                    d = row["oi_direction"] if direction == "oi" else -row["oi_direction"]
                    ret = row["fwd_4h"]
                    notional = capital * size_pct
                    pnl = d * ret * notional - cost_bps / 10000 * notional * 2
                    day = pd.Timestamp(row["time"]).normalize()
                    if hasattr(day, "tz") and day.tz is not None: day = day.tz_localize(None)
                    daily_pnl[day] = daily_pnl.get(day, 0) + pnl
                    n_trades += 1

                dpnl = pd.Series(daily_pnl).sort_index()
                if len(dpnl) < 10: continue
                cal = (dpnl.index[-1] - dpnl.index[0]).days + 1
                dr = dpnl / capital
                sr = dr.mean() / dr.std() * np.sqrt(365.25) if dr.std() > 0 else 0
                pnl_cal = dpnl.sum() / cal

                print(f"  {direction}_{size_pct*100:.0f}% [{period_label}] "
                      f"SR={sr:>5.2f} ${pnl_cal:>6.2f}/cal Tr={n_trades:>5d}")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
