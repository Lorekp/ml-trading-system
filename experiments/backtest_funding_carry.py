"""Funding Rate Carry Strategy — Walk-Forward Backtest

Hypothesis: Extreme funding rates create structural mean-reversion (Ackerer 2025).
When one side is overcrowded (paying unsustainable carry), we trade the opposite
direction to capture convergence + carry income.

Methodology:
1. Load per-asset funding data from HL asset_ctxs (1-min, May 2023+)
2. Resample to 4H, compute funding_zscore
3. Entry: zscore > threshold -> SHORT, zscore < -threshold -> LONG
4. Exit: |zscore| < exit_threshold OR max_hold OR stop_loss
5. Walk-forward: W1 (param select), W2+W3 (OOS validation)
6. Portfolio sim with position sizing and DD limits

HARD GATE: OOS Sharpe < 1.0 -> KILL strategy.
"""

import sys, os, time, warnings
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from data.hl_derivatives import load_hl_asset_ctxs, resample_to_ohlcv_freq
from signal_engine.data import load_binance_1m, resample_ohlcv
from afml.derivatives import funding_features

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/binance-1m")
HL_DIR = Path("data/hl-asset-ctxs/hl-asset-ctxs")
CACHE_DIR = Path("data/hl-asset-ctxs/cache_15m")

# Assets validated in previous research (SOL, BTC, LINK, XRP top performers)
# FARTCOIN not in HL dataset (too new). Test broader set.
TOP_ASSETS = ["SOL", "BTC", "LINK", "XRP"]
BROAD_ASSETS = [
    "SOL", "BTC", "LINK", "XRP", "ETH", "AVAX", "DOGE", "SUI",
    "APT", "FIL", "INJ", "NEAR", "ARB", "OP", "WLD", "FET",
    "TIA", "SEI", "JUP", "PENDLE",
]

WINDOWS = [
    {"label": "W1", "start": "2024-07-01", "end": "2025-02-28", "cal_days": 245},
    {"label": "W2", "start": "2025-03-01", "end": "2025-10-31", "cal_days": 244},
    {"label": "W3", "start": "2025-11-01", "end": "2026-02-28", "cal_days": 122},
]


# ============================================================
# Data Loading
# ============================================================

def load_asset_data(coin: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load 4H OHLCV + aligned funding data for one coin.

    Returns (ohlcv_4h, deriv_4h) or None if data insufficient.
    """
    # Load OHLCV
    symbol = f"{coin}USDT"
    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    except (FileNotFoundError, Exception):
        return None
    if len(ohlcv_4h) < 500:
        return None

    # Load derivatives (with parquet cache)
    cache_path = CACHE_DIR / f"{coin}.parquet"
    try:
        deriv = load_hl_asset_ctxs(str(HL_DIR), coin=coin, cache_path=str(cache_path))
    except (FileNotFoundError, ValueError, Exception):
        return None
    if len(deriv) < 1000:
        return None

    # Align derivatives to 4H bars (ffill = backward-looking, no leak)
    deriv_4h = resample_to_ohlcv_freq(deriv, ohlcv_4h.index, method="last")

    return ohlcv_4h, deriv_4h


# ============================================================
# Signal Generation
# ============================================================

def generate_signals(
    ohlcv_4h: pd.DataFrame,
    deriv_4h: pd.DataFrame,
    coin: str,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    max_hold_bars: int = 6,
    stop_loss_pct: float = 0.05,
    zscore_window: int = 50,
    start_date: str = "2024-01-01",
    end_date: str = "2026-03-01",
) -> pd.DataFrame:
    """Generate funding carry signals for one asset.

    Returns DataFrame with columns: timestamp, asset, direction,
    fwd_return, signal_strength, confidence, hold_bars.
    """
    # Compute funding features
    if "funding" not in deriv_4h.columns or deriv_4h["funding"].isna().all():
        return pd.DataFrame()

    feats = funding_features(deriv_4h["funding"], window_short=zscore_window)
    zscore = feats["funding_zscore"]

    close = ohlcv_4h["close"]
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    # Filter to date range
    mask = (close.index >= start) & (close.index <= end)
    dates = close.index[mask]

    signals = []
    in_trade = False
    entry_bar = None
    entry_price = None
    entry_direction = None
    entry_time = None

    for i, dt in enumerate(dates):
        if dt not in zscore.index or pd.isna(zscore.loc[dt]):
            continue
        z = float(zscore.loc[dt])
        price = float(close.loc[dt])

        if in_trade:
            bars_held = i - entry_bar
            unrealized = entry_direction * (price - entry_price) / entry_price

            # Exit conditions
            should_exit = False
            if abs(z) < exit_zscore:
                should_exit = True  # Funding normalized
            elif bars_held >= max_hold_bars:
                should_exit = True  # Time limit
            elif unrealized < -stop_loss_pct:
                should_exit = True  # Stop loss

            if should_exit:
                fwd_return = entry_direction * (price - entry_price) / entry_price
                signals.append({
                    "timestamp": entry_time,
                    "asset": coin,
                    "direction": entry_direction,
                    "fwd_return": fwd_return,
                    "signal_strength": min(abs(float(zscore.loc[entry_time])) / 4.0, 1.0),
                    "confidence": min(abs(float(zscore.loc[entry_time])) / 4.0, 1.0),
                    "probability": 0.5 + 0.1 * min(abs(float(zscore.loc[entry_time])) / 4.0, 1.0),
                    "hold_bars": bars_held,
                    "exit_reason": "normalize" if abs(z) < exit_zscore else
                                  "timeout" if bars_held >= max_hold_bars else "stop",
                })
                in_trade = False

        # Entry: only at settlement bars (hour % 8 == 0) when not in trade
        if not in_trade and dt.hour % 8 == 0:
            if z > entry_zscore:
                # Extreme positive funding -> SHORT (collect carry + convergence)
                in_trade = True
                entry_bar = i
                entry_price = price
                entry_direction = -1.0
                entry_time = dt
            elif z < -entry_zscore:
                # Extreme negative funding -> LONG (collect carry + convergence)
                in_trade = True
                entry_bar = i
                entry_price = price
                entry_direction = 1.0
                entry_time = dt

    return pd.DataFrame(signals) if signals else pd.DataFrame()


# ============================================================
# Portfolio Simulation (adapted for carry signals)
# ============================================================

def simulate_carry_portfolio(
    all_signals: pd.DataFrame,
    window: dict,
    starting_capital: float = 10000.0,
    max_per_trade: float = 0.05,
    max_exposure: float = 0.30,
    slippage_bps: float = 3.0,
    leverage: float = 1.0,
) -> dict | None:
    """Simulate portfolio on carry signals for one window."""
    ws = pd.Timestamp(window["start"], tz="UTC")
    we = pd.Timestamp(window["end"], tz="UTC")

    # Filter signals to window
    if "timestamp" not in all_signals.columns or len(all_signals) == 0:
        return None
    sigs = all_signals[
        (all_signals["timestamp"] >= ws) & (all_signals["timestamp"] <= we)
    ].sort_values("timestamp").reset_index(drop=True)

    if len(sigs) < 5:
        return None

    # Simple per-trade simulation
    equity = starting_capital
    equity_curve = [equity]
    daily_pnl = {}
    n_trades = 0
    n_wins = 0

    for _, row in sigs.iterrows():
        size_pct = max_per_trade * row.get("signal_strength", 0.5) * leverage
        size_usd = equity * size_pct

        # Slippage
        slip = slippage_bps / 10000.0 * size_usd * 2  # roundtrip

        pnl = size_usd * row["fwd_return"] - slip
        equity += pnl

        day = pd.Timestamp(row["timestamp"]).normalize()
        daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl

        n_trades += 1
        if pnl > 0:
            n_wins += 1

        equity_curve.append(equity)

        # DD halt
        peak = max(equity_curve)
        if (equity - peak) / peak < -0.10:
            break

    if n_trades < 3:
        return None

    total_pnl = equity - starting_capital
    pnl_cal = total_pnl / window["cal_days"]

    # Daily metrics
    dpnl = pd.Series(daily_pnl)
    daily_ret = dpnl / starting_capital
    sharpe = 0.0
    if len(daily_ret) > 5 and daily_ret.std() > 0:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(365.25)

    eq_arr = np.array(equity_curve)
    peak_arr = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak_arr) / peak_arr
    max_dd = abs(dd.min())
    worst_day = abs(daily_ret.min()) if len(daily_ret) > 0 else 0

    # Exit reason breakdown
    exits = sigs["exit_reason"].value_counts().to_dict() if "exit_reason" in sigs.columns else {}

    return {
        "pnl": total_pnl,
        "pnl_cal": pnl_cal,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "worst_day": worst_day,
        "n_trades": n_trades,
        "win_rate": n_wins / n_trades * 100 if n_trades > 0 else 0,
        "exits": exits,
    }


# ============================================================
# Signal Quality Analysis
# ============================================================

def analyze_funding_signal(
    all_data: dict[str, tuple],
    assets: list[str],
    zscore_window: int = 50,
):
    """Analyze whether extreme funding predicts returns."""
    print("\n" + "=" * 80)
    print("SIGNAL QUALITY: Does extreme funding predict returns?")
    print("=" * 80)

    all_z = []
    all_fwd = []

    for coin in assets:
        if coin not in all_data:
            continue
        ohlcv, deriv = all_data[coin]
        feats = funding_features(deriv["funding"], window_short=zscore_window)
        zscore = feats["funding_zscore"].dropna()
        close = ohlcv["close"]

        # Forward 7-day return at each point
        fwd_ret = close.pct_change(42).shift(-42)  # 42 bars * 4h = 7 days

        common = zscore.index.intersection(fwd_ret.dropna().index)
        if len(common) < 100:
            continue

        z = zscore.reindex(common)
        fr = fwd_ret.reindex(common)
        all_z.extend(z.values)
        all_fwd.extend(fr.values)

    if len(all_z) < 500:
        print("Insufficient data for signal analysis")
        return

    z_arr = np.array(all_z)
    fwd_arr = np.array(all_fwd)

    # Quintile analysis
    print(f"\n{'Zscore Bin':<20s} {'N':>6s} {'Fwd 7d ret':>12s} {'Std':>8s} {'t-stat':>8s} {'Win%':>6s}")
    print("-" * 65)

    bins = [(-np.inf, -3), (-3, -2), (-2, -1), (-1, 1), (1, 2), (2, 3), (3, np.inf)]
    labels = ["< -3", "-3 to -2", "-2 to -1", "-1 to 1", "1 to 2", "2 to 3", "> 3"]

    for (lo, hi), label in zip(bins, labels):
        mask = (z_arr >= lo) & (z_arr < hi)
        n = mask.sum()
        if n < 20:
            print(f"{label:<20s} {n:>6d}   insufficient")
            continue
        fwd = fwd_arr[mask]
        mean_r = fwd.mean()
        std_r = fwd.std()
        t = mean_r / (std_r / np.sqrt(n)) if std_r > 0 else 0
        wr = (fwd > 0).mean() * 100
        sig = '***' if abs(t) > 2.58 else ('**' if abs(t) > 1.96 else ('*' if abs(t) > 1.65 else ''))
        print(f"{label:<20s} {n:>6d} {mean_r*100:>11.3f}% {std_r*100:>7.2f}% {t:>7.2f}{sig} {wr:>5.1f}%")

    # Contrarian signal: negative zscore -> positive fwd return?
    extreme_neg = z_arr < -2
    extreme_pos = z_arr > 2
    if extreme_neg.sum() > 20 and extreme_pos.sum() > 20:
        neg_ret = fwd_arr[extreme_neg].mean()
        pos_ret = fwd_arr[extreme_pos].mean()
        print(f"\nContrarian edge:")
        print(f"  Extreme neg funding (z<-2): fwd 7d = {neg_ret*100:+.3f}% (LONG these)")
        print(f"  Extreme pos funding (z>2):  fwd 7d = {pos_ret*100:+.3f}% (SHORT these)")
        print(f"  Spread: {(neg_ret - pos_ret)*100:.3f}% per 7d")


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 100)
    print("FUNDING RATE CARRY STRATEGY -- WALK-FORWARD BACKTEST")
    print("=" * 100)

    # ---- Load all asset data ----
    print("\nLoading data (first load caches to parquet)...", flush=True)
    all_data: dict[str, tuple] = {}
    for coin in BROAD_ASSETS:
        result = load_asset_data(coin)
        if result is not None:
            all_data[coin] = result
            print(f"  {coin}: OHLCV {len(result[0])} bars, deriv {len(result[1])} rows", flush=True)
        else:
            print(f"  {coin}: SKIP (no data)", flush=True)
    print(f"\nLoaded {len(all_data)} assets in {time.time()-t0:.0f}s")

    if len(all_data) < 3:
        print("ERROR: insufficient assets loaded")
        return

    # ---- Phase 1: Signal Quality ----
    analyze_funding_signal(all_data, list(all_data.keys()))

    # ---- Phase 2: Parameter Sweep ----
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP (W1 = in-sample, W2+W3 = out-of-sample)")
    print("=" * 80)

    configs = [
        # (label, entry_z, exit_z, max_hold, stop, assets, leverage)
        ("Top4 z2.0 ex0.5 h6 sl5%", 2.0, 0.5, 6, 0.05, TOP_ASSETS, 1.0),
        ("Top4 z1.5 ex0.5 h6 sl5%", 1.5, 0.5, 6, 0.05, TOP_ASSETS, 1.0),
        ("Top4 z2.5 ex0.5 h6 sl5%", 2.5, 0.5, 6, 0.05, TOP_ASSETS, 1.0),
        ("Top4 z3.0 ex0.5 h6 sl5%", 3.0, 0.5, 6, 0.05, TOP_ASSETS, 1.0),
        ("Top4 z2.0 ex0.3 h6 sl5%", 2.0, 0.3, 6, 0.05, TOP_ASSETS, 1.0),
        ("Top4 z2.0 ex0.8 h6 sl5%", 2.0, 0.8, 6, 0.05, TOP_ASSETS, 1.0),
        ("Top4 z2.0 ex0.5 h12 sl5%", 2.0, 0.5, 12, 0.05, TOP_ASSETS, 1.0),
        ("Top4 z2.0 ex0.5 h18 sl5%", 2.0, 0.5, 18, 0.05, TOP_ASSETS, 1.0),
        ("Top4 z2.0 ex0.5 h6 sl3%", 2.0, 0.5, 6, 0.03, TOP_ASSETS, 1.0),
        ("Top4 z2.0 ex0.5 h6 sl8%", 2.0, 0.5, 6, 0.08, TOP_ASSETS, 1.0),
        ("Broad z2.0 ex0.5 h6 sl5%", 2.0, 0.5, 6, 0.05, BROAD_ASSETS, 1.0),
        ("Top4 z2.0 2x lev", 2.0, 0.5, 6, 0.05, TOP_ASSETS, 2.0),
        ("Top4 z2.0 3x lev", 2.0, 0.5, 6, 0.05, TOP_ASSETS, 3.0),
        ("Top4 z2.0 5x lev", 2.0, 0.5, 6, 0.05, TOP_ASSETS, 5.0),
    ]

    print(f"\n{'Config':<30s} {'$/cal':>7s} {'Sharpe':>7s} {'DD%':>6s} {'WD%':>6s} "
          f"{'WR%':>5s} {'#Tr':>5s}  W1$/d  W2$/d  W3$/d  Status")
    print("-" * 120)

    for label, entry_z, exit_z, max_hold, stop, assets, lev in configs:
        # Generate signals for all assets
        all_sigs = []
        for coin in assets:
            if coin not in all_data:
                continue
            ohlcv, deriv = all_data[coin]
            sigs = generate_signals(
                ohlcv, deriv, coin,
                entry_zscore=entry_z, exit_zscore=exit_z,
                max_hold_bars=max_hold, stop_loss_pct=stop,
            )
            if len(sigs) > 0:
                all_sigs.append(sigs)

        if not all_sigs:
            print(f"{label:<30s} NO SIGNALS")
            continue

        combined = pd.concat(all_sigs, ignore_index=True)

        # Run per-window
        tp = tc = tt = 0
        mdd = mwd = 0.0
        total_wr = 0
        per_window = {}

        for w in WINDOWS:
            r = simulate_carry_portfolio(combined, w, leverage=lev)
            if r is None:
                per_window[w["label"]] = 0
                continue
            per_window[w["label"]] = r["pnl_cal"]
            tp += r["pnl"]
            tc += w["cal_days"]
            tt += r["n_trades"]
            total_wr += r["win_rate"] * r["n_trades"]
            mdd = max(mdd, r["max_dd"])
            mwd = max(mwd, r["worst_day"])

        if tc == 0:
            print(f"{label:<30s} NO DATA")
            continue

        pnl_cal = tp / tc
        wr = total_wr / tt if tt > 0 else 0

        # Compute aggregate Sharpe from all windows
        all_daily = {}
        for w in WINDOWS:
            ws = pd.Timestamp(w["start"], tz="UTC")
            we = pd.Timestamp(w["end"], tz="UTC")
            wsigs = combined[
                (combined["timestamp"] >= ws) & (combined["timestamp"] <= we)
            ]
            for _, row in wsigs.iterrows():
                day = pd.Timestamp(row["timestamp"]).normalize()
                pnl = 10000.0 * 0.05 * row.get("signal_strength", 0.5) * lev * row["fwd_return"]
                all_daily[day] = all_daily.get(day, 0.0) + pnl
        if all_daily:
            dr = pd.Series(all_daily) / 10000.0
            agg_sharpe = dr.mean() / dr.std() * np.sqrt(365.25) if dr.std() > 0 else 0
        else:
            agg_sharpe = 0

        ok = mdd < 0.10 and mwd < 0.05
        w1 = per_window.get("W1", 0)
        w2 = per_window.get("W2", 0)
        w3 = per_window.get("W3", 0)

        print(f"{label:<30s} ${pnl_cal:>6.2f} {agg_sharpe:>7.2f} {mdd*100:>5.1f}% "
              f"{mwd*100:>5.2f}% {wr:>4.0f}% {tt:>5d}  "
              f"${w1:>5.2f} ${w2:>5.2f} ${w3:>5.2f}  "
              f"{'OK' if ok else 'FAIL'}")

    # ---- Phase 3: Per-asset breakdown (best config) ----
    print("\n" + "=" * 80)
    print("PER-ASSET BREAKDOWN (z=2.0, exit=0.5, hold=6, sl=5%)")
    print("=" * 80)

    print(f"\n{'Asset':<8s} {'#Sigs':>6s} {'Win%':>6s} {'Avg Ret':>8s} {'Longs':>6s} "
          f"{'Shorts':>7s} {'Norm%':>6s} {'Time%':>6s} {'Stop%':>6s}")
    print("-" * 70)

    for coin in sorted(all_data.keys()):
        ohlcv, deriv = all_data[coin]
        sigs = generate_signals(ohlcv, deriv, coin, entry_zscore=2.0, exit_zscore=0.5,
                                max_hold_bars=6, stop_loss_pct=0.05)
        if len(sigs) == 0:
            print(f"{coin:<8s}      0")
            continue

        n = len(sigs)
        wr = (sigs["fwd_return"] > 0).mean() * 100
        avg_ret = sigs["fwd_return"].mean() * 100
        n_long = (sigs["direction"] > 0).sum()
        n_short = (sigs["direction"] < 0).sum()
        exits = sigs["exit_reason"].value_counts()
        norm_pct = exits.get("normalize", 0) / n * 100
        time_pct = exits.get("timeout", 0) / n * 100
        stop_pct = exits.get("stop", 0) / n * 100

        print(f"{coin:<8s} {n:>6d} {wr:>5.1f}% {avg_ret:>+7.3f}% {n_long:>6d} "
              f"{n_short:>7d} {norm_pct:>5.1f}% {time_pct:>5.1f}% {stop_pct:>5.1f}%")

    # ---- Summary ----
    print(f"\n{'='*100}")
    print(f"Total runtime: {time.time()-t0:.0f}s")
    print(f"\nDecision criteria:")
    print(f"  OOS Sharpe > 1.0 on W2+W3 -> PROCEED to live strategy")
    print(f"  OOS Sharpe < 1.0 -> KILL strategy")
    print(f"  Max DD < 15% required")
    print(f"  Worst day < 5% required")


if __name__ == "__main__":
    main()
