"""Production pipeline with dynamic asset selection.

Each month: select top-20 by 30d median dollar volume from available data,
then run the signal model only on selected assets.

Config: 18f_C0.01, 14mo lookback, thr=0.14
Strict OOS: test 2024+
"""

import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np, pandas as pd, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_engine.config import load_config, ConfidenceFilterConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.asset_selection import compute_monthly_dollar_volume, select_assets_for_month
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
LB = 14
TEST_START = "2024-01"
CONF_THR = 0.14
TOP_N = int(os.environ.get("ASSET_TOP_N", "20"))

# All symbols with data
ALL_SYMBOLS = {}
for d in DATA_DIR.iterdir():
    if d.is_dir() and d.name.endswith("USDT"):
        base = d.name.replace("USDT", "").replace("1000", "")
        ALL_SYMBOLS[d.name] = d.name  # symbol -> symbol


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.01, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.01, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def main():
    t0 = time.time()
    print("=" * 80)
    print(f"PRODUCTION PIPELINE — DYNAMIC ASSET SELECTION")
    print(f"  Top {TOP_N} by 30d median dollar volume, recalculated monthly")
    print(f"  18f_C0.01, lb={LB}mo, thr={CONF_THR}")
    print(f"  Strict OOS: test {TEST_START}+")
    print("=" * 80)

    cfg = load_config(Path("configs/default.yaml"))

    # Phase 1: Load all available data and compute dollar volumes
    print("\nPhase 1: Loading data and computing dollar volumes...", flush=True)
    asset_data = {}  # symbol -> (ohlcv_15m, features, close, vol, fwd_ret, etc.)
    asset_volumes = {}  # symbol -> rolling median dollar volume

    symbols_to_load = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and d.name.endswith("USDT"):
            symbols_to_load.append(d.name)

    for symbol in symbols_to_load:
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            if len(ohlcv_15m) < 5000:
                continue

            close = ohlcv_15m["close"]
            features = compute_features(ohlcv_15m, cfg.features)
            feat = features[cfg.features.feature_list].copy()
            for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
                if col in feat.columns:
                    feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))
            fc = feat.dropna()
            fwd_ret = close.pct_change(4).shift(-4)
            y_all = fwd_ret.reindex(fc.index).dropna()
            valid = fc.index.intersection(y_all.index)
            if len(valid) < 3000:
                continue

            vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
            dvol = compute_monthly_dollar_volume(ohlcv_15m)

            asset_data[symbol] = {
                "ohlcv_15m": ohlcv_15m, "close": close, "fc": fc,
                "fwd_ret": fwd_ret, "y_all": y_all, "valid": valid,
                "vol": vol, "start_date": ohlcv_15m.index.min(),
                "end_date": ohlcv_15m.index.max(),
                "median_gap": ohlcv_15m.index.to_series().diff().dropna().median(),
            }
            asset_volumes[symbol] = dvol
            print(f"  {symbol:<16s} {len(ohlcv_15m):>7d} bars  dvol=${dvol.dropna().iloc[-1]:>14,.0f}" if len(dvol.dropna()) > 0 else f"  {symbol:<16s} loaded", flush=True)
        except Exception as e:
            pass

    print(f"\n  Loaded {len(asset_data)} assets with sufficient data")

    # Phase 2: Build monthly schedule
    print(f"\nPhase 2: Building monthly asset schedule...", flush=True)
    test_start_ts = pd.Timestamp(TEST_START + "-01")
    sample_idx = next(iter(asset_volumes.values())).index
    if sample_idx.tz is not None:
        test_start_ts = test_start_ts.tz_localize(sample_idx.tz)

    end_date = max(d["end_date"] for d in asset_data.values())
    test_months = pd.period_range(start=test_start_ts.to_period("M"), end=end_date.to_period("M"), freq="M")

    # Phase 3: Walk-forward with dynamic selection
    print(f"\nPhase 3: Walk-forward signal generation...", flush=True)
    all_signals = []

    for month in test_months:
        ms = pd.Timestamp(month.start_time)
        me = pd.Timestamp(month.end_time)
        # Align tz
        if sample_idx.tz is not None:
            ms = ms.tz_localize(sample_idx.tz)
            me = me.tz_localize(sample_idx.tz)

        # Select assets for this month
        selected = select_assets_for_month(asset_volumes, ms, top_n=TOP_N, min_history_days=90)
        if not selected:
            continue

        month_signals = 0
        for symbol in selected:
            if symbol not in asset_data:
                continue
            ad = asset_data[symbol]
            close = ad["close"]
            fc = ad["fc"]
            fwd_ret = ad["fwd_ret"]
            y_all = ad["y_all"]
            valid = ad["valid"]
            vol = ad["vol"]

            # Adapt tz
            ms_local = ms
            me_local = me
            if fc.index.tz is None and ms.tzinfo is not None:
                ms_local = ms.tz_localize(None)
                me_local = me.tz_localize(None)
            elif fc.index.tz is not None and ms.tzinfo is None:
                ms_local = ms.tz_localize(fc.index.tz)
                me_local = me.tz_localize(fc.index.tz)

            te = valid[(valid >= ms_local) & (valid <= me_local)]
            vol_train_start = ms_local - pd.DateOffset(months=3)
            vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms_local)].dropna()
            if len(vol_train) < 50:
                continue
            vt = vol_train.quantile(0.50)
            test_vol = vol.reindex(te)
            hv = (test_vol > vt).fillna(False)
            te = te[hv.reindex(te).fillna(False).values]
            if len(te) < 5:
                continue

            train_start = ms_local - pd.DateOffset(months=LB)
            if train_start < ad["start_date"]:
                continue
            embargo_td = ad["median_gap"] * 5
            train_cutoff = ms_local - embargo_td
            tr = valid[(valid >= train_start) & (valid < train_cutoff)]
            if len(tr) < 200:
                continue

            try:
                sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[tr])
                model = make_logreg()
                model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
                Xte = scaler.transform(fc.loc[te])
                yp = model.predict_proba(Xte)[:, 1]
            except Exception:
                continue

            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            sigs = get_signal(yp, num_classes=2)
            sd = discretize_signal(sigs, step_size=0.1)
            ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
            fv = fwd_ret.reindex(te).values
            finite = np.isfinite(fv)
            if not finite.any():
                continue

            signals = pd.DataFrame({
                "timestamp": te[finite], "direction": dirs[finite],
                "fwd_return": fv[finite], "confidence": conf[finite],
                "probability": yp[finite], "signal_strength": ss[finite],
            })
            signals["asset"] = symbol.replace("USDT", "")
            signals = signals[signals["confidence"] > CONF_THR]
            month_signals += len(signals)
            if len(signals) > 0:
                all_signals.append(signals)

        print(f"  {str(month)}: {len(selected)} assets selected, {month_signals} signals", flush=True)

    if not all_signals:
        print("NO SIGNALS"); return

    combined = pd.concat(all_signals, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    n_assets = combined["asset"].nunique()
    print(f"\n  Total: {len(combined)} signals from {n_assets} assets")

    # Phase 4: Compare dynamic vs fixed-30
    print(f"\n{'='*80}")
    print("COMPARISON: DYNAMIC TOP-20 vs FIXED 30")
    print(f"{'='*80}")

    # Dynamic top-20
    combined_nco = orig.apply_nco_wf(combined)
    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.20, delever_start=0.10,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
        base_sizing_pct=0.04, max_signal_mult=2.0,
    )
    sim = simulate_portfolio(combined_nco, pcfg)
    dpnl = compute_daily_pnl_series(sim)
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"], sim["n_trades"] / len(dpnl))
    equity = pd.Series(sim["equity"][~sim["skipped"]], index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
    dd = ((equity - equity.expanding().max()) / equity.expanding().max())
    monthly = dpnl.resample("ME").sum()
    pnl_day = (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1)

    def print_stats(label, stats, sim, dpnl, dd, monthly, pnl_day, combined):
        print(f"\n  {label}")
        print(f"    Sharpe:       {stats['sharpe']:.3f}")
        print(f"    Sortino:      {stats.get('sortino', 0):.3f}")
        print(f"    Calmar:       {stats.get('calmar', 0):.3f}")
        print(f"    Return:       {stats['total_return_pct']:.1f}%")
        print(f"    Max DD:       {dd.min()*100:.2f}%")
        print(f"    Worst Day:    {(dpnl/pcfg.starting_capital*100).min():.2f}%")
        print(f"    PnL/day:      ${pnl_day:.2f}")
        print(f"    Win% (daily): {(dpnl>0).mean()*100:.1f}%")
        print(f"    Trades:       {sim['n_trades']}")
        print(f"    Assets:       {combined['asset'].nunique()}")
        print(f"    Pos Months:   {(monthly>0).sum()}/{len(monthly)}")

        # Flat-bet stats
        pnls = combined["direction"].values * combined["fwd_return"].values - 0.0001
        print(f"    Flat Win%:    {(pnls>0).mean()*100:.1f}%")
        print(f"    Flat bps:     {pnls.mean()*10000:.1f}")

    print_stats("DYNAMIC TOP-20 (volume-based, monthly rebalanced)",
                stats, sim, dpnl, dd, monthly, pnl_day, combined)

    # Fixed 30 (from earlier pipeline)
    print(f"\n  (Run strict_oos_report.py for fixed-30 comparison)")

    # Per-asset breakdown
    print(f"\n{'='*80}")
    print("PER-ASSET CONTRIBUTION")
    print(f"{'='*80}")
    print(f"  {'Asset':<10s} {'Months':>6s} {'Signals':>7s} {'Win%':>6s} {'bps':>7s}")
    print("  " + "-" * 40)

    asset_stats = combined.groupby("asset").agg(
        months=("timestamp", lambda x: x.dt.to_period("M").nunique()),
        signals=("direction", "count"),
        win_rate=("fwd_return", lambda x: ((combined.loc[x.index, "direction"] * x - 0.0001) > 0).mean()),
        bps=("fwd_return", lambda x: (combined.loc[x.index, "direction"] * x - 0.0001).mean() * 10000),
    ).sort_values("bps", ascending=False)

    for asset, row in asset_stats.iterrows():
        print(f"  {asset:<10s} {row['months']:>6.0f} {row['signals']:>7.0f} "
              f"{row['win_rate']*100:>5.1f}% {row['bps']:>6.1f}")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
