"""Confidence-scaled leverage test.

Tests whether scaling position size with signal confidence improves
risk-adjusted returns for the prop firm challenge.

Variants:
  1. Baseline: flat sizing (current production)
  2. Linear: leverage = 1 + confidence * scale
  3. Tiered: confidence buckets → discrete leverage levels
  4. Quadratic: leverage = 1 + (confidence/0.5)^2 * extra
  5. Threshold: only scale above confidence > 0.20

For each variant, sweeps max leverage from 1.0x to 3.0x.
Reports: Sharpe, Calmar, MaxDD, PnL/day, worst day, days to 10%.
"""

import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import time, warnings, copy
from pathlib import Path

import numpy as np, pandas as pd, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_engine.config import load_config, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.asset_selection import compute_monthly_dollar_volume
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
DATA_DIR = Path("data/binance-1m")
LB = 14
TEST_START = "2024-01"
CONF_THR = 0.14
TOP_N = 30
STARTING_CAPITAL = 10000.0


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.01, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.01, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def run_portfolio_variant(signals_df, pcfg):
    """Run portfolio sim and return stats dict."""
    if len(signals_df) < 50:
        return None
    combined = orig.apply_nco_wf(signals_df)
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                               sim["n_trades"] / len(dpnl))
    equity = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
    dd = ((equity - equity.expanding().max()) / equity.expanding().max())
    monthly = dpnl.resample("ME").sum()
    pnl_day = (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1)

    # Days to 10% target
    cumulative = dpnl.cumsum()
    target = pcfg.starting_capital * 0.10
    days_to_target = None
    above = cumulative[cumulative >= target]
    if len(above) > 0:
        first_day = above.index[0]
        days_to_target = (first_day - cumulative.index[0]).days

    return {
        "sharpe": stats["sharpe"],
        "sortino": stats.get("sortino", 0),
        "calmar": stats.get("calmar", 0),
        "ret_pct": stats["total_return_pct"],
        "max_dd": dd.min() * 100,
        "worst_day": (dpnl / pcfg.starting_capital * 100).min(),
        "pnl_day": pnl_day,
        "win_daily": (dpnl > 0).mean() * 100,
        "n_trades": sim["n_trades"],
        "pos_months": f"{(monthly > 0).sum()}/{len(monthly)}",
        "final_eq": stats["final_equity"],
        "days_to_10pct": days_to_target,
    }


# --- Confidence scaling functions ---

def scale_flat(confidence, signal_strength, leverage):
    """Baseline: flat leverage applied to signal_strength."""
    return signal_strength * leverage


def scale_linear(confidence, signal_strength, max_leverage):
    """Linear: leverage = 1 + (confidence / 0.5) * (max_leverage - 1)."""
    conf_ratio = min(confidence / 0.5, 1.0)
    lev = 1.0 + conf_ratio * (max_leverage - 1.0)
    return signal_strength * lev


def scale_tiered(confidence, signal_strength, max_leverage):
    """Tiered: discrete levels based on confidence buckets."""
    if confidence >= 0.30:
        lev = max_leverage
    elif confidence >= 0.20:
        lev = 1.0 + (max_leverage - 1.0) * 0.5
    else:
        lev = 1.0
    return signal_strength * lev


def scale_quadratic(confidence, signal_strength, max_leverage):
    """Quadratic: leverage = 1 + (confidence / 0.5)^2 * (max_leverage - 1)."""
    conf_ratio = min(confidence / 0.5, 1.0)
    lev = 1.0 + (conf_ratio ** 2) * (max_leverage - 1.0)
    return signal_strength * lev


def scale_threshold(confidence, signal_strength, max_leverage):
    """Threshold: only scale if confidence > 0.20, else 1x."""
    if confidence > 0.20:
        conf_ratio = min((confidence - 0.20) / 0.30, 1.0)
        lev = 1.0 + conf_ratio * (max_leverage - 1.0)
    else:
        lev = 1.0
    return signal_strength * lev


def main():
    t0 = time.time()
    cfg = load_config(Path("configs/default.yaml"))

    print("=" * 90)
    print("CONFIDENCE-SCALED LEVERAGE TEST")
    print(f"  Dynamic top-{TOP_N}, decay_T3, NCO portfolio, OOS {TEST_START}+")
    print(f"  Prop firm target: 10% profit, <10% DD, <5% daily DD")
    print("=" * 90)

    # ==================== PHASE 1: Generate signals ====================
    print("\nPhase 1: Loading data + generating signals...", flush=True)

    asset_data = {}
    asset_volumes = {}

    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith("USDT"):
            continue
        symbol = d.name
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
            close_arr = ohlcv_15m["close"].values.astype(np.float64)
            timestamps = ohlcv_15m.index

            asset_data[symbol] = {
                "close": close, "close_arr": close_arr, "fc": fc,
                "fwd_ret": fwd_ret, "y_all": y_all, "valid": valid,
                "vol": vol, "timestamps": timestamps,
                "start_date": ohlcv_15m.index.min(),
                "end_date": ohlcv_15m.index.max(),
                "median_gap": ohlcv_15m.index.to_series().diff().dropna().median(),
            }
            asset_volumes[symbol] = compute_monthly_dollar_volume(ohlcv_15m)
        except Exception:
            pass

    print(f"  Loaded {len(asset_data)} assets in {time.time() - t0:.0f}s")

    # Walk-forward with dynamic selection + decay_T3
    sample_idx = next(iter(asset_volumes.values())).index
    test_start_ts = pd.Timestamp(TEST_START + "-01")
    if sample_idx.tz is not None:
        test_start_ts = test_start_ts.tz_localize(sample_idx.tz)
    end_date = max(d["end_date"] for d in asset_data.values())
    test_months = pd.period_range(start=test_start_ts.to_period("M"), end=end_date.to_period("M"), freq="M")

    all_signals = []
    print("\nGenerating signals with decay_T3...", flush=True)

    for month in test_months:
        ms = pd.Timestamp(month.start_time)
        if sample_idx.tz is not None:
            ms = ms.tz_localize(sample_idx.tz)

        lookback_end = ms - pd.Timedelta(days=1)
        vol_candidates = {}
        for symbol, vol_s in asset_volumes.items():
            avail = vol_s.loc[:lookback_end]
            if len(avail) < 45: continue
            latest = avail.iloc[-1]
            if np.isnan(latest) or latest <= 0: continue
            if len(avail) >= 90:
                m90 = avail.iloc[-90:].median()
                if m90 > 0 and latest / m90 > 5.0: continue
            vol_candidates[symbol] = latest
        selected = [s for s, _ in sorted(vol_candidates.items(), key=lambda x: -x[1])[:TOP_N]]

        month_count = 0
        for symbol in selected:
            if symbol not in asset_data:
                continue
            ad = asset_data[symbol]
            close_arr = ad["close_arr"]
            n = len(close_arr)
            fc = ad["fc"]
            y_all = ad["y_all"]
            valid = ad["valid"]
            vol = ad["vol"]
            timestamps_idx = ad["timestamps"]

            ms_local = ms
            me_local = ms + pd.offsets.MonthEnd(0)
            if fc.index.tz is None and ms.tzinfo is not None:
                ms_local = ms.tz_localize(None)
                me_local = me_local.tz_localize(None)

            te = valid[(valid >= ms_local) & (valid <= me_local)]
            vol_train_start = ms_local - pd.DateOffset(months=3)
            vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms_local)].dropna()
            if len(vol_train) < 50: continue
            vt = vol_train.quantile(0.50)
            test_vol = vol.reindex(te)
            hv = (test_vol > vt).fillna(False)
            te = te[hv.reindex(te).fillna(False).values]
            if len(te) < 5: continue

            train_start = ms_local - pd.DateOffset(months=LB)
            if train_start < ad["start_date"]: continue
            embargo_td = ad["median_gap"] * 5
            train_cutoff = ms_local - embargo_td
            tr = valid[(valid >= train_start) & (valid < train_cutoff)]
            if len(tr) < 200: continue

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
            mask = conf > CONF_THR
            asset_name = symbol.replace("USDT", "")

            for i in range(len(te)):
                if not mask[i]: continue
                bar_idx = timestamps_idx.get_loc(te[i])
                if bar_idx + 4 >= n: continue
                ep = close_arr[bar_idx]
                if ep <= 0: continue
                d = dirs[i]

                # decay_T3: close at T+3 if losing, else T+4
                rets = [d * (close_arr[bar_idx + k] - ep) / ep for k in range(1, 5)]
                exit_bar = 3 if rets[2] < 0 else 4
                fwd_return = rets[exit_bar - 1]

                all_signals.append({
                    "timestamp": te[i], "direction": d, "confidence": conf[i],
                    "probability": yp[i], "signal_strength": float(ss[i]),
                    "asset": asset_name, "fwd_return": fwd_return,
                })
                month_count += 1

        print(f"  {str(month)}: {len(selected)} assets, {month_count} signals", flush=True)

    signals_df = pd.DataFrame(all_signals).sort_values("timestamp").reset_index(drop=True)
    print(f"\n  Total: {len(signals_df)} signals, {signals_df['asset'].nunique()} assets")

    # Confidence distribution
    print(f"\n  Confidence distribution:")
    for lo, hi in [(0.14, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50)]:
        ct = ((signals_df["confidence"] >= lo) & (signals_df["confidence"] < hi)).sum()
        print(f"    [{lo:.2f}, {hi:.2f}): {ct} ({ct/len(signals_df)*100:.1f}%)")

    # ==================== PHASE 2: Test leverage variants ====================
    print(f"\n{'='*90}")
    print("CONFIDENCE-SCALED LEVERAGE RESULTS")
    print(f"{'='*90}")

    variants = {
        "flat": scale_flat,
        "linear": scale_linear,
        "tiered": scale_tiered,
        "quadratic": scale_quadratic,
        "threshold_0.20": scale_threshold,
    }

    leverage_levels = [1.0, 1.5, 1.7, 2.0, 2.5, 3.0]

    # Header
    print(f"\n  {'Variant':<20s} {'Lev':>4s} {'Sharpe':>7s} {'Calmar':>7s} {'MaxDD%':>7s} "
          f"{'WrstDay%':>8s} {'PnL/d':>7s} {'Days10%':>7s} {'Trades':>6s}")
    print("  " + "-" * 80)

    results = []

    for lev in leverage_levels:
        for name, scale_fn in variants.items():
            # Apply confidence scaling to signal_strength
            df = signals_df.copy()
            df["signal_strength"] = df.apply(
                lambda r: scale_fn(r["confidence"], r["signal_strength"], lev),
                axis=1,
            )

            pcfg = PortfolioConfig(
                starting_capital=STARTING_CAPITAL,
                max_dd=0.20,  # don't halt — let us see true DD
                delever_start=0.10,
                slippage=0.0001,
                max_exposure=0.90,
                max_per_asset=0.05,
                base_sizing_pct=0.04 * lev,  # THIS is the actual leverage knob
                max_signal_mult=3.0,
            )

            r = run_portfolio_variant(df, pcfg)
            if r is None:
                continue

            results.append({"variant": name, "leverage": lev, **r})

            d10 = f"{r['days_to_10pct']}" if r['days_to_10pct'] else "N/A"
            print(f"  {name:<20s} {lev:>4.1f} {r['sharpe']:>7.3f} {r['calmar']:>7.1f} "
                  f"{r['max_dd']:>7.2f} {r['worst_day']:>8.2f} ${r['pnl_day']:>6.2f} "
                  f"{d10:>7s} {r['n_trades']:>6d}")

    # ==================== PHASE 3: Analysis ====================
    print(f"\n{'='*90}")
    print("PROP FIRM FEASIBILITY (DD < 10%, daily < 5%)")
    print(f"{'='*90}")

    feasible = [r for r in results if r["max_dd"] > -10.0 and r["worst_day"] > -5.0]
    feasible.sort(key=lambda x: -x["pnl_day"])

    print(f"\n  Feasible configurations ({len(feasible)}/{len(results)}):")
    print(f"  {'Variant':<20s} {'Lev':>4s} {'Sharpe':>7s} {'MaxDD%':>7s} {'WrstDay%':>8s} "
          f"{'PnL/d':>7s} {'Days10%':>7s}")
    print("  " + "-" * 65)

    for r in feasible[:15]:
        d10 = f"{r['days_to_10pct']}" if r['days_to_10pct'] else "N/A"
        print(f"  {r['variant']:<20s} {r['leverage']:>4.1f} {r['sharpe']:>7.3f} "
              f"{r['max_dd']:>7.2f} {r['worst_day']:>8.2f} ${r['pnl_day']:>6.2f} {d10:>7s}")

    # Best by PnL/day within prop firm limits
    if feasible:
        best = feasible[0]
        print(f"\n  BEST: {best['variant']} @ {best['leverage']}x")
        print(f"    Sharpe:    {best['sharpe']:.3f}")
        print(f"    Calmar:    {best['calmar']:.1f}")
        print(f"    Max DD:    {best['max_dd']:.2f}%")
        print(f"    Worst Day: {best['worst_day']:.2f}%")
        print(f"    PnL/day:   ${best['pnl_day']:.2f}")
        print(f"    Days to 10%: {best['days_to_10pct'] or 'N/A'}")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
