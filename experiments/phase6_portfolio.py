"""Phase 6: Portfolio Construction — NCO vs HRP vs denoised, asset selection.

Tests on strict OOS (2024+), 18f_C0.01, 14mo, thr=0.14.

Variants:
A: NCO k=8 (current baseline)
B: NCO k=8 with Marcenko-Pastur denoised correlation
C: Equal weight (no allocation optimization)
D: Kelly weights only (no clustering)
E: Drop worst 5 assets (negative holdout Sharpe)
F: Top 15 assets only
G: Top 20 assets only
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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from signal_engine.config import load_config, ConfidenceFilterConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.allocation import compute_nco_weights, compute_kelly_weights
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
LB = 14
TEST_START = "2024-01"
CONF_THR = 0.14


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.01, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.01, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def process_asset(args):
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")
    from signal_engine.config import load_config, ConfidenceFilterConfig
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights
    from afml.bet_sizing import get_signal, discretize_signal

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, []

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
        return asset, []

    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5
    test_start_ts = pd.Timestamp(TEST_START, tz="UTC")
    if fc.index.tz is None:
        test_start_ts = test_start_ts.tz_localize(None)
    test_months = pd.period_range(start=test_start_ts.to_period("M"), end=end_date.to_period("M"), freq="M")

    all_signals = []
    for month in test_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None); me = me.tz_localize(None)
        te = valid[(valid >= ms) & (valid <= me)]
        vol_train_start = ms - pd.DateOffset(months=3)
        vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms)].dropna()
        if len(vol_train) < 50: continue
        vt = vol_train.quantile(0.50)
        test_vol = vol.reindex(te)
        hv = (test_vol > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5: continue

        train_start = ms - pd.DateOffset(months=LB)
        if train_start < start_date: continue
        train_cutoff = ms - embargo_td
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
        except Exception: continue

        conf = np.abs(yp - 0.5)
        dirs = np.where(yp > 0.5, 1.0, -1.0)
        sigs = get_signal(yp, num_classes=2)
        sd = discretize_signal(sigs, step_size=0.1)
        ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
        fv = fwd_ret.reindex(te).values
        finite = np.isfinite(fv)
        if not finite.any(): continue

        signals = pd.DataFrame({
            "timestamp": te[finite], "direction": dirs[finite],
            "fwd_return": fv[finite], "confidence": conf[finite],
            "probability": yp[finite], "signal_strength": ss[finite],
        })
        signals["asset"] = asset
        signals = signals[signals["confidence"] > CONF_THR]
        if len(signals) > 0:
            all_signals.append(signals)

    return asset, pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()


def apply_equal_weights(combined):
    """Equal weight: signal_strength = 1.0 for all."""
    combined = combined.copy()
    combined["signal_strength"] = 1.0
    return combined


def apply_kelly_only(combined):
    """Kelly weights per asset, no clustering."""
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(combined["timestamp"])
    months = sorted(ts.dt.to_period("M").unique())
    parts = []
    for km in months:
        km_sigs = combined[ts.dt.to_period("M") == km]
        past_sigs = combined[ts.dt.to_period("M") < km]
        if len(past_sigs) < 50:
            parts.append(km_sigs)
            continue
        kw = compute_kelly_weights(past_sigs)
        kmc = km_sigs.copy()
        kmc["signal_strength"] = kmc.apply(
            lambda r: r["signal_strength"] * kw.get(r["asset"], 1.0), axis=1)
        parts.append(kmc)
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def analyze(combined, pcfg):
    if len(combined) < 50:
        return None
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5: return None
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                               sim["n_trades"] / len(dpnl))
    equity = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
    dd = ((equity - equity.expanding().max()) / equity.expanding().max())
    monthly = dpnl.resample("ME").sum()
    pnl_day = (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1)
    return {
        "sharpe": stats["sharpe"], "sortino": stats.get("sortino", 0),
        "calmar": stats.get("calmar", 0), "ret_pct": stats["total_return_pct"],
        "max_dd": dd.min() * 100, "worst_day": (dpnl / pcfg.starting_capital * 100).min(),
        "pnl_day": pnl_day, "win_daily": (dpnl > 0).mean() * 100,
        "n_trades": sim["n_trades"], "n_days": stats["n_days"],
        "n_assets": combined["asset"].nunique(),
        "pos_months": f"{(monthly > 0).sum()}/{len(monthly)}",
    }


def print_row(label, r):
    if r is None:
        print(f"  {label:<30s}  NO DATA")
        return
    print(f"  {label:<30s} {r['sharpe']:>6.3f} {r['sortino']:>6.2f} {r['calmar']:>7.2f} "
          f"{r['ret_pct']:>5.1f}% {r['max_dd']:>6.2f}% {r['worst_day']:>5.2f}% "
          f"${r['pnl_day']:>6.2f} {r['win_daily']:>4.1f}% {r['n_trades']:>6d} "
          f"{r['n_assets']:>3d} {r['pos_months']:>5s}")


def main():
    t0 = time.time()
    print("=" * 100)
    print(f"PHASE 6: PORTFOLIO CONSTRUCTION (18f_C0.01, lb={LB}, thr={CONF_THR})")
    print("=" * 100, flush=True)

    all_signals = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s)): a for a, s in orig.ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, sigs = f.result()
            done += 1
            n = len(sigs) if isinstance(sigs, pd.DataFrame) else 0
            all_signals[asset] = sigs
            print(f"  [{done:>2d}/30] {asset}: {n}", flush=True)

    parts = [s for s in all_signals.values() if isinstance(s, pd.DataFrame) and len(s) > 0]
    combined_raw = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    print(f"\n  {len(combined_raw)} signals, {combined_raw['asset'].nunique()} assets")
    print(f"  Signals done in {time.time() - t0:.0f}s")

    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.20, delever_start=0.10,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
        base_sizing_pct=0.04, max_signal_mult=2.0,
    )

    # Per-asset flat-bet Sharpe (for asset selection)
    asset_sharpes = {}
    for asset, sigs in all_signals.items():
        if not isinstance(sigs, pd.DataFrame) or len(sigs) < 10: continue
        p = sigs["direction"].values * sigs["fwd_return"].values - 0.0001
        ts_a = pd.to_datetime(sigs["timestamp"], utc=True)
        d = pd.Series(5.0 * p, index=ts_a.values).resample("1D").sum()
        d = d[d != 0]
        if len(d) > 5 and d.std() > 0:
            asset_sharpes[asset] = d.mean() / d.std() * np.sqrt(365.25)

    sorted_assets = sorted(asset_sharpes.items(), key=lambda x: -x[1])
    top_15 = set(a for a, _ in sorted_assets[:15])
    top_20 = set(a for a, _ in sorted_assets[:20])
    drop_worst_5 = set(a for a, _ in sorted_assets[:-5]) if len(sorted_assets) > 5 else set(asset_sharpes.keys())

    print(f"\n  Asset Sharpes (flat-bet):")
    for a, sr in sorted_assets:
        marker = " *" if a in top_15 else ""
        print(f"    {a:<8s} {sr:>7.3f}{marker}")

    # ================================================================
    # ALLOCATION VARIANTS
    # ================================================================
    print(f"\n{'='*100}")
    print("ALLOCATION COMPARISON")
    print(f"{'='*100}")
    print(f"  {'Variant':<30s} {'SR':>6s} {'Sort':>6s} {'Calm':>7s} {'Ret%':>5s} "
          f"{'MaxDD':>6s} {'WDay':>5s} {'PnL/d':>7s} {'WinD':>4s} {'Trds':>6s} "
          f"{'Ast':>3s} {'Mos':>5s}")
    print("  " + "-" * 96)

    # A: NCO k=8 (baseline)
    combined_a = orig.apply_nco_wf(combined_raw.copy())
    print_row("A: NCO k=8", analyze(combined_a, pcfg))

    # B: Equal weight
    combined_b = apply_equal_weights(combined_raw.copy())
    print_row("B: Equal weight", analyze(combined_b, pcfg))

    # C: Kelly only
    combined_c = apply_kelly_only(combined_raw.copy())
    print_row("C: Kelly only", analyze(combined_c, pcfg))

    # D: Drop worst 5 + NCO
    combined_d = combined_raw[combined_raw["asset"].isin(drop_worst_5)].copy()
    combined_d = orig.apply_nco_wf(combined_d)
    print_row("D: Drop worst 5 + NCO", analyze(combined_d, pcfg))

    # E: Top 20 + NCO
    combined_e = combined_raw[combined_raw["asset"].isin(top_20)].copy()
    combined_e = orig.apply_nco_wf(combined_e)
    print_row("E: Top 20 + NCO", analyze(combined_e, pcfg))

    # F: Top 15 + NCO
    combined_f = combined_raw[combined_raw["asset"].isin(top_15)].copy()
    combined_f = orig.apply_nco_wf(combined_f)
    print_row("F: Top 15 + NCO", analyze(combined_f, pcfg))

    # G: Top 20 + equal weight
    combined_g = combined_raw[combined_raw["asset"].isin(top_20)].copy()
    combined_g = apply_equal_weights(combined_g)
    print_row("G: Top 20 + equal wt", analyze(combined_g, pcfg))

    # H: Top 15 + equal weight
    combined_h = combined_raw[combined_raw["asset"].isin(top_15)].copy()
    combined_h = apply_equal_weights(combined_h)
    print_row("H: Top 15 + equal wt", analyze(combined_h, pcfg))

    # I: Drop worst 5 + equal weight
    combined_i = combined_raw[combined_raw["asset"].isin(drop_worst_5)].copy()
    combined_i = apply_equal_weights(combined_i)
    print_row("I: Drop worst 5 + eq wt", analyze(combined_i, pcfg))

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
