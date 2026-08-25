"""Pipeline with improved asset selection vs volume-only baseline.

Improved: multi-factor score + diversification + data quality + buffer
Baseline: volume-only top-30 (current production)
"""

import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import time, warnings
from pathlib import Path

import numpy as np, pandas as pd, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_engine.config import load_config, ConfidenceFilterConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.asset_selection import (
    compute_asset_score, compute_monthly_dollar_volume,
    select_assets_for_month as select_improved,
)
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
LB = 14
TEST_START = "2024-01"
CONF_THR = 0.14
TOP_N = 30


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.01, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.01, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def run_signal(asset_data, symbol, month_start, lb_months, cfg):
    """Generate signals for one asset for one month."""
    ad = asset_data[symbol]
    close = ad["close"]
    fc = ad["fc"]
    fwd_ret = ad["fwd_ret"]
    y_all = ad["y_all"]
    valid = ad["valid"]
    vol = ad["vol"]

    ms = month_start
    me = ms + pd.offsets.MonthEnd(0)
    if fc.index.tz is None and ms.tzinfo is not None:
        ms = ms.tz_localize(None)
        me = me.tz_localize(None)
    elif fc.index.tz is not None and ms.tzinfo is None:
        ms = ms.tz_localize(fc.index.tz)
        me = me.tz_localize(fc.index.tz)

    te = valid[(valid >= ms) & (valid <= me)]
    vol_train_start = ms - pd.DateOffset(months=3)
    vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms)].dropna()
    if len(vol_train) < 50:
        return None
    vt = vol_train.quantile(0.50)
    test_vol = vol.reindex(te)
    hv = (test_vol > vt).fillna(False)
    te = te[hv.reindex(te).fillna(False).values]
    if len(te) < 5:
        return None

    train_start = ms - pd.DateOffset(months=lb_months)
    if train_start < ad["start_date"]:
        return None
    embargo_td = ad["median_gap"] * 5
    train_cutoff = ms - embargo_td
    tr = valid[(valid >= train_start) & (valid < train_cutoff)]
    if len(tr) < 200:
        return None

    try:
        sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(fc.loc[tr])
        model = make_logreg()
        model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
        Xte = scaler.transform(fc.loc[te])
        yp = model.predict_proba(Xte)[:, 1]
    except Exception:
        return None

    conf = np.abs(yp - 0.5)
    dirs = np.where(yp > 0.5, 1.0, -1.0)
    sigs = get_signal(yp, num_classes=2)
    sd = discretize_signal(sigs, step_size=0.1)
    ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
    fv = fwd_ret.reindex(te).values
    finite = np.isfinite(fv)
    if not finite.any():
        return None

    signals = pd.DataFrame({
        "timestamp": te[finite], "direction": dirs[finite],
        "fwd_return": fv[finite], "confidence": conf[finite],
        "probability": yp[finite], "signal_strength": ss[finite],
    })
    signals["asset"] = symbol.replace("USDT", "")
    signals = signals[signals["confidence"] > CONF_THR]
    return signals if len(signals) > 0 else None


def analyze(combined, pcfg, label):
    if len(combined) < 50:
        print(f"  {label}: INSUFFICIENT ({len(combined)} signals)")
        return
    combined_nco = orig.apply_nco_wf(combined)
    sim = simulate_portfolio(combined_nco, pcfg)
    if sim is None or sim["n_trades"] < 10:
        print(f"  {label}: SIM FAILED")
        return
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                               sim["n_trades"] / len(dpnl))
    equity = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
    dd = ((equity - equity.expanding().max()) / equity.expanding().max())
    monthly = dpnl.resample("ME").sum()
    pnl_day = (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1)

    pnls_flat = combined["direction"].values * combined["fwd_return"].values - 0.0001

    print(f"\n  {label}")
    print(f"    Sharpe:       {stats['sharpe']:.3f}")
    print(f"    Sortino:      {stats.get('sortino', 0):.3f}")
    print(f"    Calmar:       {stats.get('calmar', 0):.3f}")
    print(f"    Return:       {stats['total_return_pct']:.1f}%")
    print(f"    Max DD:       {dd.min()*100:.2f}%")
    print(f"    Worst Day:    {(dpnl/pcfg.starting_capital*100).min():.2f}%")
    print(f"    PnL/day:      ${pnl_day:.2f} ({pnl_day/pcfg.starting_capital*100:.3f}%)")
    print(f"    Win% (daily): {(dpnl>0).mean()*100:.1f}%")
    print(f"    Trades:       {sim['n_trades']}")
    print(f"    Signals:      {len(combined)}")
    print(f"    Assets:       {combined['asset'].nunique()}")
    print(f"    Pos Months:   {(monthly>0).sum()}/{len(monthly)}")
    print(f"    Flat Win%:    {(pnls_flat>0).mean()*100:.1f}%")
    print(f"    Flat bps:     {pnls_flat.mean()*10000:.1f}")


def main():
    t0 = time.time()
    cfg = load_config(Path("configs/default.yaml"))

    print("=" * 80)
    print("PIPELINE: IMPROVED SELECTION vs VOLUME-ONLY BASELINE")
    print(f"  18f_C0.01, lb={LB}mo, thr={CONF_THR}, top-{TOP_N}")
    print("=" * 80)

    # Phase 1: Load all data
    print("\nLoading all assets...", flush=True)
    asset_data = {}
    asset_scores = {}
    asset_volumes = {}
    daily_returns = {}
    asset_ohlcv = {}

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
            asset_data[symbol] = {
                "ohlcv_15m": ohlcv_15m, "close": close, "fc": fc,
                "fwd_ret": fwd_ret, "y_all": y_all, "valid": valid,
                "vol": vol, "start_date": ohlcv_15m.index.min(),
                "end_date": ohlcv_15m.index.max(),
                "median_gap": ohlcv_15m.index.to_series().diff().dropna().median(),
            }

            # Scores for improved selection
            asset_scores[symbol] = compute_asset_score(ohlcv_15m)
            asset_volumes[symbol] = compute_monthly_dollar_volume(ohlcv_15m)
            asset_ohlcv[symbol] = ohlcv_15m

            dr = close.resample("1D").last().pct_change().dropna()
            daily_returns[symbol] = dr
        except Exception as e:
            print(f"  WARN {symbol}: {e}", flush=True)

    print(f"  Loaded {len(asset_data)} assets, {len(asset_volumes)} volumes in {time.time() - t0:.0f}s")

    # Phase 2: Walk-forward with both selection methods
    test_start_ts = pd.Timestamp(TEST_START + "-01")
    if not asset_volumes:
        print("NO ASSET VOLUMES"); return
    sample_idx = next(iter(asset_volumes.values())).index
    if sample_idx.tz is not None:
        test_start_ts = test_start_ts.tz_localize(sample_idx.tz)

    end_date = max(d["end_date"] for d in asset_data.values())
    test_months = pd.period_range(start=test_start_ts.to_period("M"), end=end_date.to_period("M"), freq="M")

    signals_baseline = []
    signals_improved = []
    prev_improved = []

    print(f"\nGenerating signals...", flush=True)
    for month in test_months:
        ms = pd.Timestamp(month.start_time)
        if sample_idx.tz is not None:
            ms = ms.tz_localize(sample_idx.tz)

        # Baseline: volume-only top-30
        vol_candidates = {}
        lookback_end = ms - pd.Timedelta(days=1)
        for symbol, vol_s in asset_volumes.items():
            avail = vol_s.loc[:lookback_end]
            if len(avail) < 45:
                continue
            latest = avail.iloc[-1]
            if np.isnan(latest) or latest <= 0:
                continue
            if len(avail) >= 90:
                m90 = avail.iloc[-90:].median()
                if m90 > 0 and latest / m90 > 5.0:
                    continue
            vol_candidates[symbol] = latest
        baseline_assets = [s for s, _ in sorted(vol_candidates.items(), key=lambda x: -x[1])[:TOP_N]]

        # Improved: multi-factor + diversification
        improved_assets, weights = select_improved(
            asset_scores, ms, top_n=TOP_N, min_history_days=90,
            previous_assets=prev_improved, asset_ohlcv=asset_ohlcv,
            daily_returns=daily_returns, max_corr=0.85,
        )
        prev_improved = improved_assets

        # Generate signals for both universes
        for symbol in set(baseline_assets + improved_assets):
            if symbol not in asset_data:
                continue
            sigs = run_signal(asset_data, symbol, ms, LB, cfg)
            if sigs is not None and len(sigs) > 0:
                if symbol in baseline_assets:
                    signals_baseline.append(sigs.copy())
                if symbol in improved_assets:
                    s = sigs.copy()
                    # Apply rebalance weight
                    w = weights.get(symbol, 1.0)
                    s["signal_strength"] = s["signal_strength"] * w
                    signals_improved.append(s)

        nb = sum(len(s) for s in signals_baseline)
        ni = sum(len(s) for s in signals_improved)
        print(f"  {str(month)}: base={len(baseline_assets)} improved={len(improved_assets)} "
              f"sigs_b={nb} sigs_i={ni}", flush=True)

    # Phase 3: Compare
    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.20, delever_start=0.10,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
        base_sizing_pct=0.04, max_signal_mult=2.0,
    )

    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    if signals_baseline:
        combined_b = pd.concat(signals_baseline, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        analyze(combined_b, pcfg, "BASELINE (volume-only top-30)")

    if signals_improved:
        combined_i = pd.concat(signals_improved, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        analyze(combined_i, pcfg, "IMPROVED (multi-factor + diversification + buffer)")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
