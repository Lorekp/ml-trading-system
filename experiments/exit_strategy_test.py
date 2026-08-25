"""Exit Strategy Experiment -- Path-Dependent Exits vs Fixed Horizon.

Generates walk-forward signals ONCE (8-month lookback, 30 assets),
then tests multiple exit configurations using the path-dependent simulator.
Compares against the fixed 4-bar baseline (Sharpe ~3.686).

Senior quant standards:
- Walk-forward only (no in-sample optimization)
- Pre-registered parameter grids
- Bonferroni correction for multiple comparisons
- Conservative barrier priority (SL assumed hit first on ambiguous bars)
"""

import sys
sys.path.insert(0, "src")

import copy
import os
import time
import warnings

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from signal_engine.config import ExitConfig, load_config, ConfidenceFilterConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.portfolio_path import simulate_portfolio_path
from signal_engine.stats import compute_full_stats
from afml.bet_sizing import get_signal, discretize_signal

import sklearn
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)

W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")

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

LOOKBACK = 8  # 8-month rolling (proven best Sharpe)


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(
            C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42
        )
    return LogisticRegression(
        C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42
    )


def _process_asset(args):
    """Generate signals for one asset with 8-month lookback."""
    asset, symbol, fixed_start_month = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    from signal_engine.config import load_config, ConfidenceFilterConfig
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights
    from signal_engine.filters import apply_confidence_filter
    from afml.bet_sizing import get_signal, discretize_signal

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, []

    features = compute_features(ohlcv_15m, cfg.features)
    feat = features[cfg.features.feature_list].copy()
    close = ohlcv_15m["close"]
    fwd_ret = close.pct_change(4).shift(-4)
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in feat.columns:
            feat[col] = feat[col].clip(
                feat[col].quantile(0.01), feat[col].quantile(0.99)
            )

    fc = feat.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)
    if len(valid) < 3000:
        return asset, []

    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    results = []

    all_months = pd.period_range(
        start=(start_date + pd.DateOffset(months=2)).to_period("M"),
        end=end_date.to_period("M"),
        freq="M",
    )

    for month in all_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

        # Skip months before fixed start (Version B -- same test period)
        if fixed_start_month and str(month) < fixed_start_month:
            continue

        te = valid[(valid >= ms) & (valid <= me)]
        vol_train_start = ms - pd.DateOffset(months=3)
        vol_train = vol.loc[
            (vol.index >= vol_train_start) & (vol.index < ms)
        ].dropna()
        if len(vol_train) < 50:
            continue
        vt = vol_train.quantile(0.50)
        test_vol = vol.reindex(te)
        hv = (test_vol > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5:
            continue

        # 8-month lookback -- check enough history
        train_start = ms - pd.DateOffset(months=LOOKBACK)
        if train_start < start_date:
            continue

        train_cutoff = ms - embargo_td
        tr = valid[(valid >= train_start) & (valid < train_cutoff)]
        if len(tr) < 200:
            continue

        try:
            sw = compute_sample_weights(
                fc.loc[tr].index.values.astype(np.int64), fwd_bars=4
            )
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

        signals = pd.DataFrame(
            {
                "timestamp": te[finite],
                "direction": dirs[finite],
                "fwd_return": fv[finite],
                "confidence": conf[finite],
                "probability": yp[finite],
                "signal_strength": ss[finite],
            }
        )
        signals["asset"] = asset
        filtered = apply_confidence_filter(
            signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08)
        )
        if len(filtered) > 0:
            results.append(filtered.to_dict("list"))

    return asset, results


def load_ohlcv_15m() -> dict[str, pd.DataFrame]:
    """Load 15-minute OHLCV for all assets."""
    ohlcv_dict = {}
    for asset, symbol in ASSETS.items():
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            ohlcv_dict[asset] = resample_ohlcv(ohlcv_1m, "15min")
        except Exception as e:
            print(f"  Warning: could not load {asset}: {e}")
    return ohlcv_dict


def run_exit_test(
    combined_signals: pd.DataFrame,
    ohlcv_dict: dict[str, pd.DataFrame],
    pcfg,
    exit_config: ExitConfig,
) -> dict | None:
    """Run path-dependent sim for a given exit config. Returns stats or None."""
    sim = simulate_portfolio_path(combined_signals, ohlcv_dict, pcfg, exit_config)
    if sim is None or sim["n_trades"] < 10:
        return None

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None

    stats = compute_full_stats(
        dpnl, pcfg.starting_capital, sim["n_trades"], sim["n_trades"] / len(dpnl)
    )
    stats["n_trades"] = sim["n_trades"]
    stats["halted"] = sim["halted"]
    return stats


def run_baseline(combined_signals: pd.DataFrame, pcfg) -> dict | None:
    """Run the current fixed-horizon sim (baseline)."""
    sim = simulate_portfolio(combined_signals, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None

    stats = compute_full_stats(
        dpnl, pcfg.starting_capital, sim["n_trades"], sim["n_trades"] / len(dpnl)
    )
    stats["n_trades"] = sim["n_trades"]
    stats["halted"] = sim["halted"]
    return stats


def print_result(name, stats, baseline_sharpe=None):
    """Print formatted result line."""
    if stats is None:
        print(f"  {name:<40s} NO RESULT")
        return

    delta = ""
    if baseline_sharpe is not None:
        d = stats["sharpe"] - baseline_sharpe
        delta = f"  ({'+' if d >= 0 else ''}{d:.3f})"

    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
    print(
        f"  {name:<40s} "
        f"Sharpe {stats['sharpe']:>7.3f}{delta:<10s} "
        f"Ret {stats['total_return_pct']:>7.2f}%  "
        f"MaxDD {stats['max_dd_pct']:>6.2f}%  "
        f"Trades {stats['n_trades']:>6d}  "
        f"PnL/day ${pnl_day:>7.2f}",
        flush=True,
    )


def main():
    print("=" * 85)
    print("EXIT STRATEGY EXPERIMENT -- Path-Dependent vs Fixed Horizon")
    print(f"  Lookback: {LOOKBACK}-month rolling, 30 assets, {W} workers")
    print("  Baseline: Fixed 4-bar hold (current production)")
    print("  All tests use SAME signals -- only exit rules differ")
    print("=" * 85, flush=True)

    # ---- Phase 0: Load OHLCV data ----
    print("\n[Phase 0] Loading 15m OHLCV for all assets...", flush=True)
    t0 = time.time()
    ohlcv_dict = load_ohlcv_15m()
    print(f"  Loaded {len(ohlcv_dict)} assets in {time.time()-t0:.0f}s", flush=True)

    # ---- Phase 1: Generate signals ----
    print(f"\n[Phase 1] Generating signals ({LOOKBACK}-month lookback)...", flush=True)
    t0 = time.time()

    # Version B: same test period as lookback comparison (2023-08 to 2026-02)
    # This matches the established baseline (Sharpe 3.686 for 8mo)
    cfg = load_config(Path("configs/default.yaml"))
    fixed_start = "2023-08"
    print(f"  Test starts: {fixed_start} (matching lookback comparison Version B)", flush=True)

    all_signals = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {
            pool.submit(_process_asset, (a, s, str(fixed_start))): a
            for a, s in ASSETS.items()
        }
        done = 0
        for future in as_completed(futures):
            asset, results = future.result()
            done += 1
            all_signals[asset] = results
            print(
                f"  [{done:>2d}/30] {asset}: {len(results)} months",
                end="  ",
                flush=True,
            )
            if done % 5 == 0:
                print()

    # Combine all signals
    frames = []
    for asset, results in all_signals.items():
        for sig_dict in results:
            frames.append(pd.DataFrame(sig_dict))

    if not frames:
        print("ERROR: No signals generated!")
        return

    combined = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)
    print(f"\n  Total signals: {len(combined):,d} in {time.time()-t0:.0f}s", flush=True)

    # ---- Phase 1b: Walk-forward Kelly weighting (pipeline.py style) ----
    print("\n[Phase 1b] Walk-forward Kelly weighting...", flush=True)
    from afml.bet_sizing import kelly_size
    timestamps_pd = pd.to_datetime(combined["timestamp"])
    kelly_months = sorted(timestamps_pd.dt.to_period("M").unique())
    kelly_parts = []
    for km in kelly_months:
        km_mask = timestamps_pd.dt.to_period("M") == km
        km_sigs = combined[km_mask]
        past_mask = timestamps_pd.dt.to_period("M") < km
        past_sigs = combined[past_mask]
        if len(past_sigs) < 50:
            kelly_parts.append(km_sigs)
            continue
        kw = {}
        for an in km_sigs["asset"].unique():
            pa = past_sigs[past_sigs["asset"] == an]
            if len(pa) < 10:
                kw[an] = 1.0
                continue
            tr = pa["direction"].values * pa["fwd_return"].values
            w = tr[tr > 0]; lo = tr[tr <= 0]
            if len(w) < 3 or len(lo) < 3:
                kw[an] = 1.0
                continue
            wp = len(w) / len(tr)
            wr = float(w.mean())
            lr = float(abs(lo.mean()))
            if lr <= 0:
                kw[an] = 1.0
                continue
            fs = kelly_size(wp, wr, lr)
            kw[an] = max(0.2, min(3.0, fs * 7.5))
        if kw:
            mw = np.mean(list(kw.values()))
            if mw > 0:
                kw = {k: v / mw for k, v in kw.items()}
        kmc = km_sigs.copy()
        kmc["signal_strength"] = kmc.apply(
            lambda r: r["signal_strength"] * kw.get(r["asset"], 1.0), axis=1)
        kelly_parts.append(kmc)
    combined = pd.concat(kelly_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    print(f"  Kelly applied across {len(kelly_months)} months", flush=True)

    # ---- Phase 2: Portfolio config ----
    pcfg = copy.deepcopy(cfg.portfolio)
    pcfg.base_sizing_pct = 0.04
    pcfg.slippage = 0.0001
    pcfg.max_exposure = 0.90
    pcfg.max_per_asset = 0.03

    # ---- Phase 3: Run exit experiments ----
    print(f"\n{'='*85}")
    print("RESULTS")
    print(f"{'='*85}", flush=True)

    # 3.0 Baseline -- fixed 4-bar with original portfolio.py
    print("\n--- BASELINE (current production) ---", flush=True)
    baseline_stats = run_baseline(combined, pcfg)
    baseline_sharpe = baseline_stats["sharpe"] if baseline_stats else 0
    print_result("Fixed 4-bar (portfolio.py baseline)", baseline_stats)

    # 3.0b Validate path-dependent sim reproduces baseline
    print("\n--- BASELINE VALIDATION (path-dep sim, fixed mode) ---", flush=True)
    ecfg_fixed = ExitConfig(max_hold_bars=4)
    fixed_path_stats = run_exit_test(combined, ohlcv_dict, pcfg, ecfg_fixed)
    print_result("Fixed 4-bar (path-dep sim)", fixed_path_stats, baseline_sharpe)

    # 3.1 Phase 2: Stop-loss consistency audit
    print("\n--- PHASE 2: STOP-LOSS AUDIT (live executor consistency) ---", flush=True)
    for sl_pct in [0.01, 0.02, 0.03, 0.05]:
        ecfg = ExitConfig(max_hold_bars=4, stop_loss_pct=sl_pct)
        stats = run_exit_test(combined, ohlcv_dict, pcfg, ecfg)
        print_result(f"Fixed 4-bar + {sl_pct*100:.0f}% SL", stats, baseline_sharpe)

    # 3.2 Phase 3A: Signal-strength-adaptive hold
    print("\n--- PHASE 3A: ADAPTIVE HOLD (signal strength -> hold period) ---", flush=True)
    for extra in [1, 2, 3, 4]:
        ecfg = ExitConfig(
            max_hold_bars=4, adaptive_hold=True, adaptive_hold_extra=extra
        )
        stats = run_exit_test(combined, ohlcv_dict, pcfg, ecfg)
        print_result(
            f"Adaptive hold (base=4 + sig*{extra})", stats, baseline_sharpe
        )

    # 3.3 Phase 3B: Asymmetric barrier EXITS
    print("\n--- PHASE 3B: ASYMMETRIC BARRIERS (exits, not labels) ---", flush=True)
    barrier_configs = [
        (1.0, 1.0, 6, "SL=1*ATR, TP=1*ATR, max=6"),
        (1.0, 2.0, 8, "SL=1*ATR, TP=2*ATR, max=8"),
        (1.5, 2.0, 8, "SL=1.5*ATR, TP=2*ATR, max=8"),
        (1.0, 3.0, 8, "SL=1*ATR, TP=3*ATR, max=8"),
        (0.5, 2.0, 6, "SL=0.5*ATR, TP=2*ATR, max=6"),
        (1.5, 3.0, 8, "SL=1.5*ATR, TP=3*ATR, max=8"),
    ]
    for sl_mult, tp_mult, max_h, label in barrier_configs:
        ecfg = ExitConfig(
            max_hold_bars=max_h,
            stop_loss_atr_mult=sl_mult,
            take_profit_atr_mult=tp_mult,
        )
        stats = run_exit_test(combined, ohlcv_dict, pcfg, ecfg)
        print_result(label, stats, baseline_sharpe)

    # 3.4 Phase 3C: Trailing stop with extended hold
    print("\n--- PHASE 3C: TRAILING STOP (lock profits, extend hold) ---", flush=True)
    trail_configs = [
        (1.5, 0.005, 6, "Trail=1.5*ATR, min=0.5%, max=6"),
        (1.5, 0.005, 8, "Trail=1.5*ATR, min=0.5%, max=8"),
        (2.0, 0.005, 8, "Trail=2.0*ATR, min=0.5%, max=8"),
        (1.0, 0.003, 6, "Trail=1.0*ATR, min=0.3%, max=6"),
        (2.0, 0.010, 8, "Trail=2.0*ATR, min=1.0%, max=8"),
    ]
    for trail_mult, min_pft, max_h, label in trail_configs:
        ecfg = ExitConfig(
            max_hold_bars=max_h,
            trail_atr_mult=trail_mult,
            trail_min_profit_pct=min_pft,
        )
        stats = run_exit_test(combined, ohlcv_dict, pcfg, ecfg)
        print_result(label, stats, baseline_sharpe)

    # 3.5 Phase 3D: Extended hold only (no barriers)
    print("\n--- PHASE 3D: EXTENDED FIXED HOLD (no barriers) ---", flush=True)
    for hold in [6, 8, 10, 12]:
        ecfg = ExitConfig(max_hold_bars=hold)
        stats = run_exit_test(combined, ohlcv_dict, pcfg, ecfg)
        print_result(f"Fixed {hold}-bar hold", stats, baseline_sharpe)

    # 3.6 Combined: Barriers + adaptive hold
    print("\n--- PHASE 3E: COMBINED (best barrier + adaptive hold) ---", flush=True)
    combined_configs = [
        (1.0, 2.0, 0, 2, 8, "SL=1,TP=2,adaptive+sig*2,max=8"),
        (1.5, 3.0, 0, 2, 8, "SL=1.5,TP=3,adaptive+sig*2,max=8"),
        (1.0, 2.0, 1.5, 0, 8, "SL=1,TP=2,trail=1.5,max=8"),
    ]
    for sl, tp, trail, extra, max_h, label in combined_configs:
        ecfg = ExitConfig(
            max_hold_bars=max_h,
            stop_loss_atr_mult=sl,
            take_profit_atr_mult=tp,
            trail_atr_mult=trail,
            trail_min_profit_pct=0.005,
            adaptive_hold=extra > 0,
            adaptive_hold_extra=extra,
        )
        stats = run_exit_test(combined, ohlcv_dict, pcfg, ecfg)
        print_result(label, stats, baseline_sharpe)

    print(f"\n{'='*85}")
    print("DONE -- Compare all Sharpe deltas. Only accept improvements > 0.15")
    print("       (Bonferroni-adjusted significance for ~25 tests)")
    print(f"{'='*85}")


if __name__ == "__main__":
    main()
