"""Quick baseline validation - verify we reproduce the lookback comparison numbers.

Uses the exact same code path as lookback_fair.py + walk-forward Kelly from pipeline.py.
Target: Sharpe ~3.686 (8mo, Version B 2023-08 to 2026-02).
"""
import sys
sys.path.insert(0, "src")

import copy, os, time, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from signal_engine.config import load_config, ConfidenceFilterConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from afml.bet_sizing import get_signal, discretize_signal, kelly_size

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

LOOKBACK = 8
FIXED_START = "2023-08"  # Version B: same test period as lookback comparison


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def _process_asset(args):
    """Identical to lookback_fair.py _process_asset for lb=8."""
    asset, symbol = args
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
            feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))

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
        end=end_date.to_period("M"), freq="M",
    )

    for month in all_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

        month_str = str(month)
        # Generate ALL months (filter at aggregation, same as lookback_fair.py)

        te = valid[(valid >= ms) & (valid <= me)]
        vol_train_start = ms - pd.DateOffset(months=3)
        vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms)].dropna()
        if len(vol_train) < 50:
            continue
        vt = vol_train.quantile(0.50)
        test_vol = vol.reindex(te)
        hv = (test_vol > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5:
            continue

        train_start = ms - pd.DateOffset(months=LOOKBACK)
        if train_start < start_date:
            continue

        train_cutoff = ms - embargo_td
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
        signals["asset"] = asset
        filtered = apply_confidence_filter(
            signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
        if len(filtered) > 0:
            results.append((month_str, filtered.to_dict("list")))

    return asset, results


def apply_walkforward_kelly(combined):
    """Walk-forward Kelly weighting -- identical to pipeline.py lines 261-325."""
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(combined["timestamp"])
    months = sorted(timestamps.dt.to_period("M").unique())

    kelly_adjusted = []
    for month in months:
        month_mask = timestamps.dt.to_period("M") == month
        month_signals = combined[month_mask]
        past_mask = timestamps.dt.to_period("M") < month
        past_signals = combined[past_mask]

        if len(past_signals) < 50:
            kelly_adjusted.append(month_signals)
            continue

        kelly_weights = {}
        for asset_name in month_signals["asset"].unique():
            past_asset = past_signals[past_signals["asset"] == asset_name]
            if len(past_asset) < 10:
                kelly_weights[asset_name] = 1.0
                continue
            trade_rets = past_asset["direction"].values * past_asset["fwd_return"].values
            wins = trade_rets[trade_rets > 0]
            losses = trade_rets[trade_rets <= 0]
            if len(wins) < 3 or len(losses) < 3:
                kelly_weights[asset_name] = 1.0
                continue
            w_prob = len(wins) / len(trade_rets)
            w_ret = float(wins.mean())
            l_ret = float(abs(losses.mean()))
            if l_ret <= 0:
                kelly_weights[asset_name] = 1.0
                continue
            f_star = kelly_size(w_prob, w_ret, l_ret)
            kelly_weights[asset_name] = max(0.2, min(3.0, f_star * 7.5))

        if kelly_weights:
            mean_w = np.mean(list(kelly_weights.values()))
            if mean_w > 0:
                kelly_weights = {k: v / mean_w for k, v in kelly_weights.items()}

        month_copy = month_signals.copy()
        month_copy["signal_strength"] = month_copy.apply(
            lambda r: r["signal_strength"] * kelly_weights.get(r["asset"], 1.0),
            axis=1,
        )
        kelly_adjusted.append(month_copy)

    return pd.concat(kelly_adjusted, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def run_sim(combined, pcfg, label):
    """Run portfolio sim and print stats."""
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10:
        print(f"  {label}: NO RESULT")
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        print(f"  {label}: TOO FEW DAYS")
        return None
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"], sim["n_trades"] / len(dpnl))
    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
    print(
        f"  {label:<35s} Sharpe {stats['sharpe']:>7.3f}  "
        f"Ret {stats['total_return_pct']:>7.2f}%  MaxDD {stats['max_dd_pct']:>6.2f}%  "
        f"Trades {sim['n_trades']:>6d}  Days {stats['n_days']:>4d}  PnL/day ${pnl_day:>7.2f}"
    )
    return stats


def main():
    print("=" * 80)
    print("BASELINE VALIDATION")
    print(f"  8-month rolling lookback, 30 assets, Version B (start {FIXED_START})")
    print(f"  Config: conf>=0.08, 1bps slippage, 4% sizing, 90% exposure, 3% per-asset")
    print(f"  Target: Sharpe ~3.686, Return ~186%, MaxDD ~9.08%, Trades ~85426")
    print("=" * 80, flush=True)

    # Phase 1: Generate signals
    print(f"\n[1] Generating signals ({W} workers)...", flush=True)
    t0 = time.time()

    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_process_asset, (a, s)): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, results = f.result()
            done += 1
            all_results[asset] = results
            print(f"  [{done:>2d}/30] {asset}: {len(results)} months", flush=True)

    # Aggregate (same as lookback_fair.py aggregate_and_report)
    sigs = []
    for asset_results in all_results.values():
        for month_str, sig_dict in asset_results:
            if month_str < FIXED_START:
                continue
            sigs.append(pd.DataFrame(sig_dict))

    if not sigs:
        print("ERROR: No signals!")
        return

    combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
    print(f"  Signals: {len(combined):,d} in {time.time()-t0:.0f}s", flush=True)

    # Portfolio config (same as lookback_fair.py)
    cfg = load_config(Path("configs/default.yaml"))
    pcfg = copy.deepcopy(cfg.portfolio)
    pcfg.base_sizing_pct = 0.04
    pcfg.slippage = 0.0001
    pcfg.max_exposure = 0.90
    pcfg.max_per_asset = 0.03

    # Phase 2: Test configurations
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}", flush=True)

    print("\n--- Without Kelly (matches lookback_fair.py exactly) ---")
    run_sim(combined, pcfg, "No Kelly")

    print("\n--- With walk-forward Kelly (pipeline.py style) ---")
    combined_kelly = apply_walkforward_kelly(combined)
    run_sim(combined_kelly, pcfg, "Walk-forward Kelly")

    print(f"\n{'='*80}")
    print("TARGET:  Sharpe  3.686  Ret 186.27%  MaxDD  9.08%  Trades 85426  Days  885  PnL/day $ 21.05")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
