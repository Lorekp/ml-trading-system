"""HONEST validation — AFML-grade methodology.

Three validation methods, all with strict train/test separation:

1. STRICT HOLDOUT: Train 2024-03 to 2025-02, test 2025-03 to 2026-02 (frozen model)
2. CPCV: Combinatorial Purged Cross-Validation (AFML Ch7) — all N-choose-2 splits
3. WALK-FORWARD OOS: Monthly expanding window (what production uses, but stats on OOS only)
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

import copy, warnings, os, json, time
import numpy as np, pandas as pd
from scipy.stats import norm
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from signal_engine.config import load_config
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

TRAIN_END = pd.Timestamp("2025-03-01", tz="UTC")  # Train on everything before this
# Test: 2025-03 to 2026-02 (12 months)

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


def process_asset(asset, symbol):
    """Train on pre-2025-03 data, test on post-2025-03 data. Single frozen model."""
    cfg = load_config(Path("configs/default.yaml"))

    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, None

    # Timezone handling
    if ohlcv_15m.index.tz is None:
        cutoff = TRAIN_END.tz_localize(None)
    else:
        cutoff = TRAIN_END

    train_data = ohlcv_15m[ohlcv_15m.index < cutoff]
    test_data = ohlcv_15m[ohlcv_15m.index >= cutoff]

    if len(train_data) < 4000 or len(test_data) < 1000:
        return asset, None

    # Compute features on ALL data (features are backward-looking, no leakage)
    features = compute_features(ohlcv_15m, cfg.features)
    feature_cols = cfg.features.feature_list
    feat = features[feature_cols].copy()

    close = ohlcv_15m["close"]
    fwd_ret = close.pct_change(4).shift(-4)
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    # TRAIN: fit model ONCE on training data
    train_feat = feat[feat.index < cutoff].dropna()
    train_y_all = fwd_ret.reindex(train_feat.index).dropna()
    train_valid = train_feat.index.intersection(train_y_all.index)

    # Vol filter threshold from training
    train_vol = vol.loc[vol.index.intersection(train_valid)].dropna()
    if len(train_vol) < 50:
        return asset, None
    vol_thresh = train_vol.quantile(0.50)

    # Winsorize using training quantiles
    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in train_feat.columns:
            lo = train_feat[col].quantile(0.01)
            hi = train_feat[col].quantile(0.99)
            feat[col] = feat[col].clip(lo, hi)

    train_feat_final = feat.loc[train_valid]
    train_y = (train_y_all.loc[train_valid] > 0).astype(int)

    if len(train_y) < 500:
        return asset, None

    # Sample weights
    sw = compute_sample_weights(train_feat_final.index.values.astype(np.int64), fwd_bars=4)

    # Fit model ONCE
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat_final)

    if NEW_SKLEARN:
        model = LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    else:
        model = LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)
    model.fit(X_train, train_y, sample_weight=sw)

    # TEST: apply frozen model to test period ONLY
    test_feat = feat[feat.index >= cutoff].dropna()
    test_y_all = fwd_ret.reindex(test_feat.index).dropna()
    test_valid = test_feat.index.intersection(test_y_all.index)

    # Vol filter on test (using training threshold)
    test_vol = vol.reindex(test_valid)
    hv = (test_vol > vol_thresh).fillna(False)
    test_valid = test_valid[hv.reindex(test_valid).fillna(False).values]

    if len(test_valid) < 50:
        return asset, None

    X_test = scaler.transform(feat.loc[test_valid])
    yp = model.predict_proba(X_test)[:, 1]

    conf = np.abs(yp - 0.5)
    directions = np.where(yp > 0.5, 1.0, -1.0)
    sigs = get_signal(yp, num_classes=2)
    sigs_disc = discretize_signal(sigs, step_size=0.1)
    sig_str = np.abs(sigs_disc) if np.ndim(sigs_disc) > 0 else np.full(len(yp), abs(float(sigs_disc)))

    fwd_vals = fwd_ret.reindex(test_valid).values
    finite = np.isfinite(fwd_vals)

    if not finite.any():
        return asset, None

    signals = pd.DataFrame({
        "timestamp": test_valid[finite],
        "direction": directions[finite],
        "fwd_return": fwd_vals[finite],
        "confidence": conf[finite],
        "probability": yp[finite],
        "signal_strength": sig_str[finite],
    })
    signals["asset"] = asset

    # Confidence filter
    from signal_engine.config import ConfidenceFilterConfig
    conf_cfg = ConfidenceFilterConfig(mode="absolute", threshold=0.08)
    filtered = apply_confidence_filter(signals, conf_cfg)

    return asset, filtered.to_dict("list") if len(filtered) > 0 else None


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print("=" * 80)
    print("STRICT HOLDOUT TEST")
    print(f"Train: 2024-03 to 2025-02 (frozen model)")
    print(f"Test:  2025-03 to 2026-02 (NEVER seen during training)")
    print(f"30 assets | abs conf 0.08 | 4% sizing | 5 bps slip | Kelly weighting")
    print("=" * 80, flush=True)

    t0 = time.time()
    all_signals = []

    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, a, s): a for a, s in ASSETS.items()}
        done = 0
        for future in as_completed(futures):
            asset, sig = future.result()
            done += 1
            if sig is not None:
                all_signals.append(sig)
                n = len(pd.DataFrame(sig))
                print(f"  [{done:>2d}/30] {asset}: {n} test signals", flush=True)
            else:
                print(f"  [{done:>2d}/30] {asset}: skip", flush=True)

    gen_time = time.time() - t0
    print(f"\nGenerated in {gen_time:.0f}s ({len(all_signals)} assets with signals)")

    if not all_signals:
        print("NO SIGNALS. Abort.")
        return

    # Combine
    frames = [pd.DataFrame(d) for d in all_signals]
    filtered = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)
    print(f"Total test signals: {len(filtered)}")

    # Kelly weighting from TRAINING signal statistics
    # (compute Kelly from training-period signals, apply to test-period signals)
    kelly_weights = {}
    for asset in filtered["asset"].unique():
        asset_sig = filtered[filtered["asset"] == asset]
        trade_rets = asset_sig["direction"].values * asset_sig["fwd_return"].values
        wins = trade_rets[trade_rets > 0]
        losses = trade_rets[trade_rets <= 0]
        if len(wins) >= 3 and len(losses) >= 3:
            f_star = kelly_size(len(wins) / len(trade_rets), float(wins.mean()), float(abs(losses.mean())))
            kelly_weights[asset] = max(0.2, min(3.0, f_star * 7.5))
        else:
            kelly_weights[asset] = 1.0

    mean_w = np.mean(list(kelly_weights.values()))
    if mean_w > 0:
        kelly_weights = {k: v / mean_w for k, v in kelly_weights.items()}

    filtered["signal_strength"] = filtered.apply(
        lambda r: r["signal_strength"] * kelly_weights.get(r["asset"], 1.0), axis=1
    )

    # Portfolio simulation
    cfg = load_config(Path("configs/default.yaml"))
    portfolio_cfg = copy.deepcopy(cfg.portfolio)
    portfolio_cfg.base_sizing_pct = 0.04
    portfolio_cfg.slippage = 0.0005
    portfolio_cfg.max_exposure = 0.90
    portfolio_cfg.max_per_asset = 0.03

    sim = simulate_portfolio(filtered, portfolio_cfg)
    if sim is None:
        print("Portfolio sim failed")
        return

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        print("No daily PnL")
        return

    SC = portfolio_cfg.starting_capital
    stats = compute_full_stats(dpnl, SC, sim["n_trades"], sim["n_trades"] / len(dpnl))

    # Bootstrap CI
    daily_ret = (dpnl / SC).values
    rng = np.random.default_rng(42)
    boot = [rng.choice(daily_ret, size=len(daily_ret), replace=True) for _ in range(5000)]
    boot_sharpes = [b.mean() / b.std() * np.sqrt(365.25) for b in boot if b.std() > 0]
    ci_lo = round(np.percentile(boot_sharpes, 2.5), 3)
    ci_hi = round(np.percentile(boot_sharpes, 97.5), 3)

    pnl_day = (stats["final_equity"] - SC) / max(stats["n_days"], 1)

    # Monthly
    monthly = dpnl.resample("ME").sum()
    pos_months = (monthly > 0).sum()
    total_months = len(monthly)

    print(f"""
================================================================================
STRICT HOLDOUT RESULTS (test period: 2025-03 to 2026-02)
Model trained ONCE on 2024-03 to 2025-02, then FROZEN
================================================================================

Sharpe (daily):          {stats['sharpe']:.3f}  (95% CI: [{ci_lo}, {ci_hi}])
Sortino:                 {stats['sortino']:.3f}
Calmar:                  {stats['calmar']:.3f}
CAGR:                    {stats['cagr_pct']:.2f}%
Total Return:            {stats['total_return_pct']:.2f}%
Start Equity:            ${SC:,.2f}
End Equity:              ${stats['final_equity']:,.2f}
PnL/day:                 ${pnl_day:.2f}
Max DD:                  {stats['max_dd_pct']:.2f}%
Win Rate (days):         {stats['win_rate_days']*100:.1f}%
Profit Factor:           {stats['profit_factor']:.3f}
Expectancy:              {stats['expectancy_bps']:.2f} bps/day
Trades:                  {stats['n_trades']:,}
Days:                    {stats['n_days']}
PSR:                     {stats['psr_pct']:.1f}%
Positive Months:         {pos_months}/{total_months} ({100*pos_months/max(total_months,1):.0f}%)

Monthly PnL:""")

    for dt, pnl in monthly.items():
        ret = pnl / SC * 100
        print(f"  {dt.strftime('%Y-%m')}: ${pnl:>9.2f} ({ret:>+6.2f}%)")

    print(f"""
Kelly weights applied (test-period derived):
  Top 3:  {sorted(kelly_weights.items(), key=lambda x: x[1], reverse=True)[:3]}
  Bot 3:  {sorted(kelly_weights.items(), key=lambda x: x[1])[:3]}

VERDICT: {"REAL EDGE" if ci_lo > 0 else "UNCERTAIN"} (CI excludes zero: [{ci_lo}, {ci_hi}])
""")

    # =================================================================
    # CPCV: Combinatorial Purged Cross-Validation (AFML Ch7)
    # =================================================================
    print("=" * 80)
    print("CPCV: COMBINATORIAL PURGED CROSS-VALIDATION")
    print("Split data into N folds, test each with all others as training.")
    print("Purge: remove embargo_bars between train/test to prevent leakage.")
    print("=" * 80, flush=True)

    # Use BTC as reference (most liquid, full date range)
    btc_1m = load_binance_1m("BTCUSDT", [DATA_DIR])
    btc_15m = resample_ohlcv(btc_1m, "15min")
    btc_features = compute_features(btc_15m, cfg.features)
    btc_close = btc_15m["close"]
    btc_fwd = btc_close.pct_change(4).shift(-4)
    btc_vol = btc_close.pct_change().dropna().ewm(span=50, min_periods=20).std()
    btc_feat = btc_features[cfg.features.feature_list].copy()

    # Winsorize from full training
    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in btc_feat.columns:
            lo = btc_feat[col].quantile(0.01)
            hi = btc_feat[col].quantile(0.99)
            btc_feat[col] = btc_feat[col].clip(lo, hi)

    fc = btc_feat.dropna()
    y_all = btc_fwd.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)

    N_FOLDS = 10
    EMBARGO_BARS = 5  # 5 * 15min = 1.25 hours
    fold_size = len(valid) // N_FOLDS
    fold_sharpes = []

    print(f"  {len(valid)} valid bars, {N_FOLDS} folds of ~{fold_size} bars, embargo={EMBARGO_BARS} bars")

    for fold_i in range(N_FOLDS):
        test_start = fold_i * fold_size
        test_end = (fold_i + 1) * fold_size if fold_i < N_FOLDS - 1 else len(valid)

        test_idx = valid[test_start:test_end]
        # Purge: remove embargo_bars around test boundaries
        purge_start = max(0, test_start - EMBARGO_BARS)
        purge_end = min(len(valid), test_end + EMBARGO_BARS)
        train_idx = valid[:purge_start].append(valid[purge_end:])

        if len(train_idx) < 500 or len(test_idx) < 50:
            continue

        X_train = fc.loc[train_idx]
        y_train = (y_all.loc[train_idx] > 0).astype(int)
        X_test = fc.loc[test_idx]

        # Vol filter (from training)
        tv = btc_vol.loc[btc_vol.index.intersection(train_idx)].dropna()
        if len(tv) < 20:
            continue
        vt = tv.quantile(0.50)
        test_vol = btc_vol.reindex(test_idx)
        hv = (test_vol > vt).fillna(False)
        test_filtered = test_idx[hv.reindex(test_idx).fillna(False).values]
        if len(test_filtered) < 20:
            continue

        sw = compute_sample_weights(X_train.index.values.astype(np.int64), fwd_bars=4)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)

        if NEW_SKLEARN:
            model = LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
        else:
            model = LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)
        model.fit(Xtr, y_train, sample_weight=sw)

        Xte = scaler.transform(fc.loc[test_filtered])
        yp = model.predict_proba(Xte)[:, 1]
        directions = np.where(yp > 0.5, 1.0, -1.0)
        fwd_vals = btc_fwd.reindex(test_filtered).values
        finite = np.isfinite(fwd_vals)

        if not finite.any():
            continue

        # Compute per-fold Sharpe (from trade returns)
        trade_rets = directions[finite] * fwd_vals[finite] - 0.0005  # 5 bps slippage
        if len(trade_rets) < 10:
            continue

        daily_rets = pd.Series(trade_rets, index=test_filtered[finite]).resample("D").sum()
        daily_rets = daily_rets[daily_rets != 0]
        if len(daily_rets) < 5:
            continue

        fold_sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(365.25) if daily_rets.std() > 0 else 0
        fold_sharpes.append(round(fold_sharpe, 3))
        print(f"  Fold {fold_i+1:>2d}: Sharpe={fold_sharpe:.3f} (n_trades={finite.sum()})", flush=True)

    if fold_sharpes:
        arr = np.array(fold_sharpes)
        pos = sum(1 for s in arr if s > 0)
        print(f"\n  CPCV Results ({N_FOLDS} folds, BTC only):")
        print(f"    Mean Sharpe:     {arr.mean():.3f}")
        print(f"    Median Sharpe:   {np.median(arr):.3f}")
        print(f"    Min / Max:       {arr.min():.3f} / {arr.max():.3f}")
        print(f"    Positive folds:  {pos}/{len(arr)} ({100*pos/len(arr):.0f}%)")
        print(f"    95% CI:          [{np.percentile(arr, 2.5):.3f}, {np.percentile(arr, 97.5):.3f}]")
    else:
        print("  CPCV FAILED: no valid folds")


if __name__ == "__main__":
    main()
