"""Honest Validation Suite — Multiple holdout tests + fixed CPCV.

1. Multiple holdout tests: 6/12/18/21 months training, test on remainder
2. CPCV with confidence filter (FIXED bug: was missing filter + wrong slippage)
"""
import sys
sys.path.insert(0, "src")

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import copy, warnings, os, time
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
CONF_THRESH = 0.08
SLIPPAGE = 0.0001  # 1 bps (production default)

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


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def _holdout_asset(args):
    """Process one asset with strict holdout. Subprocess."""
    asset, symbol, train_end_str = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    cfg = load_config(Path("configs/default.yaml"))
    train_end = pd.Timestamp(train_end_str, tz="UTC")

    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        features = compute_features(ohlcv_15m, cfg.features)
        feature_cols = cfg.features.feature_list
        feat = features[feature_cols].copy()

        close = ohlcv_15m["close"]
        fwd_ret = close.pct_change(4).shift(-4)
        vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

        cutoff = train_end.tz_localize(None) if ohlcv_15m.index.tz is None else train_end

        # Train
        train_feat = feat[feat.index < cutoff].dropna()
        train_y_all = fwd_ret.reindex(train_feat.index).dropna()
        train_valid = train_feat.index.intersection(train_y_all.index)

        train_vol = vol.loc[vol.index.intersection(train_valid)].dropna()
        if len(train_vol) < 50 or len(train_valid) < 500:
            return asset, None

        vol_thresh = train_vol.quantile(0.50)

        # Winsorize from training
        for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
            if col in feat.columns:
                lo = train_feat[col].quantile(0.01)
                hi = train_feat[col].quantile(0.99)
                feat[col] = feat[col].clip(lo, hi)

        X_train = feat.loc[train_valid]
        y_train = (train_y_all.loc[train_valid] > 0).astype(int)
        sw = compute_sample_weights(X_train.index.values.astype(np.int64), fwd_bars=4)

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        model = make_logreg()
        model.fit(Xtr, y_train, sample_weight=sw)

        # Test (AFTER train_end only)
        test_feat = feat[feat.index >= cutoff].dropna()
        test_fwd = fwd_ret.reindex(test_feat.index).dropna()
        test_valid = test_feat.index.intersection(test_fwd.index)

        test_vol = vol.reindex(test_valid)
        hv = (test_vol > vol_thresh).fillna(False)
        test_valid = test_valid[hv.reindex(test_valid).fillna(False).values]

        if len(test_valid) < 20:
            return asset, None

        Xte = scaler.transform(feat.loc[test_valid])
        yp = model.predict_proba(Xte)[:, 1]

        conf = np.abs(yp - 0.5)
        dirs = np.where(yp > 0.5, 1.0, -1.0)
        sigs = get_signal(yp, num_classes=2)
        sd = discretize_signal(sigs, step_size=0.1)
        ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
        fv = fwd_ret.reindex(test_valid).values
        finite = np.isfinite(fv)
        if not finite.any():
            return asset, None

        signals = pd.DataFrame({
            "timestamp": test_valid[finite], "direction": dirs[finite],
            "fwd_return": fv[finite], "confidence": conf[finite],
            "probability": yp[finite], "signal_strength": ss[finite],
        })
        signals["asset"] = asset

        # Confidence filter
        conf_cfg = ConfidenceFilterConfig(mode="absolute", threshold=CONF_THRESH)
        filtered = apply_confidence_filter(signals, conf_cfg)

        return asset, filtered.to_dict("list") if len(filtered) > 0 else None
    except Exception:
        return asset, None


def run_holdout(train_end_str, label):
    """Run holdout test with given training cutoff."""
    all_signals = []
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {
            pool.submit(_holdout_asset, (a, s, train_end_str)): a
            for a, s in ASSETS.items()
        }
        for future in as_completed(futures):
            asset, sig = future.result()
            if sig is not None:
                all_signals.append(sig)

    if not all_signals:
        print(f"  {label:<55s} NO SIGNALS", flush=True)
        return

    frames = [pd.DataFrame(d) for d in all_signals]
    filtered = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)

    cfg = load_config(Path("configs/default.yaml"))
    pcfg = copy.deepcopy(cfg.portfolio)
    pcfg.base_sizing_pct = 0.04
    pcfg.slippage = SLIPPAGE
    pcfg.max_exposure = 0.90
    pcfg.max_per_asset = 0.03

    sim = simulate_portfolio(filtered, pcfg)
    if sim is None or sim["n_trades"] < 10:
        print(f"  {label:<55s} TOO FEW TRADES", flush=True)
        return

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        print(f"  {label:<55s} INSUFFICIENT DAYS", flush=True)
        return

    stats = compute_full_stats(dpnl, pcfg.starting_capital,
                                sim["n_trades"], sim["n_trades"] / len(dpnl))
    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)

    print(
        f"  {label:<55s} Sharpe={stats['sharpe']:>6.3f}  "
        f"Ret={stats['total_return_pct']:>7.2f}%  DD={stats['max_dd_pct']:>5.2f}%  "
        f"Trades={sim['n_trades']:>6d}  Days={stats['n_days']:>4d}  "
        f"PnL/day=${pnl_day:>6.2f}",
        flush=True,
    )


def run_cpcv():
    """Fixed CPCV with confidence filter + correct slippage."""
    print("\n" + "=" * 80)
    print("CPCV: 10-FOLD PURGED CV (FIXED — with conf filter + 1 bps slippage)")
    print("6 core assets: BTC, ETH, XRP, BNB, SUI, LINK")
    print("=" * 80, flush=True)

    cfg = load_config(Path("configs/default.yaml"))
    core = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
            "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT"}

    for asset, symbol in core.items():
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        except Exception:
            continue

        features = compute_features(ohlcv_15m, cfg.features)
        feature_cols = cfg.features.feature_list
        feat = features[feature_cols].copy()

        close = ohlcv_15m["close"]
        fwd_ret = close.pct_change(4).shift(-4)
        vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

        for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
            if col in feat.columns:
                lo = feat[col].quantile(0.01)
                hi = feat[col].quantile(0.99)
                feat[col] = feat[col].clip(lo, hi)

        fc = feat.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        N_FOLDS = 10
        EMBARGO = 5
        fold_size = len(valid) // N_FOLDS
        fold_sharpes = []

        for fold_i in range(N_FOLDS):
            test_s = fold_i * fold_size
            test_e = (fold_i + 1) * fold_size if fold_i < N_FOLDS - 1 else len(valid)
            test_idx = valid[test_s:test_e]

            purge_s = max(0, test_s - EMBARGO)
            purge_e = min(len(valid), test_e + EMBARGO)
            train_idx = valid[:purge_s].append(valid[purge_e:])

            if len(train_idx) < 500 or len(test_idx) < 50:
                continue

            # Vol filter from training
            tv = vol.loc[vol.index.intersection(train_idx)].dropna()
            if len(tv) < 20:
                continue
            vt = tv.quantile(0.50)

            # Apply vol filter to test
            test_vol = vol.reindex(test_idx)
            hv = (test_vol > vt).fillna(False)
            test_filtered_idx = test_idx[hv.reindex(test_idx).fillna(False).values]
            if len(test_filtered_idx) < 20:
                continue

            X_train = fc.loc[train_idx]
            y_train = (y_all.loc[train_idx] > 0).astype(int)
            sw = compute_sample_weights(X_train.index.values.astype(np.int64), fwd_bars=4)

            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_train)
            model = make_logreg()
            model.fit(Xtr, y_train, sample_weight=sw)

            Xte = scaler.transform(fc.loc[test_filtered_idx])
            yp = model.predict_proba(Xte)[:, 1]
            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)

            # CONFIDENCE FILTER (the fix!)
            conf_mask = conf >= CONF_THRESH
            if conf_mask.sum() < 5:
                continue

            fwd_vals = fwd_ret.reindex(test_filtered_idx).values
            finite = np.isfinite(fwd_vals)
            combined_mask = conf_mask & finite

            if combined_mask.sum() < 5:
                continue

            # Trade returns WITH confidence filter applied
            trade_rets = dirs[combined_mask] * fwd_vals[combined_mask] - SLIPPAGE

            daily = pd.Series(
                trade_rets, index=test_filtered_idx[combined_mask]
            ).resample("D").sum()
            daily = daily[daily != 0]
            if len(daily) < 5:
                continue

            fold_sharpe = daily.mean() / daily.std() * np.sqrt(365.25) if daily.std() > 0 else 0
            fold_sharpes.append(round(fold_sharpe, 3))

        if fold_sharpes:
            arr = np.array(fold_sharpes)
            pos = sum(1 for s in arr if s > 0)
            print(f"  {asset}: {len(fold_sharpes)}/{N_FOLDS} folds, "
                  f"mean={arr.mean():.3f} median={np.median(arr):.3f} "
                  f"pos={pos}/{len(arr)} ({100*pos/len(arr):.0f}%) "
                  f"range=[{arr.min():.3f}, {arr.max():.3f}]", flush=True)
        else:
            print(f"  {asset}: no valid folds", flush=True)


def main():
    print("=" * 80)
    print(f"HONEST VALIDATION SUITE ({W} workers)")
    print("30 assets | conf >= 0.08 | 1 bps slippage | 4% sizing")
    print("=" * 80, flush=True)

    # =====================================================================
    # 1. Multiple Holdout Tests
    # =====================================================================
    print("\n" + "=" * 80)
    print("MULTIPLE HOLDOUT TESTS (varying training length)")
    print("=" * 80, flush=True)

    holdouts = [
        ("2024-09-01", "A: 6mo train (2024-03 to 08) | 18mo test"),
        ("2025-03-01", "B: 12mo train (2024-03 to 2025-02) | 12mo test"),
        ("2025-09-01", "C: 18mo train (2024-03 to 2025-08) | 6mo test"),
        ("2025-12-01", "D: 21mo train (2024-03 to 2025-11) | 3mo test"),
    ]

    for train_end, label in holdouts:
        t0 = time.time()
        run_holdout(train_end, label)

    # =====================================================================
    # 2. CPCV (fixed)
    # =====================================================================
    run_cpcv()

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
