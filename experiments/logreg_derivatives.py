"""Experiment 1: Derivatives-Enhanced LogReg

Compare 14-feature OHLCV-only LogReg vs 22-feature (OHLCV + derivatives) LogReg.
Uses same walk-forward pipeline but with added funding, OI, premium features.
"""

import sys, os, time, warnings
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
PCACHE = Path("data/cache_15m")
HL_DIR = Path("data/hl-asset-ctxs/hl-asset-ctxs")
HL_CACHE = Path("data/hl-asset-ctxs/cache_15m")

CAPITAL = 10000.0

# Universe: top assets by volume that also have HL derivatives data
UNIVERSE = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK", "APT",
    "SUI", "ADA", "BNB", "NEAR", "DOT", "FIL", "ARB", "FET",
    "WLD", "UNI", "AAVE", "FTM",
]

BASE_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]

DERIV_FEATURES = [
    "funding_zscore", "funding_zscore_long", "funding_cumulative",
    "oi_change_5", "oi_zscore", "oi_price_divergence",
    "premium_zscore", "impact_skew",
]

ALL_FEATURES = BASE_FEATURES + DERIV_FEATURES

# T1 LogReg config
FWD_BARS = 6
C_REG = 0.02
CONFIDENCE_THRESH = 0.10


def compute_base_features(ohlcv):
    """Compute 14 base OHLCV features (same as signal_engine)."""
    from signal_engine.config import FeatureConfig
    from signal_engine.features import compute_features
    fcfg = FeatureConfig(feature_list=BASE_FEATURES)
    return compute_features(ohlcv, fcfg)


def compute_deriv_features(deriv_15m, close):
    """Compute 8 derivatives features."""
    from afml.derivatives import funding_features, oi_features, premium_features, impact_features

    feat = pd.DataFrame(index=close.index)

    # Funding features
    if "funding" in deriv_15m.columns:
        ff = funding_features(deriv_15m["funding"], window_short=50, window_long=200)
        for col in ["funding_zscore", "funding_zscore_long", "funding_cumulative"]:
            if col in ff.columns:
                feat[col] = ff[col].reindex(close.index)

    # OI features
    if "open_interest" in deriv_15m.columns:
        of = oi_features(deriv_15m["open_interest"], close, window=50)
        for col in ["oi_change_5", "oi_zscore", "oi_price_divergence"]:
            if col in of.columns:
                feat[col] = of[col].reindex(close.index)

    # Premium features
    if "mark_px" in deriv_15m.columns and "oracle_px" in deriv_15m.columns:
        pf = premium_features(deriv_15m["mark_px"], deriv_15m["oracle_px"], window=50)
        if "premium_zscore" in pf.columns:
            feat["premium_zscore"] = pf["premium_zscore"].reindex(close.index)

    # Impact features
    if "impact_bid_px" in deriv_15m.columns and "impact_ask_px" in deriv_15m.columns:
        mid = deriv_15m.get("mid_px", (deriv_15m["impact_bid_px"] + deriv_15m["impact_ask_px"]) / 2)
        imf = impact_features(deriv_15m["impact_bid_px"], deriv_15m["impact_ask_px"], mid)
        if "impact_skew" in imf.columns:
            feat["impact_skew"] = imf["impact_skew"].reindex(close.index)

    return feat


def walk_forward_logreg(features, fwd_ret, feature_cols, asset_name, fwd_bars=FWD_BARS):
    """Walk-forward LogReg, returns signals DataFrame."""
    close_ret = fwd_ret
    n = len(features)
    if n < 96 * 180:  # need 6 months
        return pd.DataFrame()

    # Monthly periods
    start = features.index[96 * 180]
    end = features.index[-1]
    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")

    embargo_bars = fwd_bars + 1
    rows = []
    model = None; scaler = None; last_retrain = -99

    for mi, month in enumerate(months):
        ms = pd.Timestamp(month.start_time)
        me = pd.Timestamp(month.end_time)
        if features.index.tz is not None:
            ms = ms.tz_localize("UTC"); me = me.tz_localize("UTC")
        elif ms.tz is not None:
            ms = ms.tz_localize(None); me = me.tz_localize(None)

        # Retrain every 2 months
        if model is None or mi - last_retrain >= 2:
            # Training data: everything before this month minus embargo
            train_end = ms - pd.Timedelta(minutes=15 * embargo_bars)
            train_mask = features.index < train_end

            X_tr = features.loc[train_mask, feature_cols].values
            y_raw = close_ret.reindex(features.index[train_mask]).values
            y_tr = (y_raw > 0).astype(int)

            valid = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
            if valid.sum() < 500:
                continue

            X_tr = np.nan_to_num(X_tr[valid], nan=0, posinf=5, neginf=-5)
            y_tr = y_tr[valid]

            # Winsorize returns features
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)

            model = LogisticRegression(C=C_REG, penalty="l1", solver="saga",
                                       max_iter=1000, random_state=42)
            model.fit(X_tr, y_tr)
            last_retrain = mi

        # Predict for this month
        month_mask = (features.index >= ms) & (features.index <= me)
        X_mo = features.loc[month_mask, feature_cols].values
        y_mo_raw = close_ret.reindex(features.index[month_mask]).values

        valid = np.isfinite(X_mo).all(axis=1) & np.isfinite(y_mo_raw)
        if valid.sum() < 10: continue

        X_mo_v = np.nan_to_num(X_mo[valid], nan=0, posinf=5, neginf=-5)
        X_mo_s = scaler.transform(X_mo_v)
        probs = model.predict_proba(X_mo_s)[:, 1]

        timestamps = features.index[month_mask][valid]
        fwd_rets = y_mo_raw[valid]

        for j in range(len(timestamps)):
            p = float(probs[j])
            conf = abs(p - 0.5)
            if conf < CONFIDENCE_THRESH: continue
            direction = 1.0 if p > 0.5 else -1.0
            rows.append({
                "timestamp": timestamps[j],
                "asset": asset_name,
                "probability": p,
                "direction": direction,
                "confidence": conf,
                "fwd_return": float(fwd_rets[j]),
                "signal_strength": min(conf * 4, 1.0),
            })

    return pd.DataFrame(rows)


def simulate_portfolio(signals_df, capital=CAPITAL):
    """Simplified portfolio simulation matching T1 config."""
    if len(signals_df) < 10: return None

    signals_df = signals_df.sort_values("timestamp").reset_index(drop=True)

    equity = capital; daily_pnl = {}

    base_sizing = 0.30  # T1: 0.06 * 5x
    max_per_asset = 0.05; slippage = 0.0001; max_exposure = 0.90

    for _, row in signals_df.iterrows():
        size_pct = base_sizing * row["signal_strength"]
        size_pct = min(size_pct, max_per_asset)

        pnl = equity * size_pct * row["direction"] * row["fwd_return"]
        pnl -= equity * size_pct * slippage

        equity += pnl
        day = pd.Timestamp(row["timestamp"]).normalize()
        if hasattr(day, "tz") and day.tz is not None: day = day.tz_localize(None)
        daily_pnl[day] = daily_pnl.get(day, 0) + pnl

        # DD halt
        peak_equity = max(capital, equity)
        if (equity - peak_equity) / peak_equity < -0.10:
            break

    return pd.Series(daily_pnl).sort_index()


def metrics(dpnl, label, capital=CAPITAL):
    if dpnl is None or len(dpnl) < 5: return None
    cal = (dpnl.index[-1] - dpnl.index[0]).days + 1
    dr = dpnl / capital
    sr = dr.mean() / dr.std() * np.sqrt(365.25) if dr.std() > 0 else 0
    cum = dpnl.cumsum(); peak = (cum + capital).cummax()
    dd = abs(((cum + capital - peak) / peak).min()); wd = abs(dr.min())
    pnl_cal = cum.iloc[-1] / cal
    wr = (dpnl > 0).sum() / max((dpnl != 0).sum(), 1)
    print(f"  {label:<30s} SR={sr:>5.2f}  ${pnl_cal:>7.2f}/cal  DD={dd*100:>5.1f}%  WD={wd*100:>5.2f}%  WR={wr*100:.0f}%")
    return {"sr": sr, "pnl_cal": pnl_cal, "dd": dd, "wd": wd}


def main():
    t0 = time.time()
    print("=" * 110)
    print("DERIVATIVES-ENHANCED LOGREG — 14 features vs 22 features")
    print("=" * 110)

    from data.hl_derivatives import load_hl_asset_ctxs, resample_to_ohlcv_freq

    # Load and process each asset
    base_signals = []
    enhanced_signals = []
    base_auc = []; enh_auc = []
    oos_start = pd.Timestamp("2025-01-01", tz="UTC")

    for coin in UNIVERSE:
        ct0 = time.time()
        symbol = f"{coin}USDT"
        ohlcv_path = PCACHE / f"{symbol}.parquet"
        if not ohlcv_path.exists():
            print(f"  {coin}: no OHLCV"); continue
        ohlcv = pd.read_parquet(ohlcv_path)
        if len(ohlcv) < 96 * 365:
            print(f"  {coin}: too short"); continue

        # Base features
        base_feat = compute_base_features(ohlcv)

        # Derivatives
        cache_path = HL_CACHE / f"{coin}.parquet"
        try:
            deriv = load_hl_asset_ctxs(str(HL_DIR), coin=coin, cache_path=str(cache_path))
            deriv_15m = resample_to_ohlcv_freq(deriv, ohlcv.index, method="last")
            deriv_feat = compute_deriv_features(deriv_15m, ohlcv["close"])
        except Exception as e:
            print(f"  {coin}: no derivatives ({e})"); continue

        # Merge
        enhanced_feat = pd.concat([base_feat, deriv_feat], axis=1)

        # Forward return
        fwd_ret = ohlcv["close"].pct_change(FWD_BARS).shift(-FWD_BARS)

        # Determine available feature columns
        base_cols = [c for c in BASE_FEATURES if c in base_feat.columns]
        enh_cols = [c for c in ALL_FEATURES if c in enhanced_feat.columns]

        # Generate signals with both feature sets
        base_sigs = walk_forward_logreg(base_feat, fwd_ret, base_cols, coin)
        enh_sigs = walk_forward_logreg(enhanced_feat, fwd_ret, enh_cols, coin)

        # OOS AUC
        for sigs, auc_list, label in [(base_sigs, base_auc, "base"), (enh_sigs, enh_auc, "enh")]:
            if len(sigs) > 0:
                oos = sigs[pd.to_datetime(sigs["timestamp"]) >= oos_start]
                if len(oos) > 50:
                    y_true = (oos["fwd_return"] > 0).astype(int)
                    try:
                        auc = roc_auc_score(y_true, oos["probability"])
                        auc_list.append(auc)
                    except: pass

        n_base = len(base_sigs); n_enh = len(enh_sigs)
        n_enh_cols = len(enh_cols)
        base_signals.append(base_sigs)
        enhanced_signals.append(enh_sigs)

        print(f"  {coin}: {len(ohlcv)} bars, {n_enh_cols} features, "
              f"base={n_base} sigs, enh={n_enh} sigs [{time.time()-ct0:.0f}s]", flush=True)

    # Combine all signals
    base_all = pd.concat(base_signals, ignore_index=True) if base_signals else pd.DataFrame()
    enh_all = pd.concat(enhanced_signals, ignore_index=True) if enhanced_signals else pd.DataFrame()

    print(f"\nTotal: base={len(base_all)} signals, enhanced={len(enh_all)} signals")

    # AUC comparison
    if base_auc and enh_auc:
        print(f"\nOOS AUC (mean across assets):")
        print(f"  Base (14 feat):     {np.mean(base_auc):.4f} +/- {np.std(base_auc):.4f}")
        print(f"  Enhanced (22 feat): {np.mean(enh_auc):.4f} +/- {np.std(enh_auc):.4f}")
        print(f"  Delta:              {np.mean(enh_auc) - np.mean(base_auc):+.4f}")

    # Portfolio simulation — OOS only
    print(f"\n{'='*110}")
    print("PORTFOLIO SIMULATION (OOS: 2025-01+)")
    print(f"{'='*110}")

    oos_start_naive = oos_start.tz_localize(None)

    for label, sigs_df in [("Base 14-feat", base_all), ("Enhanced 22-feat", enh_all)]:
        if len(sigs_df) == 0: continue
        sigs_df["timestamp"] = pd.to_datetime(sigs_df["timestamp"])
        ts_naive = sigs_df["timestamp"].dt.tz_localize(None) if sigs_df["timestamp"].dt.tz is not None else sigs_df["timestamp"]
        oos_sigs = sigs_df[ts_naive >= oos_start_naive]
        dpnl = simulate_portfolio(oos_sigs)
        metrics(dpnl, label)

    # Per-asset AUC comparison
    if base_auc and enh_auc:
        print(f"\n--- Per-Asset OOS AUC ---")
        print(f"{'Asset':<8s} {'Base AUC':>10s} {'Enh AUC':>10s} {'Delta':>8s}")
        print("-" * 40)
        for i, coin in enumerate(UNIVERSE):
            if i < len(base_auc) and i < len(enh_auc):
                d = enh_auc[i] - base_auc[i]
                flag = " *" if d > 0.01 else ""
                print(f"{coin:<8s} {base_auc[i]:>10.4f} {enh_auc[i]:>10.4f} {d:>+7.4f}{flag}")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
