"""Fixed Pairs + ML Spread Trading — Full Pipeline

Phase 2-4: Feature engineering, triple-barrier labeling, XGBoost walk-forward,
backtesting, and combination with LogReg T1.

12 pairs selected from exhaustive pre-2025 scan (no asset overlap).
"""

import sys, os, time, warnings, math, pickle
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/cache_15m")
SCACHE = Path("data/optuna/signal_cache")
OUT_DIR = Path("results/coint_ml")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAPITAL = 10000.0
COST_BPS = 5.0
BARS_PER_DAY = 96

# Selected pairs from exhaustive scan (no asset overlap, all pre-2025 profitable)
SELECTED_PAIRS = [
    ("ENA", "LTC"),    # 100% pers, score 17.09
    ("APT", "SXP"),    # 71%, 15.53
    ("ENJ", "FIL"),    # 71%, 11.37
    ("FET", "FTM"),    # 43%, 11.14, highest PnL
    ("ALPHA", "BCH"),  # 100%, 9.53
    ("WIF", "WLD"),    # 67%, 9.53
    ("DOGE", "LINA"),  # 67%, 8.50
    ("ADA", "BNB"),    # 50%, 7.46
    ("XRP", "ZEC"),    # 80%, 5.85
    ("DOT", "ETC"),    # 67%, 4.76
    ("LEVER", "NEAR"), # 67%, 6.60
    ("GALA", "LIT"),   # 67%, 6.85
]

# Walk-forward config
WF_LOOKBACK = 96 * 180  # 180 days for OLS estimation
WF_RETRAIN_BARS = 96 * 30  # retrain monthly
WF_MIN_TRAIN = 96 * 365  # 1 year minimum training data
EMBARGO_BARS = 96 * 2  # 2 days embargo
FEAT_LOOKBACK = 2000  # bars for rolling features

# Triple-barrier config
TB_ENTRY_Z = 1.5  # candidate entry threshold
TB_PT_MULT = 1.5  # profit-take at 1.5x spread vol
TB_SL_MULT = 1.5  # stop-loss at 1.5x spread vol
TB_MAX_BARS = 96  # 1 day max hold

# Trading config
ENTRY_THRESH = 0.55  # ML probability threshold
BASE_SIZE = 0.04  # 4% per pair


# ============== SPREAD COMPUTATION ==============

def ols_fit(y, x):
    xm, ym = x.mean(), y.mean(); xd = x - xm; d = np.dot(xd, xd)
    if d <= 0: return np.nan, np.nan
    b = np.dot(xd, y - ym) / d; return float(b), float(ym - b * xm)


def compute_spread_series(y_close, x_close, lookback=WF_LOOKBACK):
    """Compute rolling spread = log(Y) - beta*log(X) - intercept with rolling OLS."""
    n = len(y_close)
    spread = np.full(n, np.nan)
    beta_arr = np.full(n, np.nan)
    mu_arr = np.full(n, np.nan)
    sigma_arr = np.full(n, np.nan)
    z_arr = np.full(n, np.nan)

    log_y = np.log(y_close); log_x = np.log(x_close)

    for i in range(lookback, n):
        sl = slice(i - lookback, i)
        beta, intercept = ols_fit(log_y[sl], log_x[sl])
        if not np.isfinite(beta) or beta <= 0: continue
        resid_window = log_y[sl] - beta * log_x[sl] - intercept
        mu = float(np.mean(resid_window))
        sigma = float(np.std(resid_window))
        if sigma <= 1e-12: continue

        s = log_y[i] - beta * log_x[i] - intercept
        spread[i] = s
        beta_arr[i] = beta
        mu_arr[i] = mu
        sigma_arr[i] = sigma
        z_arr[i] = (s - mu) / sigma

    return spread, beta_arr, mu_arr, sigma_arr, z_arr


# ============== FEATURE ENGINEERING ==============

def half_life_rolling(resid, window=2000):
    """Rolling AR(1) half-life in days."""
    n = len(resid)
    hl = np.full(n, np.nan)
    for i in range(window, n):
        r = resid[i-window:i]
        if np.isnan(r).sum() > window * 0.3: continue
        valid = r[np.isfinite(r)]
        if len(valid) < 100: continue
        lag = valid[:-1]; delta = np.diff(valid)
        d = np.dot(lag - lag.mean(), lag - lag.mean())
        if d <= 0: continue
        phi = np.dot(lag - lag.mean(), delta - delta.mean()) / d
        if phi < 0:
            hl[i] = -math.log(2) / phi / BARS_PER_DAY
        else:
            hl[i] = 50.0  # cap at 50 days
    return np.clip(hl, 0.1, 50)


def hurst_rolling(resid, window=2000, max_lag=64):
    """Rolling Hurst exponent."""
    n = len(resid)
    h = np.full(n, np.nan)
    for i in range(window, n):
        r = resid[i-window:i]
        valid = r[np.isfinite(r)]
        if len(valid) < max_lag * 3: continue
        lags = np.arange(2, min(max_lag + 1, len(valid) // 3))
        tau = []
        for lag in lags:
            s = np.std(valid[lag:] - valid[:-lag])
            tau.append(s if s > 0 and np.isfinite(s) else np.nan)
        tau_arr = np.array(tau); keep = np.isfinite(tau_arr) & (tau_arr > 0)
        if keep.sum() < 5: continue
        h[i] = float(np.clip(np.polyfit(np.log(lags[keep]), np.log(tau_arr[keep]), 1)[0], 0, 1))
    return h


def rsi(series, period=14):
    """RSI on any series."""
    delta = np.diff(series, prepend=series[0])
    gain = np.where(delta > 0, delta, 0).astype(float)
    loss = np.where(delta < 0, -delta, 0).astype(float)
    # EMA
    avg_gain = np.full(len(series), np.nan)
    avg_loss = np.full(len(series), np.nan)
    if len(series) < period + 1: return np.full(len(series), 50.0)
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    for i in range(period+1, len(series)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    result = 100 - 100 / (1 + rs)
    return np.where(np.isfinite(result), result, 50.0)


def bollinger_pctb(series, window=20):
    """Bollinger %B."""
    n = len(series)
    pctb = np.full(n, np.nan)
    for i in range(window, n):
        sl = series[i-window:i]
        valid = sl[np.isfinite(sl)]
        if len(valid) < 10: continue
        mu = np.mean(valid); sigma = np.std(valid)
        if sigma <= 0: continue
        upper = mu + 2 * sigma; lower = mu - 2 * sigma
        if upper == lower: continue
        pctb[i] = (series[i] - lower) / (upper - lower)
    return pctb


def compute_features(spread, z_arr, sigma_arr, y_close, x_close,
                     y_vol, x_vol, btc_close, timestamps):
    """Compute all features for one pair. Returns DataFrame."""
    n = len(spread)

    # Z-score velocity and acceleration
    z_vel_4 = np.full(n, np.nan); z_vel_16 = np.full(n, np.nan)
    z_accel = np.full(n, np.nan)
    for i in range(16, n):
        if np.isfinite(z_arr[i]) and np.isfinite(z_arr[i-4]):
            z_vel_4[i] = z_arr[i] - z_arr[i-4]
        if np.isfinite(z_arr[i]) and np.isfinite(z_arr[i-16]):
            z_vel_16[i] = z_arr[i] - z_arr[i-16]
    for i in range(20, n):
        if np.isfinite(z_vel_4[i]) and np.isfinite(z_vel_4[i-4]):
            z_accel[i] = z_vel_4[i] - z_vel_4[i-4]

    # Spread RSI and Bollinger
    spread_clean = np.where(np.isfinite(spread), spread, 0)
    spread_rsi = rsi(spread_clean, 14)
    spread_bb = bollinger_pctb(spread_clean, 20)

    # Rolling half-life and Hurst (expensive — subsample)
    hl = half_life_rolling(spread, FEAT_LOOKBACK)
    h = hurst_rolling(spread, FEAT_LOOKBACK)

    # Rolling return correlation (100-bar)
    y_ret = np.diff(y_close, prepend=y_close[0]) / np.where(y_close > 0, y_close, 1)
    x_ret = np.diff(x_close, prepend=x_close[0]) / np.where(x_close > 0, x_close, 1)
    corr_100 = np.full(n, np.nan)
    for i in range(100, n):
        yr = y_ret[i-100:i]; xr = x_ret[i-100:i]
        v = np.isfinite(yr) & np.isfinite(xr)
        if v.sum() > 50:
            corr_100[i] = float(np.corrcoef(yr[v], xr[v])[0, 1])

    # Volume imbalance
    vol_imb = np.full(n, np.nan)
    for i in range(50, n):
        yv = np.mean(y_vol[i-50:i]); xv = np.mean(x_vol[i-50:i])
        s = yv + xv
        if s > 0: vol_imb[i] = (yv - xv) / s

    # BTC 24h return
    btc_ret_24h = np.full(n, np.nan)
    for i in range(BARS_PER_DAY, n):
        if btc_close[i] > 0 and btc_close[i-BARS_PER_DAY] > 0:
            btc_ret_24h[i] = btc_close[i] / btc_close[i-BARS_PER_DAY] - 1

    # Y volatility z-score
    y_vol_zs = np.full(n, np.nan)
    for i in range(50, n):
        yr = np.abs(y_ret[i-50:i])
        valid = yr[np.isfinite(yr)]
        if len(valid) > 10:
            mu = np.mean(valid); sig = np.std(valid)
            if sig > 0: y_vol_zs[i] = (np.abs(y_ret[i]) - mu) / sig if np.isfinite(y_ret[i]) else 0

    # Spread vol ratio (current / long-run)
    spread_vol_ratio = np.full(n, np.nan)
    for i in range(FEAT_LOOKBACK, n):
        short = np.nanstd(spread[i-100:i])
        long = np.nanstd(spread[i-FEAT_LOOKBACK:i])
        if long > 0: spread_vol_ratio[i] = short / long

    # Hour features
    hours = np.array([t.hour + t.minute / 60 for t in timestamps])
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)

    return pd.DataFrame({
        "z_score": z_arr,
        "z_velocity_4": z_vel_4,
        "z_velocity_16": z_vel_16,
        "z_acceleration": z_accel,
        "spread_rsi_14": spread_rsi,
        "spread_bb_pct": spread_bb,
        "rolling_half_life": hl,
        "rolling_hurst": h,
        "return_corr_100": corr_100,
        "volume_imbalance": vol_imb,
        "btc_ret_24h": btc_ret_24h,
        "y_vol_zscore": y_vol_zs,
        "spread_vol_ratio": spread_vol_ratio,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
    }, index=timestamps)


# ============== TRIPLE-BARRIER LABELING ==============

def label_triple_barrier(spread, z_arr, sigma_arr, entry_z=TB_ENTRY_Z,
                         pt_mult=TB_PT_MULT, sl_mult=TB_SL_MULT,
                         max_bars=TB_MAX_BARS):
    """Label candidate entries with triple-barrier on spread."""
    n = len(spread)
    labels = []  # (bar_idx, label, side)

    for i in range(n - max_bars):
        z = z_arr[i]
        if not np.isfinite(z) or abs(z) < entry_z: continue
        sigma = sigma_arr[i]
        if not np.isfinite(sigma) or sigma <= 0: continue

        side = -1.0 if z > 0 else 1.0  # mean-reversion: short if z>0, long if z<0
        entry_spread = spread[i]
        if not np.isfinite(entry_spread): continue
        target = sigma  # use spread sigma as barrier width

        label = 0  # default: time barrier
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if not np.isfinite(spread[j]): continue
            move = spread[j] - entry_spread

            # Profit-take: spread moves in our direction
            if side * move > pt_mult * target:
                label = 1; break
            # Stop-loss: spread moves against us
            if side * move < -sl_mult * target:
                label = 0; break
        else:
            # Time barrier: label by sign of final move
            if j < n and np.isfinite(spread[j]):
                label = 1 if side * (spread[j] - entry_spread) > 0 else 0

        labels.append((i, label, side))

    return labels


# ============== WALK-FORWARD TRAINING ==============

FEATURE_COLS = [
    "z_score", "z_velocity_4", "z_velocity_16", "z_acceleration",
    "spread_rsi_14", "spread_bb_pct", "rolling_half_life", "rolling_hurst",
    "return_corr_100", "volume_imbalance", "btc_ret_24h", "y_vol_zscore",
    "spread_vol_ratio", "hour_sin", "hour_cos",
]


def walk_forward_train_predict(all_features, all_labels, all_pair_ids, timestamps,
                               oos_start, retrain_every=WF_RETRAIN_BARS):
    """Walk-forward XGBoost training on pooled data. Returns predictions for OOS bars."""
    n = len(all_features)
    oos_mask = timestamps >= oos_start
    predictions = np.full(n, np.nan)
    last_train_idx = 0

    oos_indices = np.where(oos_mask)[0]
    if len(oos_indices) == 0: return predictions

    # Retrain at monthly boundaries
    retrain_points = list(range(oos_indices[0], oos_indices[-1] + 1, retrain_every))
    if retrain_points[-1] < oos_indices[-1]:
        retrain_points.append(oos_indices[-1] + 1)

    model = None
    for ri in range(len(retrain_points) - 1):
        train_end = retrain_points[ri] - EMBARGO_BARS
        pred_start = retrain_points[ri]
        pred_end = retrain_points[ri + 1] if ri + 1 < len(retrain_points) else n

        if train_end < WF_MIN_TRAIN: continue

        # Train on all data up to train_end
        train_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            if i < train_end and np.isfinite(all_labels[i]):
                train_mask[i] = True

        if train_mask.sum() < 500: continue

        X_train = all_features[train_mask]
        y_train = all_labels[train_mask]
        X_train = np.nan_to_num(X_train, nan=0, posinf=10, neginf=-10)

        model = XGBClassifier(
            max_depth=4, n_estimators=150, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", verbosity=0, random_state=42,
            enable_categorical=True,
        )
        model.fit(X_train, y_train)

        # Predict for this window
        for i in range(pred_start, min(pred_end, n)):
            if np.isfinite(all_labels[i]):  # only predict at candidate entries
                x = all_features[i:i+1]
                x = np.nan_to_num(x, nan=0, posinf=10, neginf=-10)
                predictions[i] = model.predict_proba(x)[0, 1]

    return predictions, model


# ============== BACKTESTING ==============

def simulate_ml_coint(pair_data_list, config, capital=CAPITAL):
    """Simulate ML-filtered coint trading across all pairs.

    pair_data_list: list of dicts with keys:
        y_close, x_close, spread, z_arr, sigma_arr, beta_arr,
        predictions, timestamps, y_name, x_name
    config: dict with entry_z, ml_thresh, base_size, use_ml, use_afml
    """
    entry_z = config.get("entry_z", 1.5)
    ml_thresh = config.get("ml_thresh", 0.55)
    base_size = config.get("base_size", BASE_SIZE)
    use_ml = config.get("use_ml", True)
    max_hold = config.get("max_hold", TB_MAX_BARS)

    daily_pnl = {}; n_trades = 0; n_filtered = 0

    for pd_item in pair_data_list:
        y_close = pd_item["y_close"]; x_close = pd_item["x_close"]
        z_arr = pd_item["z_arr"]; sigma_arr = pd_item["sigma_arr"]
        beta_arr = pd_item["beta_arr"]; preds = pd_item["predictions"]
        timestamps = pd_item["timestamps"]

        y_ret = np.diff(y_close, prepend=y_close[0]) / np.where(y_close > 0, y_close, 1)
        x_ret = np.diff(x_close, prepend=x_close[0]) / np.where(x_close > 0, x_close, 1)

        pos = 0; entry_bar = 0; pos_beta = 1.0; pos_size = base_size
        n = len(y_close)

        for i in range(1, n):
            if not np.isfinite(z_arr[i]): continue
            z = z_arr[i]; beta = beta_arr[i]
            if not np.isfinite(beta): continue

            notional = capital * pos_size
            cost = COST_BPS / 10000 * (1 + abs(beta)) * notional
            bar_pnl = 0.0

            # PnL for open position
            if pos != 0:
                bar_pnl = pos * (y_ret[i] - pos_beta * x_ret[i]) * capital * pos_size
                bh = i - entry_bar
                should_exit = False
                if pos == 1 and z >= 0: should_exit = True
                elif pos == -1 and z <= 0: should_exit = True
                if bh >= max_hold: should_exit = True
                if abs(z) > 5 and bh > 48: should_exit = True
                if should_exit:
                    bar_pnl -= cost; pos = 0

            # Entry
            if pos == 0 and abs(z) >= entry_z:
                enter = True
                if use_ml:
                    if not np.isfinite(preds[i]) or preds[i] < ml_thresh:
                        enter = False
                        n_filtered += 1

                if enter:
                    # AFML-style sizing: scale by ML confidence
                    if use_ml and np.isfinite(preds[i]):
                        conf = max(0.3, min(preds[i], 0.95))
                        pos_size = base_size * (0.5 + conf)  # 0.65-1.45x base
                    else:
                        pos_size = base_size

                    notional = capital * pos_size
                    cost = COST_BPS / 10000 * (1 + abs(beta)) * notional
                    pos_beta = beta

                    if z <= -entry_z:
                        pos = 1; entry_bar = i; bar_pnl -= cost; n_trades += 1
                    elif z >= entry_z:
                        pos = -1; entry_bar = i; bar_pnl -= cost; n_trades += 1

            day = timestamps[i].normalize()
            if hasattr(day, "tz") and day.tz is not None:
                day = day.tz_localize(None)
            daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl

    return pd.Series(daily_pnl).sort_index(), n_trades, n_filtered


def load_logreg_t1_dpnl():
    """Load LogReg T1 daily PnL from signal cache."""
    from signal_engine.config import PortfolioConfig
    from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
    import test_original_pipeline as orig

    W = [("W1", 245), ("W2", 244), ("W3", 122)]
    dfs = {}
    for wl, cd in W:
        p = SCACHE / f"C0.02000_lb2_{wl}.pkl"
        if not p.exists(): continue
        with open(p, "rb") as f: dfs[wl] = pickle.load(f)
    if not dfs: return pd.Series(dtype=float)

    all_dpnl = {}
    for wl, cd in W:
        raw = dfs.get(wl)
        if raw is None or raw.empty: continue
        fi = raw[raw["confidence"] > 0.10].copy()
        if len(fi) < 10: continue
        fi["fwd_return"] = fi["ret_6"].values
        combined = orig.apply_nco_wf(fi)
        pcfg = PortfolioConfig(starting_capital=CAPITAL, max_dd=0.10, delever_start=0.05,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=0.30, max_signal_mult=3.0)
        sim = simulate_portfolio(combined, pcfg)
        if sim is None: continue
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None: continue
        for dt, val in dpnl.items():
            dt2 = dt.tz_localize(None) if hasattr(dt, "tz") and dt.tz is not None else dt
            all_dpnl[dt2] = all_dpnl.get(dt2, 0) + val
    return pd.Series(all_dpnl).sort_index()


def metrics(dpnl, label, capital=CAPITAL, cal_days=None):
    if len(dpnl) < 5: return None
    if cal_days is None: cal_days = (dpnl.index[-1] - dpnl.index[0]).days + 1
    dr = dpnl / capital
    sharpe = dr.mean() / dr.std() * np.sqrt(365.25) if dr.std() > 0 else 0
    cum = dpnl.cumsum(); peak = (cum + capital).cummax()
    dd = abs(((cum + capital - peak) / peak).min()); wd = abs(dr.min())
    pnl_cal = cum.iloc[-1] / cal_days
    monthly = dpnl.resample("ME").sum(); pm = (monthly > 0).sum()
    wr = (dpnl > 0).sum() / max((dpnl != 0).sum(), 1)
    print(f"  {label:<25s} SR={sharpe:>5.2f}  ${pnl_cal:>7.2f}/cal  "
          f"DD={dd*100:>5.1f}%  WD={wd*100:>5.2f}%  +Mo={pm}/{len(monthly)} WR={wr*100:.0f}%")
    return {"sharpe": sharpe, "pnl_cal": pnl_cal, "dd": dd, "wd": wd}


def main():
    t0 = time.time()
    print("=" * 110)
    print("FIXED PAIRS + ML SPREAD TRADING")
    print(f"12 pairs, XGBoost, triple-barrier labels, walk-forward")
    print("=" * 110)

    # Load data
    print("\nLoading data...", end=" ", flush=True)
    close_dict = {}; vol_dict = {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        name = f.stem.replace("USDT", "")
        try:
            df = pd.read_parquet(f)
            if len(df) >= 2000:
                close_dict[name] = df["close"]
                vol_dict[name] = df["volume"]
        except: pass

    btc_close = close_dict.get("BTC", pd.Series(dtype=float))
    print(f"{len(close_dict)} assets")

    # OOS boundary
    oos_start = pd.Timestamp("2025-01-01", tz="UTC")

    # Process each pair
    print("\nComputing spreads and features...", flush=True)
    all_feat_rows = []
    all_labels = []
    all_pair_ids = []
    all_timestamps = []
    pair_data_list = []

    for pi, (y_name, x_name) in enumerate(SELECTED_PAIRS):
        pt0 = time.time()
        if y_name not in close_dict or x_name not in close_dict:
            print(f"  {y_name}/{x_name}: MISSING DATA"); continue

        y_s = close_dict[y_name]; x_s = close_dict[x_name]
        common = y_s.index.intersection(x_s.index)
        if len(common) < WF_LOOKBACK + 5000:
            print(f"  {y_name}/{x_name}: too short ({len(common)} bars)"); continue

        y_close = y_s.reindex(common).values.astype(float)
        x_close = x_s.reindex(common).values.astype(float)
        timestamps = y_s.reindex(common).index

        # BTC close aligned
        btc_aligned = btc_close.reindex(common).values.astype(float) if len(btc_close) > 0 else np.ones(len(common))
        btc_aligned = np.where(np.isfinite(btc_aligned), btc_aligned, 0)

        # Volume
        y_vol = vol_dict.get(y_name, pd.Series(0, index=common)).reindex(common).fillna(0).values.astype(float)
        x_vol = vol_dict.get(x_name, pd.Series(0, index=common)).reindex(common).fillna(0).values.astype(float)

        # Spread
        spread, beta_arr, mu_arr, sigma_arr, z_arr = compute_spread_series(y_close, x_close)

        # Features
        feat_df = compute_features(spread, z_arr, sigma_arr, y_close, x_close,
                                   y_vol, x_vol, btc_aligned, timestamps)

        # Triple-barrier labels
        labels_raw = label_triple_barrier(spread, z_arr, sigma_arr)
        label_arr = np.full(len(spread), np.nan)
        side_arr = np.full(len(spread), np.nan)
        for idx, lab, side in labels_raw:
            label_arr[idx] = lab
            side_arr[idx] = side

        n_labels = np.isfinite(label_arr).sum()
        pos_rate = np.nanmean(label_arr[np.isfinite(label_arr)]) if n_labels > 0 else 0

        # Store for training
        feat_vals = feat_df[FEATURE_COLS].values
        for i in range(len(spread)):
            if np.isfinite(label_arr[i]):
                all_feat_rows.append(feat_vals[i])
                all_labels.append(label_arr[i])
                all_pair_ids.append(pi)
                all_timestamps.append(timestamps[i])

        # Store for backtesting
        pair_data_list.append({
            "y_name": y_name, "x_name": x_name,
            "y_close": y_close, "x_close": x_close,
            "spread": spread, "z_arr": z_arr,
            "sigma_arr": sigma_arr, "beta_arr": beta_arr,
            "timestamps": timestamps,
            "features": feat_vals, "labels": label_arr,
            "predictions": np.full(len(spread), np.nan),  # filled later
        })

        print(f"  {y_name}/{x_name}: {len(common)} bars, {n_labels} labels "
              f"({pos_rate*100:.0f}% pos) [{time.time()-pt0:.0f}s]", flush=True)

    # Walk-forward training
    print(f"\nTraining data: {len(all_feat_rows)} labeled bars across {len(pair_data_list)} pairs")
    X_all = np.array(all_feat_rows)
    y_all = np.array(all_labels)
    pair_ids_all = np.array(all_pair_ids)
    ts_all = pd.DatetimeIndex(all_timestamps)

    print(f"Label distribution: {y_all.mean()*100:.1f}% positive")
    print(f"\nWalk-forward training...", flush=True)

    predictions, model = walk_forward_train_predict(X_all, y_all, pair_ids_all, ts_all, oos_start)

    # Map predictions back to pair_data_list
    row_idx = 0
    for pd_item in pair_data_list:
        n = len(pd_item["labels"])
        for i in range(n):
            if np.isfinite(pd_item["labels"][i]):
                pd_item["predictions"][i] = predictions[row_idx]
                row_idx += 1

    # Feature importance
    if model is not None:
        print("\nFeature importance:")
        imp = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
        for f, s in imp:
            print(f"  {f:<25s} {s:.4f}")

    # OOS accuracy
    oos_mask = ts_all >= oos_start
    if oos_mask.sum() > 0:
        oos_preds = predictions[oos_mask.values if hasattr(oos_mask, 'values') else oos_mask]
        oos_labels = y_all[oos_mask.values if hasattr(oos_mask, 'values') else oos_mask]
        valid = np.isfinite(oos_preds)
        if valid.sum() > 0:
            from sklearn.metrics import roc_auc_score, accuracy_score
            auc = roc_auc_score(oos_labels[valid], oos_preds[valid])
            acc = accuracy_score(oos_labels[valid], (oos_preds[valid] > 0.5).astype(int))
            print(f"\nOOS metrics: AUC={auc:.3f}, Acc={acc*100:.1f}%, N={valid.sum()}")

    # Filter pair_data to OOS only for backtesting
    oos_pair_data = []
    for pd_item in pair_data_list:
        ts = pd_item["timestamps"]
        oos_idx = ts >= oos_start
        if oos_idx.sum() < 100: continue
        oos_pair_data.append({
            "y_name": pd_item["y_name"], "x_name": pd_item["x_name"],
            "y_close": pd_item["y_close"][oos_idx],
            "x_close": pd_item["x_close"][oos_idx],
            "spread": pd_item["spread"][oos_idx],
            "z_arr": pd_item["z_arr"][oos_idx],
            "sigma_arr": pd_item["sigma_arr"][oos_idx],
            "beta_arr": pd_item["beta_arr"][oos_idx],
            "timestamps": pd_item["timestamps"][oos_idx],
            "predictions": pd_item["predictions"][oos_idx],
        })

    # Backtest configs
    print(f"\n{'='*110}")
    print("BACKTEST RESULTS (OOS: 2025-01 to end)")
    print(f"{'='*110}")

    configs = [
        ("Baseline z=2.0 no-ML", {"entry_z": 2.0, "ml_thresh": 0, "base_size": 0.03, "use_ml": False, "max_hold": 96*30}),
        ("Baseline z=1.5 no-ML", {"entry_z": 1.5, "ml_thresh": 0, "base_size": 0.03, "use_ml": False, "max_hold": 96*30}),
        ("ML P>0.50 z=1.5", {"entry_z": 1.5, "ml_thresh": 0.50, "base_size": 0.04, "use_ml": True, "max_hold": 96}),
        ("ML P>0.55 z=1.5", {"entry_z": 1.5, "ml_thresh": 0.55, "base_size": 0.04, "use_ml": True, "max_hold": 96}),
        ("ML P>0.60 z=1.5", {"entry_z": 1.5, "ml_thresh": 0.60, "base_size": 0.04, "use_ml": True, "max_hold": 96}),
        ("ML P>0.55 z=2.0", {"entry_z": 2.0, "ml_thresh": 0.55, "base_size": 0.04, "use_ml": True, "max_hold": 96}),
        ("ML P>0.60 z=2.0", {"entry_z": 2.0, "ml_thresh": 0.60, "base_size": 0.04, "use_ml": True, "max_hold": 96}),
        ("ML P>0.55 z=1.5 5%", {"entry_z": 1.5, "ml_thresh": 0.55, "base_size": 0.05, "use_ml": True, "max_hold": 96}),
    ]

    # Load LogReg T1
    print("\nLoading LogReg T1...", end=" ", flush=True)
    lr_dpnl = load_logreg_t1_dpnl()
    # Filter to OOS
    lr_dpnl_oos = lr_dpnl[lr_dpnl.index >= oos_start.tz_localize(None)]
    lr_cal = (lr_dpnl_oos.index[-1] - lr_dpnl_oos.index[0]).days + 1 if len(lr_dpnl_oos) > 0 else 1
    print(f"{len(lr_dpnl_oos)} days OOS")
    metrics(lr_dpnl_oos, "LogReg T1 standalone", cal_days=lr_cal)

    print(f"\n{'Config':<30s} {'$/cal':>7s} {'SR':>6s} {'DD':>5s} {'Tr':>5s} {'Filt':>5s} "
          f"{'Cmb$/c':>7s} {'CmbSR':>6s} {'CmbDD':>6s}")
    print("-" * 95)

    for name, cfg in configs:
        dpnl, n_tr, n_filt = simulate_ml_coint(oos_pair_data, cfg)
        if len(dpnl) < 5:
            print(f"{name:<30s} NO DATA"); continue

        cal = (dpnl.index[-1] - dpnl.index[0]).days + 1
        dr = dpnl / CAPITAL
        sr = dr.mean() / dr.std() * np.sqrt(365.25) if dr.std() > 0 else 0
        cum = dpnl.cumsum(); peak = (cum + CAPITAL).cummax()
        dd = abs(((cum + CAPITAL - peak) / peak).min())
        pnl_cal = cum.iloc[-1] / cal

        # Combined
        all_d = lr_dpnl_oos.index.union(dpnl.index).sort_values()
        combo = lr_dpnl_oos.reindex(all_d).fillna(0) + dpnl.reindex(all_d).fillna(0)
        cb_dr = combo / CAPITAL
        cb_sr = cb_dr.mean() / cb_dr.std() * np.sqrt(365.25) if cb_dr.std() > 0 else 0
        cb_cum = combo.cumsum()
        cb_peak = (cb_cum + CAPITAL).cummax()
        cb_dd = abs(((cb_cum + CAPITAL - cb_peak) / cb_peak).min())
        cb_cal = (combo.index[-1] - combo.index[0]).days + 1
        cb_pnl = cb_cum.iloc[-1] / cb_cal

        ok = cb_dd < 0.10
        better = pnl_cal > 0

        print(f"{name:<30s} ${pnl_cal:>6.2f} {sr:>6.2f} {dd*100:>4.1f}% {n_tr:>5d} {n_filt:>5d} "
              f"${cb_pnl:>6.2f} {cb_sr:>6.2f} {cb_dd*100:>5.1f}% "
              f"{'OK' if ok else '  '} {'$+' if better else '  '}")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
