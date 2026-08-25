#!/usr/bin/env python
"""Max PnL portfolio: combine best configs from each TF, DD < 10%.

Pick the highest-PnL config per TF (DD<10%), combine in portfolio.
Test with different DD budgets: 5%, 7.5%, 10%.
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_regime_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.pipeline import apply_vol_overlay
from signal_engine.config import *

CAPITAL = 100_000.0
BYBIT_EXTRA = 3.5
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "ADA": "ADAUSDT", "AVAX": "AVAXUSDT", "DOT": "DOTUSDT",
    "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "UNI": "UNIUSDT", "AAVE": "AAVEUSDT",
    "LTC": "LTCUSDT", "BCH": "BCHUSDT", "HBAR": "HBARUSDT", "TRX": "TRXUSDT",
    "SUI": "SUIUSDT", "APT": "APTUSDT", "ENJ": "ENJUSDT", "SXP": "SXPUSDT",
    "ZEC": "ZECUSDT", "XLM": "XLMUSDT", "FET": "FETUSDT", "WLD": "WLDUSDT",
    "ANKR": "ANKRUSDT", "TAO": "TAOUSDT", "ENA": "ENAUSDT",
    "FTM": "FTMUSDT", "FIL": "FILUSDT", "BNX": "BNXUSDT",
}
OHLCV = ["ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
         "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]
VOL_INV = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
    low_pct=0.20, high_pct=0.80, low_mult=1.25, mid_mult=1.00, high_mult=0.50)

# Best PnL configs per TF (DD < 10%), from sweep results
STRATEGIES = {
    # name: (tf, fwd_bars, C, conf, sizing)
    "1h_maxpnl":  ("1h", 6, 0.10, 0.08, 0.06),   # $107K, SR 2.29, DD 9.1%
    "1h_safe":    ("1h", 4, 0.10, 0.08, 0.06),   # $76K, SR 2.38, DD 4.7%
    "4h_maxpnl":  ("4h", 6, 0.20, 0.06, 0.06),   # $94K, SR 2.44, DD 7.6%
    "4h_safe":    ("4h", 6, 0.20, 0.08, 0.06),   # $74K, SR 2.52, DD 6.3%
    "15m_prod":   ("15min", 4, 0.10, 0.08, 0.04), # current production
}


def _gen(args):
    name, sym, tf, fwd, C = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=C, penalty="l1", solver="saga", max_iter=1000, seed=42,
                          fwd_bars=fwd, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=OHLCV, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe=tf, regime_timeframe="4h"),
        assets=ASSETS)
    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
    except FileNotFoundError:
        return None
    ohlcv_tf = resample_ohlcv(ohlcv_1m, tf)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    feat = compute_features(ohlcv_tf, config.features)
    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0: return None
    sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    return sigs if len(sigs) > 0 else None


def gen_strategy(strat_name, tf, fwd, C, conf):
    print(f"  Generating {strat_name} (tf={tf}, fwd={fwd}, C={C})...")
    t0 = time.time()
    tasks = [(n, s, tf, fwd, C) for n, s in ASSETS.items()]
    all_sigs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed({pool.submit(_gen, t): t for t in tasks}):
            r = f.result()
            if r is not None and len(r) > 0:
                all_sigs.append(r)
    if not all_sigs:
        return pd.DataFrame()
    merged = pd.concat(all_sigs, ignore_index=True)
    merged["strategy"] = strat_name
    merged["hold_bars"] = fwd
    # Apply per-strategy confidence filter
    if "confidence" in merged.columns:
        merged = merged[merged["confidence"] >= conf].reset_index(drop=True)
    print(f"    {len(merged):,} signals ({time.time()-t0:.0f}s)")
    return merged


def run_portfolio(signals, label, sizing):
    if len(signals) == 0: return None
    merged = signals.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in merged.columns: merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Load BTC daily once
    btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
    btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
    if btc_daily.index.tz is not None: btc_daily.index = btc_daily.index.tz_localize(None)
    merged = apply_vol_overlay(merged, btc_daily, VOL_INV)

    port = PortfolioConfig(
        starting_capital=CAPITAL, max_dd=0.15, delever_start=0.075,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
        max_per_strategy=0.50, base_sizing_pct=sizing, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    sim = simulate_portfolio(merged, port)
    if sim is None: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None
    if dpnl.index.tz is not None: dpnl = dpnl.copy(); dpnl.index = dpnl.index.tz_localize(None)

    n_trades = sim["n_trades"]
    avg_sz = float(sim["sizes"][~sim["skipped"]].mean()) if (~sim["skipped"]).any() else 0
    total_extra = n_trades * avg_sz * BYBIT_EXTRA / 10000
    dpnl_adj = dpnl - total_extra / max(len(dpnl), 1)

    dr = dpnl_adj / CAPITAL
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq = (1 + dr).cumprod()
    dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    rng = np.random.RandomState(42)
    bsr = np.empty(3000)
    for b in range(3000):
        s = rng.choice(dr.values, size=len(dr), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

    print(f"  {label:<45} SR={sr:.3f} [{ci[0]:.2f},{ci[1]:.2f}] Sortino={sortino:.3f} "
          f"DD={dd:.1f}% PnL=${dpnl_adj.sum():,.0f} trades={n_trades:,}")

    # Year-by-year
    for year in sorted(dpnl_adj.index.year.unique()):
        y = dpnl_adj[dpnl_adj.index.year == year]
        if len(y) > 20:
            sr_y = float(y.mean() / y.std() * np.sqrt(365.25)) if y.std() > 0 else 0
            cum = CAPITAL + y.cumsum()
            dd_y = float(((cum.cummax() - cum) / cum.cummax()).max() * 100)
            print(f"    {year}: SR={sr_y:.3f} DD={dd_y:.1f}% PnL=${y.sum():,.0f}")

    return {"sr": sr, "ci": ci, "sortino": sortino, "dd": dd,
            "pnl": float(dpnl_adj.sum()), "n_trades": n_trades}


def main():
    T0 = time.time()
    print("=" * 90)
    print("  MAX PnL PORTFOLIO (DD < 10%, 5% daily DD halt)")
    print("=" * 90)

    # Generate all strategies
    strat_signals = {}
    for sname, (tf, fwd, C, conf, sz) in STRATEGIES.items():
        strat_signals[sname] = gen_strategy(sname, tf, fwd, C, conf)

    # Individual strategy results
    print(f"\n{'='*90}")
    print("  INDIVIDUAL STRATEGIES")
    print(f"{'='*90}")
    for sname, (tf, fwd, C, conf, sz) in STRATEGIES.items():
        run_portfolio(strat_signals[sname], sname, sz)

    # Multi-strategy combos
    print(f"\n{'='*90}")
    print("  MULTI-STRATEGY COMBOS (max PnL, DD < 10%)")
    print(f"{'='*90}")

    combos = {
        "1h_safe + 4h_safe": (["1h_safe", "4h_safe"], 0.06),
        "1h_maxpnl + 4h_maxpnl": (["1h_maxpnl", "4h_maxpnl"], 0.06),
        "1h_safe + 4h_maxpnl": (["1h_safe", "4h_maxpnl"], 0.06),
        "15m + 1h_safe + 4h_safe": (["15m_prod", "1h_safe", "4h_safe"], 0.04),
        "15m + 1h_maxpnl + 4h_maxpnl": (["15m_prod", "1h_maxpnl", "4h_maxpnl"], 0.04),
        "1h_safe + 4h_safe (sz=4%)": (["1h_safe", "4h_safe"], 0.04),
        "1h_maxpnl + 4h_maxpnl (sz=4%)": (["1h_maxpnl", "4h_maxpnl"], 0.04),
    }

    for combo_name, (strat_keys, sizing) in combos.items():
        parts = [strat_signals[k] for k in strat_keys if k in strat_signals and len(strat_signals[k]) > 0]
        if not parts: continue
        combined = pd.concat(parts, ignore_index=True)
        run_portfolio(combined, combo_name, sizing)

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
