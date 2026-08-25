"""Quick correlation test: CS Momentum vs LogReg 15m daily PnL."""

import sys, os, time, warnings
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.config import SystemConfig, ModelConfig, FeatureConfig, PortfolioConfig
from signal_engine.config import RegimeFilterConfig, ConfidenceFilterConfig, DataConfig, SignalConfig
from signal_engine.pipeline import run_backtest

warnings.filterwarnings("ignore")

DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]

ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "SOL": "SOLUSDT", "ADA": "ADAUSDT",
    "AVAX": "AVAXUSDT", "DOT": "DOTUSDT", "LINK": "LINKUSDT",
    "NEAR": "NEARUSDT", "UNI": "UNIUSDT", "AAVE": "AAVEUSDT",
    "LTC": "LTCUSDT", "BCH": "BCHUSDT", "FIL": "FILUSDT",
    "FTM": "FTMUSDT", "HBAR": "HBARUSDT", "TRX": "TRXUSDT",
    "SUI": "SUIUSDT", "APT": "APTUSDT", "ENJ": "ENJUSDT",
    "SXP": "SXPUSDT", "ZEC": "ZECUSDT", "XLM": "XLMUSDT",
    "FET": "FETUSDT", "WLD": "WLDUSDT", "BNX": "BNXUSDT",
    "ANKR": "ANKRUSDT", "TAO": "TAOUSDT", "ENA": "ENAUSDT",
}

FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]


def build_cs_momentum_pnl(mom_window=5, reb_freq=5, n_leg=6, cost_bps=10.0):
    """Build daily PnL series for cross-sectional momentum."""
    # Load daily returns
    daily_rets = {}
    for name, symbol in sorted(ASSETS.items()):
        try:
            ohlcv_1m = load_binance_1m(symbol, DATA_DIRS)
            daily = resample_ohlcv(ohlcv_1m, "1D")
            daily_rets[name] = daily["close"].pct_change()
        except:
            continue

    dr = pd.DataFrame(daily_rets)
    momentum = dr.rolling(mom_window).sum()

    pnl_list = []
    dates = []
    last_reb = -reb_freq
    longs = []
    shorts = []

    for i in range(mom_window, len(dr)):
        if i - last_reb >= reb_freq:
            mom_i = momentum.iloc[i].dropna()
            if len(mom_i) < 2 * n_leg:
                continue
            ranked = mom_i.sort_values(ascending=False)
            longs = ranked.index[:n_leg].tolist()
            shorts = ranked.index[-n_leg:].tolist()
            last_reb = i
            tc = cost_bps / 10000.0
        else:
            tc = 0.0

        date = dr.index[i]
        if longs and shorts:
            l_ret = dr.loc[date, longs].mean()
            s_ret = dr.loc[date, shorts].mean()
            ls = (l_ret - s_ret) / 2.0
            pnl_list.append(float(ls) - tc)
            dates.append(date)

    return pd.Series(pnl_list, index=dates)


def main():
    t0 = time.time()
    print("=" * 80)
    print("  CS MOMENTUM vs LOGREG — CORRELATION TEST")
    print("=" * 80)

    # 1. Build CS momentum daily PnL
    print("\nBuilding CS momentum daily PnL (mom=5d, reb=5d, n=6, c=10bps)...")
    cs_pnl = build_cs_momentum_pnl(mom_window=5, reb_freq=5, n_leg=6, cost_bps=10.0)
    print(f"  CS momentum: {len(cs_pnl)} days")

    # 2. Run LogReg 15m pipeline
    print("\nRunning LogReg 15m pipeline...")
    config = SystemConfig(
        model=ModelConfig(C=0.1, penalty="l1", solver="saga",
                          max_iter=1000, seed=42, fwd_bars=4,
                          signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=FEATURES, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        portfolio=PortfolioConfig(starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
                                  slippage=0.0001, max_exposure=0.80, max_per_asset=0.06,
                                  base_sizing_pct=0.02, max_signal_mult=2.0,
                                  sizing_mode="probability_edge",
                                  use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe="15min", regime_timeframe="4h"),
        assets=ASSETS,
    )

    result = run_backtest(config, max_workers=11)
    lr_daily_pnl = result.get("daily_pnl")

    if lr_daily_pnl is None:
        print("  LogReg produced no daily PnL!")
        return

    print(f"  LogReg: {len(lr_daily_pnl)} days")

    # 3. Normalize CS momentum PnL to same scale as LogReg
    # CS momentum is in fraction returns; LogReg is in dollar PnL
    # For correlation, scale doesn't matter — just align dates
    if cs_pnl.index.tz is not None:
        cs_pnl.index = cs_pnl.index.tz_localize(None)
    if lr_daily_pnl.index.tz is not None:
        lr_daily_pnl.index = lr_daily_pnl.index.tz_localize(None)

    combined = pd.DataFrame({
        "LogReg_15m": lr_daily_pnl,
        "CS_Momentum": cs_pnl,
    }).fillna(0)

    # Only days where at least one traded
    mask = (combined["LogReg_15m"] != 0) | (combined["CS_Momentum"] != 0)
    combined = combined[mask]

    print(f"\n  Overlapping days: {len(combined)}")

    if len(combined) < 30:
        print("  Too few overlapping days for correlation")
        return

    corr = combined["LogReg_15m"].corr(combined["CS_Momentum"])
    print(f"\n{'='*80}")
    print(f"  CORRELATION: {corr:.4f}")
    if corr > 0.5:
        print(f"  FAIL: correlation > 0.5 — too correlated for diversification")
    elif corr > 0.3:
        print(f"  MODERATE: correlation 0.3-0.5 — some diversification")
    else:
        print(f"  PASS: correlation < 0.3 — good diversification")
    print(f"{'='*80}")

    # Also compute 4H LogReg correlation if we had it...
    # For now, show combined portfolio stats
    # Normalize both to zero-mean, unit-std for fair combination
    lr_norm = combined["LogReg_15m"] / combined["LogReg_15m"].std()
    cs_norm = combined["CS_Momentum"] / combined["CS_Momentum"].std()
    combined_norm = (lr_norm + cs_norm) / 2.0
    combined_nonzero = combined_norm[combined_norm != 0]

    if len(combined_nonzero) > 30:
        sr_lr = lr_norm.mean() / lr_norm.std() * np.sqrt(365.25) if lr_norm.std() > 0 else 0
        sr_cs = cs_norm.mean() / cs_norm.std() * np.sqrt(365.25) if cs_norm.std() > 0 else 0
        sr_comb = combined_nonzero.mean() / combined_nonzero.std() * np.sqrt(365.25) if combined_nonzero.std() > 0 else 0

        print(f"\n  LogReg 15m SR (normalized): {sr_lr:.2f}")
        print(f"  CS Momentum SR (normalized): {sr_cs:.2f}")
        print(f"  Combined 50/50 SR:           {sr_comb:.2f}")
        print(f"  Diversification benefit:     {sr_comb - max(sr_lr, sr_cs):+.2f}")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
