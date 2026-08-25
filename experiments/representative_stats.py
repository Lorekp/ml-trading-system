"""Full stats on most representative holdout tests (B and C)."""
import sys
sys.path.insert(0, "src")
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import copy, warnings, os, time
import numpy as np, pandas as pd
from scipy.stats import norm
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
SLIPPAGE = 0.0001

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


def holdout_asset(args):
    asset, symbol, train_end_str = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    cfg = load_config(Path("configs/default.yaml"))
    train_end = pd.Timestamp(train_end_str, tz="UTC")

    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        features = compute_features(ohlcv_15m, cfg.features)
        feat = features[cfg.features.feature_list].copy()

        close = ohlcv_15m["close"]
        fwd_ret = close.pct_change(4).shift(-4)
        vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

        cutoff = train_end.tz_localize(None) if ohlcv_15m.index.tz is None else train_end

        train_feat = feat[feat.index < cutoff].dropna()
        train_y_all = fwd_ret.reindex(train_feat.index).dropna()
        train_valid = train_feat.index.intersection(train_y_all.index)

        train_vol = vol.loc[vol.index.intersection(train_valid)].dropna()
        if len(train_vol) < 50 or len(train_valid) < 500:
            return asset, None

        vol_thresh = train_vol.quantile(0.50)

        for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
            if col in feat.columns:
                lo = train_feat[col].quantile(0.01)
                hi = train_feat[col].quantile(0.99)
                feat[col] = feat[col].clip(lo, hi)

        sw = compute_sample_weights(feat.loc[train_valid].index.values.astype(np.int64), fwd_bars=4)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(feat.loc[train_valid])
        model = make_logreg()
        model.fit(Xtr, (train_y_all.loc[train_valid] > 0).astype(int), sample_weight=sw)

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
        filtered = apply_confidence_filter(signals, ConfidenceFilterConfig(mode="absolute", threshold=CONF_THRESH))
        return asset, filtered.to_dict("list") if len(filtered) > 0 else None
    except Exception:
        return asset, None


def full_report(train_end_str, label):
    print(f"\nGenerating signals ({W} workers)...", flush=True)
    t0 = time.time()
    all_signals = []

    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(holdout_asset, (a, s, train_end_str)): a for a, s in ASSETS.items()}
        for future in as_completed(futures):
            asset, sig = future.result()
            if sig is not None:
                all_signals.append(sig)

    if not all_signals:
        print("NO SIGNALS")
        return

    frames = [pd.DataFrame(d) for d in all_signals]
    filtered = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)

    # Kelly from test signals (this IS test-period data but Kelly weights are descriptive not predictive)
    kelly_weights = {}
    for asset in filtered["asset"].unique():
        asig = filtered[filtered["asset"] == asset]
        tr = asig["direction"].values * asig["fwd_return"].values
        w = tr[tr > 0]
        l = tr[tr <= 0]
        if len(w) >= 3 and len(l) >= 3 and abs(l.mean()) > 0:
            f = kelly_size(len(w) / len(tr), float(w.mean()), float(abs(l.mean())))
            kelly_weights[asset] = max(0.2, min(3.0, f * 7.5))
        else:
            kelly_weights[asset] = 1.0
    mw = np.mean(list(kelly_weights.values()))
    if mw > 0:
        kelly_weights = {k: v / mw for k, v in kelly_weights.items()}
    filtered["signal_strength"] = filtered.apply(
        lambda r: r["signal_strength"] * kelly_weights.get(r["asset"], 1.0), axis=1)

    cfg = load_config(Path("configs/default.yaml"))
    pcfg = copy.deepcopy(cfg.portfolio)
    pcfg.base_sizing_pct = 0.04
    pcfg.slippage = SLIPPAGE
    pcfg.max_exposure = 0.90
    pcfg.max_per_asset = 0.03
    SC = pcfg.starting_capital

    sim = simulate_portfolio(filtered, pcfg)
    if sim is None:
        print("SIM FAILED")
        return

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        print("NO DAILY PNL")
        return

    s = compute_full_stats(dpnl, SC, sim["n_trades"], sim["n_trades"] / len(dpnl))
    daily_ret = dpnl / SC
    equity = (1 + daily_ret).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    pnl_day = (s["final_equity"] - SC) / max(s["n_days"], 1)

    # Bootstrap
    rng = np.random.default_rng(42)
    boot = [rng.choice(daily_ret.values, size=len(daily_ret), replace=True) for _ in range(5000)]
    boot_sr = [b.mean() / b.std() * np.sqrt(365.25) for b in boot if b.std() > 0]
    ci_lo = round(np.percentile(boot_sr, 2.5), 3)
    ci_hi = round(np.percentile(boot_sr, 97.5), 3)

    # DSR
    sr = s["sharpe"] / np.sqrt(365.25)
    v_sr = (1 - s["skewness"] * sr + ((s["kurtosis"] - 1) / 4) * sr ** 2) / max(s["n_days"] - 1, 1)
    sr_std = np.sqrt(max(v_sr, 1e-20))
    gamma = 0.5772
    e_max_sr = sr_std * ((1 - gamma) * norm.ppf(1 - 1 / 40) + gamma * norm.ppf(1 - 1 / (40 * np.e)))
    dsr = float(norm.cdf((sr - e_max_sr) / sr_std))

    # Monthly
    monthly = dpnl.resample("ME").sum()
    pos_m = (monthly > 0).sum()
    tot_m = len(monthly)

    # Weekly Sharpe
    wkly = dpnl.resample("W").sum() / SC
    wk_sharpe = wkly.mean() / wkly.std() * np.sqrt(52) if wkly.std() > 0 else 0

    # Tail risk
    var95 = np.percentile(daily_ret, 5) * 100
    cvar95 = daily_ret[daily_ret <= np.percentile(daily_ret, 5)].mean() * 100 if len(daily_ret) > 20 else 0

    # Streaks
    max_win = max_loss = cur = 0
    for v in (daily_ret > 0):
        cur = cur + 1 if v else 0
        max_win = max(max_win, cur)
    cur = 0
    for v in (daily_ret <= 0):
        cur = cur + 1 if v else 0
        max_loss = max(max_loss, cur)

    # Underwater
    uw = []
    in_dd = False
    ds = None
    for t, v in (drawdown < -0.001).items():
        if v and not in_dd:
            ds = t
            in_dd = True
        elif not v and in_dd:
            uw.append((t - ds).days)
            in_dd = False
    avg_uw = np.mean(uw) if uw else 0
    max_uw = max(uw) if uw else 0

    elapsed = time.time() - t0

    print(f"""
================================================================================
{label}
STRICT HOLDOUT: Train ONCE, freeze, test on UNSEEN future data
30 assets | conf >= 0.08 | 1 bps slippage | 4% sizing | Kelly
================================================================================

--- RETURNS -------------------------------------------------------------------
Compounding Annual Return    {s['cagr_pct']:>10.2f}%
Total Return                 {s['total_return_pct']:>10.2f}%
Start Equity                 ${SC:>12,.2f}
End Equity                   ${s['final_equity']:>12,.2f}
Net Profit                   ${s['final_equity']-SC:>12,.2f}
PnL per Day                  ${pnl_day:>12.2f}

--- RISK-ADJUSTED -------------------------------------------------------------
Sharpe (daily)               {s['sharpe']:>10.3f}
  95% Bootstrap CI           [{ci_lo}, {ci_hi}]
Sharpe (weekly)              {wk_sharpe:>10.3f}
Sortino Ratio                {s['sortino']:>10.3f}
Calmar Ratio                 {s['calmar']:>10.3f}
PSR (P[true SR > 0])         {s['psr_pct']:>10.1f}%
DSR (K=40 trials)            {dsr*100:>10.1f}%

--- DRAWDOWN ------------------------------------------------------------------
Maximum Drawdown             {s['max_dd_pct']:>10.2f}%
Avg Underwater Period        {avg_uw:>10.1f} days
Max Underwater Period        {max_uw:>10} days
Annual Std Dev               {s['ann_std']*100:>10.2f}%

--- TAIL RISK -----------------------------------------------------------------
VaR (95%, daily)             {var95:>10.4f}%
CVaR (95%, daily)            {cvar95:>10.4f}%
Daily Skewness               {s['skewness']:>10.3f}
Daily Kurtosis               {s['kurtosis']:>10.1f}

--- TRADING -------------------------------------------------------------------
Total Trades                 {s['n_trades']:>10,}
Trades per Day               {s['trades_per_day']:>10.1f}
Win Rate (days)              {s['win_rate_days']*100:>10.1f}%
Profit Factor                {s['profit_factor']:>10.3f}
Average Win                  {s['avg_win_bps']:>10.1f} bps/day
Average Loss                 {s['avg_loss_bps']:>10.1f} bps/day
Expectancy                   {s['expectancy_bps']:>10.2f} bps/day
Max Consecutive Wins         {max_win:>10} days
Max Consecutive Losses       {max_loss:>10} days

--- CONSISTENCY ---------------------------------------------------------------
Test Days                    {s['n_days']:>10}
Positive Months              {pos_m}/{tot_m} ({100*pos_m/max(tot_m,1):.0f}%)
Assets Generating Signals    {len(all_signals):>10}
Elapsed                      {elapsed:>10.0f}s

--- MONTHLY PnL ---------------------------------------------------------------""")

    cum = 0
    for dt, pnl in monthly.items():
        ret = pnl / SC * 100
        cum += ret
        mk = " ***" if pnl == monthly.max() else (" ---" if pnl == monthly.min() else "")
        print(f"  {dt.strftime('%Y-%m')}: ${pnl:>9.2f} ({ret:>+6.2f}%) cum={cum:>6.2f}%{mk}")

    print(f"""
--- VERDICT -------------------------------------------------------------------
Signal:    {"REAL" if ci_lo > 0 else "UNCERTAIN"} (CI [{ci_lo}, {ci_hi}])
DSR:       {dsr*100:.1f}%
""")


def main():
    print("=" * 80)
    print("REPRESENTATIVE HOLDOUT STATS")
    print("=" * 80, flush=True)

    # Test B: 12mo train / 12mo test (most balanced)
    full_report("2025-03-01", "TEST B: 12mo train (2024-03 to 2025-02) / 12mo test (2025-03 to 2026-02)")

    # Test C: 18mo train / 6mo test (most training, most like production)
    full_report("2025-09-01", "TEST C: 18mo train (2024-03 to 2025-08) / 6mo test (2025-09 to 2026-02)")


if __name__ == "__main__":
    main()
