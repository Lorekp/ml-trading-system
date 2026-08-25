"""Full stats report: 12-month rolling window + Kelly + 30 assets."""
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
LOOKBACK = 12
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


def process_asset(args):
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
    except:
        return asset, None

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
    if len(valid) < 5000:
        return asset, None

    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    all_months = pd.period_range(
        start=(start_date + pd.DateOffset(months=LOOKBACK + 1)).to_period("M"),
        end=end_date.to_period("M"), freq="M")
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    all_sigs = []
    for month in all_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

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
        except:
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
            signals, ConfidenceFilterConfig(mode="absolute", threshold=CONF_THRESH))
        if len(filtered) > 0:
            all_sigs.append(filtered.to_dict("list"))

    return asset, all_sigs


def main():
    print("Generating 12-month rolling signals for 30 assets...", flush=True)
    t0 = time.time()

    all_signals = []
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s)): a for a, s in ASSETS.items()}
        done = 0
        for future in as_completed(futures):
            asset, sigs = future.result()
            done += 1
            if sigs:
                all_signals.extend(sigs)

    frames = [pd.DataFrame(d) for d in all_signals]
    filtered = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)
    print(f"Generated {len(filtered)} signals in {time.time()-t0:.0f}s", flush=True)

    # Kelly
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
    dpnl = compute_daily_pnl_series(sim)
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

    # Weekly/quarterly Sharpe
    wkly = dpnl.resample("W").sum() / SC
    wk_sharpe = wkly.mean() / wkly.std() * np.sqrt(52) if wkly.std() > 0 else 0
    qtr = dpnl.resample("QE").sum() / SC
    qt_sharpe = qtr.mean() / qtr.std() * np.sqrt(4) if qtr.std() > 0 else 0

    # Monthly
    monthly = dpnl.resample("ME").sum()
    pos_m = (monthly > 0).sum()
    tot_m = len(monthly)
    hhi = float((monthly.abs() / monthly.abs().sum()).pow(2).sum()) if monthly.abs().sum() > 0 else 0

    # Tail
    var95 = np.percentile(daily_ret, 5) * 100
    cvar95 = daily_ret[daily_ret <= np.percentile(daily_ret, 5)].mean() * 100

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

    # Turnover
    ann_turnover = s["trades_per_day"] * pcfg.base_sizing_pct * 365.25
    total_fees = sim["n_trades"] * SC * pcfg.base_sizing_pct * 0.15 * SLIPPAGE

    # DD recovery
    max_dd_idx = drawdown.idxmin()
    post_dd = equity.loc[max_dd_idx:]
    recovery = post_dd[post_dd >= running_max.loc[max_dd_idx]]
    dd_recovery = (recovery.index[0] - max_dd_idx).days if len(recovery) > 0 else -1

    kelly_min = min(kelly_weights.values())
    kelly_max = max(kelly_weights.values())

    print(f"""
--- 12-MONTH ROLLING + KELLY + 30 ASSETS (Production Version) ---

PSR                          {s['psr_pct']:.1f}%
Sharpe Ratio                 {s['sharpe']:.3f}
Sortino Ratio                {s['sortino']:.3f}
Calmar Ratio                 {s['calmar']:.3f}
Total Orders                 {s['n_trades']:,}
Average Win                  {s['avg_win_bps']:.1f} bps/day
Average Loss                 {s['avg_loss_bps']:.1f} bps/day
Compounding Annual Return    {s['cagr_pct']:.2f}%
Drawdown                     {s['max_dd_pct']:.2f}%
Expectancy                   {s['expectancy_bps']:.2f} bps/day
Start Equity                 ${SC:,.2f}
End Equity                   ${s['final_equity']:,.2f}
Net Profit                   {s['total_return_pct']:.2f}%
Loss Rate                    {s['loss_rate_days']*100:.1f}%
Win Rate                     {s['win_rate_days']*100:.1f}%
Profit-Loss Ratio            {s['profit_factor']:.3f}
Annual Standard Deviation    {s['ann_std']*100:.2f}%
Annual Variance              {(s['ann_std']*100)**2/100:.2f}%
Portfolio Turnover           {ann_turnover:.0f}x/year
Drawdown Recovery            {dd_recovery} days
Total Fees (slippage)        ${total_fees:.2f} (estimated at 1 bps)

--- ADDITIONAL STATS ---
Sharpe (weekly)              {wk_sharpe:.3f}
Sharpe (quarterly)           {qt_sharpe:.3f}
Bootstrap 95% CI             [{ci_lo}, {ci_hi}]
DSR (K=40)                   {dsr*100:.1f}%
VaR (95%, daily)             {var95:.4f}%
CVaR (95%, daily)            {cvar95:.4f}%
Skewness                     {s['skewness']:.3f}
Kurtosis                     {s['kurtosis']:.1f}
Max Consecutive Wins         {max_win} days
Max Consecutive Losses       {max_loss} days
Avg Underwater Period        {avg_uw:.1f} days
Max Underwater Period        {max_uw} days
PnL per Day                  ${pnl_day:.2f}
Positive Months              {pos_m}/{tot_m} ({100*pos_m/max(tot_m,1):.0f}%)
HHI Concentration            {hhi:.4f} {"(OK)" if hhi < 0.15 else "(CONCENTRATED)"}
Days of Data                 {s['n_days']}
Kelly Sizing                 Walk-forward (30 assets, weights {kelly_min:.2f}-{kelly_max:.2f}x)
Training Window              12-month rolling (retrained monthly)

--- MONTHLY PnL ---""")

    cum = 0
    for dt, pnl in monthly.items():
        ret = pnl / SC * 100
        cum += ret
        mk = " ***" if pnl == monthly.max() else (" ---" if pnl == monthly.min() else "")
        print(f"  {dt.strftime('%Y-%m')}: ${pnl:>9.2f} ({ret:>+6.2f}%) cum={cum:>6.2f}%{mk}")

    print(f"""
--- VERDICT ---
Signal:    {"REAL" if ci_lo > 0 else "UNCERTAIN"} (CI [{ci_lo}, {ci_hi}])
DSR:       {dsr*100:.1f}%
Deploy:    {"RECOMMENDED" if ci_lo > 0 and dsr > 0.5 else "CAUTION"}
""")


if __name__ == "__main__":
    main()
