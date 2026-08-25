"""Full stats report for EVERY lookback period — saved to desktop."""
import sys
sys.path.insert(0, "src")
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import copy, warnings, os, time, json
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
LOOKBACKS = [1, 2, 3, 4, 6, 8, 10, 12, 14, 18]
DESKTOP = Path(os.path.expanduser("~/Desktop"))

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
    asset, symbol, fixed_start = args
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
        return asset, {}

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
        return asset, {}

    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    all_months = pd.period_range(
        start=(start_date + pd.DateOffset(months=2)).to_period("M"),
        end=end_date.to_period("M"), freq="M")

    results = {lb: [] for lb in LOOKBACKS}
    for month in all_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)
        te = valid[(valid >= ms) & (valid <= me)]
        vol_s = ms - pd.DateOffset(months=3)
        vt_data = vol.loc[(vol.index >= vol_s) & (vol.index < ms)].dropna()
        if len(vt_data) < 50:
            continue
        vt = vt_data.quantile(0.50)
        hv = (vol.reindex(te) > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5:
            continue

        for lb in LOOKBACKS:
            ts = ms - pd.DateOffset(months=lb)
            if ts < start_date:
                continue
            if fixed_start and str(month) < fixed_start:
                continue
            tc = ms - embargo_td
            tr = valid[(valid >= ts) & (valid < tc)]
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
                signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
            if len(filtered) > 0:
                results[lb].append(filtered.to_dict("list"))
    return asset, results


def compute_report(lb, all_results):
    """Compute full stats for one lookback, return dict."""
    sigs = []
    for asset_results in all_results.values():
        sigs.extend(asset_results.get(lb, []))
    if not sigs:
        return None

    frames = [pd.DataFrame(d) for d in sigs]
    filtered = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)

    # Kelly
    kw = {}
    for asset in filtered["asset"].unique():
        a = filtered[filtered["asset"] == asset]
        tr = a["direction"].values * a["fwd_return"].values
        w = tr[tr > 0]; l = tr[tr <= 0]
        if len(w) >= 3 and len(l) >= 3 and abs(l.mean()) > 0:
            f = kelly_size(len(w)/len(tr), float(w.mean()), float(abs(l.mean())))
            kw[asset] = max(0.2, min(3.0, f * 7.5))
        else:
            kw[asset] = 1.0
    mw = np.mean(list(kw.values()))
    if mw > 0:
        kw = {k: v/mw for k, v in kw.items()}
    filtered["signal_strength"] = filtered.apply(
        lambda r: r["signal_strength"] * kw.get(r["asset"], 1.0), axis=1)

    cfg = load_config(Path("configs/default.yaml"))
    pcfg = copy.deepcopy(cfg.portfolio)
    pcfg.base_sizing_pct = 0.04
    pcfg.slippage = 0.0001
    pcfg.max_exposure = 0.90
    pcfg.max_per_asset = 0.03
    SC = pcfg.starting_capital

    sim = simulate_portfolio(filtered, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None

    s = compute_full_stats(dpnl, SC, sim["n_trades"], sim["n_trades"] / len(dpnl))
    daily_ret = dpnl / SC
    equity = (1 + daily_ret).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    # Bootstrap
    rng = np.random.default_rng(42)
    boot = [rng.choice(daily_ret.values, size=len(daily_ret), replace=True) for _ in range(3000)]
    bsr = [b.mean()/b.std()*np.sqrt(365.25) for b in boot if b.std() > 0]
    ci_lo = round(np.percentile(bsr, 2.5), 3)
    ci_hi = round(np.percentile(bsr, 97.5), 3)

    # DSR
    sr = s["sharpe"] / np.sqrt(365.25)
    v = (1 - s["skewness"]*sr + ((s["kurtosis"]-1)/4)*sr**2) / max(s["n_days"]-1, 1)
    sr_std = np.sqrt(max(v, 1e-20))
    gamma = 0.5772
    e_max = sr_std * ((1-gamma)*norm.ppf(1-1/40) + gamma*norm.ppf(1-1/(40*np.e)))
    dsr = float(norm.cdf((sr - e_max) / sr_std))

    # Weekly/quarterly
    wk = dpnl.resample("W").sum() / SC
    wk_sr = wk.mean()/wk.std()*np.sqrt(52) if wk.std() > 0 else 0
    qt = dpnl.resample("QE").sum() / SC
    qt_sr = qt.mean()/qt.std()*np.sqrt(4) if qt.std() > 0 else 0

    # Monthly
    monthly = dpnl.resample("ME").sum()
    pos_m = (monthly > 0).sum()
    tot_m = len(monthly)
    hhi = float((monthly.abs()/monthly.abs().sum()).pow(2).sum()) if monthly.abs().sum() > 0 else 0

    # Tail
    var95 = np.percentile(daily_ret, 5) * 100
    cvar95 = daily_ret[daily_ret <= np.percentile(daily_ret, 5)].mean() * 100

    # Streaks
    mxw = mxl = c = 0
    for v in (daily_ret > 0):
        c = c+1 if v else 0; mxw = max(mxw, c)
    c = 0
    for v in (daily_ret <= 0):
        c = c+1 if v else 0; mxl = max(mxl, c)

    # Underwater
    uw = []; in_d = False; ds = None
    for t, v in (drawdown < -0.001).items():
        if v and not in_d: ds = t; in_d = True
        elif not v and in_d: uw.append((t-ds).days); in_d = False

    # DD recovery
    dd_idx = drawdown.idxmin()
    post = equity.loc[dd_idx:]
    rec = post[post >= running_max.loc[dd_idx]]
    dd_rec = (rec.index[0] - dd_idx).days if len(rec) > 0 else -1

    pnl_day = (s["final_equity"] - SC) / max(s["n_days"], 1)
    ann_turn = s["trades_per_day"] * pcfg.base_sizing_pct * 365.25

    return {
        "lookback": lb,
        "sharpe": s["sharpe"], "sortino": s["sortino"], "calmar": s["calmar"],
        "cagr_pct": s["cagr_pct"], "total_return_pct": s["total_return_pct"],
        "max_dd_pct": s["max_dd_pct"], "psr_pct": s["psr_pct"],
        "dsr_pct": round(dsr * 100, 1),
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "sharpe_weekly": round(wk_sr, 3), "sharpe_quarterly": round(qt_sr, 3),
        "n_trades": s["n_trades"], "trades_per_day": s["trades_per_day"],
        "win_rate_pct": round(s["win_rate_days"] * 100, 1),
        "profit_factor": s["profit_factor"],
        "avg_win_bps": s["avg_win_bps"], "avg_loss_bps": s["avg_loss_bps"],
        "expectancy_bps": s["expectancy_bps"],
        "ann_std_pct": round(s["ann_std"] * 100, 2),
        "var95_pct": round(var95, 4), "cvar95_pct": round(cvar95, 4),
        "skewness": s["skewness"], "kurtosis": s["kurtosis"],
        "max_win_streak": mxw, "max_loss_streak": mxl,
        "avg_underwater_days": round(np.mean(uw), 1) if uw else 0,
        "max_underwater_days": max(uw) if uw else 0,
        "dd_recovery_days": dd_rec,
        "positive_months_pct": round(100 * pos_m / max(tot_m, 1), 1),
        "hhi": round(hhi, 4),
        "pnl_per_day": round(pnl_day, 2),
        "turnover_annual": round(ann_turn, 0),
        "n_days": s["n_days"],
        "final_equity": s["final_equity"],
        "monthly": {dt.strftime("%Y-%m"): round(pnl, 2) for dt, pnl in monthly.items()},
    }


def main():
    print("=" * 70)
    print(f"GENERATING FULL STATS FOR ALL LOOKBACK PERIODS ({W} workers)")
    print("=" * 70, flush=True)

    # Determine fixed start (month 19 for 18mo lookback)
    sample = resample_ohlcv(load_binance_1m("BTCUSDT", [DATA_DIR]), "15min")
    start = sample.index.min()
    fixed_start = str((start + pd.DateOffset(months=max(LOOKBACKS) + 1)).to_period("M"))
    print(f"Data: {start.date()}, fixed start: {fixed_start}", flush=True)

    # Generate all signals
    t0 = time.time()
    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s, fixed_start)): a for a, s in ASSETS.items()}
        done = 0
        for future in as_completed(futures):
            asset, results = future.result()
            done += 1
            all_results[asset] = results
            print(f"  [{done:>2d}/30] {asset}", flush=True)
    print(f"Generated in {time.time()-t0:.0f}s\n", flush=True)

    # Compute full stats for each lookback
    all_reports = []
    for lb in LOOKBACKS:
        print(f"Computing stats for {lb}mo...", flush=True)
        report = compute_report(lb, all_results)
        if report:
            all_reports.append(report)

    # Write to desktop
    output_lines = []
    for r in all_reports:
        block = f"""
{'='*80}
{r['lookback']}-MONTH ROLLING LOOKBACK + KELLY + 30 ASSETS
Same test period: {fixed_start} to 2026-02 | Monthly retrain
{'='*80}

PSR                          {r['psr_pct']:.1f}%
Sharpe Ratio                 {r['sharpe']:.3f}
Sortino Ratio                {r['sortino']:.3f}
Calmar Ratio                 {r['calmar']:.3f}
Total Orders                 {r['n_trades']:,}
Average Win                  {r['avg_win_bps']:.1f} bps/day
Average Loss                 {r['avg_loss_bps']:.1f} bps/day
Compounding Annual Return    {r['cagr_pct']:.2f}%
Drawdown                     {r['max_dd_pct']:.2f}%
Expectancy                   {r['expectancy_bps']:.2f} bps/day
Start Equity                 $10,000.00
End Equity                   ${r['final_equity']:,.2f}
Net Profit                   {r['total_return_pct']:.2f}%
Loss Rate                    {100-r['win_rate_pct']:.1f}%
Win Rate                     {r['win_rate_pct']:.1f}%
Profit-Loss Ratio            {r['profit_factor']:.3f}
Annual Standard Deviation    {r['ann_std_pct']:.2f}%
Annual Variance              {r['ann_std_pct']**2/100:.2f}%
Portfolio Turnover           {r['turnover_annual']:.0f}x/year
Drawdown Recovery            {r['dd_recovery_days']} days

--- ADDITIONAL STATS ---
Sharpe (weekly)              {r['sharpe_weekly']:.3f}
Sharpe (quarterly)           {r['sharpe_quarterly']:.3f}
Bootstrap 95% CI             [{r['ci_lo']}, {r['ci_hi']}]
DSR (K=40)                   {r['dsr_pct']:.1f}%
VaR (95%, daily)             {r['var95_pct']:.4f}%
CVaR (95%, daily)            {r['cvar95_pct']:.4f}%
Skewness                     {r['skewness']:.3f}
Kurtosis                     {r['kurtosis']:.1f}
Max Consecutive Wins         {r['max_win_streak']} days
Max Consecutive Losses       {r['max_loss_streak']} days
Avg Underwater Period        {r['avg_underwater_days']:.1f} days
Max Underwater Period        {r['max_underwater_days']} days
PnL per Day                  ${r['pnl_per_day']:.2f}
Positive Months              {r['positive_months_pct']:.0f}%
HHI Concentration            {r['hhi']:.4f} {"(OK)" if r['hhi'] < 0.15 else "(CONCENTRATED)"}
Days of Data                 {r['n_days']}
Kelly Sizing                 Walk-forward (30 assets)
Training Window              {r['lookback']}-month rolling

--- MONTHLY PnL ---"""
        output_lines.append(block)
        for m, pnl in sorted(r["monthly"].items()):
            output_lines.append(f"  {m}: ${pnl:>9.2f}")
        output_lines.append("")

    out_path = DESKTOP / "lookback_full_stats.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\nSaved to {out_path}")

    # Also save JSON
    json_path = DESKTOP / "lookback_full_stats.json"
    with open(json_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"Saved to {json_path}")


if __name__ == "__main__":
    main()
