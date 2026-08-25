"""Stress-test the 4h edge-conditioned long: plateau vs knife-edge, per-year,
DSR with full K, and the 2026-removed robustness check."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats as sps
from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic
from scripts.longhold_hedged_vrp import _sharpe_iid, _nonoverlap_sharpe, _max_drawdown

tl = pd.read_parquet("vol_arb/output/BTC_v2_trade_log.parquet")
tl["signal"] = np.where(tl["signal"].to_numpy() != 0, 1.0, 0.0)
cfg = RealisticEvalConfig(hold="4h", real_bs_delta=True, fill_tol_minutes=60.0,
                          spread_source="tape", delta_hedge_band=0.35)
f = evaluate_realistic(tl, cfg).trade_log
cov = f[f["premium_tier"] > 0].copy().sort_index()
e = pd.DatetimeIndex(pd.to_datetime(cov.index, utc=True))
x = e + pd.Timedelta(minutes=240)
exp = pd.DatetimeIndex(pd.to_datetime(cov["expiry_dt"].values, utc=True))
x = pd.DatetimeIndex(np.where(x < exp, x, exp - pd.Timedelta(minutes=1)))
yr = e.year
pnl = cov["total_hedged_usd"].to_numpy(float)

print("=== 4h edge-conditioned long: threshold PLATEAU scan (full sample) ===")
sv = cov["edge"].to_numpy(float)
print(f"{'thr(vp)':>8} {'n':>5} {'total$':>10} {'mean$/tr':>9} {'nonovS':>7} {'2025$':>9} {'2026$':>9} {'2024$':>9} {'2023$':>9}")
for thr in [0, 20, 30, 40, 45, 50, 55, 60, 65, 70, 80]:
    k = np.isfinite(sv) & (sv >= thr)
    if k.sum() < 5:
        continue
    p = pnl[k]; ee = e[k]; xx = x[k]
    ns, _ = _nonoverlap_sharpe(p, ee, xx)
    py = {y: float(np.nansum(p[ee.year == y])) for y in [2023, 2024, 2025, 2026]}
    print(f"{thr:>8} {k.sum():>5} {p.sum():>10,.0f} {p.mean():>9,.1f} {ns:>7.2f} "
          f"{py.get(2025,0):>9,.0f} {py.get(2026,0):>9,.0f} {py.get(2024,0):>9,.0f} {py.get(2023,0):>9,.0f}")

print("\n=== full-sample edge>=50 long: per-year & 2026-removed ===")
thr = 50.0
k = np.isfinite(sv) & (sv >= thr)
p = pnl[k]; ee = e[k]; xx = x[k]
print(f"n={k.sum()} total=${p.sum():,.0f} mean=${p.mean():,.1f}/tr")
for y in sorted(set(ee.year)):
    py = p[ee.year == y]
    print(f"  {y}: ${py.sum():,.0f}  n={len(py)}  mean=${py.mean():,.1f}")
# exclude partial 2026
k25 = k & (yr <= 2025)
p25 = pnl[k25]
ns25, nn25 = _nonoverlap_sharpe(p25, e[k25], x[k25])
print(f"  2022-2025 only (drop partial 2026): total=${p25.sum():,.0f} nonovS={ns25:.3f} n={len(p25)}")

# one-sample t on per-trade pnl (full)
t = sps.ttest_1samp(p[np.isfinite(p)], 0.0)
ns_full, nn_full = _nonoverlap_sharpe(p, ee, xx)
print(f"\nfull-sample edge>=50: per-trade t-stat={t.statistic:.3f} p={t.pvalue:.3f}; nonov_S={ns_full:.3f} (n_indep={nn_full})")

# DSR / multiple-testing: K = horizons x bands x thresholds x signals tried
# horizons tried in conditioning workflow: 15m,1h,4h = 3 ; band: 1 (0.35) ; signals 7 ; thresholds 10 quantiles
K = 3 * 1 * 7 * 10
print(f"\n=== Multiple-testing K = 3 horizons x 1 band x 7 signals x 10 thr = {K} ===")
print("Deflated hurdle (Bonferroni on a one-sided t): need p < 0.05/K = "
      f"{0.05/K:.2e}  => |t| > {sps.norm.ppf(1-0.05/K):.2f}")
print(f"Best OOS non-overlap Sharpe seen anywhere = ~+0.16 (4h rv_pred_pct). "
      f"n_indep at OOS ~ <30. Implied |t| << hurdle.")
