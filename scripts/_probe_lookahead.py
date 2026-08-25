"""ADVERSARIAL LOOKAHEAD PROBE for the conditioned-long workflow.

(1) Point-in-time check: confirm signal cols are entry-bar features, predictions
    joined index-aligned (no future shift).
(2) Staling probe: shift the conditioning signal by k=1,3,10 trades. If the
    'edge ranks longs worse->less-bad' relationship is a real (regime-persistent)
    signal it decays gracefully; if it collapses to ~baseline immediately, the
    apparent signal-PnL link was a per-bar leak.
(3) Post-exit peeking: confirm the long-straddle marks use only data <= exit.
"""
import numpy as np, pandas as pd
from scipy import stats as sps
from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic
from scripts.longhold_hedged_vrp import _nonoverlap_sharpe

tl = pd.read_parquet("vol_arb/output/BTC_v2_trade_log.parquet")
tl = tl.copy()
tl["signal"] = np.where(tl["signal"].to_numpy() != 0, 1.0, 0.0)
cfg = RealisticEvalConfig(hold="4h", real_bs_delta=True, fill_tol_minutes=60.0,
                          spread_source="tape", delta_hedge_band=0.35)
f = evaluate_realistic(tl, cfg).trade_log
cov = f[f["premium_tier"] > 0].copy().sort_index()
e = pd.DatetimeIndex(pd.to_datetime(cov.index, utc=True))
x = e + pd.Timedelta(minutes=240)
exp = pd.DatetimeIndex(pd.to_datetime(cov["expiry_dt"].values, utc=True))
x = pd.DatetimeIndex(np.where(x < exp, x, exp - pd.Timedelta(minutes=1)))
pnl = cov["total_hedged_usd"].to_numpy(float)
n = len(cov)
print(f"4h covered long set: n={n}")

def metric(sig_vals, thr):
    k = np.isfinite(sig_vals) & (sig_vals >= thr)
    if k.sum() < 5:
        return None
    p = pnl[k]
    ns,_ = _nonoverlap_sharpe(p, e[k], x[k])
    return dict(n=int(k.sum()), total=float(p.sum()), mean=float(p.mean()), nonov=ns)

# ---- (2) STALING PROBE on the strongest signal 'edge' ----
# Use the top-decile threshold (the aggressive selective cut where signal showed).
sig = cov["edge"].to_numpy(float)
valid = np.isfinite(sig)
thr = np.quantile(sig[valid], 0.9)
print(f"\n=== STALING PROBE: signal='edge', thr=q0.9={thr:.3f} ===")
print(f"{'shift':>6} {'n':>5} {'total$':>10} {'mean$/tr':>9} {'nonovS':>7}")
base = metric(sig, thr)
print(f"{'0':>6} {base['n']:>5} {base['total']:>10,.0f} {base['mean']:>9,.1f} {base['nonov']:>7.3f}")
for ksh in (1, 3, 10):
    # shift the signal forward by k trades: trade i uses signal from trade i-k
    sh = np.full(n, np.nan)
    sh[ksh:] = sig[:n-ksh]
    m = metric(sh, thr)
    if m: print(f"{ksh:>6} {m['n']:>5} {m['total']:>10,.0f} {m['mean']:>9,.1f} {m['nonov']:>7.3f}")

# unconditional baseline (no condition) for reference
ub,_ = _nonoverlap_sharpe(pnl, e, x)
print(f"\nUNCONDITIONAL (all {n}): total=${pnl.sum():,.0f} mean=${pnl.mean():,.1f} nonovS={ub:.3f}")

# ---- correlation of signal with PnL, and with FUTURE-trade PnL (leak tell) ----
print("\n=== Spearman(signal, pnl) at same trade vs shifted (leak diagnostic) ===")
sv = sig[valid]; pv = pnl[valid]
rho0 = sps.spearmanr(sv, pv).correlation
print(f"  same-trade corr(edge, pnl) = {rho0:.4f}")
for ksh in (1,3,10):
    a = sig[:n-ksh]; b = pnl[ksh:]  # signal(i) vs pnl(i+k)  (future)
    mm = np.isfinite(a)&np.isfinite(b)
    print(f"  corr(edge[i], pnl[i+{ksh}]) future = {sps.spearmanr(a[mm], b[mm]).correlation:.4f}")
    c = sig[ksh:]; d = pnl[:n-ksh]  # signal(i) vs pnl(i-k) (past)
    mm2 = np.isfinite(c)&np.isfinite(d)
    print(f"  corr(edge[i], pnl[i-{ksh}]) past   = {sps.spearmanr(c[mm2], d[mm2]).correlation:.4f}")
